#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
PREPARED_MANIFEST="${EGOBODY_PREPARED_MANIFEST:-$DATASET_ROOT/prepared_manifest.json}"
PACK_ROOT="${EGOBODY_SMPL_PACK_ROOT:-$DATASET_ROOT/online_packs_smpl_v1}"
MM_SPLIT="${MM_SPLIT:-train}"
MM_TEACHER_MODE="${MM_TEACHER_MODE:-body_only}"
MM_TEACHER_ITERATIONS="${MM_TEACHER_ITERATIONS:-5000}"
MM_TEACHER_BATCH_SIZE="${MM_TEACHER_BATCH_SIZE:-100}"
MM_EVAL_METRICS_EVERY="${MM_EVAL_METRICS_EVERY:-100}"
MM_FIXED_MOTION_EVAL_BATCH_SIZE="${MM_FIXED_MOTION_EVAL_BATCH_SIZE:-10}"
MM_SCENE_POINTCLOUD_CANDIDATES="${MM_SCENE_POINTCLOUD_CANDIDATES:-2048}"
MM_SCENE_DISTANCE_LOSS_WEIGHT="${MM_SCENE_DISTANCE_LOSS_WEIGHT:-0.25}"
MM_SCENE_CONTACT_LOSS_WEIGHT="${MM_SCENE_CONTACT_LOSS_WEIGHT:-0.5}"
MM_SCENE_CONTACT_POSITIVE_WEIGHT="${MM_SCENE_CONTACT_POSITIVE_WEIGHT:-10.0}"
MM_SCENE_CONTACT_THRESHOLD_M="${MM_SCENE_CONTACT_THRESHOLD_M:-0.05}"
MM_SCENE_COUNTERFACTUAL_LOSS_WEIGHT="${MM_SCENE_COUNTERFACTUAL_LOSS_WEIGHT:-0.1}"
MM_SCENE_COUNTERFACTUAL_ACTION_MARGIN="${MM_SCENE_COUNTERFACTUAL_ACTION_MARGIN:-0.03}"
MM_SCENE_RESIDUAL_PRESERVATION_WEIGHT="${MM_SCENE_RESIDUAL_PRESERVATION_WEIGHT:-10.0}"
MM_FREEZE_BASE_ACTOR="${MM_FREEZE_BASE_ACTOR:-1}"
MM_ACTOR_LEARNING_RATE="${MM_ACTOR_LEARNING_RATE:-1e-4}"
MM_SCENE_LEARNING_RATE_MULTIPLIER="${MM_SCENE_LEARNING_RATE_MULTIPLIER:-1.0}"
MM_RESET_SCENE_ON_WARM_START="${MM_RESET_SCENE_ON_WARM_START:-0}"
MM_TEACHER_CHECKPOINT="${MM_TEACHER_CHECKPOINT:-$REPO_ROOT/data/pretrained_models/motion_tracker/smpl/last.ckpt}"
MM_TEACHER_EXPERIMENT_NAME="${MM_TEACHER_EXPERIMENT_NAME:-egobody_smpl_teacher_${MM_TEACHER_MODE}_800_v1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$PREPARED_MANIFEST" ]]; then
    echo "Missing EgoBody prepared manifest: $PREPARED_MANIFEST" >&2
    exit 2
fi
if [[ ! -f "$PACK_ROOT/manifest.json" ]]; then
    "$PYTHON_BIN" data/scripts/build_egobody_online_sft_packs.py \
        --manifest "$PREPARED_MANIFEST" \
        --output-root "$PACK_ROOT" \
        --motion-filename motion_lib_smpl.pt \
        --camera-filename ego_camera.pt
fi

if [[ "$MM_SPLIT" == "all" && ! -f "$PACK_ROOT/all/manifest.json" ]]; then
    "$PYTHON_BIN" data/scripts/build_egobody_online_sft_packs.py \
        --manifest "$PREPARED_MANIFEST" \
        --output-root "$PACK_ROOT" \
        --motion-filename motion_lib_smpl.pt \
        --camera-filename ego_camera.pt \
        --splits all
fi

split_manifest="$PACK_ROOT/$MM_SPLIT/manifest.json"
if [[ ! -f "$split_manifest" ]]; then
    echo "Missing SMPL split manifest: $split_manifest" >&2
    exit 2
fi

readarray -t inputs < <(
    "$PYTHON_BIN" - "$split_manifest" "$PREPARED_MANIFEST" <<'PY'
import json
import sys

split = json.load(open(sys.argv[1]))
prepared = json.load(open(sys.argv[2]))
print(split["motion_file"])
print(split["scene_file"])
print(prepared["egobody_root"])
print(split["num_motions"])
PY
)

num_envs="${MM_TEACHER_NUM_ENVS:-${inputs[3]}}"
if (( num_envs < inputs[3] )); then
    echo "MM_TEACHER_NUM_ENVS must be >= ${inputs[3]} paired motions." >&2
    exit 3
fi
rollout_samples=$((num_envs * 32))
if (( MM_TEACHER_BATCH_SIZE > rollout_samples || rollout_samples % MM_TEACHER_BATCH_SIZE != 0 )); then
    echo "MM_TEACHER_BATCH_SIZE must divide num_envs*32=$rollout_samples." >&2
    exit 3
fi

result_dir="$REPO_ROOT/results/$MM_TEACHER_EXPERIMENT_NAME"
mkdir -p "$result_dir"
echo "SMPL stage-1 teacher: mode=$MM_TEACHER_MODE split=$MM_SPLIT envs=$num_envs"
echo "iterations=$MM_TEACHER_ITERATIONS batch=$MM_TEACHER_BATCH_SIZE"

cd "$REPO_ROOT"
FREEZE_BASE_ACTOR_FLAG="--freeze-base-actor"
if [[ "$MM_FREEZE_BASE_ACTOR" == "0" ]]; then
    FREEZE_BASE_ACTOR_FLAG="--no-freeze-base-actor"
fi
RESET_SCENE_FLAG="--no-reset-scene-on-warm-start"
if [[ "$MM_RESET_SCENE_ON_WARM_START" == "1" ]]; then
    RESET_SCENE_FLAG="--reset-scene-on-warm-start"
fi

exec "$REPO_ROOT/scripts/run_with_memory_guard.sh" \
    "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
    "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaaclab \
    --num-envs "$num_envs" \
    --batch-size "$MM_TEACHER_BATCH_SIZE" \
    --motion-file "${inputs[0]}" \
    --scenes-file "${inputs[1]}" \
    --scene-asset-root "${inputs[2]}" \
    --teacher-condition "$MM_TEACHER_MODE" \
    --scene-pointcloud-candidates "$MM_SCENE_POINTCLOUD_CANDIDATES" \
    --scene-pointcloud-seed 0 \
    --scene-distance-loss-weight "$MM_SCENE_DISTANCE_LOSS_WEIGHT" \
    --scene-contact-loss-weight "$MM_SCENE_CONTACT_LOSS_WEIGHT" \
    --scene-contact-positive-weight "$MM_SCENE_CONTACT_POSITIVE_WEIGHT" \
    --scene-contact-threshold-m "$MM_SCENE_CONTACT_THRESHOLD_M" \
    --scene-counterfactual-loss-weight "$MM_SCENE_COUNTERFACTUAL_LOSS_WEIGHT" \
    --scene-counterfactual-action-margin "$MM_SCENE_COUNTERFACTUAL_ACTION_MARGIN" \
    --scene-residual-preservation-weight "$MM_SCENE_RESIDUAL_PRESERVATION_WEIGHT" \
    "$FREEZE_BASE_ACTOR_FLAG" \
    --actor-learning-rate "$MM_ACTOR_LEARNING_RATE" \
    --scene-learning-rate-multiplier "$MM_SCENE_LEARNING_RATE_MULTIPLIER" \
    "$RESET_SCENE_FLAG" \
    --checkpoint "$MM_TEACHER_CHECKPOINT" \
    --save-last-checkpoint-every 100 \
    --eval-metrics-every "$MM_EVAL_METRICS_EVERY" \
    --fixed-motion-eval-batch-size "$MM_FIXED_MOTION_EVAL_BATCH_SIZE" \
    --experiment-path examples/experiments/mimic/mlp_egobody_scene.py \
    --experiment-name "$MM_TEACHER_EXPERIMENT_NAME" \
    --training-max-iterations "$MM_TEACHER_ITERATIONS" \
    --seed 0 \
    --headless \
    "$@" 2>&1 | tee -a "$result_dir/console.log"
