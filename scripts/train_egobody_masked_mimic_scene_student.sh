#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
PREPARED_MANIFEST="${EGOBODY_PREPARED_MANIFEST:-$DATASET_ROOT/prepared_manifest.json}"
PACK_ROOT="${EGOBODY_SMPL_PACK_ROOT:-$DATASET_ROOT/online_packs_smpl_v1}"
MM_SPLIT="${MM_SPLIT:-train}"
MM_STUDENT_ITERATIONS="${MM_STUDENT_ITERATIONS:-5000}"
MM_STUDENT_BATCH_SIZE="${MM_STUDENT_BATCH_SIZE:-100}"
MM_STUDENT_EXPERIMENT_NAME="${MM_STUDENT_EXPERIMENT_NAME:-egobody_smpl_masked_mimic_scene_student_800_v1}"
MM_TEACHER_CHECKPOINT="${MM_TEACHER_CHECKPOINT:-${MM_SCENE_TEACHER_CHECKPOINT:-$REPO_ROOT/data/pretrained_models/motion_tracker/smpl/last.ckpt}}"
MM_DISTILL_EXPERT_INTERACTIONS="${MM_DISTILL_EXPERT_INTERACTIONS:-0}"
MM_DEPLOYABLE_ACTION_LOSS_WEIGHT="${MM_DEPLOYABLE_ACTION_LOSS_WEIGHT:-0.25}"
MM_DEPLOYABLE_ROLLOUT_START_EPOCH="${MM_DEPLOYABLE_ROLLOUT_START_EPOCH:-500}"
MM_DEPLOYABLE_ROLLOUT_END_EPOCH="${MM_DEPLOYABLE_ROLLOUT_END_EPOCH:-3000}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "$MM_TEACHER_CHECKPOINT" || ! -f "$MM_TEACHER_CHECKPOINT" ]]; then
    echo "Set MM_TEACHER_CHECKPOINT to a full-body reference-tracking teacher checkpoint." >&2
    exit 2
fi

split_manifest="$PACK_ROOT/$MM_SPLIT/manifest.json"
if [[ ! -f "$split_manifest" || ! -f "$PREPARED_MANIFEST" ]]; then
    echo "Missing EgoBody SMPL pack or prepared manifest." >&2
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
print(split["ego_camera_file"])
print(prepared["egobody_root"])
print(split["num_motions"])
PY
)

num_envs="${MM_STUDENT_NUM_ENVS:-${inputs[4]}}"
rollout_samples=$((num_envs * 32))
if (( MM_STUDENT_BATCH_SIZE > rollout_samples || rollout_samples % MM_STUDENT_BATCH_SIZE != 0 )); then
    echo "MM_STUDENT_BATCH_SIZE must divide num_envs*32=$rollout_samples." >&2
    exit 3
fi

result_dir="$REPO_ROOT/results/$MM_STUDENT_EXPERIMENT_NAME"
mkdir -p "$result_dir"
echo "SMPL scene-aware MaskedMimic student: split=$MM_SPLIT envs=$num_envs"
echo "teacher=$MM_TEACHER_CHECKPOINT"
echo "distill_expert_interactions=$MM_DISTILL_EXPERT_INTERACTIONS"
echo "deployable_action_loss=$MM_DEPLOYABLE_ACTION_LOSS_WEIGHT rollout_schedule=$MM_DEPLOYABLE_ROLLOUT_START_EPOCH..$MM_DEPLOYABLE_ROLLOUT_END_EPOCH"

DISTILL_INTERACTIONS_FLAG=()
if [[ "$MM_DISTILL_EXPERT_INTERACTIONS" == "1" ]]; then
    DISTILL_INTERACTIONS_FLAG+=(--distill-expert-interactions)
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/run_with_memory_guard.sh" \
    "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
    "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaaclab \
    --num-envs "$num_envs" \
    --batch-size "$MM_STUDENT_BATCH_SIZE" \
    --motion-file "${inputs[0]}" \
    --scenes-file "${inputs[1]}" \
    --ego-camera-file "${inputs[2]}" \
    --scene-asset-root "${inputs[3]}" \
    --expert-model-path "$MM_TEACHER_CHECKPOINT" \
    --deployable-action-loss-weight "$MM_DEPLOYABLE_ACTION_LOSS_WEIGHT" \
    --deployable-rollout-start-epoch "$MM_DEPLOYABLE_ROLLOUT_START_EPOCH" \
    --deployable-rollout-end-epoch "$MM_DEPLOYABLE_ROLLOUT_END_EPOCH" \
    "${DISTILL_INTERACTIONS_FLAG[@]}" \
    --experiment-path examples/experiments/masked_mimic/egobody_scene.py \
    --experiment-name "$MM_STUDENT_EXPERIMENT_NAME" \
    --training-max-iterations "$MM_STUDENT_ITERATIONS" \
    --seed 0 \
    --headless \
    "$@" 2>&1 | tee -a "$result_dir/console.log"
