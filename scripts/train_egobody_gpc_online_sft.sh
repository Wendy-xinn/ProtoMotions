#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/offline_sft_50}"
PREPARED_MANIFEST="${EGOBODY_PREPARED_MANIFEST:-$DATASET_ROOT/prepared_manifest.json}"
ONLINE_PACK_ROOT="${EGOBODY_ONLINE_PACK_ROOT:-$DATASET_ROOT/online_packs_orientation_v1}"
ONLINE_SPLIT="${ONLINE_SPLIT:-train}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-128}"
ONLINE_ITERATIONS="${ONLINE_ITERATIONS:-500}"
ONLINE_MINI_EPOCHS="${ONLINE_MINI_EPOCHS:-1}"
ONLINE_POINTCLOUD_CANDIDATES="${ONLINE_POINTCLOUD_CANDIDATES:-256}"
ONLINE_POINTCLOUD_SEED="${ONLINE_POINTCLOUD_SEED:-0}"
ONLINE_EXPERIMENT_NAME="${ONLINE_EXPERIMENT_NAME:-egobody_gpc_online_sft_40_head_feedback_v1}"
ONLINE_CHECKPOINT="${ONLINE_CHECKPOINT:-}"
HEAD_ORIENTATION_FEEDBACK_GAIN="${HEAD_ORIENTATION_FEEDBACK_GAIN:-1.0}"
CONDITION_MODE="${CONDITION_MODE:-full}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$PREPARED_MANIFEST" ]]; then
    echo "Missing prepared manifest: $PREPARED_MANIFEST" >&2
    exit 2
fi

if [[ ! -f "$ONLINE_PACK_ROOT/manifest.json" ]]; then
    "$PYTHON_BIN" data/scripts/build_egobody_online_sft_packs.py \
        --manifest "$PREPARED_MANIFEST" \
        --output-root "$ONLINE_PACK_ROOT"
fi

split_manifest="$ONLINE_PACK_ROOT/$ONLINE_SPLIT/manifest.json"
if [[ ! -f "$split_manifest" ]]; then
    echo "Missing online split: $split_manifest" >&2
    exit 2
fi

readarray -t online_inputs < <(
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

num_envs="${ONLINE_NUM_ENVS:-${online_inputs[4]}}"
if (( num_envs < online_inputs[4] )); then
    echo "ONLINE_NUM_ENVS must be >= the ${online_inputs[4]} motion-paired scenes." >&2
    exit 3
fi
rollout_samples=$((num_envs * 32))
if (( ONLINE_BATCH_SIZE > rollout_samples || rollout_samples % ONLINE_BATCH_SIZE != 0 )); then
    echo "ONLINE_BATCH_SIZE must divide num_envs*32=$rollout_samples." >&2
    exit 3
fi

checkpoint_args=()
if [[ -n "$ONLINE_CHECKPOINT" ]]; then
    checkpoint_args+=(
        --checkpoint "$ONLINE_CHECKPOINT"
        --resume-training-max-iterations "$ONLINE_ITERATIONS"
    )
fi

result_dir="$REPO_ROOT/results/$ONLINE_EXPERIMENT_NAME"
mkdir -p "$result_dir"
echo "EgoBody online expert SFT: split=$ONLINE_SPLIT envs=$num_envs iterations=$ONLINE_ITERATIONS"
echo "batch=$ONLINE_BATCH_SIZE mini_epochs=$ONLINE_MINI_EPOCHS pointcloud_seed=$ONLINE_POINTCLOUD_SEED"

cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/run_with_memory_guard.sh" \
    "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
    "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --num-envs "$num_envs" \
    --batch-size "$ONLINE_BATCH_SIZE" \
    --motion-file "${online_inputs[0]}" \
    --scenes-file "${online_inputs[1]}" \
    --ego-camera-file "${online_inputs[2]}" \
    --scene-asset-root "${online_inputs[3]}" \
    --episode-length 192 \
    --num-mini-epochs "$ONLINE_MINI_EPOCHS" \
    --head-orientation-feedback-gain "$HEAD_ORIENTATION_FEEDBACK_GAIN" \
    --condition-mode "$CONDITION_MODE" \
    --scene-pointcloud-candidates "$ONLINE_POINTCLOUD_CANDIDATES" \
    --scene-pointcloud-seed "$ONLINE_POINTCLOUD_SEED" \
    --save-last-checkpoint-every 25 \
    --eval-metrics-every 1000000 \
    --experiment-path examples/experiments/gpc/sft_trumans_scene_head_overfit.py \
    --experiment-name "$ONLINE_EXPERIMENT_NAME" \
    --training-max-iterations "$ONLINE_ITERATIONS" \
    --headless \
    "${checkpoint_args[@]}" \
    "$@" 2>&1 | tee -a "$result_dir/console.log"
