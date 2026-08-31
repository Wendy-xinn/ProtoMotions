#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
RECORDING="${EGOBODY_RECORDING:-recording_20211002_S17_S15_01}"
DATA_DIR="$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/$RECORDING"
OVERFIT_NUM_ENVS="${OVERFIT_NUM_ENVS:-4}"
OVERFIT_BATCH_SIZE="${OVERFIT_BATCH_SIZE:-128}"
OVERFIT_STEPS="${OVERFIT_STEPS:-1800000}"
OVERFIT_EXPERIMENT_NAME="${OVERFIT_EXPERIMENT_NAME:-egobody_ego_fov_memory_grounded_warm6000_v1}"
OVERFIT_NUM_MINI_EPOCHS="${OVERFIT_NUM_MINI_EPOCHS:-4}"
OVERFIT_SAVE_INTERVAL="${OVERFIT_SAVE_INTERVAL:-100}"
OVERFIT_EVAL_INTERVAL="${OVERFIT_EVAL_INTERVAL:-100}"
OVERFIT_MOTION_FILE="${OVERFIT_MOTION_FILE:-$DATA_DIR/motion_lib_soma23_grounded.pt}"
OVERFIT_EGO_CAMERA_FILE="${OVERFIT_EGO_CAMERA_FILE:-$DATA_DIR/ego_camera_grounded.pt}"
OVERFIT_CHECKPOINT="${OVERFIT_CHECKPOINT:-$REPO_ROOT/results/egobody_ego_fov_memory_overfit_v1/epoch_6000.ckpt}"

required=(
  "$OVERFIT_MOTION_FILE"
  "$DATA_DIR/scene_lib_training_isaaclab.pt"
  "$OVERFIT_EGO_CAMERA_FILE"
)
if [[ "$OVERFIT_CHECKPOINT" != "none" ]]; then
  required+=("$OVERFIT_CHECKPOINT")
fi
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

checkpoint_args=()
if [[ "$OVERFIT_CHECKPOINT" != "none" ]]; then
  checkpoint_args+=(--checkpoint "$OVERFIT_CHECKPOINT")
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/run_with_memory_guard.sh" \
  "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
  "$PYTHON_BIN" "$REPO_ROOT/protomotions/train_agent.py" \
  --robot-name soma23 \
  --simulator isaaclab \
  --num-envs "$OVERFIT_NUM_ENVS" \
  --batch-size "$OVERFIT_BATCH_SIZE" \
  --motion-file "$OVERFIT_MOTION_FILE" \
  --scenes-file "$DATA_DIR/scene_lib_training_isaaclab.pt" \
  --scene-asset-root /home/wenxin/projects/egobody \
  --ego-camera-file "$OVERFIT_EGO_CAMERA_FILE" \
  --episode-length 192 \
  --num-mini-epochs "$OVERFIT_NUM_MINI_EPOCHS" \
  --save-last-checkpoint-every "$OVERFIT_SAVE_INTERVAL" \
  --eval-metrics-every "$OVERFIT_EVAL_INTERVAL" \
  --experiment-path "$REPO_ROOT/examples/experiments/gpc/sft_trumans_scene_head_overfit.py" \
  --experiment-name "$OVERFIT_EXPERIMENT_NAME" \
  --training-max-steps "$OVERFIT_STEPS" \
  --headless \
  "${checkpoint_args[@]}" \
  "$@"
