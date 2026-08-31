#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-$REPO_ROOT/data/motion_for_trackers/trumans_scene_collision_v1/pilot_subsets/train/start_0000_count_0001}"
FRAME_START="${FRAME_START:-0}"
FRAME_COUNT="${FRAME_COUNT:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/motion_for_trackers/trumans_scene_collision_v1/soma23_ego_offline_overfit_v1/start_0000_frame_${FRAME_START}_count_${FRAME_COUNT}}"
OVERFIT_NUM_ENVS="${OVERFIT_NUM_ENVS:-4}"
OVERFIT_BATCH_SIZE="${OVERFIT_BATCH_SIZE:-128}"
OVERFIT_STEPS="${OVERFIT_STEPS:-200000}"
OVERFIT_EXPERIMENT_NAME="${OVERFIT_EXPERIMENT_NAME:-trumans_ego_fov_memory_256_overfit_v1}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ ! -f "$OUTPUT_DIR/source_motion_lib.pt" ]]; then
    "$PYTHON_BIN" scripts/subset_motion_lib.py \
        "$SOURCE_DIR/motion_lib.pt" "$OUTPUT_DIR/source_motion_lib.pt" \
        --indices 0 --frame-start "$FRAME_START" --frame-count "$FRAME_COUNT"
fi
if [[ ! -f "$OUTPUT_DIR/motion_lib.pt" ]]; then
    "$PYTHON_BIN" data/scripts/retarget_packaged_smpl_to_soma23.py \
        "$OUTPUT_DIR/source_motion_lib.pt" "$OUTPUT_DIR/motion_lib.pt" \
        --ik-iterations 0
fi
# Crop dynamic-object trajectories by the same source-frame interval. Static
# reconstructed geometry remains unchanged and is known for the full clip.
"$PYTHON_BIN" scripts/subset_scene_lib.py \
    "$SOURCE_DIR/scene_lib.pt" "$OUTPUT_DIR/scene_lib.pt" \
    --indices 0 --frame-start "$FRAME_START" --frame-count "$FRAME_COUNT" \
    --motion-fps 30 --freeze-near-static

exec scripts/run_with_memory_guard.sh \
    scripts/run_wsl_isaaclab.sh "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --num-envs "$OVERFIT_NUM_ENVS" \
    --batch-size "$OVERFIT_BATCH_SIZE" \
    --motion-file "$OUTPUT_DIR/motion_lib.pt" \
    --scenes-file "$OUTPUT_DIR/scene_lib.pt" \
    --experiment-path examples/experiments/gpc/sft_trumans_scene_head_overfit.py \
    --experiment-name "$OVERFIT_EXPERIMENT_NAME" \
    --training-max-steps "$OVERFIT_STEPS" \
    --headless \
    "$@"
