#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wenxin/miniconda3/envs/crisp/bin/python}"
RECORDING="${EGOBODY_RECORDING:-recording_20211002_S17_S15_01}"
DATA_DIR="$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/$RECORDING"
PORT="${VISER_PORT:-8080}"
MOTION_INDEX="${VISER_MOTION_INDEX:-0}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" examples/visualize_smpl_motion.py \
  --motion-file "$DATA_DIR/motion_lib_soma23_grounded.pt" \
  --robot soma23 \
  --initial-motion-index "$MOTION_INDEX" \
  --compare "$DATA_DIR/motion_lib_soma23.pt" \
  --compare-robot soma23 \
  --scene-pt "$DATA_DIR/scene_lib.pt" \
  --scene-asset-root /home/wenxin/projects/egobody \
  --ego-camera-pt "$DATA_DIR/ego_camera_grounded.pt" \
  --show-ego-visibility \
  --ego-max-scene-points 8000 \
  --ego-visibility-stride 2 \
  --offset 0 \
  --port "$PORT" \
  "$@"
