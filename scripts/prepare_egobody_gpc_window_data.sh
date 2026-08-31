#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
SOURCE_PACK="${EGOBODY_SOURCE_PACK:-$DATASET_ROOT/online_packs_orientation_v1}"
OUTPUT_PACK="${EGOBODY_WINDOW_PACK:-$DATASET_ROOT/window_sft_orientation_v1}"
ASSET_ROOT="${EGOBODY_ASSET_ROOT:-/home/wenxin/projects/egobody}"
SPLIT="${EGOBODY_SPLIT:-train}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" data/scripts/build_egobody_gt_ego_scene_maps.py \
    --pack-root "$SOURCE_PACK" \
    --split "$SPLIT" \
    --output-root "$OUTPUT_PACK" \
    --asset-root "$ASSET_ROOT" \
    --candidate-points "${GPC_SCENE_CANDIDATES:-8192}" \
    --output-points "${GPC_SCENE_POINTS:-256}" \
    --pointcloud-workers "${GPC_SCENE_WORKERS:-8}" \
    --seed "${GPC_SCENE_SEED:-0}" \
    --device "${GPC_SCENE_DEVICE:-cuda}"
