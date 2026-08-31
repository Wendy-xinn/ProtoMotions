#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
EGOBODY_ROOT="${EGOBODY_ROOT:-/home/wenxin/projects/egobody}"
BODY_TEXT_ROOT="${EGOBODY_BODY_TEXT_ROOT:-/home/wenxin/projects/texts/body_texts/EgoBody}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_1000_192}"
MANIFEST="$DATASET_ROOT/manifest.json"
PREPARED_MANIFEST="$DATASET_ROOT/prepared_manifest.json"

mkdir -p "$DATASET_ROOT"
cd "$REPO_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
    EGOBODY_ROOT="$EGOBODY_ROOT" \
    EGOBODY_BODY_TEXT_ROOT="$BODY_TEXT_ROOT" \
        "$REPO_ROOT/scripts/select_egobody_training_sets.sh"
fi

if [[ ! -f "$PREPARED_MANIFEST" ]]; then
    "$PYTHON_BIN" data/scripts/prepare_egobody_sft_manifest.py \
        "$MANIFEST" \
        --egobody-root "$EGOBODY_ROOT" \
        "$@"
fi

if [[ ! -f "$DATASET_ROOT/online_packs_orientation_v1/manifest.json" ]]; then
    "$PYTHON_BIN" data/scripts/build_egobody_online_sft_packs.py \
        --manifest "$PREPARED_MANIFEST" \
        --output-root "$DATASET_ROOT/online_packs_orientation_v1"
fi

if [[ ! -f "$DATASET_ROOT/online_packs_smpl_v1/manifest.json" ]]; then
    "$PYTHON_BIN" data/scripts/build_egobody_online_sft_packs.py \
        --manifest "$PREPARED_MANIFEST" \
        --output-root "$DATASET_ROOT/online_packs_smpl_v1" \
        --motion-filename motion_lib_smpl.pt \
        --camera-filename ego_camera.pt
fi

echo "Prepared shared EgoBody dataset: $DATASET_ROOT"
echo "Use EGOBODY_DATASET_ROOT=$DATASET_ROOT for both training baselines."
