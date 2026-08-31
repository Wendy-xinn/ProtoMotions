#!/usr/bin/env bash
set -euo pipefail

# Full TRUMANS reprocessing entry point:
# 1) rebuild the clip manifest from the raw release;
# 2) run the collision-mesh preprocessing pipeline.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
TRUMANS_ROOT="${TRUMANS_ROOT:-$REPO_ROOT/../TRUMANS}"
MANIFEST_OUT="${MANIFEST_OUT:-$TRUMANS_ROOT/processed/scene_expert_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[1/2] Build TRUMANS scene manifest"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" data/scripts/trumans/build_scene_manifest.py \
  --root "$TRUMANS_ROOT" \
  --output-dir "$MANIFEST_OUT"

echo "[2/2] Run collision-mesh preprocessing"
"$SCRIPT_DIR/prepare_trumans_scene_collision_v1.sh" "$@"
