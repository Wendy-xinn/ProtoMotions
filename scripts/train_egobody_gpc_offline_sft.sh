#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_ROOT="${EGOBODY_OFFLINE_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/offline_sft_50}"
PREPARED_MANIFEST="${EGOBODY_PREPARED_MANIFEST:-$DATASET_ROOT/prepared_manifest.json}"
CACHE_ROOT="${EGOBODY_CACHE_ROOT:-$DATASET_ROOT/expert_cache_grounded_v3}"
HEAD_ORIENTATION_FEEDBACK_GAIN="${HEAD_ORIENTATION_FEEDBACK_GAIN:-0.0}"
OFFLINE_BATCH_SIZE="${OFFLINE_BATCH_SIZE:-128}"
OFFLINE_EPOCHS="${OFFLINE_EPOCHS:-100}"
OFFLINE_EXPERIMENT_NAME="${OFFLINE_EXPERIMENT_NAME:-egobody_gpc_scene_offline_sft_50_grounded_v3}"
CONDITION_MODE="${CONDITION_MODE:-full}"
FSQ_SCALAR_AUX_WEIGHT="${FSQ_SCALAR_AUX_WEIGHT:-0.0}"

if [[ ! -f "$PREPARED_MANIFEST" ]]; then
  echo "Missing prepared manifest: $PREPARED_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "Missing expert cache: $CACHE_ROOT" >&2
  exit 1
fi

readarray -t representative < <(
  "$REPO_ROOT/IsaacLab/.venv/bin/python" - "$PREPARED_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.load(open(sys.argv[1]))
recording = next(
    clip["recording"] for clip in manifest["clips"] if clip["split"] == "train"
)
data_dir = Path(manifest["prepared_root"]) / recording
print(data_dir / "motion_lib_soma23_grounded.pt")
print(data_dir / "scene_lib_training_isaaclab.pt")
print(data_dir / "ego_camera_grounded.pt")
print(manifest["egobody_root"])
print(manifest["frame_count"])
PY
)

cd "$REPO_ROOT"
exec "$REPO_ROOT/scripts/run_with_memory_guard.sh" \
  "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
  "$REPO_ROOT/IsaacLab/.venv/bin/python" protomotions/train_agent.py \
  --robot-name soma23 \
  --simulator isaaclab \
  --num-envs 1 \
  --batch-size "$OFFLINE_BATCH_SIZE" \
  --motion-file "${representative[0]}" \
  --scenes-file "${representative[1]}" \
  --ego-camera-file "${representative[2]}" \
  --scene-asset-root "${representative[3]}" \
  --episode-length "${representative[4]}" \
  --offline-dataset-path "$CACHE_ROOT" \
  --offline-dataset-split train \
  --offline-num-epochs "$OFFLINE_EPOCHS" \
  --head-orientation-feedback-gain "$HEAD_ORIENTATION_FEEDBACK_GAIN" \
  --condition-mode "$CONDITION_MODE" \
  --fsq-scalar-aux-weight "$FSQ_SCALAR_AUX_WEIGHT" \
  --save-last-checkpoint-every 5 \
  --eval-metrics-every 1000000 \
  --experiment-path examples/experiments/gpc/sft_trumans_scene_head_overfit.py \
  --experiment-name "$OFFLINE_EXPERIMENT_NAME" \
  --training-max-steps 1 \
  --headless \
  "$@"
