#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
PACK_ROOT="${EGOBODY_WINDOW_PACK:-$DATASET_ROOT/window_sft_orientation_v1}"
SPLIT="${EGOBODY_SPLIT:-train}"
MANIFEST="$PACK_ROOT/$SPLIT/manifest.json"
NUM_ENVS="${GPC_NUM_ENVS:-64}"
ROLLOUT_HORIZON="${GPC_ROLLOUT_HORIZON:-32}"
BATCH_SIZE="${GPC_BATCH_SIZE:-1024}"
MAX_STEPS="${GPC_MAX_STEPS:-20000000}"
RUN_NAME="${GPC_RUN_NAME:-egobody_gpc_window_sft_v1}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing window pack: $MANIFEST" >&2
    echo "Run scripts/prepare_egobody_gpc_window_data.sh first." >&2
    exit 2
fi
"$PYTHON_BIN" data/scripts/validate_egobody_gpc_window_pack.py \
    --manifest "$MANIFEST" \
    --expected-frames 192 \
    --expected-points 256
rollout_size=$((NUM_ENVS * ROLLOUT_HORIZON))
if (( BATCH_SIZE < 1 || BATCH_SIZE > rollout_size || rollout_size % BATCH_SIZE != 0 )); then
    echo "GPC_BATCH_SIZE must divide num_envs*rollout_horizon=$rollout_size" >&2
    exit 2
fi

readarray -t inputs < <(
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
for key in ("motion_file", "scene_file", "ego_scene_map_file"):
    print(d[key])
PY
)

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO_ROOT"
exec scripts/run_with_memory_guard.sh \
    scripts/run_wsl_isaaclab.sh \
    "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --num-envs "$NUM_ENVS" \
    --batch-size "$BATCH_SIZE" \
    --motion-file "${inputs[0]}" \
    --scenes-file "${inputs[1]}" \
    --scene-asset-root "${EGOBODY_ASSET_ROOT:-/home/wenxin/projects/egobody}" \
    --ego-scene-map-file "${inputs[2]}" \
    --window-sampling-manifest "$MANIFEST" \
    --window-size-frames 32 \
    --random-windows-per-clip "${GPC_RANDOM_WINDOWS_PER_CLIP:-1}" \
    --window-sampler-seed "${GPC_WINDOW_SEED:-0}" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --num-mini-epochs "${GPC_MINI_EPOCHS:-1}" \
    --save-last-checkpoint-every "${GPC_SAVE_EVERY:-25}" \
    --eval-metrics-every "${GPC_EVAL_EVERY:-250}" \
    --experiment-path examples/experiments/gpc/sft_trumans_scene_head_overfit.py \
    --experiment-name "$RUN_NAME" \
    --training-max-steps "$MAX_STEPS" \
    --headless \
    "$@"
