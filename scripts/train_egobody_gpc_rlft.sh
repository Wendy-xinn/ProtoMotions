#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
PACK_ROOT="${EGOBODY_WINDOW_PACK:-$DATASET_ROOT/window_sft_orientation_v1}"
SPLIT="${EGOBODY_SPLIT:-train}"
MANIFEST="$PACK_ROOT/$SPLIT/manifest.json"
SFT_CHECKPOINT="${GPC_RLFT_SFT_CHECKPOINT:-$REPO_ROOT/results/egobody_gpc_dagger_student50_h192_ls05_ema99_v1/best.ckpt}"
NUM_ENVS="${GPC_RLFT_NUM_ENVS:-32}"
ROLLOUT_HORIZON="${GPC_RLFT_ROLLOUT_HORIZON:-32}"
WINDOW_SIZE="${GPC_RLFT_WINDOW_SIZE:-$ROLLOUT_HORIZON}"
BATCH_SIZE="${GPC_RLFT_BATCH_SIZE:-256}"
MAX_STEPS="${GPC_RLFT_MAX_STEPS:-5000000}"
RUN_NAME="${GPC_RLFT_RUN_NAME:-egobody_gpc_rlft_dual_prior_v1}"

for required in "$MANIFEST" "$SFT_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 2
    fi
done

rollout_size=$((NUM_ENVS * ROLLOUT_HORIZON))
if (( BATCH_SIZE < 1 || BATCH_SIZE > rollout_size || rollout_size % BATCH_SIZE != 0 )); then
    echo "GPC_RLFT_BATCH_SIZE must divide num_envs*rollout_horizon=$rollout_size" >&2
    exit 2
fi

readarray -t inputs < <(
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in ("motion_file", "scene_file", "ego_scene_map_file"):
    print(d[key])
PY
)

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO_ROOT"
training_limit_args=(--training-max-steps "$MAX_STEPS")
if [[ -n "${GPC_RLFT_MAX_ITERATIONS:-}" ]]; then
    training_limit_args=(--training-max-iterations "$GPC_RLFT_MAX_ITERATIONS")
fi
ema_args=()
if [[ -n "${GPC_RLFT_PARAMETER_EMA_DECAY:-}" ]]; then
    ema_args=(--rlft-parameter-ema-decay "$GPC_RLFT_PARAMETER_EMA_DECAY")
fi
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
    --window-size-frames "$WINDOW_SIZE" \
    --random-windows-per-clip "${GPC_RLFT_RANDOM_WINDOWS_PER_CLIP:-1}" \
    --window-sampler-seed "${GPC_RLFT_WINDOW_SEED:-0}" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --num-mini-epochs "${GPC_RLFT_MINI_EPOCHS:-2}" \
    --rlft-action-smoothness-weight "${GPC_RLFT_ACTION_SMOOTHNESS_WEIGHT:--0.002}" \
    --rlft-action-acceleration-weight "${GPC_RLFT_ACTION_ACCELERATION_WEIGHT:--0.01}" \
    --rlft-tracking-threshold-bonus-weight "${GPC_RLFT_TRACKING_THRESHOLD_BONUS_WEIGHT:-0.25}" \
    --rlft-tracking-threshold-violation-weight "${GPC_RLFT_TRACKING_THRESHOLD_VIOLATION_WEIGHT:--0.25}" \
    --rlft-actor-lr "${GPC_RLFT_ACTOR_LR:-2e-6}" \
    --rlft-base-prior-top-p "${GPC_RLFT_BASE_PRIOR_TOP_P:-0.99}" \
    --rlft-sft-kl-coeff "${GPC_RLFT_SFT_KL_COEFF:-0.01}" \
    --rlft-target-kl "${GPC_RLFT_TARGET_KL:-0.01}" \
    --rlft-head-orientation-weight "${GPC_RLFT_HEAD_ORIENTATION_WEIGHT:-0.12}" \
    --rlft-body-orientation-weight "${GPC_RLFT_BODY_ORIENTATION_WEIGHT:-0.15}" \
    --rlft-root-orientation-weight "${GPC_RLFT_ROOT_ORIENTATION_WEIGHT:-0.10}" \
    "${ema_args[@]}" \
    --save-last-checkpoint-every "${GPC_RLFT_SAVE_EVERY:-25}" \
    --eval-metrics-every "${GPC_RLFT_EVAL_EVERY:-250}" \
    --fixed-motion-eval-batch-size "${GPC_RLFT_EVAL_BATCH_SIZE:-16}" \
    --experiment-path examples/experiments/gpc/rlft_egobody_scene_head.py \
    --experiment-name "$RUN_NAME" \
    "${training_limit_args[@]}" \
    --checkpoint "$SFT_CHECKPOINT" \
    --headless \
    "$@"
