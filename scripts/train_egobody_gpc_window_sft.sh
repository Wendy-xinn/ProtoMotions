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
WINDOW_SIZE="${GPC_WINDOW_SIZE:-$ROLLOUT_HORIZON}"
BATCH_SIZE="${GPC_BATCH_SIZE:-1024}"
MAX_STEPS="${GPC_MAX_STEPS:-20000000}"
RUN_NAME="${GPC_RUN_NAME:-egobody_gpc_dagger800_beta_v1}"

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
training_limit_args=(--training-max-steps "$MAX_STEPS")
if [[ -n "${GPC_MAX_ITERATIONS:-}" ]]; then
    training_limit_args=(--training-max-iterations "$GPC_MAX_ITERATIONS")
fi
checkpoint_args=()
if [[ -n "${GPC_CHECKPOINT:-}" ]]; then
    checkpoint_args=(--checkpoint "$GPC_CHECKPOINT")
fi
termination_args=(--no-tracking-error-termination)
if [[ "${GPC_TRACKING_ERROR_TERMINATION:-0}" == "1" ]]; then
    termination_args=(--tracking-error-termination)
fi
ema_args=()
if [[ -n "${GPC_SFT_PARAMETER_EMA_DECAY:-}" ]]; then
    ema_args=(--sft-parameter-ema-decay "$GPC_SFT_PARAMETER_EMA_DECAY")
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
    --random-windows-per-clip "${GPC_RANDOM_WINDOWS_PER_CLIP:-1}" \
    --window-sampler-seed "${GPC_WINDOW_SEED:-0}" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --num-mini-epochs "${GPC_MINI_EPOCHS:-1}" \
    --fsq-scalar-aux-weight "${GPC_FSQ_SCALAR_AUX_WEIGHT:-0.5}" \
    --sequence-action-loss-weight "${GPC_SEQUENCE_ACTION_LOSS_WEIGHT:-0.0}" \
    --sequence-velocity-loss-weight "${GPC_SEQUENCE_VELOCITY_LOSS_WEIGHT:-0.0}" \
    --sequence-acceleration-loss-weight "${GPC_SEQUENCE_ACCELERATION_LOSS_WEIGHT:-0.0}" \
    --sequence-action-loss-beta "${GPC_SEQUENCE_ACTION_LOSS_BETA:-0.05}" \
    --autoregressive-student-prefix-rate "${GPC_AUTOREGRESSIVE_STUDENT_PREFIX_RATE:-0.0}" \
    --sft-actor-lr "${GPC_SFT_ACTOR_LR:-3e-4}" \
    --sft-label-smoothing "${GPC_SFT_LABEL_SMOOTHING:-0.01}" \
    "${ema_args[@]}" \
    --token-perturb-rate "${GPC_TOKEN_PERTURB_RATE:-0.1}" \
    --token-perturb-mode "${GPC_TOKEN_PERTURB_MODE:-neighbor}" \
    --sft-rollout-actor "${GPC_SFT_ROLLOUT_ACTOR:-mixed}" \
    --dagger-beta-schedule ${GPC_DAGGER_BETA_SCHEDULE:-0.95 0.90 0.80 0.70} \
    --dagger-success-thresholds ${GPC_DAGGER_SUCCESS_THRESHOLDS:-0.25 0.50 0.70} \
    --dagger-max-student-run "${GPC_DAGGER_MAX_STUDENT_RUN:-4}" \
    --save-last-checkpoint-every "${GPC_SAVE_EVERY:-25}" \
    --eval-metrics-every "${GPC_EVAL_EVERY:-250}" \
    --fixed-motion-eval-batch-size "${GPC_EVAL_BATCH_SIZE:-16}" \
    --sft-score-component "${GPC_SFT_SCORE_COMPONENT:-gt_error}" \
    --experiment-path examples/experiments/gpc/sft_trumans_scene_head_overfit.py \
    --experiment-name "$RUN_NAME" \
    "${training_limit_args[@]}" \
    "${checkpoint_args[@]}" \
    "${termination_args[@]}" \
    --headless \
    "$@"
