#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
DATASET_ROOT="${EGOBODY_DATASET_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192}"
PREPARED_MANIFEST="${EGOBODY_PREPARED_MANIFEST:-$DATASET_ROOT/prepared_manifest.json}"
PACK_ROOT="${EGOBODY_SMPL_PACK_ROOT:-$DATASET_ROOT/online_packs_smpl_v1}"
MM_EVAL_SPLIT="${MM_EVAL_SPLIT:-val}"
MM_STUDENT_CHECKPOINT="${MM_STUDENT_CHECKPOINT:-}"
MM_EVAL_BATCH_SIZE="${MM_EVAL_BATCH_SIZE:-25}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "$MM_STUDENT_CHECKPOINT" || ! -f "$MM_STUDENT_CHECKPOINT" ]]; then
    echo "Set MM_STUDENT_CHECKPOINT to a trained scene-aware MaskedMimic checkpoint." >&2
    exit 2
fi

split_manifest="$PACK_ROOT/$MM_EVAL_SPLIT/manifest.json"
if [[ ! -f "$split_manifest" || ! -f "$PREPARED_MANIFEST" ]]; then
    echo "Missing EgoBody split or prepared manifest: $split_manifest" >&2
    exit 2
fi

readarray -t inputs < <(
    "$PYTHON_BIN" - "$split_manifest" "$PREPARED_MANIFEST" <<'PY'
import json
import sys

split = json.load(open(sys.argv[1]))
prepared = json.load(open(sys.argv[2]))
print(split["motion_file"])
print(split["scene_file"])
print(split["ego_camera_file"])
print(prepared["egobody_root"])
print(split["num_motions"])
PY
)

num_envs="${MM_EVAL_NUM_ENVS:-${inputs[4]}}"
eval_root="$(dirname -- "$MM_STUDENT_CHECKPOINT")/scene_ablation_${MM_EVAL_SPLIT}"
mkdir -p "$eval_root"

run_eval() {
    local mode="$1"
    local intervention_args=()
    if [[ "$mode" != "matched" ]]; then
        intervention_args+=(
            --policy-observation-intervention "$mode"
            --policy-observation-intervention-keys ego_visible_scene_pointcloud
        )
    fi
    echo "Running scene ablation: mode=$mode split=$MM_EVAL_SPLIT envs=$num_envs"
    "$REPO_ROOT/scripts/run_wsl_isaaclab.sh" \
        "$PYTHON_BIN" protomotions/inference_agent.py \
        --checkpoint "$MM_STUDENT_CHECKPOINT" \
        --simulator isaaclab \
        --num-envs "$num_envs" \
        --motion-file "${inputs[0]}" \
        --scenes-file "${inputs[1]}" \
        --ego-camera-file "${inputs[2]}" \
        --scene-asset-root "${inputs[3]}" \
        --fixed-motion-eval-batch-size "$MM_EVAL_BATCH_SIZE" \
        --full-eval \
        --headless \
        "${intervention_args[@]}" 2>&1 | tee "$eval_root/${mode}.log"
}

cd "$REPO_ROOT"
run_eval matched
run_eval zero
run_eval shuffle
