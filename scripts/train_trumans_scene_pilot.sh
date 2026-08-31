#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ "${TRUMANS_REQUIRE_DATA_APPROVAL:-0}" == "1" && "${TRUMANS_DATA_APPROVED:-0}" != "1" ]]; then
    echo "TRUMANS training is gated by TRUMANS_REQUIRE_DATA_APPROVAL=1." >&2
    echo "After manual approval, rerun with: TRUMANS_DATA_APPROVED=1 $0" >&2
    exit 3
fi

# Fail-safe defaults: prove one scene can initialize and complete PPO updates
# before increasing scene count or parallel environments on WSL.
PILOT_NUM_ENVS="${PILOT_NUM_ENVS:-1}"
PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-32}"
PILOT_ITERATIONS="${PILOT_ITERATIONS:-20}"
PILOT_EXPERIMENT_NAME="${PILOT_EXPERIMENT_NAME:-trumans_scene_expert_pilot}"
PILOT_DATA_ROOT="${PILOT_DATA_ROOT:-$REPO_ROOT/data/motion_for_trackers/trumans_scene_collision_v1}"
PILOT_SPLIT="${PILOT_SPLIT:-train}"
# Use the final collision-mesh scene pack. The new name is scene_libs so it is
# obvious that this is the packaged training input.
PILOT_SCENE_FILE_DEFAULT="$PILOT_DATA_ROOT/scene_libs/$PILOT_SPLIT.pt"
if [[ ! -f "$PILOT_SCENE_FILE_DEFAULT" && -f "$PILOT_DATA_ROOT/mesh_collision_scenes/$PILOT_SPLIT.pt" ]]; then
    PILOT_SCENE_FILE_DEFAULT="$PILOT_DATA_ROOT/mesh_collision_scenes/$PILOT_SPLIT.pt"
fi
PILOT_SCENE_FILE="${PILOT_SCENE_FILE:-$PILOT_SCENE_FILE_DEFAULT}"
PILOT_SCENE_POINTCLOUD_CANDIDATES="${PILOT_SCENE_POINTCLOUD_CANDIDATES:-512}"
PILOT_SCENE_POINTCLOUD_WORKERS="${PILOT_SCENE_POINTCLOUD_WORKERS:-1}"
PILOT_SCENE_START_INDEX="${PILOT_SCENE_START_INDEX:-0}"
PILOT_SCENE_LOAD_COUNT="${PILOT_SCENE_LOAD_COUNT:-1}"
PILOT_ALLOW_FULL_SCENE_LIB="${PILOT_ALLOW_FULL_SCENE_LIB:-0}"
PILOT_USE_PACKAGED_SUBSET="${PILOT_USE_PACKAGED_SUBSET:-1}"
PILOT_SCENE_ASSET_ROOT="${PILOT_SCENE_ASSET_ROOT:-$REPO_ROOT/../TRUMANS}"
PILOT_BASE_CHECKPOINT="${PILOT_BASE_CHECKPOINT:-$REPO_ROOT/data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt}"
# The full split has already passed validation. Keeping this off avoids loading
# the multi-gigabyte full packs again on every safe-pilot restart.
PILOT_VALIDATE_INPUTS="${PILOT_VALIDATE_INPUTS:-0}"

if (( PILOT_SCENE_LOAD_COUNT == 0 )) && [[ "$PILOT_ALLOW_FULL_SCENE_LIB" != "1" ]]; then
    echo "Refusing to load the full SceneLib in memory-safe pilot mode." >&2
    echo "Set PILOT_ALLOW_FULL_SCENE_LIB=1 only after the small pilot is stable." >&2
    exit 4
fi
if (( PILOT_SCENE_LOAD_COUNT > PILOT_NUM_ENVS )); then
    echo "PILOT_SCENE_LOAD_COUNT must be <= PILOT_NUM_ENVS for a memory-safe pilot." >&2
    exit 4
fi
PILOT_ROLLOUT_SAMPLES=$((PILOT_NUM_ENVS * 32))
if (( PILOT_BATCH_SIZE > PILOT_ROLLOUT_SAMPLES || PILOT_ROLLOUT_SAMPLES % PILOT_BATCH_SIZE != 0 )); then
    echo "PILOT_BATCH_SIZE must divide num_envs*32 rollout samples and cannot exceed it." >&2
    echo "Current: batch=$PILOT_BATCH_SIZE, rollout_samples=$PILOT_ROLLOUT_SAMPLES" >&2
    exit 4
fi

cd "$REPO_ROOT"

echo "TRUMANS safe pilot: envs=$PILOT_NUM_ENVS batch=$PILOT_BATCH_SIZE iterations=$PILOT_ITERATIONS"
echo "SceneLib subset: start=$PILOT_SCENE_START_INDEX count=$PILOT_SCENE_LOAD_COUNT; point candidates=$PILOT_SCENE_POINTCLOUD_CANDIDATES"

if [[ "$PILOT_VALIDATE_INPUTS" == "1" ]]; then
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" data/scripts/validate_trumans_scene_training_inputs.py \
        --data-root "$PILOT_DATA_ROOT" \
        --split "$PILOT_SPLIT" \
        --motion-file "$PILOT_DATA_ROOT/motion_libs/$PILOT_SPLIT.pt" \
        --scene-file "$PILOT_SCENE_FILE" \
        --scene-asset-root "$PILOT_SCENE_ASSET_ROOT" \
        --checkpoint "$PILOT_BASE_CHECKPOINT"
fi

PILOT_MOTION_FILE="$PILOT_DATA_ROOT/motion_libs/$PILOT_SPLIT.pt"
PILOT_RUNTIME_SCENE_FILE="$PILOT_SCENE_FILE"
PILOT_RUNTIME_SCENE_START_INDEX="$PILOT_SCENE_START_INDEX"

if [[ "$PILOT_USE_PACKAGED_SUBSET" == "1" && "$PILOT_SCENE_LOAD_COUNT" -gt 0 ]]; then
    subset_tag="start_$(printf '%04d' "$PILOT_SCENE_START_INDEX")_count_$(printf '%04d' "$PILOT_SCENE_LOAD_COUNT")"
    subset_dir="$PILOT_DATA_ROOT/pilot_subsets/$PILOT_SPLIT/$subset_tag"
    subset_motion_file="$subset_dir/motion_lib.pt"
    subset_scene_file="$subset_dir/scene_lib.pt"
    subset_complete="$subset_dir/.complete"

    if [[ ! -f "$subset_complete" ]]; then
        mkdir -p "$subset_dir"
        subset_indices=()
        for ((index=PILOT_SCENE_START_INDEX; index<PILOT_SCENE_START_INDEX+PILOT_SCENE_LOAD_COUNT; index++)); do
            subset_indices+=("$index")
        done
        echo "Building one-time aligned pilot subset: ${subset_indices[*]}"
        PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/subset_motion_lib.py \
            "$PILOT_MOTION_FILE" "$subset_motion_file" \
            --indices "${subset_indices[@]}"
        PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/subset_scene_lib.py \
            "$PILOT_SCENE_FILE" "$subset_scene_file" \
            --indices "${subset_indices[@]}"
        touch "$subset_complete"
    fi

    PILOT_MOTION_FILE="$subset_motion_file"
    PILOT_RUNTIME_SCENE_FILE="$subset_scene_file"
    # The packaged subset has been remapped to local indices 0..N-1.
    PILOT_RUNTIME_SCENE_START_INDEX=0
    echo "Runtime pilot packs: $PILOT_MOTION_FILE and $PILOT_RUNTIME_SCENE_FILE"
fi

exec scripts/run_wsl_isaaclab.sh "$PYTHON_BIN" protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaaclab \
    --num-envs "$PILOT_NUM_ENVS" \
    --batch-size "$PILOT_BATCH_SIZE" \
    --motion-file "$PILOT_MOTION_FILE" \
    --scenes-file "$PILOT_RUNTIME_SCENE_FILE" \
    --scene-asset-root "$PILOT_SCENE_ASSET_ROOT" \
    --scene-pointcloud-candidates "$PILOT_SCENE_POINTCLOUD_CANDIDATES" \
    --scene-pointcloud-workers "$PILOT_SCENE_POINTCLOUD_WORKERS" \
    --scene-start-index "$PILOT_RUNTIME_SCENE_START_INDEX" \
    --scene-load-count "$PILOT_SCENE_LOAD_COUNT" \
    --experiment-path examples/experiments/mimic/mlp_trumans_scene.py \
    --experiment-name "$PILOT_EXPERIMENT_NAME" \
    --checkpoint "$PILOT_BASE_CHECKPOINT" \
    --training-max-iterations "$PILOT_ITERATIONS" \
    --headless \
    "$@"
