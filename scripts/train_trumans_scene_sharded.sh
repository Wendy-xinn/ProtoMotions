#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"

FULL_RUN_NAME="${FULL_RUN_NAME:-trumans_scene_full_safe_v1}"
FULL_DATA_ROOT="${FULL_DATA_ROOT:-$REPO_ROOT/data/motion_for_trackers/trumans_scene_collision_v1}"
FULL_SPLIT="${FULL_SPLIT:-train}"
FULL_TOTAL_CLIPS="${FULL_TOTAL_CLIPS:-432}"
FULL_SHARD_SIZE="${FULL_SHARD_SIZE:-4}"
FULL_NUM_ENVS="${FULL_NUM_ENVS:-16}"
FULL_ITERATIONS_PER_SHARD="${FULL_ITERATIONS_PER_SHARD:-100}"
FULL_CYCLES="${FULL_CYCLES:-3}"
FULL_POINTCLOUD_CANDIDATES="${FULL_POINTCLOUD_CANDIDATES:-512}"
FULL_BASE_CHECKPOINT="${FULL_BASE_CHECKPOINT:-$REPO_ROOT/data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt}"
FULL_PREPARE_SHARDS="${FULL_PREPARE_SHARDS:-1}"
FULL_PREPARE_COLLISION_ASSETS="${FULL_PREPARE_COLLISION_ASSETS:-1}"

if (( FULL_SHARD_SIZE < 1 || FULL_SHARD_SIZE > 4 )); then
    echo "FULL_SHARD_SIZE must remain in [1, 4] on the 30-GiB WSL host." >&2
    exit 4
fi
if (( FULL_NUM_ENVS < FULL_SHARD_SIZE )); then
    echo "FULL_NUM_ENVS must be >= FULL_SHARD_SIZE." >&2
    exit 4
fi
if (( FULL_NUM_ENVS % FULL_SHARD_SIZE != 0 )); then
    echo "FULL_NUM_ENVS must be divisible by FULL_SHARD_SIZE for balanced replication." >&2
    exit 4
fi
if (( FULL_TOTAL_CLIPS < 1 )); then
    echo "FULL_TOTAL_CLIPS must be positive." >&2
    exit 4
fi
if (( FULL_CYCLES < 1 )); then
    echo "FULL_CYCLES must be positive." >&2
    exit 4
fi
if (( FULL_ITERATIONS_PER_SHARD < 10 || FULL_ITERATIONS_PER_SHARD % 10 != 0 )); then
    echo "FULL_ITERATIONS_PER_SHARD must be >= 10 and divisible by 10." >&2
    echo "This guarantees that every completed shard has a current last.ckpt." >&2
    exit 4
fi

cd "$REPO_ROOT"
shard_root="$FULL_DATA_ROOT/pilot_subsets/$FULL_SPLIT"
prepared_marker=$(printf '%s/.prepared_shard_size_%04d' "$shard_root" "$FULL_SHARD_SIZE")
collision_marker="$FULL_DATA_ROOT/.dynamic_collision_assets_h16_v64_r100000_complete"

if [[ "$FULL_PREPARE_COLLISION_ASSETS" == "1" && ! -f "$collision_marker" ]]; then
    echo "[collision preflight] Completing rigid convex USD assets for all movable objects"
    scripts/run_wsl_isaaclab.sh "$PYTHON_BIN" scripts/convert_obj_scenes_to_usd.py \
        --scene-file "$FULL_DATA_ROOT/scene_libs/$FULL_SPLIT.pt" \
        --asset-root "$REPO_ROOT/../TRUMANS" \
        --bake-dynamic-collision \
        --approximation convexDecomposition \
        --max-convex-hulls 16 \
        --hull-vertex-limit 64 \
        --voxel-resolution 100000
    touch "$collision_marker"
elif [[ -f "$collision_marker" ]]; then
    echo "Reusing validated dynamic collision assets: $collision_marker"
fi

if [[ "$FULL_PREPARE_SHARDS" == "1" && ! -f "$prepared_marker" ]]; then
    echo "[prepare 1/2] Writing all MotionLib shards from one full-pack load"
    "$PYTHON_BIN" scripts/prepare_trumans_training_shards.py \
        --kind motions \
        --input "$FULL_DATA_ROOT/motion_libs/$FULL_SPLIT.pt" \
        --output-root "$shard_root" \
        --shard-size "$FULL_SHARD_SIZE"
    echo "[prepare 2/2] Writing all SceneLib shards from one full-pack load"
    "$PYTHON_BIN" scripts/prepare_trumans_training_shards.py \
        --kind scenes \
        --input "$FULL_DATA_ROOT/scene_libs/$FULL_SPLIT.pt" \
        --output-root "$shard_root" \
        --shard-size "$FULL_SHARD_SIZE"
    touch "$prepared_marker"
elif [[ -f "$prepared_marker" ]]; then
    echo "Reusing prepared shards: $prepared_marker"
fi

checkpoint="$FULL_BASE_CHECKPOINT"
for ((cycle=1; cycle<=FULL_CYCLES; cycle++)); do
    if (( cycle % 2 == 1 )); then
        first_start=0
        last_start=$((((FULL_TOTAL_CLIPS - 1) / FULL_SHARD_SIZE) * FULL_SHARD_SIZE))
        step=$FULL_SHARD_SIZE
    else
        first_start=$((((FULL_TOTAL_CLIPS - 1) / FULL_SHARD_SIZE) * FULL_SHARD_SIZE))
        last_start=0
        step=$((-FULL_SHARD_SIZE))
    fi

    shard_number=0
    for ((start=first_start; ; start+=step)); do
        count=$FULL_SHARD_SIZE
        if (( start + count > FULL_TOTAL_CLIPS )); then
            count=$((FULL_TOTAL_CLIPS - start))
        fi
        shard_number=$((shard_number + 1))
        experiment_name=$(printf '%s_cycle_%02d_shard_%04d_start_%04d_count_%02d' \
            "$FULL_RUN_NAME" "$cycle" "$shard_number" "$start" "$count")
        result_dir="$REPO_ROOT/results/$experiment_name"
        last_checkpoint="$result_dir/last.ckpt"
        complete_marker="$result_dir/.shard_complete"
        mkdir -p "$result_dir"

        if [[ -f "$complete_marker" && -f "$last_checkpoint" ]]; then
            echo "[cycle $cycle shard $shard_number] already complete: $last_checkpoint"
            checkpoint="$last_checkpoint"
        else
            echo "[cycle $cycle shard $shard_number] clips $start..$((start + count - 1)); warm start=$checkpoint"
            PILOT_EXPERIMENT_NAME="$experiment_name" \
            PILOT_NUM_ENVS="$FULL_NUM_ENVS" \
            PILOT_BATCH_SIZE="$((FULL_NUM_ENVS * 32))" \
            PILOT_ITERATIONS="$FULL_ITERATIONS_PER_SHARD" \
            PILOT_SCENE_START_INDEX="$start" \
            PILOT_SCENE_LOAD_COUNT="$count" \
            PILOT_SCENE_POINTCLOUD_CANDIDATES="$FULL_POINTCLOUD_CANDIDATES" \
            PILOT_BASE_CHECKPOINT="$checkpoint" \
            PILOT_VALIDATE_INPUTS=0 \
            PILOT_USE_PACKAGED_SUBSET=1 \
            bash scripts/train_trumans_scene_pilot.sh \
                2>&1 | tee -a "$result_dir/console.log"

            if [[ ! -f "$last_checkpoint" ]]; then
                echo "Cycle $cycle shard $shard_number ended without $last_checkpoint; stopping safely." >&2
                exit 5
            fi
            touch "$complete_marker"
            checkpoint="$last_checkpoint"
        fi

        if (( start == last_start )); then
            break
        fi
    done
done

echo "All $FULL_TOTAL_CLIPS clips completed for $FULL_CYCLES cycles. Final checkpoint: $checkpoint"
