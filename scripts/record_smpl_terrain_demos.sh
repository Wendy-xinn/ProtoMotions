#!/usr/bin/env bash
set -euo pipefail

# Record the same SMPL reference clip with two different generated height fields.
# Run from ProtoMotions (or let the script resolve its own repository root).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
WSL_RUNNER="${WSL_RUNNER:-$REPO_ROOT/scripts/run_wsl_isaaclab.sh}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt}"
MOTION_FILE="${MOTION_FILE:-$REPO_ROOT/data/motion_for_trackers/trumans_scene_v1/motions/train/2023-02-12@12-16-04.motion}"
MOTION_SUBSET="${MOTION_SUBSET:-[0]}"
RECORD_STEPS="${RECORD_STEPS:-300}"
TERRAIN_LEVELS="${TERRAIN_LEVELS:-1}"
TERRAIN_VARIANTS="${TERRAIN_VARIANTS:-1}"
STAIRS_STEP_HEIGHT="${STAIRS_STEP_HEIGHT:-0.15}"
STAIRS_STEP_WIDTH="${STAIRS_STEP_WIDTH:-0.40}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/output/terrain_demos_$(date +%Y-%m-%d-%H-%M-%S)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$MOTION_FILE" ]]; then
    echo "Motion file not found: $MOTION_FILE" >&2
    exit 2
fi

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --simulator isaaclab
    --motion-file "$MOTION_FILE"
    --num-envs 1
    --record-steps "$RECORD_STEPS"
    --overrides
    "terrain.num_levels=$TERRAIN_LEVELS"
    "terrain.num_terrains=$TERRAIN_VARIANTS"
    "terrain.stairs_step_height=$STAIRS_STEP_HEIGHT"
    "terrain.stairs_step_width=$STAIRS_STEP_WIDTH"
    "env.motion_manager.subset_method=$MOTION_SUBSET"
    "env.motion_manager.init_start_prob=1.0"
    "env.motion_manager.resample_on_reset=False"
)

run_demo() {
    local name="$1"
    local proportions="$2"
    local runner=("$PYTHON_BIN")
    if [[ -x "$WSL_RUNNER" && -e /dev/dxg ]]; then
        runner=("$WSL_RUNNER" "$PYTHON_BIN")
    fi
    echo "Recording $name terrain ($RECORD_STEPS steps)"
    PYTHONUNBUFFERED=1 "${runner[@]}" "$REPO_ROOT/protomotions/inference_agent.py" \
        "${COMMON_ARGS[@]}" \
        "simulator.experiment_name=$name" \
        "terrain.terrain_proportions=$proportions"
}

render_offline_fallback() {
    local name="$1"
    local motion_file
    motion_file="$(find "$REPO_ROOT/output/renderings" -maxdepth 1 -type f \
        -name "$name-*.motion" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "$motion_file" ]]; then
        echo "No recorded motion found for $name; skipping offline fallback." >&2
        return 0
    fi
    local terrain_file="${motion_file%.motion}.terrain.pt"
    local terrain_args=()
    if [[ -f "$terrain_file" ]]; then
        terrain_args=(--terrain "$terrain_file")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/scripts/render_recorded_motion_mp4.py" \
        --motion "$motion_file" \
        --output "$RESULT_DIR/${name}_offline.mp4" \
        "${terrain_args[@]}"
}

collect_demo() {
    local name="$1"
    shopt -s nullglob
    local files=(
        "$REPO_ROOT"/output/renderings/"$name"-*
    )
    for f in "${files[@]}"; do
        mv "$f" "$RESULT_DIR/"
    done
    shopt -u nullglob
}

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/output/renderings" "$RESULT_DIR"
echo "Reference motion source: $MOTION_FILE"
echo "Reference motion subset: $MOTION_SUBSET"
run_demo smpl_terrain_flat '[0,0,0,0,0,0,0,1]'
render_offline_fallback smpl_terrain_flat
collect_demo smpl_terrain_flat
run_demo smpl_terrain_stairs_up '[0,0,1,0,0,0,0,0]'
render_offline_fallback smpl_terrain_stairs_up
collect_demo smpl_terrain_stairs_up

echo "Demo outputs are under $RESULT_DIR/"
