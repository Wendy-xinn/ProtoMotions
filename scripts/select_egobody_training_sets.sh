#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/IsaacLab/.venv/bin/python}"
EGOBODY_ROOT="${EGOBODY_ROOT:-/public/home/wenxin/egobody}"
BODY_TEXT_ROOT="${EGOBODY_BODY_TEXT_ROOT:-}"
DATA_ROOT="${EGOBODY_SELECTION_ROOT:-$REPO_ROOT/data/motion_for_trackers/egobody_smpl_ego_v1}"

common_args=(
    --egobody-root "$EGOBODY_ROOT"
    --frames 192
    --score-stride 8
    --minimum-motion-score 0.5
)

if [[ -n "$BODY_TEXT_ROOT" ]]; then
    if [[ ! -d "$BODY_TEXT_ROOT" ]]; then
        echo "EgoBody body-text root does not exist: $BODY_TEXT_ROOT" >&2
        exit 2
    fi
    common_args+=(--body-text-root "$BODY_TEXT_ROOT")
fi

mkdir -p "$DATA_ROOT/sft_1000_192" "$DATA_ROOT/sft_diverse_800_192"
cd "$REPO_ROOT"

"$PYTHON_BIN" data/scripts/select_egobody_sft_clips.py \
    "${common_args[@]}" \
    --output "$DATA_ROOT/sft_1000_192/manifest.json" \
    --candidate-stride 64 \
    --split-clip-counts 800 100 100 \
    --min-start-gap 64

"$PYTHON_BIN" data/scripts/select_egobody_sft_clips.py \
    "${common_args[@]}" \
    --output "$DATA_ROOT/sft_diverse_800_192/manifest.json" \
    --candidate-stride 48 \
    --split-clip-counts 650 75 75 \
    --min-start-gap 96 \
    --exclude-window recording_20210923_S13_S05_01:2295 \
    --exclude-window recording_20210923_S14_S03_01:1561 \
    --exclude-window recording_20211002_S15_S17_02:3425

echo "Generated EgoBody dense-1000 and diverse-core-800 selections under $DATA_ROOT"
