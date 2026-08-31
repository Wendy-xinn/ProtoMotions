#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MM_TEACHER_MODE=body_only \
MM_TEACHER_EXPERIMENT_NAME="${MM_BODY_ONLY_NAME:-egobody_smpl_teacher_body_only_40_v1}" \
    "$SCRIPT_DIR/train_egobody_masked_mimic_teacher.sh" "$@"

MM_TEACHER_MODE=body_scene \
MM_TEACHER_EXPERIMENT_NAME="${MM_BODY_SCENE_NAME:-egobody_smpl_teacher_body_scene_40_v1}" \
    "$SCRIPT_DIR/train_egobody_masked_mimic_teacher.sh" "$@"
