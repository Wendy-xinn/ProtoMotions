#!/usr/bin/env python3
"""Rebuild and validate all grounded EgoBody cameras in a prepared dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from align_ego_camera_to_grounded_motion import align_camera


def _save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    root = Path(manifest["prepared_root"])
    recordings = sorted({clip["recording"] for clip in manifest["clips"]})

    total_clips = 0
    max_forward_p95 = 0.0
    max_distance_p95 = 0.0
    for index, recording in enumerate(recordings, 1):
        recording_root = root / recording
        camera = torch.load(
            recording_root / "ego_camera.pt", map_location="cpu", weights_only=False
        )
        grounded = torch.load(
            recording_root / "motion_lib_soma23_grounded.pt",
            map_location="cpu",
            weights_only=False,
        )
        output = align_camera(camera, grounded)
        output["grounded_motion_file"] = str(
            (recording_root / "motion_lib_soma23_grounded.pt").resolve()
        )
        _save_atomic(output, recording_root / "ego_camera_grounded.pt")
        reports = output["camera_alignment"]["per_motion"]
        total_clips += len(reports)
        max_forward_p95 = max(
            max_forward_p95,
            max(item["head_camera_forward_p95_deg"] for item in reports),
        )
        max_distance_p95 = max(
            max_distance_p95,
            max(item["head_camera_distance_p95_m"] for item in reports),
        )
        print(
            f"[{index}/{len(recordings)}] {recording}: {len(reports)} clips validated",
            flush=True,
        )
    print(
        f"Rebuilt {total_clips} clips across {len(recordings)} recordings; "
        f"worst forward p95={max_forward_p95:.2f} deg, "
        f"worst Head distance p95={max_distance_p95:.3f} m"
    )


if __name__ == "__main__":
    main()
