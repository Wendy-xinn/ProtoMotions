#!/usr/bin/env python3
"""Bind a calibrated EgoBody camera trajectory to grounded SOMA23 clips.

``world_from_camera`` is already calibrated in the static-scene coordinate
frame and must not be translated to compensate for skeleton retargeting. The
runtime visibility observation relocates it into replicated simulator scenes
using ``current_reference_root - camera_reference_root``. Store the grounded
SOMA23 root as that anchor so the difference contains only the simulator's
scene/spawn transform.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch


def align_camera(camera_data: dict, grounded_motion: dict) -> dict:
    offsets = grounded_motion.get("retarget_root_height_offsets_m")
    if offsets is None:
        raise ValueError(
            "Grounded motion is missing 'retarget_root_height_offsets_m'"
        )
    offsets = torch.as_tensor(offsets, dtype=torch.float32).cpu()
    motions = camera_data.get("motions")
    if not isinstance(motions, list) or len(motions) != offsets.numel():
        raise ValueError(
            f"Camera/motion count mismatch: camera={len(motions or [])}, "
            f"offsets={offsets.numel()}"
        )

    frame_counts = grounded_motion.get("motion_num_frames")
    if frame_counts is not None:
        frame_counts = torch.as_tensor(frame_counts, dtype=torch.long).cpu()

    target_pos = grounded_motion.get("gts", grounded_motion.get("rigid_body_pos"))
    if not torch.is_tensor(target_pos) or target_pos.ndim != 3:
        raise ValueError("Grounded motion has no valid body-position tensor")
    starts = grounded_motion.get("motion_starts")
    if starts is None:
        if frame_counts is None:
            if len(motions) != 1:
                raise ValueError(
                    "Multi-motion pack is missing motion_starts and motion_num_frames"
                )
            starts = torch.tensor([0], dtype=torch.long)
        else:
            starts = torch.cat(
                [torch.zeros(1, dtype=torch.long), frame_counts.cumsum(0)[:-1]]
            )
    else:
        starts = torch.as_tensor(starts, dtype=torch.long).cpu()

    output = copy.deepcopy(camera_data)
    for motion_index, (motion, offset) in enumerate(zip(output["motions"], offsets)):
        camera_poses = motion.get("world_from_camera")
        reference_root = motion.get("reference_root")
        if not torch.is_tensor(camera_poses) or camera_poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"Camera motion {motion_index} has invalid world_from_camera"
            )
        if not torch.is_tensor(reference_root) or reference_root.shape[-1] != 3:
            raise ValueError(f"Camera motion {motion_index} has invalid reference_root")
        if camera_poses.shape[0] != reference_root.shape[0]:
            raise ValueError(f"Camera motion {motion_index} frame arrays disagree")
        if frame_counts is not None and camera_poses.shape[0] != int(frame_counts[motion_index]):
            raise ValueError(
                f"Camera motion {motion_index} has {camera_poses.shape[0]} frames, "
                f"grounded motion has {int(frame_counts[motion_index])}"
            )

        start = int(starts[motion_index])
        count = camera_poses.shape[0]
        target_root = target_pos[start : start + count, 0].cpu()
        if target_root.shape != reference_root.shape:
            raise ValueError(
                f"Camera motion {motion_index} root shape {reference_root.shape} "
                f"does not match target {target_root.shape}"
            )
        # Preserve measured camera poses in the calibrated scene frame. Only
        # replace the anchor used later for simulator scene replication.
        motion["reference_root"] = target_root.to(dtype=reference_root.dtype)
        motion["retarget_root_height_offset_m"] = float(offset)

    output["retarget_height_alignment"] = grounded_motion.get(
        "retarget_height_alignment", "constant_per_clip"
    )
    output["retarget_root_height_offsets_m"] = offsets
    output["grounded_motion_file"] = None
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", type=Path)
    parser.add_argument("grounded_motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    camera = torch.load(args.camera, map_location="cpu", weights_only=False)
    grounded = torch.load(args.grounded_motion, map_location="cpu", weights_only=False)
    output = align_camera(camera, grounded)
    output["grounded_motion_file"] = str(args.grounded_motion.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Saved {args.output}")
    for index, offset in enumerate(output["retarget_root_height_offsets_m"]):
        print(
            f"Motion {index}: preserved GT camera; bound reference root to "
            f"grounded SOMA23 (motion Z correction {float(offset):.6f} m)"
        )


if __name__ == "__main__":
    main()
