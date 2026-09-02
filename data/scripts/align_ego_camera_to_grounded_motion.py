#!/usr/bin/env python3
"""Apply SOMA23 per-clip grounding translations to EgoBody cameras.

The measured camera is already calibrated into the reconstructed scene frame.
Grounding changes only world Z, so apply the same clip-level Z translation to
the camera while preserving its measured orientation and the static scene.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from protomotions.utils import rotations


SOMA23_HEAD_ID = 6


def validate_camera_motion_alignment(camera_data: dict, grounded_motion: dict) -> dict:
    """Reject camera/person coordinate mismatches, especially 180-degree flips."""
    starts = torch.as_tensor(grounded_motion["length_starts"], dtype=torch.long)
    counts = torch.as_tensor(grounded_motion["motion_num_frames"], dtype=torch.long)
    diagnostics = []
    for motion_index, clip in enumerate(camera_data["motions"]):
        start, count = int(starts[motion_index]), int(counts[motion_index])
        camera = clip["world_from_camera"].float()
        measured = clip["measured_world_from_camera"].float()
        offset = float(clip["retarget_root_height_offset_m"])
        if camera.shape[0] != count:
            raise ValueError(f"Camera frame mismatch for motion {motion_index}")
        if not torch.isfinite(camera).all():
            raise ValueError(f"Non-finite camera pose in motion {motion_index}")
        if not torch.allclose(camera[:, :3, :3], measured[:, :3, :3], atol=1e-6):
            raise ValueError(f"Grounding changed camera rotation in motion {motion_index}")
        expected_delta = torch.zeros_like(camera[:, :3, 3])
        expected_delta[:, 2] = offset
        actual_delta = camera[:, :3, 3] - measured[:, :3, 3]
        if not torch.allclose(actual_delta, expected_delta, atol=1e-6):
            raise ValueError(
                f"Camera grounding is not a pure Z translation in motion {motion_index}"
            )

        head_pos = grounded_motion["gts"][start : start + count, SOMA23_HEAD_ID]
        head_rot = grounded_motion["grs"][start : start + count, SOMA23_HEAD_ID]
        head_forward = rotations.quat_rotate(
            head_rot,
            head_rot.new_tensor([0.0, -1.0, 0.0]).expand(count, -1),
            w_last=True,
        )
        # EgoBody PV uses OpenGL-style -Z forward. This is the training-data
        # convention; the OpenCV conversion belongs only in Viser rendering.
        camera_forward = -camera[:, :3, 2]
        forward_angle = torch.rad2deg(
            torch.acos((head_forward * camera_forward).sum(-1).clamp(-1.0, 1.0))
        )
        head_distance = torch.linalg.vector_norm(camera[:, :3, 3] - head_pos, dim=-1)
        forward_p95 = float(torch.quantile(forward_angle, 0.95))
        distance_p95 = float(torch.quantile(head_distance, 0.95))
        if forward_p95 >= 60.0:
            raise ValueError(
                f"Camera/Head forward-axis mismatch in motion {motion_index}: "
                f"p95={forward_p95:.1f} deg"
            )
        if distance_p95 >= 0.5:
            raise ValueError(
                f"Camera is too far from Head in motion {motion_index}: "
                f"p95={distance_p95:.3f} m"
            )
        diagnostics.append(
            {
                "motion_id": motion_index,
                "head_camera_forward_median_deg": float(forward_angle.median()),
                "head_camera_forward_p95_deg": forward_p95,
                "head_camera_distance_median_m": float(head_distance.median()),
                "head_camera_distance_p95_m": distance_p95,
            }
        )
    return {"per_motion": diagnostics}


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
        motion["measured_world_from_camera"] = camera_poses.clone()
        motion["world_from_camera"] = camera_poses.clone()
        motion["world_from_camera"][:, 2, 3] += offset.to(
            dtype=camera_poses.dtype
        )
        motion["reference_root"] = target_root.to(dtype=reference_root.dtype)
        motion["retarget_root_height_offset_m"] = float(offset)

    output["retarget_height_alignment"] = grounded_motion.get(
        "retarget_height_alignment", "constant_per_clip"
    )
    output["retarget_root_height_offsets_m"] = offsets
    output["camera_alignment"] = {
        "method": "constant_per_clip_grounding_z_translation",
        "rotation_changed": False,
    }
    output["camera_alignment"].update(
        validate_camera_motion_alignment(output, grounded_motion)
    )
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
    print(output["camera_alignment"])


if __name__ == "__main__":
    main()
