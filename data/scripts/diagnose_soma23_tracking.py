#!/usr/bin/env python3
"""Diagnose SOMA23 reference-to-physics tracking without visual guesswork.

The input ``.motion`` files are recorder outputs whose quaternions are XYZW.
The final recorded frame is often a post-end reset, so it is excluded by
default.  Metrics deliberately separate world-space translation errors from
root-relative pose and orientation errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from align_soma23_retarget_foot_height import (
    FOOT_BODY_PAIRS,
    box_bottom_height,
    load_foot_box_geometries,
)
from protomotions.utils import rotations


BODY_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
]
BODY_ID = {name: index for index, name in enumerate(BODY_NAMES)}


def _load(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"rigid_body_pos", "rigid_body_rot"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return data


def _angle_deg(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(rotations.quat_diff_rad(actual, target, w_last=True))


def _forward_elevation_deg(quat: torch.Tensor) -> torch.Tensor:
    # SOMA's semantic forward axis is local -Y.
    local_forward = quat.new_tensor([0.0, -1.0, 0.0]).expand(quat.shape[:-1] + (3,))
    forward = rotations.quat_rotate(quat, local_forward, w_last=True)
    horizontal = torch.linalg.vector_norm(forward[..., :2], dim=-1)
    return torch.rad2deg(torch.atan2(forward[..., 2], horizontal))


def _relative_quat(child: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    return rotations.quat_mul(
        rotations.quat_conjugate(root, w_last=True), child, w_last=True
    )


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "p50": float(values.quantile(0.5)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def diagnose(
    rollout_path: Path,
    target_path: Path,
    mjcf_path: Path,
    *,
    trim_last: int,
    early_frames: int,
) -> dict:
    rollout = _load(rollout_path)
    target = _load(target_path)
    count = min(
        rollout["rigid_body_pos"].shape[0], target["rigid_body_pos"].shape[0]
    ) - trim_last
    if count <= 0:
        raise ValueError("No frames remain after trimming")
    pos = rollout["rigid_body_pos"][:count]
    rot = rollout["rigid_body_rot"][:count]
    gt_pos = target["rigid_body_pos"][:count]
    gt_rot = target["rigid_body_rot"][:count]

    root = BODY_ID["Hips"]
    chest = BODY_ID["Chest"]
    head = BODY_ID["Head"]
    position_error = torch.linalg.vector_norm(pos - gt_pos, dim=-1)
    root_aligned = (pos - pos[:, root:root + 1]) - (
        gt_pos - gt_pos[:, root:root + 1]
    )
    root_aligned_error = torch.linalg.vector_norm(root_aligned, dim=-1)

    root_z_error = pos[:, root, 2] - gt_pos[:, root, 2]
    head_z_error = pos[:, head, 2] - gt_pos[:, head, 2]
    relative_head_z_error = (
        pos[:, head, 2] - pos[:, root, 2]
    ) - (gt_pos[:, head, 2] - gt_pos[:, root, 2])

    body_orientation_errors = {
        name: _stats(_angle_deg(rot[:, BODY_ID[name]], gt_rot[:, BODY_ID[name]]))
        for name in ("Hips", "Chest", "Neck1", "Neck2", "Head")
    }
    head_relative_error = _angle_deg(
        _relative_quat(rot[:, head], rot[:, root]),
        _relative_quat(gt_rot[:, head], gt_rot[:, root]),
    )
    chest_relative_error = _angle_deg(
        _relative_quat(rot[:, chest], rot[:, root]),
        _relative_quat(gt_rot[:, chest], gt_rot[:, root]),
    )
    head_elevation_error = _forward_elevation_deg(rot[:, head]) - _forward_elevation_deg(
        gt_rot[:, head]
    )
    chest_elevation_error = _forward_elevation_deg(
        rot[:, chest]
    ) - _forward_elevation_deg(gt_rot[:, chest])

    geometries = load_foot_box_geometries(
        mjcf_path, [pair[3] for pair in FOOT_BODY_PAIRS]
    )
    rollout_bottoms = []
    target_bottoms = []
    for _, target_id, _, target_name in FOOT_BODY_PAIRS:
        rollout_bottoms.append(
            box_bottom_height(pos[:, target_id], rot[:, target_id], geometries[target_name])
        )
        target_bottoms.append(
            box_bottom_height(
                gt_pos[:, target_id], gt_rot[:, target_id], geometries[target_name]
            )
        )
    rollout_lowest = torch.stack(rollout_bottoms, dim=-1).min(dim=-1).values
    target_lowest = torch.stack(target_bottoms, dim=-1).min(dim=-1).values

    early = min(count, early_frames)
    return {
        "frames": count,
        "rollout": str(rollout_path),
        "target": str(target_path),
        "position_mm": {
            "mpjpe": float(position_error.mean() * 1000),
            "root": float(position_error[:, root].mean() * 1000),
            "head": float(position_error[:, head].mean() * 1000),
            "root_aligned_mpjpe": float(root_aligned_error.mean() * 1000),
            "root_z_signed": _stats(root_z_error * 1000),
            "head_z_signed": _stats(head_z_error * 1000),
            "head_minus_root_z_signed": _stats(relative_head_z_error * 1000),
        },
        "orientation_deg": {
            "global_geodesic": body_orientation_errors,
            "head_relative_to_root_geodesic": _stats(head_relative_error),
            "chest_relative_to_root_geodesic": _stats(chest_relative_error),
            "head_forward_elevation_signed": _stats(head_elevation_error),
            "chest_forward_elevation_signed": _stats(chest_elevation_error),
        },
        "initial_settling": {
            "window_frames": early,
            "root_z_error_frame0_mm": float(root_z_error[0] * 1000),
            "root_z_error_min_mm": float(root_z_error[:early].min() * 1000),
            "root_z_error_at_window_end_mm": float(root_z_error[early - 1] * 1000),
            "root_drop_from_frame0_to_min_mm": float(
                (pos[:early, root, 2].min() - pos[0, root, 2]) * 1000
            ),
            "head_elevation_error_frame0_deg": float(head_elevation_error[0]),
            "head_elevation_error_at_window_end_deg": float(
                head_elevation_error[early - 1]
            ),
        },
        "foot_collision_bottom_mm": {
            "target_frame0": float(target_lowest[0] * 1000),
            "rollout_frame0": float(rollout_lowest[0] * 1000),
            "rollout_min_first_window": float(rollout_lowest[:early].min() * 1000),
            "rollout_below_minus_2mm_fraction": float((rollout_lowest < -0.002).float().mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", nargs=3, metavar=("LABEL", "ROLLOUT", "GT"), required=True
    )
    parser.add_argument(
        "--mjcf", type=Path,
        default=Path("protomotions/data/assets/mjcf/soma23_humanoid.xml"),
    )
    parser.add_argument("--trim-last", type=int, default=1)
    parser.add_argument("--early-frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        label: diagnose(
            Path(rollout), Path(gt), args.mjcf,
            trim_last=args.trim_last, early_frames=args.early_frames,
        )
        for label, rollout, gt in args.variant
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"Saved {args.output}")
    print(text)


if __name__ == "__main__":
    main()
