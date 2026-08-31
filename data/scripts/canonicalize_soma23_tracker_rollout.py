#!/usr/bin/env python3
"""Build a scene-aligned SOMA23 motion from a physical tracker rollout.

The physical rollout supplies a dynamically feasible local pose and grounded
root height.  The reference supplies the intended scene-space root XY and
heading.  Per-frame SE(2) alignment preserves the rollout's body-relative pose
while removing tracker trajectory drift.  The post-end reset frame is trimmed
by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from protomotions.components.pose_lib import (
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils import rotations


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    return rotations.calc_heading_quat(quat, w_last=True)


def canonicalize(
    rollout: dict,
    reference: dict,
    *,
    mjcf_path: Path,
    trim_last: int,
) -> dict:
    rollout_pos = rollout["rigid_body_pos"]
    rollout_rot = rollout["rigid_body_rot"]
    reference_pos = reference.get("rigid_body_pos", reference.get("gts"))
    reference_rot = reference.get("rigid_body_rot", reference.get("grs"))
    if reference_pos is None or reference_rot is None:
        raise ValueError("Reference must contain rigid_body_pos/rot or gts/grs")
    count = min(rollout_pos.shape[0], reference_pos.shape[0]) - trim_last
    if count <= 1:
        raise ValueError("No usable frames remain")
    rollout_pos = rollout_pos[:count].clone()
    rollout_rot = rollout_rot[:count].clone()
    reference_pos = reference_pos[:count]
    reference_rot = reference_rot[:count]

    rollout_heading = _yaw_quat(rollout_rot[:, 0])
    reference_heading = _yaw_quat(reference_rot[:, 0])
    heading_delta = rotations.quat_mul(
        reference_heading,
        rotations.quat_conjugate(rollout_heading, w_last=True),
        w_last=True,
    )
    relative_pos = rollout_pos - rollout_pos[:, :1]
    expanded_delta = heading_delta[:, None].expand(-1, rollout_pos.shape[1], -1)
    aligned_relative_pos = rotations.quat_rotate(
        expanded_delta.reshape(-1, 4), relative_pos.reshape(-1, 3), w_last=True
    ).reshape_as(relative_pos)
    aligned_root = rollout_pos[:, 0].clone()
    aligned_root[:, :2] = reference_pos[:, 0, :2]
    aligned_pos = aligned_root[:, None] + aligned_relative_pos
    aligned_rot = rotations.quat_mul(expanded_delta, rollout_rot, w_last=True)

    ki = extract_kinematic_info(str(mjcf_path))
    global_mats = rotations.quaternion_to_matrix(aligned_rot, w_last=True)
    joint_mats = compute_joint_rot_mats_from_global_mats(ki, global_mats)
    fps = int(rollout.get("fps", 30))
    state = fk_from_transforms_with_velocities(
        ki, aligned_root, joint_mats, fps=fps, compute_velocities=True
    )
    qpos = extract_qpos_from_transforms(
        ki, aligned_root, joint_mats, multi_dof_decomposition_method="exp_map"
    )
    dps = qpos[:, 7:]
    dt = 1.0 / fps
    dvs = torch.zeros_like(dps)
    dvs[:-1] = (dps[1:] - dps[:-1]) / dt
    dvs[-1] = dvs[-2]
    local_effective = joint_mats.clone()
    local_effective[:, 1:] = ki.local_rot_ref_mat[1:].unsqueeze(0) @ joint_mats[:, 1:]

    contacts = rollout.get("rigid_body_contacts")
    if contacts is None:
        contacts = torch.zeros(count, aligned_pos.shape[1], dtype=torch.bool)
    else:
        contacts = contacts[:count].to(torch.bool)
    return {
        "gts": state.rigid_body_pos.cpu(),
        "grs": state.rigid_body_rot.cpu(),
        "gvs": state.rigid_body_vel.cpu(),
        "gavs": state.rigid_body_ang_vel.cpu(),
        "dps": dps.cpu(),
        "dvs": dvs.cpu(),
        "lrs": rotations.matrix_to_quaternion(local_effective, w_last=True).cpu(),
        "contacts": contacts.cpu(),
        "length_starts": torch.tensor([0], dtype=torch.long),
        "motion_num_frames": torch.tensor([count], dtype=torch.long),
        "motion_lengths": torch.tensor([(count - 1) * dt], dtype=torch.float32),
        "motion_dt": torch.tensor([dt], dtype=torch.float32),
        "motion_weights": torch.tensor([1.0], dtype=torch.float32),
        "motion_files": ("continuous_tracker_canonicalized.motion",),
        "canonicalization": "physical_local_pose_plus_reference_root_xy_heading",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mjcf", type=Path,
        default=Path("protomotions/data/assets/mjcf/soma23_humanoid.xml"),
    )
    parser.add_argument("--trim-last", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rollout = torch.load(args.rollout, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    output = canonicalize(
        rollout, reference, mjcf_path=args.mjcf, trim_last=args.trim_last
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.pt")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Saved {args.output} ({int(output['motion_num_frames'][0])} frames)")


if __name__ == "__main__":
    main()
