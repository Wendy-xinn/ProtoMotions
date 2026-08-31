#!/usr/bin/env python3
"""Apply a constant local head-pitch calibration to a SOMA23 motion library."""

from __future__ import annotations

import argparse
import math
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--degrees", type=float, required=True)
    parser.add_argument(
        "--mjcf", type=Path,
        default=Path("protomotions/data/assets/mjcf/soma23_humanoid.xml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    ki = extract_kinematic_info(str(args.mjcf))
    global_mats = rotations.quaternion_to_matrix(data["grs"], w_last=True)
    angle = torch.full((global_mats.shape[0],), math.radians(args.degrees))
    axis = torch.zeros(global_mats.shape[0], 3)
    axis[:, 0] = 1.0
    pitch = rotations.quaternion_to_matrix(
        rotations.quat_from_angle_axis(angle, axis, w_last=True), w_last=True
    )
    # Head is a leaf, so this changes only its orientation, not body positions.
    global_mats[:, 6] = global_mats[:, 6] @ pitch
    joint_mats = compute_joint_rot_mats_from_global_mats(ki, global_mats)
    root_pos = data["gts"][:, 0]
    fps = round(1.0 / float(data["motion_dt"][0]))
    state = fk_from_transforms_with_velocities(
        ki, root_pos, joint_mats, fps=fps, compute_velocities=True
    )
    qpos = extract_qpos_from_transforms(
        ki, root_pos, joint_mats, multi_dof_decomposition_method="exp_map"
    )
    local_effective = joint_mats.clone()
    local_effective[:, 1:] = ki.local_rot_ref_mat[1:].unsqueeze(0) @ joint_mats[:, 1:]
    output = dict(data)
    output.update({
        "gts": state.rigid_body_pos.cpu(), "grs": state.rigid_body_rot.cpu(),
        "gvs": state.rigid_body_vel.cpu(), "gavs": state.rigid_body_ang_vel.cpu(),
        "dps": qpos[:, 7:].cpu(),
        "lrs": rotations.matrix_to_quaternion(local_effective, w_last=True).cpu(),
        "head_pitch_calibration_degrees": float(args.degrees),
    })
    dvs = torch.zeros_like(output["dps"])
    dt = 1.0 / fps
    dvs[:-1] = (output["dps"][1:] - output["dps"][:-1]) / dt
    dvs[-1] = dvs[-2]
    output["dvs"] = dvs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.pt")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Saved {args.output}: local head pitch {args.degrees:+.1f} deg")


if __name__ == "__main__":
    main()
