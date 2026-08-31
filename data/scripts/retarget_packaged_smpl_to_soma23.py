#!/usr/bin/env python3
"""Retarget a packaged SMPL MotionLib to the SOMA23 GPC skeleton.

The default is calibrated global-orientation transfer.  This preserves the
source joint rotations without forcing incompatible SMPL joint positions onto
SOMA's different bone lengths.  Position IK remains available as an explicit
experimental option through ``--ik-iterations``; it is disabled by default
because it can create cancelling spine/neck rotations that look acceptable in
global FK but lie far outside the SOMA GPC tracker's training distribution.
Scene/object trajectories remain index-aligned with SceneLib.
"""

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
from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix


# soma23 body index -> SMPL body index. Neck1 shares the SMPL neck orientation;
# Neck2 also receives it, leaving the extra SOMA neck link geometrically valid.
SOMA23_TO_SMPL = {
    0: 0, 1: 9, 2: 10, 3: 11, 4: 12, 5: 12, 6: 13,
    7: 19, 8: 20, 9: 21, 10: 22,
    11: 14, 12: 15, 13: 16, 14: 17,
    15: 5, 16: 6, 17: 7, 18: 8,
    19: 1, 20: 2, 21: 3, 22: 4,
}

# Interaction-critical bodies receive the strongest positional constraints.
POSITION_WEIGHTS = torch.tensor([
    10, 1, 1, 2, 1, 1, 12, 1, 2, 6, 20, 1, 2, 6, 20,
    2, 8, 20, 20, 2, 8, 20, 20,
], dtype=torch.float32)


def _smpl_to_soma_root_basis(device, dtype):
    """Return the local-basis change required by the two robot conventions.

    SMPL faces local +X while SOMA23 faces local -Y.  A +90 degree rotation
    around local Z maps SOMA's -Y forward vector to SMPL's +X and, equally
    importantly, maps SOMA's +/-X hip axis to SMPL's +/-Y hip axis.
    """
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _rest_global_rotations(ki):
    """Accumulate zero-pose body frames from an extracted kinematic tree."""
    globals_list = [torch.eye(3, device=ki.local_pos.device, dtype=ki.local_pos.dtype)]
    for body_idx in range(1, ki.num_bodies):
        parent = ki.parent_indices[body_idx]
        globals_list.append(
            globals_list[parent] @ ki.local_rot_ref_mat[body_idx]
        )
    return torch.stack(globals_list)


def _build_body_frame_basis(source_ki, target_ki):
    """Calibrate every mapped body frame, including semantic root heading."""
    source_rest = _rest_global_rotations(source_ki)
    target_rest = _rest_global_rotations(target_ki)
    semantic_root_basis = _smpl_to_soma_root_basis(
        target_rest.device, target_rest.dtype
    )
    bases = []
    for target_idx in range(target_ki.num_bodies):
        source_idx = SOMA23_TO_SMPL[target_idx]
        # At the canonical pose:
        # source_global @ basis == semantic_basis @ target_rest_global.
        bases.append(
            source_rest[source_idx].transpose(-1, -2)
            @ semantic_root_basis
            @ target_rest[target_idx]
        )
    return torch.stack(bases)


def _joint_mats_from_expmap(root_rot: torch.Tensor, joint_aa: torch.Tensor):
    # Stable Rodrigues map. The shared exp-map helper has an undefined
    # derivative at exactly zero, which poisons IK gradients for identity joints.
    angle = joint_aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = joint_aa / angle
    k = torch.zeros(*joint_aa.shape[:-1], 3, 3, device=joint_aa.device, dtype=joint_aa.dtype)
    k[..., 0, 1] = -axis[..., 2]
    k[..., 0, 2] = axis[..., 1]
    k[..., 1, 0] = axis[..., 2]
    k[..., 1, 2] = -axis[..., 0]
    k[..., 2, 0] = -axis[..., 1]
    k[..., 2, 1] = axis[..., 0]
    eye = torch.eye(3, device=joint_aa.device, dtype=joint_aa.dtype).expand_as(k)
    joint_mats = eye + angle.unsqueeze(-1).sin() * k + (
        1.0 - angle.unsqueeze(-1).cos()
    ) * (k @ k)
    return torch.cat([root_rot.unsqueeze(1), joint_mats], dim=1)


def _differentiable_fk(ki, root_pos: torch.Tensor, joint_mats: torch.Tensor):
    """List-based FK avoiding in-place tensor writes that break autograd."""
    positions = [root_pos]
    rotations = [joint_mats[:, 0]]
    for body_idx in range(1, ki.num_bodies):
        parent = ki.parent_indices[body_idx]
        parent_rot = rotations[parent]
        offset = ki.local_pos[body_idx].to(root_pos).view(1, 3, 1)
        positions.append(
            positions[parent] + (parent_rot @ offset).squeeze(-1)
        )
        ref = ki.local_rot_ref_mat[body_idx].to(root_pos).unsqueeze(0)
        rotations.append(parent_rot @ ref @ joint_mats[:, body_idx])
    return torch.stack(positions, dim=1), torch.stack(rotations, dim=1)


def _optimize_soma_ik(
    ki,
    root_pos: torch.Tensor,
    initial_joint_mats: torch.Tensor,
    source_pos: torch.Tensor,
    target_global_rot: torch.Tensor,
    *,
    iterations: int,
    learning_rate: float,
    temporal_weight: float,
    position_weight: float,
    bone_direction_weight: float,
    joint_angle_weight: float,
    bend_plane_weight: float,
    endpoint_weight: float,
    rotation_weight: float,
    anchor_weight: float,
):
    """Refine rotations while preserving pose rather than source morphology.

    Raw joint positions are not a feasible target when source and destination
    skeletons have different bone lengths, hip widths, or shoulder widths.
    Build a morphology-normalized positional target from source bone directions
    and SOMA bone lengths, then preserve flexion angles and bend planes.  Raw
    hand/foot positions remain only a weak interaction-aware soft constraint.
    """
    if iterations <= 0:
        return initial_joint_mats

    from protomotions.components.pose_lib import extract_qpos_from_transforms

    device = root_pos.device
    source_mapped_pos = torch.stack(
        [source_pos[:, SOMA23_TO_SMPL[i]] for i in range(23)], dim=1
    )

    # Source pose represented as unit bone directions along the target
    # hierarchy.  Repeated mappings (the extra SOMA neck link) inherit their
    # parent's direction instead of producing a zero-length source bone.
    source_bone_dirs = source_mapped_pos.new_zeros(source_mapped_pos.shape)
    source_bone_valid = torch.zeros(
        23, dtype=torch.bool, device=source_mapped_pos.device
    )
    for body_idx in range(1, 23):
        parent = ki.parent_indices[body_idx]
        delta = source_mapped_pos[:, body_idx] - source_mapped_pos[:, parent]
        length = delta.norm(dim=-1, keepdim=True)
        valid = length.squeeze(-1) > 1e-5
        if valid.any():
            source_bone_dirs[valid, body_idx] = delta[valid] / length[valid]
            source_bone_valid[body_idx] = True
        if (~valid).any():
            inherited = source_bone_dirs[:, parent]
            source_bone_dirs[~valid, body_idx] = inherited[~valid]

    # Feasible target positions: source pose directions with SOMA morphology.
    normalized_target_parts = [root_pos]
    for body_idx in range(1, 23):
        parent = ki.parent_indices[body_idx]
        target_bone_length = ki.local_pos[body_idx].to(root_pos).norm()
        normalized_target_parts.append(
            normalized_target_parts[parent]
            + source_bone_dirs[:, body_idx] * target_bone_length
        )
    normalized_target_pos = torch.stack(normalized_target_parts, dim=1)

    # Flexion joints expressed geometrically, independent of local joint axes.
    # (parent, center, child): elbows and knees.
    flexion_triplets = ((8, 9, 10), (12, 13, 14), (15, 16, 17), (19, 20, 21))

    def _joint_geometry(pos, triplet):
        parent, center, child = triplet
        proximal = pos[:, parent] - pos[:, center]
        distal = pos[:, child] - pos[:, center]
        proximal = proximal / proximal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        distal = distal / distal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cosine = (proximal * distal).sum(dim=-1).clamp(-1.0, 1.0)
        plane = torch.cross(proximal, distal, dim=-1)
        plane_strength = plane.norm(dim=-1)
        plane = plane / plane_strength.unsqueeze(-1).clamp_min(1e-6)
        return cosine, plane, plane_strength

    source_joint_geometry = [
        _joint_geometry(source_mapped_pos, triplet)
        for triplet in flexion_triplets
    ]
    endpoint_ids = torch.tensor([10, 14, 18, 22], device=device)
    initial_qpos = extract_qpos_from_transforms(
        ki,
        root_pos,
        initial_joint_mats,
        multi_dof_decomposition_method="exp_map",
    )
    joint_aa = initial_qpos[:, 7:].reshape(-1, 22, 3).detach().clone()
    joint_aa.requires_grad_(True)
    root_rot = initial_joint_mats[:, 0].detach()
    weights = POSITION_WEIGHTS.to(device=device, dtype=root_pos.dtype)
    optimizer = torch.optim.Adam([joint_aa], lr=learning_rate)

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        joint_mats = _joint_mats_from_expmap(root_rot, joint_aa)
        pred_pos, pred_rot = _differentiable_fk(ki, root_pos, joint_mats)
        position_loss = (
            (pred_pos - normalized_target_pos).square().sum(dim=-1) * weights
        ).sum() / (weights.sum() * root_pos.shape[0])
        pred_bones = pred_pos[:, 1:] - pred_pos[:, ki.parent_indices[1:]]
        pred_bone_dirs = pred_bones / pred_bones.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        direction_mask = source_bone_valid[1:].to(root_pos.dtype).view(1, -1)
        bone_direction_loss = (
            (pred_bone_dirs - source_bone_dirs[:, 1:]).square().sum(dim=-1)
            * direction_mask
        ).sum() / (direction_mask.sum().clamp_min(1.0) * root_pos.shape[0])

        angle_losses = []
        plane_losses = []
        for triplet, (source_cosine, source_plane, source_strength) in zip(
            flexion_triplets, source_joint_geometry
        ):
            pred_cosine, pred_plane, pred_strength = _joint_geometry(
                pred_pos, triplet
            )
            angle_losses.append((pred_cosine - source_cosine).square().mean())
            # Plane direction is unreliable for a nearly straight limb, so
            # smoothly downweight it by both source and predicted sine angle.
            plane_reliability = (source_strength * pred_strength).detach()
            plane_losses.append(
                ((1.0 - (pred_plane * source_plane).sum(dim=-1))
                 * plane_reliability).mean()
            )
        joint_angle_loss = torch.stack(angle_losses).mean()
        bend_plane_loss = torch.stack(plane_losses).mean()

        endpoint_loss = (
            pred_pos[:, endpoint_ids] - source_mapped_pos[:, endpoint_ids]
        ).square().sum(dim=-1).mean()
        temporal_loss = joint_aa.new_zeros(())
        if joint_aa.shape[0] > 2:
            # Penalize angular acceleration, not angular velocity.  A
            # first-difference penalty systematically damps valid head and
            # wrist motion and makes those bodies look locked to their parent.
            temporal_loss = (
                joint_aa[2:] - 2.0 * joint_aa[1:-1] + joint_aa[:-2]
            ).square().mean()
        # The initial mapped globals carry the source limb-frame directions.
        # Keeping them close prevents position-only IK from solving hand
        # targets with anatomically implausible shoulder/forearm twist.
        relative_rot = pred_rot.transpose(-1, -2) @ target_global_rot
        identity = torch.eye(3, device=device, dtype=root_pos.dtype)
        rotation_loss = (relative_rot - identity).square().mean()
        # Stay near the orientation-transfer solution where position targets
        # are underdetermined (notably twist around long limb axes).
        anchor_loss = (joint_aa - initial_qpos[:, 7:].reshape(-1, 22, 3)).square().mean()
        loss = (
            position_weight * position_loss
            + bone_direction_weight * bone_direction_loss
            + joint_angle_weight * joint_angle_loss
            + bend_plane_weight * bend_plane_loss
            + endpoint_weight * endpoint_loss
            + temporal_weight * temporal_loss
            + rotation_weight * rotation_loss
            + anchor_weight * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([joint_aa], 10.0)
        optimizer.step()

    return _joint_mats_from_expmap(root_rot, joint_aa.detach())


def _retarget_clip(data: dict, start: int, count: int, ki, body_frame_basis, args):
    source_pos = data["gts"][start : start + count].to(args.device)
    source_rot = quaternion_to_matrix(
        data["grs"][start : start + count].to(args.device), w_last=True
    )
    target_globals = torch.stack(
        [source_rot[:, SOMA23_TO_SMPL[i]] for i in range(23)], dim=1
    )
    target_globals = target_globals @ body_frame_basis.unsqueeze(0)
    joint_rot = compute_joint_rot_mats_from_global_mats(ki, target_globals)
    root_pos = source_pos[:, 0].clone()
    joint_rot = _optimize_soma_ik(
        ki,
        root_pos,
        joint_rot,
        source_pos,
        target_globals,
        iterations=args.ik_iterations,
        learning_rate=args.ik_learning_rate,
        temporal_weight=args.temporal_weight,
        position_weight=args.position_weight,
        bone_direction_weight=args.bone_direction_weight,
        joint_angle_weight=args.joint_angle_weight,
        bend_plane_weight=args.bend_plane_weight,
        endpoint_weight=args.endpoint_weight,
        rotation_weight=args.rotation_weight,
        anchor_weight=args.anchor_weight,
    )
    dt = float(data["motion_dt"][_retarget_clip.motion_index])
    fps = max(1, round(1.0 / dt))
    state = fk_from_transforms_with_velocities(
        ki, root_pos, joint_rot, fps=fps, compute_velocities=True
    )
    qpos = extract_qpos_from_transforms(
        ki, root_pos, joint_rot, multi_dof_decomposition_method="exp_map"
    )
    dps = qpos[:, 7:]
    dvs = torch.zeros_like(dps)
    if count > 1:
        dvs[:-1] = (dps[1:] - dps[:-1]) / dt
        dvs[-1] = dvs[-2]
    local_effective = joint_rot.clone()
    local_effective[:, 1:] = (
        ki.local_rot_ref_mat[1:].unsqueeze(0) @ joint_rot[:, 1:]
    )
    contacts = torch.stack(
        [data["contacts"][start : start + count, SOMA23_TO_SMPL[i]].to(args.device) for i in range(23)],
        dim=1,
    ) if data.get("contacts") is not None else None
    return {
        "gts": state.rigid_body_pos.cpu(),
        "grs": state.rigid_body_rot.cpu(),
        "gvs": state.rigid_body_vel.cpu(),
        "gavs": state.rigid_body_ang_vel.cpu(),
        "dps": dps.cpu(),
        "dvs": dvs.cpu(),
        "lrs": matrix_to_quaternion(local_effective, w_last=True).cpu(),
        "contacts": contacts.cpu() if contacts is not None else None,
    }


_retarget_clip.motion_index = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--soma-mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/soma23_humanoid.xml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ik-iterations",
        type=int,
        default=0,
        help=(
            "Optional morphology-normalized position IK refinement. The "
            "GPC-compatible default is 0 (orientation transfer only)."
        ),
    )
    parser.add_argument("--ik-learning-rate", type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.05)
    parser.add_argument("--position-weight", type=float, default=0.25)
    parser.add_argument("--bone-direction-weight", type=float, default=1.0)
    parser.add_argument("--joint-angle-weight", type=float, default=0.5)
    parser.add_argument("--bend-plane-weight", type=float, default=0.1)
    parser.add_argument("--endpoint-weight", type=float, default=0.05)
    parser.add_argument("--rotation-weight", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    args.device = torch.device(args.device)
    ki = extract_kinematic_info(str(args.soma_mjcf))
    for field in ("local_pos", "local_rot_ref_mat"):
        setattr(ki, field, getattr(ki, field).to(args.device))
    source_ki = extract_kinematic_info(
        "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    )
    for field in ("local_pos", "local_rot_ref_mat"):
        setattr(source_ki, field, getattr(source_ki, field).to(args.device))
    body_frame_basis = _build_body_frame_basis(source_ki, ki)
    total = len(data["motion_lengths"])
    if args.limit is not None:
        total = min(total, args.limit)
    clips = []
    for motion_index in range(total):
        _retarget_clip.motion_index = motion_index
        start = int(data["length_starts"][motion_index])
        count = int(data["motion_num_frames"][motion_index])
        clips.append(
            _retarget_clip(data, start, count, ki, body_frame_basis, args)
        )
        print(f"retargeted motion {motion_index}: {count} frames")
    counts = data["motion_num_frames"][:total].clone()
    shifted = counts.roll(1)
    shifted[0] = 0
    output = {
        "length_starts": shifted.cumsum(0),
        "motion_num_frames": counts,
        "motion_lengths": data["motion_lengths"][:total].clone(),
        "motion_dt": data["motion_dt"][:total].clone(),
        "motion_weights": data["motion_weights"][:total].clone(),
    }
    for key in ("gts", "grs", "gvs", "gavs", "dps", "dvs", "lrs", "contacts"):
        values = [clip[key] for clip in clips if clip[key] is not None]
        if values:
            output[key] = torch.cat(values)
    if "motion_files" in data:
        output["motion_files"] = tuple(data["motion_files"][:total])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.tmp.pt")
    torch.save(output, temp)
    os.replace(temp, args.output)
    print(f"saved {total} SOMA23 motions to {args.output}")


if __name__ == "__main__":
    main()
