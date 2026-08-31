#!/usr/bin/env python3
"""Ground a SOMA23 retarget using its support-foot collision geometry.

The horizontal root trajectory, root orientation, and relative vertical motion
are preserved. Each clip receives one constant vertical translation computed
from SOMA collision-box bottoms on source-SMPL contact frames. A low support
quantile avoids systematic penetration without making isolated noisy contact
labels determine the whole clip. A labelled support foot at frame zero is also
required to clear the floor, preventing a reset-time PhysX depenetration
impulse. This avoids the motion distortion caused by grounding every frame
independently.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


@dataclass(frozen=True)
class BoxGeometry:
    position: torch.Tensor
    rotation: torch.Tensor
    half_size: torch.Tensor


# (source SMPL body index, target SOMA23 body index, source name, target name)
FOOT_BODY_PAIRS = (
    (7, 17, "R_Ankle", "RightFoot"),
    (8, 18, "R_Toe", "RightToeBase"),
    (3, 21, "L_Ankle", "LeftFoot"),
    (4, 22, "L_Toe", "LeftToeBase"),
)


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm().clamp_min(1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(*quat.shape[:-1], 3, 3)


def _quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    return _quat_wxyz_to_matrix(quat[..., [3, 0, 1, 2]])


def _parse_vector(text: str | None, default: tuple[float, ...]) -> torch.Tensor:
    values = default if text is None else tuple(float(value) for value in text.split())
    return torch.tensor(values, dtype=torch.float64)


def load_foot_box_geometries(
    mjcf_path: Path, body_names: list[str]
) -> dict[str, BoxGeometry]:
    root = ET.parse(mjcf_path).getroot()
    geometries: dict[str, BoxGeometry] = {}
    for name in body_names:
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"Body {name!r} was not found in {mjcf_path}")
        geom = body.find("geom[@type='box']")
        if geom is None:
            raise ValueError(f"Body {name!r} has no box geometry in {mjcf_path}")
        position = _parse_vector(geom.get("pos"), (0.0, 0.0, 0.0))
        half_size = _parse_vector(geom.get("size"), ())
        if half_size.numel() != 3:
            raise ValueError(f"Body {name!r} box must have three size values")
        quat = _parse_vector(geom.get("quat"), (1.0, 0.0, 0.0, 0.0))
        geometries[name] = BoxGeometry(
            position=position,
            rotation=_quat_wxyz_to_matrix(quat),
            half_size=half_size,
        )
    return geometries


def box_bottom_height(
    body_pos: torch.Tensor,
    body_rot_xyzw: torch.Tensor,
    geometry: BoxGeometry,
) -> torch.Tensor:
    """Return the world-space lowest box corner height [m] for every frame."""
    body_rot = _quat_xyzw_to_matrix(body_rot_xyzw.to(torch.float64))
    local_pos = geometry.position.to(body_rot)
    local_rot = geometry.rotation.to(body_rot)
    half_size = geometry.half_size.to(body_rot)
    center = body_pos.to(torch.float64) + (body_rot @ local_pos).squeeze(-1)
    world_rot = body_rot @ local_rot
    vertical_extent = (world_rot[..., 2, :].abs() * half_size).sum(dim=-1)
    return center[..., 2] - vertical_extent


def _motion_layout(
    data: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if "gts" in data:
        return (
            data["gts"],
            data["grs"],
            data["length_starts"].to(torch.long),
            data["motion_num_frames"].to(torch.long),
        )
    positions = data["rigid_body_pos"]
    rotations = data["rigid_body_rot"]
    return (
        positions,
        rotations,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([positions.shape[0]], dtype=torch.long),
    )


def align_retargeted_motion(
    source: dict,
    target: dict,
    target_geometries: dict[str, BoxGeometry],
    *,
    clearance_m: float = 0.003,
    support_quantile: float = 0.1,
    enforce_initial_support: bool = True,
    allow_unlabelled_support_fallback: bool = True,
) -> tuple[dict, torch.Tensor]:
    """Apply one contact-aware Z offset to each motion while preserving root motion."""
    if not 0.0 <= support_quantile <= 1.0:
        raise ValueError("support_quantile must be in [0, 1]")
    _, _, source_starts, source_counts = _motion_layout(source)
    target_pos, target_rot, target_starts, target_counts = _motion_layout(target)
    if not torch.equal(source_counts.cpu(), target_counts.cpu()):
        raise ValueError(
            f"Motion frame counts differ: source={source_counts.tolist()}, "
            f"target={target_counts.tolist()}"
        )
    source_contacts = source.get("contacts", source.get("rigid_body_contacts"))
    if source_contacts is None:
        if not allow_unlabelled_support_fallback:
            raise ValueError("Source motion has no contact labels")
        source_positions, _, _, _ = _motion_layout(source)
        source_contacts = torch.zeros(
            source_positions.shape[:2], dtype=torch.bool, device=source_positions.device
        )

    output = dict(target)
    position_key = "gts" if "gts" in target else "rigid_body_pos"
    contact_key = "contacts" if "gts" in target else "rigid_body_contacts"
    output[position_key] = target_pos.clone()
    if target.get(contact_key) is not None:
        output[contact_key] = target[contact_key].clone().to(torch.bool)
    else:
        output[contact_key] = torch.zeros(
            target_pos.shape[:2], dtype=torch.bool, device=target_pos.device
        )

    offsets: list[torch.Tensor | None] = []
    fallback_motion_ids = []
    for motion_index in range(len(source_counts)):
        source_start = int(source_starts[motion_index])
        target_start = int(target_starts[motion_index])
        count = int(source_counts[motion_index])
        support_bottoms = []
        initial_support_bottoms = []
        for source_id, target_id, _, target_name in FOOT_BODY_PAIRS:
            source_slice = slice(source_start, source_start + count)
            target_slice = slice(target_start, target_start + count)
            contact = source_contacts[source_slice, source_id].bool()
            target_bottom = box_bottom_height(
                target_pos[target_slice, target_id],
                target_rot[target_slice, target_id],
                target_geometries[target_name],
            )
            if not contact.any():
                continue
            support_bottoms.append(target_bottom[contact])
            if bool(contact[0]):
                initial_support_bottoms.append(target_bottom[0])
            output[contact_key][target_slice, target_id] = contact
        if not support_bottoms:
            if not allow_unlabelled_support_fallback:
                raise ValueError(
                    f"Motion {motion_index} has no support-foot contact samples"
                )
            fallback_motion_ids.append(motion_index)
            offsets.append(None)
            continue
        support_bottom = torch.quantile(torch.cat(support_bottoms), support_quantile)
        offset = clearance_m - support_bottom
        # The quantile is robust over the complete clip, but reset cannot
        # tolerate a penetrating support foot. Enforce clearance only for feet
        # labelled in contact at frame zero; using the minimum across all
        # frames would let a single noisy label lift the complete motion.
        if enforce_initial_support and initial_support_bottoms:
            initial_support_bottom = torch.stack(initial_support_bottoms).min()
            offset = torch.maximum(
                offset,
                offset.new_tensor(clearance_m) - initial_support_bottom,
            )
        offset = offset.to(target_pos.dtype)
        offsets.append(offset)

    labelled_offsets = [offset for offset in offsets if offset is not None]
    fallback_offset = (
        torch.stack(labelled_offsets).median()
        if labelled_offsets
        else target_pos.new_zeros(())
    )
    resolved_offsets = []
    for motion_index, offset in enumerate(offsets):
        if offset is None:
            offset = fallback_offset
        target_start = int(target_starts[motion_index])
        count = int(target_counts[motion_index])
        output[position_key][target_start : target_start + count, :, 2] += offset
        resolved_offsets.append(offset.cpu())

    offset_tensor = torch.stack(resolved_offsets)
    if "gts" in target:
        output["retarget_root_height_offsets_m"] = offset_tensor
    else:
        output["retarget_root_height_offset_m"] = float(offset_tensor[0])
    output["retarget_height_alignment"] = (
        "constant_per_clip_target_contact_quantile_and_initial_support"
        if enforce_initial_support
        else "constant_per_clip_target_contact_collision_box_bottom_quantile"
    )
    output["retarget_ground_clearance_m"] = float(clearance_m)
    output["retarget_ground_support_quantile"] = float(support_quantile)
    output["retarget_unlabelled_support_fallback_motion_ids"] = torch.tensor(
        fallback_motion_ids, dtype=torch.long
    )
    fallback_mask = torch.zeros(len(source_counts), dtype=torch.bool)
    fallback_mask[fallback_motion_ids] = True
    output["retarget_unlabelled_support_fallback_mask"] = fallback_mask
    output["retarget_unlabelled_support_fallback"] = (
        "median_labelled_motion_offset" if labelled_offsets else "zero_offset"
    )
    return output, offset_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_smpl", type=Path)
    parser.add_argument("soma23_motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/smpl_humanoid.xml"),
    )
    parser.add_argument(
        "--target-mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/soma23_humanoid.xml"),
    )
    parser.add_argument("--clearance-m", type=float, default=0.003)
    parser.add_argument("--support-quantile", type=float, default=0.1)
    parser.add_argument(
        "--no-initial-support-safeguard",
        action="store_true",
        help=(
            "Use only the clip-wide contact-foot bottom quantile. By default "
            "the constant offset is also raised enough to give every frame-zero "
            "support foot the requested clearance."
        ),
    )
    parser.add_argument(
        "--strict-support-contacts",
        action="store_true",
        help="Fail instead of using foot geometry when a clip has no contact labels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    source = torch.load(args.source_smpl, weights_only=False, map_location="cpu")
    target = torch.load(args.soma23_motion, weights_only=False, map_location="cpu")
    target_names = [pair[3] for pair in FOOT_BODY_PAIRS]
    target_geometries = load_foot_box_geometries(args.target_mjcf, target_names)
    output, offsets = align_retargeted_motion(
        source,
        target,
        target_geometries,
        clearance_m=args.clearance_m,
        support_quantile=args.support_quantile,
        enforce_initial_support=not args.no_initial_support_safeguard,
        allow_unlabelled_support_fallback=not args.strict_support_contacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.pt")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    print(f"Saved {args.output}")
    for motion_index, offset in enumerate(offsets):
        print(f"Motion {motion_index}: root Z offset {float(offset):.6f} m")


if __name__ == "__main__":
    main()
