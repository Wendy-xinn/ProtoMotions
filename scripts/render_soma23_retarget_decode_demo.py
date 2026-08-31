#!/usr/bin/env python3
"""Render several SOMA23 retarget/FSQ comparisons into one report-ready MP4.

The renderer is intentionally independent of Isaac Sim/RTX and Viser.  It uses
an orthographic motion-aligned view so videos can be generated headlessly on
WSL while preserving the complete physical rollout and reference trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


PARENTS = np.array(
    [-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21]
)
BODY_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "RShoulder", "RArm", "RForeArm", "RHand",
    "LShoulder", "LArm", "LForeArm", "LHand",
    "RLeg", "RShin", "RFoot", "RToe", "LLeg", "LShin", "LFoot", "LToe",
]

RIGHT = (81, 151, 233)
LEFT = (97, 202, 164)
TORSO = (210, 190, 160)
HEAD = (232, 224, 205)
INK = (42, 48, 60)
MUTED = (105, 112, 125)


def load_motion(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "pos": np.asarray(data["rigid_body_pos"], dtype=np.float32),
        "rot": np.asarray(data["rigid_body_rot"], dtype=np.float32),
        "fps": int(data.get("fps", 30)),
    }


def quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    xyz = quat[..., :3]
    uv = np.cross(xyz, vec)
    uuv = np.cross(xyz, uv)
    return vec + 2.0 * (quat[..., 3:4] * uv + uuv)


def motion_axis(gt_pos: np.ndarray) -> np.ndarray:
    root_xy = gt_pos[:, 0, :2]
    centered = root_xy - root_xy.mean(axis=0, keepdims=True)
    if np.linalg.norm(root_xy[-1] - root_xy[0]) > 0.2:
        axis = root_xy[-1] - root_xy[0]
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    axis = axis / max(np.linalg.norm(axis), 1.0e-6)
    if np.dot(axis, root_xy[-1] - root_xy[0]) < 0:
        axis = -axis
    return axis


def project(pos: np.ndarray, axis: np.ndarray) -> np.ndarray:
    side = np.array([-axis[1], axis[0]], dtype=np.float32)
    horizontal = pos[..., :2] @ axis + 0.22 * (pos[..., :2] @ side)
    return np.stack((horizontal, pos[..., 2]), axis=-1)


def body_color(body_id: int) -> tuple[int, int, int]:
    if 7 <= body_id <= 10 or 15 <= body_id <= 18:
        return RIGHT
    if 11 <= body_id <= 14 or 19 <= body_id <= 22:
        return LEFT
    if body_id == 6:
        return HEAD
    return TORSO


def panel_transform(all_projected: np.ndarray, panel_xyxy: tuple[int, int, int, int]):
    x0, y0, x1, y1 = panel_xyxy
    low = np.quantile(all_projected.reshape(-1, 2), 0.005, axis=0)
    high = np.quantile(all_projected.reshape(-1, 2), 0.995, axis=0)
    span = np.maximum(high - low, np.array([0.9, 1.9]))
    margin_x, margin_y = 54, 68
    scale = min((x1 - x0 - 2 * margin_x) / span[0], (y1 - y0 - 2 * margin_y) / span[1])
    center = (low + high) * 0.5

    def fn(points: np.ndarray) -> np.ndarray:
        result = np.empty_like(points)
        result[..., 0] = (points[..., 0] - center[0]) * scale + (x0 + x1) * 0.5
        result[..., 1] = (y0 + y1) * 0.5 - (points[..., 1] - center[1]) * scale
        return np.rint(result).astype(np.int32)

    return fn


def draw_skeleton(canvas: np.ndarray, points: np.ndarray, head_arrow: np.ndarray) -> None:
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        color = body_color(child)
        cv2.line(canvas, tuple(points[parent]), tuple(points[child]), color, 12, cv2.LINE_AA)
    for body_id, point in enumerate(points):
        radius = 11 if body_id in (0, 3, 6) else 8
        cv2.circle(canvas, tuple(point), radius, body_color(body_id), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(point), radius, INK, 2, cv2.LINE_AA)
    cv2.arrowedLine(
        canvas, tuple(points[6]), tuple(head_arrow), (45, 180, 245), 4,
        cv2.LINE_AA, tipLength=0.25,
    )


def render_clip_frames(
    label: str,
    rollout: dict,
    target: dict,
    metrics: dict,
    size: tuple[int, int],
) -> list[np.ndarray]:
    width, height = size
    count = min(len(rollout["pos"]), len(target["pos"])) - 1
    rollout_pos = rollout["pos"][:count].copy()
    target_pos = target["pos"][:count].copy()
    # Compare displacement from the same local origin; accumulated root drift is
    # retained, while irrelevant scene-world offsets are removed.
    rollout_pos[..., :2] -= rollout_pos[0, 0, :2]
    target_pos[..., :2] -= target_pos[0, 0, :2]
    axis = motion_axis(target_pos)
    projected_rollout = project(rollout_pos, axis)
    projected_target = project(target_pos, axis)
    left_panel = (24, 92, width // 2 - 12, height - 38)
    right_panel = (width // 2 + 12, 92, width - 24, height - 38)
    left_tf = panel_transform(projected_target, left_panel)
    right_tf = panel_transform(
        np.concatenate((projected_target, projected_rollout), axis=0), right_panel
    )
    target_pixels = left_tf(projected_target)
    rollout_pixels = right_tf(projected_rollout)

    local_forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    gt_forward = quat_rotate_xyzw(target["rot"][:count, 6], local_forward)
    out_forward = quat_rotate_xyzw(rollout["rot"][:count, 6], local_forward)
    gt_arrow = left_tf(project(target_pos[:, 6] + 0.22 * gt_forward, axis))
    out_arrow = right_tf(project(rollout_pos[:, 6] + 0.22 * out_forward, axis))

    mpjpe = metrics["position_mm"]["mpjpe"]
    root_aligned = metrics["position_mm"]["root_aligned_mpjpe"]
    head_angle = metrics["orientation_deg"]["global_geodesic"]["Head"]["mean"]
    head_pitch = metrics["orientation_deg"]["head_forward_elevation_signed"]["mean"]
    frames = []
    for frame_id in range(count):
        canvas = np.full((height, width, 3), (247, 248, 251), dtype=np.uint8)
        cv2.putText(canvas, "SMPL -> SOMA23 -> FSQ physical decode", (34, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.92, INK, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (34, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, MUTED, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Retargeted SOMA23 target", (82, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.64, INK, 2, cv2.LINE_AA)
        cv2.putText(canvas, "FSQ decode + physics + ego-head feedback", (width // 2 + 48, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, INK, 2, cv2.LINE_AA)
        cv2.line(canvas, (width // 2, 90), (width // 2, height - 28), (215, 218, 225), 2)
        draw_skeleton(canvas, target_pixels[frame_id], gt_arrow[frame_id])
        draw_skeleton(canvas, rollout_pixels[frame_id], out_arrow[frame_id])
        status = (
            f"frame {frame_id:03d}/{count-1:03d}    MPJPE {mpjpe:.1f} mm    "
            f"root-aligned {root_aligned:.1f} mm    Head {head_angle:.2f} deg    "
            f"pitch bias {head_pitch:+.2f} deg"
        )
        cv2.putText(canvas, status, (34, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, MUTED, 1, cv2.LINE_AA)
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", action="append", nargs=3, required=True,
        metavar=("LABEL", "ROLLOUT", "TARGET"),
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=2
    )
    total = 0
    try:
        for label, rollout_path, target_path in args.clip:
            frames = render_clip_frames(
                label,
                load_motion(Path(rollout_path)),
                load_motion(Path(target_path)),
                metrics[label],
                (args.width, args.height),
            )
            for frame in frames:
                writer.append_data(frame)
            # Brief freeze between clips improves readability after concatenation.
            for _ in range(args.fps // 2):
                writer.append_data(frames[-1])
            total += len(frames) + args.fps // 2
    finally:
        writer.close()
    print(f"Wrote {args.output} ({total} frames, {total / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
