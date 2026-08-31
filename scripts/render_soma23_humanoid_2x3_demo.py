#!/usr/bin/env python3
"""Render report-ready 2x3 SOMA23 mesh comparisons without a browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from examples.visualize_soma_scenes import build_body_local_meshes


BACKGROUND = np.array([244, 246, 249], dtype=np.uint8)
PANEL_BACKGROUND = np.array([250, 251, 253], dtype=np.uint8)
INK = (42, 48, 60)
MUTED = (105, 112, 125)
GRID = (218, 222, 228)
DIVIDER = (205, 210, 218)


def load_motion(path: Path) -> dict[str, np.ndarray | int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "pos": np.asarray(data["rigid_body_pos"], dtype=np.float32),
        "rot": np.asarray(data["rigid_body_rot"], dtype=np.float32),
        "fps": int(data.get("fps", 30)),
    }


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-8)
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def motion_axis(reference_pos: np.ndarray) -> np.ndarray:
    root = reference_pos[:-1, 0, :2]
    displacement = root[-1] - root[0]
    if np.linalg.norm(displacement) > 0.2:
        axis = displacement
    else:
        centered = root - root.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    return axis / max(float(np.linalg.norm(axis)), 1.0e-6)


def camera_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = np.array([-axis[1], axis[0]], dtype=np.float32)
    horizontal_view = side - 0.32 * axis
    view = np.array([horizontal_view[0], horizontal_view[1], -0.22], dtype=np.float32)
    view /= np.linalg.norm(view)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(view, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)
    return right, up, view


def align_motion(motion: dict, reference: dict) -> np.ndarray:
    pos = motion["pos"].copy()
    pos += reference["pos"][0, 0] - pos[0, 0]
    return pos


class MeshRenderer:
    def __init__(self) -> None:
        self.meshes = {}
        for body_id, mesh in build_body_local_meshes().items():
            colors = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float32)
            self.meshes[body_id] = (
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32),
                colors,
            )
        self.light = np.array([-0.35, -0.45, 0.82], dtype=np.float32)
        self.light /= np.linalg.norm(self.light)

    @staticmethod
    def _project(
        points: np.ndarray,
        center: np.ndarray,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray],
        scale: float,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        right, up, view = basis
        relative = points - center
        x = relative @ right
        y = relative @ up
        depth = relative @ view
        pixels = np.stack(
            (width * 0.5 + scale * x, height * 0.53 - scale * y), axis=-1
        )
        return np.rint(pixels).astype(np.int32), depth

    def _draw_ground(
        self,
        canvas: np.ndarray,
        center: np.ndarray,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray],
        scale: float,
    ) -> None:
        height, width = canvas.shape[:2]
        extent = 1.8
        for offset in np.linspace(-extent, extent, 9):
            for along_x in (True, False):
                if along_x:
                    points = np.array(
                        [[-extent, offset, 0.0], [extent, offset, 0.0]], dtype=np.float32
                    )
                else:
                    points = np.array(
                        [[offset, -extent, 0.0], [offset, extent, 0.0]], dtype=np.float32
                    )
                points[:, :2] += center[:2]
                pixels, _ = self._project(points, center, basis, scale, width, height)
                cv2.line(canvas, tuple(pixels[0]), tuple(pixels[1]), GRID, 1, cv2.LINE_AA)

    def render(
        self,
        motion: dict,
        aligned_pos: np.ndarray,
        reference: dict,
        frame_id: int,
        basis: tuple[np.ndarray, np.ndarray, np.ndarray],
        size: tuple[int, int],
        view_mode: str,
    ) -> np.ndarray:
        width, height = size
        canvas = np.full((height, width, 3), PANEL_BACKGROUND, dtype=np.uint8)
        ref_root = reference["pos"][frame_id, 0]
        if view_mode == "upper":
            center = np.array([ref_root[0], ref_root[1], ref_root[2] + 0.45], dtype=np.float32)
            scale = min(width / 1.45, height / 1.25)
        else:
            center = np.array([ref_root[0], ref_root[1], 0.93], dtype=np.float32)
            scale = min(width / 1.9, height / 1.95)
            self._draw_ground(canvas, center, basis, scale)

        triangles = []
        for body_id, (vertices, faces, face_colors) in self.meshes.items():
            rotation = quat_xyzw_to_matrix(motion["rot"][frame_id, body_id])
            world_vertices = vertices @ rotation.T + aligned_pos[frame_id, body_id]
            pixels, depth = self._project(
                world_vertices, center, basis, scale, width, height
            )
            face_vertices = world_vertices[faces]
            normals = np.cross(
                face_vertices[:, 1] - face_vertices[:, 0],
                face_vertices[:, 2] - face_vertices[:, 0],
            )
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1.0e-6)
            intensity = np.clip(0.48 + 0.52 * np.abs(normals @ self.light), 0.0, 1.0)
            colors = np.clip(face_colors * intensity[:, None], 0, 255).astype(np.uint8)
            face_depth = depth[faces].mean(axis=1)
            for face_id, face in enumerate(faces):
                triangles.append(
                    (float(face_depth[face_id]), pixels[face], tuple(int(v) for v in colors[face_id]))
                )

        for _, points, color in sorted(triangles, key=lambda item: item[0], reverse=True):
            cv2.fillConvexPoly(canvas, points, color, cv2.LINE_AA)

        head_rotation = quat_xyzw_to_matrix(motion["rot"][frame_id, 6])
        head_start = aligned_pos[frame_id, 6]
        head_end = head_start + head_rotation @ np.array(
            [0.0, -0.28, 0.0], dtype=np.float32
        )
        arrow, _ = self._project(
            np.stack((head_start, head_end)), center, basis, scale, width, height
        )
        cv2.arrowedLine(
            canvas, tuple(arrow[0]), tuple(arrow[1]), (32, 42, 52), 7,
            cv2.LINE_AA, tipLength=0.24,
        )
        cv2.arrowedLine(
            canvas, tuple(arrow[0]), tuple(arrow[1]), (35, 174, 238), 4,
            cv2.LINE_AA, tipLength=0.24,
        )
        return canvas


def metric_text(metrics: dict | None, label: str, style: str) -> str:
    if metrics is None:
        return "Kinematic SOMA23 reference"
    pos = metrics[label]["position_mm"]
    orientation = metrics[label]["orientation_deg"]
    if style == "pitch":
        head_pitch = orientation["head_forward_elevation_signed"]["mean"]
        chest_pitch = orientation["chest_forward_elevation_signed"]["mean"]
        return f"Head pitch {head_pitch:+.1f} deg   Chest pitch {chest_pitch:+.1f} deg"
    chest = orientation["global_geodesic"]["Chest"]["mean"]
    return f"RA-MPJPE {pos['root_aligned_mpjpe']:.1f} mm   Chest {chest:.1f} deg"


def draw_centered_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(8, (canvas.shape[1] - size[0]) // 2)
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def render_group_frame(
    renderer: MeshRenderer,
    rows: list[dict],
    frame_id: int,
    group_id: int,
    current_metrics: dict,
    ik_metrics: dict,
    metric_style: str,
    column_titles: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    header_h = 112
    row_h = (height - header_h) // 2
    panel_w = width // 3
    cv2.putText(
        canvas, f"SMPL to SOMA23 retarget and GPC-FSQ physical decode   |   Group {group_id}",
        (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, INK, 2, cv2.LINE_AA,
    )
    columns = list(zip(column_titles, ((53, 104, 173), (47, 137, 91), (184, 74, 68))))
    for col, (title, color) in enumerate(columns):
        x0 = col * panel_w
        cv2.rectangle(canvas, (x0, 58), (x0 + panel_w, 108), color, -1)
        segment = canvas[58:109, x0 : x0 + panel_w]
        draw_centered_text(segment, title, 32, 0.52, (255, 255, 255), 1)

    for row_id, row in enumerate(rows):
        y0 = header_h + row_id * row_h
        label_h = 47
        panel_h = row_h - label_h
        streams = [row["target"], row["current"], row["ik"]]
        metric_lines = [
            metric_text(None, row["metric_label"], metric_style),
            metric_text(current_metrics, row["metric_label"], metric_style),
            metric_text(ik_metrics, row["metric_label"], metric_style),
        ]
        for col, stream in enumerate(streams):
            x0 = col * panel_w
            panel = renderer.render(
                stream,
                row["aligned"][col],
                row["target"],
                frame_id,
                row["basis"],
                (panel_w, panel_h),
                row["view"],
            )
            canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
            footer = canvas[y0 + panel_h : y0 + row_h, x0 : x0 + panel_w]
            footer[:] = (236, 239, 243)
            draw_centered_text(footer, metric_lines[col], 18, 0.43, INK, 1)
            draw_centered_text(
                footer,
                f"{row['display']}   frame {frame_id:03d}/{row['count'] - 1:03d}",
                39,
                0.40,
                MUTED,
                1,
            )
        cv2.line(canvas, (0, y0), (width, y0), DIVIDER, 2)
    for col in (1, 2):
        cv2.line(canvas, (col * panel_w, 58), (col * panel_w, height), DIVIDER, 2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def parse_row(values: list[str]) -> dict:
    label, display, view, target_path, current_path, ik_path = values
    target = load_motion(Path(target_path))
    current = load_motion(Path(current_path))
    ik = load_motion(Path(ik_path))
    count = min(len(target["pos"]), len(current["pos"]), len(ik["pos"])) - 1
    if count < 1:
        raise ValueError(f"{label} has no valid frame transitions")
    basis = camera_basis(motion_axis(target["pos"]))
    return {
        "metric_label": label,
        "display": display,
        "view": view,
        "target": target,
        "current": current,
        "ik": ik,
        "basis": basis,
        "count": count,
        "aligned": [
            align_motion(target, target),
            align_motion(current, target),
            align_motion(ik, target),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row", action="append", nargs=6, required=True,
        metavar=("METRIC_LABEL", "DISPLAY", "VIEW", "TARGET", "CURRENT", "IK"),
    )
    parser.add_argument("--current-metrics", type=Path, required=True)
    parser.add_argument("--ik-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--metric-style", choices=("tracking", "pitch"), default="tracking"
    )
    parser.add_argument(
        "--column-title", action="append",
        help="Override the three column titles (pass exactly three times).",
    )
    args = parser.parse_args()
    if len(args.row) % 2:
        raise ValueError("Pass an even number of --row entries (two per group)")
    rows = [parse_row(row) for row in args.row]
    current_metrics = json.loads(args.current_metrics.read_text())
    ik_metrics = json.loads(args.ik_metrics.read_text())
    column_titles = args.column_title or [
        "Orientation retarget target",
        "Orientation retarget -> FSQ + physics",
        "Position IK -> FSQ + physics",
    ]
    if len(column_titles) != 3:
        raise ValueError("Pass --column-title exactly three times")
    renderer = MeshRenderer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=8, macro_block_size=2
    )
    total = 0
    try:
        for group_start in range(0, len(rows), 2):
            group_rows = rows[group_start : group_start + 2]
            group_id = group_start // 2 + 1
            frame_count = min(row["count"] for row in group_rows)
            for frame_id in range(frame_count):
                writer.append_data(
                    render_group_frame(
                        renderer, group_rows, frame_id, group_id,
                        current_metrics, ik_metrics, args.metric_style,
                        column_titles, args.width, args.height,
                    )
                )
                total += 1
            for _ in range(args.fps // 2):
                writer.append_data(
                    render_group_frame(
                        renderer, group_rows, 190, group_id,
                        current_metrics, ik_metrics, args.metric_style,
                        column_titles, args.width, args.height,
                    )
                )
                total += 1
    finally:
        writer.close()
    print(f"Wrote {args.output} ({total} frames, {total / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
