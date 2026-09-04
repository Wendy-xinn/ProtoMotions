#!/usr/bin/env python3
"""Render a synchronized SOMA23 reference and GPC rollout as a mesh video."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_soma23_humanoid_2x3_demo import (
    BACKGROUND,
    DIVIDER,
    INK,
    MUTED,
    MeshRenderer,
    camera_basis,
    load_motion,
    motion_axis,
)


def centered_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, (canvas.shape[1] - size[0]) // 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def frame_errors(reference: dict, rollout: dict, frame_id: int) -> tuple[float, float]:
    delta = rollout["pos"][frame_id] - reference["pos"][frame_id]
    per_body = np.linalg.norm(delta, axis=-1)
    return float(per_body.mean()), float(per_body.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument(
        "--corrected-rollout",
        type=Path,
        default=None,
        help="Optional second rollout rendered as a third comparison column.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EgoBody GPC closed-loop rollout")
    parser.add_argument("--rollout-label", default="GPC free rollout")
    parser.add_argument(
        "--corrected-rollout-label",
        default="GPC + Head orientation feedback",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--view",
        choices=("full", "upper"),
        default="full",
        help="Humanoid framing used for every comparison panel.",
    )
    args = parser.parse_args()

    reference = load_motion(args.reference)
    rollout = load_motion(args.rollout)
    corrected = load_motion(args.corrected_rollout) if args.corrected_rollout else None
    streams = [
        ("Synchronized SOMA23 GT", reference),
        (args.rollout_label, rollout),
    ]
    if corrected is not None:
        streams.append((args.corrected_rollout_label, corrected))
    frame_count = min(len(motion["pos"]) for _, motion in streams)
    if frame_count < 2:
        raise ValueError("The synchronized recording has fewer than two frames")

    fps = args.fps or int(reference["fps"])
    basis = camera_basis(motion_axis(reference["pos"][:frame_count]))
    renderer = MeshRenderer()
    panel_width = args.width // len(streams)
    header_height = 92
    footer_height = 58
    panel_height = args.height - header_height - footer_height
    mean_errors = []
    max_errors = []
    corrected_mean_errors = []
    corrected_max_errors = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=fps, codec="libx264", quality=8, macro_block_size=2
    )
    try:
        for frame_id in range(frame_count):
            mean_error, max_error = frame_errors(reference, rollout, frame_id)
            mean_errors.append(mean_error)
            max_errors.append(max_error)
            if corrected is not None:
                corrected_mean_error, corrected_max_error = frame_errors(
                    reference, corrected, frame_id
                )
                corrected_mean_errors.append(corrected_mean_error)
                corrected_max_errors.append(corrected_max_error)

            canvas = np.full((args.height, args.width, 3), BACKGROUND, dtype=np.uint8)
            centered_text(canvas, args.title, 38, 0.82, INK, 2)
            centered_text(
                canvas,
                f"frame {frame_id:03d}/{frame_count - 1:03d}",
                72,
                0.52,
                MUTED,
            )
            for column, (label, motion) in enumerate(streams):
                x0 = column * panel_width
                panel = renderer.render(
                    motion,
                    motion["pos"],
                    reference,
                    frame_id,
                    basis,
                    (panel_width, panel_height),
                    args.view,
                )
                canvas[
                    header_height : header_height + panel_height,
                    x0 : x0 + panel_width,
                ] = panel
                footer = canvas[header_height + panel_height :, x0 : x0 + panel_width]
                footer[:] = (236, 239, 243)
                if column == 0:
                    detail = "Reference trajectory"
                elif column == 1:
                    detail = (
                        f"mean error {mean_error * 100:.1f} cm   "
                        f"max joint {max_error * 100:.1f} cm"
                    )
                else:
                    detail = (
                        f"mean error {corrected_mean_error * 100:.1f} cm   "
                        f"max joint {corrected_max_error * 100:.1f} cm"
                    )
                centered_text(footer, label, 22, 0.53, INK, 1)
                centered_text(footer, detail, 47, 0.44, MUTED, 1)
            for column in range(1, len(streams)):
                cv2.line(
                    canvas,
                    (column * panel_width, header_height),
                    (column * panel_width, args.height),
                    DIVIDER,
                    2,
                )
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

    summary = (
        f"Wrote {args.output} ({frame_count} frames at {fps} FPS); "
        f"mean body error={np.mean(mean_errors) * 100:.2f} cm, "
        f"worst joint error={np.max(max_errors) * 100:.2f} cm"
    )
    if corrected is not None:
        summary += (
            f"; corrected mean body error="
            f"{np.mean(corrected_mean_errors) * 100:.2f} cm, "
            f"corrected worst joint error="
            f"{np.max(corrected_max_errors) * 100:.2f} cm"
        )
    print(summary)


if __name__ == "__main__":
    main()
