#!/usr/bin/env python3
"""Render IsaacLab recorder motion files without requiring an IsaacLab viewport.

This is a fallback for WSL setups where PhysX/CUDA works but Isaac Sim's
D3D12/Vulkan viewport cannot create PNG frames. It renders the recorded SMPL
body positions and, when available, the recorded terrain height field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),
]


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _terrain_height_sampler(terrain_path: Path | None):
    if terrain_path is None:
        return None
    data = _load(terrain_path)
    raw = np.asarray(data["height_field_raw"])
    horizontal = float(data["horizontal_scale"])
    vertical = float(data["vertical_scale"])

    def sample(xy: np.ndarray) -> np.ndarray:
        # Terrain's heightfield coordinates use x/y pixel indices directly.
        ix = np.clip(np.rint(xy[:, 0] / horizontal).astype(int), 0, raw.shape[0] - 1)
        iy = np.clip(np.rint(xy[:, 1] / horizontal).astype(int), 0, raw.shape[1] - 1)
        return raw[ix, iy] * vertical

    return sample


def render(motion_path: Path, output_path: Path, terrain_path: Path | None) -> None:
    motion = _load(motion_path)
    positions = np.asarray(motion["rigid_body_pos"], dtype=np.float32)
    fps = int(motion.get("fps", 30))
    sampler = _terrain_height_sampler(terrain_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame in positions:
            root = frame[0]
            fig = plt.figure(figsize=(8, 6), dpi=110)
            ax = fig.add_subplot(111, projection="3d")
            for a, b in EDGES:
                ax.plot(
                    [frame[a, 0], frame[b, 0]],
                    [frame[a, 1], frame[b, 1]],
                    [frame[a, 2], frame[b, 2]],
                    color="#168447",
                    linewidth=2.2,
                )
            ax.scatter(
                frame[:, 0], frame[:, 1], frame[:, 2],
                s=15, c="#d62728", depthshade=True,
            )

            # Draw a local terrain patch. Flat recordings use z=0.
            span = 2.8
            gx = np.linspace(root[0] - span, root[0] + span, 18)
            gy = np.linspace(root[1] - span, root[1] + span, 18)
            xx, yy = np.meshgrid(gx, gy, indexing="ij")
            xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
            if sampler is None:
                zz = np.zeros_like(xx)
            else:
                zz = sampler(xy).reshape(xx.shape)
            ax.plot_surface(xx, yy, zz, alpha=0.35, color="#c98d42", linewidth=0)

            ax.set_xlim(root[0] - span, root[0] + span)
            ax.set_ylim(root[1] - span, root[1] + span)
            ax.set_zlim(max(0.0, root[2] - 1.1), root[2] + 1.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=18, azim=-62)
            ax.set_title(output_path.stem)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
            plt.close(fig)
    finally:
        writer.close()
    print(f"Wrote {output_path} ({len(positions)} frames at {fps} FPS)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, default=None)
    args = parser.parse_args()
    render(args.motion, args.output, args.terrain)


if __name__ == "__main__":
    main()
