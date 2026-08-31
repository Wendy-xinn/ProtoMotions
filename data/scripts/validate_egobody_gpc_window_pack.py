#!/usr/bin/env python3
"""Validate scene-pooled window sampling and precomputed GT ego maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _scene_value(scene, key, default=None):
    return scene.get(key, default) if isinstance(scene, dict) else getattr(scene, key, default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=192)
    parser.add_argument("--expected-points", type=int, default=256)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = {
        "motion_file",
        "scene_file",
        "ego_scene_map_file",
        "motion_scene_ids",
        "clip_map",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Window manifest is missing: {missing}")
    motion = torch.load(manifest["motion_file"], map_location="cpu", weights_only=False)
    scenes = torch.load(manifest["scene_file"], map_location="cpu", weights_only=False)
    maps = torch.load(
        manifest["ego_scene_map_file"],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    counts = motion["motion_num_frames"].long()
    motion_count = counts.numel()
    if not torch.all(counts == args.expected_frames):
        raise ValueError("MotionLib does not contain fixed 192-frame clips")
    if len(manifest["motion_scene_ids"]) != motion_count:
        raise ValueError("motion_scene_ids is not aligned with MotionLib")
    if [str(clip["scene"]) for clip in manifest["clip_map"]] != list(
        manifest["motion_scene_ids"]
    ):
        raise ValueError("clip_map scene labels disagree with motion_scene_ids")

    original_scenes = scenes["original_scenes"]
    scene_labels = [_scene_value(scene, "scene_id") for scene in original_scenes]
    if len(scene_labels) != len(set(scene_labels)) or None in scene_labels:
        raise ValueError("Physical scene pool must contain unique non-empty scene IDs")
    if set(scene_labels) != set(manifest["motion_scene_ids"]):
        raise ValueError("Physical scene pool does not cover every motion scene")
    if any(int(_scene_value(scene, "humanoid_motion_id", -1)) != -1 for scene in original_scenes):
        raise ValueError("Window scenes must not be fixed to individual motion IDs")

    features = maps["features"]
    expected_shape = (
        motion_count,
        args.expected_frames,
        args.expected_points,
        10,
    )
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"Scene map shape {tuple(features.shape)} != {expected_shape}")
    if not torch.isfinite(features).all():
        raise ValueError("Scene maps contain non-finite values")
    if not maps.get("causal", False):
        raise ValueError("Scene maps are not marked causal")
    valid = features[..., 9] > 0.5
    remembered = (features[..., 7] < 0.5) & valid
    print(
        "Validated EgoBody GPC window pack: "
        f"{motion_count} clips x {args.expected_frames} frames, "
        f"{len(scene_labels)} physical scenes, {args.expected_points} map points, "
        f"mean valid={float(valid.sum(-1).float().mean()):.1f}, "
        f"empty frames={int((valid.sum(-1) == 0).sum())}, "
        f"history frames={int(remembered.any(-1).sum())}"
    )


if __name__ == "__main__":
    main()
