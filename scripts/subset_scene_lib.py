#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Create a small SceneLib pack aligned with a subset MotionLib."""

import argparse
import copy
import os
from pathlib import Path

import torch


def _freeze_near_static_objects(
    scenes: list,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> None:
    """Make effectively stationary clip objects kinematic in PhysX.

    TRUMANS labels an object dynamic for the complete source take even when a
    selected clip contains only tracking noise. Leaving such objects dynamic
    lets gravity/contact move them between reference-pose writes and produces
    a visible depenetration-reset jitter loop.
    """
    for scene in scenes:
        objects = scene.get("objects", []) if isinstance(scene, dict) else scene.objects
        for obj in objects:
            translations = torch.as_tensor(obj.get("translation", []), dtype=torch.float64)
            rotations = torch.as_tensor(obj.get("rotation", []), dtype=torch.float64)
            if translations.shape[0] <= 1 or rotations.shape[0] <= 1:
                continue
            max_translation = torch.linalg.norm(
                translations - translations[:1], dim=-1
            ).max()
            rotations = rotations / torch.linalg.norm(
                rotations, dim=-1, keepdim=True
            ).clamp_min(1.0e-12)
            dots = torch.abs((rotations * rotations[:1]).sum(dim=-1)).clamp(0.0, 1.0)
            max_rotation_deg = torch.rad2deg(2.0 * torch.acos(dots)).max()
            if (
                float(max_translation) <= translation_threshold_m
                and float(max_rotation_deg) <= rotation_threshold_deg
            ):
                obj["translation"] = [translations[0].tolist()]
                obj["rotation"] = [rotations[0].tolist()]
                obj["fps"] = 1.0
                obj.setdefault("options", {})["fix_base_link"] = True
                print(
                    "  froze near-static object "
                    f"{obj.get('object_path', obj.get('type', '?'))}: "
                    f"max_translation={float(max_translation):.6f}m, "
                    f"max_rotation={float(max_rotation_deg):.3f}deg"
                )


def subset_scene_lib(
    input_path: Path,
    output_path: Path,
    indices: list[int],
    remap_motion_ids: bool = True,
    frame_start: int | None = None,
    frame_count: int | None = None,
    motion_fps: float = 30.0,
    freeze_near_static: bool = False,
) -> None:
    print(f"Loading scene library from {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    scenes = data["original_scenes"]
    invalid = [index for index in indices if index < 0 or index >= len(scenes)]
    if invalid:
        raise ValueError(f"Scene indices out of range [0, {len(scenes)}): {invalid}")

    selected = [copy.deepcopy(scenes[index]) for index in indices]
    if frame_start is not None or frame_count is not None:
        start_seconds = (frame_start or 0) / motion_fps
        duration_seconds = None if frame_count is None else frame_count / motion_fps
        for scene in selected:
            objects = scene.get("objects", []) if isinstance(scene, dict) else scene.objects
            for obj in objects:
                obj_fps = float(obj.get("fps", 1.0))
                translations = obj.get("translation", [])
                rotations = obj.get("rotation", [])
                # One-sample trajectories are static and must remain available.
                if len(translations) <= 1:
                    continue
                obj_start = min(int(round(start_seconds * obj_fps)), len(translations) - 1)
                obj_end = len(translations)
                if duration_seconds is not None:
                    obj_count = max(2, int(round(duration_seconds * obj_fps)))
                    obj_end = min(obj_start + obj_count, len(translations))
                obj["translation"] = translations[obj_start:obj_end]
                if len(rotations) > 1:
                    obj["rotation"] = rotations[obj_start:obj_end]
    if freeze_near_static:
        _freeze_near_static_objects(selected, 0.01, 2.0)
    if remap_motion_ids:
        for new_motion_id, scene in enumerate(selected):
            if isinstance(scene, dict):
                scene["humanoid_motion_id"] = new_motion_id
            else:
                scene.humanoid_motion_id = new_motion_id

    output = {
        "original_scenes": selected,
        "num_original_scenes": len(selected),
        "num_objects_per_scene": data["num_objects_per_scene"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.pt")
    torch.save(output, temp_path)
    os.replace(temp_path, output_path)
    print(
        f"Saved {len(selected)} scenes to {output_path}; "
        f"motion ids remapped={remap_motion_ids}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--keep-motion-ids",
        action="store_true",
        help="Do not remap selected scene motion ids to 0..N-1.",
    )
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--motion-fps", type=float, default=30.0)
    parser.add_argument(
        "--freeze-near-static",
        action="store_true",
        help="Make objects with only clip-local tracking noise fixed/kinematic.",
    )
    args = parser.parse_args()
    subset_scene_lib(
        args.input,
        args.output,
        args.indices,
        remap_motion_ids=not args.keep_motion_ids,
        frame_start=args.frame_start,
        frame_count=args.frame_count,
        motion_fps=args.motion_fps,
        freeze_near_static=args.freeze_near_static,
    )


if __name__ == "__main__":
    main()
