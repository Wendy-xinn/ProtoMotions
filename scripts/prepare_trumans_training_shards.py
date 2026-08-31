#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Write aligned, memory-safe TRUMANS MotionLib/SceneLib training shards."""

import argparse
import os
from pathlib import Path

import torch


FRAME_FIELDS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")


def _atomic_torch_save(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.pt")
    torch.save(data, temp_path)
    os.replace(temp_path, output_path)


def _ranges(total: int, shard_size: int):
    for start in range(0, total, shard_size):
        yield start, min(shard_size, total - start)


def _tag(start: int, count: int) -> str:
    return f"start_{start:04d}_count_{count:04d}"


def prepare_motion_shards(
    input_path: Path, output_root: Path, shard_size: int, overwrite: bool
) -> int:
    print(f"Loading complete MotionLib once: {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    total = len(data["motion_lengths"])
    for shard_number, (start, count) in enumerate(_ranges(total, shard_size), 1):
        output_path = output_root / _tag(start, count) / "motion_lib.pt"
        if output_path.exists() and not overwrite:
            print(f"motion shard {shard_number}: reuse {output_path}")
            continue
        indices = list(range(start, start + count))
        frame_parts = []
        for index in indices:
            frame_start = int(data["length_starts"][index])
            frame_count = int(data["motion_num_frames"][index])
            frame_parts.append(torch.arange(frame_start, frame_start + frame_count))
        frame_indices = torch.cat(frame_parts).long()
        frame_counts = data["motion_num_frames"][indices].clone()
        shifted = frame_counts.roll(1)
        shifted[0] = 0
        output = {
            "length_starts": shifted.cumsum(0),
            "motion_num_frames": frame_counts,
            "motion_lengths": data["motion_lengths"][indices].clone(),
            "motion_dt": data["motion_dt"][indices].clone(),
            "motion_weights": data["motion_weights"][indices].clone(),
        }
        for field in FRAME_FIELDS:
            if field in data and data[field] is not None:
                output[field] = data[field][frame_indices]
        if "motion_files" in data:
            output["motion_files"] = tuple(data["motion_files"][i] for i in indices)
        _atomic_torch_save(output, output_path)
        print(
            f"motion shard {shard_number}: clips {start}..{start + count - 1}, "
            f"frames={len(frame_indices)}"
        )
    return total


def prepare_scene_shards(
    input_path: Path, output_root: Path, shard_size: int, overwrite: bool
) -> int:
    print(f"Loading complete SceneLib once: {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    scenes = data["original_scenes"]
    total = len(scenes)
    for shard_number, (start, count) in enumerate(_ranges(total, shard_size), 1):
        shard_dir = output_root / _tag(start, count)
        output_path = shard_dir / "scene_lib.pt"
        complete_path = shard_dir / ".complete"
        if output_path.exists() and not overwrite:
            print(f"scene shard {shard_number}: reuse {output_path}")
        else:
            selected = [scenes[index] for index in range(start, start + count)]
            for local_motion_id, scene in enumerate(selected):
                if isinstance(scene, dict):
                    scene["humanoid_motion_id"] = local_motion_id
                else:
                    scene.humanoid_motion_id = local_motion_id
            output = {
                "original_scenes": selected,
                "num_original_scenes": count,
                "num_objects_per_scene": data["num_objects_per_scene"],
            }
            _atomic_torch_save(output, output_path)
            print(f"scene shard {shard_number}: clips {start}..{start + count - 1}")
        motion_path = shard_dir / "motion_lib.pt"
        if motion_path.exists() and output_path.exists():
            complete_path.touch()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("motions", "scenes"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("--shard-size must be >= 1")
    if args.kind == "motions":
        total = prepare_motion_shards(
            args.input, args.output_root, args.shard_size, args.overwrite
        )
    else:
        total = prepare_scene_shards(
            args.input, args.output_root, args.shard_size, args.overwrite
        )
    print(f"Prepared {args.kind} for {total} clips")


if __name__ == "__main__":
    main()
