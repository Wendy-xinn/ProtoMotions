#!/usr/bin/env python3
"""Split an aligned EgoBody online pack into contiguous training shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch


def _atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _subset_motion(payload: dict, indices: torch.Tensor) -> dict:
    counts = payload["motion_num_frames"].long()
    starts = payload["length_starts"].long()
    total_frames = int(counts.sum())
    num_motions = len(counts)
    frame_indices = torch.cat(
        [torch.arange(starts[i], starts[i] + counts[i]) for i in indices.tolist()]
    )
    output = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim > 0:
            if value.shape[0] == total_frames:
                output[key] = value[frame_indices]
            elif value.shape[0] == num_motions:
                output[key] = value[indices]
            else:
                output[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == num_motions:
            selected = [value[i] for i in indices.tolist()]
            output[key] = tuple(selected) if isinstance(value, tuple) else selected
        else:
            output[key] = value
    selected_counts = counts[indices]
    output["length_starts"] = torch.cat(
        [selected_counts.new_zeros(1), selected_counts.cumsum(0)[:-1]]
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=50)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    count = int(source["num_motions"])
    motion = torch.load(source["motion_file"], map_location="cpu", weights_only=False)
    scene = torch.load(source["scene_file"], map_location="cpu", weights_only=False)
    camera = torch.load(source["ego_camera_file"], map_location="cpu", weights_only=False)

    for start in range(0, count, args.shard_size):
        stop = min(start + args.shard_size, count)
        indices = torch.arange(start, stop)
        shard_name = f"start_{start:04d}_count_{stop - start:04d}"
        shard_root = (args.output_root / shard_name).resolve()
        split_root = shard_root / "train"

        motion_path = split_root / Path(source["motion_file"]).name
        scene_path = split_root / Path(source["scene_file"]).name
        camera_path = split_root / Path(source["ego_camera_file"]).name
        _atomic_save(_subset_motion(motion, indices), motion_path)

        scene_subset = copy.deepcopy(scene)
        scene_subset["original_scenes"] = [
            copy.deepcopy(scene["original_scenes"][i]) for i in indices.tolist()
        ]
        for new_id, item in enumerate(scene_subset["original_scenes"]):
            if isinstance(item, dict):
                item["humanoid_motion_id"] = new_id
            else:
                item.humanoid_motion_id = new_id
        scene_subset["num_original_scenes"] = len(indices)
        _atomic_save(scene_subset, scene_path)

        camera_subset = copy.deepcopy(camera)
        camera_subset["motions"] = [
            copy.deepcopy(camera["motions"][i]) for i in indices.tolist()
        ]
        camera_subset["clip_map"] = []
        clip_map = []
        for new_id, old_id in enumerate(indices.tolist()):
            clip = dict(source["clip_map"][old_id])
            clip["source_motion_id"] = int(clip.get("source_motion_id", old_id))
            clip["motion_id"] = new_id
            clip_map.append(clip)
            camera_clip = dict(clip)
            camera_subset["clip_map"].append(camera_clip)
        _atomic_save(camera_subset, camera_path)

        split_manifest = dict(source)
        split_manifest.update(
            num_motions=len(indices),
            num_frames=int(motion["motion_num_frames"][indices].sum()),
            motion_file=str(motion_path),
            scene_file=str(scene_path),
            ego_camera_file=str(camera_path),
            clip_map=clip_map,
            subset={
                "source_manifest": str(args.manifest.resolve()),
                "start": start,
                "count": len(indices),
            },
        )
        (split_root / "manifest.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (shard_root / "manifest.json").write_text(
            json.dumps({"format_version": 1, "splits": [split_manifest]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (shard_root / ".complete").touch()
        print(f"Wrote {shard_name}")


if __name__ == "__main__":
    main()
