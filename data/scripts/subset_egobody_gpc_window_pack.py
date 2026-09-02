"""Create an aligned clip subset of an EgoBody GPC window-training pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    return parser.parse_args()


def _subset_motion_lib(payload: dict, indices: torch.Tensor) -> dict:
    frame_counts = payload["motion_num_frames"].long()
    starts = payload["length_starts"].long()
    total_frames = int(frame_counts.sum())
    num_motions = int(frame_counts.numel())
    frame_indices = torch.cat(
        [
            torch.arange(starts[index], starts[index] + frame_counts[index])
            for index in indices.tolist()
        ]
    )

    subset = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim > 0:
            if value.shape[0] == total_frames:
                subset[key] = value[frame_indices]
            elif value.shape[0] == num_motions:
                subset[key] = value[indices]
            else:
                subset[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == num_motions:
            selected = [value[index] for index in indices.tolist()]
            subset[key] = tuple(selected) if isinstance(value, tuple) else selected
        else:
            subset[key] = value

    selected_counts = frame_counts[indices]
    subset["length_starts"] = torch.cat(
        [selected_counts.new_zeros(1), selected_counts.cumsum(0)[:-1]]
    )
    if "motion_weights" in subset:
        subset["motion_weights"] = torch.ones_like(subset["motion_weights"])
    return subset


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    stop = args.start + args.count
    if args.start < 0 or args.count < 1 or stop > int(source["num_motions"]):
        raise ValueError(
            f"Invalid subset [{args.start}, {stop}) for {source['num_motions']} motions"
        )
    indices = torch.arange(args.start, stop, dtype=torch.long)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_payload = torch.load(
        source["motion_file"], map_location="cpu", weights_only=False
    )
    motion_subset = _subset_motion_lib(motion_payload, indices)
    motion_path = output_dir / "motion_lib_soma23_grounded.pt"
    torch.save(motion_subset, motion_path)

    scene_map = torch.load(
        source["ego_scene_map_file"], map_location="cpu", weights_only=False
    )
    map_subset = dict(scene_map)
    map_subset["features"] = scene_map["features"][indices]
    map_subset["num_frames"] = scene_map["num_frames"][indices]
    map_subset["motion_scene_ids"] = [
        scene_map["motion_scene_ids"][index] for index in indices.tolist()
    ]
    scene_map_path = output_dir / "gt_ego_scene_maps.pt"
    torch.save(map_subset, scene_map_path)

    clip_map = []
    for new_id, old_id in enumerate(indices.tolist()):
        clip = dict(source["clip_map"][old_id])
        clip["source_motion_id"] = int(clip.get("source_motion_id", old_id))
        clip["motion_id"] = new_id
        clip_map.append(clip)

    selected_scene_ids = [source["motion_scene_ids"][i] for i in indices.tolist()]
    unique_scene_ids = list(dict.fromkeys(selected_scene_ids))
    scene_payload = torch.load(
        source["scene_file"], map_location="cpu", weights_only=False
    )
    scenes_by_id = {
        (
            scene["scene_id"]
            if isinstance(scene, dict)
            else getattr(scene, "scene_id")
        ): scene
        for scene in scene_payload["original_scenes"]
    }
    scene_subset = dict(scene_payload)
    scene_subset["original_scenes"] = [scenes_by_id[key] for key in unique_scene_ids]
    scene_subset["num_original_scenes"] = len(unique_scene_ids)
    scene_subset["scene_ids"] = unique_scene_ids
    scene_path = output_dir / "scene_lib_window_training_isaaclab.pt"
    torch.save(scene_subset, scene_path)

    subset_manifest = dict(source)
    subset_manifest.update(
        {
            "num_motions": len(indices),
            "num_frames": int(map_subset["num_frames"].sum()),
            "motion_file": str(motion_path),
            "scene_file": str(scene_path),
            "ego_scene_map_file": str(scene_map_path),
            "clip_map": clip_map,
            "motion_scene_ids": selected_scene_ids,
            "subset": {
                "source_manifest": str(manifest_path),
                "start": args.start,
                "count": args.count,
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(subset_manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(indices)} clips to {output_dir}")


if __name__ == "__main__":
    main()
