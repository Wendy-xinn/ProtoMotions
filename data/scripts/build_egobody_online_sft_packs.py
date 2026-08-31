#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Merge per-recording EgoBody assets into motion-paired online SFT packs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch


FRAME_FIELDS = ("gts", "grs", "gvs", "gavs", "dps", "dvs", "lrs", "contacts")
MOTION_FIELDS = (
    "motion_num_frames",
    "motion_lengths",
    "motion_dt",
    "motion_weights",
)
OPTIONAL_MOTION_FIELDS = (
    "retarget_root_height_offsets_m",
    "retarget_unlabelled_support_fallback_mask",
)
RECORDING_LOCAL_METADATA = {
    "retarget_unlabelled_support_fallback_motion_ids",
    "retarget_unlabelled_support_fallback",
}
SOMA23_NUM_BODIES = 23
SOMA23_NUM_DOFS = 66


def _validate_soma23_motion(payload: dict, source: Path) -> None:
    """Reject an accidentally supplied SMPL package before merging GPC data."""
    required_shapes = {
        "gts": (SOMA23_NUM_BODIES, 3),
        "grs": (SOMA23_NUM_BODIES, 4),
        "gvs": (SOMA23_NUM_BODIES, 3),
        "gavs": (SOMA23_NUM_BODIES, 3),
        "lrs": (SOMA23_NUM_BODIES, 4),
        "contacts": (SOMA23_NUM_BODIES,),
        "dps": (SOMA23_NUM_DOFS,),
        "dvs": (SOMA23_NUM_DOFS,),
    }
    frame_count = None
    for key, trailing_shape in required_shapes.items():
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape[1:]) != trailing_shape:
            actual = None if value is None else tuple(value.shape)
            raise ValueError(
                f"{source} is not a SOMA23 MotionLib: {key} shape {actual}, "
                f"expected [frames, {', '.join(map(str, trailing_shape))}]"
            )
        if frame_count is None:
            frame_count = value.shape[0]
        elif value.shape[0] != frame_count:
            raise ValueError(f"Inconsistent frame fields in {source}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite values in {source}:{key}")
    if int(payload["motion_num_frames"].sum()) != frame_count:
        raise ValueError(f"Motion counts do not cover all frames in {source}")


def _atomic_torch_save(payload, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


def _set_motion_id(scene, motion_id: int) -> None:
    if isinstance(scene, dict):
        scene["humanoid_motion_id"] = motion_id
    else:
        scene.humanoid_motion_id = motion_id


def _merge_split(
    manifest: dict,
    split: str,
    output_root: Path,
    motion_filename: str,
    camera_filename: str,
) -> dict:
    prepared_root = Path(manifest["prepared_root"])
    clips = (
        list(manifest["clips"])
        if split == "all"
        else [clip for clip in manifest["clips"] if clip["split"] == split]
    )
    by_recording: dict[str, list[dict]] = {}
    for clip in clips:
        by_recording.setdefault(clip["recording"], []).append(clip)

    frame_parts: dict[str, list[torch.Tensor]] = {key: [] for key in FRAME_FIELDS}
    motion_parts: dict[str, list[torch.Tensor]] = {key: [] for key in MOTION_FIELDS}
    optional_motion_parts: dict[str, list[torch.Tensor]] = {
        key: [] for key in OPTIONAL_MOTION_FIELDS
    }
    motion_files: list[str] = []
    scenes = []
    camera_motions = []
    clip_map = []
    common_motion_metadata = None
    num_objects_per_scene = None

    for recording, recording_clips in by_recording.items():
        recording_dir = prepared_root / recording
        motion_payload = torch.load(
            recording_dir / motion_filename,
            map_location="cpu",
            weights_only=False,
        )
        if "retarget_unlabelled_support_fallback_mask" not in motion_payload:
            fallback_mask = torch.zeros(
                len(motion_payload["motion_num_frames"]), dtype=torch.bool
            )
            fallback_ids = motion_payload.get(
                "retarget_unlabelled_support_fallback_motion_ids"
            )
            if fallback_ids is not None:
                fallback_mask[fallback_ids] = True
            motion_payload["retarget_unlabelled_support_fallback_mask"] = fallback_mask
        if "soma23" in motion_filename.lower():
            _validate_soma23_motion(motion_payload, recording_dir / motion_filename)
        scene_payload = torch.load(
            recording_dir / "scene_lib_training_isaaclab.pt",
            map_location="cpu",
            weights_only=False,
        )
        camera_payload = torch.load(
            recording_dir / camera_filename,
            map_location="cpu",
            weights_only=False,
        )
        recording_clips = sorted(
            recording_clips, key=lambda item: (item["start"], item["clip_id"])
        )
        local_count = len(recording_clips)
        if int(motion_payload["motion_num_frames"].shape[0]) != local_count:
            raise ValueError(f"Motion count mismatch for {recording}")
        if len(camera_payload["motions"]) != local_count:
            raise ValueError(f"Camera count mismatch for {recording}")
        if len(scene_payload["original_scenes"]) != 1:
            raise ValueError(f"Expected one recording scene for {recording}")
        if num_objects_per_scene is None:
            num_objects_per_scene = int(scene_payload["num_objects_per_scene"])
        elif num_objects_per_scene != int(scene_payload["num_objects_per_scene"]):
            raise ValueError("All merged scenes must have the same object count")

        for key in FRAME_FIELDS:
            frame_parts[key].append(motion_payload[key])
        for key in MOTION_FIELDS:
            motion_parts[key].append(motion_payload[key])
        for key in OPTIONAL_MOTION_FIELDS:
            if key in motion_payload:
                optional_motion_parts[key].append(motion_payload[key])
        motion_files.extend(str(value) for value in motion_payload["motion_files"])

        if common_motion_metadata is None:
            common_motion_metadata = {
                key: copy.deepcopy(value)
                for key, value in motion_payload.items()
                if key
                not in set(FRAME_FIELDS)
                | set(MOTION_FIELDS)
                | set(OPTIONAL_MOTION_FIELDS)
                | {"length_starts", "motion_files"}
                | RECORDING_LOCAL_METADATA
            }

        source_scene = scene_payload["original_scenes"][0]
        for local_index, clip in enumerate(recording_clips):
            global_index = len(scenes)
            scene = copy.deepcopy(source_scene)
            _set_motion_id(scene, global_index)
            scenes.append(scene)
            camera_motions.append(copy.deepcopy(camera_payload["motions"][local_index]))
            clip_map.append(
                {
                    "motion_id": global_index,
                    "recording": recording,
                    "local_motion_id": local_index,
                    "clip_id": clip["clip_id"],
                    "start": clip["start"],
                    "count": clip["count"],
                    "scene": clip["scene"],
                }
            )

    merged_motion = dict(common_motion_metadata or {})
    for key, parts in frame_parts.items():
        merged_motion[key] = torch.cat(parts, dim=0)
    for key, parts in motion_parts.items():
        merged_motion[key] = torch.cat(parts, dim=0)
    for key, parts in optional_motion_parts.items():
        if parts:
            if len(parts) != len(by_recording):
                raise ValueError(f"Optional motion field {key} is only partially present")
            merged_motion[key] = torch.cat(parts, dim=0)
    counts = merged_motion["motion_num_frames"].long()
    starts = torch.zeros_like(counts)
    if counts.numel() > 1:
        starts[1:] = counts[:-1].cumsum(dim=0)
    merged_motion["length_starts"] = starts
    merged_motion["motion_files"] = tuple(motion_files)

    expected_frames = sum(int(item["count"]) for item in clips)
    if int(counts.sum()) != expected_frames:
        raise ValueError(
            f"Merged {split} frame count {int(counts.sum())} != {expected_frames}"
        )
    if not (len(clips) == len(scenes) == len(camera_motions) == counts.numel()):
        raise ValueError(f"Unaligned {split} online pack")

    split_root = output_root / split
    motion_path = split_root / motion_filename
    scene_path = split_root / "scene_lib_training_isaaclab.pt"
    camera_path = split_root / camera_filename
    _atomic_torch_save(merged_motion, motion_path)
    _atomic_torch_save(
        {
            "original_scenes": scenes,
            "num_original_scenes": len(scenes),
            "num_objects_per_scene": num_objects_per_scene,
        },
        scene_path,
    )
    _atomic_torch_save(
        {
            "recording": f"egobody_online_{split}",
            "coordinate_system": "motion_world",
            "source": str(Path(manifest["prepared_root"])),
            "motions": camera_motions,
            "clip_map": clip_map,
            "grounded_motion_file": str(motion_path),
        },
        camera_path,
    )
    split_manifest = {
        "split": split,
        "num_motions": len(clips),
        "num_frames": int(counts.sum()),
        "motion_file": str(motion_path),
        "scene_file": str(scene_path),
        "ego_camera_file": str(camera_path),
        "clip_map": clip_map,
    }
    (split_root / "manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--motion-filename",
        default="motion_lib_soma23_grounded.pt",
        help="Per-recording packaged MotionLib filename.",
    )
    parser.add_argument(
        "--camera-filename",
        default="ego_camera_grounded.pt",
        help="Per-recording camera trajectory filename.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test")
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summaries = [
        _merge_split(
            manifest,
            split,
            args.output_root,
            args.motion_filename,
            args.camera_filename,
        )
        for split in args.splits
    ]
    (args.output_root / "manifest.json").write_text(
        json.dumps({"format_version": 1, "splits": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Built EgoBody online SFT packs: "
        + ", ".join(
            f"{item['split']}={item['num_motions']} motions/{item['num_frames']} frames"
            for item in summaries
        )
    )


if __name__ == "__main__":
    main()
