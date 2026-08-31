# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a clip-level manifest for scene-aware TRUMANS motion tracking.

The released frame arrays concatenate original recordings and augmented clips.  Object
trajectories, however, are released only for the original recordings.  This script makes
that distinction explicit so a scene-aware expert cannot silently pair an augmented human
motion with an untransformed object trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _segment_ranges(segment_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = np.flatnonzero(segment_names[1:] != segment_names[:-1]) + 1
    return np.r_[0, changes], np.r_[changes, len(segment_names)]


def _asset_path(root: Path, relative: str) -> str | None:
    path = root / relative
    return relative if path.is_file() else None


def _object_track_summary(path: Path, expected_frames: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": None,
            "objects": [],
            "track_lengths": {},
            "length_matches_clip": False,
        }

    # Object_pose files are trusted files from the official TRUMANS release and contain
    # a pickled dict of {object_name: {rotation: ..., location: ...}}.
    tracks = np.load(path, allow_pickle=True).item()
    lengths: dict[str, int] = {}
    for object_name, track in tracks.items():
        rotation = np.asarray(track["rotation"])
        location = np.asarray(track["location"])
        if rotation.shape != location.shape or rotation.ndim != 2 or rotation.shape[1] != 3:
            raise ValueError(
                f"Malformed object track {path.name}:{object_name}: "
                f"rotation={rotation.shape}, location={location.shape}"
            )
        lengths[object_name] = int(rotation.shape[0])

    return {
        "path": str(path.relative_to(path.parents[2])),
        "objects": sorted(tracks),
        "track_lengths": lengths,
        "length_matches_clip": bool(lengths)
        and all(length == expected_frames for length in lengths.values()),
    }


def build_manifest(root: Path, include_augmented: bool) -> tuple[list[dict], dict]:
    segment_names = np.load(root / "seg_name.npy", mmap_mode="r")
    frame_ids = np.load(root / "frame_id.npy", mmap_mode="r")
    scene_flags = np.load(root / "scene_flag.npy", mmap_mode="r")
    scene_names = np.load(root / "scene_list.npy")
    object_flags = np.load(root / "object_flag.npy", mmap_mode="r")
    object_names = np.load(root / "object_list.npy")
    bad_frames = set(map(int, np.load(root / "bad_frames.npy").tolist()))

    starts, ends = _segment_ranges(segment_names)
    records: list[dict] = []
    skipped_augmented = 0

    for start, end in zip(starts, ends):
        clip_name = str(segment_names[start])
        is_augmented = "_augment" in clip_name
        if is_augmented and not include_augmented:
            skipped_augmented += 1
            continue

        frame_count = int(end - start)
        scene_index = int(scene_flags[start])
        if not np.all(scene_flags[start:end] == scene_index):
            raise ValueError(f"Clip {clip_name} spans more than one scene")

        scene_name = str(scene_names[scene_index])
        active_object_columns = np.flatnonzero(
            np.any(object_flags[start:end] >= 1, axis=0)
        )
        active_chair_assets = [str(object_names[i]) for i in active_object_columns]

        base_name = clip_name.split("_augment", maxsplit=1)[0]
        object_pose_path = root / "Object_all" / "Object_pose" / f"{base_name}.npy"
        object_tracks = _object_track_summary(object_pose_path, frame_count)

        bad_in_clip = sorted(i - int(start) for i in bad_frames if start <= i < end)
        records.append(
            {
                "clip_name": clip_name,
                "base_clip_name": base_name,
                "is_augmented": is_augmented,
                "global_start": int(start),
                "global_end_exclusive": int(end),
                "num_frames": frame_count,
                "source_frame_start": int(frame_ids[start]),
                "source_frame_end": int(frame_ids[end - 1]),
                "fps": 30,
                "coordinate_system": "trumans_y_up",
                "scene": {
                    "index": scene_index,
                    "name": scene_name,
                    "occupancy": _asset_path(root, f"Scene/{scene_name}.npy"),
                    "mesh": _asset_path(root, f"Scene_mesh/{scene_name}.obj"),
                },
                "object_tracks": object_tracks,
                "active_chair_assets": active_chair_assets,
                "smplx_global": _asset_path(
                    root, f"smplx_result/{base_name}_smplx_results.pkl"
                ),
                "action_text": _asset_path(root, f"Actions/{base_name}.txt"),
                "bad_local_frames": bad_in_clip,
                "eligible_scene_expert_v1": (
                    not is_augmented
                    and object_tracks["path"] is not None
                    and object_tracks["length_matches_clip"]
                ),
            }
        )

    eligible = sum(record["eligible_scene_expert_v1"] for record in records)
    with_tracks = sum(record["object_tracks"]["path"] is not None for record in records)
    mismatched_tracks = sum(
        record["object_tracks"]["path"] is not None
        and not record["object_tracks"]["length_matches_clip"]
        for record in records
    )
    summary = {
        "dataset_root": str(root.resolve()),
        "total_frames": int(len(segment_names)),
        "total_segments_in_release": int(len(starts)),
        "written_segments": len(records),
        "skipped_augmented_segments": skipped_augmented,
        "segments_with_object_tracks": with_tracks,
        "segments_with_mismatched_object_track_length": mismatched_tracks,
        "eligible_scene_expert_v1": eligible,
        "scene_count": int(len(scene_names)),
        "policy": {
            "default_subset": "original recordings only",
            "reason": (
                "Augmented human clips require the identical augmentation transform to be "
                "applied to scene/object trajectories before scene-aware training."
            ),
        },
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted TRUMANS root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/processed/scene_expert_v1)",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Write augmented clips too, but mark them ineligible until transforms are applied",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "processed" / "scene_expert_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records, summary = build_manifest(root, args.include_augmented)
    manifest_path = output_dir / "clips.jsonl"
    summary_path = output_dir / "summary.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, default=_json_value, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, default=_json_value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
