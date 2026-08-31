#!/usr/bin/env python3
"""Select motion-rich, camera-covered EgoBody clips for offline GPC SFT.

Selection is deterministic and split by recording, never by windows from the
same recording.  This prevents a static reconstruction or subject trajectory
from leaking between train and validation/test sets.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import inspect
import json
import pickle
from pathlib import Path

import numpy as np


if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--egobody-root", type=Path, default=Path("/home/wenxin/projects/egobody")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=192)
    parser.add_argument("--candidate-stride", type=int, default=48)
    parser.add_argument("--score-stride", type=int, default=8)
    parser.add_argument("--train-recordings", type=int, default=10)
    parser.add_argument("--clips-per-train-recording", type=int, default=4)
    parser.add_argument("--val-recordings", type=int, default=5)
    parser.add_argument("--test-recordings", type=int, default=5)
    parser.add_argument(
        "--split-clip-counts",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help=(
            "Select exact train/val/test clip counts from all eligible recordings. "
            "This scalable mode supersedes the recording-count options."
        ),
    )
    parser.add_argument(
        "--min-start-gap",
        type=int,
        default=96,
        help="Minimum start-frame separation within a recording in scalable mode.",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def split_members(root: Path) -> dict[str, list[str]]:
    result = {"train": [], "val": [], "test": []}
    with (root / "data_splits.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for split in result:
                value = row.get(split, "").strip()
                if value:
                    result[split].append(value)
    return result


def recording_info(root: Path) -> dict[str, dict[str, str]]:
    with (root / "data_info_release.csv").open(newline="", encoding="utf-8") as handle:
        return {row["recording_name"]: row for row in csv.DictReader(handle)}


def wearer_frame_root(root: Path, split: str, recording: str) -> Path | None:
    base = root / f"smpl_camera_wearer_{split}" / recording
    bodies = sorted(base.glob("body_idx_*"))
    return bodies[0] / "results" if len(bodies) == 1 else None


def camera_frame_ids(root: Path, recording: str) -> np.ndarray:
    image_root = root / "egocentric_color" / recording
    ids = []
    for path in image_root.glob("202*/PV/*_frame_*.jpg"):
        try:
            ids.append(int(path.stem.rsplit("_frame_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def load_sparse_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    translation = np.asarray(data["transl"], dtype=np.float64).reshape(3)
    pose = np.concatenate(
        [
            np.asarray(data["global_orient"], dtype=np.float64).reshape(3),
            np.asarray(data["body_pose"], dtype=np.float64).reshape(-1),
        ]
    )
    return translation, pose


def score_window(
    frame_root: Path, start: int, frames: int, score_stride: int
) -> float | None:
    sample_ids = list(range(start, start + frames, score_stride))
    if sample_ids[-1] != start + frames - 1:
        sample_ids.append(start + frames - 1)
    translations, poses = [], []
    for frame_id in sample_ids:
        path = frame_root / f"frame_{frame_id:05d}" / "000.pkl"
        if not path.is_file():
            return None
        translation, pose = load_sparse_pose(path)
        translations.append(translation)
        poses.append(pose)
    translations = np.stack(translations)
    poses = np.stack(poses)
    root_path = np.linalg.norm(np.diff(translations, axis=0), axis=-1).sum()
    root_span = np.linalg.norm(translations.max(0) - translations.min(0))
    pose_motion = np.linalg.norm(np.diff(poses, axis=0), axis=-1).mean()
    # Root travel favors walking; pose motion retains seated/interaction clips.
    return float(2.0 * root_path + root_span + 0.2 * pose_motion)


def candidate_windows(
    root: Path,
    split: str,
    recording: str,
    info: dict[str, str],
    frames: int,
    candidate_stride: int,
    score_stride: int,
) -> list[dict]:
    frame_root = wearer_frame_root(root, split, recording)
    camera_ids = camera_frame_ids(root, recording)
    scene_name = info["scene_name"]
    required = [
        frame_root,
        root / "scene_mesh" / scene_name / f"{scene_name}.obj",
        root / "calibrations" / recording / "cal_trans" / "holo_to_kinect12.json",
        root
        / "calibrations"
        / recording
        / "cal_trans"
        / "kinect12_to_world"
        / f"{scene_name}.json",
    ]
    if frame_root is None or camera_ids.size == 0 or any(
        path is None or not Path(path).exists() for path in required
    ):
        return []
    first = max(int(info["start_frame"]), int(camera_ids[0]))
    last = min(int(info["end_frame"]), int(camera_ids[-1]))
    candidates = []
    for start in range(first, last - frames + 2, candidate_stride):
        end = start + frames - 1
        covered = camera_ids[(camera_ids >= start) & (camera_ids <= end)]
        if covered.size < int(frames * 0.85):
            continue
        padded = np.concatenate(([start], covered, [end]))
        if int(np.diff(padded).max(initial=0)) > 8:
            continue
        score = score_window(frame_root, start, frames, score_stride)
        if score is None:
            continue
        candidates.append(
            {
                "recording": recording,
                "split": split,
                "scene": scene_name,
                "start": start,
                "count": frames,
                "end": end,
                "motion_score": score,
                "observed_camera_frames": int(covered.size),
            }
        )
    return candidates


def nonoverlapping_top(candidates: list[dict], count: int) -> list[dict]:
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["motion_score"], reverse=True):
        if all(
            candidate["end"] < prior["start"] or candidate["start"] > prior["end"]
            for prior in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda item: item["start"])


def spaced_top(candidates: list[dict], min_start_gap: int) -> list[dict]:
    """Keep high-motion windows while limiting near-duplicate temporal crops."""
    selected = []
    for candidate in sorted(
        candidates, key=lambda item: item["motion_score"], reverse=True
    ):
        if all(
            abs(candidate["start"] - prior["start"]) >= min_start_gap
            for prior in selected
        ):
            selected.append(candidate)
    return selected


def balanced_top(
    candidates_by_recording: dict[str, list[dict]],
    count: int,
    min_start_gap: int,
) -> list[dict]:
    """Select exactly ``count`` clips while spreading them across recordings."""
    queues = {
        recording: spaced_top(candidates, min_start_gap)
        for recording, candidates in candidates_by_recording.items()
    }
    queues = {recording: items for recording, items in queues.items() if items}
    selected = []
    depth = 0
    while len(selected) < count:
        round_candidates = []
        for recording, items in queues.items():
            if depth < len(items):
                candidate = items[depth]
                heapq.heappush(
                    round_candidates,
                    (-candidate["motion_score"], recording, candidate),
                )
        if not round_candidates:
            available = sum(len(items) for items in queues.values())
            raise RuntimeError(
                f"Only {available} spaced clips are available; requested {count}. "
                "Reduce --min-start-gap or the requested split count."
            )
        while round_candidates and len(selected) < count:
            _, _, candidate = heapq.heappop(round_candidates)
            selected.append(candidate)
        depth += 1
    return sorted(selected, key=lambda item: (item["recording"], item["start"]))


def main() -> None:
    args = parse_args()
    root = args.egobody_root.resolve()
    infos = recording_info(root)
    members = split_members(root)
    rng = np.random.default_rng(args.seed)
    targets = {
        "train": (args.train_recordings, args.clips_per_train_recording),
        "val": (args.val_recordings, 1),
        "test": (args.test_recordings, 1),
    }
    clips = []
    chosen_recordings = {split: [] for split in targets}
    if args.split_clip_counts is not None:
        requested = dict(zip(("train", "val", "test"), args.split_clip_counts))
        for split, count in requested.items():
            if count < 0:
                raise ValueError("Split clip counts must be non-negative")
            candidates_by_recording = {}
            for recording in members[split]:
                candidates = candidate_windows(
                    root,
                    split,
                    recording,
                    infos[recording],
                    args.frames,
                    args.candidate_stride,
                    args.score_stride,
                )
                if candidates:
                    candidates_by_recording[recording] = candidates
            split_clips = balanced_top(
                candidates_by_recording, count, args.min_start_gap
            )
            clips.extend(split_clips)
            chosen_recordings[split] = sorted(
                {item["recording"] for item in split_clips}
            )
    else:
        for split, (recording_count, clips_per_recording) in targets.items():
            pool = list(members[split])
            rng.shuffle(pool)
            ranked = []
            for recording in pool:
                candidates = candidate_windows(
                    root,
                    split,
                    recording,
                    infos[recording],
                    args.frames,
                    args.candidate_stride,
                    args.score_stride,
                )
                selected = nonoverlapping_top(candidates, clips_per_recording)
                if len(selected) == clips_per_recording:
                    ranked.append(
                        (
                            sum(item["motion_score"] for item in selected),
                            recording,
                            selected,
                        )
                    )
            ranked.sort(reverse=True)
            if len(ranked) < recording_count:
                raise RuntimeError(
                    f"Only {len(ranked)} eligible {split} recordings; "
                    f"need {recording_count}"
                )
            for _, recording, selected in ranked[:recording_count]:
                chosen_recordings[split].append(recording)
                clips.extend(selected)

    for index, clip in enumerate(clips):
        clip["clip_id"] = index
    manifest = {
        "format_version": 1,
        "egobody_root": str(root),
        "frame_count": args.frames,
        "selection_seed": args.seed,
        "split_clip_counts": (
            dict(zip(("train", "val", "test"), args.split_clip_counts))
            if args.split_clip_counts is not None
            else None
        ),
        "min_start_gap": (
            args.min_start_gap if args.split_clip_counts is not None else args.frames
        ),
        "recordings": chosen_recordings,
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(clips)} clips to {args.output}")
    for split in targets:
        split_clips = [item for item in clips if item["split"] == split]
        print(f"{split}: {len(split_clips)} clips, {len(chosen_recordings[split])} recordings")


if __name__ == "__main__":
    main()
