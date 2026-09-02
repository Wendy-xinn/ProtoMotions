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
import math
import pickle
import re
from collections import Counter
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
    parser.add_argument(
        "--body-text-root",
        type=Path,
        default=None,
        help="Optional frame-level EgoBody body-description root.",
    )
    parser.add_argument(
        "--minimum-motion-score",
        type=float,
        default=0.0,
        help="Reject low-motion candidate windows before scalable selection.",
    )
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
    parser.add_argument(
        "--exclude-window",
        action="append",
        default=[],
        metavar="RECORDING:START",
        help="Exclude a known-bad recording window. May be specified multiple times.",
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
) -> dict[str, float] | None:
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
    return {
        "motion_score": float(2.0 * root_path + root_span + 0.2 * pose_motion),
        "root_travel_m": float(root_path),
        "root_span_m": float(root_span),
        "mean_sampled_pose_delta": float(pose_motion),
    }


def load_body_text_timeline(
    body_text_root: Path, recording: str, info: dict[str, str]
) -> tuple[str, list[str], bool]:
    body_index = int(info["body_idx_fpv"].split()[0])
    body_name = f"body_idx_{body_index}"
    text_dir = body_text_root / recording / body_name
    timeline = []
    for segment_index, path in enumerate(sorted(text_dir.glob("*.json"))):
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptions = [payload[str(index)] for index in range(len(payload))]
        if segment_index > 0:
            descriptions = descriptions[1:]
        timeline.extend(descriptions)
    expected = int(info["end_frame"]) - int(info["start_frame"]) + 1
    if not timeline:
        raise ValueError(f"No body text found for {recording}/{body_name}")
    return body_name, timeline, len(timeline) == expected


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def describe_text_window(descriptions: list[str]) -> dict:
    sampled = descriptions[::8]
    if (len(descriptions) - 1) % 8 != 0:
        sampled.append(descriptions[-1])
    lowered = [text.lower() for text in sampled]
    patterns = {
        "arms_raised": r"raised|higher than (?:his |her |their )?.*shoulder|above (?:his |her |their )?.*shoulder",
        "horizontal_limb": r"horizontal|parallel to (?:the )?(?:ground|floor)",
        "torso_lean": r"\blean(?:s|ing|ed)?\b|torso is (?:almost )?horizontal",
        "deep_knee_bend": r"knees?.{0,80}(?:rather|fully|deeply|almost completely) bent",
        "hands_close": r"hands?.{0,60}(?:joined|close|near|together)",
        "asymmetric_arms": r"(?:behind|in front of|ahead of).{0,80}(?:elbow|arm|hand)",
    }
    fractions = {
        name: sum(bool(re.search(pattern, text)) for text in lowered) / len(lowered)
        for name, pattern in patterns.items()
    }
    tags = sorted(name for name, fraction in fractions.items() if fraction >= 0.1)
    changes = []
    token_sets = [_token_set(text) for text in sampled]
    for left, right in zip(token_sets, token_sets[1:]):
        union = left | right
        changes.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    representative_indices = sorted({0, len(descriptions) // 2, len(descriptions) - 1})
    return {
        "text_tags": tags,
        "text_tag_fractions": {key: round(value, 4) for key, value in fractions.items()},
        "text_change_score": round(float(np.mean(changes)) if changes else 0.0, 4),
        "representative_descriptions": [descriptions[index] for index in representative_indices],
    }


def attach_text_features(
    candidates: list[dict],
    body_text_root: Path,
    recording: str,
    info: dict[str, str],
) -> None:
    body_name, timeline, exact_alignment = load_body_text_timeline(
        body_text_root, recording, info
    )
    first_frame = int(info["start_frame"])
    for candidate in candidates:
        local_start = candidate["start"] - first_frame
        local_end = local_start + candidate["count"]
        descriptions = (
            timeline[local_start:local_end] if exact_alignment else timeline
        )
        if exact_alignment and len(descriptions) != candidate["count"]:
            raise ValueError(
                f"Incomplete body text for {recording} at {candidate['start']}"
            )
        candidate["body_text_source"] = f"EgoBody/{recording}/{body_name}"
        candidate["body_text_alignment"] = (
            "exact_frame_window" if exact_alignment else "recording_level_only"
        )
        candidate.update(describe_text_window(descriptions))
        dynamic_tags = []
        if candidate["root_span_m"] >= 0.75:
            dynamic_tags.append("locomotion")
        elif candidate["root_span_m"] >= 0.25:
            dynamic_tags.append("short_translation")
        else:
            dynamic_tags.append("in_place_motion")
        candidate["motion_tags"] = dynamic_tags
        candidate["selection_tags"] = sorted(
            set(candidate["text_tags"] + dynamic_tags)
        )
        text_tags = set(candidate["text_tags"])
        upper_body = bool(
            text_tags & {"arms_raised", "horizontal_limb", "hands_close"}
        )
        motion_tag = dynamic_tags[0]
        if motion_tag == "locomotion":
            action_type = (
                "locomotion_with_upper_body" if upper_body else "locomotion"
            )
        elif motion_tag == "short_translation":
            action_type = (
                "short_translation_with_upper_body"
                if upper_body
                else "short_translation"
            )
        elif "deep_knee_bend" in text_tags:
            action_type = "in_place_lower_body"
        elif "torso_lean" in text_tags:
            action_type = "in_place_leaning"
        elif upper_body:
            action_type = "in_place_upper_body"
        else:
            action_type = "in_place_general"
        candidate["action_type"] = action_type
        candidate["pose_motion_signature"] = "+".join(
            [motion_tag, *sorted(text_tags)]
        )


def assign_diversity_scores(candidates_by_recording: dict[str, list[dict]]) -> None:
    candidates = [item for items in candidates_by_recording.values() for item in items]
    tag_counts = Counter(
        tag for candidate in candidates for tag in candidate.get("selection_tags", [])
    )
    action_counts = Counter(
        candidate.get("action_type", "unannotated") for candidate in candidates
    )
    total = max(len(candidates), 1)
    for candidate in candidates:
        rarity = sum(
            math.log1p(total / tag_counts[tag])
            for tag in candidate.get("selection_tags", [])
        )
        action_type = candidate.get("action_type", "unannotated")
        action_rarity = math.log1p(total / action_counts[action_type])
        candidate["diversity_score"] = (
            candidate["motion_score"] + 0.15 * rarity + 0.25 * action_rarity
        )


def write_text_selection_records(output: Path, clips: list[dict]) -> None:
    annotated = [clip for clip in clips if "body_text_source" in clip]
    if not annotated:
        return
    jsonl_path = output.with_name("clip_text_records.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for clip in annotated:
            record = {
                "dataset_source": "EgoBody",
                "clip_id": clip["clip_id"],
                "split": clip["split"],
                "recording": clip["recording"],
                "source_frame_start": clip["start"],
                "source_frame_end": clip["end"],
                "frame_count": clip["count"],
                "body_text_source": clip["body_text_source"],
                "body_text_alignment": clip["body_text_alignment"],
                "motion_score": clip["motion_score"],
                "root_travel_m": clip["root_travel_m"],
                "root_span_m": clip["root_span_m"],
                "mean_sampled_pose_delta": clip["mean_sampled_pose_delta"],
                "motion_tags": clip["motion_tags"],
                "action_type": clip["action_type"],
                "pose_motion_signature": clip["pose_motion_signature"],
                "text_tags": clip["text_tags"],
                "text_change_score": clip["text_change_score"],
                "representative_descriptions": clip["representative_descriptions"],
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    tag_counts = Counter(
        tag for clip in annotated for tag in clip.get("selection_tags", [])
    )
    scores = np.asarray([clip["motion_score"] for clip in annotated])
    split_counts = Counter(clip["split"] for clip in annotated)
    recording_counts = {
        split: len({clip["recording"] for clip in annotated if clip["split"] == split})
        for split in ("train", "val", "test")
    }
    alignment_counts = Counter(clip["body_text_alignment"] for clip in annotated)
    action_counts = Counter(clip["action_type"] for clip in annotated)
    signature_counts = Counter(
        clip["pose_motion_signature"] for clip in annotated
    )
    report_path = output.with_name("SELECTION_REPORT.md")
    lines = [
        "# EgoBody Diverse Motion Selection",
        "",
        "This selection is sourced from the **EgoBody dataset**. Body descriptions",
        "come from the aligned `texts/body_texts/EgoBody` frame annotations. It is",
        "not sourced from EgoExo4D or AMASS.",
        "",
        "## Dataset",
        "",
        f"- Clips: {len(annotated)}",
        f"- Frames per clip: {annotated[0]['count']}",
        f"- Split clips: {dict(split_counts)}",
        f"- Split recordings: {recording_counts}",
        f"- Body-text alignment: {dict(alignment_counts)}",
        f"- Coarse action types: {len(action_counts)}",
        f"- Pose-motion signatures: {len(signature_counts)}",
        f"- Motion-score range: {scores.min():.4f}--{scores.max():.4f}",
        f"- Motion-score median: {np.median(scores):.4f}",
        "- Split isolation: official EgoBody recording-level train/val/test split",
        "",
        "Low-motion windows below the configured threshold were rejected; a whole",
        "recording is never included merely because that recording was eligible.",
        "Rare body-text posture tags receive a diversity bonus during ranking.",
        "Descriptions with `recording_level_only` alignment are used only for",
        "recording-level diversity; they are not claimed as exact clip captions.",
        "",
        "## Selected Tags",
        "",
        "| tag | clips |",
        "|---|---:|",
    ]
    lines.extend(f"| {tag} | {count} |" for tag, count in tag_counts.most_common())
    lines.extend(
        [
            "",
            "## Coarse Action Types",
            "",
            "These are pose-motion categories inferred from displacement and body",
            "descriptions, not object-action labels supplied by EgoBody.",
            "",
            "| action type | clips |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {action_type} | {count} |"
        for action_type, count in action_counts.most_common()
    )
    lines.extend(
        [
            "",
            "## Records",
            "",
            "Per-clip source frames, motion statistics, tags, and representative",
            f"frame descriptions are stored in `{jsonl_path.name}`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


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
        metrics = score_window(frame_root, start, frames, score_stride)
        if metrics is None:
            continue
        candidates.append(
            {
                "recording": recording,
                "split": split,
                "scene": scene_name,
                "start": start,
                "count": frames,
                "end": end,
                "observed_camera_frames": int(covered.size),
                **metrics,
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
        candidates,
        key=lambda item: item.get("diversity_score", item["motion_score"]),
        reverse=True,
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
                        (
                            -candidate.get("diversity_score", candidate["motion_score"]),
                            recording,
                            candidate,
                        ),
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
    excluded_windows = set()
    for value in args.exclude_window:
        try:
            recording, start_text = value.rsplit(":", 1)
            excluded_windows.add((recording, int(start_text)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --exclude-window {value!r}; expected RECORDING:START"
            ) from exc
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
                candidates = [
                    item
                    for item in candidates
                    if item["motion_score"] >= args.minimum_motion_score
                    and (item["recording"], item["start"]) not in excluded_windows
                ]
                if args.body_text_root is not None and candidates:
                    attach_text_features(
                        candidates,
                        args.body_text_root.resolve(),
                        recording,
                        infos[recording],
                    )
                if candidates:
                    candidates_by_recording[recording] = candidates
            assign_diversity_scores(candidates_by_recording)
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
                candidates = [
                    item
                    for item in candidates
                    if (item["recording"], item["start"]) not in excluded_windows
                ]
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
        "dataset_source": "EgoBody",
        "body_text_root": (
            str(args.body_text_root.resolve()) if args.body_text_root is not None else None
        ),
        "minimum_motion_score": args.minimum_motion_score,
        "excluded_windows": [
            {"recording": recording, "start": start}
            for recording, start in sorted(excluded_windows)
        ],
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
    write_text_selection_records(args.output, clips)
    print(f"Wrote {len(clips)} clips to {args.output}")
    for split in targets:
        split_clips = [item for item in clips if item["split"] == split]
        print(f"{split}: {len(split_clips)} clips, {len(chosen_recordings[split])} recordings")


if __name__ == "__main__":
    main()
