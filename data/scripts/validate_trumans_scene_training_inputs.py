#!/usr/bin/env python3
"""Validate TRUMANS scene-aware training inputs before launching PPO.

This is intentionally read-only.  It catches the common expensive mistakes:

* a split descriptor exists but per-clip motion/object/contact files are missing;
* a scene appears in the split but its collision OBJ/USD/pointcloud/report is missing;
* the final training SceneLib is still the intermediate room-mesh pack;
* contacts were generated but not injected/repackaged into the MotionLib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/motion_for_trackers/trumans_scene_collision_v1"


def _progress(iterable, **kwargs):
    return tqdm(iterable, dynamic_ncols=True, disable=not sys.stderr.isatty(), **kwargs)


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_asset(path: str | Path | None, asset_root: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    if asset_root is not None:
        return (asset_root / resolved).resolve()
    return resolved.resolve()


def _require_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")


def _load_scene_count(scene_file: Path) -> tuple[int, list[dict]]:
    data = torch.load(scene_file, map_location="cpu", weights_only=False)
    scenes = data.get("original_scenes") if isinstance(data, dict) else data
    if scenes is None:
        raise ValueError(f"{scene_file} does not contain original_scenes")
    return len(scenes), scenes


def _scene_object_path(scene: dict, object_index: int) -> str | None:
    objects = scene.get("objects", [])
    if object_index >= len(objects):
        return None
    return objects[object_index].get("object_path")


def _contact_union(path: Path) -> np.ndarray:
    with np.load(path) as contact:
        if "training_contact" not in contact:
            raise ValueError(f"{path} has no training_contact array")
        training = contact["training_contact"]
        if training.ndim != 3:
            raise ValueError(
                f"{path} training_contact should be [T,B,O], got {training.shape}"
            )
        return training.any(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--motion-file", type=Path, default=None)
    parser.add_argument("--scene-file", type=Path, default=None)
    parser.add_argument(
        "--scene-asset-root",
        type=Path,
        default=None,
        help="Root for relative mesh paths stored in scene files, normally ../TRUMANS.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--skip-contact-injection-check",
        action="store_true",
        help="Only check contact files exist; do not compare them against .motion files.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=30,
        help="Print at most this many detailed missing-file errors.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split = args.split
    motion_file = (
        args.motion_file.expanduser().resolve()
        if args.motion_file is not None
        else data_root / "motion_libs" / f"{split}.pt"
    )
    scene_file = (
        args.scene_file.expanduser().resolve()
        if args.scene_file is not None
        else data_root / "scene_libs" / f"{split}.pt"
    )
    scene_asset_root = (
        args.scene_asset_root.expanduser().resolve()
        if args.scene_asset_root is not None
        else None
    )

    errors: list[str] = []
    descriptor_path = data_root / "descriptors" / f"{split}.jsonl"
    _require_file(descriptor_path, "descriptor", errors)
    _require_file(motion_file, "MotionLib", errors)
    _require_file(scene_file, "SceneLib", errors)
    if args.checkpoint is not None:
        _require_file(args.checkpoint.expanduser().resolve(), "warm-start checkpoint", errors)

    if errors:
        raise SystemExit("\n".join(errors[: args.max_errors]))

    records = _records(descriptor_path)
    if not records:
        raise SystemExit(f"{descriptor_path} contains no clips")

    scene_ids = sorted({record["scene_id"] for record in records})
    print(
        f"TRUMANS {split}: {len(records)} clips, {len(scene_ids)} scenes; "
        f"checking {data_root}"
    )

    for record in records:
        _require_file(Path(record["motion_file"]), f"motion for {record['clip_id']}", errors)
        _require_file(Path(record["object_cache"]), f"object cache for {record['clip_id']}", errors)
        _require_file(
            data_root / "mesh_contacts" / split / f"{record['clip_id']}.npz",
            f"mesh contact for {record['clip_id']}",
            errors,
        )

    for scene_id in scene_ids:
        stem = data_root / "collision_meshes" / split / scene_id
        for suffix, label in (
            (".obj", "collision OBJ"),
            (".usda", "collision USD"),
            (".pointcloud.npz", "collision pointcloud"),
            (".json", "collision report"),
        ):
            _require_file(stem.with_suffix(suffix), f"{label} for scene {scene_id}", errors)

    if errors:
        print("Training inputs are not complete yet:")
        for error in errors[: args.max_errors]:
            print(f"  - {error}")
        if len(errors) > args.max_errors:
            print(f"  ... {len(errors) - args.max_errors} more")
        raise SystemExit(2)

    try:
        scene_count, scenes = _load_scene_count(scene_file)
    except Exception as error:  # noqa: BLE001 - turn loader failures into clear preflight errors
        raise SystemExit(f"failed to load SceneLib {scene_file}: {error}") from error
    if scene_count != len(records):
        raise SystemExit(
            f"SceneLib/descriptor mismatch: {scene_count} scenes vs {len(records)} descriptors"
        )

    for index, scene in enumerate(scenes):
        motion_id = scene.get("humanoid_motion_id") if isinstance(scene, dict) else None
        if motion_id != index:
            raise SystemExit(
                f"SceneLib humanoid_motion_id mismatch at scene {index}: got {motion_id}"
            )
        static_path = _resolve_asset(_scene_object_path(scene, 0), scene_asset_root)
        if static_path is None or not static_path.is_file():
            raise SystemExit(
                f"SceneLib static collision asset missing at scene {index}: {static_path}"
            )
        if "collision_meshes" not in str(static_path):
            raise SystemExit(
                f"SceneLib {scene_file} does not look like the final collision pack; "
                f"scene {index} first object is {static_path}"
            )

    try:
        motion_data = torch.load(motion_file, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"failed to load MotionLib {motion_file}: {error}") from error
    motion_count = int(motion_data["motion_num_frames"].shape[0])
    if motion_count != len(records):
        raise SystemExit(
            f"MotionLib/descriptor mismatch: {motion_count} motions vs {len(records)} descriptors"
        )
    contacts = motion_data.get("contacts")
    if contacts is None:
        raise SystemExit(f"MotionLib {motion_file} has no contacts tensor")
    if contacts.ndim != 2 or contacts.shape[1] != 24:
        raise SystemExit(f"MotionLib contacts should be [frames,24], got {tuple(contacts.shape)}")
    if int(contacts.to(torch.int64).sum().item()) <= 0:
        raise SystemExit(f"MotionLib contacts are all zero in {motion_file}")

    if not args.skip_contact_injection_check:
        expected_frames = 0
        expected_contacts = 0
        for record in _progress(records, desc=f"contact injection[{split}]", unit="clip"):
            clip_id = record["clip_id"]
            contact_path = data_root / "mesh_contacts" / split / f"{clip_id}.npz"
            expected = _contact_union(contact_path)
            motion = torch.load(record["motion_file"], map_location="cpu", weights_only=False)
            actual = motion.get("rigid_body_contacts")
            if actual is None:
                raise SystemExit(f"{record['motion_file']} has no rigid_body_contacts")
            actual_np = actual.cpu().numpy().astype(bool)
            if actual_np.shape != expected.shape:
                raise SystemExit(
                    f"contact shape mismatch for {clip_id}: motion {actual_np.shape}, "
                    f"mesh_contacts {expected.shape}"
                )
            if not np.array_equal(actual_np, expected):
                raise SystemExit(
                    f"contact injection mismatch for {clip_id}; rerun stage 4/5"
                )
            expected_frames += int(expected.shape[0])
            expected_contacts += int(expected.sum())
        if expected_frames != int(contacts.shape[0]):
            raise SystemExit(
                f"packed MotionLib frame count mismatch: {contacts.shape[0]} vs "
                f"{expected_frames} from per-clip motions"
            )
        if expected_contacts != int(contacts.to(torch.int64).sum().item()):
            raise SystemExit(
                "packed MotionLib contact count differs from injected .motion files; "
                "rerun stage 5 motion_pack"
            )

    print(
        "TRUMANS scene training inputs OK: "
        f"{len(records)} motions, {len(scene_ids)} collision scenes, "
        f"{contacts.shape[0]} frames, {int(contacts.to(torch.int64).sum().item())} contacts"
    )


if __name__ == "__main__":
    main()
