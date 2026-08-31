#!/usr/bin/env python3
"""Generate reviewable contact labels from final collision meshes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.scene_motion.contact_labels import (
    generate_collision_mesh_contact_labels,
    inject_motion_contact_union,
    save_contact_labels,
)
from data.scripts.scene_motion.transforms import TRUMANS_Y_UP_TO_Z_UP, transform_points
from data.scripts.scene_motion.trumans_adapter import TrumansAdapter
from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _progress(iterable, **kwargs):
    return tqdm(iterable, dynamic_ncols=True, disable=not sys.stderr.isatty(), **kwargs)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != "schema_version"}


def _source_joints(adapter: TrumansAdapter, clip) -> np.ndarray:
    human = adapter.load_human(clip)
    if human.joint_positions is None or human.joint_names is None:
        raise ValueError(f"{clip.clip_id}: released joints are required")
    source_index = {name: index for index, name in enumerate(human.joint_names)}
    source = transform_points(human.joint_positions, TRUMANS_Y_UP_TO_Z_UP)
    return source[:, [source_index[name] for name in SMPL_MUJOCO_NAMES]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "data/yaml_files/trumans_scene_motion.yaml",
    )
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clip-id", action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inject-into-motion",
        action="store_true",
        help="Replace MotionLib contact supervision only after Viser/PhysX validation.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else _resolve(config_dir, config["output_root"])
    )
    records = _records(output_root / "descriptors" / f"{args.split}.jsonl")
    if args.clip_id:
        selected = set(args.clip_id)
        records = [record for record in records if record["clip_id"] in selected]
        missing = selected - {record["clip_id"] for record in records}
        if missing:
            raise ValueError(f"Unknown clip IDs in {args.split}: {sorted(missing)}")
    if args.limit is not None:
        records = records[: args.limit]

    adapter = TrumansAdapter(
        root=_resolve(config_dir, config["dataset"]["root"]),
        manifest_path=_resolve(config_dir, config["dataset"]["manifest"]),
        eligible_only=bool(config["dataset"].get("eligible_only", True)),
        bad_frame_policy=config["dataset"].get("bad_frame_policy", "drop_clip"),
    )
    clips = {clip.clip_id: clip for clip in adapter.iter_clips()}
    contact_cfg = config.get("contacts", {})
    report_path = output_root / "diagnostics" / f"mesh_contacts_{args.split}.jsonl"
    reports_by_clip: dict[str, dict] = {}
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            report = json.loads(line)
            if report.get("clip_id") is not None:
                reports_by_clip[str(report["clip_id"])] = report
    for record in _progress(records, desc=f"contacts[{args.split}]", unit="clip"):
        clip_id = record["clip_id"]
        output_path = output_root / "mesh_contacts" / args.split / f"{clip_id}.npz"
        if output_path.is_file() and not args.overwrite:
            print(f"SKIP {clip_id}: {output_path} exists")
            if args.inject_into_motion:
                labels = _load_npz_dict(output_path)
                inject_motion_contact_union(Path(record["motion_file"]), labels)
                if clip_id in reports_by_clip:
                    reports_by_clip[clip_id]["injected_into_motion"] = True
            continue
        collision_path = (
            output_root / "collision_meshes" / args.split / f"{record['scene_id']}.obj"
        )
        if not collision_path.is_file():
            raise FileNotFoundError(
                f"{clip_id}: missing {collision_path}; run prepare_collision_meshes.py first"
            )
        motion = torch.load(record["motion_file"], map_location="cpu", weights_only=False)
        print(f"PROCESS {clip_id}: final collision geometry")
        compute_dynamic_contacts = bool(contact_cfg.get("compute_dynamic_contacts", False))
        if bool(contact_cfg.get("include_dynamic_contacts_in_training", False)):
            compute_dynamic_contacts = True
        contact_source = (
            "static collision pointcloud + dynamic meshes"
            if compute_dynamic_contacts
            else "static collision pointcloud"
        )
        print(f"  contact geometry: {contact_source}")
        labels = generate_collision_mesh_contact_labels(
            collision_mesh_path=collision_path,
            object_cache_path=Path(record["object_cache"]),
            source_body_pos=_source_joints(adapter, clips[clip_id]),
            target_motion=motion,
            mjcf_path=PROJECT_ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml",
            contact_threshold_m=float(contact_cfg.get("distance_threshold_m", 0.025)),
            compatibility_threshold_m=float(
                contact_cfg.get("target_compatibility_threshold_m", 0.08)
            ),
            physics_validation_threshold_m=float(
                contact_cfg.get("physics_validation_threshold_m", 0.002)
            ),
            include_dynamic_contacts_in_training=bool(
                contact_cfg.get("include_dynamic_contacts_in_training", False)
            ),
            compute_dynamic_contacts=compute_dynamic_contacts,
            temporal_dilation_frames=int(contact_cfg.get("temporal_dilation_frames", 2)),
        )
        report = {
            "clip_id": clip_id,
            "split": args.split,
            "collision_mesh": str(collision_path),
            **save_contact_labels(output_path, labels),
        }
        if args.inject_into_motion:
            inject_motion_contact_union(Path(record["motion_file"]), labels)
            report["injected_into_motion"] = True
        reports_by_clip[clip_id] = report
        print(
            f"  training body/frame={report['training_body_frame_fraction']:.4f}, "
            f"retained intent={report['training_retained_of_intended']:.4f}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    reports = [
        reports_by_clip[clip_id]
        for clip_id in sorted(reports_by_clip)
        if reports_by_clip[clip_id].get("split") == args.split
    ]
    _atomic_write_text(
        report_path,
        "".join(json.dumps(report, ensure_ascii=False) + "\n" for report in reports),
    )
    print(f"Wrote {len(reports)} reports to {report_path}")


if __name__ == "__main__":
    main()
