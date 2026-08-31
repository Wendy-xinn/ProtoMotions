#!/usr/bin/env python3
"""Build aligned, simplified static colliders and dense surface caches by scene."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.scene_motion.collision_mesh import (
    CollisionMeshConfig,
    prepare_static_collision_mesh,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "data/yaml_files/trumans_scene_motion.yaml",
    )
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--clip-id", action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-usd", action="store_true", help="Skip pxr USD export")
    parser.add_argument("--overwrite", action="store_true")
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
    if args.scene_id:
        wanted_scenes = set(args.scene_id)
        records = [record for record in records if record["scene_id"] in wanted_scenes]
        missing = wanted_scenes - {record["scene_id"] for record in records}
        if missing:
            raise ValueError(f"Unknown scene IDs in {args.split}: {sorted(missing)}")
    if args.clip_id:
        selected = set(args.clip_id)
        records = [record for record in records if record["clip_id"] in selected]
        missing = selected - {record["clip_id"] for record in records}
        if missing:
            raise ValueError(f"Unknown clip IDs in {args.split}: {sorted(missing)}")
    if args.limit is not None:
        records = records[: args.limit]

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["scene_id"], []).append(record)

    mesh_config = CollisionMeshConfig(**config.get("collision_mesh", {}))
    reports = []
    for scene_id, scene_records in _progress(
        grouped.items(), total=len(grouped), desc=f"collision_meshes[{args.split}]", unit="scene"
    ):
        output_stem = output_root / "collision_meshes" / args.split / scene_id
        report_path = output_stem.with_suffix(".json")
        if report_path.is_file() and not args.overwrite:
            print(f"SKIP {scene_id}: {report_path} exists")
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue

        body_lo = None
        body_hi = None
        swept_body_samples = 0
        dynamic_objects = []
        for record in scene_records:
            motion = torch.load(
                record["motion_file"], map_location="cpu", weights_only=False
            )
            body_points = motion["rigid_body_pos"].cpu().numpy().reshape(-1, 3)
            if body_points.size == 0 or not np.isfinite(body_points).all():
                raise ValueError(
                    f"{record['clip_id']}: rigid_body_pos contains invalid values"
                )
            clip_lo = body_points.min(axis=0)
            clip_hi = body_points.max(axis=0)
            body_lo = clip_lo if body_lo is None else np.minimum(body_lo, clip_lo)
            body_hi = clip_hi if body_hi is None else np.maximum(body_hi, clip_hi)
            swept_body_samples += int(body_points.shape[0])
            # Object caches contain per-frame world poses, but the released
            # static scan does not identify which recording/frame it captured.
            # Use one deterministic reference recording only; accumulating a
            # best-matching pose from every motion carves future placement
            # locations out of otherwise valid support surfaces.
            if not dynamic_objects:
                dynamic_objects.append(
                    {
                        "name": record["clip_id"],
                        "object_cache_path": str(record["object_cache"]),
                    }
                )
            del motion, body_points
            gc.collect()
        if body_lo is None or body_hi is None:
            raise ValueError(f"{scene_id}: no body points found")
        print(
            f"PROCESS {scene_id}: {len(scene_records)} clips, "
            f"{swept_body_samples:,} swept body samples"
        )
        report = prepare_static_collision_mesh(
            Path(scene_records[0]["scene_mesh"]),
            None,
            output_stem,
            mesh_config,
            body_bounds=(body_lo, body_hi),
            dynamic_objects=dynamic_objects,
            write_usd=not args.no_usd,
        )
        report["scene_id"] = scene_id
        report["clip_ids"] = [record["clip_id"] for record in scene_records]
        _atomic_write_text(
            report_path,
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
        reports.append(report)
        print(
            f"  faces {report['source_faces']:,} -> {report['cropped_faces']:,} "
            f"-> {report['collision_faces']:,}; "
            f"surface p95~{report['approx_surface_distance_p95_m'] * 1000:.1f} mm"
        )

    summary_path = output_root / "diagnostics" / f"collision_meshes_{args.split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        summary_path,
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"Wrote {len(reports)} scene reports to {summary_path}")


if __name__ == "__main__":
    main()
