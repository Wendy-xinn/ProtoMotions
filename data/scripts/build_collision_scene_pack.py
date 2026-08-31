#!/usr/bin/env python3
"""Replace static room assets in a SceneLib pack with prepared triangle colliders."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import trimesh
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.scene_motion.collision_mesh import write_static_collision_usda


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


def _atomic_write_pt(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".pt"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=suffix,
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_npz(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sample_collision_pointcloud(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    return np.asarray(points, dtype=np.float32), normals


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
    parser.add_argument(
        "--source-scene-file",
        type=Path,
        default=None,
        help="Defaults to <output-root>/scenes/<split>.pt.",
    )
    parser.add_argument(
        "--output-scene-file",
        type=Path,
        default=None,
        help="Defaults to <output-root>/scene_libs/<split>.pt.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else _resolve(config_path.parent, config["output_root"])
    )
    source_path = (
        args.source_scene_file.resolve()
        if args.source_scene_file is not None
        else output_root / "scenes" / f"{args.split}.pt"
    )
    target_path = (
        args.output_scene_file.resolve()
        if args.output_scene_file is not None
        else output_root / "scene_libs" / f"{args.split}.pt"
    )
    all_records = _records(output_root / "descriptors" / f"{args.split}.jsonl")
    indexed_records = list(enumerate(all_records))
    if args.clip_id:
        selected = set(args.clip_id)
        indexed_records = [
            (index, record)
            for index, record in indexed_records
            if record["clip_id"] in selected
        ]
        missing = selected - {record["clip_id"] for _, record in indexed_records}
        if missing:
            raise ValueError(f"Unknown clip IDs in {args.split}: {sorted(missing)}")
    if args.limit is not None:
        indexed_records = indexed_records[: args.limit]
    records = [record for _, record in indexed_records]
    scene_indices = [index for index, _ in indexed_records]
    data = torch.load(source_path, map_location="cpu", weights_only=False)
    scenes = data.get("original_scenes") if isinstance(data, dict) else data
    if scenes is None:
        raise ValueError(f"Scene pack not found at {source_path}")
    if len(scene_indices) == len(scenes) and scene_indices == list(range(len(scenes))):
        selected_scenes = scenes
    else:
        if max(scene_indices, default=-1) >= len(scenes):
            raise ValueError(
                f"Scene pack / descriptor mismatch: {len(scenes)} scenes, "
                f"but requested indices {scene_indices[:5]}..."
            )
        selected_scenes = [scenes[index] for index in scene_indices]
    if len(selected_scenes) != len(records):
        raise ValueError(
            f"Scene/descriptor mismatch: {len(selected_scenes)} scenes versus "
            f"{len(records)} descriptors"
        )

    output_data = copy.deepcopy(data)
    output_scenes = [copy.deepcopy(scene) for scene in selected_scenes]
    if isinstance(output_data, dict):
        output_data["original_scenes"] = output_scenes
        output_data["num_original_scenes"] = len(output_scenes)
    else:
        output_data = output_scenes
    for scene, record in _progress(
        zip(output_scenes, records),
        total=len(records),
        desc=f"scene_libs[{args.split}]",
        unit="scene",
    ):
        stem = output_root / "collision_meshes" / args.split / record["scene_id"]
        obj_path = stem.with_suffix(".obj")
        usd_path = stem.with_suffix(".usda")
        pointcloud_path = stem.with_suffix(".pointcloud.npz")
        report_path = stem.with_suffix(".json")
        if not obj_path.is_file():
            raise FileNotFoundError(
                f"Missing {obj_path}; prepare all {args.split} collision meshes first"
            )
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        if not usd_path.is_file():
            print(f"  repairing missing USD: {usd_path.name}")
            mesh = trimesh.load(obj_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            write_static_collision_usda(mesh, usd_path)
        if not pointcloud_path.is_file():
            print(f"  repairing missing pointcloud: {pointcloud_path.name}")
            mesh = trimesh.load(obj_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            config = report.get("config", {})
            samples = int(config.get("dense_pointcloud_samples", 50_000))
            seed = int(config.get("random_seed", 0))
            points, normals = _sample_collision_pointcloud(mesh, samples, seed)
            crop_min = np.asarray(report.get("crop_bounds_min", mesh.bounds[0]), dtype=np.float32)
            crop_max = np.asarray(report.get("crop_bounds_max", mesh.bounds[1]), dtype=np.float32)
            _atomic_write_npz(
                pointcloud_path,
                lambda temp_path: np.savez_compressed(
                    temp_path,
                    points=points,
                    normals=normals,
                    crop_bounds_min=crop_min,
                    crop_bounds_max=crop_max,
                ),
            )
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        bounds = mesh.bounds
        static_object = scene["objects"][0]
        if static_object.get("type") != "MeshSceneObject":
            raise ValueError(f"{record['clip_id']}: first scene object is not a room mesh")
        static_object["object_path"] = str(usd_path.resolve())
        static_object["translation"] = [[0.0, 0.0, 0.0]]
        static_object["rotation"] = [[0.0, 0.0, 0.0, 1.0]]
        static_object["fps"] = 1.0
        static_object["object_dims"] = (
            float(bounds[0, 0]),
            float(bounds[1, 0]),
            float(bounds[0, 1]),
            float(bounds[1, 1]),
            float(bounds[0, 2]),
            float(bounds[1, 2]),
        )
        static_object.setdefault("options", {})["fix_base_link"] = True

    _atomic_write_pt(target_path, lambda temp_path: torch.save(output_data, temp_path))
    print(f"Wrote {len(output_scenes)} scene_lib packs to {target_path}")


if __name__ == "__main__":
    main()
