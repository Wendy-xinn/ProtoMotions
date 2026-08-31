#!/usr/bin/env python3
"""Prepare scene-aware human motion through a dataset adapter.

Stages are intentionally explicit so every new dataset can be validated before
expensive simulation assets are built:

  validate -> motions -> alignment -> objects -> descriptors
           -> scene_pack -> motion_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Running this file directly adds data/scripts to sys.path, not the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.convert_amass_to_proto import convert_amass_to_motion
from data.scripts.scene_motion.contracts import ClipDescriptor
from data.scripts.scene_motion.transforms import (
    TRUMANS_Y_UP_TO_Z_UP,
    finite_difference,
    transform_points,
    transform_root_axis_angle_left,
    transform_world_rotations_left,
)
from data.scripts.scene_motion.trumans_adapter import TrumansAdapter
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from protomotions.components.pose_lib import compute_angular_velocity, extract_kinematic_info
from protomotions.components.scene_lib import (
    BoxSceneObject,
    MeshSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)


ALL_STAGES = (
    "validate",
    "motions",
    "alignment",
    "objects",
    "descriptors",
    "scene_pack",
    "motion_pack",
)


def _progress(iterable, **kwargs):
    return tqdm(iterable, dynamic_ncols=True, disable=not sys.stderr.isatty(), **kwargs)


def _atomic_write(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix if target.suffix else ".tmp"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only scene-motion config schema_version=1 is supported")
    return config


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _adapter_from_config(config: dict, config_dir: Path):
    dataset = config["dataset"]
    if dataset["adapter"] != "trumans":
        raise ValueError(
            f"Unknown adapter {dataset['adapter']!r}; implement DatasetAdapter for it"
        )
    root = _resolve(config_dir, dataset["root"])
    manifest = _resolve(config_dir, dataset["manifest"])
    return TrumansAdapter(
        root=root,
        manifest_path=manifest,
        eligible_only=bool(dataset.get("eligible_only", True)),
        bad_frame_policy=dataset.get("bad_frame_policy", "drop_clip"),
    )


def _split_for_scene(scene_id: str, split_config: dict) -> str:
    if not split_config.get("enabled", True):
        return split_config.get("single_split", "train")
    ratios = split_config.get("ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    value = int(hashlib.sha1(scene_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += float(ratio)
        if value < cumulative:
            return name
    return list(ratios)[-1]


def _select_clips(
    adapter,
    config: dict,
    limit: int | None,
    clip_ids: set[str] | None = None,
) -> dict[str, list[ClipDescriptor]]:
    splits: dict[str, list[ClipDescriptor]] = {}
    selected_ids = set()
    for clip in adapter.iter_clips():
        if clip_ids is not None and clip.clip_id not in clip_ids:
            continue
        split = _split_for_scene(clip.scene_id, config.get("split", {}))
        splits.setdefault(split, []).append(clip)
        selected_ids.add(clip.clip_id)
        if limit is not None and sum(map(len, splits.values())) >= limit:
            break
    if clip_ids is not None:
        missing = clip_ids - selected_ids
        if missing:
            raise ValueError(
                "Requested clips were not eligible/found: " + ", ".join(sorted(missing))
            )
    for clips in splits.values():
        clips.sort(key=lambda clip: clip.clip_id)
    if not splits:
        raise ValueError("No clips selected")
    return splits


def _validate(adapter, splits: dict[str, list[ClipDescriptor]]) -> dict:
    scene_ids = set()
    object_names = set()
    frame_count = 0
    for clips in splits.values():
        for clip in _progress(clips, desc="validate", unit="clip"):
            human = adapter.load_human(clip)
            objects = adapter.load_objects(clip)
            scene = adapter.load_scene(clip)
            human.validate(clip)
            for obj in objects:
                obj.validate(clip)
                object_names.add(obj.object_id)
            scene_ids.add(scene.scene_id)
            frame_count += clip.num_frames
    return {
        "clips": sum(map(len, splits.values())),
        "frames": frame_count,
        "hours": frame_count / 30.0 / 3600.0,
        "scenes": len(scene_ids),
        "object_mesh_types": len(object_names),
        "split_counts": {name: len(clips) for name, clips in splits.items()},
    }


_SCALE_DIAGNOSTIC_CHAINS = {
    "left_leg": (("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle")),
    "right_leg": (("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle")),
    "left_arm": (("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist")),
    "right_arm": (("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist")),
}


def _target_chain_lengths(kinematic_info) -> dict[str, float]:
    body_index = {name: index for index, name in enumerate(kinematic_info.body_names)}
    result = {}
    for chain_name, edges in _SCALE_DIAGNOSTIC_CHAINS.items():
        length = 0.0
        for parent_name, child_name in edges:
            child_index = body_index[child_name]
            parent_index = kinematic_info.parent_indices[child_index]
            if kinematic_info.body_names[parent_index] != parent_name:
                raise ValueError(
                    f"Target skeleton edge mismatch: {parent_name}->{child_name}"
                )
            length += float(torch.linalg.norm(kinematic_info.local_pos[child_index]))
        result[chain_name] = length
    return result


def _scale_diagnostics(
    adapter,
    splits: dict[str, list[ClipDescriptor]],
    output_root: Path,
    config: dict,
) -> dict:
    """Measure source-body to fixed-target limb scale without changing scene units."""
    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    )
    target_lengths = _target_chain_lengths(kinematic_info)
    records = []
    for split, clips in splits.items():
        for clip in _progress(clips, desc=f"scale[{split}]", unit="clip"):
            human = adapter.load_human(clip)
            if human.joint_positions is None or human.joint_names is None:
                continue
            joint_index = {name: index for index, name in enumerate(human.joint_names)}
            chain_ratios = {}
            for chain_name, edges in _SCALE_DIAGNOSTIC_CHAINS.items():
                source_length = 0.0
                for parent_name, child_name in edges:
                    parent = human.joint_positions[:, joint_index[parent_name]]
                    child = human.joint_positions[:, joint_index[child_name]]
                    source_length += float(
                        np.median(np.linalg.norm(child - parent, axis=-1))
                    )
                chain_ratios[chain_name] = source_length / target_lengths[chain_name]
            records.append(
                {
                    "clip_id": clip.clip_id,
                    "split": split,
                    "scene_id": clip.scene_id,
                    "uniform_scale_ratio": float(np.median(list(chain_ratios.values()))),
                    "chain_scale_ratios": chain_ratios,
                }
            )

    report_path = output_root / "diagnostics" / "human_scale.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not records:
        return {"available": False, "report": str(report_path)}

    ratios = np.asarray([record["uniform_scale_ratio"] for record in records])
    max_deviation = float(
        config.get("human", {}).get("scale_warning_relative_deviation", 0.10)
    )
    return {
        "available": True,
        "policy": config.get("human", {}).get(
            "scale_policy", "fixed_target_preserve_scene_metric"
        ),
        "target_robot": "smpl_humanoid.xml",
        "ratio_definition": "median(source limb-chain length / target limb-chain length)",
        "clips": len(records),
        "ratio_p05": float(np.quantile(ratios, 0.05)),
        "ratio_median": float(np.median(ratios)),
        "ratio_p95": float(np.quantile(ratios, 0.95)),
        "warning_relative_deviation": max_deviation,
        "clips_outside_warning": int(np.sum(np.abs(ratios - 1.0) > max_deviation)),
        "report": str(report_path),
    }


def _alignment_diagnostics(
    adapter,
    splits: dict[str, list[ClipDescriptor]],
    output_root: Path,
) -> dict:
    """Compare converted fixed-humanoid joints to released source joints in world space."""
    records = []
    for split, clips in splits.items():
        for clip in _progress(clips, desc=f"alignment[{split}]", unit="clip"):
            human = adapter.load_human(clip)
            if human.joint_positions is None or human.joint_names is None:
                continue
            motion_path = output_root / "motions" / split / f"{clip.clip_id}.motion"
            if not motion_path.is_file():
                raise FileNotFoundError(
                    f"Alignment requires converted motion {motion_path}; run motions first"
                )
            motion = torch.load(motion_path, map_location="cpu", weights_only=False)
            target = motion["rigid_body_pos"].cpu().numpy()
            source_index = {name: index for index, name in enumerate(human.joint_names)}
            source = transform_points(human.joint_positions, TRUMANS_Y_UP_TO_Z_UP)
            source = source[:, [source_index[name] for name in SMPL_MUJOCO_NAMES]]
            if source.shape != target.shape:
                raise ValueError(
                    f"{clip.clip_id}: source joints {source.shape} != target {target.shape}; "
                    "alignment diagnostics currently require equal source/output FPS"
                )
            errors = np.linalg.norm(source - target, axis=-1)
            per_body_median = {
                name: float(np.median(errors[:, index]))
                for index, name in enumerate(SMPL_MUJOCO_NAMES)
            }
            records.append(
                {
                    "clip_id": clip.clip_id,
                    "split": split,
                    "median_joint_error_m": float(np.median(errors)),
                    "p95_joint_error_m": float(np.quantile(errors, 0.95)),
                    "max_joint_error_m": float(np.max(errors)),
                    "per_body_median_error_m": per_body_median,
                }
            )

    report_path = output_root / "diagnostics" / "human_alignment.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not records:
        return {"available": False, "report": str(report_path)}
    medians = np.asarray([record["median_joint_error_m"] for record in records])
    p95s = np.asarray([record["p95_joint_error_m"] for record in records])
    return {
        "available": True,
        "clips": len(records),
        "median_of_clip_medians_m": float(np.median(medians)),
        "p95_of_clip_medians_m": float(np.quantile(medians, 0.95)),
        "median_of_clip_p95_m": float(np.median(p95s)),
        "report": str(report_path),
    }


def _convert_motion(adapter, clip: ClipDescriptor, output_path: Path, config: dict) -> None:
    human = adapter.load_human(clip)
    basis = TRUMANS_Y_UP_TO_Z_UP
    translation = transform_points(human.root_translation, basis)
    root_axis_angle = transform_root_axis_angle_left(human.root_orientation, basis)
    pose_axis_angle = np.concatenate([root_axis_angle, human.body_pose], axis=-1)

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/smpl_humanoid.xml"
    )
    human_cfg = config.get("human", {})
    motion, _ = convert_amass_to_motion(
        pose_aa=pose_axis_angle,
        amass_trans=translation,
        mocap_fr=int(round(clip.fps)),
        output_fps=int(human_cfg.get("output_fps", 30)),
        humanoid_type="smpl",
        joint_names=SMPL_BONE_ORDER_NAMES,
        mujoco_joint_names=SMPL_MUJOCO_NAMES,
        kinematic_info=kinematic_info,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fix_height=bool(human_cfg.get("fix_height", False)),
        compute_ground_contacts=bool(human_cfg.get("compute_ground_contacts", True)),
    )
    _atomic_write(output_path, lambda temp_path: torch.save(motion.to_dict(), temp_path))


def _scale_slug(scale: np.ndarray) -> str:
    values = [float(value) for value in scale.reshape(-1)]
    return "_".join(f"{value:g}".replace("-", "m").replace(".", "p") for value in values)


def _object_mesh_override_path(
    *,
    object_id: str,
    mesh_path: Path,
    output_root: Path,
    override_cfg: dict,
) -> tuple[Path, np.ndarray]:
    scale_value = override_cfg.get("scale", override_cfg.get("mesh_scale", 1.0))
    scale = np.asarray(scale_value, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(3, float(scale), dtype=np.float64)
    else:
        scale = scale.reshape(-1)
        if len(scale) != 3:
            raise ValueError(
                f"object_mesh_overrides.{object_id}.scale must be scalar or XYZ triplet"
            )
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(
            f"object_mesh_overrides.{object_id}.scale must contain positive finite values"
        )
    if np.allclose(scale, 1.0):
        return mesh_path, scale.astype(np.float32)

    override_root = output_root / "object_mesh_overrides"
    override_path = override_root / f"{object_id}.scale_{_scale_slug(scale)}.obj"
    if not override_path.is_file():
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"Expected a non-empty object mesh at {mesh_path}")
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale[None]
        _atomic_write(override_path, lambda temp_path: mesh.export(temp_path))
    return override_path, scale.astype(np.float32)


def _convert_objects(
    adapter,
    clip: ClipDescriptor,
    output_path: Path,
    config: dict,
    output_root: Path,
) -> dict:
    basis = TRUMANS_Y_UP_TO_Z_UP
    override_cfg = config.get("object_mesh_overrides", {})
    objects = adapter.load_objects(clip)
    names = []
    mesh_paths = []
    source_mesh_paths = []
    mesh_scales = []
    translations = []
    rotations = []
    linear_velocities = []
    angular_velocities = []
    for obj in objects:
        mesh_path, mesh_scale = _object_mesh_override_path(
            object_id=obj.object_id,
            mesh_path=Path(obj.mesh_path),
            output_root=output_root,
            override_cfg=override_cfg.get(obj.object_id, {}),
        )
        translation = transform_points(obj.translation, basis)
        quaternion = transform_world_rotations_left(
            obj.rotation, obj.rotation_format, basis
        )
        rotation_matrices = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
        angular_velocity = compute_angular_velocity(
            torch.from_numpy(rotation_matrices), fps=clip.fps
        ).cpu().numpy()
        names.append(obj.object_id)
        mesh_paths.append(str(mesh_path))
        source_mesh_paths.append(str(obj.mesh_path))
        mesh_scales.append(mesh_scale)
        translations.append(translation)
        rotations.append(quaternion)
        linear_velocities.append(finite_difference(translation, clip.fps))
        angular_velocities.append(angular_velocity)

    _atomic_write(
        output_path,
        lambda temp_path: np.savez_compressed(
            temp_path,
            names=np.asarray(names, dtype=str),
            mesh_paths=np.asarray(mesh_paths, dtype=str),
            source_mesh_paths=np.asarray(source_mesh_paths, dtype=str),
            mesh_scales=np.asarray(mesh_scales, dtype=np.float32),
            translations=np.asarray(translations, dtype=np.float32),
            rotations_xyzw=np.asarray(rotations, dtype=np.float32),
            linear_velocities=np.asarray(linear_velocities, dtype=np.float32),
            angular_velocities=np.asarray(angular_velocities, dtype=np.float32),
            fps=np.asarray(clip.fps, dtype=np.float32),
        ),
    )
    return {"objects": len(objects), "path": str(output_path)}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_descriptors(
    adapter,
    split: str,
    clips: list[ClipDescriptor],
    output_root: Path,
) -> Path:
    descriptor_path = output_root / "descriptors" / f"{split}.jsonl"
    basis_quaternion = Rotation.from_matrix(TRUMANS_Y_UP_TO_Z_UP).as_quat().tolist()
    def _write(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as stream:
            for motion_id, clip in enumerate(_progress(clips, desc=f"descriptors[{split}]", unit="clip")):
                scene = adapter.load_scene(clip)
                record = {
                    "motion_id": motion_id,
                    "clip_id": clip.clip_id,
                    "split": split,
                    "fps": clip.fps,
                    "num_frames": clip.num_frames,
                    "scene_id": clip.scene_id,
                    "motion_file": str(output_root / "motions" / split / f"{clip.clip_id}.motion"),
                    "object_cache": str(output_root / "objects" / split / f"{clip.clip_id}.npz"),
                    "scene_mesh": str(scene.mesh_path),
                    "scene_mesh_rotation_xyzw": basis_quaternion,
                    "occupancy": str(scene.occupancy_path) if scene.occupancy_path else None,
                    "occupancy_axis_order": scene.occupancy_axis_order,
                    "occupancy_bounds_min": scene.occupancy_bounds_min,
                    "occupancy_bounds_max": scene.occupancy_bounds_max,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    _atomic_write(descriptor_path, _write)
    return descriptor_path


def _mesh_object_with_cached_dims(cache: dict, **kwargs) -> MeshSceneObject:
    path = str(kwargs["object_path"])
    if path in cache:
        kwargs["object_dims"] = cache[path]
    obj = MeshSceneObject(**kwargs)
    cache[path] = obj.object_dims
    return obj


def _build_scene_pack(
    adapter,
    split: str,
    clips: list[ClipDescriptor],
    output_root: Path,
    dataset_root: Path,
    config: dict,
) -> Path:
    scene_cfg = config.get("scene_pack", {})
    object_counts = [len(adapter.load_objects(clip)) for clip in clips]
    max_objects = scene_cfg.get("max_objects", "auto")
    max_objects = max(object_counts) if max_objects == "auto" else int(max_objects)
    if max(object_counts) > max_objects:
        raise ValueError(
            f"max_objects={max_objects}, but selected clips require {max(object_counts)}"
        )

    basis_quaternion = Rotation.from_matrix(TRUMANS_Y_UP_TO_Z_UP).as_quat()
    scenes = []
    dims_cache: dict[str, tuple] = {}
    for motion_id, clip in enumerate(_progress(clips, desc=f"scene_pack[{split}]", unit="clip")):
        scene_asset = adapter.load_scene(clip)
        scene_objects = [
            _mesh_object_with_cached_dims(
                dims_cache,
                object_path=str(scene_asset.mesh_path),
                translation=(0.0, 0.0, 0.0),
                rotation=basis_quaternion,
                options=ObjectOptions(fix_base_link=True),
            )
        ]
        cache = np.load(output_root / "objects" / split / f"{clip.clip_id}.npz")
        for index, object_id in enumerate(cache["names"].tolist()):
            scene_objects.append(
                _mesh_object_with_cached_dims(
                    dims_cache,
                    object_path=str(cache["mesh_paths"][index]),
                    translation=cache["translations"][index],
                    rotation=cache["rotations_xyzw"][index],
                    fps=float(cache["fps"]),
                    options=ObjectOptions(
                        fix_base_link=False,
                        density=float(scene_cfg.get("object_density", 500.0)),
                        static_friction=float(scene_cfg.get("static_friction", 0.8)),
                        dynamic_friction=float(scene_cfg.get("dynamic_friction", 0.6)),
                    ),
                )
            )
        for _ in range(max_objects - len(cache["names"])):
            scene_objects.append(
                BoxSceneObject(
                    width=0.001,
                    depth=0.001,
                    height=0.001,
                    translation=(0.0, 0.0, -100.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    options=ObjectOptions(fix_base_link=True),
                )
            )
        scenes.append(Scene(objects=scene_objects, humanoid_motion_id=motion_id))

    output_path = output_root / "scenes" / f"{split}.pt"
    _atomic_write(
        output_path,
        lambda temp_path: SceneLib.save_scenes_to_file(
            scenes, str(temp_path), asset_root=str(dataset_root)
        ),
    )
    return output_path


def _write_motion_yaml(split: str, clips: list[ClipDescriptor], output_root: Path) -> Path:
    path = output_root / "motion_configs" / f"{split}.yaml"
    motions = [
        {
            "file": str(output_root / "motions" / split / f"{clip.clip_id}.motion"),
            "fps": clip.fps,
            "weight": 1.0,
            "idx": index,
        }
        for index, clip in enumerate(clips)
    ]
    _atomic_write(
        path,
        lambda temp_path: temp_path.write_text(
            yaml.safe_dump({"motions": motions}, sort_keys=False), encoding="utf-8"
        ),
    )
    return path


def _package_motion(split: str, yaml_path: Path, output_root: Path) -> Path:
    output_path = output_root / "motion_libs" / f"{split}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=output_path.suffix if output_path.suffix else ".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        sys.executable,
        "protomotions/components/motion_lib.py",
        "--motion-path",
        str(yaml_path),
        "--output-file",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated subset of: {','.join(ALL_STAGES)}",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--clip-id",
        action="append",
        default=None,
        help="Process only this clip ID; repeat the option to select multiple clips.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override config output_root, useful for isolated review runs.",
    )
    parser.add_argument(
        "--existing-motions-only",
        action="store_true",
        help="Restrict later stages to clip IDs already cached under output_root/motions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    config_dir = config_path.parent
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _resolve(config_dir, config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = _adapter_from_config(config, config_dir)
    selected_clip_ids = set(args.clip_id) if args.clip_id else None
    if args.existing_motions_only:
        existing_ids = {
            path.stem
            for path in (output_root / "motions").glob("*/*.motion")
        }
        if not existing_ids:
            raise ValueError(f"No existing motions found under {output_root / 'motions'}")
        selected_clip_ids = (
            existing_ids
            if selected_clip_ids is None
            else selected_clip_ids & existing_ids
        )
    splits = _select_clips(
        adapter,
        config,
        args.limit,
        selected_clip_ids,
    )
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    summary_path = output_root / "preparation_summary.json"
    if "validate" in stages:
        summary = _validate(adapter, splits)
    elif summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    if "validate" in stages:
        summary["human_scale"] = _scale_diagnostics(
            adapter, splits, output_root, config
        )
    dataset_root = adapter.root
    for split, clips in splits.items():
        if "motions" in stages:
            for clip in _progress(clips, desc=f"motions[{split}]", unit="clip"):
                path = output_root / "motions" / split / f"{clip.clip_id}.motion"
                if args.overwrite or not path.exists():
                    _convert_motion(adapter, clip, path, config)
        if "objects" in stages:
            for clip in _progress(clips, desc=f"objects[{split}]", unit="clip"):
                path = output_root / "objects" / split / f"{clip.clip_id}.npz"
                if args.overwrite or not path.exists():
                    _convert_objects(adapter, clip, path, config, output_root)
        if "descriptors" in stages:
            _write_descriptors(adapter, split, clips, output_root)
        yaml_path = _write_motion_yaml(split, clips, output_root)
        if "scene_pack" in stages:
            _build_scene_pack(
                adapter, split, clips, output_root, dataset_root, config
            )
        if "motion_pack" in stages:
            _package_motion(split, yaml_path, output_root)

    if "alignment" in stages:
        summary["human_alignment"] = _alignment_diagnostics(
            adapter, splits, output_root
        )
    summary.update(
        {
            "schema_version": config["schema_version"],
            "adapter": config["dataset"]["adapter"],
            "output_root": str(output_root),
            "stages": stages,
            "coordinate_basis_source_to_target": TRUMANS_Y_UP_TO_Z_UP.tolist(),
        }
    )
    _atomic_write(
        summary_path,
        lambda temp_path: temp_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
