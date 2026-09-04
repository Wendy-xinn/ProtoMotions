#!/usr/bin/env python3
"""Prepare one or more native-SMPL EgoBody clips with an aligned static scene.

EgoBody wearer SMPL fits are expressed in the master Kinect RGB frame.  The
official ``kinect12_to_world/<scene>.json`` matrix maps that frame into the
released scene-mesh frame.  Both motion and mesh are then rotated together
from the released Y-up convention into ProtoMotions' Z-up convention.

When egocentric RGB files are available, their ``timestamp_frame_xxxxx.jpg``
names provide the exact frame-ID-to-timestamp association. The corresponding
PV-to-HoloLens transforms are chained through the official calibration files
and saved in ProtoMotions world coordinates for visualization and training.
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
import smplx
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Legacy official SMPL pickles contain chumpy arrays. Chumpy still calls the
# Python <=3.10 inspection alias while IsaacLab uses Python 3.12.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.convert_amass_to_proto import convert_amass_to_motion
from data.scripts.scene_motion.transforms import TRUMANS_Y_UP_TO_Z_UP
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.components.scene_lib import (
    MeshSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egobody-root", type=Path, default=Path("/home/wenxin/projects/egobody"))
    parser.add_argument("--recording", default="recording_20210918_S05_S06_01")
    parser.add_argument("--frame-start", type=int, default=None, help="Official Kinect frame ID")
    parser.add_argument("--frame-count", type=int, default=256)
    parser.add_argument("--num-clips", type=int, default=1)
    parser.add_argument("--clip-stride", type=int, default=256)
    parser.add_argument(
        "--clip-starts",
        type=int,
        nargs="+",
        default=None,
        help="Explicit frame IDs for non-uniform clips; overrides --frame-start/--num-clips.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/motion_for_trackers/egobody_smpl_ego_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _recording_info(root: Path, recording: str) -> dict[str, str]:
    with (root / "data_info_release.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["recording_name"]: row for row in csv.DictReader(handle)}
    if recording not in rows:
        raise KeyError(f"Recording {recording!r} is absent from data_info_release.csv")
    return rows[recording]


def _split(root: Path, recording: str) -> str:
    with (root / "data_splits.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for split_name in ("train", "val", "test"):
                if row.get(split_name) == recording:
                    return split_name
    raise KeyError(f"Recording {recording!r} is absent from data_splits.csv")


def _load_fit(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    pose = np.concatenate(
        [
            np.asarray(data["global_orient"], dtype=np.float64).reshape(3),
            np.asarray(data["body_pose"], dtype=np.float64).reshape(69),
        ]
    )
    translation = np.asarray(data["transl"], dtype=np.float64).reshape(3)
    betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:10]
    return pose, translation, betas


def _transform_motion(
    poses: np.ndarray,
    translations: np.ndarray,
    scene_from_kinect: np.ndarray,
    floor_height_scene_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proto_from_scene = np.eye(4, dtype=np.float64)
    proto_from_scene[:3, :3] = TRUMANS_Y_UP_TO_Z_UP
    # Put the scanned floor at simulator z=0. Apply this to both scene and
    # motion; shifting only the humanoid would destroy physical alignment.
    proto_from_scene[2, 3] = -floor_height_scene_y
    proto_from_kinect = proto_from_scene @ scene_from_kinect
    rotation = proto_from_kinect[:3, :3]
    transformed = poses.copy()
    root_rotation = Rotation.from_rotvec(poses[:, :3]).as_matrix()
    transformed[:, :3] = Rotation.from_matrix(rotation[None] @ root_rotation).as_rotvec()
    transformed_translation = (
        np.einsum("ij,tj->ti", rotation, translations) + proto_from_kinect[:3, 3]
    )
    return transformed.astype(np.float32), transformed_translation.astype(np.float32), proto_from_kinect


def _floor_height_scene_y(scene_mesh: trimesh.Trimesh) -> float:
    """Estimate the dominant lowest walkable Y-up surface of a room scan."""
    normals = np.asarray(scene_mesh.face_normals)
    centers = np.asarray(scene_mesh.triangles_center)
    areas = np.asarray(scene_mesh.area_faces)
    upward = normals[:, 1] > 0.9
    if not upward.any():
        raise ValueError("Scene mesh has no upward-facing surfaces; cannot find floor")
    heights = centers[upward, 1]
    weights = areas[upward]
    # Area-weighted histogram rejects isolated low scan artifacts and selects
    # the strongest horizontal layer in the lower quarter of the room.
    lower_limit = float(np.quantile(heights, 0.01))
    upper_limit = float(np.quantile(heights, 0.25))
    bins = max(32, int(np.ceil((upper_limit - lower_limit) / 0.01)))
    histogram, edges = np.histogram(
        heights, bins=bins, range=(lower_limit, upper_limit), weights=weights
    )
    index = int(histogram.argmax())
    return float(0.5 * (edges[index] + edges[index + 1]))


def _smpl_pelvis_positions(
    poses: np.ndarray,
    translations: np.ndarray,
    betas: np.ndarray,
    gender: str,
) -> np.ndarray:
    """Evaluate shaped SMPL joint 0; EgoBody ``transl`` is not pelvis position."""
    model_file = PROJECT_ROOT / "data/smpl" / f"SMPL_{gender.upper()}.pkl"
    model = smplx.SMPL(
        str(model_file),
        batch_size=len(poses),
    )
    with torch.no_grad():
        output = model(
            global_orient=torch.from_numpy(poses[:, :3]).float(),
            body_pose=torch.from_numpy(poses[:, 3:]).float(),
            transl=torch.from_numpy(translations).float(),
            betas=torch.from_numpy(betas).float(),
        )
    return output.joints[:, 0].cpu().numpy().astype(np.float64)


def _load_scene_transform(root: Path, recording: str, scene_name: str) -> np.ndarray:
    path = (
        root
        / "calibrations"
        / recording
        / "cal_trans"
        / "kinect12_to_world"
        / f"{scene_name}.json"
    )
    matrix = np.asarray(json.loads(path.read_text(encoding="utf-8"))["trans"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid Kinect-to-scene transform at {path}")
    det = np.linalg.det(matrix[:3, :3])
    orthogonality = np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))
    if abs(det - 1.0) > 1.0e-3 or orthogonality > 1.0e-3:
        raise ValueError(f"Non-rigid calibration at {path}: det={det}, orthogonality={orthogonality}")
    return matrix


def _find_pv_recording_dir(root: Path, recording: str) -> Path | None:
    """Accept both the official archive layout and a directly copied recording."""
    candidates = []
    for parent in (root / "egocentric_color" / recording, root / recording):
        if parent.is_dir():
            candidates.extend(path.parent for path in parent.glob("202*/*_pv.txt"))
    unique = sorted(set(candidates))
    if not unique:
        return None
    if len(unique) != 1:
        raise ValueError(f"Expected one PV stream for {recording}, got {unique}")
    return unique[0]


def _load_pv_stream(root: Path, recording: str) -> dict | None:
    pv_dir = _find_pv_recording_dir(root, recording)
    if pv_dir is None:
        return None
    pv_files = list(pv_dir.glob("*_pv.txt"))
    if len(pv_files) != 1:
        raise ValueError(f"Expected one *_pv.txt under {pv_dir}, got {pv_files}")
    lines = pv_files[0].read_text(encoding="utf-8").splitlines()
    cx, cy, width, height = ast.literal_eval(lines[0])
    by_timestamp = {}
    for line in lines[1:]:
        fields = line.split(",")
        by_timestamp[fields[0]] = (
            float(fields[1]),
            float(fields[2]),
            np.asarray(fields[3:20], dtype=np.float64).reshape(4, 4),
        )
    by_frame = {}
    for image_path in (pv_dir / "PV").glob("*_frame_*.jpg"):
        frame_text = image_path.stem.rsplit("_frame_", 1)
        if len(frame_text) != 2:
            continue
        timestamp, frame_id = frame_text
        if timestamp in by_timestamp:
            by_frame[int(frame_id)] = (*by_timestamp[timestamp], image_path)
    if not by_frame:
        raise ValueError(f"No PV images could be matched to {pv_files[0]}")
    return {
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height),
        "by_frame": by_frame,
        "pv_info_path": str(pv_files[0]),
    }


def _camera_clip(
    pv_stream: dict,
    frame_ids: np.ndarray,
    proto_from_kinect: np.ndarray,
    holo_to_kinect: np.ndarray,
) -> dict:
    """Build a dense camera trajectory, interpolating dropped PV frames."""
    by_frame = pv_stream["by_frame"]
    known_ids = np.asarray(sorted(by_frame), dtype=np.int64)
    if frame_ids[0] < known_ids[0] or frame_ids[-1] > known_ids[-1]:
        raise ValueError(
            f"PV coverage {known_ids[0]}..{known_ids[-1]} does not cover "
            f"clip {frame_ids[0]}..{frame_ids[-1]}"
        )
    known_camera = np.stack(
        [proto_from_kinect @ holo_to_kinect @ by_frame[int(fid)][2] for fid in known_ids]
    )
    translations = np.column_stack(
        [np.interp(frame_ids, known_ids, known_camera[:, axis, 3]) for axis in range(3)]
    )
    rotations = Slerp(
        known_ids.astype(np.float64), Rotation.from_matrix(known_camera[:, :3, :3])
    )(frame_ids.astype(np.float64)).as_matrix()
    world_from_camera = np.tile(np.eye(4, dtype=np.float32), (len(frame_ids), 1, 1))
    world_from_camera[:, :3, :3] = rotations.astype(np.float32)
    world_from_camera[:, :3, 3] = translations.astype(np.float32)
    known_fx = np.asarray([by_frame[int(fid)][0] for fid in known_ids])
    known_fy = np.asarray([by_frame[int(fid)][1] for fid in known_ids])
    observed = np.asarray([int(fid) in by_frame for fid in frame_ids], dtype=np.bool_)
    image_paths = [
        str(by_frame[int(fid)][3]) if int(fid) in by_frame else "" for fid in frame_ids
    ]
    return {
        "frame_ids": torch.from_numpy(frame_ids.copy()),
        "world_from_camera": torch.from_numpy(world_from_camera),
        "fx": torch.from_numpy(np.interp(frame_ids, known_ids, known_fx).astype(np.float32)),
        "fy": torch.from_numpy(np.interp(frame_ids, known_ids, known_fy).astype(np.float32)),
        "cx": pv_stream["cx"],
        "cy": pv_stream["cy"],
        "width": pv_stream["width"],
        "height": pv_stream["height"],
        "observed": torch.from_numpy(observed),
        "image_paths": image_paths,
    }


def _motion_diagnostics(
    motion,
    scene_mesh: trimesh.Trimesh,
    body_names: list[str],
    scene_z_translation: float,
) -> dict:
    rng = np.random.default_rng(20260825)
    sample_count = min(200_000, max(20_000, len(scene_mesh.faces) // 2))
    # trimesh's sampler uses NumPy's global RNG; preserve reproducibility here.
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        surface, _ = trimesh.sample.sample_surface(scene_mesh, sample_count)
    finally:
        np.random.set_state(state)
    surface_proto = np.einsum("ij,nj->ni", TRUMANS_Y_UP_TO_Z_UP, surface)
    surface_proto[:, 2] += scene_z_translation
    tree = cKDTree(surface_proto)
    foot_names = [name for name in body_names if any(token in name.lower() for token in ("ankle", "foot", "toe"))]
    foot_ids = [body_names.index(name) for name in foot_names]
    feet = motion.rigid_body_pos[:, foot_ids].cpu().numpy().reshape(-1, 3)
    distances = tree.query(feet, workers=-1)[0]
    root = motion.rigid_body_pos[:, 0].cpu().numpy()
    return {
        "frames": int(root.shape[0]),
        "root_bounds_proto_m": [root.min(0).tolist(), root.max(0).tolist()],
        "foot_bodies": foot_names,
        "foot_to_scene_sampled_surface_m": {
            "median": float(np.median(distances)),
            "p95": float(np.quantile(distances, 0.95)),
            "max": float(distances.max()),
        },
    }


def main() -> None:
    args = _parse_args()
    root = args.egobody_root.resolve()
    info = _recording_info(root, args.recording)
    split = _split(root, args.recording)
    first_frame = int(info["start_frame"]) if args.frame_start is None else args.frame_start
    last_official = int(info["end_frame"])
    scene_name = info["scene_name"]
    wearer_root = root / f"smpl_camera_wearer_{split}" / args.recording
    scene_path = root / "scene_mesh" / scene_name / f"{scene_name}.obj"
    if not wearer_root.is_dir() or not scene_path.is_file():
        raise FileNotFoundError(f"Missing wearer fits ({wearer_root}) or scene mesh ({scene_path})")

    scene_from_kinect = _load_scene_transform(root, args.recording, scene_name)
    holo_path = root / "calibrations" / args.recording / "cal_trans" / "holo_to_kinect12.json"
    holo_to_kinect = np.asarray(
        json.loads(holo_path.read_text(encoding="utf-8"))["trans"], dtype=np.float64
    )
    pv_stream = _load_pv_stream(root, args.recording)
    scene_mesh = trimesh.load(scene_path, force="mesh", process=False)
    floor_height_scene_y = _floor_height_scene_y(scene_mesh)
    kinematic_info = extract_kinematic_info(
        str(PROJECT_ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    )
    output_dir = args.output_root.resolve() / args.recording
    motion_dir = output_dir / "motions"
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion_entries = []
    scenes = []
    camera_clips = []
    diagnostics = {
        "recording": args.recording,
        "split": split,
        "scene": scene_name,
        "coordinate_chain": "master_kinect -> released_scene_mesh -> ProtoMotions_Z_up",
        "scene_from_kinect": scene_from_kinect.tolist(),
        "proto_from_scene_rotation": TRUMANS_Y_UP_TO_Z_UP.tolist(),
        "floor_height_scene_y_m": floor_height_scene_y,
        "world_z_shift_m": -floor_height_scene_y,
        "clips": [],
    }
    scene_rotation = Rotation.from_matrix(TRUMANS_Y_UP_TO_Z_UP).as_quat()

    wearer_dirs = sorted(wearer_root.glob("body_idx_*"))
    if len(wearer_dirs) != 1:
        raise FileNotFoundError(f"Expected one camera-wearer body directory, got {wearer_dirs}")
    wearer_index = int(wearer_dirs[0].name.rsplit("_", 1)[1])
    wearer_gender = info[f"body_idx_{wearer_index}"].split()[-1].lower()

    clip_starts = (
        args.clip_starts
        if args.clip_starts is not None
        else [first_frame + index * args.clip_stride for index in range(args.num_clips)]
    )
    for clip_index, start in enumerate(clip_starts):
        frame_ids = np.arange(start, start + args.frame_count, dtype=np.int64)
        if int(frame_ids[-1]) > last_official:
            raise ValueError(f"Clip ending at {frame_ids[-1]} exceeds official end {last_official}")
        poses, translations, beta_values = [], [], []
        for frame_id in frame_ids:
            matches = list(wearer_root.glob(f"body_idx_*/results/frame_{frame_id:05d}/000.pkl"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one wearer fit for frame {frame_id:05d}, got {matches}")
            pose, translation, betas = _load_fit(matches[0])
            poses.append(pose)
            translations.append(translation)
            beta_values.append(betas)
        poses = np.stack(poses)
        translations = np.stack(translations)
        beta_values = np.stack(beta_values)
        pelvis_positions = _smpl_pelvis_positions(
            poses, translations, beta_values, wearer_gender
        )
        poses, translations, proto_from_kinect = _transform_motion(
            poses, pelvis_positions, scene_from_kinect, floor_height_scene_y
        )
        motion, actual_fps = convert_amass_to_motion(
            pose_aa=poses,
            amass_trans=translations,
            mocap_fr=args.fps,
            output_fps=args.fps,
            humanoid_type="smpl",
            joint_names=SMPL_BONE_ORDER_NAMES,
            mujoco_joint_names=SMPL_MUJOCO_NAMES,
            kinematic_info=kinematic_info,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fix_height=True,
            compute_ground_contacts=True,
        )
        ungrounded_root = torch.from_numpy(translations).to(
            dtype=motion.rigid_body_pos.dtype
        )
        root_translation_delta = motion.rigid_body_pos[:, 0].cpu() - ungrounded_root
        constant_delta = root_translation_delta[0].expand_as(root_translation_delta)
        if not torch.allclose(
            root_translation_delta, constant_delta, atol=1.0e-5, rtol=0.0
        ) or not torch.allclose(
            root_translation_delta[0, :2], torch.zeros(2), atol=1.0e-5
        ):
            raise ValueError(
                "Official SMPL height fixing must be one constant Z translation"
            )
        grounding_z_offset = float(root_translation_delta[0, 2])
        if pv_stream is not None:
            camera_clip = _camera_clip(
                pv_stream, frame_ids, proto_from_kinect, holo_to_kinect
            )
            camera_clip["world_from_camera"][:, 2, 3] += grounding_z_offset
            camera_clip["grounding_z_offset_m"] = grounding_z_offset
            camera_clip["reference_root"] = motion.rigid_body_pos[:, 0].cpu()
            camera_clips.append(camera_clip)
        clip_name = f"frame_{start:05d}_count_{len(frame_ids):04d}"
        motion_path = motion_dir / f"{clip_name}.motion"
        if motion_path.exists() and not args.overwrite:
            raise FileExistsError(f"{motion_path} exists; pass --overwrite")
        torch.save(motion.to_dict(), motion_path)
        motion_entries.append({"file": f"motions/{motion_path.name}", "fps": actual_fps, "weight": 1.0, "idx": clip_index})
        scenes.append(
            Scene(
                objects=[
                    MeshSceneObject(
                        object_path=str(scene_path),
                        translation=(0.0, 0.0, -floor_height_scene_y),
                        rotation=scene_rotation,
                        options=ObjectOptions(fix_base_link=True),
                    )
                ],
                humanoid_motion_id=clip_index,
            )
        )
        clip_diag = _motion_diagnostics(
            motion,
            scene_mesh,
            kinematic_info.body_names,
            -floor_height_scene_y,
        )
        clip_diag.update(
            {
                "name": clip_name,
                "source_frame_ids": [int(frame_ids[0]), int(frame_ids[-1])],
                "wearer_gender": wearer_gender,
                "root_translation_semantics": "shaped_SMPL_joint_0",
                "median_betas": np.median(beta_values, axis=0).tolist(),
                "proto_from_kinect": proto_from_kinect.tolist(),
                "grounding_z_offset_m": grounding_z_offset,
            }
        )
        diagnostics["clips"].append(clip_diag)

    (output_dir / "motion_lib.yaml").write_text(
        yaml.safe_dump({"motions": motion_entries}, sort_keys=False), encoding="utf-8"
    )
    SceneLib.save_scenes_to_file(scenes, str(output_dir / "scene_lib.pt"), asset_root=str(root))
    # All clips share one static reconstruction. Training with fewer simulator
    # environments than clips must therefore replicate one universal scene and
    # sample motion IDs at episode reset instead of permanently dropping clips.
    universal_scene = Scene(objects=scenes[0].objects, humanoid_motion_id=-1)
    SceneLib.save_scenes_to_file(
        [universal_scene], str(output_dir / "scene_lib_training.pt"), asset_root=str(root)
    )
    if camera_clips:
        torch.save(
            {
                "recording": args.recording,
                "coordinate_system": "ProtoMotions_Z_up",
                "source": pv_stream["pv_info_path"],
                "motions": camera_clips,
            },
            output_dir / "ego_camera.pt",
        )
    (output_dir / "coordinate_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(scenes)} clip(s) under {output_dir}")
    print(json.dumps(diagnostics["clips"], indent=2))


if __name__ == "__main__":
    main()
