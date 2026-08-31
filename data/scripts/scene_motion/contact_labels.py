"""Automatic reference contact labels for aligned human-scene motion."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

import numpy as np
import torch
import trimesh
from scipy.ndimage import maximum_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".tmp"
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

def _floats(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    return np.fromstring(value, sep=" ", dtype=np.float32)


def load_body_collision_samples(
    mjcf_path: Path,
    samples_along_capsule: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate every SMPL collider by local spheres.

    Returns padded local centers/radii and a validity mask with shapes
    ``(B,S,3)``, ``(B,S)`` and ``(B,S)``.
    """
    root = ET.parse(mjcf_path).getroot()
    samples_by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        geom = body.find("geom")
        if name is None or geom is None:
            continue
        geom_type = geom.get("type", "sphere")
        if geom_type == "capsule":
            fromto = _floats(geom.get("fromto"), (0, 0, 0, 0, 0, 0))
            radius = float(_floats(geom.get("size"), (0.03,))[0])
            alpha = np.linspace(0.0, 1.0, samples_along_capsule, dtype=np.float32)
            centers = fromto[:3] + alpha[:, None] * (fromto[3:] - fromto[:3])
            radii = np.full(len(centers), radius, dtype=np.float32)
        elif geom_type == "box":
            center = _floats(geom.get("pos"), (0, 0, 0))
            half = _floats(geom.get("size"), (0.03, 0.03, 0.03))
            # Sample all six actual box faces. The former six-sphere cover
            # rounded pelvis/hand/foot boxes and disagreed with PhysX at faces
            # and corners. A 3x3 grid per face preserves the MJCF collider.
            grid = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
            offsets = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    other = [dim for dim in range(3) if dim != axis]
                    for first in grid:
                        for second in grid:
                            offset = np.zeros(3, dtype=np.float32)
                            offset[axis] = sign * half[axis]
                            offset[other[0]] = first * half[other[0]]
                            offset[other[1]] = second * half[other[1]]
                            offsets.append(offset)
            centers = center[None] + np.asarray(offsets, dtype=np.float32)
            radii = np.zeros(len(centers), dtype=np.float32)
        else:
            center = _floats(geom.get("pos"), (0, 0, 0))
            radius = float(_floats(geom.get("size"), (0.03,))[0])
            centers = center[None]
            radii = np.asarray([radius], dtype=np.float32)
        samples_by_name[name] = (centers.astype(np.float32), radii)

    missing = [name for name in SMPL_MUJOCO_NAMES if name not in samples_by_name]
    if missing:
        raise ValueError(f"MJCF has no collision geometry for: {missing}")
    max_samples = max(len(samples_by_name[name][0]) for name in SMPL_MUJOCO_NAMES)
    centers = np.zeros((len(SMPL_MUJOCO_NAMES), max_samples, 3), dtype=np.float32)
    radii = np.zeros((len(SMPL_MUJOCO_NAMES), max_samples), dtype=np.float32)
    valid = np.zeros((len(SMPL_MUJOCO_NAMES), max_samples), dtype=bool)
    for body_id, name in enumerate(SMPL_MUJOCO_NAMES):
        body_centers, body_radii = samples_by_name[name]
        count = len(body_centers)
        centers[body_id, :count] = body_centers
        radii[body_id, :count] = body_radii
        valid[body_id, :count] = True
    return centers, radii, valid


def transform_body_spheres(
    body_pos: np.ndarray,
    body_rot_xyzw: np.ndarray,
    local_centers: np.ndarray,
) -> np.ndarray:
    frames, bodies = body_pos.shape[:2]
    samples = local_centers.shape[1]
    expanded = np.broadcast_to(local_centers[None], (frames, bodies, samples, 3))
    rotation_matrices = Rotation.from_quat(body_rot_xyzw.reshape(-1, 4)).as_matrix()
    rotation_matrices = rotation_matrices.reshape(frames, bodies, 3, 3)
    rotated = np.einsum("tbij,tbsj->tbsi", rotation_matrices, expanded)
    return rotated + body_pos[:, :, None]


def _query_static_mesh(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    radii: np.ndarray,
    valid: np.ndarray,
    broadphase_margin_m: float | None = None,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, bodies, samples = centers.shape[:3]
    flat = centers.reshape(-1, 3)
    if broadphase_margin_m is None:
        candidate = valid
    else:
        # One exact body-anchor query safely rejects most body/frame pairs. The
        # bound encloses every local collider sample, so contacts inside the
        # requested margin cannot be discarded by this broad phase.
        valid_count = np.maximum(valid.sum(axis=-1, keepdims=True), 1)
        anchor = (centers * valid[..., None]).sum(axis=2) / valid_count
        sample_reach = np.linalg.norm(centers - anchor[:, :, None], axis=-1) + radii
        sample_reach = np.where(valid, sample_reach, 0.0).max(axis=-1)
        flat_anchor = anchor.reshape(-1, 3)
        anchor_distance_parts = []
        for start in range(0, len(flat_anchor), chunk_size):
            _, distance, _ = trimesh.proximity.closest_point(
                mesh, flat_anchor[start : start + chunk_size]
            )
            anchor_distance_parts.append(distance)
        anchor_distance = np.nan_to_num(
            np.concatenate(anchor_distance_parts).reshape(frames, bodies),
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        )
        body_candidate = (
            anchor_distance - sample_reach <= float(broadphase_margin_m)
        )
        candidate = valid & body_candidate[..., None]

    candidate_ids = np.flatnonzero(candidate.reshape(-1))
    flat_closest = np.zeros((len(flat), 3), dtype=np.float32)
    flat_distance = np.full(len(flat), np.inf, dtype=np.float32)
    flat_face = np.full(len(flat), -1, dtype=np.int64)
    for start in range(0, len(candidate_ids), chunk_size):
        ids = candidate_ids[start : start + chunk_size]
        closest, distance, face_id = trimesh.proximity.closest_point(
            mesh, flat[ids]
        )
        flat_closest[ids] = closest.astype(np.float32)
        flat_distance[ids] = distance.astype(np.float32)
        flat_face[ids] = face_id
    closest = flat_closest.reshape(frames, bodies, samples, 3)
    raw_distance = flat_distance.reshape(frames, bodies, samples)
    raw_distance = np.nan_to_num(raw_distance, nan=np.inf, posinf=np.inf, neginf=np.inf)
    face_ids = flat_face.reshape(frames, bodies, samples)
    signed = np.where(valid, raw_distance - radii, np.inf)
    sample_id = np.argmin(signed, axis=-1)
    distance = np.take_along_axis(signed, sample_id[..., None], axis=-1)[..., 0]
    point = np.take_along_axis(closest, sample_id[..., None, None], axis=2)[..., 0, :]
    selected_face = np.take_along_axis(face_ids, sample_id[..., None], axis=-1)[..., 0]
    safe_face = np.clip(selected_face, 0, max(len(mesh.faces) - 1, 0))
    normal = mesh.face_normals[safe_face]
    normal[~np.isfinite(distance)] = 0.0
    return distance.astype(np.float32), point.astype(np.float32), normal.astype(np.float32)


def _load_collision_pointcloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Empty collision pointcloud: {path}")
    if normals.shape != points.shape:
        normals = np.zeros_like(points, dtype=np.float32)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    return points, normals


def _query_static_pointcloud(
    points: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    valid: np.ndarray,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate nearest-surface query against sampled collision mesh points.

    This is intentionally used for large offline label generation.  Exact
    triangle proximity on a 100k-face room mesh can allocate very large
    intermediate arrays in trimesh/rtree, while a KD-tree over the same
    collision mesh's dense surface samples gives stable memory usage.
    """
    frames, bodies, samples = centers.shape[:3]
    flat = centers.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    candidate_ids = np.flatnonzero(flat_valid)
    tree = cKDTree(points)

    flat_distance = np.full(len(flat), np.inf, dtype=np.float32)
    flat_point = np.zeros((len(flat), 3), dtype=np.float32)
    flat_normal = np.zeros((len(flat), 3), dtype=np.float32)
    for start in range(0, len(candidate_ids), chunk_size):
        ids = candidate_ids[start : start + chunk_size]
        distance, point_id = tree.query(flat[ids], k=1, workers=1)
        point_id = np.asarray(point_id, dtype=np.int64)
        flat_distance[ids] = np.asarray(distance, dtype=np.float32)
        flat_point[ids] = points[point_id]
        flat_normal[ids] = normals[point_id]

    signed = flat_distance.reshape(frames, bodies, samples) - radii
    signed = np.where(valid, signed, np.inf)
    point = flat_point.reshape(frames, bodies, samples, 3)
    normal = flat_normal.reshape(frames, bodies, samples, 3)
    sample_id = np.argmin(signed, axis=-1)
    distance = np.take_along_axis(signed, sample_id[..., None], axis=-1)[..., 0]
    selected_point = np.take_along_axis(
        point, sample_id[..., None, None], axis=2
    )[..., 0, :]
    selected_normal = np.take_along_axis(
        normal, sample_id[..., None, None], axis=2
    )[..., 0, :]
    invalid = ~np.isfinite(distance)
    selected_point[invalid] = 0.0
    selected_normal[invalid] = 0.0
    return (
        distance.astype(np.float32),
        selected_point.astype(np.float32),
        selected_normal.astype(np.float32),
    )


def _query_dynamic_mesh(
    mesh: trimesh.Trimesh,
    centers_world: np.ndarray,
    radii: np.ndarray,
    valid: np.ndarray,
    translation: np.ndarray,
    rotation_xyzw: np.ndarray,
    broadphase_margin_m: float,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query a moving object mesh, using its local AABB as a broad phase."""
    frames, bodies, samples = centers_world.shape[:3]
    if len(translation) != frames or len(rotation_xyzw) != frames:
        raise ValueError(
            "Dynamic object trajectory length does not match the human motion: "
            f"{len(translation)}/{len(rotation_xyzw)} != {frames}"
        )
    rotation = Rotation.from_quat(rotation_xyzw).as_matrix()
    relative = centers_world - translation[:, None, None]
    local = np.einsum("tji,tbsj->tbsi", rotation, relative)

    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    outside_vector = np.maximum(bounds[0] - local, 0.0)
    outside_vector += np.maximum(local - bounds[1], 0.0)
    outside_distance = np.linalg.norm(outside_vector, axis=-1)
    candidate = valid & (
        outside_distance - radii <= float(broadphase_margin_m)
    )

    flat_local = local.reshape(-1, 3)
    flat_candidate = candidate.reshape(-1)
    candidate_ids = np.flatnonzero(flat_candidate)
    flat_distance = np.full(len(flat_local), np.inf, dtype=np.float32)
    flat_point = np.zeros((len(flat_local), 3), dtype=np.float32)
    flat_face = np.full(len(flat_local), -1, dtype=np.int64)
    for start in range(0, len(candidate_ids), chunk_size):
        ids = candidate_ids[start : start + chunk_size]
        closest, distance, face_id = trimesh.proximity.closest_point(mesh, flat_local[ids])
        distance = np.nan_to_num(distance, nan=np.inf, posinf=np.inf, neginf=np.inf)
        flat_distance[ids] = distance.astype(np.float32)
        flat_point[ids] = closest.astype(np.float32)
        flat_face[ids] = face_id

    signed = flat_distance.reshape(frames, bodies, samples) - radii
    signed = np.where(valid, signed, np.inf)
    point_local = flat_point.reshape(frames, bodies, samples, 3)
    face_ids = flat_face.reshape(frames, bodies, samples)
    sample_id = np.argmin(signed, axis=-1)
    distance = np.take_along_axis(signed, sample_id[..., None], axis=-1)[..., 0]
    chosen_local = np.take_along_axis(
        point_local, sample_id[..., None, None], axis=2
    )[..., 0, :]
    chosen_face = np.take_along_axis(
        face_ids, sample_id[..., None], axis=-1
    )[..., 0]
    safe_face = np.clip(chosen_face, 0, max(len(mesh.faces) - 1, 0))
    normal_local = np.asarray(mesh.face_normals[safe_face], dtype=np.float32)
    point_world = np.einsum("tij,tbj->tbi", rotation, chosen_local)
    point_world += translation[:, None]
    normal_world = np.einsum("tij,tbj->tbi", rotation, normal_local)
    invalid = ~np.isfinite(distance)
    point_world[invalid] = 0.0
    normal_world[invalid] = 0.0
    return (
        distance.astype(np.float32),
        point_world.astype(np.float32),
        normal_world.astype(np.float32),
    )


def _dilate_contacts(contact: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        return contact
    size = frames * 2 + 1
    return maximum_filter1d(contact.astype(np.uint8), size=size, axis=0, mode="nearest").astype(bool)


def _assemble_contact_labels(
    *,
    object_names: list[str],
    target_distances: list[np.ndarray],
    source_distances: list[np.ndarray],
    target_points: list[np.ndarray],
    target_normals: list[np.ndarray],
    source_points: list[np.ndarray],
    source_normals: list[np.ndarray],
    contact_threshold_m: float,
    compatibility_threshold_m: float,
    physics_validation_threshold_m: float,
    include_dynamic_contacts_in_training: bool,
    temporal_dilation_frames: int,
    geometry_source: str,
) -> dict[str, np.ndarray]:
    if not target_distances:
        raise ValueError("Contact geometry contains no collision objects")
    target_distance = np.stack(target_distances, axis=-1)
    source_distance = np.stack(source_distances, axis=-1)
    target_contact = target_distance <= contact_threshold_m
    source_contact = source_distance <= contact_threshold_m
    target_physics_contact = target_distance <= physics_validation_threshold_m
    intended_contact = _dilate_contacts(source_contact, temporal_dilation_frames)
    target_compatible = target_distance <= compatibility_threshold_m
    training_object_mask = np.asarray(
        [name.startswith("__static_") for name in object_names], dtype=bool
    )
    if include_dynamic_contacts_in_training:
        training_object_mask[:] = True
    training_contact = (
        intended_contact
        & target_compatible
        & training_object_mask[None, None, :]
    )

    target_nearest_object = np.argmin(target_distance, axis=-1)
    target_point_stack = np.stack(target_points, axis=2)
    target_normal_stack = np.stack(target_normals, axis=2)
    target_nearest_point = np.take_along_axis(
        target_point_stack, target_nearest_object[..., None, None], axis=2
    )[..., 0, :]
    target_nearest_normal = np.take_along_axis(
        target_normal_stack, target_nearest_object[..., None, None], axis=2
    )[..., 0, :]

    source_nearest_object = np.argmin(source_distance, axis=-1)
    source_point_stack = np.stack(source_points, axis=2)
    source_normal_stack = np.stack(source_normals, axis=2)
    source_nearest_point = np.take_along_axis(
        source_point_stack, source_nearest_object[..., None, None], axis=2
    )[..., 0, :]
    source_nearest_normal = np.take_along_axis(
        source_normal_stack, source_nearest_object[..., None, None], axis=2
    )[..., 0, :]
    return {
        "object_names": np.asarray(object_names, dtype=str),
        "geometry_source": np.asarray(geometry_source),
        "source_signed_distance": source_distance.astype(np.float16),
        "target_signed_distance": target_distance.astype(np.float16),
        "source_contact": source_contact,
        "target_contact": target_contact,
        "target_physics_contact": target_physics_contact,
        "intended_contact": intended_contact,
        "target_compatible": target_compatible,
        "training_object_mask": training_object_mask,
        "training_contact": training_contact,
        "source_nearest_object_id": source_nearest_object.astype(np.int16),
        "source_nearest_point": source_nearest_point.astype(np.float32),
        "source_nearest_normal": source_nearest_normal.astype(np.float32),
        "target_nearest_object_id": target_nearest_object.astype(np.int16),
        "target_nearest_point": target_nearest_point.astype(np.float32),
        "target_nearest_normal": target_nearest_normal.astype(np.float32),
        "nearest_object_id": target_nearest_object.astype(np.int16),
        "nearest_point": target_nearest_point.astype(np.float32),
        "nearest_normal": target_nearest_normal.astype(np.float32),
        "contact_threshold_m": np.asarray(contact_threshold_m, dtype=np.float32),
        "compatibility_threshold_m": np.asarray(
            compatibility_threshold_m, dtype=np.float32
        ),
        "physics_validation_threshold_m": np.asarray(
            physics_validation_threshold_m, dtype=np.float32
        ),
    }


def generate_collision_mesh_contact_labels(
    *,
    collision_mesh_path: Path,
    object_cache_path: Path,
    source_body_pos: np.ndarray,
    target_motion: dict,
    mjcf_path: Path,
    contact_threshold_m: float = 0.025,
    compatibility_threshold_m: float = 0.080,
    physics_validation_threshold_m: float = 0.002,
    include_dynamic_contacts_in_training: bool = False,
    compute_dynamic_contacts: bool = False,
    temporal_dilation_frames: int = 2,
) -> dict[str, np.ndarray]:
    """Generate labels from the exact meshes used by the collision pipeline.

    The static mesh is already aligned to the simulator's Z-up world. Dynamic
    object meshes stay in their released local frame and are queried under the
    cached per-frame object transform.
    """
    target_pos = np.asarray(target_motion["rigid_body_pos"], dtype=np.float32)
    target_rot = np.asarray(target_motion["rigid_body_rot"], dtype=np.float32)
    source_body_pos = np.asarray(source_body_pos, dtype=np.float32)
    if source_body_pos.shape != target_pos.shape:
        raise ValueError(f"source {source_body_pos.shape} != target {target_pos.shape}")

    local_centers, local_radii, local_valid = load_body_collision_samples(mjcf_path)
    frames = len(target_pos)
    valid = np.broadcast_to(local_valid[None], (frames,) + local_valid.shape)
    radii = np.broadcast_to(local_radii[None], (frames,) + local_radii.shape)
    target_centers = transform_body_spheres(target_pos, target_rot, local_centers)
    source_centers = transform_body_spheres(source_body_pos, target_rot, local_centers)

    object_names = ["__static_collision_mesh__"]
    pointcloud_path = collision_mesh_path.with_suffix(".pointcloud.npz")
    static_query_source = "static_collision_pointcloud"
    if pointcloud_path.is_file():
        static_points, static_normals = _load_collision_pointcloud(pointcloud_path)
        target_static = _query_static_pointcloud(
            static_points,
            static_normals,
            target_centers,
            radii,
            valid,
        )
        source_static = _query_static_pointcloud(
            static_points,
            static_normals,
            source_centers,
            radii,
            valid,
        )
    else:
        static_mesh = trimesh.load(collision_mesh_path, force="mesh", process=False)
        if isinstance(static_mesh, trimesh.Scene):
            static_mesh = trimesh.util.concatenate(tuple(static_mesh.geometry.values()))
        if not isinstance(static_mesh, trimesh.Trimesh) or len(static_mesh.faces) == 0:
            raise ValueError(f"Empty collision mesh: {collision_mesh_path}")
        static_query_source = "static_collision_mesh"
        target_static = _query_static_mesh(
            static_mesh,
            target_centers,
            radii,
            valid,
            broadphase_margin_m=compatibility_threshold_m,
        )
        source_static = _query_static_mesh(
            static_mesh,
            source_centers,
            radii,
            valid,
            broadphase_margin_m=compatibility_threshold_m,
        )
    target_distances = [target_static[0]]
    source_distances = [source_static[0]]
    target_points = [target_static[1]]
    target_normals = [target_static[2]]
    source_points = [source_static[1]]
    source_normals = [source_static[2]]

    if include_dynamic_contacts_in_training:
        compute_dynamic_contacts = True
    geometry_source = static_query_source
    if compute_dynamic_contacts:
        with np.load(object_cache_path) as objects:
            names = objects["names"].tolist()
            mesh_paths = objects["mesh_paths"].tolist()
            translations = np.asarray(objects["translations"], dtype=np.float32)
            rotations = np.asarray(objects["rotations_xyzw"], dtype=np.float32)
        for object_id, (name, mesh_path) in enumerate(zip(names, mesh_paths)):
            object_mesh = trimesh.load(Path(mesh_path), force="mesh", process=False)
            if isinstance(object_mesh, trimesh.Scene):
                object_mesh = trimesh.util.concatenate(tuple(object_mesh.geometry.values()))
            if not isinstance(object_mesh, trimesh.Trimesh) or len(object_mesh.faces) == 0:
                raise ValueError(f"Empty dynamic object mesh: {mesh_path}")
            target_dynamic = _query_dynamic_mesh(
                object_mesh,
                target_centers,
                radii,
                valid,
                translations[object_id],
                rotations[object_id],
                broadphase_margin_m=compatibility_threshold_m,
            )
            source_dynamic = _query_dynamic_mesh(
                object_mesh,
                source_centers,
                radii,
                valid,
                translations[object_id],
                rotations[object_id],
                broadphase_margin_m=compatibility_threshold_m,
            )
            object_names.append(str(name))
            target_distances.append(target_dynamic[0])
            source_distances.append(source_dynamic[0])
            target_points.append(target_dynamic[1])
            target_normals.append(target_dynamic[2])
            source_points.append(source_dynamic[1])
            source_normals.append(source_dynamic[2])
        geometry_source = f"{static_query_source}+dynamic_source_meshes"

    # Broad-phase rejects are known to be farther than the compatibility
    # margin, but their exact distance is intentionally not computed. Store a
    # finite truncated value so downstream tensor code never receives inf.
    distance_truncation_m = float(compatibility_threshold_m) + 0.001
    source_distances = [
        np.minimum(
            np.nan_to_num(
                value,
                nan=distance_truncation_m,
                posinf=distance_truncation_m,
                neginf=-distance_truncation_m,
            ),
            distance_truncation_m,
        )
        for value in source_distances
    ]
    target_distances = [
        np.minimum(
            np.nan_to_num(
                value,
                nan=distance_truncation_m,
                posinf=distance_truncation_m,
                neginf=-distance_truncation_m,
            ),
            distance_truncation_m,
        )
        for value in target_distances
    ]
    labels = _assemble_contact_labels(
        object_names=object_names,
        target_distances=target_distances,
        source_distances=source_distances,
        target_points=target_points,
        target_normals=target_normals,
        source_points=source_points,
        source_normals=source_normals,
        contact_threshold_m=contact_threshold_m,
        compatibility_threshold_m=compatibility_threshold_m,
        physics_validation_threshold_m=physics_validation_threshold_m,
        include_dynamic_contacts_in_training=include_dynamic_contacts_in_training,
        temporal_dilation_frames=temporal_dilation_frames,
        geometry_source=geometry_source,
    )
    labels["distance_truncation_m"] = np.asarray(
        distance_truncation_m, dtype=np.float32
    )
    return labels


def save_contact_labels(path: Path, labels: dict[str, np.ndarray]) -> dict:
    _atomic_write(
        path,
        lambda temp_path: np.savez_compressed(
            temp_path, schema_version=np.asarray(2, dtype=np.int32), **labels
        ),
    )
    intended = labels["intended_contact"]
    target = labels["target_contact"]
    training = labels["training_contact"]
    union = training.any(axis=-1)
    overlap = intended & target
    return {
        "path": str(path),
        "intended_contact_fraction": float(intended.mean()),
        "training_contact_fraction": float(training.mean()),
        "training_body_frame_fraction": float(union.mean()),
        "training_retained_of_intended": float(
            training.sum() / max(intended.sum(), 1)
        ),
        "target_recall_of_intended": float(overlap.sum() / max(intended.sum(), 1)),
    }


def inject_motion_contact_union(motion_path: Path, labels: dict[str, np.ndarray]) -> None:
    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    training_union = torch.from_numpy(
        labels.get("training_contact", labels["intended_contact"]).any(axis=-1)
    )
    if training_union.shape != motion["rigid_body_pos"].shape[:2]:
        raise ValueError(
            f"Contact union {training_union.shape} does not match motion bodies "
            f"{motion['rigid_body_pos'].shape[:2]}"
        )
    motion["rigid_body_contacts"] = training_union.to(torch.bool)
    _atomic_write(motion_path, lambda temp_path: torch.save(motion, temp_path))
