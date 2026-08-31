"""Prepare aligned static triangle meshes for simulator collision and observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import gc
from pathlib import Path
import tempfile

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .transforms import TRUMANS_Y_UP_TO_Z_UP, transform_points


def _atomic_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(fd)
    return Path(tmp_name)


@dataclass(frozen=True)
class CollisionMeshConfig:
    """Dataset-configurable static collision-mesh preparation parameters."""

    interaction_margin_m: float = 1.5
    vertical_margin_below_m: float = 1.0
    vertical_margin_above_m: float = 1.5
    weld_tolerance_m: float = 0.002
    min_component_faces: int = 100
    min_component_area_m2: float = 0.01
    target_faces: int = 100_000
    simplification_aggression: int = 7
    dense_pointcloud_samples: int = 50_000
    diagnostic_samples: int = 20_000
    random_seed: int = 0
    remove_dynamic_duplicates: bool = True
    duplicate_object_samples: int = 512
    duplicate_frame_samples: int = 8
    duplicate_detection_threshold_m: float = 0.020
    duplicate_min_overlap_ratio: float = 0.35
    duplicate_carve_distance_m: float = 0.015
    duplicate_carve_aabb_margin_m: float = 0.030
    duplicate_query_workers: int = 1
    max_faces_for_dynamic_duplicate_removal: int = 300_000
    preserve_dynamic_carve_boundaries: bool = True


def _load_aligned_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Expected a non-empty triangle mesh at {path}")
    mesh = mesh.copy()
    mesh.vertices = transform_points(mesh.vertices, TRUMANS_Y_UP_TO_Z_UP)
    return mesh


def _crop_to_interaction(
    mesh: trimesh.Trimesh,
    config: CollisionMeshConfig,
    body_points: np.ndarray | None = None,
    body_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    if body_bounds is not None:
        raw_lo = np.asarray(body_bounds[0], dtype=np.float64).reshape(3)
        raw_hi = np.asarray(body_bounds[1], dtype=np.float64).reshape(3)
        if not np.isfinite(raw_lo).all() or not np.isfinite(raw_hi).all():
            raise ValueError("body_bounds must contain finite XYZ bounds")
    else:
        points = np.asarray(body_points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError(
                "body_points must contain finite world-space XYZ positions"
            )
        raw_lo = points.min(axis=0)
        raw_hi = points.max(axis=0)

    lo = raw_lo - config.interaction_margin_m
    hi = raw_hi + config.interaction_margin_m
    lo[2] = raw_lo[2] - config.vertical_margin_below_m
    hi[2] = raw_hi[2] + config.vertical_margin_above_m

    triangles = mesh.triangles
    # Retain triangles intersecting the swept AABB instead of testing only the
    # centroid, which can cut large wall/floor triangles prematurely.
    face_mask = np.all(triangles.max(axis=1) >= lo, axis=1) & np.all(
        triangles.min(axis=1) <= hi, axis=1
    )
    face_ids = np.flatnonzero(face_mask)
    if len(face_ids) == 0:
        raise ValueError("Human interaction crop does not overlap the scene mesh")
    cropped = mesh.submesh([face_ids], append=True, repair=False)
    return cropped, lo.astype(np.float32), hi.astype(np.float32)


def _remove_invalid_and_duplicate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    finite_vertices = np.isfinite(mesh.vertices).all(axis=1)
    valid = finite_vertices[mesh.faces].all(axis=1)
    mesh.update_faces(valid)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("Mesh contains no finite faces")
    mesh.update_faces(mesh.nondegenerate_faces(height=1.0e-10))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _weld_vertices(mesh: trimesh.Trimesh, tolerance_m: float) -> trimesh.Trimesh:
    if tolerance_m <= 0.0:
        return mesh
    # Trimesh's merge is decimal-grid based. Use the nearest conservative
    # number of decimal digits and report the effective tolerance.
    digits = max(0, int(math.ceil(-math.log10(tolerance_m))))
    welded = mesh.copy()
    # Collision geometry has no UV/normal seam semantics. Requiring matching
    # texture or normal indices would keep coincident scan vertices separated
    # and artificially explode the connected-component count.
    welded.merge_vertices(digits_vertex=digits)
    welded.remove_unreferenced_vertices()
    return welded


def _load_object_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Expected a non-empty dynamic object mesh at {path}")
    return _remove_invalid_and_duplicate_faces(mesh)


def _closest_distance(
    mesh: trimesh.Trimesh, points: np.ndarray, chunk_size: int = 100_000
) -> np.ndarray:
    distances = []
    for start in range(0, len(points), chunk_size):
        _, distance, _ = trimesh.proximity.closest_point(
            mesh, points[start : start + chunk_size]
        )
        distances.append(distance)
    if not distances:
        return np.empty(0, dtype=np.float64)
    return np.nan_to_num(
        np.concatenate(distances), nan=np.inf, posinf=np.inf, neginf=np.inf
    )


def _transform_object_points(
    points: np.ndarray, translation: np.ndarray, rotation_xyzw: np.ndarray
) -> np.ndarray:
    return Rotation.from_quat(rotation_xyzw).apply(points) + translation


def _remove_dynamic_object_duplicates(
    mesh: trimesh.Trimesh,
    dynamic_objects: list[dict] | None,
    config: CollisionMeshConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None, list[dict]]:
    """Detect scan/object overlap and carve only confirmed reference-pose surfaces.

    The complete swept object volume is deliberately not removed: doing so for a
    door or drawer would punch an artificial hole through the surrounding frame.
    """
    if not config.remove_dynamic_duplicates or not dynamic_objects:
        return mesh, None, []

    scene_vertex_tree = cKDTree(np.asarray(mesh.vertices))
    remove_mask = np.zeros(len(mesh.faces), dtype=bool)
    reports = []
    object_mesh_cache: dict[str, trimesh.Trimesh] = {}
    local_sample_cache: dict[str, np.ndarray] = {}
    rng_state = np.random.get_state()
    try:
        for object_index, record in enumerate(dynamic_objects):
            cache_path = record.get("object_cache_path")
            if cache_path is None:
                # Backward-compatible fallback for already materialized records.
                object_records = [record]
            else:
                with np.load(cache_path, allow_pickle=True) as object_cache:
                    object_names = object_cache["names"].tolist()
                    object_mesh_paths = object_cache["mesh_paths"]
                    object_translations = object_cache["translations"]
                    object_rotations = object_cache["rotations_xyzw"]
                    object_records = [
                        {
                            "name": f"{record['name']}:{object_name}",
                            "mesh_path": str(object_mesh_paths[obj_index]),
                            "translations": object_translations[obj_index],
                            "rotations_xyzw": object_rotations[obj_index],
                        }
                        for obj_index, object_name in enumerate(object_names)
                    ]

            for object_offset, object_record in enumerate(object_records):
                object_mesh_path = str(object_record["mesh_path"])
                if object_mesh_path in object_mesh_cache:
                    object_mesh = object_mesh_cache[object_mesh_path]
                else:
                    object_mesh = _load_object_mesh(Path(object_mesh_path))
                    object_mesh_cache[object_mesh_path] = object_mesh

                np.random.seed(
                    config.random_seed + object_index * 1000 + object_offset + 101
                )
                if object_mesh_path in local_sample_cache:
                    local_samples = local_sample_cache[object_mesh_path]
                else:
                    local_samples, _ = trimesh.sample.sample_surface(
                        object_mesh, config.duplicate_object_samples
                    )
                    local_sample_cache[object_mesh_path] = local_samples
                translations = np.asarray(object_record["translations"], dtype=np.float64)
                rotations = np.asarray(object_record["rotations_xyzw"], dtype=np.float64)
                if translations.shape[0] != rotations.shape[0]:
                    raise ValueError(
                        f"{object_record['name']}: translation/rotation length mismatch"
                    )
                print(
                    f"  duplicate scan {object_index + 1}/{len(dynamic_objects)} "
                    f"{object_record['name']}",
                    flush=True,
                )
                frame_ids = np.unique(
                    np.linspace(
                        0,
                        len(translations) - 1,
                        min(config.duplicate_frame_samples, len(translations)),
                        dtype=np.int64,
                    )
                )

                # Vertex-distance screening cheaply identifies the scan pose. The
                # winning frame is then verified against actual scene triangles.
                approximate_ratios = []
                for frame in frame_ids:
                    world_samples = _transform_object_points(
                        local_samples, translations[frame], rotations[frame]
                    )
                    approximate_distance = scene_vertex_tree.query(
                        world_samples, workers=max(1, int(config.duplicate_query_workers))
                    )[0]
                    approximate_ratios.append(
                        float(
                            np.mean(
                                approximate_distance
                                <= config.duplicate_detection_threshold_m * 1.5
                            )
                        )
                    )
                best_frame = int(frame_ids[int(np.argmax(approximate_ratios))])
                world_samples = _transform_object_points(
                    local_samples, translations[best_frame], rotations[best_frame]
                )
                sample_tree = cKDTree(world_samples)
                best_frame_distance = scene_vertex_tree.query(
                    world_samples, workers=max(1, int(config.duplicate_query_workers))
                )[0]
                overlap_ratio = float(
                    np.mean(best_frame_distance <= config.duplicate_detection_threshold_m)
                )
                duplicate = overlap_ratio >= config.duplicate_min_overlap_ratio

                removed_faces = 0
                candidate_faces = 0
                if duplicate:
                    world_vertices = _transform_object_points(
                        np.asarray(object_mesh.vertices),
                        translations[best_frame],
                        rotations[best_frame],
                    )
                    margin = config.duplicate_carve_aabb_margin_m
                    object_lo = world_vertices.min(axis=0) - margin
                    object_hi = world_vertices.max(axis=0) + margin
                    triangles = mesh.triangles
                    candidate_mask = np.all(triangles.max(axis=1) >= object_lo, axis=1)
                    candidate_mask &= np.all(triangles.min(axis=1) <= object_hi, axis=1)
                    candidate_ids = np.flatnonzero(candidate_mask & ~remove_mask)
                    candidate_faces = int(len(candidate_ids))
                    if len(candidate_ids):
                        candidate_triangles = triangles[candidate_ids]
                        candidate_centers = candidate_triangles.mean(axis=1)
                        query_points = np.concatenate(
                            [
                                candidate_centers[:, None],
                                candidate_triangles,
                            ],
                            axis=1,
                        )
                        distances = sample_tree.query(
                            query_points.reshape(-1, 3),
                            workers=max(1, int(config.duplicate_query_workers)),
                        )[0].reshape(len(candidate_ids), 4)
                        object_duplicate_faces = candidate_ids[
                            distances.min(axis=1) <= config.duplicate_carve_distance_m
                        ]
                        # Preserve an upward-facing support surface at the
                        # object's bottom.  Otherwise a laptop bottom resting
                        # on a table is indistinguishable from baked geometry
                        # under a distance-only query and the tabletop is cut.
                        if len(object_duplicate_faces):
                            local_ids = np.searchsorted(candidate_ids, object_duplicate_faces)
                            normals = mesh.face_normals[object_duplicate_faces]
                            centers = candidate_centers[local_ids]
                            object_bottom_z = float(np.quantile(world_vertices[:, 2], 0.02))
                            support_faces = (
                                (normals[:, 2] >= 0.8)
                                & (np.abs(centers[:, 2] - object_bottom_z) <= 0.025)
                            )
                            object_duplicate_faces = object_duplicate_faces[~support_faces]
                        remove_mask[object_duplicate_faces] = True
                        removed_faces = int(len(object_duplicate_faces))

                reports.append(
                    {
                        "name": str(object_record["name"]),
                        "mesh_path": str(object_record["mesh_path"]),
                        "best_reference_frame": best_frame,
                        "overlap_ratio": overlap_ratio,
                        "distance_median_m": float(np.median(best_frame_distance)),
                        "distance_p95_m": float(np.quantile(best_frame_distance, 0.95)),
                        "duplicate_detected": duplicate,
                        "candidate_static_faces": candidate_faces,
                        "removed_static_faces": removed_faces,
                        "distance_method": "sampled_kdtree",
                        "carve_method": "sampled_kdtree",
                    }
                )
    finally:
        np.random.set_state(rng_state)

    removed_ids = np.flatnonzero(remove_mask)
    if len(removed_ids) == 0:
        return mesh, None, reports
    kept_ids = np.flatnonzero(~remove_mask)
    removed_mesh = mesh.submesh([removed_ids], append=True, repair=False)
    deduplicated = mesh.submesh([kept_ids], append=True, repair=False)
    deduplicated = _remove_invalid_and_duplicate_faces(deduplicated)
    return deduplicated, removed_mesh, reports


def _filter_components(
    mesh: trimesh.Trimesh, config: CollisionMeshConfig
) -> tuple[trimesh.Trimesh, list[dict], list[dict]]:
    components = mesh.split(only_watertight=False)
    kept = []
    kept_stats = []
    removed_stats = []
    for component in components:
        stats = {
            "faces": int(len(component.faces)),
            "area_m2": float(component.area),
            "extents_m": np.asarray(component.extents, dtype=float).tolist(),
            "centroid_m": np.asarray(component.centroid, dtype=float).tolist(),
        }
        # Require either meaningful topology or meaningful physical area. This
        # removes floating scan speckles while retaining thin but broad panels.
        keep = (
            len(component.faces) >= config.min_component_faces
            or component.area >= config.min_component_area_m2
        )
        if keep:
            kept.append(component)
            kept_stats.append(stats)
        else:
            removed_stats.append(stats)
    if not kept:
        raise ValueError("Component filtering removed the complete interaction mesh")
    combined = trimesh.util.concatenate(kept)
    combined.remove_unreferenced_vertices()
    return combined, kept_stats, removed_stats


def _simplify(mesh: trimesh.Trimesh, config: CollisionMeshConfig) -> trimesh.Trimesh:
    if config.target_faces <= 0 or len(mesh.faces) <= config.target_faces:
        return mesh.copy()
    try:
        simplified = mesh.simplify_quadric_decimation(
            face_count=config.target_faces,
            aggression=config.simplification_aggression,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Mesh simplification needs fast-simplification. Install "
            "requirements_scene_preprocessing.txt or the scene-preprocess extra."
        ) from error
    simplified = _remove_invalid_and_duplicate_faces(simplified)
    if len(simplified.faces) == 0:
        raise ValueError("Quadric simplification produced an empty mesh")
    return simplified


def _sample_surface(
    mesh: trimesh.Trimesh, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        points, face_ids = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(rng_state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    return np.asarray(points, dtype=np.float32), normals


def _surface_error(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    sample_count: int,
    seed: int,
) -> dict:
    count = max(1_000, int(sample_count))
    source_points, _ = _sample_surface(source, count, seed)
    target_points, _ = _sample_surface(target, count, seed + 1)
    if max(len(source.faces), len(target.faces)) > 200_000:
        # Exact surface proximity on a million-face room can spike memory
        # dramatically inside trimesh's proximity helpers. For large scenes,
        # use a symmetric nearest-neighbour estimate over sampled surface
        # points instead. This is diagnostic-only and keeps preprocessing
        # stable.
        target_tree = cKDTree(target_points)
        source_tree = cKDTree(source_points)
        source_to_target = target_tree.query(source_points, workers=1)[0]
        target_to_source = source_tree.query(target_points, workers=1)[0]
        error_method = "sampled_kdtree"
    else:
        # Query the actual opposing triangle surface. Point-set nearest
        # neighbours mostly measure sample spacing on a large room and greatly
        # overstate the geometric error of decimation.
        _, source_to_target, _ = trimesh.proximity.closest_point(target, source_points)
        _, target_to_source, _ = trimesh.proximity.closest_point(source, target_points)
        error_method = "exact_surface"
    symmetric = np.concatenate([source_to_target, target_to_source])
    return {
        "surface_error_method": error_method,
        "approx_surface_distance_median_m": float(np.median(symmetric)),
        "approx_surface_distance_p95_m": float(np.quantile(symmetric, 0.95)),
        "approx_surface_distance_max_m": float(symmetric.max()),
    }


def _component_summary(stats: list[dict], top_k: int = 50) -> dict:
    """Keep diagnostics useful without writing tens of thousands of fragments."""
    ordered = sorted(stats, key=lambda item: item["area_m2"], reverse=True)
    return {
        "count": len(stats),
        "faces": int(sum(item["faces"] for item in stats)),
        "area_m2": float(sum(item["area_m2"] for item in stats)),
        "largest_by_area": ordered[:top_k],
        "truncated": len(ordered) > top_k,
    }


def write_static_collision_usda(mesh: trimesh.Trimesh, path: Path) -> None:
    """Write a fixed-base triangle collider compatible with ``RigidObjectCfg``.

    The ProtoMotions IsaacLab scene builder represents every scene slot as a
    ``RigidObjectCfg``.  Therefore the room root needs a kinematic
    ``RigidBodyAPI`` even though it never moves; omitting it makes IsaacLab
    reject the asset while activating contact sensors.  Kinematic + fixed-base
    has static collision semantics and does not add a simulated dynamic body.
    """
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    except ImportError as error:
        raise RuntimeError(
            "USD export requires pxr; run with the IsaacLab/Isaac Sim Python."
        ) from error

    temp_path = _atomic_temp_path(path)
    try:
        stage = Usd.Stage.CreateNew(str(temp_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Root")
        usd_mesh = UsdGeom.Mesh.Define(stage, "/Root/CollisionMesh")
        usd_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*vertex) for vertex in mesh.vertices])
        )
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mesh.faces)))
        usd_mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray(mesh.faces.reshape(-1).tolist())
        )
        collision_api = UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim())
        mesh_collision_api.CreateApproximationAttr("none")
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid_api.CreateKinematicEnabledAttr(True)
        stage.SetDefaultPrim(root.GetPrim())
        stage.Save()
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def prepare_static_collision_mesh(
    source_mesh_path: Path,
    body_points: np.ndarray | None,
    output_stem: Path,
    config: CollisionMeshConfig,
    *,
    body_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    dynamic_objects: list[dict] | None = None,
    write_usd: bool = True,
) -> dict:
    """Prepare OBJ/USD collider, dense surface cache and JSON diagnostics."""
    source = _load_aligned_mesh(source_mesh_path)
    source_faces = int(len(source.faces))
    cropped, crop_min, crop_max = _crop_to_interaction(
        source, config, body_points=body_points, body_bounds=body_bounds
    )
    cropped_faces = int(len(cropped.faces))
    del source
    gc.collect()
    cleaned = _remove_invalid_and_duplicate_faces(cropped)
    del cropped
    gc.collect()
    cleaned = _weld_vertices(cleaned, config.weld_tolerance_m)
    pre_dedup_faces = len(cleaned.faces)
    # Dynamic duplicates must be carved from the unsimplified scan.  Carving
    # after decimation makes each selected triangle cover a much larger surface
    # area, producing oversized holes around doors, screens and appliance
    # panels.  The removal routine already restricts expensive face queries to
    # each object's local AABB, so scene face count is not a correctness reason
    # to change operation order.
    deduplicated, duplicate_mesh, duplicate_reports = _remove_dynamic_object_duplicates(
        cleaned, dynamic_objects, config
    )
    for item in duplicate_reports:
        item["stage"] = "before_simplification"
    duplicate_removed_faces = int(
        len(duplicate_mesh.faces) if duplicate_mesh is not None else 0
    )
    deduplicated_faces = int(len(deduplicated.faces))
    del cleaned
    gc.collect()
    deduplicated, kept_components, removed_components = _filter_components(
        deduplicated, config
    )
    filtered_faces = int(len(deduplicated.faces))
    gc.collect()
    carve_boundaries_preserved = bool(
        duplicate_removed_faces and config.preserve_dynamic_carve_boundaries
    )
    if carve_boundaries_preserved:
        # fast-simplification 0.1.x cannot preserve open boundaries.  Running
        # unconstrained QEM after carving dynamic scan duplicates can collapse
        # or bridge those openings, undoing the high-resolution removal.  Keep
        # the cropped collider at native resolution until a boundary-constrained
        # simplifier is available.
        simplified = deduplicated.copy()
    else:
        simplified = _simplify(deduplicated, config)
    post_duplicate_removed_faces = 0
    collision_faces = int(len(simplified.faces))
    if carve_boundaries_preserved:
        # ``simplified`` is an exact copy here.  Comparing two independently
        # sampled point clouds would report their sample spacing as geometric
        # error (several centimetres on a large scene), despite identical
        # vertices and faces.
        surface_error = {
            "surface_error_method": "identity_copy",
            "approx_surface_distance_median_m": 0.0,
            "approx_surface_distance_p95_m": 0.0,
            "approx_surface_distance_max_m": 0.0,
        }
    else:
        surface_error = _surface_error(
            deduplicated,
            simplified,
            config.diagnostic_samples,
            config.random_seed + 10,
        )
    del deduplicated
    gc.collect()

    obj_path = output_stem.with_suffix(".obj")
    usd_path = output_stem.with_suffix(".usda")
    pointcloud_path = output_stem.with_suffix(".pointcloud.npz")
    duplicate_path = output_stem.with_suffix(".duplicate_faces.obj")
    diagnostics_path = output_stem.with_suffix(".json")
    obj_tmp = _atomic_temp_path(obj_path)
    try:
        simplified.export(obj_tmp)
        obj_tmp.replace(obj_path)
    finally:
        if obj_tmp.exists():
            obj_tmp.unlink()
    if duplicate_mesh is not None and len(duplicate_mesh.faces):
        duplicate_tmp = _atomic_temp_path(duplicate_path)
        try:
            duplicate_mesh.export(duplicate_tmp)
            duplicate_tmp.replace(duplicate_path)
        finally:
            if duplicate_tmp.exists():
                duplicate_tmp.unlink()
    elif duplicate_path.exists():
        duplicate_path.unlink()
    if write_usd:
        write_static_collision_usda(simplified, usd_path)
    points, normals = _sample_surface(
        simplified, config.dense_pointcloud_samples, config.random_seed
    )
    del simplified
    gc.collect()
    pointcloud_tmp = _atomic_temp_path(pointcloud_path)
    try:
        np.savez_compressed(
            pointcloud_tmp,
            points=points,
            normals=normals,
            crop_bounds_min=crop_min,
            crop_bounds_max=crop_max,
        )
        pointcloud_tmp.replace(pointcloud_path)
    finally:
        if pointcloud_tmp.exists():
            pointcloud_tmp.unlink()

    report = {
        "source_mesh": str(source_mesh_path),
        "collision_obj": str(obj_path),
        "collision_usd": str(usd_path) if write_usd else None,
        "pointcloud": str(pointcloud_path),
        "source_faces": source_faces,
        "cropped_faces": cropped_faces,
        "pre_dedup_faces": int(pre_dedup_faces),
        "duplicate_removed_faces": duplicate_removed_faces,
        "post_simplification_duplicate_removed_faces": post_duplicate_removed_faces,
        "dynamic_carve_boundaries_preserved": carve_boundaries_preserved,
        "deduplicated_faces": deduplicated_faces,
        "cleaned_faces": filtered_faces,
        "collision_faces": collision_faces,
        "duplicate_faces_obj": (
            str(duplicate_path)
            if duplicate_mesh is not None and len(duplicate_mesh.faces)
            else None
        ),
        "dynamic_duplicate_diagnostics": duplicate_reports,
        "crop_bounds_min": crop_min.tolist(),
        "crop_bounds_max": crop_max.tolist(),
        "kept_component_summary": _component_summary(kept_components),
        "removed_component_summary": _component_summary(removed_components),
        "config": asdict(config),
        **surface_error,
    }
    diagnostics_tmp = _atomic_temp_path(diagnostics_path)
    try:
        diagnostics_tmp.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        diagnostics_tmp.replace(diagnostics_path)
    finally:
        if diagnostics_tmp.exists():
            diagnostics_tmp.unlink()
    return report
