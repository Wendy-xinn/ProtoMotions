#!/usr/bin/env python3
"""Inspect one TRUMANS conversion in a shared, browser-based world frame.

This is a data-validation viewer, not a simulator.  It overlays:

* the released TRUMANS joints after the declared Y-up -> Z-up transform;
* the fixed-neutral SMPL humanoid produced by preprocessing;
* the released room mesh and occupancy voxels;
* the converted dynamic-object trajectories.
* the fitted primitive collision proxy and generated contact labels.

Run with the ``crisp`` environment because it already owns the Viser stack::

    /home/wenxin/miniconda3/envs/crisp/bin/python \
      examples/visualize_trumans_preprocessing.py --split train --clip-index 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
import yaml
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.scene_motion.transforms import TRUMANS_Y_UP_TO_Z_UP, transform_points
from data.scripts.scene_motion.trumans_adapter import TrumansAdapter
from data.smpl.smpl_joint_names import SMPL_MUJOCO_NAMES
from examples.visualize_smpl_motion import (
    REGION_COLORS,
    SMPL_GEOM_DEFS,
    SKELETONS,
    _build_body_meshes,
    create_skeleton_meshes,
    quat_xyzw_to_wxyz,
)


def _resolve(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing through the descriptors stage first."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_mesh(path: Path, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.visual.face_colors = color
    return mesh


def _sample_scene_surface(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    """Sample an intact room surface instead of displaying disconnected faces."""
    # Fixed seed makes screenshots and visual comparisons reproducible without
    # altering NumPy's process-global random state.
    state = np.random.get_state()
    try:
        np.random.seed(0)
        points, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float32)


def _error_mesh(source: np.ndarray, target: np.ndarray) -> trimesh.Trimesh | None:
    pieces = []
    for p0, p1 in zip(source, target):
        if np.linalg.norm(p1 - p0) > 1.0e-5:
            pieces.append(trimesh.creation.cylinder(radius=0.0025, sections=6, segment=[p0, p1]))
    if not pieces:
        return None
    mesh = trimesh.util.concatenate(pieces)
    mesh.visual.face_colors = [245, 40, 40, 255]
    return mesh


def _wxyz_align_z(direction: np.ndarray) -> np.ndarray:
    """Return a WXYZ quaternion rotating local +Z onto ``direction``."""
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    unit = direction / norm
    dot = float(unit[2])
    if dot < -1.0 + 1.0e-7:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    w = np.sqrt((1.0 + dot) * 0.5)
    xyz = np.array([-unit[1], unit[0], 0.0], dtype=np.float32) / (2.0 * w)
    return np.asarray([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)


def _occupancy_points(record: dict, max_points: int) -> np.ndarray | None:
    path_value = record.get("occupancy")
    bounds_min = record.get("occupancy_bounds_min")
    bounds_max = record.get("occupancy_bounds_max")
    if not path_value or bounds_min is None or bounds_max is None:
        return None
    occupancy = np.load(path_value, mmap_mode="r")
    indices = np.argwhere(occupancy)
    if len(indices) == 0:
        return None
    if len(indices) > max_points:
        sample_ids = np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)
        indices = indices[sample_ids]
    lo = np.asarray(bounds_min, dtype=np.float32)
    hi = np.asarray(bounds_max, dtype=np.float32)
    points_source = lo + (indices.astype(np.float32) + 0.5) / np.asarray(occupancy.shape) * (hi - lo)
    return transform_points(points_source, TRUMANS_Y_UP_TO_Z_UP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "data/yaml_files/trumans_scene_motion.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the processed output root declared by the config.",
    )
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Viser bind address. Keep loopback unless remote network exposure is intended.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--scene-display",
        choices=("points", "mesh"),
        default="points",
        help="Use points for responsive inspection; mesh uploads the full released room.",
    )
    parser.add_argument("--max-scene-points", type=int, default=200_000)
    parser.add_argument("--max-occupancy-points", type=int, default=50_000)
    parser.add_argument(
        "--local-scene-samples",
        type=int,
        default=None,
        help="Override the configured moving local point-cloud sample count.",
    )
    parser.add_argument(
        "--local-scene-radius",
        type=float,
        default=None,
        help="Override the configured local scene crop radius in metres.",
    )
    parser.add_argument(
        "--physx-trace",
        type=Path,
        default=None,
        help="Optional replay trace NPZ with reference/actual PhysX body poses and forces.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    dataset_root = _resolve(config_dir, config["dataset"]["root"])
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _resolve(config_dir, config["output_root"])
    )
    descriptor_records = _load_records(output_root / "descriptors" / f"{args.split}.jsonl")

    if args.clip_id is not None:
        matches = [record for record in descriptor_records if record["clip_id"] == args.clip_id]
        if not matches:
            raise ValueError(f"Clip {args.clip_id!r} is not in processed split {args.split!r}")
        record = matches[0]
    else:
        if not 0 <= args.clip_index < len(descriptor_records):
            raise IndexError(f"clip-index must be in [0, {len(descriptor_records) - 1}]")
        record = descriptor_records[args.clip_index]

    adapter = TrumansAdapter(
        root=dataset_root,
        manifest_path=_resolve(config_dir, config["dataset"]["manifest"]),
        eligible_only=bool(config["dataset"].get("eligible_only", True)),
        bad_frame_policy=config["dataset"].get("bad_frame_policy", "drop_clip"),
    )
    clips = {clip.clip_id: clip for clip in adapter.iter_clips()}
    clip = clips[record["clip_id"]]
    source_human = adapter.load_human(clip)

    source_index = {name: index for index, name in enumerate(source_human.joint_names)}
    source_joints = transform_points(source_human.joint_positions, TRUMANS_Y_UP_TO_Z_UP)
    source_joints = source_joints[:, [source_index[name] for name in SMPL_MUJOCO_NAMES]]

    motion = torch.load(record["motion_file"], map_location="cpu", weights_only=False)
    target_pos = motion["rigid_body_pos"].numpy()
    target_rot = motion["rigid_body_rot"].numpy()
    contacts = motion.get("rigid_body_contacts")
    contacts = contacts.numpy() if contacts is not None else None
    if target_pos.shape != source_joints.shape:
        raise ValueError(f"Source joints {source_joints.shape} != converted joints {target_pos.shape}")

    frame_errors = np.linalg.norm(source_joints - target_pos, axis=-1)
    print(f"Clip: {record['clip_id']} ({len(target_pos)} frames at {record['fps']} FPS)")
    print(
        "Joint alignment: "
        f"median={np.median(frame_errors):.4f} m, "
        f"p95={np.quantile(frame_errors, 0.95):.4f} m, "
        f"max={np.max(frame_errors):.4f} m"
    )

    room = _load_mesh(Path(record["scene_mesh"]), (150, 155, 165, 90))
    room.vertices = transform_points(room.vertices, TRUMANS_Y_UP_TO_Z_UP)
    object_cache = np.load(record["object_cache"])
    primitive_path = Path(
        record.get(
            "primitive_proxy",
            output_root / "primitives" / args.split / f"{record['clip_id']}.npz",
        )
    )
    mesh_contact_path = (
        output_root / "mesh_contacts" / args.split / f"{record['clip_id']}.npz"
    )
    primitive_contact_path = Path(
        record.get(
            "contact_labels",
            output_root / "contacts" / args.split / f"{record['clip_id']}.npz",
        )
    )
    contact_path = (
        mesh_contact_path if mesh_contact_path.is_file() else primitive_contact_path
    )
    primitive_proxy = np.load(primitive_path) if primitive_path.is_file() else None
    contact_labels = np.load(contact_path) if contact_path.is_file() else None
    default_trace_path = contact_path.with_name(
        f"{contact_path.stem}.physx_target_physics_contact.trace.npz"
    )
    physx_trace_path = (
        args.physx_trace.expanduser().resolve()
        if args.physx_trace is not None
        else default_trace_path
    )
    physx_trace = None
    if physx_trace_path.is_file():
        with np.load(physx_trace_path) as trace:
            physx_trace = {key: np.asarray(trace[key]) for key in trace.files}
        if physx_trace["actual_body_pos"].shape[1:] != target_pos.shape[1:]:
            raise ValueError(
                f"PhysX trace body shape {physx_trace['actual_body_pos'].shape} "
                f"does not match motion {target_pos.shape}"
            )
    object_meshes = []
    for object_id, mesh_path in zip(object_cache["names"].tolist(), object_cache["mesh_paths"].tolist()):
        object_meshes.append(
            (object_id, _load_mesh(Path(mesh_path), (70, 145, 235, 180)))
        )
    occupancy_points = _occupancy_points(record, args.max_occupancy_points)
    collision_stem = output_root / "collision_meshes" / args.split / record["scene_id"]
    collision_obj_path = collision_stem.with_suffix(".obj")
    collision_pointcloud_path = collision_stem.with_suffix(".pointcloud.npz")
    duplicate_faces_path = collision_stem.with_suffix(".duplicate_faces.obj")
    collision_mesh = None
    duplicate_faces_mesh = None
    collision_points = None
    collision_normals = None
    collision_tree = None
    if collision_obj_path.is_file():
        collision_mesh = _load_mesh(collision_obj_path, (255, 105, 20, 255))
    if duplicate_faces_path.is_file():
        duplicate_faces_mesh = _load_mesh(duplicate_faces_path, (255, 20, 180, 255))
    if collision_pointcloud_path.is_file():
        with np.load(collision_pointcloud_path) as cache:
            collision_points = np.asarray(cache["points"], dtype=np.float32)
            collision_normals = np.asarray(cache["normals"], dtype=np.float32)
        collision_tree = cKDTree(collision_points)
    obs_cfg = config.get("observations", {}).get("static_geometry", {})
    local_scene_samples = int(
        args.local_scene_samples
        if args.local_scene_samples is not None
        else obs_cfg.get("samples", 256)
    )
    local_scene_radius = float(
        args.local_scene_radius
        if args.local_scene_radius is not None
        else obs_cfg.get("crop_radius_m", 3.0)
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Open http://localhost:{args.port} (use VS Code port forwarding when remote)")
    if collision_mesh is None:
        print(f"WARNING: collision mesh not found: {collision_obj_path}")
    else:
        print(
            f"Collision mesh: {collision_obj_path} "
            f"({len(collision_mesh.faces):,} faces)"
        )
    if duplicate_faces_mesh is not None:
        print(
            f"Removed dynamic duplicates: {duplicate_faces_path} "
            f"({len(duplicate_faces_mesh.faces):,} faces)"
        )
    if contact_labels is not None:
        geometry_source = (
            str(contact_labels["geometry_source"])
            if "geometry_source" in contact_labels.files
            else "legacy"
        )
        print(f"Contact labels: {contact_path} ({geometry_source})")
    if physx_trace is not None:
        print(
            f"PhysX replay trace: {physx_trace_path} "
            f"({len(physx_trace['actual_body_pos'])} frames)"
        )
    if collision_points is None:
        print(f"WARNING: collision point cache not found: {collision_pointcloud_path}")
    else:
        print(
            f"Collision surface cache: {collision_pointcloud_path} "
            f"({len(collision_points):,} points)"
        )
    server.scene.add_grid("/grid", width=12.0, height=12.0, cell_size=0.5, plane="xy")
    released_scene_visible = collision_mesh is None
    if args.scene_display == "points":
        room_points = _sample_scene_surface(room, args.max_scene_points)
        room_handle = server.scene.add_point_cloud(
            "/released_scene_surface",
            room_points,
            colors=np.array([135, 140, 150], dtype=np.uint8),
            point_size=0.006,
            visible=released_scene_visible,
        )
    else:
        print(
            f"Uploading the complete released scene mesh ({len(room.faces):,} faces); "
            "this can be slow."
        )
        room_handle = server.scene.add_mesh_trimesh(
            "/released_scene_mesh", room, visible=released_scene_visible
        )
    collision_mesh_handle = None
    if collision_mesh is not None:
        # Explicit material keeps the layer visibly orange in both upstream
        # Viser and the CRISP vendored build; GLB face alpha was easy to miss.
        collision_mesh_handle = server.scene.add_mesh_simple(
            "/final_collision_mesh",
            vertices=np.asarray(collision_mesh.vertices, dtype=np.float32),
            faces=np.asarray(collision_mesh.faces, dtype=np.uint32),
            color=(255, 105, 20),
            opacity=0.82,
            flat_shading=True,
            side="double",
            visible=True,
        )
    duplicate_faces_handle = None
    if duplicate_faces_mesh is not None:
        duplicate_faces_handle = server.scene.add_mesh_simple(
            "/removed_dynamic_duplicates",
            vertices=np.asarray(duplicate_faces_mesh.vertices, dtype=np.float32),
            faces=np.asarray(duplicate_faces_mesh.faces, dtype=np.uint32),
            color=(255, 20, 180),
            opacity=0.95,
            flat_shading=True,
            side="double",
            visible=False,
        )
    replay_actual_handle = None
    replay_expected_contact_handle = None
    replay_actual_contact_handle = None
    if physx_trace is not None:
        replay_actual_handle = server.scene.add_point_cloud(
            "/physx_replay/actual_body_centers",
            physx_trace["actual_body_pos"][0],
            colors=np.tile(np.asarray([[255, 70, 40]], dtype=np.uint8), (len(SMPL_MUJOCO_NAMES), 1)),
            point_size=0.035,
            point_shape="circle",
            visible=False,
        )
        replay_expected_contact_handle = server.scene.add_point_cloud(
            "/physx_replay/expected_contacts",
            physx_trace["reference_body_pos"][0],
            colors=np.tile(np.asarray([[255, 220, 30]], dtype=np.uint8), (len(SMPL_MUJOCO_NAMES), 1)),
            point_size=0.055,
            point_shape="circle",
            visible=False,
        )
        replay_actual_contact_handle = server.scene.add_point_cloud(
            "/physx_replay/actual_contacts",
            physx_trace["actual_body_pos"][0],
            colors=np.tile(np.asarray([[40, 220, 255]], dtype=np.uint8), (len(SMPL_MUJOCO_NAMES), 1)),
            point_size=0.045,
            point_shape="circle",
            visible=False,
        )
    occupancy_handle = None
    if occupancy_points is not None:
        occupancy_handle = server.scene.add_point_cloud(
            "/released_occupancy",
            occupancy_points,
            colors=np.array([245, 190, 40], dtype=np.uint8),
            point_size=0.008,
            visible=False,
        )

    with server.gui.add_folder("Playback"):
        clip_label = server.gui.add_text("Clip", initial_value=record["clip_id"], disabled=True)
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=len(target_pos) - 1, step=1, initial_value=0
        )
        play = server.gui.add_checkbox("Play", initial_value=False)
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        error_label = server.gui.add_text("Frame joint error", initial_value="", disabled=True)
    with server.gui.add_folder("Layers"):
        room_layer_name = (
            "Released scene surface points"
            if args.scene_display == "points"
            else "Released scene mesh"
        )
        show_room = server.gui.add_checkbox(
            room_layer_name, initial_value=released_scene_visible
        )
        show_collision_mesh = server.gui.add_checkbox(
            "Final static collision mesh", initial_value=collision_mesh_handle is not None
        )
        show_duplicate_faces = server.gui.add_checkbox(
            "Removed dynamic duplicates (magenta)", initial_value=False
        )
        show_local_scene = server.gui.add_checkbox(
            f"Moving local scene points ({local_scene_samples})",
            initial_value=collision_points is not None,
        )
        show_occupancy = server.gui.add_checkbox("Released occupancy", initial_value=False)
        show_objects = server.gui.add_checkbox("Dynamic object meshes", initial_value=True)
        show_primitives = server.gui.add_checkbox(
            "Primitive collision proxy", initial_value=collision_mesh is None
        )
        show_contacts = server.gui.add_checkbox(
            "Training contacts: body red / surface cyan", initial_value=True
        )
        show_source = server.gui.add_checkbox("Released joints (green)", initial_value=True)
        show_target = server.gui.add_checkbox("Converted humanoid", initial_value=True)
        show_errors = server.gui.add_checkbox("Joint error vectors (red)", initial_value=False)
        show_replay_actual = server.gui.add_checkbox(
            "PhysX actual body centers (red)",
            initial_value=False,
            disabled=physx_trace is None,
        )
        show_replay_contacts = server.gui.add_checkbox(
            "PhysX contacts: expected yellow / actual cyan",
            initial_value=False,
            disabled=physx_trace is None,
        )

    body_regions = create_skeleton_meshes("smpl")
    body_meshes = _build_body_meshes(SMPL_GEOM_DEFS, body_regions)
    target_frames = []
    for body_id, body_name in enumerate(SMPL_MUJOCO_NAMES):
        frame_handle = server.scene.add_frame(f"/converted/{body_name}", show_axes=False)
        mesh = body_meshes[body_id].copy()
        server.scene.add_mesh_trimesh(f"/converted/{body_name}/geom", mesh)
        target_frames.append(frame_handle)

    object_handles = []
    for object_index, (object_id, mesh) in enumerate(object_meshes):
        handle = server.scene.add_mesh_trimesh(f"/objects/{object_id}_{object_index}", mesh)
        object_handles.append(handle)

    static_primitive_handles = []
    dynamic_primitive_handles = []
    if primitive_proxy is not None:
        for primitive_id, extents in enumerate(primitive_proxy["static_extents"]):
            mesh = trimesh.creation.box(extents=extents)
            mesh.visual.face_colors = [245, 170, 35, 85]
            handle = server.scene.add_mesh_trimesh(
                f"/collision_proxy/static_{primitive_id:02d}", mesh
            )
            handle.position = primitive_proxy["static_translations"][primitive_id]
            handle.wxyz = quat_xyzw_to_wxyz(
                primitive_proxy["static_rotations_xyzw"][primitive_id]
            )
            static_primitive_handles.append(handle)
        for primitive_id, extents in enumerate(primitive_proxy["dynamic_extents"]):
            mesh = trimesh.creation.box(extents=extents)
            mesh.visual.face_colors = [220, 70, 225, 105]
            dynamic_primitive_handles.append(
                server.scene.add_mesh_trimesh(
                    f"/collision_proxy/dynamic_{primitive_id:02d}", mesh
                )
            )

    contact_handles = []
    contact_surface_handles = []
    contact_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.026)
    contact_mesh.visual.face_colors = [255, 35, 35, 255]
    contact_surface_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.018)
    contact_surface_mesh.visual.face_colors = [20, 225, 235, 255]
    if contact_labels is not None:
        for body_name in SMPL_MUJOCO_NAMES:
            contact_handles.append(
                server.scene.add_mesh_trimesh(
                    f"/training_contacts/body/{body_name}", contact_mesh
                )
            )
            contact_surface_handles.append(
                server.scene.add_mesh_trimesh(
                    f"/training_contacts/proxy_surface/{body_name}",
                    contact_surface_mesh,
                )
            )

    parents = SKELETONS["smpl"]["parents"]
    source_joint_handles = []
    source_bone_handles: list[tuple[int, int, object]] = []
    joint_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.016)
    joint_mesh.visual.face_colors = [45, 220, 90, 255]
    for body_id, body_name in enumerate(SMPL_MUJOCO_NAMES):
        handle = server.scene.add_mesh_trimesh(
            f"/released_human/joints/{body_name}", joint_mesh
        )
        source_joint_handles.append(handle)
        parent = parents[body_id]
        if parent < 0:
            continue
        median_length = float(
            np.median(np.linalg.norm(source_joints[:, body_id] - source_joints[:, parent], axis=-1))
        )
        cylinder = trimesh.creation.cylinder(radius=0.008, height=median_length, sections=8)
        cylinder.visual.face_colors = [45, 220, 90, 255]
        bone_frame = server.scene.add_frame(
            f"/released_human/bones/{body_name}", show_axes=False
        )
        server.scene.add_mesh_trimesh(
            f"/released_human/bones/{body_name}/geom", cylinder
        )
        source_bone_handles.append((parent, body_id, bone_frame))

    error_handle = None
    replay_error_handle = None
    local_scene_handle = None
    last_local_indices = None

    def update_local_scene_points(frame: int) -> None:
        """Show body-stratified nearest collision samples for the current pose."""
        nonlocal local_scene_handle, last_local_indices
        if (
            not show_local_scene.value
            or collision_tree is None
            or collision_points is None
            or collision_normals is None
        ):
            if local_scene_handle is not None:
                local_scene_handle.visible = False
            return
        anchors = target_pos[frame]
        per_body = max(1, int(np.ceil(local_scene_samples / len(anchors))))
        distances, indices = collision_tree.query(anchors, k=per_body, workers=-1)
        distances = np.asarray(distances).reshape(-1)
        indices = np.asarray(indices).reshape(-1)
        valid = distances <= local_scene_radius
        selected = []
        seen = set()
        for index in indices[valid]:
            index = int(index)
            if index not in seen:
                seen.add(index)
                selected.append(index)
        if len(selected) < local_scene_samples:
            root_distances, root_indices = collision_tree.query(
                anchors[0], k=min(local_scene_samples * 4, len(collision_points))
            )
            for distance, index in zip(
                np.asarray(root_distances).reshape(-1),
                np.asarray(root_indices).reshape(-1),
            ):
                index = int(index)
                if distance > local_scene_radius:
                    continue
                if index not in seen:
                    seen.add(index)
                    selected.append(index)
                if len(selected) >= local_scene_samples:
                    break
        selected = np.asarray(selected[:local_scene_samples], dtype=np.int64)
        if len(selected) == 0:
            if local_scene_handle is not None:
                local_scene_handle.visible = False
            return
        if (
            local_scene_handle is not None
            and last_local_indices is not None
            and np.array_equal(selected, last_local_indices)
        ):
            local_scene_handle.visible = True
            return
        # Encode normals as RGB so surface orientation can be inspected without
        # uploading hundreds of line segments every playback frame.
        colors = np.clip((collision_normals[selected] * 0.5 + 0.5) * 255, 0, 255).astype(
            np.uint8
        )
        # Re-sending the same node name replaces its point buffer directly.
        # Removing it first creates a visible blank frame in the browser.
        local_scene_handle = server.scene.add_point_cloud(
            "/moving_local_scene_points",
            collision_points[selected],
            colors=colors,
            point_size=0.018,
            point_shape="circle",
        )
        last_local_indices = selected.copy()

    def update_frame(frame: int, *, update_local: bool = True) -> None:
        nonlocal error_handle, replay_error_handle
        source = source_joints[frame]
        target = target_pos[frame]
        current_errors = frame_errors[frame]
        error_label.value = (
            f"median {np.median(current_errors) * 100:.1f} cm | "
            f"max {np.max(current_errors) * 100:.1f} cm"
        )

        with server.atomic():
            for body_id, handle in enumerate(source_joint_handles):
                handle.position = source[body_id]
                handle.visible = show_source.value
            for parent, child, handle in source_bone_handles:
                direction = source[child] - source[parent]
                handle.position = (source[child] + source[parent]) * 0.5
                handle.wxyz = _wxyz_align_z(direction)
                handle.visible = show_source.value

        if error_handle is not None:
            error_handle.remove()
            error_handle = None
        # Error geometry changes size every frame.  Only upload it while paused so
        # playback does not accumulate full mesh messages in the browser queue.
        if show_errors.value and not play.value:
            mesh = _error_mesh(source, target)
            if mesh is not None:
                error_handle = server.scene.add_mesh_trimesh("/joint_errors", mesh)

        if physx_trace is not None:
            trace_frame = min(frame, len(physx_trace["actual_body_pos"]) - 1)
            actual_body = physx_trace["actual_body_pos"][trace_frame]
            reference_body = physx_trace["reference_body_pos"][trace_frame]
            expected_body_contact = physx_trace["expected_contact"][trace_frame]
            actual_body_contact = physx_trace["actual_contact"][trace_frame]
            server.scene.add_point_cloud(
                "/physx_replay/actual_body_centers",
                actual_body,
                colors=np.tile(
                    np.asarray([[255, 70, 40]], dtype=np.uint8),
                    (len(actual_body), 1),
                ),
                point_size=0.035,
                point_shape="circle",
                visible=bool(show_replay_actual.value),
            )
            server.scene.add_point_cloud(
                "/physx_replay/expected_contacts",
                reference_body[expected_body_contact],
                colors=np.tile(
                    np.asarray([[255, 220, 30]], dtype=np.uint8),
                    (int(expected_body_contact.sum()), 1),
                ),
                point_size=0.055,
                point_shape="circle",
                visible=bool(show_replay_contacts.value),
            )
            server.scene.add_point_cloud(
                "/physx_replay/actual_contacts",
                actual_body[actual_body_contact],
                colors=np.tile(
                    np.asarray([[40, 220, 255]], dtype=np.uint8),
                    (int(actual_body_contact.sum()), 1),
                ),
                point_size=0.045,
                point_shape="circle",
                visible=bool(show_replay_contacts.value),
            )
            if replay_error_handle is not None:
                replay_error_handle.remove()
                replay_error_handle = None
            if show_replay_actual.value and not play.value:
                replay_error_handle = _error_mesh(reference_body, actual_body)
                if replay_error_handle is not None:
                    replay_error_handle.visual.face_colors = [255, 70, 40, 220]
                    replay_error_handle = server.scene.add_mesh_trimesh(
                        "/physx_replay/body_error_vectors", replay_error_handle
                    )
            if replay_actual_handle is not None:
                replay_actual_handle.visible = bool(show_replay_actual.value)
            if replay_expected_contact_handle is not None:
                replay_expected_contact_handle.visible = bool(show_replay_contacts.value)
            if replay_actual_contact_handle is not None:
                replay_actual_contact_handle.visible = bool(show_replay_contacts.value)

        with server.atomic():
            for body_id, handle in enumerate(target_frames):
                handle.position = target[body_id]
                handle.wxyz = quat_xyzw_to_wxyz(target_rot[frame, body_id])
                handle.visible = show_target.value

            for object_index, handle in enumerate(object_handles):
                pose_frame = min(frame, object_cache["translations"].shape[1] - 1)
                handle.position = object_cache["translations"][object_index, pose_frame]
                handle.wxyz = quat_xyzw_to_wxyz(
                    object_cache["rotations_xyzw"][object_index, pose_frame]
                )
                handle.visible = show_objects.value

            for handle in static_primitive_handles:
                handle.visible = show_primitives.value
            for primitive_id, handle in enumerate(dynamic_primitive_handles):
                pose_frame = min(
                    frame, primitive_proxy["dynamic_translations"].shape[1] - 1
                )
                handle.position = primitive_proxy["dynamic_translations"][primitive_id, pose_frame]
                handle.wxyz = quat_xyzw_to_wxyz(
                    primitive_proxy["dynamic_rotations_xyzw"][primitive_id, pose_frame]
                )
                handle.visible = show_primitives.value
            if contact_labels is not None:
                contact_key = (
                    "training_contact"
                    if "training_contact" in contact_labels.files
                    else "intended_contact"
                )
                frame_contacts = contact_labels[contact_key][frame].any(axis=-1)
                point_key = (
                    "source_nearest_point"
                    if "source_nearest_point" in contact_labels.files
                    else "nearest_point"
                )
                for body_id, handle in enumerate(contact_handles):
                    handle.position = source[body_id]
                    handle.visible = bool(show_contacts.value and frame_contacts[body_id])
                    surface_handle = contact_surface_handles[body_id]
                    surface_handle.position = contact_labels[point_key][frame, body_id]
                    surface_handle.visible = bool(
                        show_contacts.value and frame_contacts[body_id]
                    )
        if update_local:
            update_local_scene_points(frame)

    def apply_static_layer_visibility() -> None:
        """Apply cheap visibility changes from the main thread."""
        room_handle.visible = show_room.value
        if collision_mesh_handle is not None:
            collision_mesh_handle.visible = show_collision_mesh.value
        if duplicate_faces_handle is not None:
            duplicate_faces_handle.visible = show_duplicate_faces.value
        if occupancy_handle is not None:
            occupancy_handle.visible = show_occupancy.value

    def current_layer_state() -> tuple[bool, ...]:
        return (
            bool(show_room.value),
            bool(show_collision_mesh.value),
            bool(show_duplicate_faces.value),
            bool(show_local_scene.value),
            bool(show_occupancy.value),
            bool(show_objects.value),
            bool(show_primitives.value),
            bool(show_contacts.value),
            bool(show_source.value),
            bool(show_target.value),
            bool(show_errors.value),
            bool(show_replay_actual.value),
            bool(show_replay_contacts.value),
        )

    apply_static_layer_visibility()
    update_frame(0)
    last_rendered_frame = 0
    last_layer_state = current_layer_state()
    last_local_update_time = time.time()
    last_time = time.time()
    frame_accumulator = 0.0
    try:
        while True:
            now = time.time()
            if play.value:
                frame_accumulator += (now - last_time) * float(record["fps"]) * speed.value
                if frame_accumulator >= 1.0:
                    advance = int(frame_accumulator)
                    frame_accumulator -= advance
                    frame_slider.value = (int(frame_slider.value) + advance) % len(target_pos)
            frame = int(frame_slider.value)
            layer_state = current_layer_state()
            layer_changed = layer_state != last_layer_state
            frame_changed = frame != last_rendered_frame
            if layer_changed:
                apply_static_layer_visibility()
            if frame_changed or layer_changed:
                # While playing, update the changing point-cloud payload at 10Hz.
                # Pose transforms still update at the requested playback rate.
                update_local = (
                    not play.value
                    or layer_changed
                    or now - last_local_update_time >= 0.1
                )
                if update_local:
                    update_frame(frame)
                    last_local_update_time = now
                else:
                    update_frame(frame, update_local=False)
                last_rendered_frame = frame
                last_layer_state = layer_state
            last_time = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
