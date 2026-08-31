# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viser-based visualizer for SMPL .motion files.

Supports both individual .motion files and packaged .pt motion libraries.
Can display two motions side-by-side for comparison (e.g., SOMA23 vs SMPL).

Usage::

    # Single SMPL motion
    python examples/visualize_smpl_motion.py --motion-file /path/to/motion.motion

    # Side-by-side comparison (SOMA23 left, SMPL right)
    python examples/visualize_smpl_motion.py \
        --motion-file /path/to/smpl.motion \
        --compare /path/to/soma23.motion --compare-robot soma23

    # Specify robot (default: auto-detect from body count)
    python examples/visualize_smpl_motion.py --motion-file /path/to/motion.motion --robot smpl
"""

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

# Make torch.load() able to resolve ProtoMotions classes even when this file is
# launched through the CRISP conda environment or from a different directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
import trimesh.creation
import trimesh.transformations as tf
import viser

from protomotions.components.terrains.terrain_utils import (
    convert_heightfield_to_trimesh,
)

# ── Skeleton definitions ──────────────────────────────────────────────────

SKELETONS = {
    "smpl": {
        "bodies": [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Thorax",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
        ],
        "parents": [
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            12,
            11,
            14,
            15,
            16,
            17,
            11,
            19,
            20,
            21,
            22,
        ],
        "regions": {
            "spine": [0, 9, 10, 11, 12, 13],
            "left_arm": [14, 15, 16, 17, 18],
            "right_arm": [19, 20, 21, 22, 23],
            "left_leg": [1, 2, 3, 4],
            "right_leg": [5, 6, 7, 8],
        },
    },
    "soma23": {
        "bodies": [
            "Hips",
            "Spine1",
            "Spine2",
            "Chest",
            "Neck1",
            "Neck2",
            "Head",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightLeg",
            "RightShin",
            "RightFoot",
            "RightToeBase",
            "LeftLeg",
            "LeftShin",
            "LeftFoot",
            "LeftToeBase",
        ],
        "parents": [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            3,
            7,
            8,
            9,
            3,
            11,
            12,
            13,
            0,
            15,
            16,
            17,
            0,
            19,
            20,
            21,
        ],
        "regions": {
            "spine": [0, 1, 2, 3, 4, 5, 6],
            "right_arm": [7, 8, 9, 10],
            "left_arm": [11, 12, 13, 14],
            "right_leg": [15, 16, 17, 18],
            "left_leg": [19, 20, 21, 22],
        },
    },
}

REGION_COLORS = {
    "spine": [220, 200, 170],
    "left_arm": [240, 140, 100],
    "right_arm": [100, 170, 240],
    "left_leg": [210, 130, 210],
    "right_leg": [100, 210, 150],
}

# SMPL body geom definitions from smpl_humanoid.xml (body_idx, type, params)
# fmt: off
SMPL_GEOM_DEFS = [
    (0, "box", {"half": [0.083, 0.1069, 0.0722], "pos": [-0.0055, 0.0, -0.0121], "quat": [1, 0, 0, 0]}),
    (1, "capsule", {"radius": 0.0615, "fromto": [-0.0009, 0.0069, -0.075, -0.0036, 0.0274, -0.3002]}),
    (2, "capsule", {"radius": 0.0541, "fromto": [-0.0087, -0.0027, -0.0796, -0.035, -0.0109, -0.3184]}),
    (3, "box", {"half": [0.085, 0.0483, 0.0464], "pos": [0.0242, 0.0233, -0.0239], "quat": [1, 0, 0, 0]}),
    (4, "box", {"half": [0.0496, 0.0478, 0.02], "pos": [0.0248, -0.003, 0.0055], "quat": [1, 0, 0, 0]}),
    (5, "capsule", {"radius": 0.0606, "fromto": [-0.0018, -0.0077, -0.0765, -0.0071, -0.0306, -0.3061]}),
    (6, "capsule", {"radius": 0.0541, "fromto": [-0.0085, 0.0032, -0.0797, -0.0338, 0.0126, -0.3187]}),
    (7, "box", {"half": [0.0865, 0.0483, 0.0478], "pos": [0.0256, -0.0212, -0.0174], "quat": [1, 0, 0, 0]}),
    (8, "box", {"half": [0.0493, 0.0479, 0.0216], "pos": [0.0227, 0.0042, 0.0045], "quat": [1, 0, 0, 0]}),
    (9, "capsule", {"radius": 0.0769, "fromto": [0.0005, 0.0025, 0.0608, 0.0006, 0.003, 0.0743]}),
    (10, "capsule", {"radius": 0.0755, "fromto": [0.0114, 0.0007, 0.0238, 0.014, 0.0008, 0.0291]}),
    (11, "capsule", {"radius": 0.1002, "fromto": [-0.0173, -0.0009, 0.0682, -0.0212, -0.001, 0.0833]}),
    (12, "capsule", {"radius": 0.0436, "fromto": [0.0103, 0.001, 0.013, 0.0411, 0.0041, 0.052]}),
    (13, "box", {"half": [0.076, 0.0606, 0.1154], "pos": [-0.0116, -0.0042, 0.0876], "quat": [1, 0, 0, 0]}),
    (14, "capsule", {"radius": 0.0521, "fromto": [-0.0018, 0.0182, 0.0061, -0.0071, 0.0728, 0.0244]}),
    (15, "capsule", {"radius": 0.0517, "fromto": [-0.0055, 0.0519, -0.0026, -0.022, 0.2077, -0.0102]}),
    (16, "capsule", {"radius": 0.0405, "fromto": [-0.0002, 0.0498, 0.0018, -0.0009, 0.1994, 0.0072]}),
    (17, "capsule", {"radius": 0.0318, "fromto": [-0.003, 0.0168, -0.0016, -0.012, 0.0672, -0.0065]}),
    (18, "box", {"half": [0.0538, 0.0585, 0.0158], "pos": [-0.0058, 0.0493, 0.001], "quat": [1, 0, 0, 0]}),
    (19, "capsule", {"radius": 0.0511, "fromto": [-0.0018, -0.0192, 0.0065, -0.0073, -0.0768, 0.026]}),
    (20, "capsule", {"radius": 0.0531, "fromto": [-0.0043, -0.0507, -0.0027, -0.0171, -0.203, -0.0107]}),
    (21, "capsule", {"radius": 0.0408, "fromto": [-0.0011, -0.0511, 0.0016, -0.0044, -0.2042, 0.0062]}),
    (22, "capsule", {"radius": 0.0326, "fromto": [-0.0021, -0.0169, -0.0012, -0.0083, -0.0677, -0.0049]}),
    (23, "box", {"half": [0.0546, 0.0569, 0.0164], "pos": [-0.0079, -0.0462, -0.0009], "quat": [1, 0, 0, 0]}),
]
# fmt: on


def _build_body_meshes(geom_defs, body_regions):
    """Build trimesh for each body in its local frame from MJCF geom specs."""
    meshes = {}
    for body_idx, geom_type, params in geom_defs:
        if geom_type == "capsule":
            p0 = np.array(params["fromto"][:3])
            p1 = np.array(params["fromto"][3:])
            length = np.linalg.norm(p1 - p0)
            if length < 1e-6:
                m = trimesh.creation.icosphere(subdivisions=1, radius=params["radius"])
            else:
                m = trimesh.creation.capsule(
                    height=length, radius=params["radius"], count=[8, 8]
                )
                mid = (p0 + p1) / 2
                direction = (p1 - p0) / length
                rot_mat = _rotation_between(np.array([0, 0, 1.0]), direction)
                T = np.eye(4)
                T[:3, :3] = rot_mat
                T[:3, 3] = mid
                m.apply_transform(T)
        elif geom_type == "box":
            m = trimesh.creation.box(extents=[2 * h for h in params["half"]])
            T = np.eye(4)
            T[:3, 3] = params["pos"]
            m.apply_transform(T)
        elif geom_type == "sphere":
            m = trimesh.creation.icosphere(subdivisions=1, radius=params["radius"])
            m.apply_translation(params["pos"])
        else:
            continue
        region = body_regions.get(body_idx, "spine")
        color = REGION_COLORS[region]
        m.visual.face_colors = color + [200]
        meshes[body_idx] = m
    return meshes


def _rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.9999:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return tf.rotation_matrix(np.pi, perp)[:3, :3]
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def detect_robot(num_bodies: int) -> str:
    if num_bodies == 24:
        return "smpl"
    elif num_bodies == 23:
        return "soma23"
    raise ValueError(f"Unknown robot with {num_bodies} bodies")


def load_motion(path: str) -> dict:
    """Load a .motion file (individual or packaged .pt)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "length_starts" in data:
        # Packaged motion lib
        motions = []
        for i in range(len(data["length_starts"])):
            start = data["length_starts"][i].item()
            n = data["motion_num_frames"][i].item()
            m = {
                "gts": data["gts"][start : start + n].numpy(),
                "grs": data["grs"][start : start + n].numpy(),
                "contacts": data["contacts"][start : start + n]
                if "contacts" in data
                else None,
                "fps": 1.0 / data["motion_dt"][i].item(),
                "name": data["motion_files"][i]
                if "motion_files" in data
                else f"motion_{i}",
            }
            motions.append(m)
        return {"motions": motions}
    elif "gts" in data and data["gts"].dim() == 3:
        # Individual .motion file: [T, B, 3]
        return {
            "motions": [
                {
                    "gts": data["gts"].numpy(),
                    "grs": data["grs"].numpy(),
                    "contacts": data.get("contacts", data.get("contacts_ground")),
                    "fps": data.get("fps", 30),
                    "name": path.split("/")[-1],
                }
            ]
        }
    elif "rigid_body_pos" in data and "rigid_body_rot" in data:
        # IsaacLab recorder output.  Its COMMON convention is also xyzw,
        # matching the motion-library files consumed above.
        positions = data["rigid_body_pos"]
        rotations = data["rigid_body_rot"]
        contacts = data.get("rigid_body_contacts")
        return {
            "motions": [
                {
                    "gts": positions.numpy() if isinstance(positions, torch.Tensor) else np.asarray(positions),
                    "grs": rotations.numpy() if isinstance(rotations, torch.Tensor) else np.asarray(rotations),
                    "contacts": contacts,
                    "fps": data.get("fps", 30),
                    "name": path.split("/")[-1],
                }
            ]
        }
    raise ValueError(f"Unknown format in {path}")


def create_skeleton_meshes(skeleton_name: str):
    """Create simple capsule meshes for skeleton visualization."""
    skel = SKELETONS[skeleton_name]
    body_to_region = {}
    for region, indices in skel["regions"].items():
        for idx in indices:
            body_to_region[idx] = region
    return body_to_region


def main():
    parser = argparse.ArgumentParser(description="Viser SMPL/SOMA23 motion visualizer")
    parser.add_argument(
        "--motion-file", required=True, help="Path to .motion or .pt file"
    )
    parser.add_argument(
        "--initial-motion-index",
        type=int,
        default=0,
        help="Packaged-motion index selected when Viser first opens.",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Robot type (smpl/soma23). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Second .motion file for side-by-side comparison",
    )
    parser.add_argument(
        "--compare-robot", default=None, help="Robot type for comparison motion"
    )
    parser.add_argument(
        "--scene-mesh",
        default=None,
        nargs="*",
        help="One or more .obj mesh files to display as static scene objects",
    )
    parser.add_argument(
        "--scene-dir",
        default=None,
        help="Directory with per-motion scene .obj files (auto-loaded by motion name)",
    )
    parser.add_argument(
        "--scene-pt",
        default=None,
        help="Packaged scene .pt file (indexed by motion slider, matching motion .pt)",
    )
    parser.add_argument(
        "--scene-asset-root",
        default=None,
        help="Root used to resolve relative mesh paths stored in --scene-pt",
    )
    parser.add_argument(
        "--terrain",
        default=None,
        help="Recorded IsaacLab terrain .pt file (height_field_raw) to display",
    )
    parser.add_argument(
        "--objects",
        default=None,
        help=(
            "Recorded .objects.pt sidecar with rollout-synchronized object poses. "
            "If omitted, a same-stem sidecar beside --motion-file is used when present."
        ),
    )
    parser.add_argument(
        "--ignore-recorded-objects",
        action="store_true",
        help=(
            "Do not auto-load the same-stem .objects.pt sidecar; display poses "
            "from --scene-pt instead."
        ),
    )
    parser.add_argument(
        "--smooth-object-window",
        type=int,
        default=1,
        help=(
            "Centered moving-average window for displayed object trajectories. "
            "Use an odd value such as 9 to suppress tracking/physics jitter."
        ),
    )
    parser.add_argument(
        "--freeze-scene-objects",
        action="store_true",
        help="Display every scene object at its first pose (visualization only).",
    )
    parser.add_argument(
        "--terrain-stride",
        type=int,
        default=1,
        help="Downsampling stride for the displayed terrain height field",
    )
    parser.add_argument(
        "--terrain-radius",
        type=float,
        default=5.0,
        help="Meters to keep around the motion when cropping a large terrain",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--show-ego-visibility",
        action="store_true",
        help="Show virtual head-camera visible/remembered/unseen scene layers.",
    )
    parser.add_argument("--ego-horizontal-fov", type=float, default=90.0)
    parser.add_argument("--ego-vertical-fov", type=float, default=60.0)
    parser.add_argument("--ego-near", type=float, default=0.05)
    parser.add_argument("--ego-far", type=float, default=6.0)
    parser.add_argument(
        "--ego-max-scene-points",
        type=int,
        default=6000,
        help="Maximum scene points used by ego-visibility visualization.",
    )
    parser.add_argument(
        "--ego-camera-pt",
        help="Measured per-motion camera trajectories in Proto world coordinates.",
    )
    parser.add_argument(
        "--ego-visibility-stride",
        type=int,
        default=1,
        help="Refresh colored visibility points every N animation frames.",
    )
    parser.add_argument(
        "--record-dir",
        default="output/viser_recordings",
        help="Directory for MP4 files generated from the Viser browser render.",
    )
    parser.add_argument("--record-width", type=int, default=960)
    parser.add_argument("--record-height", type=int, default=540)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument(
        "--offset", type=float, default=1.5, help="X offset between compared motions"
    )
    parser.add_argument(
        "--align-compare-root",
        action="store_true",
        help=(
            "Translate the comparison trajectory so its first pelvis position "
            "matches the primary motion (then apply --offset). Useful for "
            "IsaacLab recordings stored in terrain/world coordinates."
        ),
    )
    args = parser.parse_args()

    # Load primary motion
    motion_data = load_motion(args.motion_file)
    motions = motion_data["motions"]
    if not 0 <= args.initial_motion_index < len(motions):
        raise ValueError(
            f"--initial-motion-index must be in [0, {len(motions) - 1}], "
            f"got {args.initial_motion_index}"
        )
    initial_motion_index = args.initial_motion_index
    num_bodies = motions[0]["gts"].shape[1]
    robot = args.robot or detect_robot(num_bodies)
    skel = SKELETONS[robot]
    body_to_region = create_skeleton_meshes(robot)

    # Load comparison motion
    cmp_motions = None
    cmp_skel = None
    cmp_body_to_region = None
    if args.compare:
        cmp_data = load_motion(args.compare)
        cmp_motions = cmp_data["motions"]
        cmp_num_bodies = cmp_motions[0]["gts"].shape[1]
        cmp_robot = args.compare_robot or detect_robot(cmp_num_bodies)
        cmp_skel = SKELETONS[cmp_robot]
        cmp_body_to_region = create_skeleton_meshes(cmp_robot)

    # Load scene meshes
    scene_meshes = []
    if args.scene_mesh:
        for mesh_path in args.scene_mesh:
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
                scene_meshes.append((mesh_path.split("/")[-1], mesh))
                print(f"Loaded scene mesh: {mesh_path} ({len(mesh.vertices)} verts)")
            except Exception as e:
                print(f"Warning: failed to load {mesh_path}: {e}")

    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    # Motion recordings may live tens of metres from the world origin (for
    # example, terrain tiles around x=58, y=44).  Aim the initial browser
    # camera at the first pelvis instead of leaving the user looking at the
    # empty origin.
    initial_target = np.asarray(
        motions[initial_motion_index]["gts"][0, 0], dtype=np.float64
    )

    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle):
        client.camera.position = initial_target + np.array([3.5, -5.0, 2.8])
        client.camera.look_at = initial_target + np.array([0.0, 0.0, 0.8])

    # GUI controls
    motion_idx_slider = server.gui.add_slider(
        "Motion",
        min=0,
        max=len(motions) - 1,
        step=1,
        initial_value=initial_motion_index,
    )
    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=motions[initial_motion_index]["gts"].shape[0] - 1,
        step=1,
        initial_value=0,
    )
    playing = server.gui.add_checkbox("Play", initial_value=False)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
    )
    _sphere_radius = server.gui.add_slider(
        "Body radius", min=0.01, max=0.08, step=0.005, initial_value=0.04
    )
    with server.gui.add_folder("Ego scene visibility"):
        show_scene_mesh = server.gui.add_checkbox(
            "Scene mesh", initial_value=True
        )
        show_ego_current = server.gui.add_checkbox(
            "Currently visible", initial_value=args.show_ego_visibility
        )
        show_ego_memory = server.gui.add_checkbox(
            "Seen but currently hidden", initial_value=args.show_ego_visibility
        )
        show_ego_unseen = server.gui.add_checkbox(
            "Not seen yet", initial_value=False
        )
        show_ego_frustum = server.gui.add_checkbox(
            "Measured GT/input camera", initial_value=args.show_ego_visibility
        )
        show_student_head_frustum = server.gui.add_checkbox(
            # Keep report views uncluttered by default. These diagnostic
            # frusta can still be enabled explicitly from the Viser panel.
            "Student head camera", initial_value=False
        )
        show_compare_head_frustum = server.gui.add_checkbox(
            "GT-body head camera", initial_value=False
        )
        ego_current_count = server.gui.add_text(
            "Visible points", initial_value="0", disabled=True
        )
        ego_memory_count = server.gui.add_text(
            "Memory points", initial_value="0", disabled=True
        )
        ego_unseen_count = server.gui.add_text(
            "Unseen points", initial_value="0", disabled=True
        )
    record_button = server.gui.add_button("Generate MP4 from Viser")
    record_status = server.gui.add_text("Record status", initial_value="Idle", disabled=True)

    record_state = {
        "requested": False,
        "active": False,
        "client": None,
        "frames": [],
        "path": None,
    }

    @record_button.on_click
    def _(_event: viser.GuiEvent):
        if record_state["active"] or record_state["requested"]:
            return
        record_state["requested"] = True
        record_status.value = "Preparing frames..."

    # Add static scene meshes (from --scene-mesh)
    for i, (name, mesh) in enumerate(scene_meshes):
        if not hasattr(mesh.visual, "face_colors") or mesh.visual.face_colors is None:
            mesh.visual.face_colors = [180, 180, 180, 200]
        server.scene.add_mesh_trimesh(f"/scene/{name}", mesh)

    # Add ground plane
    ground = trimesh.creation.box(extents=[20, 20, 0.005])
    ground.visual.face_colors = [200, 200, 200, 100]
    ground.apply_translation([initial_target[0], initial_target[1], -0.0025])
    server.scene.add_mesh_trimesh("/scene/ground", ground)

    # Optional height-field terrain.  The full IsaacLab height field can be
    # very large, so it is decimated for interactive browser rendering while
    # preserving the original height values and stair/profile geometry.
    if args.terrain:
        try:
            terrain_data = torch.load(
                args.terrain, weights_only=False, map_location="cpu"
            )
            raw = np.asarray(terrain_data["height_field_raw"])
            stride = max(1, int(args.terrain_stride))
            hs = float(terrain_data["horizontal_scale"]) * stride
            vs = float(terrain_data["vertical_scale"])
            # IsaacLab height fields are stored in world-aligned x/y cells.
            # Crop to the trajectory neighbourhood so the browser camera does
            # not fit a 100m terrain around a 2m humanoid.
            all_positions = np.concatenate(
                [np.asarray(m["gts"])[:, :, :2].reshape(-1, 2) for m in motions],
                axis=0,
            )
            lo = all_positions.min(axis=0) - float(args.terrain_radius)
            hi = all_positions.max(axis=0) + float(args.terrain_radius)
            raw_x0 = max(0, int(np.floor(lo[0] / (hs / stride))))
            raw_y0 = max(0, int(np.floor(lo[1] / (hs / stride))))
            raw_x1 = min(raw.shape[0], int(np.ceil(hi[0] / (hs / stride))) + 1)
            raw_y1 = min(raw.shape[1], int(np.ceil(hi[1] / (hs / stride))) + 1)
            raw = raw[raw_x0:raw_x1:stride, raw_y0:raw_y1:stride]
            origin_x = raw_x0 * (hs / stride)
            origin_y = raw_y0 * (hs / stride)
            terrain_vertices, terrain_faces = convert_heightfield_to_trimesh(
                raw,
                hs,
                vs,
                slope_threshold=float(terrain_data.get("slope_threshold", 0.9)),
            )
            terrain_vertices[:, 0] += origin_x
            terrain_vertices[:, 1] += origin_y
            terrain_mesh = trimesh.Trimesh(
                vertices=terrain_vertices, faces=terrain_faces, process=False
            )
            terrain_mesh.visual.face_colors = [191, 137, 69, 180]
            server.scene.add_mesh_trimesh("/scene/terrain", terrain_mesh)
            print(
                f"Loaded terrain {args.terrain}: {raw.shape[0]}x{raw.shape[1]} "
                f"(stride={stride})"
            )
        except Exception as exc:
            print(f"Warning: failed to load terrain {args.terrain}: {exc}")

    # Per-motion scene loading
    scene_dir = Path(args.scene_dir) if args.scene_dir else None
    # Load packaged scene .pt if provided
    packaged_scenes = None
    if args.scene_pt:
        packaged_scenes = torch.load(
            args.scene_pt, weights_only=False, map_location="cpu"
        )
        print(
            f"Loaded packaged scenes: {packaged_scenes['num_original_scenes']} scenes, "
            f"{packaged_scenes['num_objects_per_scene']} objects/scene"
        )

    ego_cameras = None
    if args.ego_camera_pt:
        ego_camera_data = torch.load(
            args.ego_camera_pt, weights_only=False, map_location="cpu"
        )
        ego_cameras = ego_camera_data["motions"]
        if len(ego_cameras) != len(motions):
            raise ValueError(
                f"Camera pack has {len(ego_cameras)} motions, expected {len(motions)}"
            )
        print(f"Loaded measured ego cameras: {args.ego_camera_pt}")

    recorded_objects = None
    objects_path = Path(args.objects) if args.objects else None
    if (
        objects_path is None
        and not args.ignore_recorded_objects
        and str(args.motion_file).endswith(".motion")
    ):
        candidate = Path(str(args.motion_file)[: -len(".motion")] + ".objects.pt")
        if candidate.is_file():
            objects_path = candidate
    if objects_path is not None:
        try:
            objects_data = torch.load(
                objects_path, weights_only=False, map_location="cpu"
            )
            recorded_objects = objects_data.get("objects", [])
            print(
                f"Loaded synchronized object poses: {objects_path} "
                f"({len(recorded_objects)} objects)"
            )
        except Exception as exc:
            print(f"Warning: failed to load recorded objects {objects_path}: {exc}")

    current_scene_handles = []
    current_scene_objects = []
    current_scene_idx = [-1]
    mesh_cache = {}
    ego_visibility_state = {
        "seen_static_ids": set(),
        "last_frame": -1,
        "scene_index": -1,
        "render_signature": None,
    }
    empty_points = np.zeros((0, 3), dtype=np.float32)
    # CRISP currently vendors an older Viser whose PointCloudHandle cannot
    # update points/colors in place. Keep mutable slots and replace the handles
    # whenever the frame changes.
    ego_layer_handles = {"current": None, "memory": None, "unseen": None}
    ego_frustum_handle = server.scene.add_camera_frustum(
        "/ego/camera_frustum",
        fov=np.deg2rad(args.ego_vertical_fov),
        aspect=(
            np.tan(np.deg2rad(args.ego_horizontal_fov) * 0.5)
            / np.tan(np.deg2rad(args.ego_vertical_fov) * 0.5)
        ),
        # This is an orientation icon, not a metric drawing of the 6m far plane.
        scale=0.18,
        color=(255, 220, 40),
        thickness=2.0,
        visible=args.show_ego_visibility,
    )
    student_head_frustum_handle = server.scene.add_camera_frustum(
        "/ego/student_head_camera_frustum",
        fov=np.deg2rad(args.ego_vertical_fov),
        aspect=np.tan(np.deg2rad(args.ego_horizontal_fov) * 0.5)
        / np.tan(np.deg2rad(args.ego_vertical_fov) * 0.5),
        scale=0.15,
        color=(40, 210, 255),
        thickness=2.0,
        visible=False,
    )
    compare_head_frustum_handle = server.scene.add_camera_frustum(
        "/ego/gt_body_head_camera_frustum",
        fov=np.deg2rad(args.ego_vertical_fov),
        aspect=np.tan(np.deg2rad(args.ego_horizontal_fov) * 0.5)
        / np.tan(np.deg2rad(args.ego_vertical_fov) * 0.5),
        scale=0.15,
        color=(255, 70, 210),
        thickness=2.0,
        visible=False,
    )

    def _clear_scene():
        for h in current_scene_handles:
            h.remove()
        current_scene_handles.clear()
        current_scene_objects.clear()
        ego_visibility_state["seen_static_ids"].clear()
        ego_visibility_state["last_frame"] = -1
        ego_visibility_state["render_signature"] = None

    def _mesh_for_object(obj: dict):
        object_path = obj.get("object_path")
        scale = np.asarray(obj.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if object_path:
            path = Path(object_path)
            if not path.is_absolute() and args.scene_asset_root:
                path = Path(args.scene_asset_root) / path
            # PhysX scene packs reference USD collision assets, which trimesh
            # cannot read. The preparation pipeline writes a same-stem OBJ in
            # the *same transformed coordinates* for inspection. The original
            # TRUMANS visual OBJ uses a different source coordinate frame.
            if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                path = path.with_suffix(".obj")
            cache_key = (str(path.resolve()), tuple(scale.tolist()))
            if path.is_file():
                try:
                    if cache_key not in mesh_cache:
                        mesh = trimesh.load(path, force="mesh")
                        if isinstance(mesh, trimesh.Scene):
                            mesh = trimesh.util.concatenate(
                                tuple(mesh.geometry.values())
                            )
                        mesh = mesh.copy()
                        mesh.apply_scale(scale)
                        # Preserve texture/material visuals on the original
                        # scene mesh. Only color geometry with no visual data.
                        if getattr(mesh.visual, "kind", None) is None:
                            mesh.visual.face_colors = [150, 155, 175, 255]
                        mesh_cache[cache_key] = mesh
                    return mesh_cache[cache_key]
                except (NotImplementedError, ValueError) as exc:
                    print(
                        f"Warning: cannot load scene mesh {path} ({exc}); "
                        "using its packaged bounding box."
                    )
            print(
                f"Warning: scene mesh not found: {path}. "
                "Using its bounding box; pass --scene-asset-root for relative paths."
            )

        dims = obj.get("object_dims")
        if dims is None:
            return None
        min_x, max_x, min_y, max_y, min_z, max_z = dims
        extents = np.array([max_x - min_x, max_y - min_y, max_z - min_z]) * scale
        if np.all(extents < 0.01):
            return None
        center = np.array(
            [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
        ) * scale
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        mesh.visual.face_colors = [160, 160, 180, 120]
        return mesh

    def _render_scene_objects(scene: dict):
        """Render true scene meshes and retain their complete pose trajectories."""
        def smooth_trajectory(values: np.ndarray, window: int) -> np.ndarray:
            if window <= 1 or len(values) <= 1:
                return values
            window = min(int(window), len(values))
            if window % 2 == 0:
                window = max(1, window - 1)
            if window <= 1:
                return values
            radius = window // 2
            padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
            cumulative = np.concatenate(
                [np.zeros((1, values.shape[1]), dtype=np.float64),
                 np.cumsum(padded, axis=0, dtype=np.float64)],
                axis=0,
            )
            return ((cumulative[window:] - cumulative[:-window]) / window).astype(
                values.dtype
            )

        for j, obj in enumerate(scene.get("objects", [])):
            mesh = _mesh_for_object(obj)
            if mesh is None:
                continue
            translations = np.asarray(
                obj.get("translation", [[0, 0, 0]]), dtype=np.float32
            )
            rotations = np.asarray(
                obj.get("rotation", [[0, 0, 0, 1]]), dtype=np.float32
            )
            # A recorded rollout may start from a random reference phase.
            # Its same-stem objects sidecar contains the exact object poses
            # seen by physics and must override SceneLib's frame-zero timeline.
            if recorded_objects is not None and j < len(recorded_objects):
                recorded_obj = recorded_objects[j]
                translations = np.asarray(
                    recorded_obj["translation"], dtype=np.float32
                )
                rotations = np.asarray(recorded_obj["rotation"], dtype=np.float32)
            if args.freeze_scene_objects and len(translations) > 1:
                translations = translations[:1]
                rotations = rotations[:1]
            elif args.smooth_object_window > 1 and len(translations) > 1:
                translations = smooth_trajectory(
                    translations, args.smooth_object_window
                )
                # Resolve the q/-q ambiguity before filtering, then normalize.
                rotations = rotations.copy()
                for q_idx in range(1, len(rotations)):
                    if np.dot(rotations[q_idx - 1], rotations[q_idx]) < 0.0:
                        rotations[q_idx] *= -1.0
                rotations = smooth_trajectory(rotations, args.smooth_object_window)
                rotations /= np.maximum(
                    np.linalg.norm(rotations, axis=-1, keepdims=True), 1.0e-8
                )
            # Skip padding objects parked below the world.
            if translations[0, 2] < -5:
                continue
            rot_xyzw = rotations[0]
            wxyz = quat_xyzw_to_wxyz(np.array(rot_xyzw))
            h = server.scene.add_mesh_trimesh(
                f"/scene/obj_{j}", mesh, position=translations[0], wxyz=wxyz
            )
            current_scene_handles.append(h)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            max_per_object = max(
                256,
                int(args.ego_max_scene_points)
                // max(1, len(scene.get("objects", []))),
            )
            if len(vertices) > max_per_object:
                sample_indices = np.linspace(
                    0, len(vertices) - 1, max_per_object, dtype=np.int64
                )
                vertices = vertices[sample_indices]
            is_static = bool(obj.get("options", {}).get("fix_base_link", False))
            current_scene_objects.append(
                {
                    "handle": h,
                    "translations": translations,
                    "rotations": rotations,
                    "local_points": vertices,
                    "static": is_static,
                }
            )

    def _object_world_points(frame_index: int):
        point_chunks = []
        static_chunks = []
        id_chunks = []
        next_id = 0
        for item in current_scene_objects:
            local = item["local_points"]
            pose_index = min(frame_index, len(item["translations"]) - 1)
            rot = tf.quaternion_matrix(
                quat_xyzw_to_wxyz(item["rotations"][pose_index])
            )[:3, :3]
            world = local @ rot.T + item["translations"][pose_index]
            point_chunks.append(world)
            static_chunks.append(
                np.full(len(world), item["static"], dtype=np.bool_)
            )
            id_chunks.append(np.arange(next_id, next_id + len(world), dtype=np.int64))
            next_id += len(world)
        if not point_chunks:
            return empty_points, np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.int64)
        return (
            np.concatenate(point_chunks, axis=0),
            np.concatenate(static_chunks),
            np.concatenate(id_chunks),
        )

    def _ego_visibility_for_frame(mi: int, frame_index: int):
        world_points, static_mask, point_ids = _object_world_points(frame_index)
        if len(world_points) == 0:
            return world_points, static_mask, point_ids, np.zeros(0, dtype=np.bool_), None
        if ego_cameras is not None:
            camera_info = ego_cameras[mi]
            world_from_camera = np.asarray(
                camera_info["world_from_camera"][frame_index], dtype=np.float64
            )
            camera_pos = world_from_camera[:3, 3]
            camera_rot = world_from_camera[:3, :3]
            tan_h = 0.5 * float(camera_info["width"]) / float(
                camera_info["fx"][frame_index]
            )
            tan_v = 0.5 * float(camera_info["height"]) / float(
                camera_info["fy"][frame_index]
            )
            camera_points = (world_points - camera_pos) @ camera_rot
            # EgoBody's PV pose uses the OpenGL camera convention: local -Z
            # is forward.  The official renderer applies diag(1, -1, -1)
            # before OpenCV projection. Viser frusta instead look along +Z.
            depth = -camera_points[:, 2]
            vertical_sign = -1.0
            frustum_rot = camera_rot @ np.diag([1.0, -1.0, -1.0])
        else:
            motion = motions[mi]
            head_id = skel["bodies"].index("Head")
            head_pos = np.asarray(motion["gts"][frame_index, head_id], dtype=np.float64)
            head_q = np.asarray(motion["grs"][frame_index, head_id], dtype=np.float64)
            head_rot = tf.quaternion_matrix(quat_xyzw_to_wxyz(head_q))[:3, :3]
            camera_pos = head_pos + head_rot @ np.array([0.0, -0.08, 0.03])
            camera_rot = head_rot @ tf.rotation_matrix(
                np.pi * 0.5, [1.0, 0.0, 0.0]
            )[:3, :3]
            camera_points = (world_points - camera_pos) @ camera_rot
            depth = camera_points[:, 2]
            vertical_sign = 1.0
            frustum_rot = camera_rot
            tan_h = np.tan(np.deg2rad(args.ego_horizontal_fov) * 0.5)
            tan_v = np.tan(np.deg2rad(args.ego_vertical_fov) * 0.5)
        safe_depth = np.maximum(depth, 1.0e-6)
        x_ratio = camera_points[:, 0] / safe_depth
        z_ratio = vertical_sign * camera_points[:, 1] / safe_depth
        in_frustum = (
            (depth >= args.ego_near)
            & (depth <= args.ego_far)
            & (np.abs(x_ratio) <= tan_h)
            & (np.abs(z_ratio) <= tan_v)
        )
        width, height = 64, 36
        px = np.clip((((x_ratio / tan_h) + 1.0) * 0.5 * width).astype(np.int64), 0, width - 1)
        py = np.clip((((z_ratio / tan_v) + 1.0) * 0.5 * height).astype(np.int64), 0, height - 1)
        pixel = py * width + px
        z_buffer = np.full(width * height, np.inf, dtype=np.float64)
        np.minimum.at(z_buffer, pixel[in_frustum], depth[in_frustum])
        visible = in_frustum & (depth <= z_buffer[pixel] + 0.04)
        camera = (camera_pos, frustum_rot, tan_h, tan_v)
        return world_points, static_mask, point_ids, visible, camera

    def _frustum_pose(camera):
        camera_pos, world_from_camera, tan_h, tan_v = camera
        del tan_h, tan_v
        transform = np.eye(4)
        transform[:3, :3] = world_from_camera
        return camera_pos, tf.quaternion_from_matrix(transform)

    def _head_camera_pose(motion, skeleton, frame_index, x_offset=0.0):
        head_id = skeleton["bodies"].index("Head")
        head_pos = np.asarray(
            motion["gts"][frame_index, head_id], dtype=np.float64
        ).copy()
        head_pos[0] += x_offset
        head_q = np.asarray(
            motion["grs"][frame_index, head_id], dtype=np.float64
        )
        head_rot = tf.quaternion_matrix(quat_xyzw_to_wxyz(head_q))[:3, :3]
        camera_pos = head_pos + head_rot @ np.array([0.0, -0.08, 0.03])
        camera_rot = head_rot @ tf.rotation_matrix(
            np.pi * 0.5, [1.0, 0.0, 0.0]
        )[:3, :3]
        return _frustum_pose(
            (
                camera_pos,
                camera_rot,
                np.tan(np.deg2rad(args.ego_horizontal_fov) * 0.5),
                np.tan(np.deg2rad(args.ego_vertical_fov) * 0.5),
            )
        )

    def update_head_camera_frusta(mi: int, frame_index: int):
        student_head_frustum_handle.visible = show_student_head_frustum.value
        student_pos, student_wxyz = _head_camera_pose(
            motions[mi], skel, frame_index
        )
        student_head_frustum_handle.position = student_pos
        student_head_frustum_handle.wxyz = student_wxyz

        compare_visible = cmp_motions is not None and show_compare_head_frustum.value
        compare_head_frustum_handle.visible = compare_visible
        if compare_visible:
            cmp_mi = min(mi, len(cmp_motions) - 1)
            cmp_fi = min(frame_index, cmp_motions[cmp_mi]["gts"].shape[0] - 1)
            compare_offset = float(args.offset)
            if args.align_compare_root:
                primary_root0 = motions[mi]["gts"][0, 0]
                compare_root0 = cmp_motions[cmp_mi]["gts"][0, 0]
                # Head-camera helper only accepts the displayed X offset; root
                # alignment in Y/Z is applied explicitly below.
                alignment = primary_root0 - compare_root0
            else:
                alignment = np.zeros(3, dtype=np.float64)
            compare_pos, compare_wxyz = _head_camera_pose(
                cmp_motions[cmp_mi], cmp_skel, cmp_fi, compare_offset
            )
            compare_head_frustum_handle.position = compare_pos + alignment
            compare_head_frustum_handle.wxyz = compare_wxyz

    def update_ego_visibility(mi: int, frame_index: int):
        state = ego_visibility_state
        for handle in current_scene_handles:
            handle.visible = show_scene_mesh.value
        for key, checkbox in (
            ("current", show_ego_current),
            ("memory", show_ego_memory),
            ("unseen", show_ego_unseen),
        ):
            if ego_layer_handles[key] is not None:
                ego_layer_handles[key].visible = checkbox.value
        ego_frustum_handle.visible = show_ego_frustum.value
        signature = (mi, frame_index)
        stride = max(1, int(args.ego_visibility_stride))
        if (
            state["render_signature"] is not None
            and state["render_signature"][0] == mi
            and frame_index > state["last_frame"]
            and frame_index % stride != 0
        ):
            return
        if state["render_signature"] == signature:
            return
        state["render_signature"] = signature
        if state["scene_index"] != mi or frame_index < state["last_frame"]:
            state["scene_index"] = mi
            state["seen_static_ids"].clear()
            state["last_frame"] = -1
        # Reconstruct causal memory when the user jumps forward with the slider.
        for history_frame in range(state["last_frame"] + 1, frame_index + 1):
            _, static, ids, visible, _ = _ego_visibility_for_frame(mi, history_frame)
            state["seen_static_ids"].update(ids[visible & static].tolist())
        state["last_frame"] = frame_index
        points, static, ids, visible, camera = _ego_visibility_for_frame(mi, frame_index)
        seen = np.isin(ids, np.fromiter(state["seen_static_ids"], dtype=np.int64))
        remembered_hidden = static & seen & ~visible
        unseen = static & ~seen

        def replace_layer(key, path, layer_points, color, point_size, visible_value):
            old_handle = ego_layer_handles[key]
            if old_handle is not None:
                old_handle.remove()
            colors = np.tile(
                np.asarray(color, dtype=np.uint8), (len(layer_points), 1)
            )
            ego_layer_handles[key] = server.scene.add_point_cloud(
                path,
                layer_points.astype(np.float32),
                colors=colors,
                point_size=point_size,
                point_shape="circle",
                visible=visible_value,
            )

        replace_layer(
            "current", "/ego/currently_visible", points[visible],
            [30, 255, 90], 0.022, show_ego_current.value,
        )
        replace_layer(
            "memory", "/ego/seen_but_hidden", points[remembered_hidden],
            [255, 145, 30], 0.016, show_ego_memory.value,
        )
        replace_layer(
            "unseen", "/ego/not_seen_yet", points[unseen],
            [70, 75, 85], 0.011, show_ego_unseen.value,
        )
        ego_current_count.value = str(int(visible.sum()))
        ego_memory_count.value = str(int(remembered_hidden.sum()))
        ego_unseen_count.value = str(int(unseen.sum()))
        if camera is not None:
            frustum_pos, frustum_wxyz = _frustum_pose(camera)
            ego_frustum_handle.position = frustum_pos
            ego_frustum_handle.wxyz = frustum_wxyz

    def update_scene_frame(frame_index: int):
        for item in current_scene_objects:
            pose_index = min(frame_index, len(item["translations"]) - 1)
            item["handle"].position = item["translations"][pose_index]
            item["handle"].wxyz = quat_xyzw_to_wxyz(
                item["rotations"][pose_index]
            )

    def load_scene_for_motion(mi: int):
        """Load/swap scene for the current motion."""
        if mi == current_scene_idx[0]:
            return
        _clear_scene()
        current_scene_idx[0] = mi

        # Packaged scene .pt — index directly by motion index
        if packaged_scenes is not None:
            scenes = packaged_scenes["original_scenes"]
            if mi < len(scenes):
                _render_scene_objects(scenes[mi])
                n_real = sum(
                    1
                    for o in scenes[mi]["objects"]
                    if o.get("object_dims", (0, 0, 0, 0, 0, -10))[5] > -5
                )
                print(f"Scene {mi}: {n_real} objects")
            return

        # Per-file scene loading from --scene-dir
        if scene_dir is None or not scene_dir.exists():
            return

        motion_name = motions[mi].get("name", f"motion_{mi}")
        base = re.sub(r"(_[abcd])?(\.(motion|pkl))?$", "", motion_name)

        # Per-file .pt scene
        pt_path = scene_dir / f"{base}.pt"
        if pt_path.exists():
            try:
                sd = torch.load(str(pt_path), weights_only=False, map_location="cpu")
                if "original_scenes" in sd and sd["original_scenes"]:
                    _render_scene_objects(sd["original_scenes"][0])
            except Exception as e:
                print(f"Warning: failed to load {pt_path}: {e}")

    # Track handles for cleanup
    body_handles = {}
    bone_handles = {}
    cmp_body_handles = {}
    cmp_bone_handles = {}

    def update_frame_range():
        mi = motion_idx_slider.value
        n_frames = motions[mi]["gts"].shape[0]
        frame_slider.max = n_frames - 1
        if frame_slider.value >= n_frames:
            frame_slider.value = 0
        load_scene_for_motion(mi)

    @motion_idx_slider.on_update
    def _(_):
        update_frame_range()

    def build_robot_body_meshes(robot_name, regions):
        if robot_name == "soma23":
            # Keep SOMA's MJCF-derived body geometry and indices in one source.
            from visualize_soma_scenes import build_body_local_meshes

            return build_body_local_meshes()
        return _build_body_meshes(SMPL_GEOM_DEFS, regions)

    primary_body_meshes = build_robot_body_meshes(robot, body_to_region)
    comparison_body_meshes = (
        build_robot_body_meshes(cmp_robot, cmp_body_to_region)
        if cmp_motions is not None
        else None
    )

    def render_skeleton(
        gts,
        grs,
        contacts,
        skel_info,
        body_reg,
        handles_body,
        handles_bone,
        body_meshes,
        prefix="",
        x_offset=0.0,
    ):
        """Render one skeleton frame using body-local capsule/box meshes."""
        num_b = gts.shape[0]

        for i in range(num_b):
            pos = gts[i].copy()
            pos[0] += x_offset
            rot_xyzw = grs[i]
            wxyz = quat_xyzw_to_wxyz(rot_xyzw)
            name = f"{prefix}{skel_info['bodies'][i]}"

            # Get body mesh (capsule/box from MJCF)
            body_mesh = body_meshes.get(i)
            if body_mesh is None:
                continue

            # Keep handles alive and only update their transforms. Removing and
            # re-adding all bodies each browser frame causes visible flashing.
            if name not in handles_body:
                handles_body[name] = server.scene.add_mesh_trimesh(
                    f"/skel/{name}",
                    body_mesh,
                    position=pos,
                    wxyz=wxyz,
                )
            else:
                handles_body[name].position = pos
                handles_body[name].wxyz = wxyz

    # Load initial scene
    load_scene_for_motion(initial_motion_index)

    # Main loop
    prev_time = time.time()
    frame_accum = 0.0

    while True:
        mi = motion_idx_slider.value
        load_scene_for_motion(mi)
        fi = frame_slider.value
        motion = motions[mi]
        update_scene_frame(fi)
        if packaged_scenes is not None:
            update_ego_visibility(mi, fi)
        update_head_camera_frusta(mi, fi)

        gts = motion["gts"][fi]
        grs = motion["grs"][fi]
        contacts = motion["contacts"][fi] if motion["contacts"] is not None else None
        if contacts is not None:
            contacts = (
                contacts.numpy() if isinstance(contacts, torch.Tensor) else contacts
            )

        render_skeleton(
            gts,
            grs,
            contacts,
            skel,
            body_to_region,
            body_handles,
            bone_handles,
            primary_body_meshes,
        )

        if cmp_motions is not None:
            cmp_mi = min(mi, len(cmp_motions) - 1)
            cmp_fi = min(fi, cmp_motions[cmp_mi]["gts"].shape[0] - 1)
            cmp = cmp_motions[cmp_mi]
            cmp_contacts = (
                cmp["contacts"][cmp_fi] if cmp["contacts"] is not None else None
            )
            if cmp_contacts is not None:
                cmp_contacts = (
                    cmp_contacts.numpy()
                    if isinstance(cmp_contacts, torch.Tensor)
                    else cmp_contacts
                )
            cmp_gts = cmp["gts"][cmp_fi].copy()
            if args.align_compare_root:
                primary_root0 = motions[mi]["gts"][0, 0]
                compare_root0 = cmp["gts"][0, 0]
                cmp_gts += primary_root0 - compare_root0
            render_skeleton(
                cmp_gts,
                cmp["grs"][cmp_fi],
                cmp_contacts,
                cmp_skel,
                cmp_body_to_region,
                cmp_body_handles,
                cmp_bone_handles,
                comparison_body_meshes,
                prefix="cmp_",
                x_offset=args.offset,
            )

        # Capture the browser's actual Viser render, including meshes, colors,
        # lighting, and the user's current camera. This is independent of
        # Isaac Sim's RTX/Vulkan viewport and therefore works in WSLg.
        if record_state["requested"] and not record_state["active"]:
            clients = list(server.get_clients().values())
            if clients:
                record_state["requested"] = False
                record_state["active"] = True
                record_state["client"] = clients[0]
                record_state["frames"] = []
                Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                record_state["path"] = str(
                    Path(args.record_dir) / f"{Path(args.motion_file).stem}-{stamp}.mp4"
                )
                record_status.value = "Recording..."
            else:
                record_status.value = "Open the Viser page before recording"

        if record_state["active"]:
            try:
                client = record_state["client"]
                image = client.camera.get_render(
                    height=args.record_height,
                    width=args.record_width,
                    transport_format="jpeg",
                )
                record_state["frames"].append(image)
                next_frame = fi + 1
                if next_frame >= motion["gts"].shape[0]:
                    imageio.mimsave(
                        record_state["path"],
                        record_state["frames"],
                        fps=args.record_fps,
                        codec="libx264",
                    )
                    record_status.value = f"Saved: {record_state['path']}"
                    print(f"Viser MP4 saved to {record_state['path']}")
                    record_state["active"] = False
                    record_state["client"] = None
                    record_state["frames"] = []
                else:
                    frame_slider.value = next_frame
            except Exception as exc:
                record_status.value = f"Record failed: {exc}"
                print(f"Warning: Viser recording failed: {exc}")
                record_state["active"] = False
                record_state["client"] = None
                record_state["frames"] = []

        # Playback
        now = time.time()
        dt = now - prev_time
        prev_time = now

        if playing.value:
            fps = motion["fps"] if isinstance(motion["fps"], (int, float)) else 30
            frame_accum += dt * fps * speed.value
            if frame_accum >= 1.0:
                advance = int(frame_accum)
                frame_accum -= advance
                new_frame = fi + advance
                max_frame = motion["gts"].shape[0] - 1
                if new_frame > max_frame:
                    new_frame = 0
                frame_slider.value = new_frame

        time.sleep(0.016)  # ~60 fps render


if __name__ == "__main__":
    main()
