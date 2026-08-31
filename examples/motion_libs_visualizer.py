# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Motion Visualizer with Smoothness Metrics
# Supports normalized jerk, oscillation index, and purposeful jerk metrics
# Uses threshold-based highlighting similar to the original visualizer

from typing import Dict, List
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

FPS = 30

# Parse arguments first (argparse is safe, doesn't import torch)
parser = argparse.ArgumentParser(
    description="Motion Visualizer with Smoothness Metrics"
)
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    required=True,
    help="Paths to MotionLib .pt files (e.g., predicted_motion_lib.pt motion_lib.pt). Each file will be displayed in a separate environment.",
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["isaacgym", "isaaclab", "newton"],
    default="isaacgym",
    help="Simulator to use (isaacgym, isaaclab, newton)",
)
parser.add_argument(
    "--robot",
    type=str,
    choices=["g1", "h1_2", "smpl", "soma23"],
    default="g1",
    help="Robot to load (g1, h1_2, smpl, or soma23)",
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier (1.0 = normal speed)",
)
parser.add_argument(
    "--smoothness_threshold",
    type=float,
    default=6500.0,
    help="Smoothness threshold to highlight bodies (higher values = less smooth). FPS-invariant metric.",
)
parser.add_argument(
    "--metric",
    type=str,
    choices=["nj", "oi", "pj"],
    default="nj",
    help="Smoothness metric: 'nj' for normalized jerk, 'oi' for oscillation index, 'pj' for purposeful jerk",
)
parser.add_argument(
    "--use-data-vel",
    action="store_true",
    help="Use stored rigid_body_vel from motion data instead of computing velocities via finite differences (default: False, use finite differences)",
)
parser.add_argument(
    "--window_sec",
    type=float,
    default=0.4,
    help="Sliding window length in seconds for computing smoothness metrics",
)
parser.add_argument(
    "--origin_xy",
    type=float,
    nargs=2,
    default=[0.0, 0.0],
    help="Target x,y position to move all motions to (default: 0.0 0.0)",
)
parser.add_argument(
    "--scene_file",
    type=str,
    default=None,
    help="Motion-aligned SceneLib .pt file. Enables world-coordinate scene playback.",
)
parser.add_argument(
    "--scene_asset_root",
    type=str,
    default=None,
    help="Root for relative mesh paths in --scene_file (TRUMANS dataset root).",
)
parser.add_argument(
    "--scene_index",
    type=int,
    default=None,
    help="Scene/motion index to inspect. Defaults to --start_motion_index.",
)
parser.add_argument(
    "--start_motion_index", type=int, default=0, help="Initial motion index."
)
parser.add_argument(
    "--mesh_collision_approximation",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere"],
    default=None,
    help="Optional collision approximation; requires USD scene assets.",
)
parser.add_argument(
    "--contact_labels",
    type=str,
    default=None,
    help="Per-clip contact NPZ used for PhysX reference-replay validation.",
)
parser.add_argument(
    "--validate_contacts",
    action="store_true",
    help="Replay one motion once and compare geometric labels with simulator contacts.",
)
parser.add_argument(
    "--contact_label_key",
    choices=[
        "source_contact",
        "target_contact",
        "target_physics_contact",
        "intended_contact",
        "training_contact",
    ],
    default="target_physics_contact",
    help="Label tensor to compare. target_physics_contact matches PhysX contact offset.",
)
parser.add_argument(
    "--contact_force_threshold",
    type=float,
    default=1.0,
    help="PhysX net-force threshold in Newtons for a positive contact.",
)
parser.add_argument(
    "--contact_object_mode",
    choices=["kinematic", "dynamic"],
    default="kinematic",
    help=(
        "During contact validation, drive scene objects kinematically along the "
        "released trajectory or leave them as free dynamic bodies. Kinematic is "
        "the deterministic reference-replay default."
    ),
)
parser.add_argument(
    "--validation_frames",
    type=int,
    default=0,
    help="Frames to validate; 0 means one complete motion.",
)
parser.add_argument(
    "--contact_report",
    type=str,
    default=None,
    help="Output JSON report. Defaults beside --contact_labels.",
)
parser.add_argument(
    "--replay_trace",
    type=str,
    default=None,
    help=(
        "NPZ output containing reference/actual body poses, forces and contact "
        "booleans for Viser replay inspection. Defaults beside --contact_labels."
    ),
)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import torch  # noqa: E402
import numpy as np  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

from protomotions.simulator.base_simulator.config import (  # noqa: E402
    VisualizationMarkerConfig,
    MarkerConfig,
    MarkerState,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.robot_configs.base import ControlType  # noqa: E402
from protomotions.components.motion_lib import MotionLib  # noqa: E402
from protomotions.components.scene_lib import (  # noqa: E402
    SceneLib,
    MeshSceneObject,
    Scene,
    ObjectOptions,
    SceneLibConfig,
    ReplicationMethod,
    SubsetMethod,
)
import os  # noqa: E402


@dataclass
class RobotSpec:
    """Robot specification with joint/body names for visualization"""

    # Body names to visualize (these are the rigid body names, not joint names)
    viz_bodies: List[str]


# Define robot specifications
ROBOT_SPECS = {
    "g1": RobotSpec(
        viz_bodies=[],
    ),
    "h1_2": RobotSpec(
        viz_bodies=[],
    ),
    "smpl": RobotSpec(
        viz_bodies=[],
    ),
    "soma23": RobotSpec(
        viz_bodies=[],
    ),
}


# ----- Smoothness Metrics Implementation -----
def _diff(x, dt):
    """Compute finite difference with given time step"""
    return (x[1:] - x[:-1]) / dt


def normalized_jerk_from_vel(vel, dt, eps=0.1):
    """
    Compute normalized jerk from velocity trajectory.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_nj: [N] normalized jerk per body
        mean_nj: scalar mean normalized jerk

        --smoothness_threshold 6500.0 --window_sec 0.4 (using finite differences, which is default) seems to be good qualitative measures
        Uses T^5 for dimensionless, FPS-invariant normalization.
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    j = _diff(a, dt)  # [T-2, N, 3]

    speed = torch.linalg.norm(vel, dim=-1)  # [T, N]
    jnorm2 = torch.linalg.norm(j, dim=-1) ** 2  # [T-2, N]

    T_tot = vel.shape[0] * dt
    L = (speed * dt).sum(dim=0).clamp_min(eps)  # [N] - path length per body
    int_j2 = (jnorm2 * dt).sum(dim=0)  # [N] - integrated squared jerk
    # Using T^5 (not T^3) for dimensionless, FPS-invariant normalization
    nj = (T_tot**5 * int_j2) / (L**2 + eps)  # [N] - normalized jerk
    return nj, nj.mean()


def oscillation_index_from_vel(vel, dt, eps=0.001):
    """
    Compute oscillation index from velocity trajectory.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_oi: [N] oscillation index per body (0-2, higher = more oscillatory)
        mean_oi: scalar mean oscillation index

        threshold 1.2 (slightly larger than 1) seems meaningful
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    a1, a2 = a[:-1], a[1:]  # [T-2, N, 3]

    fps = 1.0 / dt
    a1 = a1 / fps
    a2 = a2 / fps

    num = (a1 * a2).sum(-1)  # [T-2, N]
    den = (torch.linalg.norm(a1, dim=-1) * torch.linalg.norm(a2, dim=-1)).clamp_min(eps)
    # print(torch.mean(den))
    cos = (num / den).clamp(-1, 1)  # [T-2, N]
    oi = (1 - cos).mean(dim=0)  # [N]
    return oi, oi.mean()


def purposeful_jerk_from_vel(vel, dt, eps=1e-8):
    """
    Compute purposeful jerk from velocity trajectory.
    High values indicate jerk that coincides with velocity direction changes.
    Args:
        vel: [T, N, 3] velocity trajectory
        dt: time step
    Returns:
        per_body_pj: [N] purposeful jerk per body
        mean_pj: scalar mean purposeful jerk
    """
    a = _diff(vel, dt)  # [T-1, N, 3]
    j = _diff(a, dt)  # [T-2, N, 3]
    v1, v2 = vel[:-1], vel[1:]  # [T-1, N, 3]

    num = (v1 * v2).sum(-1)  # [T-1, N]
    den = (torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)).clamp_min(eps)
    misalign = 1 - (num / den).clamp(-1, 1)  # [T-1, N], in [0,2]
    jn = torch.linalg.norm(j, dim=-1)  # [T-2, N]

    # Align shapes: use minimum length
    Tm = min(misalign.shape[0] - 1, jn.shape[0])
    pj = (jn[:Tm] * misalign[1 : 1 + Tm]).mean(dim=0)  # [N]
    return pj, pj.mean()


def create_checkerboard_ground(
    num_envs: int, device: torch.device, simulator_type: str = "isaacgym"
) -> SceneLib:
    """
    Create a visual checkerboard ground plane using a textured mesh.

    Args:
        num_envs: Number of environments
        device: Torch device
        simulator_type: Type of simulator (isaacgym, isaaclab, etc.)

    Returns:
        SceneLib with checkerboard ground for each environment
    """
    # Get path to the checkerboard asset (URDF for IsaacGym, USD for IsaacLab)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    checkerboard_dir = os.path.join(
        project_root, "protomotions/data/assets/checkerboard"
    )

    if simulator_type == "isaaclab":
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.usda")
        asset_type = "USD"
    else:
        # IsaacGym, Newton, Genesis use URDF
        asset_path = os.path.join(checkerboard_dir, "checkerboard_ground.urdf")
        asset_type = "URDF"

    if not os.path.exists(asset_path):
        print(f"Warning: Checkerboard ground {asset_type} not found at {asset_path}")
        print(f"Assets should be in: {checkerboard_dir}")
        return None

    # Get texture path for IsaacGym (IsaacLab loads it from USD)
    texture_path = None
    if simulator_type != "isaaclab":
        texture_file = os.path.join(checkerboard_dir, "checkerboard_texture.png")
        if os.path.exists(texture_file):
            texture_path = texture_file

    # Create scenes for each environment
    # IMPORTANT: Each scene needs its own MeshSceneObject instance,
    # otherwise attributes get overwritten during _process_scene_objects()
    scenes = []
    for _ in range(num_envs):
        ground_mesh = MeshSceneObject(
            object_path=asset_path,
            translation=(0.0, 0.0, -0.005),  # Slightly below zero
            rotation=(0.0, 0.0, 0.0, 1.0),  # No rotation (x, y, z, w)
            options=ObjectOptions(
                fix_base_link=True,  # Static object
                vhacd_enabled=False,  # Disable convex decomposition for simple plane
                texture_path=texture_path,  # Texture for IsaacGym (None for IsaacLab)
            ),
        )
        scenes.append(Scene(objects=[ground_mesh], offset=(0.0, 0.0)))

    # Configure scene lib
    scene_lib_config = SceneLibConfig(
        scene_file=None,  # No file, using inline scene
        replicate_method=ReplicationMethod.SEQUENTIAL,
        subset_method=SubsetMethod.FIRST,
        pointcloud_samples_per_object=None,
    )

    # Return a SceneLib without terrain (avoids collision geometry in simulators)
    return SceneLib(
        config=scene_lib_config,
        num_envs=num_envs,
        scenes=scenes,
        device=device,
        terrain=None,  # No terrain to avoid unwanted collisions
    )


class MotionVisualizerSmoothness:
    def __init__(
        self,
        motion_files: List[str],
        robot_name: str = "g1",
        simulator_type: str = "isaacgym",
        headless: bool = False,
        cpu_only: bool = False,
        extra_simulator_params: dict = None,
        playback_speed: float = 1.0,
        metric: str = "nj",
        use_data_vel: bool = False,
        window_sec: float = 2.0,
        scene_file: str = None,
        scene_asset_root: str = None,
        scene_index: int = None,
        start_motion_index: int = 0,
        mesh_collision_approximation: str = None,
        contact_labels: str = None,
        validate_contacts: bool = False,
        contact_label_key: str = "target_physics_contact",
        contact_force_threshold: float = 1.0,
        contact_object_mode: str = "kinematic",
        validation_frames: int = 0,
        contact_report: str = None,
        replay_trace: str = None,
    ):
        self.motion_files = [Path(f) for f in motion_files]
        self.robot_name = robot_name
        self.robot_spec = ROBOT_SPECS[robot_name]
        self.num_envs = len(motion_files)
        self.simulator_type = simulator_type
        self.headless = headless
        self.playback_speed = playback_speed
        self.device = torch.device("cuda:0" if not cpu_only else "cpu")
        self.smoothness_threshold = args.smoothness_threshold
        self.metric = metric
        self.use_data_vel = use_data_vel  # If False (default), use finite differences
        self.window_frames = max(4, int(round(window_sec * FPS)))
        self.scene_file = scene_file
        self.scene_asset_root = scene_asset_root
        self.scene_index = start_motion_index if scene_index is None else scene_index
        self.validate_contacts = validate_contacts
        self.contact_label_key = contact_label_key
        self.contact_force_threshold = contact_force_threshold
        self.contact_object_mode = contact_object_mode
        self.validation_frames = validation_frames
        self.contact_report = Path(contact_report) if contact_report else None
        self.replay_trace_path = Path(replay_trace) if replay_trace else None
        self.contact_labels_path = Path(contact_labels) if contact_labels else None
        self.contact_reference = None
        self.contact_reference_by_object = None
        self.contact_object_names = None
        self.contact_tp = None
        self.contact_fp = None
        self.contact_fn = None
        self.contact_pair_tp = None
        self.contact_pair_fp = None
        self.contact_pair_fn = None
        self.validated_contact_frames = 0
        self._trace_reference_pos = []
        self._trace_pre_step_pos = []
        self._trace_actual_pos = []
        self._trace_reference_rot = []
        self._trace_pre_step_rot = []
        self._trace_actual_rot = []
        self._trace_reference_object_pos = []
        self._trace_pre_step_object_pos = []
        self._trace_post_step_object_pos = []
        self._trace_reference_object_rot = []
        self._trace_pre_step_object_rot = []
        self._trace_post_step_object_rot = []
        self._trace_force = []
        self._trace_object_force = []
        self._trace_expected = []
        self._trace_expected_by_object = []
        self._trace_actual = []
        self._trace_actual_by_object = []
        if self.validate_contacts:
            if self.num_envs != 1:
                raise ValueError("Contact validation requires exactly one --motion_files input")
            if self.scene_file is None:
                raise ValueError("Contact validation requires --scene_file")
            if self.contact_labels_path is None:
                raise ValueError("Contact validation requires --contact_labels")
            with np.load(self.contact_labels_path) as labels:
                if self.contact_label_key not in labels.files:
                    raise KeyError(
                        f"{self.contact_label_key!r} is absent from {self.contact_labels_path}"
                    )
                reference = labels[self.contact_label_key]
                if reference.ndim == 3:
                    self.contact_reference_by_object = torch.from_numpy(
                        reference.astype(bool)
                    ).to(self.device)
                    self.contact_object_names = labels["object_names"].astype(str)
                    reference = reference.any(axis=-1)
                self.contact_reference = torch.from_numpy(reference.astype(bool)).to(self.device)
            if self.contact_report is None:
                self.contact_report = self.contact_labels_path.with_name(
                    f"{self.contact_labels_path.stem}.physx_{self.contact_label_key}.json"
                )
            if self.replay_trace_path is None:
                self.replay_trace_path = self.contact_labels_path.with_name(
                    f"{self.contact_labels_path.stem}.physx_{self.contact_label_key}.trace.npz"
                )

        # Load motion libraries (.pt files)
        from protomotions.components.motion_lib import MotionLibConfig

        self.motion_libs = [
            MotionLib(
                config=MotionLibConfig(motion_file=str(motion_file)), device=self.device
            )
            for motion_file in self.motion_files
        ]

        # Scene-aware playback must retain the dataset's metric world frame.
        if self.scene_file is None:
            for i, motion_lib in enumerate(self.motion_libs):
                target_xy = torch.tensor(args.origin_xy, device=self.device)
                target_xy = target_xy + torch.tensor([1.0 * i, 0.0], device=self.device)
                print(f"Translating motion library {i} to origin {target_xy}")
                motion_lib.translate_all_motions_to_origin(target_xy)
        else:
            print("Preserving motion world coordinates for scene alignment")

        # Motion playback state
        self.current_motion_idx = start_motion_index
        self.current_frame = 0
        # Use the first motion lib to determine total motions and current motion length
        self.total_motions = self.motion_libs[0].num_motions()
        self.current_motion_length = (
            self.motion_libs[0]
            .get_motion_num_frames(None)[self.current_motion_idx]
            .item()
        )
        print(
            f"Loaded {len(self.motion_files)} motion files with {self.total_motions} motions each"
        )
        print(f"Motion files: {[str(f) for f in self.motion_files]}")
        print(
            f"Current motion {self.current_motion_idx} has {self.current_motion_length} frames"
        )

        # Load robot configuration using factory function
        self.robot_cfg = robot_config(robot_name)

        # Store kinematic info for later use
        self.kinematic_info = self.robot_cfg.kinematic_info
        if self.validate_contacts:
            body_count = self.kinematic_info.num_bodies
            if tuple(self.contact_reference.shape) != (
                self.current_motion_length,
                body_count,
            ):
                raise ValueError(
                    f"Contact labels {tuple(self.contact_reference.shape)} do not match "
                    f"motion {(self.current_motion_length, body_count)}"
                )
            self.contact_tp = torch.zeros(body_count, dtype=torch.long)
            self.contact_fp = torch.zeros(body_count, dtype=torch.long)
            self.contact_fn = torch.zeros(body_count, dtype=torch.long)
            if self.contact_reference_by_object is not None:
                object_count = self.contact_reference_by_object.shape[-1]
                pair_shape = (body_count, object_count)
                self.contact_pair_tp = torch.zeros(pair_shape, dtype=torch.long)
                self.contact_pair_fp = torch.zeros(pair_shape, dtype=torch.long)
                self.contact_pair_fn = torch.zeros(pair_shape, dtype=torch.long)
            # The visualizer normally creates no contact sensors. Validation
            # needs one sensor per humanoid body. IsaacLab's filtered matrix
            # preserves the body-by-object attribution used by scene labels.
            self.robot_cfg.contact_bodies = list(self.kinematic_info.body_names)

        # Create simulator configuration using factory function
        self.simulator_cfg = simulator_config(
            simulator_type,
            self.robot_cfg,
            headless=headless,
            num_envs=self.num_envs,
            experiment_name="motion_viz_smoothness",
        )
        if (
            self.validate_contacts
            and self.contact_reference_by_object is not None
            and simulator_type == "isaaclab"
        ):
            self.simulator_cfg.enable_body_contact_filter_matrix = True

        # Override robot asset settings for motion visualization
        self.robot_cfg.asset.disable_gravity = True
        self.robot_cfg.asset.fix_base_link = False
        self.robot_cfg.asset.self_collisions = False

        # Use torque control (zero torque) to maintain poses
        self.robot_cfg.control.control_type = ControlType.TORQUE

        # Create visualization markers
        self.viz_markers = self._create_visualization_markers()

        # Initialize body markers after kinematic info is loaded
        self._initialize_body_markers()

        # Create custom key handlers. R must be REGISTERED here (not left to a legacy
        # simulator.user_requested_reset flag that the UserInterface sim layer no longer sets) — otherwise
        # IsaacGym, which only delivers keys it subscribed via subscribe_viewer_keyboard_event, never sees it.
        custom_key_handlers = {
            "R": self._request_next_motion,  # Key R: switch to the next motion
            "1": self.increase_speed,  # Key 1: Increase playback speed
            "2": self.decrease_speed,  # Key 2: Decrease playback speed
            "3": self.increase_smoothness_threshold,  # Key 3: Increase smoothness threshold
            "4": self.decrease_smoothness_threshold,  # Key 4: Decrease smoothness threshold
        }

        if self.scene_file is None:
            print("Creating checkerboard ground plane...")
            scene_lib = create_checkerboard_ground(
                self.num_envs, self.device, self.simulator_type
            )
            print("Checkerboard ground loaded successfully")
        else:
            scene_lib = SceneLib(
                config=SceneLibConfig(
                    scene_file=self.scene_file,
                    asset_root=self.scene_asset_root,
                    scene_indices=[self.scene_index],
                    subset_method=SubsetMethod.FIRST,
                    replicate_method=ReplicationMethod.FIRST,
                    mesh_collision_approximation=mesh_collision_approximation,
                ),
                num_envs=self.num_envs,
                device=self.device,
                terrain=None,
            )
            if self.validate_contacts and self.contact_object_mode == "kinematic":
                for scene in scene_lib.scenes:
                    for scene_object in scene.objects:
                        scene_object.options.fix_base_link = True
                print("Contact validation: driving all scene objects kinematically")
            scene_motion_ids = scene_lib.get_humanoid_motion_ids()
            if scene_motion_ids is not None and any(
                motion_id != self.current_motion_idx for motion_id in scene_motion_ids
            ):
                raise ValueError(
                    f"Scene {self.scene_index} maps to motion {scene_motion_ids}, "
                    f"not requested motion {self.current_motion_idx}"
                )
            print(
                f"Loaded physical scene {self.scene_index} with "
                f"{scene_lib.num_objects_per_scene} object slots"
            )
        self.scene_lib = scene_lib
        terrain = None

        # Get simulator class and instantiate
        SimulatorClass = get_class(self.simulator_cfg._target_)

        extra_params = extra_simulator_params or {}
        self.simulator = SimulatorClass(
            config=self.simulator_cfg,
            robot_config=self.robot_cfg,
            terrain=terrain,
            device=self.device,
            scene_lib=scene_lib,
            custom_key_handlers=custom_key_handlers,
            **extra_params,
        )
        # Initialize the simulator with visualization markers
        self.simulator._initialize_with_markers(self.viz_markers)

        print(f"Loaded {robot_name} robot using {simulator_type}")
        print(f"Visualizing bodies: {self.robot_spec.viz_bodies}")
        vel_source = "data_vel" if self.use_data_vel else "finite_diff"
        print(
            f"Smoothness metric: {self.metric} | velocity source: {vel_source} | window: {self.window_frames} frames"
        )
        print(f"Smoothness threshold: {self.smoothness_threshold}")
        print("Visualization:")
        print("  Red spheres - Specified body markers")
        print("  Yellow spheres - Bodies exceeding smoothness threshold")
        print("  Purple spheres - Bodies in contact with ground")
        print("Controls:")
        print("  'R' - Switch to next motion")
        print("  '1' - Increase playback speed by 150% (NumPad 1 for IsaacLab)")
        print("  '2' - Decrease playback speed by 150% (NumPad 2 for IsaacLab)")
        print("  '3' - Increase smoothness threshold by 1.5x (NumPad 3 for IsaacLab)")
        print("  '4' - Decrease smoothness threshold by 1.5x (NumPad 4 for IsaacLab)")
        print("Motion will play automatically and loop")
        if self.validate_contacts:
            print(
                "Contact validation: "
                f"label={self.contact_label_key}, force>{self.contact_force_threshold:g} N, "
                f"report={self.contact_report}"
            )

        self.simulator.user_requested_reset = False

        # Speed control state
        self.speed_change_factor = 1.5  # 150% speed change
        self.min_speed = 0.01  # Minimum playback speed
        self.max_speed = 10.0  # Maximum playback speed

        # Pre-computed smoothness metrics for current motion
        # Shape: [num_frames, num_envs, num_bodies] - stores smoothness score per body per frame
        self.precomputed_smoothness = None

        # Pre-compute smoothness for the initial motion
        print("Pre-computing smoothness metrics for initial motion...")
        self._precompute_motion_smoothness()

    def _create_visualization_markers(self) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for specified body locations"""
        # Create one marker config for each body we want to visualize
        marker_configs = [
            MarkerConfig(size="regular") for _ in self.robot_spec.viz_bodies
        ]

        # Yellow joint markers for ALL bodies (get count from kinematic_info)
        # Note: kinematic_info will be set after _create_simulator_config is called
        self.joint_marker_name = "joint_highlight_markers"
        # We'll create these markers in the simulator initialization

        # Purple contact markers for ALL bodies
        self.contact_marker_name = "contact_markers"
        # We'll create these markers in the simulator initialization

        # Create visualization marker groups (initially empty, will be populated after config loading)
        markers = {
            "body_markers": VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=marker_configs
            ),
        }

        return markers

    def _initialize_body_markers(self):
        """Initialize body markers after kinematic info is loaded"""
        if self.kinematic_info is None:
            return

        num_bodies = self.kinematic_info.num_bodies
        joint_marker_configs = [MarkerConfig(size="regular") for _ in range(num_bodies)]

        contact_marker_configs = [
            MarkerConfig(size="regular")  # Smaller size for contact markers
            for _ in range(num_bodies)
        ]

        # Add the body markers to the existing visualization markers
        self.viz_markers[self.joint_marker_name] = VisualizationMarkerConfig(
            type="sphere",
            color=(1.0, 1.0, 0.0),  # yellow
            markers=joint_marker_configs,
        )

        self.viz_markers[self.contact_marker_name] = VisualizationMarkerConfig(
            type="sphere",
            color=(0.8, 0.0, 0.8),  # purple
            markers=contact_marker_configs,
        )

    def _request_next_motion(self):
        """R key press: ask the main loop to advance to the next motion. Deferred via the flag (rather than
        switching here) so the heavy per-motion smoothness recompute runs on the loop, not in the key
        callback fired from inside the viewer's event poll."""
        if self.scene_file is not None:
            print("Scene-aware mode is fixed to one motion; restart with another --scene_index")
        else:
            self.simulator.user_requested_reset = True

    def _switch_to_next_motion(self):
        """Switch to the next motion in the dataset"""
        self.current_motion_idx = (self.current_motion_idx + 1) % self.total_motions
        self.current_frame = 0
        self.current_motion_length = (
            self.motion_libs[0]
            .get_motion_num_frames(None)[self.current_motion_idx]
            .item()
        )

        print(
            f"Switched to motion {self.current_motion_idx}/{self.total_motions-1} "
            f"(length: {self.current_motion_length} frames)"
        )
        print(
            f"Current motion: {self.motion_libs[0].motion_files[self.current_motion_idx]}"
        )

        # Pre-compute smoothness for new motion
        print("Pre-computing smoothness metrics for new motion...")
        self._precompute_motion_smoothness()

    def _precompute_motion_smoothness(self):
        """Pre-compute smoothness metrics for the entire current motion"""
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        dt = 1.0 / FPS

        # Load all frames for all environments
        all_positions = []
        all_velocities = []

        for frame_idx in range(self.current_motion_length):
            frame_tensor = torch.tensor([frame_idx], device=self.device)

            # Get state for all environments
            pos_list = []
            vel_list = []
            for motion_lib in self.motion_libs:
                state = motion_lib.get_motion_state_exact_frame(
                    motion_idx, frame_tensor
                )
                pos_list.append(state.rigid_body_pos[0])  # [num_bodies, 3]
                if state.rigid_body_vel is not None:
                    vel_list.append(state.rigid_body_vel[0])
                else:
                    vel_list.append(torch.zeros_like(state.rigid_body_pos[0]))

            # Stack: [num_envs, num_bodies, 3]
            all_positions.append(torch.stack(pos_list, dim=0))
            all_velocities.append(torch.stack(vel_list, dim=0))

        # Stack to [num_frames, num_envs, num_bodies, 3]
        positions_tensor = torch.stack(all_positions, dim=0)
        velocities_tensor = torch.stack(all_velocities, dim=0)

        T, E, B, _ = positions_tensor.shape

        # Compute smoothness using sliding window
        # Result shape: [num_frames, num_envs, num_bodies]
        smoothness_scores = torch.zeros(T, E, B, device=self.device)

        for frame_idx in range(T):
            # Get window around this frame
            window_start = max(0, frame_idx - self.window_frames // 2)
            window_end = min(T, frame_idx + self.window_frames // 2 + 1)

            if window_end - window_start < 4:  # Need at least 4 frames for jerk
                continue

            # Get windowed data
            pos_window = positions_tensor[window_start:window_end]  # [W, E, B, 3]
            vel_window = velocities_tensor[window_start:window_end]  # [W, E, B, 3]

            W = pos_window.shape[0]
            N = E * B

            # Reshape to [W, N, 3]
            pos_reshaped = pos_window.view(W, N, 3)
            vel_reshaped = vel_window.view(W, N, 3)

            # Use finite differences if configured
            if not self.use_data_vel:
                vel_reshaped = _diff(pos_reshaped, dt)
                # Pad velocity
                if vel_reshaped.shape[0] >= 2:
                    v_extrapolated = 2 * vel_reshaped[:1] - vel_reshaped[1:2]
                else:
                    v_extrapolated = torch.zeros_like(vel_reshaped[:1])
                vel_reshaped = torch.cat([v_extrapolated, vel_reshaped], dim=0)

            # Compute smoothness metric
            if self.metric == "nj":
                per_body_scores, _ = normalized_jerk_from_vel(vel_reshaped, dt)
            elif self.metric == "oi":
                per_body_scores, _ = oscillation_index_from_vel(vel_reshaped, dt)
            else:  # pj
                per_body_scores, _ = purposeful_jerk_from_vel(vel_reshaped, dt)

            # Reshape back to [E, B]
            per_body_scores = per_body_scores.view(E, B)
            smoothness_scores[frame_idx] = per_body_scores

        # Store pre-computed scores
        self.precomputed_smoothness = smoothness_scores
        print(f"Smoothness pre-computed for {T} frames")

    def _get_current_pose(self):
        """Get the current pose for the selected motion and frame using MotionLib API for all environments"""
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        # Get poses from all motion libraries
        dof_pos_list = []
        rigid_body_pos_list = []
        rigid_body_rot_list = []
        rigid_body_vel_list = []

        for motion_lib in self.motion_libs:
            state = motion_lib.get_motion_state_exact_frame(
                motion_idx, torch.tensor([clamped_frame], device=self.device)
            )
            dof_pos_list.append(state.dof_pos[0])
            rigid_body_pos_list.append(state.rigid_body_pos[0])
            rigid_body_rot_list.append(state.rigid_body_rot[0])
            # Handle case where rigid_body_vel might be None
            if state.rigid_body_vel is not None:
                rigid_body_vel_list.append(state.rigid_body_vel[0])
            else:
                rigid_body_vel_list.append(torch.zeros_like(state.rigid_body_pos[0]))

        # Stack to create batch dimension for environments
        dof_pos = torch.stack(dof_pos_list, dim=0)  # [num_envs, num_dofs]
        rigid_body_pos = torch.stack(
            rigid_body_pos_list, dim=0
        )  # [num_envs, num_bodies, 3]
        rigid_body_rot = torch.stack(
            rigid_body_rot_list, dim=0
        )  # [num_envs, num_bodies, 4]
        rigid_body_vel = torch.stack(
            rigid_body_vel_list, dim=0
        )  # [num_envs, num_bodies, 3]

        return dof_pos, rigid_body_pos, rigid_body_rot, rigid_body_vel

    def _update_contact_markers(self) -> Dict[str, MarkerState]:
        """Update contact markers to show which bodies are in contact with the ground."""
        # Get contact data for current frame from the first motion library
        motion_idx = torch.tensor(
            [self.current_motion_idx], device=self.device, dtype=torch.long
        )
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        # Get contact state from motion library
        contact_states = []
        for motion_lib in self.motion_libs:
            state = motion_lib.get_motion_state_exact_frame(
                motion_idx, torch.tensor([clamped_frame], device=self.device)
            )
            if state.rigid_body_contacts is not None:
                contact_states.append(state.rigid_body_contacts[0])  # [num_bodies]
            else:
                # Fallback if no contact data
                contact_states.append(
                    torch.zeros(
                        self.kinematic_info.num_bodies,
                        dtype=torch.bool,
                        device=self.device,
                    )
                )

        # Stack contact states for all environments
        contact_mask = torch.stack(contact_states, dim=0)  # [num_envs, num_bodies]

        # Get positions/orientations for ALL bodies
        all_body_state = self.simulator.get_bodies_state()
        all_translations = (
            all_body_state.rigid_body_pos.detach().clone()
        )  # [num_envs, all_bodies, 3]
        all_orientations = (
            all_body_state.rigid_body_rot.detach().clone()
        )  # [num_envs, all_bodies, 4]

        # Only show contact markers for bodies that are in contact
        # Hide non-contact markers below ground
        mask = contact_mask.unsqueeze(-1)  # [num_envs, all_bodies, 1]
        hidden_pos = torch.tensor([0.0, 0.0, -100.0], device=self.device).view(1, 1, 3)
        contact_translations = torch.where(mask, all_translations, hidden_pos)

        # # Offset contact markers slightly below the body center for visibility
        # contact_offset = torch.tensor([0.0, 0.0, -0.05], device=self.device).view(1, 1, 3)
        # contact_translations = torch.where(mask, contact_translations + contact_offset, hidden_pos)

        return {
            self.contact_marker_name: MarkerState(
                translation=contact_translations, orientation=all_orientations
            )
        }

    def _update_joint_highlights(self) -> Dict[str, MarkerState]:
        """Get which joints to highlight based on pre-computed smoothness metrics and return marker states."""

        # Look up pre-computed smoothness for current frame
        clamped_frame = min(self.current_frame, self.current_motion_length - 1)

        if (
            self.precomputed_smoothness is None
            or clamped_frame >= self.precomputed_smoothness.shape[0]
        ):
            # No pre-computed data available, no highlighting
            self.highlight_mask = torch.zeros(
                self.num_envs,
                self.kinematic_info.num_bodies,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            # Get pre-computed scores for this frame: [num_envs, num_bodies]
            per_body_scores = self.precomputed_smoothness[clamped_frame]

            # Determine which bodies exceed threshold
            highlight = (
                per_body_scores > self.smoothness_threshold
            )  # [num_envs, num_bodies]
            self.highlight_mask = highlight

        # Get positions/orientations for ALL bodies
        all_body_state = self.simulator.get_bodies_state()
        all_translations = (
            all_body_state.rigid_body_pos.detach().clone()
        )  # [num_envs, all_bodies, 3]
        all_orientations = (
            all_body_state.rigid_body_rot.detach().clone()
        )  # [num_envs, all_bodies, 4]

        # Only show for highlighted bodies by hiding non-highlighted markers below ground
        mask = self.highlight_mask.unsqueeze(-1)  # [num_envs, all_bodies, 1]
        hidden_pos = torch.tensor([0.0, 0.0, -100.0], device=self.device).view(1, 1, 3)
        translations = torch.where(mask, all_translations, hidden_pos)

        return {
            self.joint_marker_name: MarkerState(
                translation=translations, orientation=all_orientations
            )
        }

    def _set_robot_pose(self, dof_pos, rigid_body_pos=None, rigid_body_rot=None):
        """Set the robot to the specified pose"""
        # for visualize, so we don't need to set the velocities, so just put to zero so it does not move before we reset pose
        current_state = self.simulator.get_robot_state()

        # Set DOF positions (already has the correct shape [num_envs, num_dofs])
        current_state.dof_pos = dof_pos.detach()
        current_state.dof_vel = torch.zeros_like(current_state.dof_pos).detach()

        # set root position and orientation
        current_state.rigid_body_pos[:, 0, :] = rigid_body_pos.detach()[:, 0, :]
        current_state.rigid_body_rot[:, 0, :] = rigid_body_rot.detach()[:, 0, :]
        current_state.rigid_body_vel[:, 0, :] = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        current_state.rigid_body_ang_vel[:, 0, :] = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        # if rigid_body_pos is not None and rigid_body_rot is not None:
        #     current_state.rigid_body_pos = rigid_body_pos.detach()  # Already [num_envs, num_bodies, 3]
        #     current_state.rigid_body_rot = rigid_body_rot.detach()  # Already [num_envs, num_bodies, 4]
        #     current_state.rigid_body_vel = torch.zeros(self.num_envs, rigid_body_pos.shape[1], 3, device=self.device)
        #     current_state.rigid_body_ang_vel = torch.zeros(self.num_envs, rigid_body_pos.shape[1], 3, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        object_state = None
        if self.scene_file is not None:
            dt = float(self.motion_libs[0].motion_dt[self.current_motion_idx])
            motion_time = torch.full(
                (self.num_envs,), self.current_frame * dt, device=self.device
            )
            object_state = self.scene_lib.get_scene_pose(env_ids, motion_time)
            object_state.root_vel = torch.zeros_like(object_state.root_pos)
            object_state.root_ang_vel = torch.zeros_like(object_state.root_pos)
        self.simulator.reset_envs(
            current_state, new_object_states=object_state, env_ids=env_ids
        )
        return object_state

    def _get_updated_marker_positions(self):
        """Update marker positions to follow the specified bodies"""
        if not self.viz_markers:
            return

        # this will convert to sim common ordering, which is the MJCF ordering
        current_state = self.simulator.get_bodies_state()

        idx_in_common = [
            self.simulator._body_names.index(body_name)
            for body_name in self.robot_spec.viz_bodies
        ]

        all_positions = (
            current_state.rigid_body_pos[:, idx_in_common, :].detach().clone()
        )
        all_orientations = (
            current_state.rigid_body_rot[:, idx_in_common, :].detach().clone()
        )

        marker_states = {}

        marker_states["body_markers"] = MarkerState(
            translation=all_positions, orientation=all_orientations
        )

        # Add/update joint highlight markers
        joint_marker_states = self._update_joint_highlights()
        marker_states.update(joint_marker_states)

        # Add/update contact markers
        contact_marker_states = self._update_contact_markers()
        marker_states.update(contact_marker_states)

        return marker_states

    def increase_speed(self):
        """Increase playback speed by the speed change factor"""
        new_speed = min(self.playback_speed * self.speed_change_factor, self.max_speed)
        if new_speed != self.playback_speed:
            self.playback_speed = new_speed
            print(f"Playback speed increased to {self.playback_speed:.3f}x")
            return True
        return False

    def decrease_speed(self):
        """Decrease playback speed by the speed change factor"""
        new_speed = max(self.playback_speed / self.speed_change_factor, self.min_speed)
        if new_speed != self.playback_speed:
            self.playback_speed = new_speed
            print(f"Playback speed decreased to {self.playback_speed:.3f}x")
            return True
        return False

    def increase_smoothness_threshold(self):
        """Increase smoothness threshold by 1.5x"""
        self.smoothness_threshold *= 1.5
        print(f"Smoothness threshold increased to {self.smoothness_threshold:.3f}")

    def decrease_smoothness_threshold(self):
        """Decrease smoothness threshold by 1.5x"""
        new_threshold = max(
            self.smoothness_threshold / 1.5, 0.001
        )  # Minimum threshold of 0.001
        if new_threshold != self.smoothness_threshold:
            self.smoothness_threshold = new_threshold
            print(f"Smoothness threshold decreased to {self.smoothness_threshold:.3f}")
        else:
            print(f"Smoothness threshold at minimum: {self.smoothness_threshold:.3f}")

    def run(self):
        """Main simulation loop"""
        step_count = 0
        marker_states = None
        target_dt = 1.0 / FPS  # wall-clock time per motion frame

        while True:
            frame_start = time.perf_counter()

            # Check for reset request (the registered R key handler sets this flag)
            if self.simulator.user_requested_reset:
                self._switch_to_next_motion()
                self.simulator.user_requested_reset = False

            # Validation always visits every reference frame exactly once;
            # playback_speed only removes wall-clock throttling in this mode.
            if self.validate_contacts:
                frames_per_step = 1
                frame_skip = 1
            elif self.playback_speed < 1.0:
                frames_per_step = max(1, int(1.0 / self.playback_speed))
                frame_skip = 1  # Don't skip frames when slowing down
            else:
                frames_per_step = 1  # Update every step when speeding up
                frame_skip = max(
                    1, int(self.playback_speed)
                )  # Skip frames for fast playback

            # Update motion frame based on playback speed
            replay_frame = None
            pre_step_state = None
            pre_step_object_state = None
            reference_object_state = None
            if step_count % frames_per_step == 0:
                # Get current pose for display
                replay_frame = self.current_frame
                dof_pos, rigid_body_pos, rigid_body_rot, _ = self._get_current_pose()

                # Set robot pose
                reference_object_state = self._set_robot_pose(
                    dof_pos, rigid_body_pos, rigid_body_rot
                )
                if self.validate_contacts and self.replay_trace_path is not None:
                    pre_step_state = self.simulator.get_bodies_state()
                    if self.scene_file is not None:
                        pre_step_object_state = self.simulator.get_object_root_state()

                # Advance frame with skip for fast playback
                self.current_frame += frame_skip

                # Loop motion when finished
                if self.current_frame >= self.current_motion_length:
                    self.current_frame = 0

            # Zero torque control to maintain pose
            _common_actions = torch.zeros(
                self.num_envs, self.kinematic_info.num_dofs, device=self.device
            )

            if marker_states is None or step_count % frames_per_step == 0:
                marker_states = self._get_updated_marker_positions()

            self.simulator.step(_common_actions, markers_callback=lambda: marker_states)

            if self.validate_contacts and replay_frame is not None:
                contact_state = self.simulator.get_bodies_contact_buf()
                force = torch.linalg.norm(
                    contact_state.rigid_body_contact_forces[0], dim=-1
                )
                actual = force > self.contact_force_threshold
                expected = self.contact_reference[replay_frame]
                object_force = None
                actual_by_object = None
                expected_by_object = None
                if self.contact_reference_by_object is not None:
                    if self.simulator_type != "isaaclab":
                        raise RuntimeError(
                            "Per-object contact validation currently requires IsaacLab."
                        )
                    filtered = self.simulator.get_body_filtered_contact_forces()
                    # SceneLib object filters are contiguous and preserve scene
                    # slot order. Terrain, when present, occupies the prefix.
                    object_count = self.contact_reference_by_object.shape[-1]
                    object_offset = self.simulator.body_contact_object_filter_offset
                    if filtered.shape[2] < object_count + object_offset:
                        raise RuntimeError(
                            "Filtered contact matrix has too few partners: "
                            f"{filtered.shape[2]} for {object_count} labeled objects."
                        )
                    object_force = torch.linalg.norm(
                        filtered[
                            0,
                            :,
                            object_offset : object_offset + object_count,
                        ],
                        dim=-1,
                    )
                    actual_by_object = object_force > self.contact_force_threshold
                    expected_by_object = self.contact_reference_by_object[replay_frame]
                    self.contact_pair_tp += (actual_by_object & expected_by_object).cpu()
                    self.contact_pair_fp += (actual_by_object & ~expected_by_object).cpu()
                    self.contact_pair_fn += (~actual_by_object & expected_by_object).cpu()
                if self.replay_trace_path is not None:
                    actual_state = self.simulator.get_bodies_state()
                    self._trace_reference_pos.append(
                        rigid_body_pos[0].detach().cpu().numpy()
                    )
                    self._trace_actual_pos.append(
                        actual_state.rigid_body_pos[0].detach().cpu().numpy()
                    )
                    self._trace_pre_step_pos.append(
                        pre_step_state.rigid_body_pos[0].detach().cpu().numpy()
                    )
                    self._trace_reference_rot.append(
                        rigid_body_rot[0].detach().cpu().numpy()
                    )
                    self._trace_actual_rot.append(
                        actual_state.rigid_body_rot[0].detach().cpu().numpy()
                    )
                    self._trace_pre_step_rot.append(
                        pre_step_state.rigid_body_rot[0].detach().cpu().numpy()
                    )
                    if reference_object_state is not None:
                        post_object_state = self.simulator.get_object_root_state()
                        self._trace_reference_object_pos.append(
                            reference_object_state.root_pos[0].detach().cpu().numpy()
                        )
                        self._trace_pre_step_object_pos.append(
                            pre_step_object_state.root_pos[0].detach().cpu().numpy()
                        )
                        self._trace_post_step_object_pos.append(
                            post_object_state.root_pos[0].detach().cpu().numpy()
                        )
                        self._trace_reference_object_rot.append(
                            reference_object_state.root_rot[0].detach().cpu().numpy()
                        )
                        self._trace_pre_step_object_rot.append(
                            pre_step_object_state.root_rot[0].detach().cpu().numpy()
                        )
                        self._trace_post_step_object_rot.append(
                            post_object_state.root_rot[0].detach().cpu().numpy()
                        )
                    self._trace_force.append(force.detach().cpu().numpy())
                    if object_force is not None:
                        self._trace_object_force.append(
                            object_force.detach().cpu().numpy()
                        )
                    self._trace_expected.append(expected.detach().cpu().numpy())
                    if expected_by_object is not None:
                        self._trace_expected_by_object.append(
                            expected_by_object.detach().cpu().numpy()
                        )
                    self._trace_actual.append(actual.detach().cpu().numpy())
                    if actual_by_object is not None:
                        self._trace_actual_by_object.append(
                            actual_by_object.detach().cpu().numpy()
                        )
                self.contact_tp += (actual & expected).cpu()
                self.contact_fp += (actual & ~expected).cpu()
                self.contact_fn += (~actual & expected).cpu()
                self.validated_contact_frames += 1
                requested_frames = (
                    self.validation_frames
                    if self.validation_frames > 0
                    else self.current_motion_length
                )
                if self.validated_contact_frames >= requested_frames:
                    self._write_contact_validation_report()
                    return

            step_count += 1

            # Throttle to real-time (adjusted by playback speed)
            elapsed = time.perf_counter() - frame_start
            sleep_time = target_dt / max(self.playback_speed, 0.01) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _write_contact_validation_report(self) -> None:
        if self.replay_trace_path is not None and self._trace_reference_pos:
            self.replay_trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.replay_trace_path,
                frame_indices=np.arange(len(self._trace_reference_pos), dtype=np.int32),
                reference_body_pos=np.asarray(self._trace_reference_pos, dtype=np.float32),
                pre_step_body_pos=np.asarray(self._trace_pre_step_pos, dtype=np.float32),
                actual_body_pos=np.asarray(self._trace_actual_pos, dtype=np.float32),
                reference_body_rot=np.asarray(self._trace_reference_rot, dtype=np.float32),
                pre_step_body_rot=np.asarray(self._trace_pre_step_rot, dtype=np.float32),
                actual_body_rot=np.asarray(self._trace_actual_rot, dtype=np.float32),
                reference_object_pos=np.asarray(
                    self._trace_reference_object_pos, dtype=np.float32
                ),
                pre_step_object_pos=np.asarray(
                    self._trace_pre_step_object_pos, dtype=np.float32
                ),
                post_step_object_pos=np.asarray(
                    self._trace_post_step_object_pos, dtype=np.float32
                ),
                reference_object_rot=np.asarray(
                    self._trace_reference_object_rot, dtype=np.float32
                ),
                pre_step_object_rot=np.asarray(
                    self._trace_pre_step_object_rot, dtype=np.float32
                ),
                post_step_object_rot=np.asarray(
                    self._trace_post_step_object_rot, dtype=np.float32
                ),
                contact_force=np.asarray(self._trace_force, dtype=np.float32),
                object_contact_force=np.asarray(
                    self._trace_object_force, dtype=np.float32
                ),
                expected_contact=np.asarray(self._trace_expected, dtype=bool),
                expected_contact_by_object=np.asarray(
                    self._trace_expected_by_object, dtype=bool
                ),
                actual_contact=np.asarray(self._trace_actual, dtype=bool),
                actual_contact_by_object=np.asarray(
                    self._trace_actual_by_object, dtype=bool
                ),
                body_names=np.asarray(self.kinematic_info.body_names, dtype=str),
                object_names=np.asarray(
                    self.contact_object_names
                    if self.contact_object_names is not None
                    else [],
                    dtype=str,
                ),
                contact_force_threshold=np.asarray(
                    self.contact_force_threshold, dtype=np.float32
                ),
            )
        tp = self.contact_tp.numpy()
        fp = self.contact_fp.numpy()
        fn = self.contact_fn.numpy()
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp + fn, 1)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1.0e-12)
        active = (tp + fp + fn) > 0
        total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        micro_precision = total_tp / max(total_tp + total_fp, 1)
        micro_recall = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = (
            2.0 * micro_precision * micro_recall
            / max(micro_precision + micro_recall, 1.0e-12)
        )
        report = {
            "schema_version": 1,
            "motion_file": str(self.motion_files[0]),
            "scene_file": str(self.scene_file),
            "scene_index": self.scene_index,
            "contact_labels": str(self.contact_labels_path),
            "label_key": self.contact_label_key,
            "force_threshold_n": self.contact_force_threshold,
            "frames": self.validated_contact_frames,
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "macro_active_bodies": {
                "precision": float(precision[active].mean()) if active.any() else 0.0,
                "recall": float(recall[active].mean()) if active.any() else 0.0,
                "f1": float(f1[active].mean()) if active.any() else 0.0,
            },
            "per_body": {
                name: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "tp": int(tp[index]),
                    "fp": int(fp[index]),
                    "fn": int(fn[index]),
                }
                for index, name in enumerate(self.kinematic_info.body_names)
            },
        }
        if self.contact_pair_tp is not None:
            pair_tp = self.contact_pair_tp.numpy()
            pair_fp = self.contact_pair_fp.numpy()
            pair_fn = self.contact_pair_fn.numpy()
            pair_total_tp = int(pair_tp.sum())
            pair_total_fp = int(pair_fp.sum())
            pair_total_fn = int(pair_fn.sum())
            pair_precision = pair_total_tp / max(pair_total_tp + pair_total_fp, 1)
            pair_recall = pair_total_tp / max(pair_total_tp + pair_total_fn, 1)
            pair_f1 = (
                2.0
                * pair_precision
                * pair_recall
                / max(pair_precision + pair_recall, 1.0e-12)
            )
            report["object_mode"] = self.contact_object_mode
            report["pair_micro"] = {
                "precision": pair_precision,
                "recall": pair_recall,
                "f1": pair_f1,
                "tp": pair_total_tp,
                "fp": pair_total_fp,
                "fn": pair_total_fn,
            }
            report["per_object"] = {}
            for object_id, object_name in enumerate(self.contact_object_names):
                object_tp = int(pair_tp[:, object_id].sum())
                object_fp = int(pair_fp[:, object_id].sum())
                object_fn = int(pair_fn[:, object_id].sum())
                object_precision = object_tp / max(object_tp + object_fp, 1)
                object_recall = object_tp / max(object_tp + object_fn, 1)
                object_f1 = (
                    2.0
                    * object_precision
                    * object_recall
                    / max(object_precision + object_recall, 1.0e-12)
                )
                report["per_object"][str(object_name)] = {
                    "precision": object_precision,
                    "recall": object_recall,
                    "f1": object_f1,
                    "tp": object_tp,
                    "fp": object_fp,
                    "fn": object_fn,
                }
            print(
                "PhysX object-pair validation: "
                f"precision={pair_precision:.3f}, recall={pair_recall:.3f}, "
                f"F1={pair_f1:.3f}"
            )
        self.contact_report.parent.mkdir(parents=True, exist_ok=True)
        self.contact_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "PhysX contact validation: "
            f"precision={micro_precision:.3f}, recall={micro_recall:.3f}, "
            f"F1={micro_f1:.3f} ({self.validated_contact_frames} frames)"
        )
        print(f"Wrote {self.contact_report}")
        if self.replay_trace_path is not None:
            print(f"Wrote {self.replay_trace_path}")


def main():
    # Use the global args that were parsed early
    global args, AppLauncher

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Extra simulator parameters for IsaacLab
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {
            "headless": args.headless,
            "device": str(device),
            # # Performance settings for faster-than-realtime rendering
            # "rendering_mode": "performance",  # Options: "performance", "balanced", "quality"
        }
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    visualizer = MotionVisualizerSmoothness(
        motion_files=args.motion_files,
        robot_name=args.robot,
        simulator_type=args.simulator,
        headless=args.headless,
        cpu_only=args.cpu_only,
        extra_simulator_params=extra_simulator_params,
        playback_speed=args.playback_speed,
        metric=args.metric,
        use_data_vel=args.use_data_vel,
        window_sec=args.window_sec,
        scene_file=args.scene_file,
        scene_asset_root=args.scene_asset_root,
        scene_index=args.scene_index,
        start_motion_index=args.start_motion_index,
        mesh_collision_approximation=args.mesh_collision_approximation,
        contact_labels=args.contact_labels,
        validate_contacts=args.validate_contacts,
        contact_label_key=args.contact_label_key,
        contact_force_threshold=args.contact_force_threshold,
        contact_object_mode=args.contact_object_mode,
        validation_frames=args.validation_frames,
        contact_report=args.contact_report,
        replay_trace=args.replay_trace,
    )

    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        visualizer.simulator.close()


if __name__ == "__main__":
    main()
