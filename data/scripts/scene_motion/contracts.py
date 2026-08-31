"""Typed input contracts shared by scene-motion dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np


class RotationFormat(str, Enum):
    AXIS_ANGLE = "axis_angle"
    EULER_XYZ = "euler_xyz"
    QUATERNION_XYZW = "quaternion_xyzw"
    MATRIX = "matrix"


@dataclass(frozen=True)
class ClipDescriptor:
    """Dataset-independent identity and alignment metadata for one clip."""

    clip_id: str
    num_frames: int
    fps: float
    scene_id: str
    source_start: int = 0
    source_end_exclusive: Optional[int] = None
    bad_local_frames: tuple[int, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass
class HumanMotionInput:
    """Required human input before conversion to a ProtoMotions robot.

    `root_translation` and `root_orientation` are world-space. `body_pose` is a
    flattened sequence of local joint rotations. All arrays must share T.
    """

    root_translation: np.ndarray
    root_orientation: np.ndarray
    body_pose: np.ndarray
    root_rotation_format: RotationFormat
    body_rotation_format: RotationFormat
    source_model: str
    root_translation_semantics: str = "skeleton_root_world"
    betas: Optional[np.ndarray] = None
    left_hand_pose: Optional[np.ndarray] = None
    right_hand_pose: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    joint_names: Optional[Sequence[str]] = None

    def validate(self, clip: ClipDescriptor) -> None:
        arrays = {
            "root_translation": self.root_translation,
            "root_orientation": self.root_orientation,
            "body_pose": self.body_pose,
        }
        for name, array in arrays.items():
            if array.shape[0] != clip.num_frames:
                raise ValueError(
                    f"{clip.clip_id}: {name} has {array.shape[0]} frames, "
                    f"expected {clip.num_frames}"
                )
        if self.root_translation.shape[1:] != (3,):
            raise ValueError(f"{clip.clip_id}: root_translation must have shape (T, 3)")
        if self.joint_positions is not None:
            if self.joint_positions.ndim != 3 or self.joint_positions.shape[2] != 3:
                raise ValueError(
                    f"{clip.clip_id}: joint_positions must have shape (T, J, 3)"
                )
            if self.joint_positions.shape[0] != clip.num_frames:
                raise ValueError(
                    f"{clip.clip_id}: joint_positions has "
                    f"{self.joint_positions.shape[0]} frames, expected {clip.num_frames}"
                )
            if self.joint_names is None:
                raise ValueError(
                    f"{clip.clip_id}: joint_names is required with joint_positions"
                )
            if len(self.joint_names) != self.joint_positions.shape[1]:
                raise ValueError(
                    f"{clip.clip_id}: {len(self.joint_names)} joint names for "
                    f"{self.joint_positions.shape[1]} joints"
                )


@dataclass
class ObjectMotionInput:
    """Pose sequence for one independently spawned scene object or object part."""

    object_id: str
    mesh_path: Path
    translation: np.ndarray
    rotation: np.ndarray
    rotation_format: RotationFormat
    pose_semantics: str = "mesh_local_to_world"
    movable: bool = True
    articulated_group: Optional[str] = None

    def validate(self, clip: ClipDescriptor) -> None:
        if self.translation.shape != (clip.num_frames, 3):
            raise ValueError(
                f"{clip.clip_id}:{self.object_id}: translation shape "
                f"{self.translation.shape}, expected {(clip.num_frames, 3)}"
            )
        if self.rotation.shape[0] != clip.num_frames:
            raise ValueError(
                f"{clip.clip_id}:{self.object_id}: rotation has "
                f"{self.rotation.shape[0]} frames, expected {clip.num_frames}"
            )
        if not self.mesh_path.is_file():
            raise FileNotFoundError(self.mesh_path)


@dataclass(frozen=True)
class SceneAssetInput:
    """Static scene geometry associated with a clip."""

    scene_id: str
    mesh_path: Path
    occupancy_path: Optional[Path] = None
    occupancy_axis_order: Optional[str] = None
    occupancy_bounds_min: Optional[tuple[float, float, float]] = None
    occupancy_bounds_max: Optional[tuple[float, float, float]] = None


class DatasetAdapter(ABC):
    """Minimal interface required to add another scene-motion dataset."""

    @abstractmethod
    def iter_clips(self) -> Iterator[ClipDescriptor]:
        """Yield stable, deterministic clip descriptors."""

    @abstractmethod
    def load_human(self, clip: ClipDescriptor) -> HumanMotionInput:
        """Load the frame-aligned human parameters for `clip`."""

    @abstractmethod
    def load_objects(self, clip: ClipDescriptor) -> list[ObjectMotionInput]:
        """Load zero or more frame-aligned object/part trajectories."""

    @abstractmethod
    def load_scene(self, clip: ClipDescriptor) -> SceneAssetInput:
        """Return static geometry and optional occupancy for `clip`."""
