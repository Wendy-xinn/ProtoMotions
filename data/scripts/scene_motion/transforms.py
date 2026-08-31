"""Coordinate and rotation conversion utilities for dataset adapters."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import RotationFormat


TRUMANS_Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def as_rotation_matrix(values: np.ndarray, rotation_format: RotationFormat) -> np.ndarray:
    values = np.asarray(values)
    flat = values.reshape(values.shape[0], -1)
    if rotation_format is RotationFormat.AXIS_ANGLE:
        return Rotation.from_rotvec(flat.reshape(-1, 3)).as_matrix()
    if rotation_format is RotationFormat.EULER_XYZ:
        return Rotation.from_euler("xyz", flat.reshape(-1, 3)).as_matrix()
    if rotation_format is RotationFormat.QUATERNION_XYZW:
        return Rotation.from_quat(flat.reshape(-1, 4)).as_matrix()
    if rotation_format is RotationFormat.MATRIX:
        return values.reshape(-1, 3, 3)
    raise ValueError(rotation_format)


def transform_points(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply a source-world to target-world basis rotation."""
    return np.einsum("ij,...j->...i", basis, np.asarray(points)).astype(np.float32)


def transform_world_rotations_left(
    values: np.ndarray,
    rotation_format: RotationFormat,
    basis: np.ndarray,
) -> np.ndarray:
    """Transform mesh-local-to-world rotations while leaving mesh local axes intact."""
    matrices = as_rotation_matrix(values, rotation_format)
    transformed = np.einsum("ij,tjk->tik", basis, matrices)
    return Rotation.from_matrix(transformed).as_quat().astype(np.float32)


def transform_root_axis_angle_left(axis_angle: np.ndarray, basis: np.ndarray) -> np.ndarray:
    quaternions = transform_world_rotations_left(
        axis_angle, RotationFormat.AXIS_ANGLE, basis
    )
    return Rotation.from_quat(quaternions).as_rotvec().astype(np.float32)


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Central finite-difference derivative with one-sided endpoints."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=1).astype(np.float32)
