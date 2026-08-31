# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact per-object state observations for scene-aware policies."""

import torch
from torch import Tensor
from torch.nn import functional as F

from protomotions.utils import rotations


def compute_scene_object_state_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    object_vel: Tensor,
    object_ang_vel: Tensor,
    object_valid_mask: Tensor,
) -> Tensor:
    """Return ego-heading-local pose, velocity and validity for every object.

    Each object contributes 16 values: position (3), rotation-6D (6), linear
    velocity (3), angular velocity (3), and validity (1). Invalid padding objects
    have all state features zeroed while retaining a zero validity channel.
    """
    num_envs, num_objects = object_pos.shape[:2]
    if num_objects == 0:
        return object_pos.new_zeros((num_envs, 0))

    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)
    heading_inv = heading_inv.unsqueeze(1).expand(-1, num_objects, -1)

    local_pos = rotations.quat_rotate(
        heading_inv.reshape(-1, 4),
        (object_pos - root_pos.unsqueeze(1)).reshape(-1, 3),
        True,
    ).reshape(num_envs, num_objects, 3)
    local_rot = rotations.quat_mul(heading_inv, object_rot, True)
    local_rot_6d = rotations.quat_to_tan_norm(local_rot, True)
    local_vel = rotations.quat_rotate(
        heading_inv.reshape(-1, 4), object_vel.reshape(-1, 3), True
    ).reshape(num_envs, num_objects, 3)
    local_ang_vel = rotations.quat_rotate(
        heading_inv.reshape(-1, 4), object_ang_vel.reshape(-1, 3), True
    ).reshape(num_envs, num_objects, 3)

    valid = object_valid_mask.to(dtype=object_pos.dtype).unsqueeze(-1)
    state = torch.cat(
        [local_pos, local_rot_6d, local_vel, local_ang_vel], dim=-1
    ) * valid
    return torch.cat([state, valid], dim=-1).reshape(num_envs, -1)


def compute_scene_object_token_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    object_vel: Tensor,
    object_ang_vel: Tensor,
    object_valid_mask: Tensor,
    object_bbox_extents: Tensor,
    object_static_mask: Tensor,
    object_class_ids: Tensor,
    num_classes: int = 6,
) -> Tensor:
    """One geometry-aware token per fixed scene-object slot.

    The first 16 channels retain the original state contract.  Extents, static
    status and a primitive/mesh type one-hot add the geometry needed to reason
    about collisions without re-inferring shape from pose.
    """
    num_envs, num_objects = object_pos.shape[:2]
    if num_objects == 0:
        return object_pos.new_zeros((num_envs, 0))
    state = compute_scene_object_state_obs(
        root_pos,
        root_rot,
        object_pos,
        object_rot,
        object_vel,
        object_ang_vel,
        object_valid_mask,
    ).reshape(num_envs, num_objects, 16)
    valid = object_valid_mask.to(object_pos.dtype).unsqueeze(-1)
    class_one_hot = F.one_hot(
        object_class_ids.clamp(min=0, max=num_classes - 1), num_classes=num_classes
    ).to(object_pos.dtype)
    geometry = torch.cat(
        [
            object_bbox_extents * valid,
            object_static_mask.to(object_pos.dtype).unsqueeze(-1) * valid,
            class_one_hot * valid,
        ],
        dim=-1,
    )
    return torch.cat([state, geometry], dim=-1).reshape(num_envs, -1)
