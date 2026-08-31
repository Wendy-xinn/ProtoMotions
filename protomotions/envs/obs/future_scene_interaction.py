# SPDX-License-Identifier: Apache-2.0

"""Privileged future body-to-scene queries and supervision targets."""

from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.obs.nearest_surface_obs import compute_nearest_surface_vectors
from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate


def compute_future_scene_query_obs(
    future_body_pos: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
    body_ids: Optional[list[int]] = None,
) -> Tensor:
    """Express privileged future body query positions in the current heading frame."""
    if body_ids is not None:
        future_body_pos = future_body_pos[:, :, body_ids]
    num_envs, num_steps, num_bodies = future_body_pos.shape[:3]
    relative = future_body_pos - root_pos[:, None, None]
    heading_inv = calc_heading_quat_inv(root_rot, True)
    heading_inv = heading_inv[:, None, None].expand(-1, num_steps, num_bodies, -1)
    local = quat_rotate(
        heading_inv.reshape(-1, 4), relative.reshape(-1, 3), True
    )
    return local.reshape(num_envs, -1)


def compute_future_scene_interaction_targets(
    future_body_pos: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
    height_points: Optional[Tensor] = None,
    height_samples: Optional[Tensor] = None,
    object_pos: Optional[Tensor] = None,
    object_rot: Optional[Tensor] = None,
    neutral_pointclouds: Optional[Tensor] = None,
    object_valid_mask: Optional[Tensor] = None,
    terrain_horizontal_scale: float = 0.1,
    body_ids: Optional[list[int]] = None,
    contact_threshold_m: float = 0.05,
    distance_scale_m: float = 0.5,
) -> Tensor:
    """Return normalized distance/contact labels for future body queries.

    Output is flattened from ``[env, future_step, body, 2]``. Channel zero is
    unsigned distance clipped and normalized by ``distance_scale_m``; channel
    one is a binary proximity-contact label. These are privileged loss targets,
    never policy inputs.
    """
    if body_ids is not None:
        future_body_pos = future_body_pos[:, :, body_ids]
    num_envs, num_steps, num_bodies = future_body_pos.shape[:3]
    query_pos = future_body_pos.reshape(num_envs, num_steps * num_bodies, 3)
    nearest_vectors = compute_nearest_surface_vectors(
        rigid_body_pos=query_pos,
        root_pos=root_pos,
        root_rot=root_rot,
        height_points=height_points,
        height_samples=height_samples,
        object_pos=object_pos,
        object_rot=object_rot,
        neutral_pointclouds=neutral_pointclouds,
        object_valid_mask=object_valid_mask,
        terrain_horizontal_scale=terrain_horizontal_scale,
    ).reshape(num_envs, num_steps, num_bodies, 3)
    distance_m = torch.linalg.vector_norm(nearest_vectors, dim=-1)
    normalized_distance = (distance_m / distance_scale_m).clamp(0.0, 1.0)
    contact = (distance_m <= contact_threshold_m).to(distance_m.dtype)
    return torch.stack((normalized_distance, contact), dim=-1).reshape(num_envs, -1)
