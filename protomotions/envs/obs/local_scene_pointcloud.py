# SPDX-License-Identifier: Apache-2.0

"""Fixed-size ego-local scene surface observations."""

import math

import torch
from torch import Tensor

from protomotions.utils import rotations


def compute_local_scene_pointcloud_obs(
    root_pos: Tensor,
    root_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    neutral_pointcloud_normals: Tensor,
    object_valid_mask: Tensor,
    object_static_mask: Tensor,
    num_samples: int = 128,
    crop_radius_m: float = 3.0,
    rigid_body_pos: Tensor | None = None,
) -> Tensor:
    """Select body-stratified samples and express points/normals in heading frame.

    Each sample has eight channels: XYZ, normal, static flag, validity.  The
    fixed-size result remains compatible across clips with different primitive
    counts and is intentionally small enough for the stage-1 residual MLP.
    """
    num_envs, num_objects = object_pos.shape[:2]
    if num_objects == 0 or neutral_pointclouds.shape[2] == 0:
        return root_pos.new_zeros((num_envs, num_samples * 8))

    points_per_object = neutral_pointclouds.shape[2]
    expanded_rot = object_rot.unsqueeze(2).expand(-1, -1, points_per_object, -1)
    world_points = rotations.quat_rotate(
        expanded_rot.reshape(-1, 4), neutral_pointclouds.reshape(-1, 3), True
    ).reshape(num_envs, num_objects, points_per_object, 3)
    world_points = world_points + object_pos.unsqueeze(2)
    world_normals = rotations.quat_rotate(
        expanded_rot.reshape(-1, 4), neutral_pointcloud_normals.reshape(-1, 3), True
    ).reshape(num_envs, num_objects, points_per_object, 3)

    relative = world_points - root_pos[:, None, None]
    root_distances = torch.linalg.norm(relative, dim=-1)
    valid = object_valid_mask.unsqueeze(-1).expand_as(root_distances)
    valid = valid & (root_distances <= crop_radius_m)
    flat_valid = valid.reshape(num_envs, -1)
    available = flat_valid.shape[1]
    selected_count = min(num_samples, available)
    flat_world = world_points.reshape(num_envs, available, 3)
    if rigid_body_pos is None or rigid_body_pos.shape[1] == 0:
        flat_distance = root_distances.reshape(num_envs, -1).masked_fill(
            ~flat_valid, float("inf")
        )
        _, indices = torch.topk(
            flat_distance, k=selected_count, dim=-1, largest=False
        )
        selected_distance = flat_distance.gather(1, indices)
    else:
        # Give every body anchor an equal quota. This prevents a global nearest
        # query from spending almost all samples on the floor around the root.
        num_bodies = rigid_body_pos.shape[1]
        quota = min(max(1, math.ceil(selected_count / num_bodies)), available)
        body_distance = torch.linalg.norm(
            flat_world.unsqueeze(1) - rigid_body_pos.unsqueeze(2), dim=-1
        ).masked_fill(~flat_valid.unsqueeze(1), float("inf"))
        selected_distance, indices = torch.topk(
            body_distance, k=quota, dim=-1, largest=False
        )
        indices = indices.reshape(num_envs, -1)[:, :selected_count]
        selected_distance = selected_distance.reshape(num_envs, -1)[:, :selected_count]

    flat_relative = relative.reshape(num_envs, available, 3)
    flat_normals = world_normals.reshape(num_envs, available, 3)
    selected_relative = flat_relative.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
    selected_normals = flat_normals.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
    selected_valid = torch.isfinite(selected_distance)

    static = object_static_mask.unsqueeze(-1).expand(-1, -1, points_per_object)
    selected_static = static.reshape(num_envs, available).gather(1, indices)
    heading_inv = rotations.calc_heading_quat_inv(root_rot, True)
    heading = heading_inv.unsqueeze(1).expand(-1, selected_count, -1)
    local_points = rotations.quat_rotate(
        heading.reshape(-1, 4), selected_relative.reshape(-1, 3), True
    ).reshape(num_envs, selected_count, 3)
    local_normals = rotations.quat_rotate(
        heading.reshape(-1, 4), selected_normals.reshape(-1, 3), True
    ).reshape(num_envs, selected_count, 3)
    mask = selected_valid.to(root_pos.dtype).unsqueeze(-1)
    features = torch.cat(
        [
            local_points * mask,
            local_normals * mask,
            selected_static.to(root_pos.dtype).unsqueeze(-1) * mask,
            mask,
        ],
        dim=-1,
    )
    if selected_count < num_samples:
        features = torch.cat(
            [features, features.new_zeros(num_envs, num_samples - selected_count, 8)],
            dim=1,
        )
    return features.reshape(num_envs, num_samples * 8)
