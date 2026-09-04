# SPDX-License-Identifier: Apache-2.0

"""Offline ego-camera visibility observations from reconstructed scene surfaces."""

import math

import torch
from torch import Tensor

from protomotions.utils import rotations


# Observation kernels are plain callables rather than nn.Modules.  Keep the
# causal visibility state outside the TensorDict, keyed by the backing scene
# tensor so independent train/eval environments do not share memory.
_EGO_SCENE_MEMORY: dict[tuple, dict[str, Tensor]] = {}


def compute_ego_visible_scene_pointcloud_obs(
    reference_body_pos: Tensor,
    reference_body_rot: Tensor,
    object_pos: Tensor,
    object_rot: Tensor,
    neutral_pointclouds: Tensor,
    neutral_pointcloud_normals: Tensor,
    object_valid_mask: Tensor,
    object_static_mask: Tensor,
    head_body_id: int,
    num_samples: int = 512,
    map_point_budget: int = 8192,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 60.0,
    near_m: float = 0.05,
    far_m: float = 6.0,
    camera_offset_head: tuple[float, float, float] = (0.0, -0.08, 0.03),
    image_width: int = 64,
    image_height: int = 36,
    occlusion_tolerance_m: float = 0.04,
    accumulate_history: bool = False,
    include_history_metadata: bool = False,
    history_age_scale_steps: float = 256.0,
    minimum_valid_points: int = 0,
    progress_buf: Tensor | None = None,
    motion_ids: Tensor | None = None,
    motion_times: Tensor | None = None,
    camera_world_from: Tensor | None = None,
    camera_tan_h: Tensor | None = None,
    camera_tan_v: Tensor | None = None,
    camera_tan_left: Tensor | None = None,
    camera_tan_right: Tensor | None = None,
    camera_tan_top: Tensor | None = None,
    camera_tan_bottom: Tensor | None = None,
    camera_num_frames: Tensor | None = None,
    camera_reference_root: Tensor | None = None,
    camera_fps: float = 30.0,
) -> Tensor:
    """Return surfaces visible from a virtual pinhole camera between the eyes.

    SOMA23's semantic forward direction is local ``-Y``; local ``+X`` is image
    right and local ``+Z`` is image up.  The camera follows the *reference* Head
    rather than the simulated Head because the offline input video supplies the
    complete camera trajectory independently of rollout tracking error.

    The legacy 8 channels are camera-local XYZ, normal, static flag and valid.
    With history metadata enabled, current-visible and normalized time-since-
    last-seen are inserted before validity, yielding 10 channels.
    """
    num_envs, num_objects = object_pos.shape[:2]
    if num_objects == 0 or neutral_pointclouds.shape[2] == 0:
        if minimum_valid_points > 0:
            raise RuntimeError("Ego scene point cloud has no candidate geometry")
        feature_dim = 10 if include_history_metadata else 8
        return reference_body_pos.new_zeros((num_envs, num_samples * feature_dim))

    points_per_object = neutral_pointclouds.shape[2]
    if map_point_budget > 0 and points_per_object > map_point_budget:
        stride = (points_per_object + map_point_budget - 1) // map_point_budget
        keep = torch.arange(0, points_per_object, stride, device=neutral_pointclouds.device)[:map_point_budget]
        neutral_pointclouds = neutral_pointclouds.index_select(2, keep)
        neutral_pointcloud_normals = neutral_pointcloud_normals.index_select(2, keep)
        points_per_object = neutral_pointclouds.shape[2]
    expanded_object_rot = object_rot.unsqueeze(2).expand(
        -1, -1, points_per_object, -1
    )
    world_points = rotations.quat_rotate(
        expanded_object_rot.reshape(-1, 4),
        neutral_pointclouds.reshape(-1, 3),
        True,
    ).reshape(num_envs, num_objects, points_per_object, 3)
    world_points = world_points + object_pos.unsqueeze(2)
    world_normals = rotations.quat_rotate(
        expanded_object_rot.reshape(-1, 4),
        neutral_pointcloud_normals.reshape(-1, 3),
        True,
    ).reshape(num_envs, num_objects, points_per_object, 3)

    available = num_objects * points_per_object
    if camera_world_from is not None:
        if motion_ids is None or motion_times is None:
            raise ValueError("motion_ids and motion_times are required for measured cameras")
        frame_indices = torch.floor(motion_times * camera_fps + 1.0e-5).long()
        if camera_num_frames is not None:
            frame_indices = torch.minimum(
                frame_indices, camera_num_frames[motion_ids] - 1
            )
        frame_indices = frame_indices.clamp_min(0)
        world_from_camera = camera_world_from[motion_ids, frame_indices]
        camera_pos = world_from_camera[:, :3, 3]
        if camera_reference_root is not None:
            source_root = camera_reference_root[motion_ids, frame_indices]
            camera_pos = camera_pos + reference_body_pos[:, 0] - source_root
        camera_rot = world_from_camera[:, :3, :3]
        relative_world = (
            world_points.reshape(num_envs, available, 3) - camera_pos[:, None]
        )
        pv_points = torch.einsum("eij,epj->epi", camera_rot.transpose(1, 2), relative_world)
        pv_normals = torch.einsum(
            "eij,epj->epi",
            camera_rot.transpose(1, 2),
            world_normals.reshape(num_envs, available, 3),
        )
        # EgoBody PV is OpenGL-like: right=+X, up=+Y, forward=-Z.
        # Convert it to the existing canonical SOMA camera axes
        # right=+X, forward=-Y, up=+Z so encoder semantics stay unchanged.
        camera_points = torch.stack(
            (pv_points[..., 0], pv_points[..., 2], pv_points[..., 1]), dim=-1
        )
        camera_normals = torch.stack(
            (pv_normals[..., 0], pv_normals[..., 2], pv_normals[..., 1]), dim=-1
        )
        tan_h = camera_tan_h[motion_ids, frame_indices].unsqueeze(-1)
        tan_v = camera_tan_v[motion_ids, frame_indices].unsqueeze(-1)
        if camera_tan_left is None:
            tan_left = tan_right = tan_h
            tan_top = tan_bottom = tan_v
        else:
            assert camera_tan_right is not None
            assert camera_tan_top is not None
            assert camera_tan_bottom is not None
            tan_left = camera_tan_left[motion_ids, frame_indices].unsqueeze(-1)
            tan_right = camera_tan_right[motion_ids, frame_indices].unsqueeze(-1)
            tan_top = camera_tan_top[motion_ids, frame_indices].unsqueeze(-1)
            tan_bottom = camera_tan_bottom[motion_ids, frame_indices].unsqueeze(-1)
    else:
        head_pos = reference_body_pos[:, head_body_id]
        head_rot = reference_body_rot[:, head_body_id]
        offset = head_pos.new_tensor(camera_offset_head).expand(num_envs, -1)
        camera_pos = head_pos + rotations.quat_rotate(head_rot, offset, True)
        camera_rot_inv = rotations.quat_conjugate(head_rot, True)
        relative_world = (
            world_points.reshape(num_envs, available, 3) - camera_pos[:, None]
        )
        expanded_camera_inv = camera_rot_inv[:, None].expand(-1, available, -1)
        camera_points = rotations.quat_rotate(
            expanded_camera_inv.reshape(-1, 4), relative_world.reshape(-1, 3), True
        ).reshape(num_envs, available, 3)
        camera_normals = rotations.quat_rotate(
            expanded_camera_inv.reshape(-1, 4),
            world_normals.reshape(-1, 3),
            True,
        ).reshape(num_envs, available, 3)
        tan_h = math.tan(math.radians(horizontal_fov_deg) * 0.5)
        tan_v = math.tan(math.radians(vertical_fov_deg) * 0.5)
        tan_left = tan_right = tan_h
        tan_top = tan_bottom = tan_v

    # SOMA local axes: right=+X, forward=-Y, up=+Z.
    depth = -camera_points[..., 1]
    x_ratio = camera_points[..., 0] / depth.clamp_min(1.0e-6)
    z_ratio = camera_points[..., 2] / depth.clamp_min(1.0e-6)
    object_valid = object_valid_mask.unsqueeze(-1).expand(
        -1, -1, points_per_object
    ).reshape(num_envs, available)
    in_frustum = (
        object_valid
        & (depth >= near_m)
        & (depth <= far_m)
        & (x_ratio >= -tan_left)
        & (x_ratio <= tan_right)
        & (z_ratio >= -tan_bottom)
        & (z_ratio <= tan_top)
    )

    # Coarse angular z-buffer. scatter_reduce is deterministic and avoids
    # depending on a renderer, so this observation also works headlessly.
    pixel_x = ((x_ratio + tan_left) / (tan_left + tan_right) * image_width).long()
    pixel_y = ((z_ratio + tan_bottom) / (tan_bottom + tan_top) * image_height).long()
    pixel_x = pixel_x.clamp(0, image_width - 1)
    pixel_y = pixel_y.clamp(0, image_height - 1)
    pixel_index = pixel_y * image_width + pixel_x
    z_buffer = depth.new_full((num_envs, image_width * image_height), float("inf"))
    z_buffer.scatter_reduce_(
        1,
        pixel_index,
        depth.masked_fill(~in_frustum, float("inf")),
        reduce="amin",
        include_self=True,
    )
    nearest_depth = z_buffer.gather(1, pixel_index)
    visible = in_frustum & (depth <= nearest_depth + occlusion_tolerance_m)

    static = object_static_mask.unsqueeze(-1).expand(
        -1, -1, points_per_object
    ).reshape(num_envs, available)
    candidate = visible
    if accumulate_history:
        if progress_buf is None:
            raise ValueError("progress_buf is required when accumulate_history=True")
        if (motion_ids is None) != (motion_times is None):
            raise ValueError("motion_ids and motion_times must be provided together")
        use_motion_clock = motion_ids is not None
        memory_key = (
            neutral_pointclouds.device.type,
            neutral_pointclouds.device.index,
            neutral_pointclouds.data_ptr(),
            num_envs,
            available,
            use_motion_clock,
        )
        state = _EGO_SCENE_MEMORY.get(memory_key)
        if state is None or state["seen_static"].shape != visible.shape:
            state = {
                "seen_static": torch.zeros_like(visible),
                "last_seen_step": torch.full_like(visible, -1, dtype=torch.long),
                "previous_progress": torch.full_like(progress_buf, -1),
                "previous_motion_ids": (
                    torch.full_like(motion_ids, -1) if motion_ids is not None else None
                ),
                "previous_motion_times": (
                    torch.full_like(motion_times, -1.0)
                    if motion_times is not None
                    else None
                ),
            }
            _EGO_SCENE_MEMORY[memory_key] = state
        previous_progress = state["previous_progress"]
        if use_motion_clock:
            assert motion_ids is not None
            assert motion_times is not None
            assert state["previous_motion_ids"] is not None
            assert state["previous_motion_times"] is not None
            reset = (motion_ids != state["previous_motion_ids"]) | (
                motion_times + 1.0e-6 < state["previous_motion_times"]
            )
            history_step = torch.floor(motion_times * camera_fps + 1.0e-5).long()
            state["previous_motion_ids"].copy_(motion_ids)
            state["previous_motion_times"].copy_(motion_times)
        else:
            reset = (progress_buf == 0) | (progress_buf < previous_progress)
            history_step = progress_buf.to(torch.long)
        if reset.any():
            state["seen_static"][reset] = False
            state["last_seen_step"][reset] = -1
        static_points = static
        state["seen_static"] |= visible & static_points
        current_step = history_step.unsqueeze(-1).expand_as(visible)
        state["last_seen_step"] = torch.where(
            visible, current_step, state["last_seen_step"]
        )
        state["previous_progress"].copy_(progress_buf)
        # Dynamic objects are current-frame observations only: remembering
        # their old point identities would incorrectly leave geometry behind.
        candidate = visible | state["seen_static"]

    selected_count = min(num_samples, available)
    # Once memory is enabled, remembered points may be behind the camera and
    # therefore have negative signed depth. Rank the fixed-size memory by true
    # camera distance so negative depth cannot incorrectly win the top-k.
    candidate_distance = torch.linalg.vector_norm(camera_points, dim=-1).masked_fill(
        ~candidate, float("inf")
    )
    selected_distance, indices = torch.topk(
        candidate_distance, k=selected_count, dim=-1, largest=False
    )
    selected_valid = torch.isfinite(selected_distance)
    gather_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
    selected_points = camera_points.gather(1, gather_xyz)
    selected_normals = camera_normals.gather(1, gather_xyz)
    selected_static = static.gather(1, indices)
    selected_visible = visible.gather(1, indices)
    mask = selected_valid.to(camera_points.dtype).unsqueeze(-1)
    feature_parts = [
        selected_points * mask,
        selected_normals * mask,
        selected_static.to(camera_points.dtype).unsqueeze(-1) * mask,
    ]
    if include_history_metadata:
        if accumulate_history:
            selected_last_seen = state["last_seen_step"].gather(1, indices)
            current_step = history_step.unsqueeze(-1)
            selected_age = (current_step - selected_last_seen).clamp_min(0)
            selected_age = selected_age.to(camera_points.dtype)
            selected_age = (selected_age / history_age_scale_steps).clamp_max(1.0)
        else:
            selected_age = torch.zeros_like(selected_distance)
        feature_parts.extend([
            selected_visible.to(camera_points.dtype).unsqueeze(-1) * mask,
            selected_age.unsqueeze(-1) * mask,
        ])
    feature_parts.append(mask)
    features = torch.cat(feature_parts, dim=-1)
    feature_dim = 10 if include_history_metadata else 8
    if selected_count < num_samples:
        features = torch.cat(
            [features, features.new_zeros(num_envs, num_samples - selected_count, feature_dim)],
            dim=1,
        )
    if minimum_valid_points > 0:
        valid_counts = selected_valid.sum(dim=1)
        failed = valid_counts < minimum_valid_points
        if failed.any():
            failed_envs = failed.nonzero(as_tuple=False).flatten().tolist()
            frame_text = "unknown"
            if progress_buf is not None:
                frame_text = str(progress_buf[failed].tolist())
            raise RuntimeError(
                "Ego scene memory below minimum_valid_points="
                f"{minimum_valid_points}; envs={failed_envs}, frames={frame_text}, "
                f"counts={valid_counts[failed].tolist()}"
            )
    return features.reshape(num_envs, num_samples * feature_dim)
