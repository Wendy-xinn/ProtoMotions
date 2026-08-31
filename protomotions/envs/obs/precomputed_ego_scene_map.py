# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-aligned GT ego scene-map lookup."""

import torch
from torch import Tensor


def compute_precomputed_ego_scene_map_obs(
    motion_ids: Tensor,
    motion_times: Tensor,
    scene_maps: Tensor,
    scene_map_num_frames: Tensor,
    scene_map_fps: float,
) -> Tensor:
    """Gather the offline causal ego map for each active motion frame."""
    frame_ids = torch.floor(motion_times * scene_map_fps + 1.0e-5).long()
    frame_ids = torch.minimum(frame_ids, scene_map_num_frames[motion_ids] - 1)
    frame_ids = frame_ids.clamp_min(0)
    features = scene_maps[motion_ids, frame_ids]
    return features.to(dtype=motion_times.dtype)
