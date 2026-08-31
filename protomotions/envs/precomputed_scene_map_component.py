# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy file-backed MDP component for large precomputed ego maps."""

from __future__ import annotations

import torch

from protomotions.envs.context_paths import resolve_path
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.precomputed_ego_scene_map import (
    compute_precomputed_ego_scene_map_obs,
)


class PrecomputedEgoSceneMapComponent(MdpComponent):
    """Keep the large map outside serialized configs and load it on first use."""

    def __init__(self, scene_map_file: str):
        super().__init__(
            compute_func=compute_precomputed_ego_scene_map_obs,
            dynamic_vars={
                "motion_ids": EnvContext.motion_ids,
                "motion_times": EnvContext.motion_times,
            },
            static_params={},
            compile=False,
        )
        self.scene_map_file = scene_map_file
        self._loaded_params = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_loaded_params"] = None
        return state

    def resolve_args(self, ctx):
        resolved = {
            name: resolve_path(ctx, field_path.path)
            for name, field_path in self.dynamic_vars.items()
        }
        device = resolved["motion_times"].device
        if self._loaded_params is None:
            payload = torch.load(
                self.scene_map_file, map_location="cpu", weights_only=False
            )
            scene_maps = payload["features"]
            num_frames = payload["num_frames"].long()
            if scene_maps.ndim != 4:
                raise ValueError(
                    "Precomputed scene maps must have shape [M, T, P, C]"
                )
            if scene_maps.shape[0] != num_frames.numel():
                raise ValueError("Scene-map motion count does not match num_frames")
            feature_dim = int(payload.get("feature_dim", scene_maps.shape[-1]))
            if feature_dim != scene_maps.shape[-1]:
                raise ValueError("Scene-map feature_dim metadata is inconsistent")
            self._loaded_params = {
                "scene_maps": scene_maps.reshape(
                    scene_maps.shape[0], scene_maps.shape[1], -1
                ).to(device),
                "scene_map_num_frames": num_frames.to(device),
                "scene_map_fps": float(payload.get("fps", 30.0)),
            }
        return resolved, self._loaded_params
