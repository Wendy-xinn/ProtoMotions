# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from protomotions.envs.component_factories import scene_object_state_obs_factory
from protomotions.envs.obs.scene_object_state import compute_scene_object_state_obs


def test_scene_object_state_obs_contains_local_state_and_masks_padding():
    obs = compute_scene_object_state_obs(
        root_pos=torch.tensor([[1.0, 2.0, 0.0]]),
        root_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        object_pos=torch.tensor([[[2.0, 4.0, 3.0], [9.0, 9.0, 9.0]]]),
        object_rot=torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
        ),
        object_vel=torch.tensor([[[0.5, 1.0, 1.5], [8.0, 8.0, 8.0]]]),
        object_ang_vel=torch.tensor([[[2.0, 2.5, 3.0], [7.0, 7.0, 7.0]]]),
        object_valid_mask=torch.tensor([[True, False]]),
    ).reshape(1, 2, 16)

    assert torch.equal(obs[0, 0, :3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(obs[0, 0, 9:12], torch.tensor([0.5, 1.0, 1.5]))
    assert obs[0, 0, -1] == 1
    assert torch.count_nonzero(obs[0, 1]) == 0


def test_scene_object_state_obs_handles_empty_scenes_and_factory_paths():
    obs = compute_scene_object_state_obs(
        root_pos=torch.zeros(3, 3),
        root_rot=torch.zeros(3, 4),
        object_pos=torch.zeros(3, 0, 3),
        object_rot=torch.zeros(3, 0, 4),
        object_vel=torch.zeros(3, 0, 3),
        object_ang_vel=torch.zeros(3, 0, 3),
        object_valid_mask=torch.zeros(3, 0, dtype=torch.bool),
    )
    assert obs.shape == (3, 0)

    component = scene_object_state_obs_factory()
    assert component.dynamic_vars["object_vel"].path == "scene.object_vel"
    assert component.dynamic_vars["object_ang_vel"].path == "scene.object_ang_vel"
