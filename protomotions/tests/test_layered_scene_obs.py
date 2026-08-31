# SPDX-License-Identifier: Apache-2.0

import torch

from protomotions.envs.component_factories import (
    body_contact_feedback_obs_factory,
    local_scene_pointcloud_obs_factory,
    reference_contact_obs_factory,
    scene_object_token_obs_factory,
)
from protomotions.envs.obs.contact_feedback import compute_body_contact_feedback_obs
from protomotions.envs.obs.local_scene_pointcloud import compute_local_scene_pointcloud_obs
from protomotions.envs.obs.scene_object_state import compute_scene_object_token_obs


def test_layered_scene_observation_shapes_and_padding():
    envs, objects, points = 2, 3, 8
    root_pos = torch.zeros(envs, 3)
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(envs, 1)
    object_pos = torch.zeros(envs, objects, 3)
    object_pos[:, 1, 0] = 1.0
    object_pos[:, 2, 0] = 10.0
    object_rot = root_rot[:, None].repeat(1, objects, 1)
    pointcloud = torch.randn(envs, objects, points, 3) * 0.1
    normals = torch.nn.functional.normalize(
        torch.randn(envs, objects, points, 3), dim=-1
    )
    valid = torch.tensor([[True, True, False], [True, True, False]])
    static = torch.tensor([[True, False, True], [True, False, True]])

    local = compute_local_scene_pointcloud_obs(
        root_pos,
        root_rot,
        object_pos,
        object_rot,
        pointcloud,
        normals,
        valid,
        static,
        num_samples=10,
        crop_radius_m=3.0,
        rigid_body_pos=torch.stack(
            [root_pos, root_pos + torch.tensor([1.0, 0.0, 0.0])], dim=1
        ),
    )
    assert local.shape == (envs, 80)
    assert torch.isfinite(local).all()

    tokens = compute_scene_object_token_obs(
        root_pos,
        root_rot,
        object_pos,
        object_rot,
        torch.zeros_like(object_pos),
        torch.zeros_like(object_pos),
        valid,
        torch.ones(envs, objects, 3),
        static,
        torch.tensor([[3, 3, 0], [3, 3, 0]]),
    ).reshape(envs, objects, 26)
    assert torch.count_nonzero(tokens[:, 2]) == 0


def test_contact_feedback_and_factory_context_paths():
    obs = compute_body_contact_feedback_obs(
        torch.tensor([[True, False]]), torch.tensor([[20.0, 0.0]])
    )
    assert obs.shape == (1, 4)
    assert torch.isfinite(obs).all()

    point_factory = local_scene_pointcloud_obs_factory()
    token_factory = scene_object_token_obs_factory()
    feedback_factory = body_contact_feedback_obs_factory()
    reference_factory = reference_contact_obs_factory()
    assert point_factory.dynamic_vars["neutral_pointcloud_normals"].path == (
        "scene.neutral_pointcloud_normals"
    )
    assert point_factory.dynamic_vars["rigid_body_pos"].path == (
        "current.rigid_body_pos"
    )
    assert token_factory.dynamic_vars["object_bbox_extents"].path == (
        "scene.object_bbox_extents"
    )
    assert feedback_factory.dynamic_vars["current_contact_force_magnitudes"].path == (
        "current_contact_force_magnitudes"
    )
    assert reference_factory.dynamic_vars["rigid_body_contacts"].path == (
        "mimic.ref_state.rigid_body_contacts"
    )
