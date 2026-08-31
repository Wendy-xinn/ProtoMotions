import torch

from protomotions.envs.obs.future_scene_interaction import (
    compute_future_scene_interaction_targets,
    compute_future_scene_query_obs,
)


def test_future_scene_query_is_current_root_relative():
    future = torch.tensor([[[[2.0, 3.0, 1.0]]]])
    root_pos = torch.tensor([[1.0, 1.0, 0.0]])
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    query = compute_future_scene_query_obs(future, root_pos, root_rot)
    torch.testing.assert_close(query, torch.tensor([[1.0, 2.0, 1.0]]))


def test_future_scene_targets_encode_distance_and_contact():
    future = torch.tensor([[[[0.0, 0.0, 0.03], [0.0, 0.0, 0.20]]]])
    root_pos = torch.zeros(1, 3)
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    object_pos = torch.zeros(1, 1, 3)
    object_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    pointcloud = torch.zeros(1, 1, 1, 3)
    target = compute_future_scene_interaction_targets(
        future_body_pos=future,
        root_pos=root_pos,
        root_rot=root_rot,
        object_pos=object_pos,
        object_rot=object_rot,
        neutral_pointclouds=pointcloud,
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        contact_threshold_m=0.05,
        distance_scale_m=0.5,
    ).reshape(1, 1, 2, 2)
    torch.testing.assert_close(target[0, 0, :, 0], torch.tensor([0.06, 0.40]))
    torch.testing.assert_close(target[0, 0, :, 1], torch.tensor([1.0, 0.0]))
