import torch
import pytest

from protomotions.envs.obs.ego_visible_scene_pointcloud import (
    compute_ego_visible_scene_pointcloud_obs,
)


def test_ego_camera_rejects_behind_and_occluded_points():
    body_pos = torch.zeros(1, 1, 3)
    body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    object_pos = torch.zeros(1, 1, 3)
    object_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    # Identity Head looks along -Y. The second front point is hidden in the
    # same angular pixel; the +Y point is behind the camera.
    points = torch.tensor(
        [[[[0.0, -1.0, 0.03], [0.0, -2.0, 0.03], [0.0, 1.0, 0.03]]]]
    )
    normals = torch.zeros_like(points)
    output = compute_ego_visible_scene_pointcloud_obs(
        reference_body_pos=body_pos,
        reference_body_rot=body_rot,
        object_pos=object_pos,
        object_rot=object_rot,
        neutral_pointclouds=points,
        neutral_pointcloud_normals=normals,
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=3,
        camera_offset_head=(0.0, 0.0, 0.0),
        image_width=8,
        image_height=8,
        occlusion_tolerance_m=0.01,
    ).reshape(1, 3, 8)
    valid = output[0, :, -1] > 0.5
    assert valid.sum().item() == 1
    torch.testing.assert_close(output[0, valid, 1], torch.tensor([-1.0]))


def test_ego_camera_keeps_points_in_distinct_pixels():
    body_pos = torch.zeros(1, 1, 3)
    body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    points = torch.tensor([[[[-0.5, -2.0, 0.0], [0.5, -2.0, 0.0]]]])
    output = compute_ego_visible_scene_pointcloud_obs(
        reference_body_pos=body_pos,
        reference_body_rot=body_rot,
        object_pos=torch.zeros(1, 1, 3),
        object_rot=body_rot,
        neutral_pointclouds=points,
        neutral_pointcloud_normals=torch.zeros_like(points),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=2,
        camera_offset_head=(0.0, 0.0, 0.0),
        image_width=16,
        image_height=8,
    ).reshape(1, 2, 8)
    assert (output[0, :, -1] > 0.5).sum().item() == 2


def test_measured_pv_camera_uses_negative_z_as_forward():
    # The simulator replicated this scene at x=10. The measured camera and its
    # stored source root are at the original x=0 and must follow that offset.
    body_pos = torch.tensor([[[10.0, 0.0, 0.0]]])
    body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    points = torch.tensor([[[[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]]])
    output = compute_ego_visible_scene_pointcloud_obs(
        reference_body_pos=body_pos,
        reference_body_rot=body_rot,
        object_pos=torch.tensor([[[10.0, 0.0, 0.0]]]),
        object_rot=body_rot,
        neutral_pointclouds=points,
        neutral_pointcloud_normals=torch.zeros_like(points),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=2,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
        camera_world_from=torch.eye(4).reshape(1, 1, 4, 4),
        camera_tan_h=torch.ones(1, 1),
        camera_tan_v=torch.ones(1, 1),
        camera_num_frames=torch.tensor([1]),
        camera_reference_root=torch.zeros(1, 1, 3),
    ).reshape(1, 2, 8)
    valid = output[0, :, -1] > 0.5
    assert valid.sum().item() == 1
    torch.testing.assert_close(output[0, valid, 1], torch.tensor([-2.0]))


def test_measured_camera_uses_asymmetric_principal_point_frustum():
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    point = torch.tensor([[[[0.75, 0.0, -1.0]]]])
    output = compute_ego_visible_scene_pointcloud_obs(
        reference_body_pos=torch.zeros(1, 1, 3),
        reference_body_rot=identity,
        object_pos=torch.zeros(1, 1, 3),
        object_rot=identity,
        neutral_pointclouds=point,
        neutral_pointcloud_normals=torch.zeros_like(point),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=1,
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
        camera_world_from=torch.eye(4).reshape(1, 1, 4, 4),
        camera_tan_h=torch.full((1, 1), 0.5),
        camera_tan_v=torch.full((1, 1), 0.5),
        camera_tan_left=torch.full((1, 1), 0.5),
        camera_tan_right=torch.full((1, 1), 1.0),
        camera_tan_top=torch.full((1, 1), 0.5),
        camera_tan_bottom=torch.full((1, 1), 0.5),
        camera_num_frames=torch.tensor([1]),
        camera_reference_root=torch.zeros(1, 1, 3),
    ).reshape(1, 1, 8)
    assert output[0, 0, -1] == 1.0


def test_static_point_remains_in_causal_memory_after_camera_turns_away():
    body_pos = torch.zeros(1, 1, 3)
    body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    points = torch.tensor([[[[0.0, -2.0, 0.0]]]])
    kwargs = dict(
        reference_body_pos=body_pos,
        object_pos=torch.zeros(1, 1, 3),
        object_rot=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
        neutral_pointclouds=points,
        neutral_pointcloud_normals=torch.zeros_like(points),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=1,
        camera_offset_head=(0.0, 0.0, 0.0),
        accumulate_history=True,
    )
    first = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=body_rot,
        progress_buf=torch.tensor([0]),
        **kwargs,
    ).reshape(1, 1, 8)
    assert first[0, 0, -1] > 0.5

    # 180 degrees around Z: the remembered point is now behind the camera but
    # must remain present, expressed in the new camera coordinate frame.
    turned = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    second = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=turned,
        progress_buf=torch.tensor([1]),
        **kwargs,
    ).reshape(1, 1, 8)
    assert second[0, 0, -1] > 0.5
    assert second[0, 0, 1] > 0.0


def test_history_metadata_tracks_visibility_age_and_episode_reset():
    body_pos = torch.zeros(1, 1, 3)
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    kwargs = dict(
        reference_body_pos=body_pos,
        object_pos=torch.zeros(1, 1, 3),
        object_rot=identity,
        neutral_pointclouds=torch.tensor([[[[0.0, -2.0, 0.0]]]]),
        neutral_pointcloud_normals=torch.zeros(1, 1, 1, 3),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=1,
        camera_offset_head=(0.0, 0.0, 0.0),
        accumulate_history=True,
        include_history_metadata=True,
        history_age_scale_steps=4.0,
    )
    current = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=identity, progress_buf=torch.tensor([0]), **kwargs
    ).reshape(1, 1, 10)
    assert current[0, 0, 7] == 1.0
    assert current[0, 0, 8] == 0.0

    turned = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    remembered = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=turned, progress_buf=torch.tensor([2]), **kwargs
    ).reshape(1, 1, 10)
    assert remembered[0, 0, 7] == 0.0
    assert remembered[0, 0, 8] == 0.5
    assert remembered[0, 0, 9] == 1.0

    reset = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=turned, progress_buf=torch.tensor([0]), **kwargs
    ).reshape(1, 1, 10)
    assert reset[0, 0, 9] == 0.0


def test_tracking_reset_preserves_history_until_motion_time_rewinds():
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    turned = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    kwargs = dict(
        reference_body_pos=torch.zeros(1, 1, 3),
        object_pos=torch.zeros(1, 1, 3),
        object_rot=identity,
        neutral_pointclouds=torch.tensor([[[[0.0, -2.0, 0.0]]]]),
        neutral_pointcloud_normals=torch.zeros(1, 1, 1, 3),
        object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        object_static_mask=torch.ones(1, 1, dtype=torch.bool),
        head_body_id=0,
        num_samples=1,
        camera_offset_head=(0.0, 0.0, 0.0),
        accumulate_history=True,
        include_history_metadata=True,
        history_age_scale_steps=4.0,
        motion_ids=torch.tensor([0]),
    )
    compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=identity,
        progress_buf=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
        **kwargs,
    )
    continued = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=turned,
        progress_buf=torch.tensor([0]),
        motion_times=torch.tensor([1.0 / 30.0]),
        **kwargs,
    ).reshape(1, 1, 10)
    assert continued[0, 0, 9] == 1.0
    assert continued[0, 0, 8] == pytest.approx(0.25)

    rewound = compute_ego_visible_scene_pointcloud_obs(
        reference_body_rot=turned,
        progress_buf=torch.tensor([0]),
        motion_times=torch.tensor([0.0]),
        **kwargs,
    ).reshape(1, 1, 10)
    assert rewound[0, 0, 9] == 0.0


def test_minimum_valid_points_reports_empty_frame():
    identity = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    with pytest.raises(RuntimeError, match="minimum_valid_points=1"):
        compute_ego_visible_scene_pointcloud_obs(
            reference_body_pos=torch.zeros(1, 1, 3),
            reference_body_rot=identity,
            object_pos=torch.zeros(1, 1, 3),
            object_rot=identity,
            neutral_pointclouds=torch.tensor([[[[0.0, 2.0, 0.0]]]]),
            neutral_pointcloud_normals=torch.zeros(1, 1, 1, 3),
            object_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            object_static_mask=torch.ones(1, 1, dtype=torch.bool),
            head_body_id=0,
            num_samples=1,
            camera_offset_head=(0.0, 0.0, 0.0),
            minimum_valid_points=1,
            progress_buf=torch.tensor([7]),
        )
