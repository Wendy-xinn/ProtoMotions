import torch

from data.scripts.align_ego_camera_to_grounded_motion import align_camera


def test_grounding_translates_camera_only_along_z():
    camera_pose = torch.eye(4).repeat(2, 1, 1)
    camera_pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [1.1, 2.2, 3.3]])
    camera = {
        "motions": [
            {
                "world_from_camera": camera_pose,
                "reference_root": torch.zeros(2, 3),
            }
        ]
    }
    grounded = {
        "retarget_root_height_offsets_m": torch.tensor([0.125]),
        "motion_num_frames": torch.tensor([2]),
        "length_starts": torch.tensor([0]),
        "gts": torch.zeros(2, 23, 3),
        "grs": torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 23, 1),
    }
    # SOMA -Y forward and EgoBody PV -Z forward must agree for validation.
    grounded["grs"][:, 6] = torch.tensor(
        [0.7071068, 0.0, 0.0, 0.7071068]
    )
    grounded["gts"][:, 6] = camera_pose[:, :3, 3] + torch.tensor([0.0, 0.0, 0.125])

    result = align_camera(camera, grounded)
    aligned = result["motions"][0]["world_from_camera"]

    torch.testing.assert_close(aligned[:, :2, 3], camera_pose[:, :2, 3])
    torch.testing.assert_close(aligned[:, 2, 3], camera_pose[:, 2, 3] + 0.125)
    torch.testing.assert_close(aligned[:, :3, :3], camera_pose[:, :3, :3])
    torch.testing.assert_close(
        result["motions"][0]["measured_world_from_camera"], camera_pose
    )
