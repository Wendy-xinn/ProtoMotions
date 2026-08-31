import torch
from types import SimpleNamespace

from protomotions.envs.obs.precomputed_ego_scene_map import (
    compute_precomputed_ego_scene_map_obs,
)
from protomotions.envs.precomputed_scene_map_component import (
    PrecomputedEgoSceneMapComponent,
)


def test_precomputed_scene_map_gathers_motion_and_frame():
    maps = torch.arange(2 * 4 * 3, dtype=torch.float16).reshape(2, 4, 3)
    output = compute_precomputed_ego_scene_map_obs(
        motion_ids=torch.tensor([1, 0]),
        motion_times=torch.tensor([1.0 / 30.0, 3.0 / 30.0]),
        scene_maps=maps,
        scene_map_num_frames=torch.tensor([4, 4]),
        scene_map_fps=30.0,
    )
    torch.testing.assert_close(output[0], maps[1, 1].float())
    torch.testing.assert_close(output[1], maps[0, 3].float())


def test_precomputed_scene_map_clamps_last_frame():
    maps = torch.arange(6, dtype=torch.float16).reshape(1, 2, 3)
    output = compute_precomputed_ego_scene_map_obs(
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([99.0]),
        scene_maps=maps,
        scene_map_num_frames=torch.tensor([2]),
        scene_map_fps=30.0,
    )
    torch.testing.assert_close(output[0], maps[0, 1].float())


def test_file_backed_component_drops_loaded_map_when_serialized(tmp_path):
    map_file = tmp_path / "maps.pt"
    torch.save(
        {
            "features": torch.arange(12, dtype=torch.float16).reshape(1, 2, 2, 3),
            "num_frames": torch.tensor([2]),
            "fps": 30.0,
            "feature_dim": 3,
        },
        map_file,
    )
    component = PrecomputedEgoSceneMapComponent(str(map_file))
    ctx = SimpleNamespace(
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([1.0 / 30.0]),
    )
    output = component.compute(ctx)
    assert output.shape == (1, 6)
    serialized = tmp_path / "component.pt"
    torch.save(component, serialized)
    restored = torch.load(serialized, weights_only=False)
    assert restored._loaded_params is None
    assert serialized.stat().st_size < 10_000
