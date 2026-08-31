import torch
from tensordict import TensorDict

from protomotions.agents.scene_ppo.config import (
    SceneFeatureEncoderConfig,
    SceneResidualPPOActorConfig,
    TrajectorySceneCrossAttentionEncoderConfig,
)
from protomotions.agents.scene_ppo.model import (
    SceneFeatureEncoder,
    SceneResidualPPOActor,
    TrajectorySceneCrossAttentionEncoder,
)
from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig


def _inputs(batch_size=3):
    points = torch.randn(batch_size, 16, 8)
    points[:, -3:, -1] = 0
    objects = torch.randn(batch_size, 4, 26)
    objects[:, :, 15] = 1
    objects[:, -1, 15] = 0
    return points, objects, TensorDict(
        {
            "local_scene_pointcloud": points.reshape(batch_size, -1),
            "scene_object_tokens": objects.reshape(batch_size, -1),
            "nearest_surface": torch.randn(batch_size, 12),
        },
        batch_size=batch_size,
    )


def test_scene_feature_encoder_is_point_and_object_permutation_invariant():
    torch.manual_seed(7)
    config = SceneFeatureEncoderConfig(
        in_keys=["local_scene_pointcloud", "scene_object_tokens", "nearest_surface"],
        out_keys=["scene_delta"],
        num_out=9,
    )
    model = SceneFeatureEncoder(config).eval()
    points, objects, inputs = _inputs()
    expected = model(inputs.clone())["scene_delta"]
    permuted = inputs.clone()
    permuted["local_scene_pointcloud"] = points[:, torch.randperm(16)].reshape(3, -1)
    permuted["scene_object_tokens"] = objects[:, torch.randperm(4)].reshape(3, -1)
    actual = model(permuted)["scene_delta"]
    torch.testing.assert_close(actual, expected)


def test_scene_feature_encoder_ignores_invalid_padding_values():
    torch.manual_seed(11)
    config = SceneFeatureEncoderConfig(
        in_keys=["local_scene_pointcloud", "scene_object_tokens", "nearest_surface"],
        out_keys=["scene_delta"],
        num_out=5,
    )
    model = SceneFeatureEncoder(config).eval()
    points, objects, inputs = _inputs()
    expected = model(inputs.clone())["scene_delta"]
    points[:, -3:, :-1] = 1e6
    objects[:, -1, :] = -1e6
    objects[:, -1, 15] = 0
    changed_padding = inputs.clone()
    changed_padding["local_scene_pointcloud"] = points.reshape(3, -1)
    changed_padding["scene_object_tokens"] = objects.reshape(3, -1)
    actual = model(changed_padding)["scene_delta"]
    torch.testing.assert_close(actual, expected)


def test_scene_feature_encoder_no_scene_ablation_is_strictly_zero():
    torch.manual_seed(12)
    config = SceneFeatureEncoderConfig(
        in_keys=["local_scene_pointcloud", "scene_object_tokens", "nearest_surface"],
        out_keys=["scene_delta"],
        num_out=5,
        condition_mode="no_scene",
    )
    model = SceneFeatureEncoder(config)
    _, _, inputs = _inputs()
    output = model(inputs)["scene_delta"]
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_scene_feature_encoder_future_interaction_loss_updates_scene_branch():
    torch.manual_seed(17)
    batch = 3
    targets = 6
    config = SceneFeatureEncoderConfig(
        in_keys=[
            "local_scene_pointcloud",
            "nearest_surface",
            "future_scene_query",
            "future_scene_interactions",
        ],
        out_keys=["scene_delta"],
        num_out=5,
        num_interaction_targets=targets,
        distance_loss_weight=0.25,
        contact_loss_weight=0.5,
    )
    model = SceneFeatureEncoder(config)
    _, _, inputs = _inputs(batch)
    inputs.pop("scene_object_tokens")
    inputs["future_scene_query"] = torch.randn(batch, targets * 3)
    distance = torch.rand(batch, targets)
    contact = torch.randint(0, 2, (batch, targets)).float()
    inputs["future_scene_interactions"] = torch.stack(
        (distance, contact), dim=-1
    ).reshape(batch, -1)
    output = model(inputs)
    loss, metrics = model.compute_model_loss(
        output, current_epoch=0, zero_loss=output["scene_delta"].sum() * 0.0
    )
    loss.backward()
    assert loss.item() > 0
    assert "model/scene_interaction_loss" in metrics
    assert model.point_encoder[0].weight.grad is not None


def test_counterfactual_scene_loss_reaches_gated_action_adapter():
    torch.manual_seed(19)
    batch, targets, actions = 4, 3, 5
    scene_config = SceneFeatureEncoderConfig(
        in_keys=[
            "local_scene_pointcloud",
            "nearest_surface",
            "future_scene_query",
            "future_scene_interactions",
        ],
        out_keys=["scene_delta"],
        num_out=actions,
        num_interaction_targets=targets,
    )
    actor = SceneResidualPPOActor(
        SceneResidualPPOActorConfig(
            num_out=actions,
            actor_logstd=-2.9,
            in_keys=["body", *scene_config.in_keys],
            mu_key="base_action",
            mu_model=MLPWithConcatConfig(
                in_keys=["body"],
                out_keys=["base_action"],
                num_out=actions,
                layers=[MLPLayerConfig(units=16, activation="relu")],
            ),
            scene_model=scene_config,
            scene_gate_init=0.01,
            counterfactual_loss_weight=0.1,
            counterfactual_action_margin=0.03,
            counterfactual_min_interaction_delta=0.01,
            counterfactual_scene_keys=["local_scene_pointcloud"],
        )
    )
    points = torch.zeros(batch, 8, 8)
    points[..., 0] = torch.tensor([0.0, 0.3, 0.6, 0.9])[:, None]
    points[..., -1] = 1
    distance = torch.tensor([0.0, 0.3, 0.6, 0.9])[:, None].expand(-1, targets)
    contact = torch.tensor([0.0, 1.0, 0.0, 1.0])[:, None].expand(-1, targets)
    inputs = TensorDict(
        {
            "body": torch.randn(batch, 7),
            "local_scene_pointcloud": points.reshape(batch, -1),
            "nearest_surface": torch.randn(batch, 6),
            "future_scene_query": torch.zeros(batch, targets * 3),
            "future_scene_interactions": torch.stack(
                (distance, contact), dim=-1
            ).reshape(batch, -1),
        },
        batch_size=batch,
    )
    output = actor(inputs, log_internals=True)
    loss, metrics = actor.compute_model_loss(
        output,
        current_epoch=0,
        zero_loss=output["mean_action"].sum() * 0.0,
        log_prefix="actor_model",
    )
    loss.backward()
    assert loss.item() > 0
    assert metrics["actor_model/scene/counterfactual_active_fraction"].item() > 0
    assert actor.scene_gate.grad is not None
    assert actor.scene_gate.grad.abs().item() > 0
    assert actor.scene_model.scene_output.weight.grad is not None


def test_scene_feature_encoder_accepts_offline_head_trajectory_landmarks():
    torch.manual_seed(13)
    steps = 16
    config = SceneFeatureEncoderConfig(
        in_keys=[
            "local_scene_pointcloud",
            "scene_object_tokens",
            "head_target_poses",
            "head_target_masks",
            "head_target_times",
        ],
        out_keys=["task_cond"],
        num_out=7,
    )
    model = SceneFeatureEncoder(config).eval()
    _, _, inputs = _inputs()
    batch = inputs.batch_size[0]
    inputs["head_target_poses"] = torch.randn(batch, steps * 12)
    inputs["head_target_masks"] = torch.ones(batch, steps * 2)
    inputs["head_target_times"] = torch.linspace(0.0, 8.0, steps).repeat(batch, 1)
    output = model(inputs)["task_cond"]
    assert output.shape == (batch, 7)
    assert torch.isfinite(output).all()


def test_trajectory_scene_cross_attention_has_finite_gradients():
    torch.manual_seed(17)
    batch, point_count, steps = 3, 12, 8
    config = TrajectorySceneCrossAttentionEncoderConfig(
        in_keys=[
            "ego_visible_scene_pointcloud",
            "head_target_poses",
            "head_target_masks",
            "head_target_times",
            "body_contact_feedback",
        ],
        out_keys=["task_cond"],
        num_out=11,
        point_key="ego_visible_scene_pointcloud",
        model_dim=32,
        num_attention_heads=4,
        feedforward_dim=64,
        use_scene_history_token=True,
    )
    points = torch.randn(batch, point_count, 10)
    points[..., -1] = 1
    inputs = TensorDict(
        {
            "ego_visible_scene_pointcloud": points.reshape(batch, -1),
            "head_target_poses": torch.randn(batch, steps * 12),
            "head_target_masks": torch.ones(batch, steps * 2),
            "head_target_times": torch.linspace(0.1, 4.0, steps).repeat(batch, 1),
            "body_contact_feedback": torch.randn(batch, 6),
        },
        batch_size=batch,
    )
    model = TrajectorySceneCrossAttentionEncoder(config)
    loss = model(inputs)["task_cond"].square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_scene_history_features_summarize_current_and_remembered_points():
    points = torch.zeros(1, 4, 10)
    points[0, :2, :3] = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    points[0, :2, 6] = 1.0
    points[0, 0, 7] = 1.0
    points[0, 1, 8] = 0.5
    points[0, :2, 9] = 1.0
    valid = points[..., 9] > 0.5

    features = TrajectorySceneCrossAttentionEncoder._scene_history_features(
        points, valid
    )
    assert features.shape == (1, 16)
    torch.testing.assert_close(features[0, :6], torch.tensor([0.5, 0.5, 0.5, 1.0, 0.25, 0.5]))
    torch.testing.assert_close(features[0, 6:9], torch.tensor([2.0, 0.0, 0.0]))


def _trajectory_inputs(batch=2, point_count=12, steps=8):
    points = torch.randn(batch, point_count, 10)
    points[..., -1] = 1
    return TensorDict(
        {
            "ego_visible_scene_pointcloud": points.reshape(batch, -1),
            "head_target_poses": torch.randn(batch, steps * 12),
            "head_target_masks": torch.ones(batch, steps * 2),
            "head_target_times": torch.linspace(0.1, 4.0, steps).repeat(batch, 1),
            "body_contact_feedback": torch.randn(batch, 6),
        },
        batch_size=batch,
    )


def _ablation_model(mode):
    return TrajectorySceneCrossAttentionEncoder(
        TrajectorySceneCrossAttentionEncoderConfig(
            in_keys=[
                "ego_visible_scene_pointcloud",
                "head_target_poses",
                "head_target_masks",
                "head_target_times",
                "body_contact_feedback",
            ],
            out_keys=["task_cond"],
            num_out=11,
            point_key="ego_visible_scene_pointcloud",
            model_dim=32,
            num_attention_heads=4,
            feedforward_dim=64,
            condition_mode=mode,
        )
    ).eval()


def test_head_only_ablation_is_invariant_to_scene_geometry():
    torch.manual_seed(23)
    inputs = _trajectory_inputs()
    model = _ablation_model("head_only")
    expected = model(inputs.clone())["task_cond"]
    changed = inputs.clone()
    changed["ego_visible_scene_pointcloud"] = torch.randn_like(
        changed["ego_visible_scene_pointcloud"]
    )
    actual = model(changed)["task_cond"]
    torch.testing.assert_close(actual, expected)


def test_scene_only_ablation_is_invariant_to_head_trajectory():
    torch.manual_seed(29)
    inputs = _trajectory_inputs()
    model = _ablation_model("scene_only")
    expected = model(inputs.clone())["task_cond"]
    changed = inputs.clone()
    changed["head_target_poses"] = torch.randn_like(changed["head_target_poses"])
    changed["head_target_times"] = torch.flip(
        changed["head_target_times"], dims=[-1]
    )
    actual = model(changed)["task_cond"]
    torch.testing.assert_close(actual, expected)
