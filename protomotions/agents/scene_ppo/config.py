"""Configuration for checkpoint-compatible scene-residual PPO."""

from dataclasses import dataclass, field
from typing import List

from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.ppo.config import PPOActorConfig, PPOAgentConfig


@dataclass
class SceneFeatureEncoderConfig:
    """Encode scene geometry, objects, and an offline head trajectory separately."""

    _target_: str = "protomotions.agents.scene_ppo.model.SceneFeatureEncoder"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    num_out: int = 1
    point_key: str = "local_scene_pointcloud"
    object_key: str = "scene_object_tokens"
    head_pose_key: str = "head_target_poses"
    head_mask_key: str = "head_target_masks"
    head_time_key: str = "head_target_times"
    point_feature_dim: int = 8
    object_feature_dim: int = 26
    point_embedding_dim: int = 128
    object_embedding_dim: int = 128
    head_embedding_dim: int = 128
    hidden_dim: int = 256
    condition_mode: str = "full"
    future_query_key: str = "future_scene_query"
    interaction_target_key: str = "future_scene_interactions"
    interaction_latent_key: str = "scene_interaction_latent"
    distance_prediction_key: str = "future_scene_distance_pred"
    contact_prediction_key: str = "future_scene_contact_logits"
    interaction_latent_dim: int = 128
    num_interaction_targets: int = 0
    distance_loss_weight: float = 0.0
    contact_loss_weight: float = 0.0
    contact_focal_gamma: float = 2.0
    contact_positive_weight: float = 1.0
    zero_init_output: bool = False


@dataclass
class TrajectorySceneCrossAttentionEncoderConfig(SceneFeatureEncoderConfig):
    """Causal scene memory queried by an ordered offline head trajectory."""

    _target_: str = (
        "protomotions.agents.scene_ppo.model."
        "TrajectorySceneCrossAttentionEncoder"
    )
    point_feature_dim: int = 10
    model_dim: int = 128
    num_attention_heads: int = 4
    num_head_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    condition_mode: str = "full"
    use_scene_history_token: bool = False


@dataclass
class SceneResidualPPOActorConfig(PPOActorConfig):
    _target_: str = "protomotions.agents.scene_ppo.model.SceneResidualPPOActor"
    scene_model: object = None
    scene_gate_init: float = 0.0
    scene_gate_learnable: bool = True
    counterfactual_loss_weight: float = 0.0
    counterfactual_action_margin: float = 0.03
    counterfactual_min_interaction_delta: float = 0.05
    counterfactual_contact_threshold_m: float = 0.05
    counterfactual_distance_scale_m: float = 0.5
    counterfactual_scene_keys: List[str] = field(default_factory=list)
    residual_preservation_loss_weight: float = 0.0
    interaction_num_bodies: int = 0
    interaction_body_ids: List[int] = field(default_factory=list)


@dataclass
class SceneResidualCriticConfig(MLPWithConcatConfig):
    _target_: str = "protomotions.agents.scene_ppo.model.SceneResidualCritic"
    scene_model: object = None
    scene_gate_init: float = 0.0


@dataclass
class SceneResidualPPOAgentConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.scene_ppo.agent.SceneResidualPPO"
    freeze_base_actor: bool = True
    scene_learning_rate_multiplier: float = 1.0
    reset_scene_on_warm_start: bool = False
