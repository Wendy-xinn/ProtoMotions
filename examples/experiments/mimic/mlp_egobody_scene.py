"""SMPL stage-1 teachers for the small EgoBody scene-motion split."""

import argparse

from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
from protomotions.agents.evaluators.config import (
    MimicEvaluatorConfig,
    MotionWeightsRulesConfig,
)
from protomotions.agents.ppo.config import AdvantageNormalizationConfig, PPOModelConfig
from protomotions.agents.scene_ppo.config import (
    SceneFeatureEncoderConfig,
    SceneResidualCriticConfig,
    SceneResidualPPOActorConfig,
    SceneResidualPPOAgentConfig,
)
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import ReplicationMethod, SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


SCENE_KEYS = [
    "nearest_surface",
    "local_scene_pointcloud",
    "body_contact_feedback",
    "future_scene_query",
    "future_scene_interactions",
]
CRITIC_SCENE_KEYS = SCENE_KEYS + ["reference_contacts"]
BASE_KEYS = ["max_coords_obs", "mimic_target_poses", "previous_actions"]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--scene-asset-root", required=True)
    parser.add_argument("--scene-pointcloud-candidates", type=int, default=2048)
    parser.add_argument("--scene-pointcloud-workers", type=int, default=1)
    parser.add_argument("--scene-pointcloud-seed", type=int, default=0)
    parser.add_argument("--save-last-checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-metrics-every", type=int, default=250)
    parser.add_argument("--fixed-motion-eval-batch-size", type=int, default=10)
    parser.add_argument("--scene-distance-loss-weight", type=float, default=0.25)
    parser.add_argument("--scene-contact-loss-weight", type=float, default=0.5)
    parser.add_argument("--scene-contact-positive-weight", type=float, default=10.0)
    parser.add_argument("--scene-contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--scene-counterfactual-loss-weight", type=float, default=0.5)
    parser.add_argument("--scene-counterfactual-action-margin", type=float, default=0.03)
    parser.add_argument("--scene-residual-preservation-weight", type=float, default=10.0)
    parser.add_argument(
        "--freeze-base-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--scene-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--reset-scene-on-warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--teacher-condition",
        choices=("body_only", "body_scene"),
        default="body_scene",
        help="Controlled full-body teacher ablation with or without full scene input.",
    )


def terrain_config(args: argparse.Namespace):
    del args
    return TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        border_size=2.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        minimal_humanoid_spacing=0.0,
    )


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(
        scene_file=args.scenes_file,
        asset_root=args.scene_asset_root,
        replicate_method=ReplicationMethod.SEQUENTIAL,
        pointcloud_samples_per_object=args.scene_pointcloud_candidates,
        pointcloud_max_workers=args.scene_pointcloud_workers,
        pointcloud_sampling_seed=args.scene_pointcloud_seed,
        mesh_collision_approximation=None,
    )


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        body_contact_feedback_obs_factory,
        contact_force_change_rew_factory,
        future_scene_interaction_target_obs_factory,
        future_scene_query_obs_factory,
        contact_match_rew_factory,
        local_scene_pointcloud_obs_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        nearest_surface_obs_factory,
        pow_rew_factory,
        previous_actions_factory,
        reference_contact_obs_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.control.scene_object_reference_control import (
        SceneObjectReferenceControlConfig,
    )
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig

    body_ids = list(range(len(robot_cfg.kinematic_info.body_names)))
    observations = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True, future_steps=1
        ),
        "nearest_surface": nearest_surface_obs_factory(body_ids=body_ids),
        # The mesh is pre-sampled densely; 256 body-stratified points keep the
        # privileged teacher input fixed-size.
        "local_scene_pointcloud": local_scene_pointcloud_obs_factory(
            num_samples=256, crop_radius_m=6.0
        ),
        "body_contact_feedback": body_contact_feedback_obs_factory(
            body_ids=body_ids
        ),
        "reference_contacts": reference_contact_obs_factory(),
        "future_scene_query": future_scene_query_obs_factory(body_ids=body_ids),
        "future_scene_interactions": (
            future_scene_interaction_target_obs_factory(
                body_ids=body_ids,
                contact_threshold_m=args.scene_contact_threshold_m,
            )
        ),
    }
    rewards = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.02, zero_during_grace_period=True
        ),
        "scene_impact_rew": contact_force_change_rew_factory(
            weight=-1e-5,
            min_value=-0.5,
            threshold=30.0,
            zero_during_grace_period=True,
        ),
    }
    return EnvConfig(
        ref_respawn_offset=0.0,
        ref_contact_smooth_window=7,
        max_episode_length=192,
        num_state_history_steps=2,
        control_components={
            "mimic": MimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=[1, 5, 10, 20, 30],
            ),
            "scene_objects": SceneObjectReferenceControlConfig(),
        },
        observation_components=observations,
        termination_components={
            "tracking_error": tracking_error_term_factory(threshold=0.5)
        },
        reward_components=rewards,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=1.0,
            resample_on_reset=False,
        ),
    )


def _scene_encoder(
    args,
    out_key: str,
    num_out: int,
    in_keys,
    num_bodies: int,
    auxiliary: bool = False,
    zero_init_output: bool = False,
):
    return SceneFeatureEncoderConfig(
        in_keys=list(in_keys),
        out_keys=[out_key],
        num_out=num_out,
        condition_mode=(
            "full" if args.teacher_condition == "body_scene" else "no_scene"
        ),
        num_interaction_targets=(5 * num_bodies if auxiliary else 0),
        distance_loss_weight=(
            args.scene_distance_loss_weight if auxiliary else 0.0
        ),
        contact_loss_weight=(
            args.scene_contact_loss_weight if auxiliary else 0.0
        ),
        contact_positive_weight=(
            args.scene_contact_positive_weight if auxiliary else 1.0
        ),
        zero_init_output=zero_init_output,
    )


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    del env_config
    num_actions = robot_config.number_of_actions
    body_names = robot_config.kinematic_info.body_names
    nonfoot_interaction_body_ids = [
        body_id
        for body_id, body_name in enumerate(body_names)
        if "Ankle" not in body_name and "Toe" not in body_name
    ]
    base_actor = MLPWithConcatConfig(
        in_keys=BASE_KEYS,
        out_keys=["actor_trunk_out"],
        num_out=num_actions,
        normalize_obs=True,
        norm_clamp_value=5,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
    )
    actor = SceneResidualPPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=BASE_KEYS + SCENE_KEYS,
        mu_key="actor_trunk_out",
        mu_model=base_actor,
        scene_model=_scene_encoder(
            args,
            "scene_action_delta",
            num_actions,
            BASE_KEYS + SCENE_KEYS,
            len(robot_config.kinematic_info.body_names),
            auxiliary=True,
            zero_init_output=True,
        ),
        # A fixed gain removes the gate/output scaling ambiguity and gives the
        # counterfactual loss a direct gradient path into the adapter.
        scene_gate_init=0.1,
        scene_gate_learnable=False,
        counterfactual_loss_weight=(
            args.scene_counterfactual_loss_weight
            if args.teacher_condition == "body_scene"
            else 0.0
        ),
        counterfactual_action_margin=args.scene_counterfactual_action_margin,
        counterfactual_contact_threshold_m=args.scene_contact_threshold_m,
        counterfactual_scene_keys=[
            "local_scene_pointcloud",
        ],
        residual_preservation_loss_weight=(
            args.scene_residual_preservation_weight
            if args.teacher_condition == "body_scene"
            else 0.0
        ),
        interaction_num_bodies=len(body_names),
        interaction_body_ids=nonfoot_interaction_body_ids,
    )
    critic = SceneResidualCriticConfig(
        in_keys=BASE_KEYS,
        out_keys=["value"],
        num_out=1,
        normalize_obs=True,
        norm_clamp_value=5,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
        scene_model=_scene_encoder(
            args,
            "scene_value_delta",
            1,
            CRITIC_SCENE_KEYS,
            len(robot_config.kinematic_info.body_names),
        ),
        scene_gate_init=0.0,
    )
    return SceneResidualPPOAgentConfig(
        # Preserve the pretrained full-body controller while learning how scene
        # context should adjust its actions through the gated residual branch.
        freeze_base_actor=args.freeze_base_actor,
        scene_learning_rate_multiplier=args.scene_learning_rate_multiplier,
        reset_scene_on_warm_start=args.reset_scene_on_warm_start,
        model=PPOModelConfig(
            in_keys=BASE_KEYS + CRITIC_SCENE_KEYS,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor,
            critic=critic,
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=args.actor_learning_rate
            ),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        save_last_checkpoint_every=args.save_last_checkpoint_every,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            eval_metrics_every=args.eval_metrics_every,
            fixed_motion_eval_batch_size=args.fixed_motion_eval_batch_size,
            policy_observation_intervention_keys=[
                "local_scene_pointcloud",
                "nearest_surface",
            ],
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args
):
    del args
    simulator_cfg.default_robot_friction = 0.5
    robot_cfg.update_fields(contact_bodies=list(robot_cfg.kinematic_info.body_names))


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    del robot_cfg, simulator_cfg, agent_cfg, terrain_cfg, motion_lib_cfg, scene_lib_cfg, args
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.init_start_prob = 1.0
    env_cfg.motion_manager.resample_on_reset = False
