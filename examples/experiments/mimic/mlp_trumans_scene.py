"""Stage-1 TRUMANS scene-aware expert, warm-started from smpl-terrains."""

import argparse

from protomotions.agents.ppo.config import (
    AdvantageNormalizationConfig,
    PPOModelConfig,
)
from protomotions.agents.base_agent.config import OptimizerConfig
from protomotions.agents.common.config import MLPLayerConfig, MLPWithConcatConfig
from protomotions.agents.evaluators.config import MimicEvaluatorConfig, MotionWeightsRulesConfig
from protomotions.agents.scene_ppo.config import (
    SceneFeatureEncoderConfig,
    SceneResidualCriticConfig,
    SceneResidualPPOActorConfig,
    SceneResidualPPOAgentConfig,
)
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.scene_lib import ReplicationMethod
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


SCENE_KEYS = [
    "nearest_surface",
    "local_scene_pointcloud",
    "scene_object_tokens",
    "body_contact_feedback",
]
CRITIC_SCENE_KEYS = SCENE_KEYS + ["reference_contacts"]
BASE_KEYS = ["max_coords_obs", "terrain", "mimic_target_poses", "previous_actions"]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--scene-asset-root",
        default="/home/wenxin/projects/TRUMANS",
        help="Root used to resolve relative mesh paths stored in the scene pack.",
    )
    parser.add_argument(
        "--scene-pointcloud-candidates",
        type=int,
        default=512,
        help=(
            "Surface candidates sampled per scene object before selecting the "
            "256 body-stratified policy points. The 10-env pilot can raise this; "
            "large-scale training needs the cached spatial-index path."
        ),
    )
    parser.add_argument(
        "--scene-pointcloud-workers",
        type=int,
        default=1,
        help=(
            "Maximum worker processes used while building cached scene pointclouds. "
            "Keep this low on WSL to avoid memory spikes."
        ),
    )
    parser.add_argument(
        "--scene-start-index",
        type=int,
        default=0,
        help="First SceneLib clip index loaded by the memory-safe pilot.",
    )
    parser.add_argument(
        "--scene-load-count",
        type=int,
        default=1,
        help=(
            "Number of original clip-scenes to deserialize into runtime tensors. "
            "Use a small value for WSL pilots; 0 explicitly requests the full split."
        ),
    )


def terrain_config(args: argparse.Namespace):
    # TRUMANS scenes are placed in Terrain's flat object-playground region and
    # provide their own collision floor/geometry. Generating the full complex
    # terrain here created ~1M vertices before PPO even started. Keep the
    # checkpoint-compatible 16x16 height observation, but use a tiny flat base.
    return TerrainConfig(
        # At least two 10 m scene slots are needed along x. An 8 m map put
        # the first scene center exactly one pixel outside tot_rows.
        map_length=20.0,
        # Terrain's current heightfield generator assumes square subterrain
        # tiles (length and width axes are assigned in transposed order).
        map_width=20.0,
        border_size=2.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        minimal_humanoid_spacing=0.0,
    )


def scene_lib_config(args: argparse.Namespace):
    if args.scene_start_index < 0:
        raise ValueError("--scene-start-index must be >= 0")
    if args.scene_load_count < 0:
        raise ValueError("--scene-load-count must be >= 0")
    scene_indices = None
    if args.scene_load_count > 0:
        scene_indices = list(
            range(
                args.scene_start_index,
                args.scene_start_index + args.scene_load_count,
            )
        )
    return SceneLibConfig(
        scene_file=args.scenes_file,
        asset_root=args.scene_asset_root,
        scene_indices=scene_indices,
        # Replicate K unique aligned pairs evenly across N parallel envs.
        replicate_method=ReplicationMethod.SEQUENTIAL,
        pointcloud_samples_per_object=args.scene_pointcloud_candidates,
        pointcloud_max_workers=args.scene_pointcloud_workers,
        # The scene pack already points at the prepared static mesh and
        # per-object convex USD assets. Do not rebake them per environment.
        mesh_collision_approximation=None,
    )


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        action_smoothness_factory,
        contact_force_change_rew_factory,
        contact_match_rew_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        mimic_tracking_rewards_factory,
        nearest_surface_obs_factory,
        local_scene_pointcloud_obs_factory,
        body_contact_feedback_obs_factory,
        reference_contact_obs_factory,
        pow_rew_factory,
        previous_actions_factory,
        scene_object_token_obs_factory,
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
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        "nearest_surface": nearest_surface_obs_factory(body_ids=body_ids),
        "local_scene_pointcloud": local_scene_pointcloud_obs_factory(
            num_samples=256, crop_radius_m=3.0
        ),
        "scene_object_tokens": scene_object_token_obs_factory(num_classes=6),
        "body_contact_feedback": body_contact_feedback_obs_factory(body_ids=body_ids),
        "reference_contacts": reference_contact_obs_factory(),
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
            # Conservative pilot weight: only validated static-proxy contacts
            # are injected. Increase after full-clip PhysX validation passes.
            weight=-0.02, zero_during_grace_period=True
        ),
        # Penalize obstacle impacts across all humanoid bodies. Scene geometry
        # still participates in PhysX contact resolution even without this term;
        # the explicit penalty makes collision avoidance visible to PPO.
        "scene_impact_rew": contact_force_change_rew_factory(
            weight=-1e-5,
            min_value=-0.5,
            threshold=30.0,
            zero_during_grace_period=True,
        ),
    }
    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
        control_components={
            "mimic": MimicControlConfig(bootstrap_on_episode_end=True),
            "scene_objects": SceneObjectReferenceControlConfig(),
        },
        observation_components=observations,
        termination_components={"tracking_error": tracking_error_term_factory(threshold=0.5)},
        reward_components=rewards,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(init_start_prob=0.2, resample_on_reset=True),
    )


def _scene_encoder(
    out_key: str, num_out: int, in_keys=SCENE_KEYS
) -> SceneFeatureEncoderConfig:
    return SceneFeatureEncoderConfig(
        in_keys=in_keys,
        out_keys=[out_key],
        num_out=num_out,
    )


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace):
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

    num_actions = robot_config.number_of_actions
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
        scene_model=_scene_encoder("scene_action_delta", num_actions),
        scene_gate_init=0.0,
    )
    critic = SceneResidualCriticConfig(
        in_keys=BASE_KEYS,
        out_keys=["value"],
        num_out=1,
        normalize_obs=True,
        norm_clamp_value=5,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
        scene_model=_scene_encoder("scene_value_delta", 1, CRITIC_SCENE_KEYS),
        scene_gate_init=0.0,
    )
    model_keys = BASE_KEYS + CRITIC_SCENE_KEYS
    return SceneResidualPPOAgentConfig(
        freeze_base_actor=True,
        model=PPOModelConfig(
            in_keys=model_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor,
            critic=critic,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
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
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    # Contact labels cover every SMPL body, including hands/torso interactions
    # with beds, walls and manipulated objects.
    robot_cfg.update_fields(contact_bodies=list(robot_cfg.kinematic_info.body_names))


def apply_inference_overrides(
    robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
    motion_lib_cfg, scene_lib_cfg, args,
):
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
