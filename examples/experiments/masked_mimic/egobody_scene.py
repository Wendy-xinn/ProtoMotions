"""Scene-aware SMPL MaskedMimic student for EgoBody distillation."""

import argparse

import torch

from examples.experiments.masked_mimic import transformer as base
from protomotions.agents.common.config import (
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleOperationForwardConfig,
    ModuleOperationReshapeConfig,
)
from protomotions.agents.common.supervision import SupervisionLossConfig
from protomotions.agents.scene_ppo.config import (
    TrajectorySceneCrossAttentionEncoderConfig,
)
from protomotions.components.scene_lib import ReplicationMethod, SceneLibConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


INTERACTION_FUTURE_STEPS = [1, 5, 10, 20, 30]
INTERACTION_LATENT_DIM = 128


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    base.additional_experiment_arguments(parser)
    parser.add_argument("--scene-asset-root", required=True)
    parser.add_argument("--ego-camera-file", default=None)
    parser.add_argument("--scene-pointcloud-candidates", type=int, default=2048)
    parser.add_argument("--scene-pointcloud-seed", type=int, default=0)
    parser.add_argument("--scene-contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--interaction-latent-loss-weight", type=float, default=0.1)
    parser.add_argument("--interaction-distance-loss-weight", type=float, default=0.25)
    parser.add_argument("--interaction-contact-loss-weight", type=float, default=0.5)


def terrain_config(args: argparse.Namespace):
    return base.terrain_config(args)


def scene_lib_config(args: argparse.Namespace):
    return SceneLibConfig(
        scene_file=args.scenes_file,
        asset_root=args.scene_asset_root,
        replicate_method=ReplicationMethod.SEQUENTIAL,
        pointcloud_samples_per_object=args.scene_pointcloud_candidates,
        pointcloud_max_workers=1,
        pointcloud_sampling_seed=args.scene_pointcloud_seed,
        mesh_collision_approximation=None,
    )


def motion_lib_config(args: argparse.Namespace):
    return base.motion_lib_config(args)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.component_factories import (
        body_contact_feedback_obs_factory,
        ego_visible_scene_pointcloud_obs_factory,
        future_scene_interaction_target_obs_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.masked_mimic_control import FixedBodyCondition
    from protomotions.envs.control.scene_object_reference_control import (
        SceneObjectReferenceControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import (
        compute_target_masks_only,
        compute_target_poses_only,
        compute_target_time_offsets,
    )

    config = base.env_config(robot_cfg, args)
    control = config.control_components["masked_mimic"]
    control.future_steps = INTERACTION_FUTURE_STEPS
    control.fixed_conditioning = [FixedBodyCondition("Head", 1)]
    control.repeat_mask_probability = 1.0
    control.visible_target_pose_prob = 1.0
    config.control_components["scene_objects"] = SceneObjectReferenceControlConfig()

    body_ids = list(range(len(robot_cfg.kinematic_info.body_names)))
    head_id = robot_cfg.kinematic_info.body_names.index("Head")
    head_tensor = torch.tensor([head_id], dtype=torch.long)
    observations = config.observation_components
    observations.update(
        {
            "ego_visible_scene_pointcloud": ego_visible_scene_pointcloud_obs_factory(
                head_body_id=head_id,
                num_samples=256,
                horizontal_fov_deg=90.0,
                vertical_fov_deg=60.0,
                near_m=0.05,
                far_m=6.0,
                accumulate_history=True,
                include_history_metadata=True,
                history_age_scale_steps=float(config.max_episode_length),
                camera_trajectory_file=args.ego_camera_file,
                camera_fps=30.0,
            ),
            "body_contact_feedback": body_contact_feedback_obs_factory(
                body_ids=body_ids
            ),
            "head_target_poses": MdpComponent(
                compute_func=compute_target_poses_only,
                dynamic_vars={
                    "current_state_body_pos": EnvContext.current.rigid_body_pos,
                    "current_state_body_rot": EnvContext.current.rigid_body_rot,
                    "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
                    "masked_mimic_ref_rot": EnvContext.masked_mimic.ref_rot,
                    "masked_mimic_target_bodies_masks": (
                        EnvContext.masked_mimic.target_bodies_masks
                    ),
                },
                static_params={
                    "conditionable_body_ids": head_tensor,
                    "include_root_relative": True,
                },
            ),
            "head_target_masks": MdpComponent(
                compute_func=compute_target_masks_only,
                dynamic_vars={
                    "masked_mimic_target_bodies_masks": (
                        EnvContext.masked_mimic.target_bodies_masks
                    )
                },
                static_params={"conditionable_body_ids": head_tensor},
            ),
            "head_target_times": MdpComponent(
                compute_func=compute_target_time_offsets,
                dynamic_vars={
                    "masked_mimic_time_offsets": EnvContext.masked_mimic.time_offsets
                },
            ),
            "future_scene_interactions": (
                future_scene_interaction_target_obs_factory(
                    body_ids=body_ids,
                    contact_threshold_m=args.scene_contact_threshold_m,
                )
            ),
        }
    )
    config.ref_respawn_offset = 0.0
    config.ref_contact_smooth_window = 7
    return config


def agent_config(robot_config: RobotConfig, env_config: EnvConfig, args):
    config = base.agent_config(robot_config, env_config, args)
    prior = config.model.prior
    scene_keys = [
        "ego_visible_scene_pointcloud",
        "head_target_poses",
        "head_target_masks",
        "head_target_times",
        "body_contact_feedback",
    ]
    prior.in_keys = list(dict.fromkeys(prior.in_keys + scene_keys))

    scene_encoder = TrajectorySceneCrossAttentionEncoderConfig(
        in_keys=scene_keys,
        out_keys=["student_scene_interaction_latent"],
        num_out=INTERACTION_LATENT_DIM,
        point_key="ego_visible_scene_pointcloud",
        point_feature_dim=10,
        condition_mode="full",
    )
    num_targets = len(INTERACTION_FUTURE_STEPS) * len(
        robot_config.kinematic_info.body_names
    )
    interaction_models = [
        scene_encoder,
        MLPWithConcatConfig(
            in_keys=["student_scene_interaction_latent"],
            out_keys=["student_scene_distance_pred"],
            num_out=num_targets,
            layers=[MLPLayerConfig(units=256, activation="relu")],
        ),
        MLPWithConcatConfig(
            in_keys=["student_scene_interaction_latent"],
            out_keys=["student_scene_contact_logits"],
            num_out=num_targets,
            layers=[MLPLayerConfig(units=256, activation="relu")],
        ),
        MLPWithConcatConfig(
            in_keys=["student_scene_interaction_latent"],
            out_keys=["scene_interaction_token"],
            num_out=512,
            layers=[MLPLayerConfig(units=256, activation="relu")],
            module_operations=[
                ModuleOperationReshapeConfig(
                    new_shape=["batch_size", 1, -1]
                ),
                ModuleOperationForwardConfig(),
            ],
        ),
    ]
    transformer_index = next(
        index
        for index, model in enumerate(prior.models)
        if model._target_.endswith(".Transformer")
    )
    prior.models[transformer_index:transformer_index] = interaction_models
    transformer = prior.models[transformer_index + len(interaction_models)]
    transformer.in_keys = list(
        dict.fromkeys(transformer.in_keys + ["scene_interaction_token"])
    )

    config.model.interaction_target_key = "future_scene_interactions"
    config.model.interaction_num_targets = num_targets
    config.model.interaction_distance_loss_weight = (
        args.interaction_distance_loss_weight
    )
    config.model.interaction_contact_loss_weight = args.interaction_contact_loss_weight

    config.expert_output_map = {
        "scene_interaction_latent": "expert_scene_interaction_latent",
        "future_scene_distance_pred": "expert_scene_distance_pred",
        "future_scene_contact_logits": "expert_scene_contact_logits",
    }
    config.auxiliary_losses = [
        SupervisionLossConfig(
            prediction_key="student_scene_interaction_latent",
            target_key="expert_scene_interaction_latent",
            weight=args.interaction_latent_loss_weight,
            log_prefix="scene_distill/latent",
        ),
        SupervisionLossConfig(
            prediction_key="student_scene_distance_pred",
            target_key="expert_scene_distance_pred",
            weight=args.interaction_distance_loss_weight,
            log_prefix="scene_distill/distance",
        ),
        SupervisionLossConfig(
            prediction_key="student_scene_contact_logits",
            target_key="expert_scene_contact_logits",
            weight=args.interaction_contact_loss_weight,
            log_prefix="scene_distill/contact",
        ),
    ]
    return config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    args: argparse.Namespace,
):
    base.apply_inference_overrides(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, args
    )
    agent_cfg.expert_model_path = None
