"""Small-data SFT probe: SOMA GPC conditioned on TRUMANS scene + head only."""

import argparse
from collections import Counter
import json
import torch

from protomotions.agents.scene_ppo.config import (
    TrajectorySceneCrossAttentionEncoderConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

PRIOR_CHECKPOINT = "data/pretrained_models/gpc_prior/soma_bones/inference_last.ckpt"
TRACKER_CHECKPOINT = "data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt"
SCENE_KEYS = [
    "head_target_poses",
    "head_target_masks",
    "head_target_times",
    "ego_visible_scene_pointcloud",
    "body_contact_feedback",
]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--prior-checkpoint", default=PRIOR_CHECKPOINT)
    parser.add_argument("--tracker-checkpoint", default=TRACKER_CHECKPOINT)
    parser.add_argument("--scene-asset-root", default="/home/wenxin/projects/TRUMANS")
    parser.add_argument("--scene-pointcloud-candidates", type=int, default=256)
    parser.add_argument("--scene-pointcloud-input-samples", type=int, default=256)
    parser.add_argument("--scene-pointcloud-workers", type=int, default=None)
    parser.add_argument("--scene-pointcloud-seed", type=int, default=None)
    parser.add_argument("--minimum-scene-points", type=int, default=0)
    parser.add_argument(
        "--tracking-error-termination",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="End the active rollout window when tracking error exceeds 0.5 m.",
    )
    parser.add_argument(
        "--scene-history-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a learned summary token for the accumulated causal scene map.",
    )
    parser.add_argument("--ego-head-landmarks", type=int, default=32)
    parser.add_argument("--ego-camera-file", default=None)
    parser.add_argument("--ego-scene-map-file", default=None)
    parser.add_argument("--window-sampling-manifest", default=None)
    parser.add_argument("--window-size-frames", type=int, default=32)
    parser.add_argument("--random-windows-per-clip", type=int, default=1)
    parser.add_argument("--window-sampler-seed", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--episode-length", type=int, default=256)
    parser.add_argument("--num-mini-epochs", type=int, default=4)
    parser.add_argument("--save-last-checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-metrics-every", type=int, default=100)
    parser.add_argument("--offline-cache-output", default=None)
    parser.add_argument("--offline-cache-split", default="train")
    parser.add_argument("--offline-dataset-path", default=None)
    parser.add_argument("--offline-dataset-split", default="train")
    parser.add_argument("--offline-num-epochs", type=int, default=100)
    parser.add_argument(
        "--condition-mode",
        choices=("full", "head_only", "scene_only", "no_condition"),
        default="full",
        help="Controlled task-conditioning ablation for offline SFT.",
    )
    parser.add_argument(
        "--fsq-scalar-aux-weight",
        type=float,
        default=0.0,
        help="Weight of the 40-way unpacked FSQ scalar auxiliary objective.",
    )
    parser.add_argument(
        "--head-translation-only",
        action="store_true",
        help="At inference, condition on Head translation without Head rotation.",
    )
    parser.add_argument(
        "--head-orientation-feedback-gain",
        type=float,
        default=0.0,
        help=(
            "Overwrite the decoded Head PD target with calibrated reference "
            "orientation feedback. Use the same value for cache collection and "
            "student inference."
        ),
    )


def _tracker_future_steps(args):
    from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
    resolved = load_resolved_configs_from_checkpoint(args.tracker_checkpoint)
    return resolved["env"].control_components["mimic"].future_steps


def terrain_config(args):
    from protomotions.components.terrains.config import TerrainConfig
    return TerrainConfig(
        map_length=20.0, map_width=20.0, border_size=2.0,
        num_levels=1, num_terrains=1,
        terrain_proportions=[0, 0, 0, 0, 0, 0, 0, 1],
        minimal_humanoid_spacing=0.0,
    )


def scene_lib_config(args):
    from protomotions.components.scene_lib import ReplicationMethod, SceneLibConfig
    replication_weights = None
    if args.window_sampling_manifest is not None:
        with open(args.window_sampling_manifest, encoding="utf-8") as handle:
            motion_scene_ids = json.load(handle)["motion_scene_ids"]
        scene_counts = Counter(motion_scene_ids)
        # The generated scene pack preserves first-occurrence order while
        # deduplicating physical scenes.
        physical_scene_ids = list(dict.fromkeys(motion_scene_ids))
        replication_weights = [scene_counts[value] for value in physical_scene_ids]

    return SceneLibConfig(
        scene_file=args.scenes_file,
        asset_root=args.scene_asset_root,
        replicate_method=(
            ReplicationMethod.WEIGHTED
            if replication_weights is not None
            else ReplicationMethod.SEQUENTIAL
        ),
        replication_weights=replication_weights,
        pointcloud_samples_per_object=(
            None
            if args.ego_scene_map_file is not None
            else args.scene_pointcloud_candidates
        ),
        pointcloud_max_workers=args.scene_pointcloud_workers,
        pointcloud_sampling_seed=args.scene_pointcloud_seed,
        mesh_collision_approximation=None,
    )


def motion_lib_config(args):
    from protomotions.components.motion_lib import MotionLibConfig
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args) -> EnvConfig:
    from protomotions.envs.action import make_pd_action_config
    from protomotions.envs.component_factories import (
        body_contact_feedback_obs_factory,
        ego_visible_scene_pointcloud_obs_factory,
        local_scene_pointcloud_obs_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        precomputed_ego_scene_map_obs_factory,
        tracking_error_term_factory,
    )
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.masked_mimic_control import (
        FixedBodyCondition,
        MaskedMimicControlConfig,
    )
    from protomotions.envs.control.scene_object_reference_control import (
        SceneObjectReferenceControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.obs import (
        compute_target_masks_only,
        compute_target_poses_only,
        compute_target_time_offsets,
    )

    conditionable = torch.tensor(
        [robot_cfg.kinematic_info.body_names.index(name)
         for name in robot_cfg.trackable_bodies_subset], dtype=torch.long
    )
    body_ids = list(range(len(robot_cfg.kinematic_info.body_names)))
    if args.ego_scene_map_file is not None:
        scene_observation = precomputed_ego_scene_map_obs_factory(
            args.ego_scene_map_file
        )
    else:
        scene_observation = ego_visible_scene_pointcloud_obs_factory(
            head_body_id=robot_cfg.kinematic_info.body_names.index("Head"),
            num_samples=args.scene_pointcloud_input_samples,
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
            near_m=0.05,
            far_m=6.0,
            accumulate_history=True,
            include_history_metadata=True,
            history_age_scale_steps=float(args.episode_length),
            minimum_valid_points=args.minimum_scene_points,
            camera_trajectory_file=args.ego_camera_file,
            camera_fps=30.0,
        )

    observations = {
        "max_coords_obs": max_coords_obs_factory(),
        # Full target is used only by the frozen FSQ target encoder to label SFT.
        "mimic_target_poses": mimic_target_poses_max_coords_factory(
            with_velocities=True, with_relative=True
        ),
        "head_target_poses": MdpComponent(
            compute_func=compute_target_poses_only,
            dynamic_vars={
                "current_state_body_pos": EnvContext.current.rigid_body_pos,
                "current_state_body_rot": EnvContext.current.rigid_body_rot,
                "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
                "masked_mimic_ref_rot": EnvContext.masked_mimic.ref_rot,
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
            },
            static_params={"conditionable_body_ids": conditionable, "include_root_relative": True},
        ),
        "head_target_masks": MdpComponent(
            compute_func=compute_target_masks_only,
            dynamic_vars={"masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks},
            static_params={"conditionable_body_ids": conditionable},
        ),
        "head_target_times": MdpComponent(
            compute_func=compute_target_time_offsets,
            dynamic_vars={"masked_mimic_time_offsets": EnvContext.masked_mimic.time_offsets},
        ),
        "ego_visible_scene_pointcloud": scene_observation,
        "body_contact_feedback": body_contact_feedback_obs_factory(body_ids=body_ids),
    }
    if not 0.0 <= args.head_orientation_feedback_gain <= 1.0:
        raise ValueError("--head-orientation-feedback-gain must be in [0, 1]")
    if args.window_sampling_manifest is not None:
        from protomotions.envs.motion_manager.config import (
            SceneWindowMotionManagerConfig,
        )

        with open(args.window_sampling_manifest, encoding="utf-8") as handle:
            window_manifest = json.load(handle)
        motion_manager = SceneWindowMotionManagerConfig(
            init_start_prob=0.0,
            resample_on_reset=True,
            motion_scene_ids=list(window_manifest["motion_scene_ids"]),
            window_size_frames=args.window_size_frames,
            fixed_window_stride_frames=args.window_size_frames,
            random_windows_per_clip=args.random_windows_per_clip,
            sampler_seed=args.window_sampler_seed,
        )
        # BaseEnv resets at progress >= max_episode_length - 1.
        max_episode_length = args.window_size_frames + 1
    else:
        motion_manager = MimicMotionManagerConfig(
            init_start_prob=1.0,
            resample_on_reset=True,
        )
        max_episode_length = args.episode_length
    return EnvConfig(
        # The packaged retarget is already collision-box grounded.  Adding the
        # generic 5 cm spawn clearance would undo that calibration.
        ref_respawn_offset=0.0,
        ref_contact_smooth_window=7,
        max_episode_length=max_episode_length,
        reset_grace_period=2,
        control_components={
            "masked_mimic": MaskedMimicControlConfig(
                bootstrap_on_episode_end=True,
                future_steps=_tracker_future_steps(args),
                # Sparse landmarks span the remaining complete offline clip.
                num_masked_future_steps=args.ego_head_landmarks,
                uniform_future_times=True,
                fixed_conditioning=[FixedBodyCondition("Head", 1)],
                repeat_mask_probability=1.0,
                visible_target_pose_prob=1.0,
                fully_hidden_pose_prob=0.0,
            ),
            "scene_objects": SceneObjectReferenceControlConfig(),
        },
        observation_components=observations,
        termination_components=(
            {"tracking_error": tracking_error_term_factory(threshold=0.5)}
            if args.tracking_error_termination
            else {}
        ),
        action_config=make_pd_action_config(robot_cfg),
        head_orientation_feedback_gain=args.head_orientation_feedback_gain,
        motion_manager=motion_manager,
    )


def agent_config(robot_config, env_config, args):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import PretrainedModelConfig
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTActorConfig,
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTSFTAgentConfig,
        DiscretePriorPEFTSFTModelConfig,
    )
    # Ada-class and newer NVIDIA GPUs accelerate the frozen GPC prior's large
    # float32 matrix multiplications substantially with TF32. This keeps model
    # parameters, optimizer state, losses, and checkpoints in float32; only the
    # internal matmul implementation uses TensorFloat-32 precision.
    torch.set_float32_matmul_precision("high")

    condition_model = TrajectorySceneCrossAttentionEncoderConfig(
        in_keys=SCENE_KEYS,
        out_keys=["task_cond"],
        num_out=256,
        point_key="ego_visible_scene_pointcloud",
        point_feature_dim=10,
        condition_mode=args.condition_mode,
        use_scene_history_token=args.scene_history_token,
    )
    return DiscretePriorPEFTSFTAgentConfig(
        pretrained_modules={"prior": PretrainedModelConfig(
            checkpoint_path=args.prior_checkpoint, module_path=""
        )},
        model=DiscretePriorPEFTSFTModelConfig(
            actor=DiscretePriorPEFTActorConfig(
                in_keys=SCENE_KEYS,
                peft=DiscretePriorPEFTConfig(
                    model=condition_model,
                    peft_type="dora", rank=16, alpha=32,
                    temperature=1.0, top_p=0.9,
                    sampling_mode="prior_constraint", prior_top_p=0.99,
                    film_input_norm=True,
                ),
            ),
            actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", lr=3e-4),
            fsq_scalar_aux_weight=args.fsq_scalar_aux_weight,
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=args.rollout_horizon,
        num_mini_epochs=args.num_mini_epochs,
        gradient_clip_val=10.0,
        save_last_checkpoint_every=args.save_last_checkpoint_every,
        evaluator=MimicEvaluatorConfig(
            eval_metrics_every=args.eval_metrics_every,
            max_eval_steps=191,
            save_predicted_motion_lib_every=None,
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
        ),
        offline_cache_output=args.offline_cache_output,
        offline_cache_split=args.offline_cache_split,
        offline_dataset_path=args.offline_dataset_path,
        offline_dataset_split=args.offline_dataset_split,
        offline_num_epochs=args.offline_num_epochs,
    )


def configure_robot_and_simulator(robot_cfg, simulator_cfg: SimulatorConfig, args):
    robot_cfg.update_fields(contact_bodies=list(robot_cfg.kinematic_info.body_names))


def apply_inference_overrides(robot_cfg, simulator_cfg, env_cfg, agent_cfg,
                              terrain_cfg, motion_lib_cfg, scene_lib_cfg, args):
    env_cfg.termination_components = {}
    env_cfg.max_episode_length = 1_000_000
    if hasattr(env_cfg.motion_manager, "window_size_frames"):
        env_cfg.motion_manager.window_size_frames = 192
        env_cfg.motion_manager.fixed_window_stride_frames = 192
        env_cfg.motion_manager.random_windows_per_clip = 0
    env_cfg.record_reference_motion = True
    if args.head_translation_only:
        masked_mimic_cfg = env_cfg.control_components["masked_mimic"]
        for condition in masked_mimic_cfg.fixed_conditioning or []:
            if condition.body_name == "Head":
                condition.constraint_state = 0
