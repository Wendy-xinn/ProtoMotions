"""On-policy RLFT of the EgoBody scene-and-Head GPC adapter."""

from __future__ import annotations

import torch

from examples.experiments.gpc import sft_trumans_scene_head_overfit as sft
from protomotions.agents.scene_ppo.config import (
    TrajectorySceneCrossAttentionEncoderConfig,
)


PRIOR_CHECKPOINT = sft.PRIOR_CHECKPOINT
TRACKER_CHECKPOINT = sft.TRACKER_CHECKPOINT
SCENE_KEYS = sft.SCENE_KEYS

configure_robot_and_simulator = sft.configure_robot_and_simulator
terrain_config = sft.terrain_config
scene_lib_config = sft.scene_lib_config
motion_lib_config = sft.motion_lib_config
apply_inference_overrides = sft.apply_inference_overrides


def additional_experiment_arguments(parser):
    sft.additional_experiment_arguments(parser)
    parser.add_argument(
        "--rlft-action-smoothness-weight",
        type=float,
        default=-0.005,
        help="Penalty weight for frame-to-frame processed-action changes.",
    )
    parser.add_argument(
        "--rlft-action-acceleration-weight",
        type=float,
        default=-0.02,
        help="Penalty weight for processed-action second differences.",
    )
    parser.add_argument(
        "--rlft-tracking-threshold-bonus-weight",
        type=float,
        default=0.25,
        help="Soft bonus aligned with the evaluator's three success thresholds.",
    )
    parser.add_argument(
        "--rlft-tracking-threshold-violation-weight",
        type=float,
        default=-0.5,
        help="Hinge penalty for frames outside evaluator success thresholds.",
    )
    parser.add_argument(
        "--rlft-actor-lr",
        type=float,
        default=2e-6,
        help="Actor learning rate for the KL-anchored RLFT adapter.",
    )
    parser.add_argument("--rlft-base-prior-top-p", type=float, default=0.99)
    parser.add_argument("--rlft-sft-kl-coeff", type=float, default=0.01)
    parser.add_argument("--rlft-target-kl", type=float, default=0.01)
    parser.add_argument("--rlft-head-orientation-weight", type=float, default=0.12)
    parser.add_argument("--rlft-body-orientation-weight", type=float, default=0.15)
    parser.add_argument("--rlft-root-orientation-weight", type=float, default=0.10)
    parser.add_argument("--rlft-parameter-ema-decay", type=float, default=None)
    parser.add_argument(
        "--rlft-early-termination",
        action="store_true",
        help=(
            "Terminate drifted training rollouts early. Disabled by default so "
            "RLFT learns from its own off-reference states and recovery attempts."
        ),
    )


def env_config(robot_cfg, args):
    from protomotions.envs.component_factories import (
        anchor_pos_error_term_factory,
        action_acceleration_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        global_anchor_pos_rew_factory,
        global_body_ang_vel_rew_factory,
        global_body_lin_vel_rew_factory,
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
        relative_body_ori_rew_factory,
        relative_body_pos_error_term_factory,
        relative_body_pos_rew_factory,
        tracking_threshold_bonus_factory,
        tracking_threshold_violation_factory,
    )

    config = sft.env_config(robot_cfg, args)
    body_names = robot_cfg.kinematic_info.body_names

    def body_ids(names):
        return torch.tensor([body_names.index(name) for name in names], dtype=torch.long)

    end_effectors = body_ids(
        ["Head", "RightHand", "LeftHand", "RightFoot", "LeftFoot"]
    )
    head = body_ids(["Head"])
    body_orientation_ids = torch.tensor(
        [
            index
            for index, name in enumerate(body_names)
            if name not in {"Head", "Chest"}
        ],
        dtype=torch.long,
    )
    dof_names = robot_cfg.kinematic_info.dof_names
    acceleration_weights = torch.ones(len(dof_names), dtype=torch.float)
    for index, name in enumerate(dof_names):
        if any(part in name for part in ("Leg", "Shin", "Foot", "ToeBase")):
            acceleration_weights[index] = 0.35
        elif any(
            part in name
            for part in (
                "Spine",
                "Chest",
                "Neck",
                "Head",
                "Shoulder",
                "Arm",
                "ForeArm",
                "Hand",
            )
        ):
            acceleration_weights[index] = 1.5
    acceleration_weights /= acceleration_weights.mean()

    # Separate global root tracking from root-relative pose tracking. Dedicated
    # end-effector and Head/Chest terms prevent a few bad bodies from being
    # diluted by the 23-body average.
    config.reward_components = {
        "root_position": global_anchor_pos_rew_factory(weight=0.25, sigma=0.25),
        "root_orientation": global_anchor_ori_rew_factory(
            weight=args.rlft_root_orientation_weight, sigma=0.40
        ),
        "relative_pose": relative_body_pos_rew_factory(weight=0.30, sigma=0.30),
        "end_effector_pose": relative_body_pos_rew_factory(
            weight=0.20, sigma=0.18, body_indices=end_effectors
        ),
        "relative_orientation": relative_body_ori_rew_factory(
            weight=args.rlft_body_orientation_weight,
            sigma=0.50,
            body_indices=body_orientation_ids,
        ),
        "head_orientation": relative_body_ori_rew_factory(
            weight=args.rlft_head_orientation_weight,
            sigma=0.35,
            body_indices=head,
        ),
        "linear_velocity": global_body_lin_vel_rew_factory(weight=0.05, sigma=1.0),
        "angular_velocity": global_body_ang_vel_rew_factory(weight=0.025, sigma=3.14),
        "action_smoothness": action_smoothness_factory(
            weight=args.rlft_action_smoothness_weight
        ),
        "action_acceleration": action_acceleration_factory(
            weight=args.rlft_action_acceleration_weight,
            joint_weights=acceleration_weights,
        ),
        "tracking_threshold_bonus": tracking_threshold_bonus_factory(
            weight=args.rlft_tracking_threshold_bonus_weight,
            position_threshold=0.35,
            rotation_threshold=0.70,
            max_joint_threshold=0.75,
            temperature=0.05,
        ),
        "tracking_threshold_violation": tracking_threshold_violation_factory(
            weight=args.rlft_tracking_threshold_violation_weight,
            position_threshold=0.35,
            rotation_threshold=0.70,
            max_joint_threshold=0.75,
        ),
    }
    config.num_state_history_steps = max(config.num_state_history_steps, 2)
    config.termination_components = (
        {
            "root_drift": anchor_pos_error_term_factory(threshold=0.50),
            "relative_body_drift": relative_body_pos_error_term_factory(
                threshold=0.75
            ),
        }
        if args.rlft_early_termination
        else {}
    )
    # Feedback is an evaluation ablation, not part of the initial RLFT policy.
    config.head_orientation_feedback_gain = 0.0
    return config


def agent_config(robot_config, env_config, args):
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        PretrainedModelConfig,
    )
    from protomotions.agents.evaluators.config import MimicEvaluatorConfig
    from protomotions.agents.peft.prior_config import (
        DiscretePriorPEFTActorConfig,
        DiscretePriorPEFTConfig,
        DiscretePriorPEFTRLFTAgentConfig,
        DiscretePriorPEFTRLFTModelConfig,
    )
    from protomotions.envs.component_factories import (
        gr_error_factory,
        gt_error_factory,
        max_joint_error_factory,
    )

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
    critic_keys = ["max_coords_obs", *SCENE_KEYS]

    return DiscretePriorPEFTRLFTAgentConfig(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=args.prior_checkpoint,
                module_path="",
            )
        },
        e_clip=0.15,
        entropy_coef=0.0,
        target_kl=args.rlft_target_kl,
        parameter_ema_decay=args.rlft_parameter_ema_decay,
        evaluate_parameter_ema=True,
        tau=0.95,
        model=DiscretePriorPEFTRLFTModelConfig(
            actor=DiscretePriorPEFTActorConfig(
                in_keys=SCENE_KEYS,
                peft=DiscretePriorPEFTConfig(
                    model=condition_model,
                    peft_type="dora",
                    rank=16,
                    alpha=32,
                    temperature=1.0,
                    # The adapter-free GPC prior defines the support; the frozen
                    # SFT actor separately supplies the KL anchor.
                    top_p=1.0,
                    sampling_mode="prior_constraint",
                    prior_top_p=args.rlft_base_prior_top_p,
                    kl_coeff=args.rlft_sft_kl_coeff,
                    film_input_norm=True,
                ),
            ),
            critic=MLPWithConcatConfig(
                in_keys=critic_keys,
                out_keys=["value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(3)
                ],
            ),
            actor_optimizer=OptimizerConfig(
                _target_="torch.optim.AdamW", lr=args.rlft_actor_lr, weight_decay=0.0
            ),
            critic_optimizer=OptimizerConfig(
                _target_="torch.optim.AdamW", lr=1e-4, weight_decay=0.0
            ),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        num_steps=args.rollout_horizon,
        num_mini_epochs=args.num_mini_epochs,
        normalize_rewards=True,
        gradient_clip_val=1.0,
        save_last_checkpoint_every=args.save_last_checkpoint_every,
        evaluator=MimicEvaluatorConfig(
            eval_metrics_every=args.eval_metrics_every,
            max_eval_steps=191,
            deterministic_policy=True,
            fixed_motion_eval_batch_size=args.fixed_motion_eval_batch_size,
            eval_action_ema_alpha=args.eval_action_ema_alpha,
            score_component="gr_error",
            save_predicted_motion_lib_every=None,
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.35),
                "gr_error": gr_error_factory(threshold=0.70),
                "max_joint_error": max_joint_error_factory(threshold=0.75),
            },
        ),
    )
