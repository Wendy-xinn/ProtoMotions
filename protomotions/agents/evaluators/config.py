# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for evaluators."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    evaluation_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of MdpComponent evaluation metrics for success/failure tracking."}
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1}
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={"help": "Evaluate metrics every N epochs. None = disabled.", "min": 1}
    )
    policy_observation_intervention: str = "none"
    policy_observation_intervention_keys: List[str] = field(default_factory=list)
    deterministic_policy: bool = field(
        default=False,
        metadata={
            "help": (
                "Use greedy generation for policies exposing a "
                "deterministic_generation switch during evaluation."
            )
        },
    )
    eval_token_switch_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Greedy decoding bias added to each previous-frame token."
        },
    )
    score_component: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Evaluation component mean used for checkpoint selection. "
                "None preserves success-rate selection."
            )
        },
    )
    score_component_minimize: bool = field(
        default=True,
        metadata={"help": "Whether lower values of score_component are better."},
    )


@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={"help": "Discount factor for successful motion weights.", "min": 0.0, "max": 1.0}
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={"help": "Discount for failed motions. 0 = set weight straight to 1.", "min": 0.0, "max": 1.0}
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={"help": "Minimum weight for any motion. '1/num_motions' or float value."}
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    fixed_motion_eval_batch_size: Optional[int] = None
    deactivate_failed_motions: bool = field(
        default=True,
        metadata={
            "help": (
                "Stop and park a motion after its first evaluation-threshold "
                "failure. Disable for full-horizon diagnostics while retaining "
                "the first-failure success metric."
            )
        },
    )
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={"help": "Save pred_motion_lib every M evals. None = disabled.", "min": 1}
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."}
    )
    eval_action_ema_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "EMA smoothing factor for actions during evaluation only. "
                "Simulates deployment low-pass filtering. "
                "a_applied = alpha * a_policy + (1-alpha) * a_prev. "
                "None = disabled (raw actions). Typical values: 0.5-0.8."
                "Smaller alpha = more smoothing."
            ),
            "min": 0.0,
            "max": 1.0,
        }
    )
