# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration objects for discrete-token GPC prior PEFT.

SFT and RLFT get separate public configs instead of one config hidden behind a
``training_mode`` flag, so the config you instantiate determines which training
loop runs. These configs assume the prior emits discrete autoregressive tokens
(FSQ code indices), which is why the actor head is a token classifier and SFT
supervises with cross-entropy over token indices rather than continuous action
regression.

The public actor config mirrors the normal actor/critic style: ``actor.in_keys``
names the task observations, ``actor.peft.model`` builds the conditioning
tensor, and ``actor.peft.condition_key`` names the tensor consumed by adapter
layers. The frozen prior context is discovered from the loaded prior checkpoint.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from protomotions.agents.base_agent.config import (
    BaseModelConfig,
    OptimizerConfig,
)
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    ModuleContainerConfig,
    ModuleOperationForwardConfig,
    ObsProcessorConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.latent import (
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.supervised.config import SupervisedAgentConfig, RolloutActor
from protomotions.agents.fine_tuning.config import FineTuningAgentConfig
from protomotions.agents.peft.config import TransformerPEFTConfig


DEFAULT_PEFT_CONDITION_KEY = "task_cond"
DEFAULT_PEFT_NORM_CLAMP_VALUE = 5.0


@dataclass
class GaussianActionResidualConfig:
    """Small stochastic correction applied after the frozen FSQ decoder."""

    enabled: bool = False
    action_dim: int = 0
    hidden_dim: int = 256
    max_delta: float = 0.15
    initial_logstd: float = -3.5
    state_key: str = "max_coords_obs"
    previous_action_key: Optional[str] = None


def default_peft_model_config(
    in_keys: List[str],
    condition_key: str = DEFAULT_PEFT_CONDITION_KEY,
) -> ModuleContainerConfig:
    """Build the default task-observation-to-PEFT-condition module."""
    return ModuleContainerConfig(
        in_keys=list(in_keys),
        out_keys=[condition_key],
        models=[
            ObsProcessorConfig(
                in_keys=list(in_keys),
                out_keys=[condition_key],
                normalize_obs=True,
                norm_clamp_value=DEFAULT_PEFT_NORM_CLAMP_VALUE,
                module_operations=[ModuleOperationForwardConfig()],
            )
        ],
    )


@dataclass
class DiscretePriorPEFTConfig(TransformerPEFTConfig):
    """Discrete-prior PEFT adapter and conditioning-network configuration.

    ``model`` is the task-side TensorDict module that writes ``condition_key``.
    Adapter layers receive that condition tensor plus the frozen prior context
    discovered from the checkpoint; they do not read task or terrain-specific
    observation keys directly.
    """

    model: Optional[ModuleContainerConfig] = field(
        default=None,
        metadata={
            "help": (
                "TensorDict module that writes PEFT conditioning features. If "
                "omitted, a parameter-free ObsProcessor is built from actor in_keys."
            )
        },
    )
    condition_key: str = field(
        default=DEFAULT_PEFT_CONDITION_KEY,
        metadata={"help": "TensorDict key produced by model and consumed by PEFT."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature during generation."},
    )
    top_p: float = field(
        default=0.8,
        metadata={
            "help": (
                "Student nucleus sampling threshold. Ignored when "
                "sampling_mode='prior_constraint', where prior_top_p controls "
                "the frozen-prior nucleus."
            )
        },
    )
    sampling_mode: str = field(
        default="nucleus",
        metadata={"help": "Sampling strategy: 'nucleus' or 'prior_constraint'."},
    )
    prior_top_p: float = field(
        default=0.99,
        metadata={
            "help": "Nucleus threshold for the adapter-free base prior when "
            "sampling_mode='prior_constraint'."
        },
    )
    kl_coeff: float = field(
        default=0.0,
        metadata={
            "help": "KL divergence loss coefficient for training (0 = disabled)."
        },
    )
    clear_peft: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, zero adapter residuals once after the RL reference "
                "is pinned. This lets RLFT keep an SFT prior constraint while "
                "starting the active student from the base prior."
            )
        },
    )
    m_clamp: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set, clamp DoRA m parameters to [-m_clamp, m_clamp] after "
                "each actor optimizer step."
            )
        },
    )
    film_input_norm: bool = field(
        default=False,
        metadata={
            "help": (
                "Normalize the full PEFT conditioning vector before FiLM "
                "gamma/beta layers. Enable in both SFT and RLFT so the "
                "SFT-learned stats load into RLFT."
            )
        },
    )
    film_input_norm_clamp: float = field(
        default=5.0,
        metadata={"help": "Clamp value for normalized PEFT conditioning."},
    )

    def resolve_model(self, actor_in_keys: List[str]) -> None:
        if self.model is None:
            self.model = default_peft_model_config(
                actor_in_keys,
                condition_key=self.condition_key,
            )
        missing = [key for key in self.model.in_keys if key not in actor_in_keys]
        if missing:
            raise AssertionError(
                "DiscretePriorPEFTActorConfig in_keys must include PEFT model in_keys. "
                f"Missing: {missing}; in_keys={actor_in_keys}"
            )
        if self.condition_key not in self.model.out_keys:
            raise AssertionError(
                "DiscretePriorPEFTConfig model must produce condition_key "
                f"{self.condition_key!r}. out_keys={self.model.out_keys}"
            )


@dataclass
class DiscretePriorPEFTActorConfig:
    """Task-facing actor config for a PEFT-adapted discrete GPC prior.

    ``in_keys`` should contain only observations needed by ``peft.model``. The
    pretrained prior's context keys are resolved from the checkpoint at model
    construction time.
    """

    _target_: str = "protomotions.agents.peft.actor.DiscretePriorPEFTActor"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Task-specific PEFT observation keys. The frozen prior context "
                "keys are discovered from the loaded pretrained prior and "
                "appended at runtime."
            )
        },
    )
    out_keys: List[str] = field(
        default_factory=lambda: [
            "action",
            "mean_action",
            "neglogp",
            "prior_tokens",
        ],
        metadata={"help": "Actor rollout output keys."},
    )
    peft: DiscretePriorPEFTConfig = field(default_factory=DiscretePriorPEFTConfig)
    action_residual: GaussianActionResidualConfig = field(
        default_factory=GaussianActionResidualConfig
    )
    oracle_target_tokens: bool = False

    def __post_init__(self):
        self.peft.resolve_model(self.in_keys)
        if self.action_residual.enabled:
            if self.action_residual.action_dim < 1:
                raise ValueError("Enabled action residual requires action_dim > 0")
            if self.action_residual.hidden_dim < 1:
                raise ValueError("Action residual hidden_dim must be positive")
            if not 0.0 < self.action_residual.max_delta <= 1.0:
                raise ValueError("Action residual max_delta must be in (0, 1]")
        if self.oracle_target_tokens and not self.action_residual.enabled:
            raise ValueError("Oracle-token RLFT requires an enabled action residual")


@dataclass
class DiscretePriorPEFTBaseModelConfig(BaseModelConfig):
    """Shared actor config for discrete-prior PEFT models."""

    actor: DiscretePriorPEFTActorConfig = field(
        default_factory=DiscretePriorPEFTActorConfig
    )
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            _target_="torch.optim.AdamW",
            lr=2.5e-4,
            weight_decay=0.01,
        )
    )


@dataclass
class DiscretePriorPEFTRLFTModelConfig(DiscretePriorPEFTBaseModelConfig):
    """RLFT model config: discrete-prior PEFT actor plus task critic."""

    _target_: str = "protomotions.agents.peft.model.DiscretePriorPEFTModel"
    critic: Optional[MLPWithConcatConfig] = field(
        default=None,
        metadata={"help": "Task critic config. RLFT requires this to be set."},
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            _target_="torch.optim.AdamW",
            lr=1e-4,
        )
    )


@dataclass
class DiscretePriorPEFTSFTModelConfig(DiscretePriorPEFTBaseModelConfig):
    """SFT model config for supervised discrete-token labels."""

    _target_: str = "protomotions.agents.peft.sft_model.DiscretePriorPEFTSFTModel"
    token_perturb_rate: float = field(
        default=0.0,
        metadata={"help": "Probability of perturbing teacher-forced input tokens."},
    )
    token_perturb_mode: str = field(
        default="replace",
        metadata={"help": "Token perturbation mode: replace, neighbor, or mixed."},
    )
    autoregressive_student_prefix_rate: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of feeding a straight-through student token instead "
                "of the expert token at each within-frame autoregressive step."
            )
        },
    )
    fsq_scalar_aux_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary cross-entropy weight for the unpacked FSQ scalars. "
                "Zero preserves packed-token-only SFT."
            )
        },
    )
    sequence_action_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for differentiable decoded-action supervision."},
    )
    sequence_velocity_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for adjacent decoded-action delta matching."},
    )
    sequence_acceleration_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for decoded-action second-difference matching."},
    )
    sequence_action_loss_beta: float = field(
        default=0.05,
        metadata={"help": "Smooth-L1 beta used by decoded-action sequence losses."},
    )


@dataclass
class DiscretePriorPEFTRLFTAgentConfig(FineTuningAgentConfig):
    """RLFT config for PPO fine-tuning of a discrete-prior PEFT adapter."""

    _target_: str = "protomotions.agents.peft.prior_agent.DiscretePriorPEFTRLFTAgent"

    pretrained_modules: Dict[str, PretrainedModelConfig] = field(
        default_factory=dict,
        metadata={
            "help": "Frozen modules keyed by name. PEFT expects a whole prior "
            "model under 'prior'."
        },
    )
    model: DiscretePriorPEFTRLFTModelConfig = field(
        default_factory=DiscretePriorPEFTRLFTModelConfig
    )
    save_inference_checkpoint: bool = True

    num_mini_epochs: int = 2
    gradient_clip_val: float = 25.0
    entropy_coef: float = field(
        default=0.0,
        metadata={
            "help": "Entropy bonus coefficient added to PPO actor loss. Keep "
            "0.0 for SFT-warmed PEFT unless deliberately exploring."
        },
    )
    target_kl: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set, skip actor updates once minibatch actor/kl exceeds "
            "target_kl * 1.5 for the current rollout update."
        },
    )
    parameter_ema_decay: Optional[float] = field(
        default=None,
        metadata={"help": "EMA decay for trainable RLFT actor parameters."},
    )
    evaluate_parameter_ema: bool = field(
        default=True,
        metadata={"help": "Use RLFT actor EMA weights for physical evaluation."},
    )


@dataclass
class DiscretePriorPEFTSFTAgentConfig(SupervisedAgentConfig):
    """SFT config for supervised discrete-token PEFT training."""

    _target_: str = "protomotions.agents.peft.sft_agent.DiscretePriorPEFTSFTAgent"

    pretrained_modules: Dict[str, PretrainedModelConfig] = field(
        default_factory=dict,
        metadata={
            "help": "Frozen modules keyed by name. SFT expects a whole prior "
            "model under 'prior'."
        },
    )
    model: DiscretePriorPEFTSFTModelConfig = field(
        default_factory=DiscretePriorPEFTSFTModelConfig
    )
    rollout_actor: RolloutActor = RolloutActor.EXPERT
    dagger_beta_schedule: List[float] = field(
        default_factory=lambda: [0.95, 0.90, 0.80, 0.70],
        metadata={"help": "Expert-action probabilities for mixed DAgger stages."},
    )
    dagger_success_thresholds: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.70],
        metadata={
            "help": "Student-only success required to advance to each next beta."
        },
    )
    dagger_max_consecutive_student_steps: int = field(
        default=4,
        metadata={
            "help": "Safety cap on consecutive student actions during mixed rollout."
        },
    )
    parameter_ema_decay: Optional[float] = field(
        default=None,
        metadata={"help": "EMA decay for trainable SFT parameters."},
    )
    evaluate_parameter_ema: bool = field(
        default=True,
        metadata={"help": "Use parameter EMA weights for physical evaluation."},
    )
    loss: SupervisionLossConfig = field(
        default_factory=lambda: SupervisionLossConfig(
            loss_type=SupervisionLossType.DISCRETE_CROSS_ENTROPY,
            prediction_key=LATENT_LOGITS_KEY,
            target_key=TARGET_LATENT_KEY,
            label_smoothing=0.01,
            log_prefix="sft",
        ),
        metadata={"help": "Supervised loss over PEFT latent logits."},
    )
    save_inference_checkpoint: bool = True
    num_mini_epochs: int = 2
    gradient_clip_val: float = 25.0
    offline_cache_output: Optional[str] = field(
        default=None,
        metadata={"help": "Collect one deterministic expert cache and exit."},
    )
    offline_cache_split: str = field(
        default="train",
        metadata={"help": "Dataset split recorded in newly collected cache files."},
    )
    offline_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Train from cached expert transitions instead of simulator rollouts."},
    )
    offline_dataset_split: str = field(
        default="train",
        metadata={"help": "Comma-separated cached splits used for optimization."},
    )
    offline_num_epochs: int = field(
        default=100,
        metadata={"help": "Complete passes over an offline SFT dataset."},
    )
