# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT model for PEFT adapters on a frozen discrete-token GPC prior."""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.peft.model import DiscretePriorPEFTModel


def factorized_fsq_cross_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    num_levels: int,
    scalars_per_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize packed-token logits into small FSQ scalar classifiers.

    A packed token is a mixed-radix integer with the first FSQ scalar as the
    least-significant digit. Reshaping the vocabulary therefore exposes axes in
    reverse scalar order. Marginal log probabilities preserve the original
    packed-token head while giving partial credit for correct scalar digits.
    """
    expected_vocab = num_levels**scalars_per_token
    if logits.shape[-1] != expected_vocab:
        raise ValueError(
            f"Expected packed vocabulary {expected_vocab}, got {logits.shape[-1]}"
        )
    if target_tokens.shape != logits.shape[:-1]:
        raise ValueError(
            f"Target shape {target_tokens.shape} does not match logits {logits.shape[:-1]}"
        )

    log_joint = F.log_softmax(logits, dim=-1).reshape(
        *logits.shape[:-1], *([num_levels] * scalars_per_token)
    )
    basis = target_tokens.new_tensor(
        [num_levels**i for i in range(scalars_per_token)]
    )
    scalar_targets = (
        target_tokens.unsqueeze(-1).div(basis, rounding_mode="floor") % num_levels
    )
    packed_rank = logits.ndim - 1
    scalar_losses = []
    scalar_correct = []
    for scalar_index in range(scalars_per_token):
        keep_axis = packed_rank + (scalars_per_token - 1 - scalar_index)
        reduce_axes = tuple(
            axis
            for axis in range(packed_rank, packed_rank + scalars_per_token)
            if axis != keep_axis
        )
        scalar_log_probs = (
            torch.logsumexp(log_joint, dim=reduce_axes)
            if reduce_axes
            else log_joint
        )
        scalar_target = scalar_targets[..., scalar_index]
        scalar_losses.append(
            F.nll_loss(
                scalar_log_probs.reshape(-1, num_levels),
                scalar_target.reshape(-1),
            )
        )
        scalar_correct.append(
            (scalar_log_probs.argmax(dim=-1) == scalar_target).float().mean()
        )
    return torch.stack(scalar_losses).mean(), torch.stack(scalar_correct).mean()


def factorized_fsq_expected_indices(
    logits: torch.Tensor,
    *,
    num_levels: int,
    scalars_per_token: int,
    straight_through_hard: bool = False,
) -> torch.Tensor:
    """Return differentiable FSQ indices from packed categorical logits.

    With ``straight_through_hard=True`` the forward pass exactly unpacks the
    argmax token used at deployment, while the backward pass follows the soft
    categorical probabilities.
    """
    expected_vocab = num_levels**scalars_per_token
    if logits.shape[-1] != expected_vocab:
        raise ValueError(
            f"Expected packed vocabulary {expected_vocab}, got {logits.shape[-1]}"
        )
    joint_probs = F.softmax(logits, dim=-1)
    if straight_through_hard:
        hard = F.one_hot(
            joint_probs.argmax(dim=-1), num_classes=expected_vocab
        ).to(joint_probs.dtype)
        joint_probs = hard + joint_probs - joint_probs.detach()
    joint_probs = joint_probs.reshape(
        *logits.shape[:-1], *([num_levels] * scalars_per_token)
    )
    packed_rank = logits.ndim - 1
    levels = torch.arange(num_levels, device=logits.device, dtype=logits.dtype)
    expected_scalars = []
    for scalar_index in range(scalars_per_token):
        keep_axis = packed_rank + (scalars_per_token - 1 - scalar_index)
        reduce_axes = tuple(
            axis
            for axis in range(packed_rank, packed_rank + scalars_per_token)
            if axis != keep_axis
        )
        marginal = joint_probs.sum(dim=reduce_axes) if reduce_axes else joint_probs
        expected_scalars.append((marginal * levels).sum(dim=-1))
    return torch.stack(expected_scalars, dim=-1).flatten(start_dim=-2)


class DiscretePriorPEFTSFTModel(DiscretePriorPEFTModel):
    """Discrete-prior PEFT model used by the supervised SFT agent.

    Rollout uses the frozen target encoder as the expert: encode the target
    motion into prior tokens, decode those tokens to an action, and store the
    tokens as supervision labels. Optimization replays the batch with teacher
    forcing and writes ``latent_logits`` for the generic supervision loss.
    """

    def collect_expert_rollout(self, tensordict: TensorDict) -> TensorDict:
        target_prior_tokens = self._actor.predict_target_prior_tokens(tensordict)
        fsq_indices = self._actor.prior_tokens_to_fsq_indices(target_prior_tokens)
        fsq_codes = self._actor.fsq_indices_to_codes(fsq_indices)
        action = self._actor._decode(tensordict, fsq_codes)

        tensordict["action"] = action
        tensordict["mean_action"] = action
        tensordict["prior_tokens"] = target_prior_tokens
        tensordict[LATENT_KEY] = target_prior_tokens
        tensordict[TARGET_LATENT_KEY] = target_prior_tokens
        # The expert encoder is deterministic and SFT trains with
        # cross-entropy, so neglogp is an unused rollout-contract placeholder.
        if "neglogp" in self.out_keys:
            tensordict["neglogp"] = torch.zeros(
                action.shape[0],
                self._actor.num_prior_tokens,
                device=action.device,
                dtype=action.dtype,
            )
        return tensordict

    def collect_student_rollout(self, tensordict: TensorDict) -> TensorDict:
        """Generate tokens autoregressively from deployable task inputs.

        ``forward`` is intentionally the teacher-forced optimization path for
        this SFT model.  Evaluation and inference must call this explicit
        method instead; otherwise a TensorDict without cached target tokens
        would silently enter ``collect_expert_rollout`` and execute the oracle.
        """
        return super().forward_rollout(tensordict)

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        expert_td = self.collect_expert_rollout(tensordict.clone())
        return self.forward(expert_td)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if not isinstance(tensordict, TensorDict):
            raise TypeError(
                "DiscretePriorPEFTSFTModel.forward expects a TensorDict input."
            )
        if TARGET_LATENT_KEY not in tensordict:
            tensordict = self.collect_expert_rollout(tensordict)
        target_prior_tokens = tensordict[TARGET_LATENT_KEY].detach()

        teacher_tokens = self._actor.perturb_tokens(
            target_prior_tokens,
            rate=self.config.token_perturb_rate,
            mode=self.config.token_perturb_mode,
        )
        prior_dict = self._actor.build_prior_input(tensordict, tokens=teacher_tokens)
        student_prefix_rate = float(
            getattr(self.config, "autoregressive_student_prefix_rate", 0.0)
        )
        if student_prefix_rate > 0.0:
            tensordict[LATENT_LOGITS_KEY] = self._actor.forward_with_student_prefixes(
                prior_dict, student_prefix_rate
            )
        else:
            tensordict[LATENT_LOGITS_KEY] = self._actor(prior_dict)
        return tensordict

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss,
        log_prefix: str = "model",
    ):
        loss, logs = super().compute_model_loss(
            tensordict,
            current_epoch=current_epoch,
            zero_loss=zero_loss,
            log_prefix=log_prefix,
        )
        scalar_weight = float(self.config.fsq_scalar_aux_weight)
        if scalar_weight > 0.0:
            scalar_loss, scalar_accuracy = factorized_fsq_cross_entropy(
                tensordict[LATENT_LOGITS_KEY],
                tensordict[TARGET_LATENT_KEY],
                num_levels=self._actor.num_fsq_levels,
                scalars_per_token=self._actor.fsq_scalars_per_prior_token,
            )
            weighted = scalar_loss * scalar_weight
            loss = loss + weighted
            logs.update(
                {
                    f"{log_prefix}/fsq_scalar_cross_entropy": scalar_loss.detach(),
                    f"{log_prefix}/fsq_scalar_accuracy": scalar_accuracy.detach(),
                    f"{log_prefix}/fsq_scalar_aux_loss": weighted.detach(),
                }
            )

        sequence_loss, sequence_logs = self._compute_sequence_action_loss(
            tensordict, log_prefix=log_prefix
        )
        loss = loss + sequence_loss
        logs.update(sequence_logs)
        return loss, logs

    def _expected_decoded_action(
        self, tensordict: TensorDict, logits: torch.Tensor
    ) -> torch.Tensor:
        expected_indices = factorized_fsq_expected_indices(
            logits,
            num_levels=self._actor.num_fsq_levels,
            scalars_per_token=self._actor.fsq_scalars_per_prior_token,
            straight_through_hard=True,
        )
        fsq_codes = self._actor.fsq_indices_to_codes(expected_indices)
        return self._actor._decode(tensordict.clone(), fsq_codes)

    def _temporal_tensordict(
        self, tensordict: TensorDict, prefix: str
    ) -> TensorDict:
        keys = [*self._actor.in_keys, TARGET_LATENT_KEY]
        return TensorDict(
            {key: tensordict[f"{prefix}{key}"] for key in keys},
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _compute_sequence_action_loss(
        self, tensordict: TensorDict, *, log_prefix: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        action_weight = float(
            getattr(self.config, "sequence_action_loss_weight", 0.0)
        )
        velocity_weight = float(
            getattr(self.config, "sequence_velocity_loss_weight", 0.0)
        )
        acceleration_weight = float(
            getattr(self.config, "sequence_acceleration_loss_weight", 0.0)
        )
        zero = tensordict[LATENT_LOGITS_KEY].sum() * 0.0
        if max(action_weight, velocity_weight, acceleration_weight) <= 0.0:
            return zero, {}

        beta = float(getattr(self.config, "sequence_action_loss_beta", 0.05))
        if beta <= 0.0:
            raise ValueError("sequence_action_loss_beta must be positive")

        predicted = self._expected_decoded_action(
            tensordict, tensordict[LATENT_LOGITS_KEY]
        )
        oracle = tensordict["oracle_action"]
        total = zero
        logs = {}

        action_loss = F.smooth_l1_loss(
            predicted, oracle, beta=beta, reduction="none"
        ).mean(dim=-1).mean()
        if action_weight > 0.0:
            total = total + action_weight * action_loss
        logs[f"{log_prefix}/sequence_action_loss"] = action_loss.detach()

        if velocity_weight > 0.0 or acceleration_weight > 0.0:
            previous_td = self._temporal_tensordict(tensordict, "sequence_prev_")
            previous_td = self.forward(previous_td)
            previous_predicted = self._expected_decoded_action(
                previous_td, previous_td[LATENT_LOGITS_KEY]
            )
            previous_oracle = tensordict["sequence_prev_oracle_action"]
            predicted_velocity = predicted - previous_predicted
            oracle_velocity = oracle - previous_oracle
            velocity_error = F.smooth_l1_loss(
                predicted_velocity,
                oracle_velocity,
                beta=beta,
                reduction="none",
            ).mean(dim=-1)
            velocity_mask = tensordict["sequence_velocity_mask"].float()
            velocity_loss = self._masked_mean(velocity_error, velocity_mask)
            if velocity_weight > 0.0:
                total = total + velocity_weight * velocity_loss
            logs[f"{log_prefix}/sequence_velocity_loss"] = velocity_loss.detach()

        if acceleration_weight > 0.0:
            previous2_td = self._temporal_tensordict(tensordict, "sequence_prev2_")
            previous2_td = self.forward(previous2_td)
            previous2_predicted = self._expected_decoded_action(
                previous2_td, previous2_td[LATENT_LOGITS_KEY]
            )
            previous2_oracle = tensordict["sequence_prev2_oracle_action"]
            predicted_acceleration = (
                predicted - 2.0 * previous_predicted + previous2_predicted
            )
            oracle_acceleration = oracle - 2.0 * previous_oracle + previous2_oracle
            acceleration_error = F.smooth_l1_loss(
                predicted_acceleration,
                oracle_acceleration,
                beta=beta,
                reduction="none",
            ).mean(dim=-1)
            acceleration_mask = tensordict["sequence_acceleration_mask"].float()
            acceleration_loss = self._masked_mean(
                acceleration_error, acceleration_mask
            )
            total = total + acceleration_weight * acceleration_loss
            logs[f"{log_prefix}/sequence_acceleration_loss"] = (
                acceleration_loss.detach()
            )

        logs[f"{log_prefix}/sequence_action_aux_loss"] = total.detach()
        return total, logs
