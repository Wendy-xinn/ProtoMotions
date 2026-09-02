# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLFT agent for PEFT adapters on a frozen discrete-token GPC prior."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from lightning.fabric import Fabric
import torch
from torch import Tensor
from tensordict import TensorDict

from protomotions.agents.common.autoregressive import (
    kl_divergence_on_prior_support,
    kl_divergence_sampling_distribution,
)
from protomotions.agents.common.latent import compute_discrete_latent_ppo_loss
from protomotions.agents.fine_tuning.agent import FineTuningAgent
from protomotions.agents.peft.prior_setup import DiscretePriorPEFTSetupMixin

log = logging.getLogger(__name__)


class DiscretePriorPEFTRLFTAgent(DiscretePriorPEFTSetupMixin, FineTuningAgent):
    """PPO/RLFT for a PEFT-adapted discrete GPC prior.

    This class is intentionally RL-only: it owns PPO ratio loss, critic loss,
    RLFT checkpoint warm-start from SFT, and the frozen KL/sampling reference.
    Supervised PEFT training lives in ``sft_agent.py`` and uses the generic
    supervised loop.
    """

    def __init__(self, fabric: Fabric, env, config, root_dir: Optional[Path] = None):
        if config.model.critic is None:
            raise ValueError(
                "DiscretePriorPEFTRLFTAgent requires config.model.critic for RLFT."
            )
        super().__init__(fabric, env, config, root_dir=root_dir)

    def create_optimizers(self, model):
        super().create_optimizers(model)
        self._initialize_parameter_ema()

    def _initialize_parameter_ema(self):
        decay = getattr(self.config, "parameter_ema_decay", None)
        if decay is not None and not 0.0 < decay < 1.0:
            raise ValueError("parameter_ema_decay must be in (0, 1)")
        self._parameter_ema = (
            {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if name.startswith("_actor.") and parameter.requires_grad
            }
            if decay is not None
            else {}
        )
        self._parameter_ema_backup = None

    @torch.no_grad()
    def _update_parameter_ema(self):
        parameter_ema = getattr(self, "_parameter_ema", {})
        if not parameter_ema:
            return
        decay = float(self.config.parameter_ema_decay)
        for name, parameter in self.model.named_parameters():
            if name in parameter_ema:
                parameter_ema[name].lerp_(parameter.detach(), 1.0 - decay)

    @torch.no_grad()
    def begin_evaluation_weights(self):
        if (
            not getattr(self.config, "evaluate_parameter_ema", True)
            or not getattr(self, "_parameter_ema", {})
        ):
            return
        if self._parameter_ema_backup is not None:
            raise RuntimeError("Parameter EMA evaluation weights are already active")
        self._parameter_ema_backup = {}
        for name, parameter in self.model.named_parameters():
            if name in self._parameter_ema:
                self._parameter_ema_backup[name] = parameter.detach().clone()
                parameter.copy_(self._parameter_ema[name])

    @torch.no_grad()
    def end_evaluation_weights(self):
        if getattr(self, "_parameter_ema_backup", None) is None:
            return
        for name, parameter in self.model.named_parameters():
            if name in self._parameter_ema_backup:
                parameter.copy_(self._parameter_ema_backup[name])
        self._parameter_ema_backup = None

    def load(self, checkpoint, load_env=True, load_training_state: bool = True):
        warm_start = checkpoint is not None and not load_training_state
        self._peft_loading_training_state = load_training_state
        self._peft_warm_started_from_sft = False
        try:
            super().load(
                checkpoint,
                load_env=load_env,
                load_training_state=load_training_state,
            )
        finally:
            self._peft_loading_training_state = False
        if warm_start:
            # Lazy model materialization can call generate() before checkpoint
            # loading and pin an initialization-time reference. RLFT must
            # instead constrain updates to the loaded SFT policy.
            self.model._actor.reset_reference()
        require_existing = (
            checkpoint is not None
            and load_training_state
            and not getattr(self, "_peft_warm_started_from_sft", False)
        )
        self._prepare_rlft_prior_reference(
            require_existing=require_existing,
        )

    def fit(self):
        self._prepare_rlft_prior_reference()
        return super().fit()

    @torch.no_grad()
    def _after_load_model_state_dict(self, state_dict) -> None:
        super()._after_load_model_state_dict(state_dict)
        if getattr(self, "_peft_loading_training_state", False):
            return
        parameter_ema = state_dict.get("parameter_ema")
        if parameter_ema:
            named_parameters = dict(self.model.named_parameters())
            missing = sorted(set(parameter_ema) - set(named_parameters))
            if missing:
                raise KeyError(
                    "SFT EMA checkpoint contains parameters absent from the RLFT model: "
                    f"{missing[:10]}"
                )
            for name, value in parameter_ema.items():
                named_parameters[name].copy_(
                    value.to(
                        device=named_parameters[name].device,
                        dtype=named_parameters[name].dtype,
                    )
                )
            log.info(
                "Warm-started RLFT actor from %d SFT EMA parameters.",
                len(parameter_ema),
            )
        self._initialize_parameter_ema()

    def _prepare_rlft_prior_reference(self, *, require_existing: bool = False):
        """Capture or validate the reference policy used by RLFT KL/sampling.

        Warm-starts capture the loaded SFT/current student policy once at fit
        start and may then clear only the active student adapter. True resumes
        must already carry reference state in the checkpoint so the reference is
        not silently rebuilt from a changed student or configured prior.
        """
        actor = self.model._actor
        if require_existing:
            actor.require_reference()
            return

        captured = actor.capture_reference()
        if captured and self.config.model.actor.peft.clear_peft:
            actor.prior_with_peft.clear_peft()

    @torch.no_grad()
    def _clamp_peft_m(self):
        # DoRA magnitude m is unbounded by construction. Clamping keeps RLFT
        # adapter updates near the frozen prior's scale instead of letting a few
        # large magnitudes dominate the token logits.
        bound = getattr(self.config.model.actor.peft, "m_clamp", None)
        if bound is None:
            return
        peft = self.model._actor.prior_with_peft
        for module in peft.base_prior._transformer.modules():
            if hasattr(module, "m"):
                module.m.clamp_(-bound, bound)

    def actor_step(self, batch_dict: Dict) -> Tuple[Tensor, Dict]:
        return self._actor_step_discrete_ppo(batch_dict)

    def _actor_step_discrete_ppo(self, batch_dict: Dict) -> Tuple[Tensor, Dict]:
        """Compute the discrete-token PPO loss for the PEFT actor."""
        actor = self.model._actor
        prior_tokens = batch_dict["prior_tokens"].detach()
        old_neglogp = batch_dict["neglogp"].detach()
        advantages = batch_dict["advantages"].detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO is applied in token space: rollout sampled GPC prior tokens from
        # the adapter, then optimization replays those exact tokens under
        # the current adapter to compute a categorical likelihood ratio.
        prior_dict = actor.build_prior_input(batch_dict, tokens=prior_tokens)
        logits = self.actor(prior_dict)

        base_prior_logits = None
        prior_with_peft = getattr(actor, "prior_with_peft", None)
        loss_temperature = getattr(prior_with_peft, "temperature", 1.0)
        loss_top_p = getattr(prior_with_peft, "top_p", 1.0)
        if (
            prior_with_peft is not None
            and getattr(prior_with_peft, "sampling_mode", None) == "prior_constraint"
        ):
            base_forward = getattr(
                prior_with_peft,
                "forward_base_prior",
                prior_with_peft.forward_prior,
            )
            base_prior_logits = base_forward(prior_dict)
            loss_top_p = prior_with_peft.prior_top_p

        ppo_loss, ppo_log_dict = compute_discrete_latent_ppo_loss(
            logits=logits,
            selected=prior_tokens,
            old_neglogp=old_neglogp,
            advantages=advantages,
            e_clip=self.e_clip,
            entropy_coef=self.config.entropy_coef,
            temperature=loss_temperature,
            top_p=loss_top_p,
            prior_logits=base_prior_logits,
            log_prefix="actor",
        )

        kl_coeff = actor.kl_coeff
        kl_prior_loss = torch.tensor(0.0, device=logits.device)
        if kl_coeff > 0:
            reference_prior_dict = actor.build_reference_prior_input(
                batch_dict,
                tokens=prior_tokens,
            )
            sft_reference_logits = actor.prior_with_peft.forward_prior(
                reference_prior_dict
            )
            if base_prior_logits is not None:
                kl_prior_loss = kl_divergence_on_prior_support(
                    logits,
                    sft_reference_logits,
                    base_prior_logits,
                    p=loss_top_p,
                    temperature=loss_temperature,
                    reduction="mean",
                )
            else:
                kl_prior_loss = kl_divergence_sampling_distribution(
                    logits,
                    sft_reference_logits,
                    p=loss_top_p,
                    temperature=loss_temperature,
                    prior_constraint=False,
                    reduction="mean",
                )

        loss = ppo_loss + kl_coeff * kl_prior_loss
        log_dict = {
            # Keep the historical key for dashboards and old checkpoint tests.
            "actor/kl_prior_loss": kl_prior_loss.detach(),
            "actor/kl_sft_reference_loss": kl_prior_loss.detach(),
            "actor/kl_coeff": kl_coeff,
            "actor/adv_mean": advantages.mean().detach(),
            "actor/adv_std": advantages.std().detach(),
            "losses/actor_loss": loss.detach(),
            "stats/reward_mean": batch_dict["rewards"].mean().detach(),
        }
        log_dict.update(ppo_log_dict)
        return loss, log_dict

    def critic_step(self, batch_dict: Dict) -> Tuple[Tensor, Dict]:
        """Critic MSE loss against computed returns."""
        batch_td = TensorDict(
            batch_dict,
            batch_size=batch_dict["returns"].shape[0],
        )
        batch_td = self.critic(batch_td)
        out_key = self.model._critic.out_keys[0]
        values = batch_td[out_key].squeeze(-1)
        returns = batch_dict["returns"].detach()

        critic_loss = torch.nn.functional.mse_loss(values, returns)
        log_dict = {
            "losses/critic_loss": critic_loss.detach(),
            "stats/value_mean": values.mean().detach(),
            "stats/return_mean": returns.mean().detach(),
        }
        return critic_loss, log_dict

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        """Run one PEFT PPO minibatch update."""
        if batch_idx == 0:
            self._kl_early_stop_triggered = False

        iter_log_dict = {}
        actor_loss, actor_log_dict = self.actor_step(batch_dict)
        iter_log_dict.update(actor_log_dict)

        skip_actor = bool(getattr(self, "_kl_early_stop_triggered", False))
        target_kl = getattr(self.config, "target_kl", None)
        if not skip_actor and target_kl is not None and "actor/kl" in actor_log_dict:
            actor_kl = actor_log_dict["actor/kl"].detach().item()
            if actor_kl > target_kl * 1.5:
                self._kl_early_stop_triggered = True
                skip_actor = True
                log.warning(
                    "Epoch %s batch %s: skipping PEFT actor update "
                    "(actor/kl %.4f > target_kl * 1.5 %.4f)",
                    getattr(self, "current_epoch", 0),
                    batch_idx,
                    actor_kl,
                    target_kl * 1.5,
                )

        iter_log_dict["ppo/kl_early_stopped"] = torch.tensor(
            float(getattr(self, "_kl_early_stop_triggered", False)),
            device=self.device,
        )

        if skip_actor:
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                1.0, device=self.device
            )
        else:
            actor_grad_clip_dict = self._step_optimizer(
                loss=actor_loss,
                model=self.actor,
                optimizer=self.actor_optimizer,
                model_name="actor",
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self._clamp_peft_m()
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                0.0, device=self.device
            )

        critic_loss, critic_log_dict = self.critic_step(batch_dict)
        iter_log_dict.update(critic_log_dict)
        critic_grad_clip_dict = self._step_optimizer(
            loss=critic_loss,
            model=self.critic,
            optimizer=self.critic_optimizer,
            model_name="critic",
        )
        iter_log_dict.update(critic_grad_clip_dict)
        self._update_parameter_ema()
        return iter_log_dict

    def _load_training_state(self, state_dict):
        warm_start_from_sft = self.has_critic and "critic_optimizer" not in state_dict
        self._peft_warm_started_from_sft = warm_start_from_sft
        if not warm_start_from_sft:
            super()._load_training_state(state_dict)
            saved_ema = state_dict.get("rlft_parameter_ema")
            if saved_ema is not None and self._parameter_ema:
                for name, value in saved_ema.items():
                    if name in self._parameter_ema:
                        self._parameter_ema[name].copy_(value.to(self.device))
            return

        log.info(
            "Using SFT checkpoint as RLFT initialization; restoring actor "
            "optimizer state and leaving critic optimizer fresh."
        )
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.current_epoch = 0
        self.step_count = 0
        self.fit_start_time = None
        self.best_evaluated_score = None

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        if getattr(self, "_parameter_ema", {}):
            state_dict["rlft_parameter_ema"] = {
                name: value.detach().cpu().clone()
                for name, value in self._parameter_ema.items()
            }
        return state_dict

    def get_inference_state_dict(self, state_dict, model_state_dict=None):
        if model_state_dict is None and getattr(self, "_parameter_ema", {}):
            model_state_dict = dict(self.model.state_dict())
            model_state_dict.update(self._parameter_ema)
        return super().get_inference_state_dict(
            state_dict, model_state_dict=model_state_dict
        )
