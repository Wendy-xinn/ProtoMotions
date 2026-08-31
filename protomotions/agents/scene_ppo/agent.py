"""PPO warm-start that tolerates only newly added scene-residual parameters."""

import torch

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.utils.normalization import (
    materialize_lazy_running_stats_from_state_dict,
)


class SceneResidualPPO(PPO):
    """Warm-start a terrain tracker while preserving its base actor exactly."""

    _ALLOWED_MISSING = (
        "_actor.scene_gate",
        "_actor.scene_model.",
        "_critic.scene_gate",
        "_critic.scene_model.",
    )

    def create_model(self):
        model = super().create_model()
        # PPO's tree-wide initializer runs after module construction, so it
        # would otherwise overwrite SceneFeatureEncoder's zero-init request.
        # Re-apply it here to guarantee an exact motion-only policy at startup.
        scene_model = model._actor.scene_model
        if scene_model.config.zero_init_output:
            torch.nn.init.zeros_(scene_model.scene_output.weight)
            torch.nn.init.zeros_(scene_model.scene_output.bias)
        return model

    def create_optimizers(self, model):
        multiplier = self.config.scene_learning_rate_multiplier
        if multiplier == 1.0:
            return super().create_optimizers(model)
        if multiplier <= 0.0:
            raise ValueError("scene_learning_rate_multiplier must be positive")

        scene_params = []
        base_params = []
        for name, parameter in model._actor.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "scene_gate" or name.startswith("scene_model."):
                scene_params.append(parameter)
            else:
                base_params.append(parameter)
        base_lr = self.config.model.actor_optimizer.lr
        actor_optimizer = instantiate_optimizer(
            self.config.model.actor_optimizer,
            model._actor,
            params=[
                {"params": base_params},
                {"params": scene_params, "lr": base_lr * multiplier},
            ],
        )
        self.actor, self.actor_optimizer = self._setup_model_optimizer(
            model._actor, actor_optimizer
        )

        critic_optimizer = instantiate_optimizer(
            self.config.model.critic_optimizer, model._critic
        )
        self.critic, self.critic_optimizer = self._setup_model_optimizer(
            model._critic, critic_optimizer
        )

        if self.config.adaptive_lr.enabled:
            self.actor_lr = base_lr
            self.critic_lr = self.config.model.critic_optimizer.lr

    def _materialize_lazy_modules(self, dummy_obs_td):
        """Materialize first, then freeze the checkpoint-compatible actor path."""
        super()._materialize_lazy_modules(dummy_obs_td)
        if self.config.freeze_base_actor:
            for parameter in self.model._actor.mu.parameters():
                parameter.requires_grad_(False)
            for module in self.model._actor.mu.modules():
                if hasattr(module, "_freeze_running"):
                    module._freeze_running = True

    def _load_model_state_dict(self, model_state_dict):
        if self.config.reset_scene_on_warm_start:
            model_state_dict = {
                key: value
                for key, value in model_state_dict.items()
                if not key.startswith("_actor.scene_")
                and not key.startswith("_critic.scene_")
            }
        current_logstd = self.actor_module.logstd.data.clone()
        materialize_lazy_running_stats_from_state_dict(self.model, model_state_dict)
        self.model.materialize_from_state_dict(model_state_dict)
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)

        invalid_missing = [
            key
            for key in missing
            if not any(key == prefix or key.startswith(prefix) for prefix in self._ALLOWED_MISSING)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "Scene residual warm-start mismatch: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
        if not self.config.model.actor.learnable_std:
            self.actor_module.logstd.data.copy_(current_logstd)
        print(
            "Loaded base terrain checkpoint; initialized new scene residuals "
            f"({len(missing)} tensors missing by design)."
        )
