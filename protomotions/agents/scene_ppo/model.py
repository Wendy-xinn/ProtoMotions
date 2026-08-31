"""Scene residual actor/critic that retain old checkpoint parameter names."""

import torch
import torch.nn.functional as F
from torch import distributions, nn
from tensordict import TensorDict

from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.ppo.model import PPOActor
from protomotions.utils.hydra_replacement import get_class

from .config import SceneResidualCriticConfig, SceneResidualPPOActorConfig


class SceneFeatureEncoder(nn.Module):
    """Permutation-invariant point/object encoders followed by late fusion.

    Point samples and object slots are deliberately encoded by separate shared
    networks.  The validity channel masks padding before max pooling, so neither
    point order nor the number of padded objects changes the representation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_keys = list(config.in_keys)
        self.out_keys = list(config.out_keys)
        if len(self.out_keys) != 1:
            raise ValueError("SceneFeatureEncoder requires exactly one output key")
        if config.condition_mode not in {"full", "no_scene"}:
            raise ValueError(
                "SceneFeatureEncoder condition_mode must be full or no_scene; "
                f"got {config.condition_mode!r}"
            )
        self.point_encoder = nn.Sequential(
            nn.Linear(config.point_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.point_embedding_dim),
            nn.ReLU(),
        )
        self.object_encoder = None
        if config.object_key in self.in_keys:
            self.object_encoder = nn.Sequential(
                nn.Linear(config.object_feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.object_embedding_dim),
                nn.ReLU(),
            )
        self._head_keys = (
            config.head_pose_key,
            config.head_mask_key,
            config.head_time_key,
        )
        self.query_encoder = None
        if config.future_query_key in self.in_keys:
            self.query_encoder = nn.Sequential(
                nn.LazyLinear(config.hidden_dim),
                nn.ReLU(),
            )
        self.head_encoder = None
        if all(key in self.in_keys for key in self._head_keys):
            # Encode sparse, variable-horizon landmarks per time step instead
            # of flattening the complete clip into an unordered auxiliary MLP.
            self.head_encoder = nn.Sequential(
                nn.LazyLinear(config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.head_embedding_dim),
                nn.ReLU(),
            )
        reserved = {
            config.point_key,
            config.object_key,
            config.future_query_key,
            config.interaction_target_key,
            *self._head_keys,
        }
        self._aux_keys = [key for key in self.in_keys if key not in reserved]
        self.aux_encoder = None
        if self._aux_keys:
            self.aux_encoder = nn.Sequential(
                nn.LazyLinear(config.hidden_dim),
                nn.ReLU(),
            )
        self.fusion = nn.Sequential(
            nn.LazyLinear(config.hidden_dim),
            nn.ReLU(),
        )
        self.scene_output = nn.Linear(config.hidden_dim, config.num_out)
        if config.zero_init_output:
            nn.init.zeros_(self.scene_output.weight)
            nn.init.zeros_(self.scene_output.bias)
        self.interaction_projection = None
        self.distance_head = None
        self.contact_head = None
        if config.num_interaction_targets > 0:
            self.interaction_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.interaction_latent_dim),
                nn.ReLU(),
            )
            self.distance_head = nn.Linear(
                config.interaction_latent_dim, config.num_interaction_targets
            )
            self.contact_head = nn.Linear(
                config.interaction_latent_dim, config.num_interaction_targets
            )

    @staticmethod
    def _masked_max(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = features.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        batch = tensordict[self.config.point_key].shape[0]
        points = tensordict[self.config.point_key].reshape(
            batch, -1, self.config.point_feature_dim
        )
        point_latent = self._masked_max(
            self.point_encoder(points), points[..., -1] > 0.5
        )
        if self.config.object_key in self.in_keys:
            assert self.object_encoder is not None
            objects = tensordict[self.config.object_key].reshape(
                batch, -1, self.config.object_feature_dim
            )
            # Object validity is channel 15 in the 16-channel state prefix.
            object_latent = self._masked_max(
                self.object_encoder(objects), objects[..., 15] > 0.5
            )
        else:
            # A strict ego-only experiment must not leak complete scene object
            # states for objects outside the reconstructed camera history.
            object_latent = points.new_zeros(
                (batch, self.config.object_embedding_dim)
            )
        if self.head_encoder is not None:
            head_times = tensordict[self.config.head_time_key]
            num_head_steps = head_times.shape[-1]
            head_poses = tensordict[self.config.head_pose_key].reshape(
                batch, num_head_steps, -1
            )
            head_masks = tensordict[self.config.head_mask_key].reshape(
                batch, num_head_steps, -1
            )
            # Time offsets make trajectory order explicit. The masked-mimic
            # control publishes future landmarks in chronological order.
            head_features = torch.cat(
                [head_poses, head_masks, head_times.unsqueeze(-1)], dim=-1
            )
            head_step_latent = self.head_encoder(head_features)
            head_valid = head_masks.to(torch.bool).any(dim=-1)
            head_latent = self._masked_max(head_step_latent, head_valid)
        else:
            head_latent = points.new_zeros(
                (batch, self.config.head_embedding_dim)
            )
        fused_parts = [point_latent, object_latent, head_latent]
        if self.query_encoder is not None:
            fused_parts.append(
                self.query_encoder(tensordict[self.config.future_query_key])
            )
        if self.aux_encoder is not None:
            auxiliary = self.aux_encoder(
                torch.cat([tensordict[key] for key in self._aux_keys], dim=-1)
            )
            fused_parts.append(auxiliary)
        fused = self.fusion(torch.cat(fused_parts, dim=-1))
        scene_output = self.scene_output(fused)
        if self.config.condition_mode == "no_scene":
            scene_output = scene_output * 0.0
        tensordict[self.out_keys[0]] = scene_output
        if self.interaction_projection is not None:
            assert self.distance_head is not None and self.contact_head is not None
            interaction_latent = self.interaction_projection(fused)
            if self.config.condition_mode == "no_scene":
                interaction_latent = interaction_latent * 0.0
            tensordict[self.config.interaction_latent_key] = interaction_latent
            tensordict[self.config.distance_prediction_key] = torch.sigmoid(
                self.distance_head(interaction_latent)
            )
            tensordict[self.config.contact_prediction_key] = self.contact_head(
                interaction_latent
            )
        if log_internals:
            tensordict["scene_point_latent"] = point_latent
            tensordict["scene_object_latent"] = object_latent
            tensordict["ego_head_trajectory_latent"] = head_latent
        return tensordict

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss,
        log_prefix: str = "model",
    ):
        del current_epoch
        config = self.config
        enabled = (
            config.num_interaction_targets > 0
            and config.condition_mode != "no_scene"
            and config.interaction_target_key in tensordict
        )
        if not enabled:
            return zero_loss * 0.0, {}

        target = tensordict[config.interaction_target_key].reshape(
            tensordict.batch_size[0], config.num_interaction_targets, 2
        )
        distance_target = target[..., 0]
        contact_target = target[..., 1]
        distance_pred = tensordict[config.distance_prediction_key]
        contact_logits = tensordict[config.contact_prediction_key]

        distance_loss = F.smooth_l1_loss(distance_pred, distance_target)
        contact_bce = F.binary_cross_entropy_with_logits(
            contact_logits, contact_target, reduction="none"
        )
        contact_probability = torch.sigmoid(contact_logits)
        focal_probability = torch.where(
            contact_target > 0.5,
            contact_probability,
            1.0 - contact_probability,
        )
        contact_loss = (
            (1.0 - focal_probability).pow(config.contact_focal_gamma)
            * contact_bce
            * torch.where(
                contact_target > 0.5,
                contact_bce.new_tensor(config.contact_positive_weight),
                contact_bce.new_tensor(1.0),
            )
        ).mean()
        weighted_distance = config.distance_loss_weight * distance_loss
        weighted_contact = config.contact_loss_weight * contact_loss
        loss = weighted_distance + weighted_contact
        with torch.no_grad():
            contact_pred = contact_logits > 0.0
            contact_true = contact_target > 0.5
            contact_accuracy = (contact_pred == contact_true).float().mean()
            true_positive = (contact_pred & contact_true).float().sum()
            predicted_positive = contact_pred.float().sum()
            target_positive = contact_true.float().sum()
            contact_precision = true_positive / predicted_positive.clamp_min(1.0)
            contact_recall = true_positive / target_positive.clamp_min(1.0)
            contact_f1 = (
                2.0
                * contact_precision
                * contact_recall
                / (contact_precision + contact_recall).clamp_min(1e-8)
            )
        return loss, {
            f"{log_prefix}/scene_distance_loss": distance_loss.detach(),
            f"{log_prefix}/scene_contact_focal_loss": contact_loss.detach(),
            f"{log_prefix}/scene_contact_accuracy": contact_accuracy,
            f"{log_prefix}/scene_contact_precision": contact_precision,
            f"{log_prefix}/scene_contact_recall": contact_recall,
            f"{log_prefix}/scene_contact_f1": contact_f1,
            f"{log_prefix}/scene_contact_positive_rate": contact_true.float().mean(),
            f"{log_prefix}/scene_interaction_loss": loss.detach(),
        }


class TrajectorySceneCrossAttentionEncoder(nn.Module):
    """Encode ordered head intent and use it to query causal scene memory.

    Point order is intentionally irrelevant. Head landmarks retain their time
    order through a temporal Transformer. Cross-attention then selects geometry
    that is relevant to the future camera path instead of globally max-pooling
    the complete reconstructed scene.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_keys = list(config.in_keys)
        self.out_keys = list(config.out_keys)
        if len(self.out_keys) != 1:
            raise ValueError("Encoder requires exactly one output key")
        if config.condition_mode not in {
            "full",
            "head_only",
            "scene_only",
            "no_condition",
        }:
            raise ValueError(
                "condition_mode must be one of full, head_only, scene_only, "
                f"or no_condition; got {config.condition_mode!r}"
            )

        dim = config.model_dim
        self.point_projection = nn.Sequential(
            nn.Linear(config.point_feature_dim, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.history_projection = None
        self.history_type_embedding = None
        if config.use_scene_history_token:
            if config.point_feature_dim < 10:
                raise ValueError(
                    "Scene history token requires 10-channel point features"
                )
            self.history_projection = nn.Sequential(
                nn.Linear(16, dim), nn.LayerNorm(dim), nn.GELU()
            )
            self.history_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.head_pose_projection = nn.Sequential(
            nn.LazyLinear(dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.head_time_projection = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.head_transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_head_layers, norm=nn.LayerNorm(dim)
        )
        self.scene_attention = nn.MultiheadAttention(
            dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.aux_projection = nn.Sequential(
            nn.LazyLinear(dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_out),
        )

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
        """MultiheadAttention cannot accept a row with every token masked."""
        padding = ~valid
        empty = ~valid.any(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        return padding

    @staticmethod
    def _masked_mean(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    @staticmethod
    def _scene_history_features(
        points: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Summarize the complete causal map in one order-invariant token."""
        weights = valid.to(points.dtype)
        count = weights.sum(dim=1, keepdim=True)
        denominator = count.clamp_min(1.0)
        xyz = points[..., :3]
        current = points[..., 7].clamp(0.0, 1.0)
        age = points[..., 8].clamp(0.0, 1.0)
        static = points[..., 6].clamp(0.0, 1.0)

        centroid = (xyz * weights.unsqueeze(-1)).sum(1) / denominator
        centered = xyz - centroid.unsqueeze(1)
        spread = torch.sqrt(
            (centered.square() * weights.unsqueeze(-1)).sum(1) / denominator
            + 1.0e-8
        )
        distances = torch.linalg.vector_norm(xyz, dim=-1)
        nearest = distances.masked_fill(~valid, float("inf")).amin(1, keepdim=True)
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
        farthest = distances.masked_fill(~valid, 0.0).amax(1, keepdim=True)
        mean_distance = (distances * weights).sum(1, keepdim=True) / denominator
        age_mean = (age * weights).sum(1, keepdim=True) / denominator
        age_max = age.masked_fill(~valid, 0.0).amax(1, keepdim=True)

        return torch.cat(
            [
                count / points.shape[1],
                (current * weights).sum(1, keepdim=True) / denominator,
                ((1.0 - current) * weights).sum(1, keepdim=True) / denominator,
                (static * weights).sum(1, keepdim=True) / denominator,
                age_mean,
                age_max,
                centroid,
                spread,
                nearest,
                mean_distance,
                farthest,
                (count > 0).to(points.dtype),
            ],
            dim=-1,
        )

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        points_flat = tensordict[self.config.point_key]
        batch = points_flat.shape[0]
        points = points_flat.reshape(batch, -1, self.config.point_feature_dim)
        point_valid = points[..., -1] > 0.5
        point_tokens = self.point_projection(points)
        point_tokens = point_tokens * point_valid.unsqueeze(-1)
        scene_tokens = point_tokens
        scene_valid = point_valid
        history_token = None
        if self.history_projection is not None:
            assert self.history_type_embedding is not None
            history_features = self._scene_history_features(points, point_valid)
            history_token = (
                self.history_projection(history_features).unsqueeze(1)
                + self.history_type_embedding
            )
            history_valid = point_valid.any(dim=1, keepdim=True)
            history_token = history_token * history_valid.unsqueeze(-1)
            scene_tokens = torch.cat([history_token, point_tokens], dim=1)
            scene_valid = torch.cat([history_valid, point_valid], dim=1)

        head_times = tensordict[self.config.head_time_key]
        steps = head_times.shape[-1]
        head_poses = tensordict[self.config.head_pose_key].reshape(batch, steps, -1)
        head_masks = tensordict[self.config.head_mask_key].reshape(batch, steps, -1)
        head_valid = head_masks.bool().any(dim=-1)
        head_input = torch.cat([head_poses, head_masks.to(head_poses.dtype)], dim=-1)
        # Offsets are normalized per sample, preserving order without making
        # the embedding depend on the clip's absolute frame rate or duration.
        time_scale = head_times.amax(dim=1, keepdim=True).clamp_min(1.0e-3)
        normalized_time = (head_times / time_scale).unsqueeze(-1)
        head_tokens = (
            self.head_pose_projection(head_input)
            + self.head_time_projection(normalized_time)
        )
        head_tokens = self.head_transformer(
            head_tokens, src_key_padding_mask=self._safe_padding_mask(head_valid)
        )
        attended_scene, _ = self.scene_attention(
            query=head_tokens,
            key=scene_tokens,
            value=scene_tokens,
            key_padding_mask=self._safe_padding_mask(scene_valid),
            need_weights=False,
        )
        trajectory_latent = self._masked_mean(head_tokens, head_valid)
        attended_scene_latent = self._masked_mean(attended_scene, head_valid)
        # Scene-only cannot pool through head queries without leaking the head
        # trajectory.  Use a permutation-invariant point summary instead.  The
        # zero-valued links keep every branch in the autograd graph so these
        # controlled ablations also work under DDP without unused parameters.
        independent_scene_latent = self._masked_mean(scene_tokens, scene_valid)
        mode = self.config.condition_mode
        if mode == "full":
            scene_latent = attended_scene_latent
        elif mode == "head_only":
            scene_latent = attended_scene_latent * 0.0
        elif mode == "scene_only":
            trajectory_latent = trajectory_latent * 0.0
            scene_latent = independent_scene_latent + attended_scene_latent * 0.0
        else:
            trajectory_latent = trajectory_latent * 0.0
            scene_latent = attended_scene_latent * 0.0

        reserved = {
            self.config.point_key,
            self.config.object_key,
            self.config.head_pose_key,
            self.config.head_mask_key,
            self.config.head_time_key,
        }
        aux_keys = [key for key in self.in_keys if key not in reserved]
        if aux_keys:
            aux_latent = self.aux_projection(
                torch.cat([tensordict[key] for key in aux_keys], dim=-1)
            )
        else:
            aux_latent = torch.zeros_like(scene_latent)
        output = self.output(
            torch.cat([trajectory_latent, scene_latent, aux_latent], dim=-1)
        )
        tensordict[self.out_keys[0]] = output
        if log_internals:
            tensordict["ego_head_trajectory_latent"] = trajectory_latent
            if history_token is not None:
                tensordict["ego_scene_history_token"] = history_token.squeeze(1)
            tensordict["scene_memory_latent"] = scene_latent
        return tensordict


class SceneResidualPPOActor(PPOActor):
    """Keep the pretrained actor in ``mu`` and add a gated scene delta."""

    config: SceneResidualPPOActorConfig

    def __init__(self, config: SceneResidualPPOActorConfig):
        super().__init__(config)
        scene_cls = get_class(config.scene_model._target_)
        self.scene_model = scene_cls(config=config.scene_model)
        self.scene_gate = nn.Parameter(
            torch.tensor(float(config.scene_gate_init)),
            requires_grad=config.scene_gate_learnable,
        )
        self.in_keys = list(dict.fromkeys(self.in_keys + self.scene_model.in_keys))

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        tensordict = self.mu(tensordict, log_internals=log_internals)
        base_mu = tensordict[self.config.mu_key]
        tensordict = self.scene_model(tensordict, log_internals=log_internals)
        scene_delta = tensordict[self.config.scene_model.out_keys[0]]
        mu = base_mu + self.scene_gate * scene_delta
        if (
            self.config.counterfactual_loss_weight > 0.0
            or self.config.residual_preservation_loss_weight > 0.0
        ):
            tensordict["scene_action_delta_raw"] = scene_delta

        std = torch.exp(self.logstd)
        dist = distributions.Normal(mu, mu * 0 + std)
        action = dist.sample()
        tensordict["action"] = action
        tensordict["mean_action"] = mu
        tensordict["neglogp"] = -dist.log_prob(action).sum(dim=-1)
        if log_internals:
            tensordict["scene_action_delta"] = self.scene_gate * scene_delta
            tensordict["scene_gate"] = self.scene_gate.expand(mu.shape[0], 1)
        return tensordict

    def _counterfactual_scene_loss(self, tensordict: TensorDict, zero_loss):
        config = self.config
        target_key = self.scene_model.config.interaction_target_key
        batch_size = tensordict.batch_size[0]
        enabled = (
            config.counterfactual_loss_weight > 0.0
            and self.scene_model.config.condition_mode != "no_scene"
            and batch_size > 1
            and target_key in tensordict
            and self.scene_model.config.num_interaction_targets > 0
            and bool(config.counterfactual_scene_keys)
        )
        if not enabled:
            return zero_loss * 0.0, {}

        # Keep body state and future motion intent fixed. Only scene geometry is
        # replaced, otherwise the actor could satisfy this loss by responding to
        # a different future pose rather than to a different scene.
        order = torch.randperm(batch_size, device=tensordict.device)
        permutation = torch.empty_like(order)
        permutation[order] = order.roll(1)
        counterfactual = TensorDict(
            {key: tensordict[key] for key in self.scene_model.in_keys},
            batch_size=tensordict.batch_size,
        )
        for key in config.counterfactual_scene_keys:
            if key in counterfactual:
                counterfactual[key] = tensordict[key][permutation]

        counterfactual = self.scene_model(counterfactual)
        factual_delta = tensordict["scene_action_delta_raw"]
        counterfactual_delta = counterfactual[
            self.config.scene_model.out_keys[0]
        ]
        action_delta = (
            self.scene_gate * (factual_delta - counterfactual_delta)
        ).square().mean(dim=-1).add(1e-12).sqrt()

        point_key = self.scene_model.config.point_key
        query_key = self.scene_model.config.future_query_key
        points = tensordict[point_key].reshape(
            batch_size, -1, self.scene_model.config.point_feature_dim
        )
        counterfactual_points = counterfactual[point_key].reshape_as(points)
        queries = tensordict[query_key].reshape(batch_size, -1, 3)

        def interaction_from_points(point_features):
            valid = point_features[..., -1] > 0.5
            distances = torch.cdist(queries, point_features[..., :3])
            distances = distances.masked_fill(~valid.unsqueeze(1), float("inf"))
            nearest = distances.min(dim=-1).values
            nearest = torch.where(
                torch.isfinite(nearest),
                nearest,
                nearest.new_full(nearest.shape, config.counterfactual_distance_scale_m),
            )
            normalized = (
                nearest / config.counterfactual_distance_scale_m
            ).clamp(0.0, 1.0)
            contact = (
                nearest <= config.counterfactual_contact_threshold_m
            ).to(normalized.dtype)
            return torch.stack((normalized, contact), dim=-1)

        # Compute both labels through the same sampled-point approximation so
        # their difference measures only the geometry intervention.
        factual_target = interaction_from_points(points)
        counterfactual_target = interaction_from_points(counterfactual_points)
        distance_delta = (
            factual_target[..., 0] - counterfactual_target[..., 0]
        ).abs().mean(dim=-1)
        contact_changed = (
            factual_target[..., 1] != counterfactual_target[..., 1]
        ).any(dim=-1).to(distance_delta.dtype)
        interaction_delta = torch.maximum(distance_delta, contact_changed)
        active = interaction_delta >= config.counterfactual_min_interaction_delta
        margin = config.counterfactual_action_margin * interaction_delta
        per_sample_loss = torch.relu(margin - action_delta)
        if active.any():
            raw_loss = per_sample_loss[active].mean()
        else:
            raw_loss = per_sample_loss.sum() * 0.0
        weighted_loss = config.counterfactual_loss_weight * raw_loss
        return weighted_loss, {
            "counterfactual_loss": raw_loss.detach(),
            "counterfactual_weighted_loss": weighted_loss.detach(),
            "counterfactual_active_fraction": active.float().mean().detach(),
            "counterfactual_interaction_delta": interaction_delta.mean().detach(),
            "counterfactual_action_delta": action_delta.mean().detach(),
            "scene_gate_value": self.scene_gate.detach(),
        }

    def _residual_preservation_loss(self, tensordict: TensorDict, zero_loss):
        config = self.config
        target_key = self.scene_model.config.interaction_target_key
        enabled = (
            config.residual_preservation_loss_weight > 0.0
            and config.interaction_num_bodies > 0
            and bool(config.interaction_body_ids)
            and target_key in tensordict
            and "scene_action_delta_raw" in tensordict
        )
        if not enabled:
            return zero_loss * 0.0, {}

        batch_size = tensordict.batch_size[0]
        targets = tensordict[target_key].reshape(
            batch_size, -1, config.interaction_num_bodies, 2
        )
        body_ids = torch.as_tensor(
            config.interaction_body_ids,
            dtype=torch.long,
            device=targets.device,
        )
        # Feet are excluded by configuration, so this mask captures object/body
        # interactions rather than ordinary support contact with the floor.
        interaction = (
            targets[:, :, body_ids, 1] > 0.5
        ).any(dim=(1, 2)).to(targets.dtype)
        preserve = 1.0 - interaction
        gated_residual_energy = (
            self.scene_gate * tensordict["scene_action_delta_raw"]
        ).square().mean(dim=-1)
        raw_loss = (
            gated_residual_energy * preserve
        ).sum() / preserve.sum().clamp_min(1.0)
        weighted_loss = config.residual_preservation_loss_weight * raw_loss
        return weighted_loss, {
            "residual_preservation_loss": raw_loss.detach(),
            "residual_preservation_weighted_loss": weighted_loss.detach(),
            "nonfoot_interaction_fraction": interaction.mean().detach(),
            "factual_scene_residual_rms": gated_residual_energy.mean().sqrt().detach(),
        }

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss,
        log_prefix: str = "model",
    ):
        loss, log_dict = super().compute_model_loss(
            tensordict,
            current_epoch=current_epoch,
            zero_loss=zero_loss,
            log_prefix=log_prefix,
        )
        scene_loss_fn = getattr(self.scene_model, "compute_model_loss", None)
        if scene_loss_fn is None:
            return loss, log_dict
        scene_loss, scene_log_dict = scene_loss_fn(
            tensordict,
            current_epoch=current_epoch,
            zero_loss=zero_loss,
            log_prefix=f"{log_prefix}/scene",
        )
        log_dict.update(scene_log_dict)
        counterfactual_loss, counterfactual_log = self._counterfactual_scene_loss(
            tensordict, zero_loss
        )
        preservation_loss, preservation_log = self._residual_preservation_loss(
            tensordict, zero_loss
        )
        log_dict.update(
            {
                f"{log_prefix}/scene/{key}": value
                for key, value in counterfactual_log.items()
            }
        )
        log_dict.update(
            {
                f"{log_prefix}/scene/{key}": value
                for key, value in preservation_log.items()
            }
        )
        return loss + scene_loss + counterfactual_loss + preservation_loss, log_dict


class SceneResidualCritic(MLPWithConcat):
    """Retain the pretrained value MLP and add a gated privileged scene value."""

    config: SceneResidualCriticConfig

    def __init__(self, config: SceneResidualCriticConfig):
        super().__init__(config)
        scene_cls = get_class(config.scene_model._target_)
        self.scene_model = scene_cls(config=config.scene_model)
        self.scene_gate = nn.Parameter(torch.tensor(float(config.scene_gate_init)))
        self.in_keys = list(dict.fromkeys(self.in_keys + self.scene_model.in_keys))

    def forward(self, tensordict: TensorDict, log_internals: bool = False) -> TensorDict:
        tensordict = super().forward(tensordict, log_internals=log_internals)
        value_key = self.config.out_keys[0]
        base_value = tensordict[value_key]
        tensordict = self.scene_model(tensordict, log_internals=log_internals)
        scene_value = tensordict[self.config.scene_model.out_keys[0]]
        tensordict[value_key] = base_value + self.scene_gate * scene_value
        return tensordict
