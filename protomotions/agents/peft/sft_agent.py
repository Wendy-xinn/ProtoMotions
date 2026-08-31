# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT agent for PEFT adapters on a frozen discrete-token GPC prior."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from lightning.fabric import Fabric
import torch
from torch.utils.data import DataLoader

from protomotions.agents.fine_tuning.pretrained_modules import PretrainedModulesMixin
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.peft.prior_setup import DiscretePriorPEFTSetupMixin
from protomotions.agents.utils.training import aggregate_scalar_metrics
from protomotions.agents.common.latent import TARGET_LATENT_KEY
from protomotions.agents.peft.offline_dataset import OfflineSFTDataset


# SFT uses SupervisedAgent directly, so it opts into the shared frozen-module
# lifecycle here. RLFT receives the same mixin through FineTuningAgent.
class DiscretePriorPEFTSFTAgent(
    DiscretePriorPEFTSetupMixin,
    PretrainedModulesMixin,
    SupervisedAgent,
):
    """Train a discrete-prior PEFT adapter with target-token supervision.

    The model owns the expert labeling path through the frozen target encoder;
    the generic supervised loop stores those labels and applies the configured
    supervised loss during optimization.
    """

    def __init__(self, fabric: Fabric, env, config, root_dir: Optional[Path] = None):
        if getattr(config.model, "critic", None) is not None:
            raise ValueError("DiscretePriorPEFTSFTAgent does not use a critic.")
        super().__init__(fabric, env, config, root_dir=root_dir)

    @property
    def has_critic(self):
        return False

    def create_model(self):
        # SupervisedAgent's external-expert slot is unused here. PEFT SFT gets
        # labels from the frozen target encoder inside DiscretePriorPEFTSFTModel.
        self.expert_model = None
        return super().create_model()

    def _should_build_target_encoder(self, mimic_target_poses_dim: int) -> bool:
        if mimic_target_poses_dim <= 0:
            raise ValueError(
                "DiscretePriorPEFTSFTAgent requires environment observations to include "
                "mimic_target_poses so the frozen target encoder can build "
                "supervision labels."
            )
        return True

    def create_optimizers(self, model):
        optimizer = instantiate_optimizer(
            self.config.model.actor_optimizer,
            model,
            params=self._actor_optimizer_params(model),
        )
        self.training_model, self.supervised_optimizer = self._setup_model_optimizer(
            model,
            optimizer,
        )

    def post_epoch_logging(self, training_log_dict):
        """Select a periodic token-accuracy best checkpoint for SFT runs.

        The generic evaluator has no score when no physical evaluation
        components are configured.  In that case ``best_evaluated_score`` used
        to remain None and no score checkpoint was ever written.
        """
        interval = getattr(self.evaluator.config, "eval_metrics_every", None)
        metric = training_log_dict.get("sft/accuracy")
        if interval and metric is not None and self.current_epoch % interval == 0:
            score = aggregate_scalar_metrics(
                {"sft/accuracy": metric}, self.fabric, weight=self.num_envs
            )["sft/accuracy"]
            score = float(score.detach().cpu() if torch.is_tensor(score) else score)
            if self.best_evaluated_score is None or score >= self.best_evaluated_score:
                self.best_evaluated_score = score
                self.save(checkpoint_name="last.ckpt", new_high_score=True)
        return super().post_epoch_logging(training_log_dict)

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        # RLFT warm-start reads the actor optimizer state from SFT checkpoints.
        state_dict["actor_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def fit(self):
        if self.config.offline_cache_output is not None:
            return self.collect_offline_cache(Path(self.config.offline_cache_output))
        if self.config.offline_dataset_path is not None:
            return self.fit_offline(Path(self.config.offline_dataset_path))
        return super().fit()

    @staticmethod
    def _cache_tensor(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        return value.to(torch.float16) if value.is_floating_point() else value

    @torch.no_grad()
    def collect_offline_cache(self, output_dir: Path):
        """Collect every local motion once under frozen expert control."""
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.eval()
        num_motions = int(self.motion_lib.num_motions())
        env_count = self.num_envs
        actor = self.model._actor
        cache_keys = list(
            dict.fromkeys(
                list(self.config.model.actor.in_keys)
                + list(actor.frozen_prior_input_keys)
                + ["mimic_target_poses"]
            )
        )
        clip_entries = []
        for batch_start in range(0, num_motions, env_count):
            active = min(env_count, num_motions - batch_start)
            local_ids = torch.arange(active, device=self.device, dtype=torch.long)
            motion_ids = torch.arange(
                batch_start, batch_start + active, device=self.device, dtype=torch.long
            )
            assigned = motion_ids[torch.arange(env_count, device=self.device) % active]
            self.motion_manager.motion_ids.copy_(assigned)
            self.motion_manager.motion_times.zero_()
            self.model.reset_rollout_context(
                num_envs=self.num_envs, device=self.device
            )
            obs, _ = self.env.reset(disable_motion_resample=True)
            obs = self.add_agent_info_to_obs(obs)
            obs_td = self.obs_dict_to_tensordict(obs)
            frame_counts = self.motion_lib.get_motion_num_frames(assigned).to(torch.long)
            max_frames = int(frame_counts[:active].max().item())
            failed = torch.zeros(env_count, dtype=torch.bool, device=self.device)
            per_env: list[dict[str, list[torch.Tensor]]] = [dict() for _ in range(active)]
            for frame in range(max_frames):
                expert_td = self.model.collect_expert_rollout(obs_td.clone())
                missing = [key for key in cache_keys if key not in expert_td]
                if missing:
                    raise KeyError(
                        f"Expert cache is missing required observation keys: {missing}"
                    )
                valid_now = (~failed) & (frame < frame_counts)
                for env_index in range(active):
                    target = per_env[env_index]
                    for key in cache_keys:
                        target.setdefault(key, []).append(
                            self._cache_tensor(expert_td[key][env_index])
                        )
                    target.setdefault(TARGET_LATENT_KEY, []).append(
                        self._cache_tensor(expert_td[TARGET_LATENT_KEY][env_index])
                    )
                    target.setdefault("action", []).append(
                        self._cache_tensor(expert_td["action"][env_index])
                    )
                    target.setdefault("motion_time", []).append(
                        self._cache_tensor(self.motion_manager.motion_times[env_index])
                    )
                    target.setdefault("valid", []).append(valid_now[env_index].cpu())
                next_obs, _, dones, terminated, _ = self.env.step(expert_td["action"])
                failed |= terminated.bool()
                for env_index in range(active):
                    per_env[env_index].setdefault("terminated", []).append(
                        terminated[env_index].detach().cpu().bool()
                    )
                next_obs = self.add_agent_info_to_next_obs(next_obs)
                obs_td = self.obs_dict_to_tensordict(next_obs)

            for env_index in range(active):
                motion_id = int(motion_ids[env_index].item())
                tensors = {
                    key: torch.stack(values) for key, values in per_env[env_index].items()
                }
                valid_frames = int(tensors["valid"].sum().item())
                file_name = f"motion_{motion_id:04d}.pt"
                metadata = {
                    "motion_id": motion_id,
                    "frames": int(tensors["valid"].shape[0]),
                    "valid_frames": valid_frames,
                    "complete": valid_frames == int(frame_counts[env_index].item()),
                }
                payload = {"format_version": 1, "metadata": metadata, "tensors": tensors}
                temporary = output_dir / f".{file_name}.tmp"
                torch.save(payload, temporary)
                os.replace(temporary, output_dir / file_name)
                clip_entries.append({"file": file_name, **metadata})
                print(
                    f"Cached motion {motion_id}: {valid_frames}/{metadata['frames']} valid frames",
                    flush=True,
                )

        manifest = {
            "format_version": 1,
            "split": self.config.offline_cache_split,
            "clips": sorted(clip_entries, key=lambda entry: entry["motion_id"]),
        }
        temporary_manifest = output_dir / ".cache_manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary_manifest, output_dir / "cache_manifest.json")
        print(f"Expert cache complete: {output_dir}", flush=True)

    def _offline_loader(self, dataset: OfflineSFTDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def _move_offline_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    @torch.no_grad()
    def _evaluate_offline(self, loader: DataLoader) -> dict[str, float]:
        self.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = self._move_offline_batch(batch)
            _, logs = self.supervised_step(batch)
            weight = int(next(iter(batch.values())).shape[0])
            count += weight
            for key, value in logs.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[key] = totals.get(key, 0.0) + float(value) * weight
        return {key: value / max(count, 1) for key, value in totals.items()}

    def fit_offline(self, dataset_root: Path):
        """Optimize the adapter from fixed expert-state transitions."""
        train_splits = {
            value.strip()
            for value in self.config.offline_dataset_split.split(",")
            if value.strip()
        }
        train_dataset = OfflineSFTDataset(dataset_root, train_splits)
        train_loader = self._offline_loader(train_dataset, shuffle=True)
        try:
            val_dataset = OfflineSFTDataset(dataset_root, {"val"})
            val_loader = self._offline_loader(val_dataset, shuffle=False)
        except (FileNotFoundError, RuntimeError):
            val_loader = None

        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        if self._fit_session_start_monotonic is None:
            self._fit_session_start_monotonic = time.monotonic()
        self.fabric.call("on_fit_start", self)
        print(
            f"Offline SFT: {len(train_dataset)} train frames, "
            f"{0 if val_loader is None else len(val_loader.dataset)} validation frames",
            flush=True,
        )
        best_score = self.best_evaluated_score
        final_epoch = int(self.config.offline_num_epochs)
        while self.current_epoch < final_epoch and not self.should_stop:
            epoch_start = time.monotonic()
            self.train()
            totals: dict[str, float] = {}
            examples = 0
            for batch_index, batch in enumerate(train_loader):
                batch = self._move_offline_batch(batch)
                weight = int(next(iter(batch.values())).shape[0])
                logs = self.perform_optimization_step(batch, batch_index)
                examples += weight
                self.step_count += weight
                for key, value in logs.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        totals[key] = totals.get(key, 0.0) + float(value.detach()) * weight
            train_logs = {
                f"offline/train/{key}": value / max(examples, 1)
                for key, value in totals.items()
            }
            val_logs = {}
            if val_loader is not None:
                val_logs = {
                    f"offline/val/{key}": value
                    for key, value in self._evaluate_offline(val_loader).items()
                }
            self.current_epoch += 1
            elapsed = time.monotonic() - epoch_start
            log_dict = {
                **train_logs,
                **val_logs,
                "offline/epoch_seconds": elapsed,
                "offline/examples": examples,
            }
            self.fabric.log_dict(log_dict, step=self.current_epoch)
            score_key = "offline/val/sft/accuracy"
            fallback_key = "offline/train/sft/accuracy"
            score = log_dict.get(score_key, log_dict.get(fallback_key))
            is_best = score is not None and (best_score is None or score >= best_score)
            if is_best:
                best_score = float(score)
                self.best_evaluated_score = best_score
                self.save(checkpoint_name="last.ckpt", new_high_score=True)
            elif self.current_epoch % self.config.save_last_checkpoint_every == 0:
                self.save(checkpoint_name="last.ckpt")
            print(
                f"Offline epoch {self.current_epoch}/{final_epoch} | "
                f"{elapsed:.2f}s | train_acc={log_dict.get(fallback_key, float('nan')):.4f} | "
                f"val_acc={log_dict.get(score_key, float('nan')):.4f}",
                flush=True,
            )
        self.save(checkpoint_name="last.ckpt")
        self.fabric.call("on_fit_end", self)
