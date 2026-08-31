# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-matched shuffled finite-window motion sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch

from protomotions.envs.motion_manager.config import SceneWindowMotionManagerConfig
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager


class SceneWindowMotionManager(MimicMotionManager):
    """Draw complete transition coverage plus random windows within each scene."""

    config: SceneWindowMotionManagerConfig

    def __init__(
        self,
        config: SceneWindowMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
        scene_ids_per_env: Optional[list[Optional[str]]] = None,
    ):
        if fixed_motion_ids_per_env is not None:
            raise ValueError("SceneWindowMotionManager cannot use fixed motion IDs")
        if scene_ids_per_env is None or len(scene_ids_per_env) != num_envs:
            raise ValueError("A physical scene_id is required for every environment")
        if any(scene_id is None for scene_id in scene_ids_per_env):
            raise ValueError("SceneWindowMotionManager received an unlabeled scene")
        if len(config.motion_scene_ids) != motion_lib.num_motions():
            raise ValueError(
                "motion_scene_ids must align one-to-one with MotionLib motions"
            )
        if config.window_size_frames < 1 or config.fixed_window_stride_frames < 1:
            raise ValueError("Window size and stride must be positive")
        if config.random_windows_per_clip < 0:
            raise ValueError("random_windows_per_clip cannot be negative")

        super().__init__(
            config,
            num_envs,
            env_dt,
            device,
            motion_lib,
            fixed_motion_ids_per_env=None,
        )
        self.scene_ids_per_env = [str(value) for value in scene_ids_per_env]
        self._rng = random.Random(config.sampler_seed)
        self._motions_by_scene: dict[str, list[int]] = defaultdict(list)
        for motion_id, scene_id in enumerate(config.motion_scene_ids):
            self._motions_by_scene[str(scene_id)].append(motion_id)
        missing = sorted(set(self.scene_ids_per_env) - self._motions_by_scene.keys())
        if missing:
            raise ValueError(f"No training motions for physical scenes: {missing}")

        self._queues: dict[str, list[tuple[int, int]]] = {}
        self._scene_epochs: dict[str, int] = defaultdict(int)
        self.window_end_times = torch.zeros(num_envs, device=device)

    def _motion_num_frames(self, motion_id: int) -> int:
        if hasattr(self.motion_lib, "motion_num_frames"):
            return int(self.motion_lib.motion_num_frames[motion_id])
        dt = float(self.motion_lib.motion_dt[motion_id])
        return int(round(float(self.motion_lib.motion_lengths[motion_id]) / dt)) + 1

    def _motion_dt(self, motion_id: int) -> float:
        if hasattr(self.motion_lib, "motion_dt"):
            return float(self.motion_lib.motion_dt[motion_id])
        return self.env_dt

    def _refill(self, scene_id: str) -> None:
        queue: list[tuple[int, int]] = []
        for motion_id in self._motions_by_scene[scene_id]:
            frame_count = self._motion_num_frames(motion_id)
            available_transitions = max(0, frame_count - 1)
            max_start = max(
                0, available_transitions - self.config.window_size_frames
            )
            fixed_starts = list(
                range(0, max_start + 1, self.config.fixed_window_stride_frames)
            )
            if not fixed_starts or fixed_starts[-1] != max_start:
                fixed_starts.append(max_start)
            queue.extend((motion_id, start) for start in fixed_starts)
            for _ in range(self.config.random_windows_per_clip):
                queue.append((motion_id, self._rng.randint(0, max_start)))
        self._rng.shuffle(queue)
        self._queues[scene_id] = queue
        self._scene_epochs[scene_id] += 1

    def _next_window(self, scene_id: str) -> tuple[int, int]:
        if not self._queues.get(scene_id):
            self._refill(scene_id)
        return self._queues[scene_id].pop()

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        if new_motion_ids is not None:
            raise ValueError("Explicit motion IDs bypass scene-matched window sampling")
        for env_id in env_ids.detach().cpu().tolist():
            motion_id, start_frame = self._next_window(self.scene_ids_per_env[env_id])
            motion_dt = self._motion_dt(motion_id)
            start_time = start_frame * motion_dt
            end_frame = min(
                start_frame + self.config.window_size_frames,
                self._motion_num_frames(motion_id) - 1,
            )
            self.motion_ids[env_id] = motion_id
            self.motion_times[env_id] = start_time
            self.window_end_times[env_id] = min(
                end_frame * motion_dt,
                float(self.motion_lib.motion_lengths[motion_id]),
            )

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Called after motion_times advances. End when all requested transitions
        # have run, not one step early as the full-clip manager intentionally does.
        done = self.motion_times >= self.window_end_times
        return done if env_ids is None else done[env_ids]

    def build_scene_matched_eval_batches(self):
        """Pair every motion with matching envs while filling all scenes per round."""
        envs_by_scene: dict[str, list[int]] = defaultdict(list)
        for env_id, scene_id in enumerate(self.scene_ids_per_env):
            envs_by_scene[scene_id].append(env_id)

        offsets = {scene_id: 0 for scene_id in self._motions_by_scene}
        batches = []
        while True:
            batch_env_ids = []
            batch_motion_ids = []
            for scene_id, motion_ids in self._motions_by_scene.items():
                env_ids = envs_by_scene[scene_id]
                if not env_ids:
                    raise ValueError(f"No evaluation environment for scene {scene_id}")
                start = offsets[scene_id]
                if start >= len(motion_ids):
                    continue
                selected = motion_ids[start : start + len(env_ids)]
                batch_env_ids.extend(env_ids[: len(selected)])
                batch_motion_ids.extend(selected)
                offsets[scene_id] += len(selected)
            if not batch_motion_ids:
                break
            batches.append(
                (
                    torch.tensor(
                        batch_env_ids, device=self.device, dtype=torch.long
                    ),
                    torch.tensor(
                        batch_motion_ids, device=self.device, dtype=torch.long
                    ),
                )
            )
        return batches

    def get_state_dict(self):
        state = super().get_state_dict()
        state["scene_window_sampler"] = {
            "queues": self._queues,
            "scene_epochs": dict(self._scene_epochs),
            "rng_state": self._rng.getstate(),
        }
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        sampler = state_dict.get("scene_window_sampler")
        if sampler is not None:
            self._queues = sampler["queues"]
            self._scene_epochs = defaultdict(int, sampler["scene_epochs"])
            self._rng.setstate(sampler["rng_state"])
