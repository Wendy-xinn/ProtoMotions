"""MaskedMimic control component whose visible target is an external Head path."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import numpy as np
import torch

from protomotions.envs.control.masked_mimic_control import MaskedMimicControl


class ExternalHeadMaskedMimicControl(MaskedMimicControl):
    """Replace only the Head entries of the sparse future target.

    The stock component still supplies timing and spawn offsets.  All body masks
    are forcibly disabled except Head translation and Head rotation, preventing
    the bootstrap MotionLib from exposing GT targets for the remaining body.
    """

    def __init__(self, config, env):
        super().__init__(config, env)
        trajectory_path = os.environ.get("EGOBODY_MM_HEAD_CONDITION")
        if not trajectory_path:
            raise RuntimeError("EGOBODY_MM_HEAD_CONDITION is not set")
        data = np.load(trajectory_path)
        self.external_head_pos = torch.as_tensor(
            data["head_pos"], dtype=torch.float32, device=self.env.device
        )
        self.external_head_rot = torch.as_tensor(
            data["head_quat_xyzw"], dtype=torch.float32, device=self.env.device
        )
        self.external_fps = float(np.asarray(data["fps"]).item())
        self.head_body_id = self._all_body_names.index("Head")
        self.head_conditionable_id = self.conditionable_body_ids.tolist().index(self.head_body_id)
        self.output_path = os.environ.get("EGOBODY_MM_OUTPUT")
        self.stop_after_steps = int(
            os.environ.get("EGOBODY_MM_STOP_AFTER_STEPS", len(self.external_head_pos))
        )
        self._step_count = 0
        self._last_recorded_time = None
        self._recording = {
            "motion_time": [],
            "rigid_body_pos": [],
            "rigid_body_rot": [],
            "rigid_body_vel": [],
            "rigid_body_ang_vel": [],
            "dof_pos": [],
            "dof_vel": [],
        }
        atexit.register(self._save_recording)

    def _force_head_only_masks(self) -> None:
        masks = self.masked_mimic_target_bodies_masks.view(
            self.env.num_envs, self.config.num_masked_future_steps, self.num_conditionable_bodies, 2
        )
        masks.zero_()
        masks[:, :, self.head_conditionable_id, 0] = True
        masks[:, :, self.head_conditionable_id, 1] = True
        self.masked_mimic_target_poses_masks.fill_(True)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._force_head_only_masks()

    def step(self):
        super().step()
        self._force_head_only_masks()
        self._step_count += 1
        if self._step_count >= self.stop_after_steps:
            self._save_recording()
            raise KeyboardInterrupt

    def _sample_external(self, times: torch.Tensor):
        frame = torch.clamp(times * self.external_fps, 0, len(self.external_head_pos) - 1)
        lower = torch.floor(frame).long()
        upper = torch.clamp(lower + 1, max=len(self.external_head_pos) - 1)
        alpha = (frame - lower.float()).unsqueeze(-1)
        pos = torch.lerp(self.external_head_pos[lower], self.external_head_pos[upper], alpha)

        q0 = self.external_head_rot[lower]
        q1 = self.external_head_rot[upper]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        rot = torch.nn.functional.normalize(torch.lerp(q0, q1, alpha), dim=-1)
        return pos, rot

    def _record_current(self, ctx) -> None:
        if self.output_path is None:
            return
        current_time = float(self.env.motion_manager.motion_times[0].item())
        if self._last_recorded_time is not None and abs(current_time - self._last_recorded_time) < 1e-8:
            return
        self._last_recorded_time = current_time
        self._recording["motion_time"].append(current_time)
        for key in self._recording:
            if key == "motion_time":
                continue
            value = getattr(ctx.current, key, None)
            if value is not None:
                self._recording[key].append(value[0].detach().cpu().numpy())

    def populate_context(self, ctx) -> None:
        super().populate_context(ctx)
        self._force_head_only_masks()
        self._record_current(ctx)

        env_ids = getattr(ctx, "env_ids", None)
        if env_ids is None:
            motion_ids = self.env.motion_manager.motion_ids
            target_times = self.target_times
        else:
            motion_ids = self.env.motion_manager.motion_ids[env_ids]
            target_times = self.target_times[env_ids]
        num_envs, num_steps = target_times.shape

        flat_motion_ids = motion_ids[:, None].expand(num_envs, num_steps).reshape(-1)
        flat_times = target_times.reshape(-1)
        raw_state = self.env.motion_lib.get_motion_state(flat_motion_ids, flat_times)
        raw_head = raw_state.rigid_body_pos[:, self.head_body_id].view(num_envs, num_steps, 3)
        # The stock context already contains the environment/terrain spawn offset.
        spawn_offset = ctx.masked_mimic.ref_pos[:, :, self.head_body_id] - raw_head

        external_pos, external_rot = self._sample_external(target_times)
        ctx.masked_mimic.ref_pos[:, :, self.head_body_id] = external_pos + spawn_offset
        ctx.masked_mimic.ref_rot[:, :, self.head_body_id] = external_rot
        ctx.masked_mimic.target_bodies_masks = self.masked_mimic_target_bodies_masks[
            env_ids if env_ids is not None else slice(None)
        ]
        ctx.masked_mimic.target_poses_masks = self.masked_mimic_target_poses_masks[
            env_ids if env_ids is not None else slice(None)
        ]

    def _save_recording(self) -> None:
        if self.output_path is None or not self._recording["motion_time"]:
            return
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, values in self._recording.items():
            if values:
                arrays[key] = np.asarray(values, dtype=np.float32)
        arrays["fps"] = np.asarray(self.external_fps, dtype=np.float32)
        arrays["head_condition_path"] = np.asarray(os.environ["EGOBODY_MM_HEAD_CONDITION"])
        np.savez_compressed(output, **arrays)
        self.output_path = None
        print(f"Saved MaskedMimic rollout to {output}")
