# SPDX-License-Identifier: Apache-2.0

"""Kinematic reference trajectories for dynamic scene primitives."""

from dataclasses import dataclass

import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig


@dataclass
class SceneObjectReferenceControlConfig(ControlComponentConfig):
    _target_: str = (
        "protomotions.envs.control.scene_object_reference_control."
        "SceneObjectReferenceControl"
    )


class SceneObjectReferenceControl(ControlComponent):
    """Replay dynamic object roots while the PPO humanoid remains physical."""

    config: SceneObjectReferenceControlConfig

    def reset(self, env_ids):
        del env_ids

    def step(self):
        if self.env.scene_lib.num_objects_per_scene == 0:
            return
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        times = self.env.motion_manager.motion_times
        state = self.env.scene_lib.get_scene_pose(
            env_ids, times, self.env.config.ref_object_respawn_offset
        )
        state.root_pos += self.env.respawn_root_offset.unsqueeze(1)
        dt = float(self.env.dt)
        previous = self.env.scene_lib.get_scene_pose(
            env_ids,
            torch.clamp_min(times - dt, 0.0),
            self.env.config.ref_object_respawn_offset,
        )
        previous.root_pos += self.env.respawn_root_offset.unsqueeze(1)
        state.root_vel = (state.root_pos - previous.root_pos) / max(dt, 1.0e-6)
        state.root_ang_vel = torch.zeros_like(state.root_vel)
        dynamic_mask = self.env.scene_lib.get_per_object_valid_mask(env_ids)
        dynamic_mask &= ~self.env.scene_lib.get_object_static_mask(env_ids)
        self.env.simulator.set_object_root_state(state, env_ids, dynamic_mask)

    def populate_context(self, ctx: EnvContext) -> None:
        del ctx
