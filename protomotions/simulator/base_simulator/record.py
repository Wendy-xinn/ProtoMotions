# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording and rendering logic extracted from the Simulator base class.

This module contains the RecordingMixin class which provides video recording,
frame capture, marker management, and motion/object serialization functionality.
The mixin is designed to be used with the Simulator class, accessing simulator
state and methods via self.
"""

from collections import deque
from datetime import datetime
import logging
import os
from typing import Dict, Optional

import torch

from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.simulator.base_simulator.utils import build_motion_data

log = logging.getLogger(__name__)


class RecordingMixin:
    """Mixin providing recording and rendering capabilities for simulators.

    This mixin expects the following attributes/methods to be provided by the
    host class (Simulator):
        - self.headless, self.config, self.scene_lib, self.num_envs
        - self._num_dof, self._proj_config
        - self._original_marker_configs
        - self.get_robot_state(), self.get_object_root_state()
        - self._get_projectile_positions_rotations()
        - self._write_viewport_to_file(file_name)
        - self._update_simulator_markers(markers_state)
    """

    # -------------------------
    # Initialization
    # -------------------------

    def _init_recording_state(self) -> None:
        """Initialize all recording-related attributes."""
        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        self._show_markers: bool = True

        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )

        # Last markers state for recording (set each step)
        self._last_markers_state: Optional[Dict[str, MarkerState]] = None

    # -------------------------
    # Recording state control
    # -------------------------

    def _toggle_video_record(self):
        self._user_is_recording = not self._user_is_recording
        self._user_recording_state_change = True

    def _cancel_video_record(self):
        self._user_is_recording = False
        self._user_recording_state_change = False
        self._delete_user_viewer_recordings = True

    # -------------------------
    # Camera target
    # -------------------------

    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.

        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib.num_objects_per_scene > 0:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (
                self.scene_lib.num_objects_per_scene + 1
            )
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (
                self._camera_target["env"] + 1
            ) % self.num_envs
            print("Updated camera target to env", self._camera_target["env"])

    # -------------------------
    # Marker management
    # -------------------------

    def _get_recording_scene_offset(self) -> torch.Tensor:
        """Return the simulator-world offset of the recorded scene.

        Scene environments are placed in a terrain playground away from the
        local asset origin.  Deriving the complete XYZ offset from a fixed
        object's live pose also captures terrain-height correction, which is
        not represented by SceneLib's XY-only ``scene_offsets`` property.
        """
        zero = torch.zeros(3, dtype=torch.float32)
        if (
            self.scene_lib is None
            or self.scene_lib.num_scenes() == 0
            or not self._recorded_objects
        ):
            return zero

        eid = self._recording_env_id
        scene_idx = self.scene_lib._scene_to_original_scene_id[eid].item()
        scene = self.scene_lib._original_scenes[scene_idx]
        live_positions = self._recorded_objects[0][0]
        for obj_idx, obj in enumerate(scene.objects):
            if obj.options.fix_base_link:
                local_position = torch.as_tensor(
                    obj.translation[0], dtype=live_positions.dtype
                )
                return (live_positions[obj_idx].cpu() - local_position.cpu()).clone()
        return zero

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        print(f"Markers are now {'visible' if self._show_markers else 'hidden'}")

    def _update_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """

        if not markers_state or len(markers_state) == 0:
            return

        if not self.config.w_last:
            for key in markers_state.keys():
                markers_state[key].orientation = rotations.xyzw_to_wxyz(
                    markers_state[key].orientation
                )
        # Headless recordings may contain data-only marker streams (notably
        # synchronized reference poses) that intentionally have no simulator
        # visualization object. Keep them in ``_last_markers_state`` for the
        # recorder, but only send instantiated markers to the renderer.
        render_markers_state = {
            key: value
            for key, value in markers_state.items()
            if key in self._original_marker_configs
        }
        if not self._show_markers:
            for key in render_markers_state.keys():
                # Throw it out of view
                render_markers_state[key].translation = (
                    torch.zeros_like(render_markers_state[key].translation) - 1000000
                )
        if render_markers_state:
            self._update_simulator_markers(render_markers_state)

    def _build_markers_save_data(
        self, scene_offset: Optional[torch.Tensor] = None
    ) -> dict:
        """Build markers data dictionary for saving to .markers.pt file."""
        markers_data = {"fps": 30, "markers": {}}
        for name, frame_list in self._recorded_markers.items():
            translations = torch.stack([f[0] for f in frame_list], dim=0)
            if scene_offset is not None:
                translations = translations - scene_offset
            orientations = torch.stack([f[1] for f in frame_list], dim=0)
            # Get marker config metadata from the original (pre-simulator)
            # configs, since simulator-specific init may wrap/replace them
            marker_config = self._original_marker_configs.get(name)
            marker_type = "sphere"
            color = (1.0, 0.0, 0.0)
            sizes = []
            if marker_config is not None:
                marker_type = marker_config.type
                color = marker_config.color
                sizes = [m.size for m in marker_config.markers]

            markers_data["markers"][name] = {
                "type": marker_type,
                "color": color,
                "sizes": sizes,
                "translation": translations,
                "orientation": orientations,
            }
        return markers_data

    # -------------------------
    # Object serialization
    # -------------------------

    def _build_terrain_save_data(self) -> Optional[dict]:
        """Build terrain data dictionary for saving to .terrain.pt file.

        A flat terrain is saved as an all-zero height field as well.  Keeping
        the sidecar for both flat and non-flat runs makes visualisation and
        dataset comparisons reproducible.
        """
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None

        return {
            "height_field_raw": terrain.height_field_raw,
            "horizontal_scale": terrain.horizontal_scale,
            "vertical_scale": terrain.vertical_scale,
            "slope_threshold": terrain.config.slope_threshold,
        }

    def _build_objects_save_data(
        self, scene_offset: Optional[torch.Tensor] = None
    ) -> dict:
        """Build objects data dictionary for saving to .objects.pt file."""
        objects_list = []

        # Scene objects
        if self._recorded_objects:
            from protomotions.components.scene_lib import (
                BoxSceneObject,
                SphereSceneObject,
                CylinderSceneObject,
                MeshSceneObject,
            )

            translations = torch.stack([f[0] for f in self._recorded_objects], dim=0)
            if scene_offset is not None:
                translations = translations - scene_offset
            rotations = torch.stack([f[1] for f in self._recorded_objects], dim=0)

            eid = self._recording_env_id
            scene_idx = self.scene_lib._scene_to_original_scene_id[eid].item()
            scene = self.scene_lib._original_scenes[scene_idx]

            for obj_idx, obj in enumerate(scene.objects):
                obj_info = {
                    "name": f"object_{obj_idx}",
                    "translation": translations[:, obj_idx, :],
                    "rotation": rotations[:, obj_idx, :],
                }
                if isinstance(obj, BoxSceneObject):
                    obj_info["shape"] = "box"
                    obj_info["size"] = [obj.width, obj.depth, obj.height]
                elif isinstance(obj, SphereSceneObject):
                    obj_info["shape"] = "sphere"
                    obj_info["size"] = [obj.radius]
                elif isinstance(obj, CylinderSceneObject):
                    obj_info["shape"] = "cylinder"
                    obj_info["size"] = [obj.radius, obj.height]
                elif isinstance(obj, MeshSceneObject):
                    obj_info["shape"] = "mesh"
                    obj_info["size"] = []
                    obj_info["mesh_path"] = obj.object_path
                    obj_info["scale"] = list(obj.scale)
                else:
                    obj_info["shape"] = "box"
                    dims = obj.object_dims
                    if dims is not None:
                        obj_info["size"] = [
                            dims[1] - dims[0],
                            dims[3] - dims[2],
                            dims[5] - dims[4],
                        ]
                    else:
                        obj_info["size"] = [0.1, 0.1, 0.1]

                objects_list.append(obj_info)

        # Projectiles
        if self._recorded_projectiles:
            proj_pos = torch.stack(
                [f[0] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 3]
            if scene_offset is not None:
                proj_pos = proj_pos - scene_offset
            proj_rot = torch.stack(
                [f[1] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 4]

            half_sizes = self._proj_config.get_sizes()
            hide_z = self._proj_config.hide_z

            for p in range(proj_pos.shape[1]):
                # Only include projectiles that were visible at some point
                if (proj_pos[:, p, 2] > hide_z + 0.5).any():
                    hs = half_sizes[p]
                    full_size = hs * 2
                    objects_list.append(
                        {
                            "name": f"projectile_{p}",
                            "shape": "box",
                            "size": [full_size, full_size, full_size],
                            "translation": proj_pos[:, p, :],
                            "rotation": proj_rot[:, p, :],
                        }
                    )

        return {"fps": 30, "objects": objects_list}

    # -------------------------
    # Main render loop
    # -------------------------

    def render(self):
        """
        Render the current simulation state and handle video recording if enabled.

        This method manages:
        1. Video recording state transitions and initialization
        2. Frame capture and saving during recording
        3. Video compilation when recording ends
        4. Cleanup of temporary image files
        """
        # Motion/state recording is independent of viewport rendering.  In
        # headless runs the same state-change machinery records and serializes
        # motion/GT/object sidecars, while PNG capture is skipped entirely.
        # This avoids constructing a Kit/RTX window merely to export data for
        # an external viewer such as Viser.
        if (
            not self.headless
            or self._user_is_recording
            or self._user_recording_state_change
            or self._delete_user_viewer_recordings
        ):
            # Handle recording state transitions
            if self._user_recording_state_change:
                if self._user_is_recording:
                    # Initialize new recording
                    self._user_recording_video_queue = deque(
                        maxlen=self._user_recording_video_queue_size
                    )
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self._curr_user_recording_name = (
                        self._user_recording_video_path % curr_date_time
                    )
                    self._user_recording_frame = 0

                    self._recorded_motion = {
                        "gts": [],  # rigid_body_pos (global translations)
                        "grs": [],  # rigid_body_rot (global rotations)
                        "gvs": [],  # rigid_body_vel (global velocities)
                        "gavs": [],  # rigid_body_ang_vel (global angular velocities)
                        "dps": [],  # dof_pos
                        "dvs": [],  # dof_vel
                        "contacts": [],  # rigid_body_contacts
                    }
                    self._recorded_markers = {}
                    self._recorded_objects = []
                    self._recorded_projectiles = []
                    self._recording_env_id = self._camera_target["env"]

                    if not os.path.exists(self._curr_user_recording_name):
                        os.makedirs(self._curr_user_recording_name)
                    print(
                        f"Started recording to folder {self._curr_user_recording_name}"
                    )
                else:
                    # Finalize recording and create video
                    image_dir = self._curr_user_recording_name
                    images = sorted(
                        [
                            os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.endswith(".png")
                        ]
                    )

                    if self.headless:
                        print(
                            "Headless recording: skipped PNG/MP4 rendering; "
                            "saving motion data only."
                        )
                    elif not images:
                        # A viewer can fail to create a GLFW window (for
                        # example when DISPLAY/WSLg is missing). Do not call
                        # MoviePy with an empty sequence. Report the actual
                        # rendering problem and leave the run debuggable.
                        print(
                            "Warning: video recording produced no PNG frames; "
                            "check DISPLAY/WSLg and the IsaacLab GLFW logs."
                        )
                    else:
                        try:
                            from moviepy import ImageSequenceClip
                        except ImportError:
                            # MoviePy 1.x exposes clips through ``moviepy.editor``.
                            from moviepy.editor import ImageSequenceClip

                        clip = ImageSequenceClip(images, fps=30)
                        clip.write_videofile(
                            f"{self._curr_user_recording_name}.mp4",
                            codec="libx264",
                            audio=False,
                            threads=32,
                            preset="veryfast",
                            ffmpeg_params=[
                                "-profile:v",
                                "main",
                                "-level",
                                "4.0",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                "-crf",
                                "23",
                                "-x264-params",
                                "keyint=60:min-keyint=30",
                            ],
                        )
                    self._delete_user_viewer_recordings = True
                    if images:
                        print(f"Video saved to {self._curr_user_recording_name}.mp4")

                    # Save the recorded motion as a .motion file
                    motion_data = build_motion_data(
                        self._recorded_motion,
                        fps=30,  # Video recording FPS
                        num_dof=self._num_dof,
                    )
                    scene_offset = self._get_recording_scene_offset().to(
                        motion_data["rigid_body_pos"]
                    )
                    if torch.any(scene_offset != 0):
                        motion_data["rigid_body_pos"] -= scene_offset
                        motion_data["coordinate_frame"] = "scene_local"
                        motion_data["recording_world_offset"] = scene_offset.cpu()
                    motion_file_path = f"{self._curr_user_recording_name}.motion"
                    torch.save(motion_data, motion_file_path)
                    print(f"Motion saved to {motion_file_path}")
                    self._recorded_motion = None

                    # A BaseEnv can expose the current reference pose as a
                    # reserved marker stream.  Save it as an ordinary motion
                    # file so Viser can overlay policy and synchronized GT.
                    reference_frames = self._recorded_markers.get(
                        "recording_reference_pose"
                    )
                    if reference_frames:
                        reference_pos = torch.stack(
                            [frame[0] for frame in reference_frames], dim=0
                        )
                        reference_rot = torch.stack(
                            [frame[1] for frame in reference_frames], dim=0
                        )
                        if not self.config.w_last:
                            # _update_markers converts xyzw to the backend's
                            # wxyz convention in-place before recording.
                            reference_rot = rotations.wxyz_to_xyzw(reference_rot)
                        reference_pos = reference_pos - scene_offset.cpu()
                        gt_motion_data = {
                            "fps": 30,
                            "rigid_body_pos": reference_pos,
                            "rigid_body_rot": reference_rot,
                            "coordinate_frame": "scene_local",
                            "recording_world_offset": scene_offset.cpu(),
                        }
                        gt_motion_path = (
                            f"{self._curr_user_recording_name}.gt.motion"
                        )
                        torch.save(gt_motion_data, gt_motion_path)
                        print(
                            "Synchronized reference motion saved to "
                            f"{gt_motion_path}"
                        )

                    # Save markers and objects files
                    try:
                        save_sidecars = getattr(
                            self.config, "save_recording_sidecars", True
                        )
                        if save_sidecars and self._recorded_markers:
                            markers_data = self._build_markers_save_data(scene_offset)
                            markers_data["coordinate_frame"] = "scene_local"
                            markers_data["recording_world_offset"] = scene_offset.cpu()
                            markers_path = (
                                f"{self._curr_user_recording_name}.markers.pt"
                            )
                            torch.save(markers_data, markers_path)
                            print(f"Markers saved to {markers_path}")

                        if save_sidecars and (
                            self._recorded_objects or self._recorded_projectiles
                        ):
                            objects_data = self._build_objects_save_data(scene_offset)
                            objects_data["coordinate_frame"] = "scene_local"
                            objects_data["recording_world_offset"] = scene_offset.cpu()
                            objects_path = (
                                f"{self._curr_user_recording_name}.objects.pt"
                            )
                            torch.save(objects_data, objects_path)
                            print(f"Objects saved to {objects_path}")
                        terrain_data = (
                            self._build_terrain_save_data()
                            if save_sidecars
                            else None
                        )
                        if terrain_data is not None:
                            terrain_path = (
                                f"{self._curr_user_recording_name}.terrain.pt"
                            )
                            torch.save(terrain_data, terrain_path)
                            print(f"Terrain saved to {terrain_path}")

                    except Exception as e:
                        print(f"Warning: failed to save markers/objects/terrain: {e}")
                    self._recorded_markers = None
                    self._recorded_objects = None
                    self._recorded_projectiles = None

                self._user_recording_state_change = False

            # Capture frame if recording
            if self._user_is_recording:
                if not self.headless:
                    file_name = (
                        self._curr_user_recording_name
                        + "/%04d.png" % self._user_recording_frame
                    )
                    self._write_viewport_to_file(file_name)
                self._user_recording_frame += 1

                eid = self._recording_env_id

                # Record motion (single env only)
                robot_state = self.get_robot_state()
                self._recorded_motion["gts"].append(
                    robot_state.rigid_body_pos[eid].cpu().clone()
                )
                self._recorded_motion["grs"].append(
                    robot_state.rigid_body_rot[eid].cpu().clone()
                )
                if robot_state.rigid_body_vel is not None:
                    self._recorded_motion["gvs"].append(
                        robot_state.rigid_body_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_ang_vel is not None:
                    self._recorded_motion["gavs"].append(
                        robot_state.rigid_body_ang_vel[eid].cpu().clone()
                    )
                if robot_state.dof_pos is not None:
                    self._recorded_motion["dps"].append(
                        robot_state.dof_pos[eid].cpu().clone()
                    )
                if robot_state.dof_vel is not None:
                    self._recorded_motion["dvs"].append(
                        robot_state.dof_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_contacts is not None:
                    self._recorded_motion["contacts"].append(
                        robot_state.rigid_body_contacts[eid].cpu().clone()
                    )

                # Record markers (single env only, skip terrain markers)
                save_sidecars = getattr(self.config, "save_recording_sidecars", True)
                if self._last_markers_state:
                    for name, ms in self._last_markers_state.items():
                        if name == "terrain_markers":
                            continue
                        if not save_sidecars and name != "recording_reference_pose":
                            continue
                        if name not in self._recorded_markers:
                            self._recorded_markers[name] = []
                        self._recorded_markers[name].append(
                            (
                                ms.translation[eid].cpu().clone(),
                                ms.orientation[eid].cpu().clone(),
                            )
                        )

                # Record objects (single env only)
                if (
                    self.scene_lib is not None
                    and self.scene_lib.num_objects_per_scene > 0
                    and (save_sidecars or not self._recorded_objects)
                ):
                    obj_state = self.get_object_root_state()
                    self._recorded_objects.append(
                        (
                            obj_state.root_pos[eid].cpu().clone(),
                            obj_state.root_rot[eid].cpu().clone(),
                        )
                    )

                # Record projectiles (single env only)
                if (
                    save_sidecars
                    and
                    self._proj_config is not None
                    and self._proj_config.num_projectiles > 0
                ):
                    pos, rot = self._get_projectile_positions_rotations()
                    self._recorded_projectiles.append(
                        (
                            pos[eid].cpu().clone(),
                            rot[eid].cpu().clone(),
                        )
                    )

            # Clean up temporary files if needed
            if self._delete_user_viewer_recordings:
                images = [
                    img
                    for img in os.listdir(self._curr_user_recording_name)
                    if img.endswith(".png")
                ]
                for image in images:
                    os.remove(os.path.join(self._curr_user_recording_name, image))
                os.removedirs(self._curr_user_recording_name)
                self._delete_user_viewer_recordings = False
                self._recorded_motion = None
