"""TRUMANS implementation of the generic scene-motion adapter contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES

from .contracts import (
    ClipDescriptor,
    DatasetAdapter,
    HumanMotionInput,
    ObjectMotionInput,
    RotationFormat,
    SceneAssetInput,
)


class TrumansAdapter(DatasetAdapter):
    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        eligible_only: bool = True,
        bad_frame_policy: str = "drop_clip",
    ):
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.eligible_only = eligible_only
        self.bad_frame_policy = bad_frame_policy
        if bad_frame_policy not in {"drop_clip", "keep"}:
            raise ValueError("bad_frame_policy must be 'drop_clip' or 'keep'")

        self._human_pose = np.load(self.root / "human_pose.npy", mmap_mode="r")
        self._human_orient = np.load(self.root / "human_orient.npy", mmap_mode="r")
        self._human_transl = np.load(self.root / "human_transl.npy", mmap_mode="r")
        self._human_joints = np.load(self.root / "human_joints.npy", mmap_mode="r")
        self._betas = np.load(self.root / "betas.npy", mmap_mode="r")
        self._left_hand = np.load(self.root / "left_hand_pose.npy", mmap_mode="r")
        self._right_hand = np.load(self.root / "right_hand_pose.npy", mmap_mode="r")

        self._records = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._record_by_id = {record["clip_name"]: record for record in self._records}

    def iter_clips(self) -> Iterator[ClipDescriptor]:
        for record in self._records:
            if self.eligible_only and not record["eligible_scene_expert_v1"]:
                continue
            bad_frames = tuple(map(int, record["bad_local_frames"]))
            if bad_frames and self.bad_frame_policy == "drop_clip":
                continue
            yield ClipDescriptor(
                clip_id=record["clip_name"],
                num_frames=int(record["num_frames"]),
                fps=float(record["fps"]),
                scene_id=record["scene"]["name"],
                source_start=int(record["global_start"]),
                source_end_exclusive=int(record["global_end_exclusive"]),
                bad_local_frames=bad_frames,
                metadata={"manifest_record": record},
            )

    def load_human(self, clip: ClipDescriptor) -> HumanMotionInput:
        sl = slice(clip.source_start, clip.source_end_exclusive)
        result = HumanMotionInput(
            # SMPL-X `transl` is not the simulated pelvis position in this release;
            # it differs from joint 0 by up to several decimetres due to the body
            # model's shaped pelvis offset. ProtoMotions expects the free-joint/root
            # world position, so use the released 24-joint root directly.
            root_translation=np.asarray(self._human_joints[sl, 0], dtype=np.float32),
            root_orientation=np.asarray(self._human_orient[sl], dtype=np.float32),
            body_pose=np.asarray(self._human_pose[sl], dtype=np.float32),
            root_rotation_format=RotationFormat.AXIS_ANGLE,
            body_rotation_format=RotationFormat.AXIS_ANGLE,
            source_model="smplx_body_21",
            root_translation_semantics="released_joint_0_world",
            betas=np.asarray(self._betas[sl], dtype=np.float32),
            left_hand_pose=np.asarray(self._left_hand[sl], dtype=np.float32),
            right_hand_pose=np.asarray(self._right_hand[sl], dtype=np.float32),
            joint_positions=np.asarray(self._human_joints[sl], dtype=np.float32),
            joint_names=SMPL_BONE_ORDER_NAMES,
        )
        result.validate(clip)
        return result

    def load_objects(self, clip: ClipDescriptor) -> list[ObjectMotionInput]:
        record = self._record_by_id[clip.clip_id]
        relative_path = record["object_tracks"]["path"]
        if relative_path is None:
            return []
        raw_tracks = np.load(self.root / relative_path, allow_pickle=True).item()
        objects = []
        for object_id in sorted(raw_tracks):
            track = raw_tracks[object_id]
            obj = ObjectMotionInput(
                object_id=object_id,
                mesh_path=self.root / "Object_all" / "Object_mesh" / f"{object_id}.obj",
                translation=np.asarray(track["location"], dtype=np.float32),
                rotation=np.asarray(track["rotation"], dtype=np.float32),
                rotation_format=RotationFormat.EULER_XYZ,
                movable=True,
            )
            obj.validate(clip)
            objects.append(obj)
        return objects

    def load_scene(self, clip: ClipDescriptor) -> SceneAssetInput:
        record = self._record_by_id[clip.clip_id]
        scene = record["scene"]
        asset = SceneAssetInput(
            scene_id=scene["name"],
            mesh_path=self.root / scene["mesh"],
            occupancy_path=self.root / scene["occupancy"],
            occupancy_axis_order="xyz",
            occupancy_bounds_min=(-3.0, 0.0, -4.0),
            occupancy_bounds_max=(3.0, 2.0, 4.0),
        )
        if not asset.mesh_path.is_file() or not asset.occupancy_path.is_file():
            raise FileNotFoundError(f"Missing scene asset for {clip.clip_id}: {asset}")
        return asset
