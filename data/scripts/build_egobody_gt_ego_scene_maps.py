#!/usr/bin/env python3
"""Build frame-aligned causal GT ego scene maps and window-sampling scenes."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from protomotions.components.scene_lib import ReplicationMethod, SceneLib, SceneLibConfig
from protomotions.envs.component_factories import load_ego_camera_trajectory_params
from protomotions.envs.obs.ego_visible_scene_pointcloud import (
    _EGO_SCENE_MEMORY,
    compute_ego_visible_scene_pointcloud_obs,
)


def _resolve(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = manifest_path.parent / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def _save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _set_window_scene_metadata(scene, scene_id: str) -> None:
    if isinstance(scene, dict):
        scene["humanoid_motion_id"] = -1
        scene["scene_id"] = scene_id
    else:
        scene.humanoid_motion_id = -1
        scene.scene_id = scene_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--candidate-points", type=int, default=8192)
    parser.add_argument("--output-points", type=int, default=256)
    parser.add_argument("--pointcloud-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.candidate_points < args.output_points:
        raise ValueError("--candidate-points must be >= --output-points")

    source_manifest_path = args.pack_root / args.split / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    motion_path = _resolve(source_manifest["motion_file"], source_manifest_path)
    scene_path = _resolve(source_manifest["scene_file"], source_manifest_path)
    camera_path = _resolve(source_manifest["ego_camera_file"], source_manifest_path)
    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    scene_payload = torch.load(scene_path, map_location="cpu", weights_only=False)
    clip_map = source_manifest["clip_map"]
    count = len(clip_map)
    if count != int(motion["motion_num_frames"].numel()):
        raise ValueError("MotionLib and clip_map are not aligned")
    if not torch.all(motion["motion_num_frames"] == motion["motion_num_frames"][0]):
        raise ValueError("GT ego map builder currently requires equal clip lengths")

    source_scenes = scene_payload["original_scenes"]
    unique_scenes: dict[str, object] = {}
    motion_scene_ids = []
    for scene, clip in zip(source_scenes, clip_map):
        scene_id = str(clip["scene"])
        motion_scene_ids.append(scene_id)
        if scene_id not in unique_scenes:
            pooled_scene = copy.deepcopy(scene)
            _set_window_scene_metadata(pooled_scene, scene_id)
            unique_scenes[scene_id] = pooled_scene
    window_scenes = list(unique_scenes.values())
    scene_index = {scene_id: index for index, scene_id in enumerate(unique_scenes)}
    clip_scene_indices = torch.tensor(
        [scene_index[value] for value in motion_scene_ids], dtype=torch.long
    )

    split_root = args.output_root / args.split
    window_scene_path = split_root / "scene_lib_window_training_isaaclab.pt"
    window_scene_payload = copy.deepcopy(scene_payload)
    window_scene_payload["original_scenes"] = window_scenes
    window_scene_payload["num_original_scenes"] = len(window_scenes)
    window_scene_payload["scene_ids"] = list(unique_scenes)
    _save(window_scene_payload, window_scene_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=str(window_scene_path),
            asset_root=str(args.asset_root),
            replicate_method=ReplicationMethod.SEQUENTIAL,
            pointcloud_samples_per_object=args.candidate_points,
            pointcloud_max_workers=args.pointcloud_workers,
            pointcloud_sampling_seed=args.seed,
        ),
        num_envs=len(window_scenes),
        device=device,
        terrain=None,
    )
    camera_payload = torch.load(camera_path, map_location="cpu", weights_only=False)
    cameras = load_ego_camera_trajectory_params(str(camera_path), args.fps)
    cameras = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cameras.items()
    }
    motion_ids = torch.arange(count, device=device, dtype=torch.long)
    scene_ids = clip_scene_indices.to(device)
    frame_count = int(motion["motion_num_frames"][0])
    starts = motion["length_starts"].long()
    motion_dt = motion["motion_dt"].to(device)
    features = torch.empty(
        count,
        frame_count,
        args.output_points,
        10,
        dtype=torch.float16,
        device="cpu",
    )
    neutral_points = scene_lib.get_scene_neutral_pointcloud(scene_ids)
    neutral_normals = scene_lib.get_scene_neutral_pointcloud_normals(scene_ids)
    object_valid = scene_lib.get_per_object_valid_mask(scene_ids)
    object_static = scene_lib.get_object_static_mask(scene_ids)
    _EGO_SCENE_MEMORY.clear()

    for frame_id in range(frame_count):
        flat_ids = starts + frame_id
        reference_pos = motion["gts"][flat_ids].to(device)
        reference_rot = motion["grs"][flat_ids].to(device)
        motion_times = motion_dt * frame_id
        object_state = scene_lib.get_scene_pose(scene_ids, motion_times)
        frame_features = compute_ego_visible_scene_pointcloud_obs(
            reference_body_pos=reference_pos,
            reference_body_rot=reference_rot,
            object_pos=object_state.root_pos,
            object_rot=object_state.root_rot,
            neutral_pointclouds=neutral_points,
            neutral_pointcloud_normals=neutral_normals,
            object_valid_mask=object_valid,
            object_static_mask=object_static,
            head_body_id=0,
            num_samples=args.output_points,
            near_m=0.05,
            far_m=6.0,
            accumulate_history=True,
            include_history_metadata=True,
            history_age_scale_steps=float(frame_count),
            progress_buf=torch.full(
                (count,), frame_id, dtype=torch.long, device=device
            ),
            motion_ids=motion_ids,
            motion_times=motion_times,
            camera_world_from=cameras["camera_world_from"],
            camera_tan_h=cameras["camera_tan_h"],
            camera_tan_v=cameras["camera_tan_v"],
            camera_tan_left=cameras["camera_tan_left"],
            camera_tan_right=cameras["camera_tan_right"],
            camera_tan_top=cameras["camera_tan_top"],
            camera_tan_bottom=cameras["camera_tan_bottom"],
            camera_num_frames=cameras["camera_num_frames"],
            camera_reference_root=cameras["camera_reference_root"],
            camera_fps=args.fps,
        ).reshape(count, args.output_points, 10)
        features[:, frame_id].copy_(frame_features.to("cpu", dtype=torch.float16))
        if (frame_id + 1) % 16 == 0 or frame_id + 1 == frame_count:
            print(f"GT ego scene maps: {frame_id + 1}/{frame_count} frames")

    scene_map_path = split_root / "gt_ego_scene_maps.pt"
    valid_mask = features[..., -1] > 0.5
    valid_counts = valid_mask.sum(dim=-1)
    empty_frames = int((valid_counts == 0).sum())
    _save(
        {
            "format_version": 1,
            "features": features,
            "num_frames": motion["motion_num_frames"].long(),
            "fps": args.fps,
            "feature_dim": 10,
            "num_points": args.output_points,
            "candidate_points": args.candidate_points,
            "motion_scene_ids": motion_scene_ids,
            "causal": True,
            "coordinate_system": "gt_ego_camera",
            "camera_pose_convention": "EgoBody_PV_OpenGL_right_X_up_Y_forward_neg_Z",
            "feature_coordinate_system": "SOMA_camera_right_X_forward_neg_Y_up_Z",
            "camera_alignment": camera_payload.get("camera_alignment"),
            "validity_summary": {
                "empty_frames": empty_frames,
                "total_frames": int(valid_counts.numel()),
                "minimum_valid_points": int(valid_counts.min()),
                "mean_valid_points": float(valid_counts.float().mean()),
            },
        },
        scene_map_path,
    )
    output_manifest = copy.deepcopy(source_manifest)
    output_manifest.update(
        scene_file=str(window_scene_path.resolve()),
        ego_scene_map_file=str(scene_map_path.resolve()),
        motion_scene_ids=motion_scene_ids,
        sampling={
            "window_size_frames": 32,
            "fixed_window_stride_frames": 32,
            "random_windows_per_clip": 1,
        },
    )
    (split_root / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote window pack: {split_root}")
    print(
        f"Scene validity: empty={empty_frames}/{valid_counts.numel()} frames, "
        f"mean={float(valid_counts.float().mean()):.1f}/{args.output_points} points"
    )


if __name__ == "__main__":
    main()
