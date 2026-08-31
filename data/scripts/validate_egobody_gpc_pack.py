#!/usr/bin/env python3
"""Validate a motion-paired EgoBody pack before SOMA23 GPC training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_egobody_online_sft_packs import _validate_soma23_motion


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    candidate = manifest_path.parent / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(path_value)


def _motion_id(scene) -> int:
    return int(scene["humanoid_motion_id"] if isinstance(scene, dict) else scene.humanoid_motion_id)


def _validate_checkpoint_config(path: Path, label: str) -> None:
    config = _load_json(path)
    if config.get("robot_name") != "soma23":
        raise ValueError(f"{label} is not SOMA23: {path}")


def _validate_split(manifest_path: Path, expected_frames: int | None) -> dict:
    manifest = _load_json(manifest_path)
    motion_path = _resolve(manifest["motion_file"], manifest_path)
    scene_path = _resolve(manifest["scene_file"], manifest_path)
    camera_path = _resolve(manifest["ego_camera_file"], manifest_path)
    motion = torch.load(motion_path, map_location="cpu", weights_only=False)
    scenes = torch.load(scene_path, map_location="cpu", weights_only=False)
    cameras = torch.load(camera_path, map_location="cpu", weights_only=False)
    _validate_soma23_motion(motion, motion_path)

    counts = motion["motion_num_frames"].long()
    num_motions = counts.numel()
    if expected_frames is not None and not torch.all(counts == expected_frames):
        raise ValueError(f"{manifest_path}: not every clip has {expected_frames} frames")
    if manifest["num_motions"] != num_motions or manifest["num_frames"] != int(counts.sum()):
        raise ValueError(f"Manifest counts disagree with MotionLib in {manifest_path}")
    original_scenes = scenes["original_scenes"]
    camera_motions = cameras["motions"]
    clip_map = manifest["clip_map"]
    if not (len(original_scenes) == len(camera_motions) == len(clip_map) == num_motions):
        raise ValueError(f"Motion/scene/camera/clip alignment mismatch in {manifest_path}")

    observed_frames = 0
    for motion_id, (count_tensor, scene, camera, clip) in enumerate(
        zip(counts, original_scenes, camera_motions, clip_map)
    ):
        count = int(count_tensor)
        if _motion_id(scene) != motion_id or int(clip["motion_id"]) != motion_id:
            raise ValueError(f"Non-sequential scene mapping at motion {motion_id}")
        frame_ids = torch.as_tensor(camera["frame_ids"])
        world_from_camera = torch.as_tensor(camera["world_from_camera"])
        observed = torch.as_tensor(camera["observed"], dtype=torch.bool)
        if frame_ids.numel() != count or world_from_camera.shape != (count, 4, 4):
            raise ValueError(f"Camera length mismatch at motion {motion_id}")
        if observed.numel() != count or not torch.isfinite(world_from_camera).all():
            raise ValueError(f"Invalid camera data at motion {motion_id}")
        if not torch.all(frame_ids[1:] > frame_ids[:-1]):
            raise ValueError(f"Camera frame ids are not increasing at motion {motion_id}")
        expected_frame_ids = torch.arange(
            int(clip["start"]), int(clip["start"]) + count, dtype=frame_ids.dtype
        )
        if not torch.equal(frame_ids.cpu(), expected_frame_ids):
            raise ValueError(
                f"Camera/clip-map frame mismatch at motion {motion_id}: "
                f"camera={int(frame_ids[0])}..{int(frame_ids[-1])}, "
                f"clip={int(clip['start'])}..{int(clip['start']) + count - 1}"
            )
        rotation = world_from_camera[:, :3, :3]
        identity = torch.eye(3).expand(count, -1, -1)
        if not torch.allclose(rotation.transpose(1, 2) @ rotation, identity, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Non-rigid camera transform at motion {motion_id}")
        observed_frames += int(observed.sum())

    quaternion_norm = torch.linalg.vector_norm(motion["grs"], dim=-1)
    max_quaternion_error = float((quaternion_norm - 1.0).abs().max())
    if max_quaternion_error > 2e-3:
        raise ValueError(f"Invalid SOMA23 quaternion norm: max error {max_quaternion_error:.6f}")
    total_frames = int(counts.sum())
    fallback_mask = motion.get("retarget_unlabelled_support_fallback_mask")
    fallback_motions = 0 if fallback_mask is None else int(fallback_mask.sum())
    return {
        "split": manifest.get("split", manifest_path.parent.name),
        "motions": num_motions,
        "frames": total_frames,
        "camera_observed": observed_frames,
        "camera_coverage": observed_frames / max(total_frames, 1),
        "max_quaternion_error": max_quaternion_error,
        "unlabelled_support_fallback_motions": fallback_motions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--expected-frames", type=int, default=192)
    parser.add_argument(
        "--prior-config",
        type=Path,
        default=Path("data/pretrained_models/gpc_prior/soma_bones/config.yaml"),
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("data/pretrained_models/motion_tracker/soma_bones_fsq/config.yaml"),
    )
    args = parser.parse_args()

    _validate_checkpoint_config(args.prior_config, "GPC prior")
    _validate_checkpoint_config(args.tracker_config, "FSQ tracker")
    summaries = []
    for split in args.splits:
        manifest_path = args.pack_root / split / "manifest.json"
        if manifest_path.is_file():
            summaries.append(_validate_split(manifest_path, args.expected_frames))
    if not summaries:
        raise FileNotFoundError(f"No split manifests below {args.pack_root}")
    print("Validated SOMA23 GPC pack and frame-aligned scene cameras:")
    for summary in summaries:
        print(
            f"  {summary['split']}: {summary['motions']} clips, "
            f"{summary['frames']} frames, camera coverage "
            f"{summary['camera_coverage']:.2%}, max |quat_norm-1| "
            f"{summary['max_quaternion_error']:.2e}, contact fallback "
            f"{summary['unlabelled_support_fallback_motions']}"
        )


if __name__ == "__main__":
    main()
