#!/usr/bin/env python3
"""Build an EgoBody Head 6-DoF condition for ProtoMotions MaskedMimic.

The GT path is deliberately derived from the official per-frame SMPL fits.  The
fits are transformed from Kinect-12 coordinates to the HoloLens world, made
Z-up, and converted with ProtoMotions' own SMPL kinematics.  Consequently the
saved Head pose uses exactly the body-frame convention of the checkpoint's
``Head`` rigid body rather than an approximate shoulder-facing frame.

The PV path maps the synchronized PV camera trajectory into that same Head
frame with one rigid camera-to-head mount.  Translation and rotation are both
preserved.  ``pre`` is reserved as a stable CLI/API slot; no prediction source
is silently substituted for it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
PROTO_ROOT = HERE.parents[1]
DEFAULT_EGOBODY_ROOT = Path("/public/home/wenxin/egobody")
DEFAULT_HUMAN3R_DATA = Path("/public/home/wenxin/Human3R/data")
# Rigid-body tensors follow SMPL_MUJOCO_NAMES (depth-first MJCF order), not
# the canonical SMPL parameter order where Head is joint 15.
HEAD_INDEX = 13

# Executing a file below ``integrations/`` puts only that directory on
# sys.path; make the ProtoMotions namespace packages (notably ``data`` and
# ``protomotions``) importable without requiring a caller-specific PYTHONPATH.
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Head position+rotation condition for MaskedMimic"
    )
    parser.add_argument("--recording", required=True)
    parser.add_argument("--cam-input", required=True, choices=("gt", "pv", "pre"))
    parser.add_argument("--start", type=int, default=0, help="Index in the aligned recording manifest")
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--egobody-root", type=Path, default=DEFAULT_EGOBODY_ROOT)
    parser.add_argument("--human3r-data-root", type=Path, default=DEFAULT_HUMAN3R_DATA)
    parser.add_argument(
        "--output-root", type=Path, default=PROTO_ROOT / "outputs" / "egobody_masked_mimic"
    )
    parser.add_argument(
        "--pv-calibration-frames",
        type=int,
        default=0,
        help="Frames used to estimate the fixed PV-to-Head mount; 0 uses the selected clip",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _find_smpl_recording(root: Path, recording: str) -> Path:
    matches = sorted(root.glob(f"smpl_camera_wearer_*/{recording}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one extracted SMPL wearer directory for {recording}, got {matches}. "
            "Extract smpl_camera_wearer_<split>.zip under the EgoBody root first."
        )
    return matches[0]


def _resolve_frame_pkl(recording_dir: Path, frame_id: int) -> Path:
    matches = list(recording_dir.glob(f"body_idx_*/results/frame_{frame_id:05d}/000.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one wearer SMPL fit for frame {frame_id:05d} below {recording_dir}, got {matches}"
        )
    return matches[0]


def _slice_indices(length: int, start: int, num_frames: int, stride: int) -> np.ndarray:
    if start < 0 or stride <= 0 or num_frames <= 1:
        raise ValueError("start must be >= 0, stride > 0, and num_frames > 1")
    stop = min(length, start + num_frames * stride)
    indices = np.arange(start, stop, stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError(f"The requested clip contains only {len(indices)} frame(s)")
    return indices


def _load_smpl_sequence(recording_dir: Path, frame_ids: np.ndarray):
    poses, trans, betas = [], [], []
    for frame_id in frame_ids.tolist():
        with _resolve_frame_pkl(recording_dir, int(frame_id)).open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        poses.append(
            np.concatenate(
                [
                    np.asarray(data["global_orient"], dtype=np.float32).reshape(3),
                    np.asarray(data["body_pose"], dtype=np.float32).reshape(69),
                ]
            )
        )
        trans.append(np.asarray(data["transl"], dtype=np.float32).reshape(3))
        betas.append(np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:10])
    return np.stack(poses), np.stack(trans), np.median(np.stack(betas), axis=0)


def _y_up_to_z_up() -> np.ndarray:
    # HoloLens spatial coordinates are Y-up.  This proper rotation maps +Y to +Z.
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    return result


def _transform_root_pose(
    pose_aa: np.ndarray, trans: np.ndarray, t_holo_from_kinect: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_zup_from_kinect = _y_up_to_z_up() @ t_holo_from_kinect
    root_rotation = Rotation.from_rotvec(pose_aa[:, :3]).as_matrix()
    transformed_rotation = t_zup_from_kinect[:3, :3][None] @ root_rotation
    transformed_trans = (
        np.einsum("ij,tj->ti", t_zup_from_kinect[:3, :3], trans)
        + t_zup_from_kinect[:3, 3]
    )

    # Keep metric height, but put the first root horizontally near the simulator origin.
    xy_origin = transformed_trans[0, :2].copy()
    transformed_trans[:, :2] -= xy_origin
    pose_out = pose_aa.copy()
    pose_out[:, :3] = Rotation.from_matrix(transformed_rotation).as_rotvec()

    t_proto_from_holo = _y_up_to_z_up()
    t_proto_from_holo[:2, 3] = -xy_origin
    return pose_out, transformed_trans, t_proto_from_holo


def _convert_to_proto_motion(pose_aa: np.ndarray, trans: np.ndarray, fps: int):
    scripts_dir = PROTO_ROOT / "data" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from convert_amass_to_proto import convert_amass_to_motion
        from protomotions.components.pose_lib import extract_kinematic_info
        from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
    finally:
        sys.path.pop(0)

    kinematic_info = extract_kinematic_info(
        str(PROTO_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "smpl_humanoid.xml")
    )
    motion, actual_fps = convert_amass_to_motion(
        pose_aa=pose_aa,
        amass_trans=trans,
        mocap_fr=fps,
        output_fps=fps,
        humanoid_type="smpl",
        joint_names=SMPL_BONE_ORDER_NAMES,
        mujoco_joint_names=SMPL_MUJOCO_NAMES,
        kinematic_info=kinematic_info,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return motion, int(actual_fps)


def _poses_to_transforms(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(position), axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    transforms[:, :3, 3] = position
    return transforms


def _average_mount(relative: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(relative[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.median(relative[:, :3, 3], axis=0)
    return result


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(a, -1, -2) @ b
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def _infer_fps(timestamps: np.ndarray, stride: int) -> int:
    """Infer FPS while tolerating seconds, ms, us, 100-ns ticks, or ns."""
    raw_dt = float(np.median(np.diff(timestamps)))
    candidates = []
    for ticks_per_second in (1.0, 1e3, 1e6, 1e7, 1e9):
        fps = ticks_per_second / raw_dt if raw_dt > 0 else 0.0
        if 5.0 <= fps <= 240.0:
            candidates.append(fps)
    if not candidates:
        # EgoBody PV is nominally 30 Hz. Respect explicit temporal subsampling.
        return max(1, int(round(30.0 / stride)))
    return int(round(min(candidates, key=lambda value: abs(value - 30.0 / stride))))


def main() -> None:
    args = parse_args()
    if args.cam_input == "pre":
        raise NotImplementedError(
            "--cam-input pre is reserved, but no predicted-camera producer has been selected yet. "
            "The interface intentionally refuses to substitute GT or PV data."
        )

    record_data = args.human3r_data_root / args.recording
    manifest = _load_torch(record_data / "manifest.pt")
    calibration = _load_torch(record_data / "calibration.pt")
    camera_head = _load_torch(record_data / "camera_head_traj.pt")
    frame_ids_all = np.asarray(manifest["frame_ids"], dtype=np.int64)
    indices = _slice_indices(len(frame_ids_all), args.start, args.num_frames, args.stride)
    frame_ids = frame_ids_all[indices]

    timestamps = np.asarray(manifest["query_timestamps"], dtype=np.float64)[indices]
    fps = _infer_fps(timestamps, args.stride)

    recording_dir = _find_smpl_recording(args.egobody_root, args.recording)
    pose_aa, trans, betas = _load_smpl_sequence(recording_dir, frame_ids)
    t_holo_from_kinect = np.asarray(calibration["T_kinect12_to_holo"], dtype=np.float64)
    pose_proto, trans_proto, t_proto_from_holo = _transform_root_pose(
        pose_aa, trans, t_holo_from_kinect
    )
    motion, fps = _convert_to_proto_motion(pose_proto, trans_proto, fps)

    motion_dict = motion.to_dict()
    gt_pos = motion.rigid_body_pos[:, HEAD_INDEX].cpu().numpy().astype(np.float64)
    gt_quat = motion.rigid_body_rot[:, HEAD_INDEX].cpu().numpy().astype(np.float64)
    gt_transform = _poses_to_transforms(gt_pos, gt_quat)

    # ProtoMotions' official converter vertically fixes the motion.  Apply the
    # identical scalar shift to camera poses before estimating the rigid mount.
    height_shift = float(motion.rigid_body_pos[0, 0, 2] - trans_proto[0, 2])
    t_proto_from_holo[2, 3] += height_shift

    diagnostics = {}
    mount = np.eye(4, dtype=np.float64)
    if args.cam_input == "gt":
        head_transform = gt_transform
    else:
        pv_holo = np.asarray(camera_head["T_holo_pv"], dtype=np.float64)[indices]
        pv_proto = t_proto_from_holo[None] @ pv_holo
        calibration_frames = args.pv_calibration_frames or len(indices)
        calibration_frames = min(calibration_frames, len(indices))
        relative = np.linalg.inv(pv_proto[:calibration_frames]) @ gt_transform[:calibration_frames]
        mount = _average_mount(relative)
        head_transform = pv_proto @ mount[None]
        diagnostics = {
            "pv_head_translation_rmse_m": float(
                np.sqrt(np.mean(np.sum((head_transform[:, :3, 3] - gt_pos) ** 2, axis=-1)))
            ),
            "pv_head_rotation_mean_deg": float(
                _rotation_error_deg(head_transform[:, :3, :3], gt_transform[:, :3, :3]).mean()
            ),
            "pv_calibration_frames": int(calibration_frames),
        }

    output_dir = args.output_root / args.recording / (
        f"{args.cam_input}_start{args.start}_frames{len(indices)}_stride{args.stride}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = output_dir / "head_condition.npz"
    motion_path = output_dir / "bootstrap_gt_smpl.motion"
    metadata_path = output_dir / "metadata.json"
    if not args.force and (condition_path.exists() or motion_path.exists()):
        raise FileExistsError(f"Output already exists below {output_dir}; pass --force to replace it")

    head_quat = Rotation.from_matrix(head_transform[:, :3, :3]).as_quat().astype(np.float32)
    np.savez_compressed(
        condition_path,
        head_pos=head_transform[:, :3, 3].astype(np.float32),
        head_quat_xyzw=head_quat,
        head_transform=head_transform.astype(np.float32),
        gt_head_transform=gt_transform.astype(np.float32),
        frame_ids=frame_ids,
        manifest_indices=indices,
        fps=np.asarray(fps, dtype=np.int64),
        cam_input=np.asarray(args.cam_input),
        pv_to_head=mount.astype(np.float32),
        t_proto_from_holo=t_proto_from_holo.astype(np.float32),
    )
    torch.save(motion_dict, motion_path)
    metadata = {
        "recording": args.recording,
        "cam_input": args.cam_input,
        "start": args.start,
        "num_frames": len(indices),
        "stride": args.stride,
        "fps": fps,
        "head_body_name": "Head",
        "head_body_index": HEAD_INDEX,
        "condition_has_translation": True,
        "condition_has_rotation": True,
        "smpl_recording_dir": str(recording_dir),
        "condition_path": str(condition_path),
        "bootstrap_motion_path": str(motion_path),
        "diagnostics": diagnostics,
        "note": "bootstrap full-body GT initializes MotionLib; MaskedMimic inference is masked to Head only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
