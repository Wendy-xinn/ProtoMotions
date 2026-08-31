# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SAMP SMPL motions to SOMA23 format using SOMA-X pose inversion.

Pipeline per motion:
    1. Load SMPL .motion file (dof_pos, rigid_body_pos/rot in z-up)
    2. Reconstruct SMPL body_pose + global_orient (axis-angle)
    3. Run smplx forward pass → posed vertices [T, 6890, 3]
    4. Batch through SOMA-X PoseInversion → SOMA 78-joint rotations [T, 78, 3, 3]
    5. Subsample 78 → 23 MJCF joints
    6. Convert via create_motion_from_soma23_global_rotations (handles y-up → z-up)
    7. Transfer contact labels from SMPL motion
    8. Save as .motion file

The SMPL vertices are in SMPL's native y-up frame (smplx handles this).
SOMA-X outputs y-up rotations, which create_motion_from_soma23_global_rotations
converts to z-up.

Usage::

    python data/scripts/convert_smpl_samp_to_soma23.py \\
        ~/protomotions_assets/samp/motions/ \\
        ~/protomotions_assets/samp/motions_soma23/ \\
        --batch-size 128

    # Single file
    python data/scripts/convert_smpl_samp_to_soma23.py \\
        ~/protomotions_assets/samp/motions/armchair001_stageII.motion \\
        ~/protomotions_assets/samp/motions_soma23/ \\
        --batch-size 128
"""

from __future__ import annotations

import inspect
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

# chumpy 0.70 is required to unpickle the public SMPL model but predates
# Python 3.11's removal of inspect.getargspec.  Install the narrow compatibility
# shim before importing smplx (which triggers the chumpy import).
if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import smplx
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.spatial.transform import Rotation as Rot

from soma import SOMALayer
from soma.pose_inversion import PoseInversion

# Add data/scripts to path for convert_soma23_to_proto imports
sys.path.insert(0, str(Path(__file__).parent))
from convert_soma23_to_proto import (
    MJCF_BODY_NAMES,
    SOMASKEL77_TO_MJCF_INDICES,
    create_motion_from_soma23_global_rotations,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils.rotations import quaternion_to_matrix

# MJCF body order → AMASS body order permutation.
# smpl_2_mujoco[mj_idx] = amass_idx (used during amass→proto conversion).
# We need the inverse: for each amass_idx, which mj_idx holds that joint.
_SMPL_2_MUJOCO = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES]
_MJCF_TO_AMASS = [0] * 24
for _mj, _am in enumerate(_SMPL_2_MUJOCO):
    _MJCF_TO_AMASS[_am] = _mj

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# SOMA-X uses 78 joints: Root (index 0) + 77 SOMASKEL77 joints (1-77).
# To get SOMA23 MJCF bodies, offset SOMASKEL77_TO_MJCF_INDICES by +1.
SOMAX78_TO_MJCF_INDICES = [i + 1 for i in SOMASKEL77_TO_MJCF_INDICES]

# SMPL pkl path (chumpy-free version cached by SOMA-X setup)
SMPL_PKL = Path(__file__).resolve().parent.parent / "smpl" / "SMPL_NEUTRAL.pkl"
SOMA_X_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "soma_x_v2" / "assets"

# ``convert_amass_to_proto.py`` does not merely rotate the root.  Before it
# stores ``local_rigid_body_rot`` it right-multiplies *every global rotation*
# by this matrix and then recomputes locals.  Consequently every non-root
# local is conjugated.  Undoing only a z-up/y-up root transform (the previous
# implementation) changes the anatomical axes of shoulders, elbows, hips and
# knees and produces the characteristic twisted limbs.
_AMASS_TO_PROTO_ROT = torch.tensor(
    Rot.from_euler("xyz", [-np.pi / 2, -np.pi / 2, 0]).as_matrix(),
    dtype=torch.float32,
)
_ZUP_TO_YUP = torch.tensor(
    Rot.from_euler("x", -90, degrees=True).as_matrix(), dtype=torch.float32
)


def recover_smpl_local_rotations(
    proto_local_rotations: torch.Tensor,
) -> torch.Tensor:
    """Invert the rotation storage transform in convert_amass_to_proto.py.

    Args:
        proto_local_rotations: ``[T, 24, 3, 3]`` in SMPL/AMASS joint order.

    Returns:
        Original SMPL local rotation matrices in the source coordinate frame.
    """
    recovered = torch.empty_like(proto_local_rotations)
    r = _AMASS_TO_PROTO_ROT.to(proto_local_rotations)
    # Stored root is G_root @ R.
    recovered[:, 0] = proto_local_rotations[:, 0] @ r.T
    # Stored child local is R^T @ L_child @ R.
    recovered[:, 1:] = r @ proto_local_rotations[:, 1:] @ r.T
    return recovered


def create_soma23_motion_from_zup_globals(
    global_rot_mats: torch.Tensor,
    root_pos: torch.Tensor,
    kinematic_info,
    fps: int,
):
    """Build SOMA23 motion when the SOMA FK result is already z-up.

    The checked-in BONES/SEED bridge performs a y-up -> z-up conversion.  SMPL
    reconstructed from a Proto/AMASS motion remains z-up, so applying that
    conversion a second time puts the character below the floor.
    """
    from protomotions.components.pose_lib import (
        compute_angular_velocity,
        compute_joint_rot_mats_from_global_mats,
        extract_qpos_from_transforms,
        fk_from_transforms_with_velocities,
    )
    from protomotions.utils.rotations import matrix_to_quaternion

    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=global_rot_mats,
    )
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.local_rigid_body_rot = matrix_to_quaternion(local_rot_mats, w_last=True)
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    motion.dof_vel = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats[:, 1:], fps=fps
    ).reshape(local_rot_mats.shape[0], -1)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)
    return motion


def _quat_to_rotvec(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to axis-angle (rotvec). Shape [..., 4] → [..., 3]."""
    # scipy expects wxyz
    q_np = q_xyzw.cpu().numpy()
    shape = q_np.shape[:-1]
    q_flat = q_np.reshape(-1, 4)
    rv = Rot.from_quat(q_flat).as_rotvec()  # scipy uses xyzw
    return torch.from_numpy(rv.reshape(*shape, 3)).float()


def convert_motion(
    smpl_motion_path: Path,
    soma: SOMALayer,
    inv: PoseInversion,
    smpl_model: smplx.SMPL,
    kinematic_info,
    batch_size: int,
    device: torch.device,
    max_frames: int | None = None,
) -> dict:
    """Convert a single SMPL .motion file to SOMA23 .motion dict."""
    motion_data = torch.load(smpl_motion_path, weights_only=False, map_location="cpu")
    local_rots = motion_data[
        "local_rigid_body_rot"
    ]  # [T, 24, 4] xyzw quats, MJCF order
    rigid_body_pos = motion_data["rigid_body_pos"]  # [T, 24, 3] z-up
    fps = motion_data.get("fps", 30)
    if max_frames is not None:
        local_rots = local_rots[:max_frames]
        rigid_body_pos = rigid_body_pos[:max_frames]
    T = local_rots.shape[0]

    # --- Step 1: Reconstruct SMPL parameters from local_rigid_body_rot ---
    # Reorder from MJCF body order → AMASS body order (what smplx expects)
    local_rots_amass = local_rots[:, _MJCF_TO_AMASS]  # [T, 24, 4] xyzw, AMASS order

    stored_local_mats = quaternion_to_matrix(local_rots_amass, w_last=True)
    smpl_local_mats = recover_smpl_local_rotations(stored_local_mats)
    smpl_local_aa = (
        torch.from_numpy(
            Rot.from_matrix(smpl_local_mats.reshape(-1, 3, 3).numpy()).as_rotvec()
        )
        .float()
        .reshape(T, 24, 3)
    )
    global_orient = smpl_local_aa[:, 0]
    body_pose = smpl_local_aa[:, 1:].reshape(T, 69)

    # AMASS translation was never rotated by convert_amass_to_proto.py.
    root_pos_source = rigid_body_pos[:, 0]

    # --- Step 2: Batch smplx forward → vertices ---
    all_rotations = []
    all_root_trans = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bp = body_pose[start:end].to(device)
        go = global_orient[start:end].to(device)
        tr = root_pos_source[start:end].to(device)

        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=bp,
                global_orient=go,
                transl=tr,
            )
            # The checked-in SOMA Uniform -> SOMA23 bridge is y-up. Rotate the
            # complete reconstructed mesh (not individual joint parameters)
            # into that world frame before pose inversion.
            verts = smpl_out.vertices @ _ZUP_TO_YUP.to(device).T

            # --- Step 3: SOMA-X pose inversion ---
            result = inv.fit(
                verts,
                body_iters=3,
                full_iters=1,
                lie_iters=5,
                constrain_1dof=True,
                leaf_weight={"head": 2.0, "hands": 3.0, "feet": 5.0, "heels": 8.0},
            )
            # result["rotations"]: [B, 78, 3, 3] absolute rotations (y-up)
            # result["root_translation"]: [B, 3]
            all_rotations.append(result["rotations"].cpu())
            all_root_trans.append(result["root_translation"].cpu())

    soma_rotations = torch.cat(all_rotations, dim=0)  # [T, 78, 3, 3]
    soma_root_trans = torch.cat(all_root_trans, dim=0)  # [T, 3]

    # --- Step 4+5+6: SOMA-X FK -> BONES/SOMA-Uniform standard frame -> SOMA23 ---
    # PoseInversion returns absolute *local* rotations with SOMA joint orient
    # baked in.  Do not reinterpret them as globals and do not remove joint
    # orient before passing them to the SOMA rig.  The official reconstruction
    # contract is BatchedSkinning.pose(..., absolute_pose=True).
    public_local_abs = soma.to_public_rotations(soma_rotations.to(device))
    with torch.no_grad():
        _, public_world = soma.public_batched_skinning.pose(
            local_rotations=public_local_abs,
            hips_translations=soma_root_trans.to(device),
            absolute_pose=True,
            return_transforms=True,
        )
    # The GPC SOMA23 motions were produced from BONES/SEED SOMA Uniform, whose
    # standard frame is defined by this checked-in 77-joint offset tensor.  It
    # is not identical to SOMA-X's public rig bind/joint-orient convention.
    standard_offsets = torch.load(
        Path(__file__).resolve().parents[1]
        / "soma"
        / "standard_t_pose_global_offsets_rots.p",
        weights_only=False,
        map_location=device,
    ).to(device=device, dtype=public_world.dtype)
    soma77_world = public_world[:, 1:, :3, :3]
    standard_world = soma77_world @ standard_offsets.transpose(-2, -1)
    soma23_standard_world = standard_world[:, SOMASKEL77_TO_MJCF_INDICES]
    bones_convention_motion = create_motion_from_soma23_global_rotations(
        global_rot_mats=soma23_standard_world.cpu(),
        root_pos=soma_root_trans.cpu(),
        kinematic_info=kinematic_info,
        fps=fps,
    )
    # The BONES converter includes a 180-degree heading convention that is
    # appropriate for its BVH files.  Our source is already a Proto z-up motion;
    # remove that fixed yaw and retain its exact scene-space root trajectory.
    bones_world = quaternion_to_matrix(
        bones_convention_motion.rigid_body_rot, w_last=True
    )
    undo_bones_heading = torch.tensor(
        Rot.from_euler("z", 180, degrees=True).as_matrix(), dtype=bones_world.dtype
    )
    proto_world = undo_bones_heading @ bones_world
    motion = create_soma23_motion_from_zup_globals(
        global_rot_mats=proto_world,
        root_pos=root_pos_source,
        kinematic_info=kinematic_info,
        fps=fps,
    )

    # Build output dict
    out = motion.to_dict()
    out["fps"] = fps
    return out


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., help="Single .motion file or directory of .motion files"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for SOMA23 .motion files"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Frames per GPU batch"),
    device_str: str = typer.Option("cuda", "--device"),
    max_frames: int | None = typer.Option(
        None, "--max-frames", help="Convert only the first N frames (review/debug)"
    ),
) -> None:
    """Convert SMPL SAMP motions to SOMA23 via SOMA-X pose inversion."""
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_file():
        motion_files = [input_path]
    elif input_path.is_dir():
        motion_files = sorted(input_path.glob("*.motion"))
    else:
        console.print(f"[red]ERROR[/]: {input_path} not found")
        raise typer.Exit(1)

    if not motion_files:
        console.print(f"[red]ERROR[/]: no .motion files found in {input_path}")
        raise typer.Exit(1)

    console.print(f"Converting {len(motion_files)} motions | batch_size={batch_size}")

    # --- Initialize models ---
    console.print("Loading SMPL model...")
    smpl_model = smplx.create(str(SMPL_PKL), model_type="smpl").to(device)

    console.print("Loading SOMA-X...")
    soma = SOMALayer(
        data_root=SOMA_X_DATA_ROOT,
        identity_model_type="smpl",
        identity_model_kwargs={"model_path": str(SMPL_PKL)},
        device=device_str,
        mode="warp",
        low_lod=True,
    )
    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(torch.zeros(1, 10).to(device))

    console.print("Loading SOMA23 kinematic info...")
    mjcf_path = (
        Path(__file__).parent.parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
        / "soma23_humanoid.xml"
    )
    kinematic_info = extract_kinematic_info(str(mjcf_path))

    # --- Convert ---
    successes = 0
    failures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Converting", total=len(motion_files), status="")

        for mf in motion_files:
            name = mf.stem
            progress.update(task, status=f"[cyan]{name[:40]}[/]")
            out_path = output_dir / mf.name

            if out_path.exists():
                progress.console.print(f"  [dim]SKIP[/] {name} (exists)")
                progress.advance(task)
                successes += 1
                continue

            try:
                out_dict = convert_motion(
                    mf,
                    soma,
                    inv,
                    smpl_model,
                    kinematic_info,
                    batch_size,
                    device,
                    max_frames,
                )
                torch.save(out_dict, out_path)
                successes += 1
            except Exception as e:
                failures.append((name, str(e)))
                progress.console.print(f"  [red]FAIL[/] {name}: {e}")

            progress.advance(task)

    console.print(f"\n[green]Done[/]: {successes}/{len(motion_files)} converted")
    if failures:
        console.print(f"[red]Failures ({len(failures)}):[/]")
        for name, err in failures:
            console.print(f"  {name}: {err}")


if __name__ == "__main__":
    app()
