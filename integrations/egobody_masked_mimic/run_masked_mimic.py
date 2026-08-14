#!/usr/bin/env python3
"""Launch the official MaskedMimic checkpoint with an external Head condition."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTO_ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROTO_ROOT / "data/pretrained_models/masked_mimic/smpl/last.ckpt",
    )
    parser.add_argument(
        "--simulator", choices=("isaacgym", "isaaclab", "mujoco"), default="isaaclab"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    for path in (args.condition, args.motion_file, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROTO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Absolute ``.../envs/isaacgym/bin/python`` invocations do not activate the
    # conda environment, so its ninja executable would otherwise be invisible
    # while IsaacGym builds gymtorch.
    env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["EGOBODY_MM_HEAD_CONDITION"] = str(args.condition.resolve())
    env["EGOBODY_MM_OUTPUT"] = str(args.output.resolve())
    if args.max_steps > 0:
        env["EGOBODY_MM_STOP_AFTER_STEPS"] = str(args.max_steps)

    command = [
        sys.executable,
        str(PROTO_ROOT / "protomotions/inference_agent.py"),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--simulator",
        args.simulator,
        "--motion-file",
        str(args.motion_file.resolve()),
        "--num-envs",
        str(args.num_envs),
        "--overrides",
        "env.control_components.masked_mimic._target_=integrations.egobody_masked_mimic.external_head_control.ExternalHeadMaskedMimicControl",
        "env.control_components.masked_mimic.visible_target_pose_prob=1.0",
        "env.control_components.masked_mimic.force_max_conditioned_bodies_prob=0.0",
        "env.control_components.masked_mimic.force_small_num_conditioned_bodies_prob=0.0",
        "env.motion_manager.init_start_prob=1.0",
        "env.motion_manager.resample_on_reset=False",
        *args.overrides,
    ]
    if args.headless:
        command.insert(command.index("--overrides"), "--headless")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROTO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
