#!/usr/bin/env python3
"""Collect one expert-physics transition cache for each prepared recording."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--head-orientation-feedback-gain", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.head_orientation_feedback_gain <= 1.0:
        parser.error("--head-orientation-feedback-gain must be in [0, 1]")
    manifest_path = args.prepared_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_root = Path(manifest["prepared_root"])
    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else manifest_path.parent / "expert_cache_grounded_v3"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    python = PROJECT_ROOT / "IsaacLab/.venv/bin/python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    by_recording = {}
    for clip in manifest["clips"]:
        by_recording.setdefault(clip["recording"], []).append(clip)
    for index, (recording, clips) in enumerate(sorted(by_recording.items()), 1):
        output = cache_root / recording
        existing_manifest = output / "cache_manifest.json"
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if len(existing.get("clips", [])) == len(clips):
                print(f"[{index}/{len(by_recording)}] Reusing {recording}", flush=True)
                continue
        data_dir = prepared_root / recording
        split = clips[0]["split"]
        num_envs = min(args.num_envs, len(clips))
        command = [
            str(PROJECT_ROOT / "scripts/run_with_memory_guard.sh"),
            str(PROJECT_ROOT / "scripts/run_wsl_isaaclab.sh"),
            str(python),
            str(PROJECT_ROOT / "protomotions/train_agent.py"),
            "--robot-name",
            "soma23",
            "--simulator",
            "isaaclab",
            "--num-envs",
            str(num_envs),
            "--batch-size",
            "128",
            "--motion-file",
            str(data_dir / "motion_lib_soma23_grounded.pt"),
            "--scenes-file",
            str(data_dir / "scene_lib_training_isaaclab.pt"),
            "--scene-asset-root",
            manifest["egobody_root"],
            "--ego-camera-file",
            str(data_dir / "ego_camera_grounded.pt"),
            "--episode-length",
            str(manifest["frame_count"]),
            "--offline-cache-output",
            str(output),
            "--offline-cache-split",
            split,
            "--head-orientation-feedback-gain",
            str(args.head_orientation_feedback_gain),
            "--experiment-path",
            str(PROJECT_ROOT / "examples/experiments/gpc/sft_trumans_scene_head_overfit.py"),
            "--experiment-name",
            f"_egobody_cache_{recording}",
            "--training-max-steps",
            "1",
            "--headless",
        ]
        print(
            f"[{index}/{len(by_recording)}] Collecting {recording} ({len(clips)} clips)",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    print(f"All expert caches are under {cache_root}")


if __name__ == "__main__":
    main()
