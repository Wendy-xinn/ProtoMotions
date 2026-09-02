#!/usr/bin/env python3
"""Prepare, retarget and ground every recording in an EgoBody SFT manifest.

The command is resumable: an existing output is reused unless ``--overwrite``
is supplied.  Each recording is packaged independently so its static scene is
loaded once during later expert-cache collection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--egobody-root",
        type=Path,
        default=None,
        help="Override the source root recorded in the selection manifest.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--retarget-device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Rebuild only these recordings while retaining the complete manifest.",
    )
    parser.add_argument(
        "--skip-usd",
        action="store_true",
        help="Defer OBJ-to-USD scene conversion until cache collection.",
    )
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def base_recording_is_complete(
    output_dir: Path, starts: list[str], frame_count: int
) -> bool:
    required = [
        output_dir / "ego_camera.pt",
        output_dir / "scene_lib_training.pt",
        output_dir / "motion_lib.yaml",
    ]
    required.extend(
        output_dir
        / "motions"
        / f"frame_{int(start):05d}_count_{frame_count:04d}.motion"
        for start in starts
    )
    return all(path.is_file() for path in required)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    egobody_root = (
        args.egobody_root.resolve()
        if args.egobody_root is not None
        else Path(manifest["egobody_root"]).resolve()
    )
    manifest["egobody_root"] = str(egobody_root)
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else manifest_path.parent / "recordings"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    recordings = sorted({clip["recording"] for clip in manifest["clips"]})
    if args.recordings is not None:
        requested = set(args.recordings)
        unknown = requested.difference(recordings)
        if unknown:
            raise ValueError(f"Requested recordings are absent from manifest: {unknown}")
        recordings = [recording for recording in recordings if recording in requested]
    for index, recording in enumerate(recordings, 1):
        clips = [clip for clip in manifest["clips"] if clip["recording"] == recording]
        starts = [str(clip["start"]) for clip in sorted(clips, key=lambda item: item["start"])]
        output_dir = output_root / recording
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(recordings)}] {recording}: starts={starts}", flush=True)

        smpl_pack = output_dir / "motion_lib_smpl.pt"
        soma_pack = output_dir / "motion_lib_soma23.pt"
        grounded_pack = output_dir / "motion_lib_soma23_grounded.pt"
        grounded_camera = output_dir / "ego_camera_grounded.pt"
        scene_usd = output_dir / "scene_lib_training_isaaclab.pt"
        base_complete = base_recording_is_complete(
            output_dir, starts, manifest["frame_count"]
        )
        rebuild_recording = args.overwrite or not base_complete
        overwrite = ["--overwrite"] if rebuild_recording else []
        if rebuild_recording:
            run(
                [
                    python,
                    "data/scripts/prepare_egobody_smpl_scene.py",
                    "--egobody-root",
                    str(egobody_root),
                    "--recording",
                    recording,
                    "--frame-count",
                    str(manifest["frame_count"]),
                    "--clip-starts",
                    *starts,
                    "--output-root",
                    str(output_root),
                    *overwrite,
                ],
                env=env,
            )

        if rebuild_recording or not smpl_pack.is_file():
            run(
                [
                    python,
                    "protomotions/components/motion_lib.py",
                    "--motion-path",
                    str(output_dir / "motion_lib.yaml"),
                    "--output-file",
                    str(smpl_pack),
                    "--device",
                    "cpu",
                ],
                env=env,
            )

        if rebuild_recording or not soma_pack.is_file():
            run(
                [
                    python,
                    "data/scripts/retarget_packaged_smpl_to_soma23.py",
                    str(smpl_pack),
                    str(soma_pack),
                    "--device",
                    args.retarget_device,
                    "--ik-iterations",
                    "0",
                    *overwrite,
                ],
                env=env,
            )

        if rebuild_recording or not grounded_pack.is_file():
            run(
                [
                    python,
                    "data/scripts/align_soma23_retarget_foot_height.py",
                    str(smpl_pack),
                    str(soma_pack),
                    str(grounded_pack),
                    *overwrite,
                ],
                env=env,
            )

        if rebuild_recording or not grounded_camera.is_file():
            run(
                [
                    python,
                    "data/scripts/align_ego_camera_to_grounded_motion.py",
                    str(output_dir / "ego_camera.pt"),
                    str(grounded_pack),
                    str(grounded_camera),
                    *overwrite,
                ],
                env=env,
            )

        if not args.skip_usd and (rebuild_recording or not scene_usd.is_file()):
            run(
                [
                    str(PROJECT_ROOT / "scripts/run_wsl_isaaclab.sh"),
                    python,
                    "scripts/convert_obj_scenes_to_usd.py",
                    "--scene-file",
                    str(output_dir / "scene_lib_training.pt"),
                    "--asset-root",
                    str(egobody_root),
                    "--output-scene-file",
                    str(scene_usd),
                ],
                env=env,
            )

    prepared = dict(manifest)
    prepared["prepared_root"] = str(output_root)
    prepared_path = manifest_path.parent / "prepared_manifest.json"
    prepared_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(f"Prepared manifest: {prepared_path}")


if __name__ == "__main__":
    main()
