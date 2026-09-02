# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit
- **W/A/S/D**: Move target when running with ``--command-source target=keyboard``

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--fixed-motion-eval-batch-size",
        type=int,
        default=None,
        help="Evaluate fixed motions in smaller batches for stable PhysX comparisons.",
    )
    parser.add_argument(
        "--eval-action-ema-alpha",
        type=float,
        default=None,
        help=(
            "Apply causal action EMA during physical evaluation: "
            "applied=alpha*policy+(1-alpha)*previous."
        ),
    )
    parser.add_argument(
        "--eval-token-switch-penalty",
        type=float,
        default=None,
        help="Bias previous-frame tokens during greedy physical evaluation.",
    )
    parser.add_argument(
        "--policy-observation-intervention",
        choices=("none", "zero", "shuffle"),
        default="none",
        help="Intervene on selected policy observations during full evaluation.",
    )
    parser.add_argument(
        "--disable-scene-adapter",
        action="store_true",
        default=False,
        help="Set the scene residual actor gain to zero for a motion-only ablation.",
    )
    parser.add_argument(
        "--policy-observation-intervention-keys",
        nargs="+",
        default=None,
        help="Observation keys affected by --policy-observation-intervention.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        default=None,
        help=(
            "Restrict a single-environment rollout to one motion inside the "
            "motion file and reset it at time zero."
        ),
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--scene-asset-root",
        type=str,
        default=None,
        help="Asset root used to resolve relative object paths in --scenes-file.",
    )
    parser.add_argument(
        "--ego-camera-file",
        type=str,
        default=None,
        help="Packaged ego-camera trajectories aligned with --motion-file.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--command-source",
        nargs="*",
        default=[],
        help=(
            "Override task command sources for inference, e.g. "
            "target=keyboard. A bare value applies to the single target "
            "control component."
        ),
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=0,
        help=(
            "Automatically record this many simulation steps and exit. Headless "
            "runs save motion sidecars without PNG/MP4 output; files are written "
            "under output/renderings."
        ),
    )
    parser.add_argument(
        "--record-sidecars",
        action="store_true",
        help=(
            "Also save .markers.pt, .objects.pt, and .terrain.pt beside an "
            "automatic motion recording. By default --record-steps saves only "
            ".motion and synchronized .gt.motion; static scene geometry can be "
            "reloaded from --scenes-file."
        ),
    )
    parser.add_argument(
        "--head-translation-only",
        action="store_true",
        help=(
            "For masked-mimic checkpoints with a fixed Head condition, hide "
            "the Head rotation and retain translation only."
        ),
    )
    parser.add_argument(
        "--free-scene-objects",
        action="store_true",
        help=(
            "Disable kinematic scene-object reference replay during this rollout. "
            "Dynamic objects then move only under physics/contact; static objects "
            "remain fixed according to SceneLib."
        ),
    )
    parser.add_argument(
        "--oracle-target-tokens",
        action="store_true",
        help=(
            "Bypass learned prior-token prediction and decode frozen FSQ tokens "
            "obtained from the complete reference motion. This is a diagnostic "
            "for retarget, tokenization, decoder, and physical tracking."
        ),
    )
    parser.add_argument(
        "--oracle-takeover-step",
        type=int,
        default=None,
        help=(
            "Run student tokens before this zero-based rollout step, then use "
            "oracle target tokens for the remainder of each episode."
        ),
    )
    parser.add_argument(
        "--deterministic-tokens",
        action="store_true",
        help=(
            "Decode the highest-probability student token at every autoregressive "
            "step instead of sampling. This remains a closed-loop student rollout."
        ),
    )
    parser.add_argument(
        "--head-orientation-feedback-gain",
        type=float,
        default=0.0,
        help=(
            "Use the known offline ego Head orientation as world-space feedback, "
            "converted to a local Head PD target using simulated Neck2."
        ),
    )
    parser.add_argument(
        "--record-target-tokens",
        type=str,
        default=None,
        help=(
            "Save the frozen target encoder's per-step packed FSQ tokens. "
            "Requires --oracle-target-tokens and a finite --record-steps run."
        ),
    )
    parser.add_argument(
        "--offline-token-eval-cache",
        type=str,
        default=None,
        help=(
            "Evaluate teacher-forced and greedy-autoregressive token accuracy "
            "on one offline SFT cache .pt file, then exit."
        ),
    )
    parser.add_argument(
        "--record-student-oracle-tokens",
        type=str,
        default=None,
        help=(
            "During student rollout, save predicted tokens and frozen-encoder "
            "oracle tokens computed on the same current student state."
        ),
    )
    parser.add_argument(
        "--recovery-push-steps",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Completed simulation steps after which to apply a deterministic "
            "root-velocity impulse. Intended for finite oracle recovery tests."
        ),
    )
    parser.add_argument(
        "--recovery-push-linear-velocity",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
        help="Root linear-velocity impulse in m/s for recovery diagnostics.",
    )
    parser.add_argument(
        "--recovery-push-angular-velocity",
        type=float,
        nargs=3,
        metavar=("WX", "WY", "WZ"),
        default=(0.0, 0.0, 0.0),
        help="Root angular-velocity impulse in rad/s for recovery diagnostics.",
    )
    parser.add_argument(
        "--rollout-metrics-output",
        type=str,
        default=None,
        help="Optional JSON output containing per-step simple-rollout metrics.",
    )
    parser.add_argument(
        "--offline-token-eval-output",
        type=str,
        default=None,
        help="Optional JSON output for --offline-token-eval-cache.",
    )
    parser.add_argument(
        "--offline-token-eval-observations",
        type=str,
        default=None,
        help=(
            "Optional student rollout recording whose observations are substituted "
            "into the offline cache one field at a time."
        ),
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_offline_token_cache(
    agent,
    cache_path: str | Path,
    observation_path: str | Path | None = None,
) -> dict:
    """Compare teacher-forced and free-prefix predictions on cached states."""
    import json

    import torch.nn.functional as F
    from tensordict import TensorDict

    from protomotions.agents.common.latent import LATENT_LOGITS_KEY, TARGET_LATENT_KEY

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    tensors = payload["tensors"]
    valid = tensors.get("valid")
    valid_indices = (
        torch.arange(next(iter(tensors.values())).shape[0])
        if valid is None
        else valid.bool().nonzero(as_tuple=False).squeeze(-1)
    )
    if valid_indices.numel() == 0:
        raise ValueError(f"Offline token cache has no valid frames: {cache_path}")

    model = agent.model
    actor = getattr(model, "_actor", None)
    prior_with_peft = getattr(actor, "prior_with_peft", None)
    if actor is None or prior_with_peft is None:
        raise ValueError("Offline token evaluation requires a PEFT discrete-prior model")

    model.eval()
    old_deterministic = prior_with_peft.deterministic_generation
    prior_with_peft.deterministic_generation = True
    batch_size = int(getattr(agent.config, "batch_size", 128))
    target_batches = []
    teacher_batches = []
    autoregressive_batches = []
    cross_entropy_sum = 0.0
    try:
        for start in range(0, valid_indices.numel(), batch_size):
            frame_ids = valid_indices[start : start + batch_size]
            batch = {}
            for key, value in tensors.items():
                if key in {"valid", "terminated", "motion_time"}:
                    continue
                value = value[frame_ids]
                if value.is_floating_point():
                    value = value.float()
                batch[key] = value.to(agent.device)
            td = TensorDict(batch, batch_size=frame_ids.numel(), device=agent.device)
            target = td[TARGET_LATENT_KEY].long()

            teacher_td = model(td.clone())
            logits = teacher_td[LATENT_LOGITS_KEY]
            teacher = logits.argmax(dim=-1)
            cross_entropy_sum += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    reduction="sum",
                )
            )

            student_td = model.collect_student_rollout(td.clone())
            autoregressive = student_td["prior_tokens"].long()
            target_batches.append(target.cpu())
            teacher_batches.append(teacher.cpu())
            autoregressive_batches.append(autoregressive.cpu())
    finally:
        prior_with_peft.deterministic_generation = old_deterministic

    target = torch.cat(target_batches)
    teacher = torch.cat(teacher_batches)
    autoregressive = torch.cat(autoregressive_batches)

    def summarize(predicted: torch.Tensor) -> dict:
        packed_correct = predicted.eq(target)
        predicted_fsq = actor.prior_tokens_to_fsq_indices(predicted.to(agent.device)).cpu()
        target_fsq = actor.prior_tokens_to_fsq_indices(target.to(agent.device)).cpu()
        frame_exact = packed_correct.all(dim=-1)
        mismatch = (~frame_exact).nonzero(as_tuple=False).squeeze(-1)
        return {
            "packed_token_accuracy": float(packed_correct.float().mean()),
            "packed_token_accuracy_by_position": [
                float(value) for value in packed_correct.float().mean(dim=0)
            ],
            "frame_exact_8_tokens": float(frame_exact.float().mean()),
            "fsq_scalar_accuracy": float(predicted_fsq.eq(target_fsq).float().mean()),
            "first_mismatch_valid_frame_index": (
                None if mismatch.numel() == 0 else int(mismatch[0])
            ),
        }

    report = {
        "cache": str(Path(cache_path)),
        "metadata": payload.get("metadata", {}),
        "valid_frames": int(target.shape[0]),
        "tokens_per_frame": int(target.shape[1]),
        "teacher_forced": summarize(teacher),
        "greedy_autoregressive_on_cached_state": summarize(autoregressive),
        "teacher_forced_cross_entropy": cross_entropy_sum / target.numel(),
    }

    if observation_path is not None:
        observation_payload = torch.load(
            observation_path, map_location="cpu", weights_only=False
        )
        online_observations = observation_payload["observations"]
        common_keys = [key for key in online_observations if key in tensors]
        num_comparison_frames = min(
            int(valid_indices.numel()),
            min(int(online_observations[key].shape[0]) for key in common_keys),
        )
        comparison_frame_ids = valid_indices[:num_comparison_frames]
        comparison_target = tensors[TARGET_LATENT_KEY][comparison_frame_ids].long()

        def predict_with_online_keys(keys: list[str]) -> torch.Tensor:
            predictions = []
            for start in range(0, num_comparison_frames, batch_size):
                end = min(start + batch_size, num_comparison_frames)
                frame_ids = comparison_frame_ids[start:end]
                batch = {}
                for key, value in tensors.items():
                    if key in {"valid", "terminated", "motion_time"}:
                        continue
                    value = value[frame_ids]
                    if key in keys:
                        value = online_observations[key][start:end]
                    if value.is_floating_point():
                        value = value.float()
                    batch[key] = value.to(agent.device)
                td = TensorDict(batch, batch_size=end - start, device=agent.device)
                predictions.append(
                    model.collect_student_rollout(td)["prior_tokens"].long().cpu()
                )
            return torch.cat(predictions)

        def summarize_comparison(predicted: torch.Tensor) -> dict:
            packed_correct = predicted.eq(comparison_target)
            frame_exact = packed_correct.all(dim=-1)
            return {
                "packed_token_accuracy": float(packed_correct.float().mean()),
                "packed_token_accuracy_by_position": [
                    float(value) for value in packed_correct.float().mean(dim=0)
                ],
                "frame_exact_8_tokens": float(frame_exact.float().mean()),
                "predicted_tokens_by_frame": predicted.tolist(),
            }

        old_deterministic = prior_with_peft.deterministic_generation
        prior_with_peft.deterministic_generation = True
        try:
            substitutions = {
                "cached_observations": summarize_comparison(
                    predict_with_online_keys([])
                )
            }
            for key in common_keys:
                substitutions[f"online_{key}"] = summarize_comparison(
                    predict_with_online_keys([key])
                )
            substitutions["all_online_observations"] = summarize_comparison(
                predict_with_online_keys(common_keys)
            )
        finally:
            prior_with_peft.deterministic_generation = old_deterministic

        recorded_student = observation_payload.get("student_prior_tokens")
        report["online_observation_counterfactual"] = {
            "recording": str(Path(observation_path)),
            "comparison_frames": num_comparison_frames,
            "target_tokens_by_frame": comparison_target.tolist(),
            "recorded_student": (
                None
                if recorded_student is None
                else summarize_comparison(
                    recorded_student[:num_comparison_frames].long()
                )
            ),
            "substitutions": substitutions,
        }
    # Validate that the report remains JSON serializable before returning it.
    json.dumps(report)
    return report


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Example for quick inference-only config experiments.
#
#     Keep this commented out unless you are doing a local smoke test and need a
#     richer temporary override than the CLI can express. For reusable behavior,
#     put the override in an experiment file's apply_inference_overrides hook.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )
#
#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01
#
#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None,
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None,
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")
#

def apply_command_source_overrides(env_config, command_source_specs):
    """Apply inference-only task command source overrides."""
    if len(command_source_specs) == 0:
        return

    from protomotions.envs.control.target_control import (
        KeyboardTargetCommandSourceConfig,
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )

    control_components = env_config.control_components
    for spec in command_source_specs:
        if "=" in spec:
            component_name, source_name = spec.split("=", 1)
        else:
            target_components = [
                name
                for name, component in control_components.items()
                if isinstance(component, TargetControlConfig)
            ]
            if len(target_components) != 1:
                raise ValueError(
                    "Bare --command-source values require exactly one "
                    "TargetControlConfig component"
                )
            component_name = target_components[0]
            source_name = spec

        if component_name not in control_components:
            raise ValueError(
                f"Cannot override command source for unknown control component "
                f"'{component_name}'"
            )

        component_config = control_components[component_name]
        if not isinstance(component_config, TargetControlConfig):
            raise ValueError(
                f"Command source override '{component_name}={source_name}' only "
                "supports TargetControlConfig components"
            )

        source_name = source_name.lower()
        if source_name in ("keyboard", "manual", "user", "user-control"):
            component_config.command_source = KeyboardTargetCommandSourceConfig()
        elif source_name in ("random", "training"):
            component_config.command_source = RandomTargetCommandSourceConfig()
        else:
            raise ValueError(
                f"Unsupported command source '{source_name}' for component "
                f"'{component_name}'"
            )


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    if args.motion_id is not None:
        if args.num_envs != 1:
            raise ValueError("--motion-id currently requires --num-envs 1")
        if args.motion_id < 0:
            raise ValueError("--motion-id must be non-negative")
        env_config.motion_manager.subset_method = [args.motion_id]
        env_config.motion_manager.init_start_prob = 1.0
        log.info(
            "Inference override: fixed motion_id=%d at time zero", args.motion_id
        )

    if args.free_scene_objects:
        removed = []
        for name, component in list(env_config.control_components.items()):
            target = getattr(component, "_target_", "")
            if target.endswith("SceneObjectReferenceControl"):
                removed.append(name)
                del env_config.control_components[name]
        if not removed:
            log.warning(
                "--free-scene-objects requested, but the checkpoint has no "
                "SceneObjectReferenceControl component"
            )
        else:
            log.info(
                "Inference override: disabled scene reference controls %s; "
                "dynamic objects are free-running",
                removed,
            )

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # # Temporary: Enable domain randomization for local inference testing.
    # # Prefer --overrides or apply_inference_overrides for reusable changes.
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # from protomotions.robot_configs.base import ControlType
    # robot_config.control.control_type = ControlType.PROPORTIONAL

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present
        motion_lib_config.motion_file_shard_indices = None

    if args.scenes_file is not None:
        # Normalise "None"/"null" strings to actual None (disable scenes)
        scenes_file = (
            None if args.scenes_file.lower() in ("none", "null") else args.scenes_file
        )
        log.info(f"CLI override: scenes_file = {scenes_file}")
        scene_lib_config.scene_file = scenes_file
        if scenes_file is None:
            scene_lib_config.asset_root = None
        elif args.scene_asset_root is not None:
            scene_lib_config.asset_root = args.scene_asset_root
        else:
            # A CLI scene package replaces the checkpoint package, so its
            # conventional ``<asset_root>/scenes/<file>`` root must replace the
            # stale checkpoint root as well. Nonstandard layouts can pass the
            # explicit --scene-asset-root override above.
            scene_lib_config.asset_root = str(
                Path(scenes_file).resolve().parent.parent
            )
    elif args.scene_asset_root is not None:
        scene_lib_config.asset_root = args.scene_asset_root

    if args.ego_camera_file is not None:
        from protomotions.envs.component_factories import (
            load_ego_camera_trajectory_params,
        )

        component = env_config.observation_components.get(
            "ego_visible_scene_pointcloud"
        )
        if component is None:
            raise ValueError(
                "--ego-camera-file requires the ego_visible_scene_pointcloud "
                "observation component in the checkpoint config"
            )
        log.info(f"CLI override: ego_camera_file = {args.ego_camera_file}")
        component.static_params.update(
            load_ego_camera_trajectory_params(args.ego_camera_file)
        )

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    if args.command_source:
        log.info(f"CLI override: command_source = {args.command_source}")
        apply_command_source_overrides(env_config, args.command_source)

    # Automatic recordings include an exactly synchronized full reference
    # pose stream.  The viewer recorder writes it beside the rollout as
    # ``<name>.gt.motion`` for direct Viser comparison.
    if args.record_steps > 0:
        env_config.record_reference_motion = True
        # This inference-only attribute deliberately is not part of the
        # serialized simulator config contract. RecordingMixin defaults to the
        # legacy sidecar behaviour when the attribute is absent, while CLI
        # automatic recordings are compact unless explicitly requested.
        simulator_config.save_recording_sidecars = args.record_sidecars

    if args.head_translation_only:
        changed_head_condition = False
        for component_config in env_config.control_components.values():
            fixed_conditions = getattr(
                component_config, "fixed_conditioning", None
            )
            for condition in fixed_conditions or []:
                if condition.body_name == "Head":
                    condition.constraint_state = 0
                    changed_head_condition = True
        if not changed_head_condition:
            raise ValueError(
                "--head-translation-only requires a fixed Head condition in "
                "the checkpoint environment config"
            )
        log.info("Inference override: Head condition uses translation only")

    if args.head_orientation_feedback_gain:
        if not 0.0 <= args.head_orientation_feedback_gain <= 1.0:
            raise ValueError("--head-orientation-feedback-gain must be in [0, 1]")
        env_config.head_orientation_feedback_gain = args.head_orientation_feedback_gain
        log.info(
            "Inference: world Head orientation feedback gain = "
            f"{args.head_orientation_feedback_gain}"
        )

    motion_lib_config.validate()

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        # Old checkpoints may still have w_last=False from IsaacLab 2.x.
        if hasattr(simulator_config, "w_last") and not simulator_config.w_last:
            log.info(
                "Overriding w_last=False -> True for IsaacLab 3 (xyzw quaternions)"
            )
            simulator_config.w_last = True
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        if not args.headless:
            app_launcher_flags["visualizer"] = ["kit"]
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Convert friction for simulator compatibility
    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False, load_training_state=False)
    if args.fixed_motion_eval_batch_size is not None:
        agent.evaluator.config.fixed_motion_eval_batch_size = (
            args.fixed_motion_eval_batch_size
        )
    if args.eval_action_ema_alpha is not None:
        if not 0.0 < args.eval_action_ema_alpha <= 1.0:
            raise ValueError("--eval-action-ema-alpha must be in (0, 1]")
        agent.evaluator.config.eval_action_ema_alpha = args.eval_action_ema_alpha
        log.info(
            "Inference evaluator action EMA alpha: %.3f",
            args.eval_action_ema_alpha,
        )
    if args.eval_token_switch_penalty is not None:
        if args.eval_token_switch_penalty < 0.0:
            raise ValueError("--eval-token-switch-penalty must be non-negative")
        agent.evaluator.config.eval_token_switch_penalty = (
            args.eval_token_switch_penalty
        )
    if args.disable_scene_adapter:
        actor = getattr(agent.model, "_actor", None)
        scene_gate = getattr(actor, "scene_gate", None)
        if scene_gate is None:
            raise ValueError(
                "--disable-scene-adapter requires an actor with scene_gate"
            )
        with torch.no_grad():
            scene_gate.zero_()
        log.info("Inference ablation: disabled scene residual actor adapter")
    if args.policy_observation_intervention != "none":
        agent.evaluator.config.policy_observation_intervention = (
            args.policy_observation_intervention
        )
        if args.policy_observation_intervention_keys is not None:
            agent.evaluator.config.policy_observation_intervention_keys = (
                args.policy_observation_intervention_keys
            )
        log.info(
            "Inference evaluator policy intervention: %s on %s",
            args.policy_observation_intervention,
            agent.evaluator.config.policy_observation_intervention_keys,
        )
    if args.deterministic_tokens:
        actor = getattr(agent.model, "_actor", None)
        prior_with_peft = getattr(actor, "prior_with_peft", None)
        if prior_with_peft is None:
            raise ValueError(
                "--deterministic-tokens requires a PEFT discrete-prior checkpoint"
            )
        prior_with_peft.deterministic_generation = True
        log.info("Inference override: using deterministic greedy student tokens")
    if args.oracle_target_tokens:
        if not hasattr(agent.model, "collect_expert_rollout"):
            raise ValueError(
                "--oracle-target-tokens requires a checkpoint model that defines "
                "collect_expert_rollout()."
            )
        agent.evaluator.use_expert_rollout = True
        log.info("Inference override: using oracle target FSQ tokens")
    if args.oracle_takeover_step is not None:
        if args.oracle_target_tokens:
            raise ValueError(
                "--oracle-takeover-step cannot be combined with "
                "--oracle-target-tokens"
            )
        if args.oracle_takeover_step < 0:
            raise ValueError("--oracle-takeover-step must be non-negative")
        if not hasattr(agent.model, "collect_expert_rollout"):
            raise ValueError(
                "--oracle-takeover-step requires collect_expert_rollout()."
            )
        agent.evaluator.oracle_takeover_step = args.oracle_takeover_step
        log.info(
            "Inference diagnostic: student rollout followed by oracle takeover "
            "at step %d",
            args.oracle_takeover_step,
        )
    if args.record_target_tokens is not None:
        if not args.oracle_target_tokens:
            raise ValueError("--record-target-tokens requires --oracle-target-tokens")
        if args.record_steps <= 0:
            raise ValueError("--record-target-tokens requires --record-steps > 0")
        agent.evaluator.target_token_record_path = args.record_target_tokens
    if args.record_student_oracle_tokens is not None:
        if args.oracle_target_tokens:
            raise ValueError(
                "--record-student-oracle-tokens cannot be combined with "
                "--oracle-target-tokens"
            )
        if args.record_steps <= 0:
            raise ValueError(
                "--record-student-oracle-tokens requires --record-steps > 0"
            )
        agent.evaluator.student_oracle_token_record_path = (
            args.record_student_oracle_tokens
        )
    if args.recovery_push_steps:
        if args.record_steps <= 0 and not args.full_eval:
            raise ValueError(
                "--recovery-push-steps requires --record-steps > 0 or --full-eval"
            )
        if min(args.recovery_push_steps) <= 0:
            raise ValueError("--recovery-push-steps values must be positive")
        agent.evaluator.recovery_push_steps = set(args.recovery_push_steps)
        agent.evaluator.recovery_push_linear_velocity = tuple(
            args.recovery_push_linear_velocity
        )
        agent.evaluator.recovery_push_angular_velocity = tuple(
            args.recovery_push_angular_velocity
        )
        log.info(
            "Recovery diagnostic: push after steps %s with linear=%s m/s, "
            "angular=%s rad/s",
            sorted(agent.evaluator.recovery_push_steps),
            agent.evaluator.recovery_push_linear_velocity,
            agent.evaluator.recovery_push_angular_velocity,
        )
    agent.evaluator.rollout_metrics_output = args.rollout_metrics_output
    headless = getattr(env.simulator, "headless", True)
    ui = getattr(env.simulator, "user_interface", None)
    if not headless and ui is not None:
        help_text = ui.help_text()
        if help_text:
            log.info("Viewer keybinds:\n%s", help_text)

    try:
        if args.offline_token_eval_cache is not None:
            import json

            report = evaluate_offline_token_cache(
                agent,
                args.offline_token_eval_cache,
                args.offline_token_eval_observations,
            )
            rendered = json.dumps(report, indent=2)
            print(rendered)
            if args.offline_token_eval_output is not None:
                output_path = Path(args.offline_token_eval_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered + "\n", encoding="utf-8")
                log.info("Saved offline token evaluation to %s", output_path)
        elif args.full_eval:
            agent.evaluator.eval_count = 0
            evaluation_log, evaluated_score, num_eval_items = (
                agent.evaluator.evaluate()
            )

            # Print evaluation metrics
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(evaluation_log.items()):
                print(f"  {key}: {value:.6f}")
            print(f"  Items Evaluated: {num_eval_items}")
            print("=" * 60)
            if evaluated_score is not None:
                print(f"  Overall Score: {evaluated_score:.6f}")
            print("=" * 60 + "\n")
        else:
            if args.record_steps > 0:
                agent.evaluator.simple_test_policy(
                    collect_metrics=True,
                    max_steps=args.record_steps,
                    auto_record=True,
                )
            else:
                agent.evaluator.simple_test_policy(collect_metrics=True)
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
