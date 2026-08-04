from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


_prepend_import_path(SRC_ROOT)

from open_wam.configs import (
    ParallelRuntimeMode,
    load_experiment_config,
)
from open_wam.configs.enums import (
    DeadlineMissPolicy,
    FallbackHistoryPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerMode,
    RealtimeSchedulerProfile,
    RolloutArtifactProfile,
)
from open_wam.data.latent_temporal import raw_window_frames_for_latents
from open_wam.evals import libero_realtime_runtime as realtime_runtime
from open_wam.evals import libero_rollout_artifacts as rollout_artifacts
from open_wam.evals import libero_visualization as exact_viz
from open_wam.evals import (
    realtime_history,
    realtime_speculation,
)
from open_wam.integrations import (
    LiberoControlConfig,
    ensure_local_libero_config,
    libero_rollout,
    load_libero_task_init_states,
    resolve_libero_task_by_id,
)
from open_wam.integrations.realtime_contracts import PlannedControlStep
from open_wam.integrations.realtime_control import (
    build_live_rollout_summary,
)
from open_wam.integrations.realtime_plan_queue import (
    drop_control_steps_from,
    future_control_depth,
    merge_future_control_steps,
    missing_control_action_indices,
    required_control_action_indices,
)
from open_wam.integrations.realtime_scheduling import (
    frame_index_to_action_start,
    resolve_realtime_planner_mode,
    resolve_realtime_scheduler_defaults,
    should_submit_frame_grouped_planner,
    should_submit_sequence_planner,
)
from open_wam.models.policy_variants.dual_expert.runtime_routing import (
    ensure_dual_expert_inference_backend,
)
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot
from open_wam.pipelines import (
    LingbotExactRunner,
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
)
from open_wam.runtime import checkpoints as runtime_checkpoints
from open_wam.runtime import rollout as rollout_runtime
from open_wam.utils import (
    apply_config_overrides,
    merge_runtime_config_from_checkpoint,
    parse_override_assignments,
    resolve_transformer_dir_override,
    seed_everywhere,
    validate_positive_step_override,
)
from open_wam.utils.libero_paradigm import (
    require_current_libero_policy_paradigm,
)

VERBOSE = False


EVAL_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "debug_short": {},
    "libero_10hz_full": {
        "max_actions": 3000,
        "env_horizon": 5000,
        "target_action_hz": 10.0,
        "deadline_miss_policy": DeadlineMissPolicy.HOLD_STATE.value,
    },
}


def _apply_realtime_cli_profiles(args: argparse.Namespace, argv: list[str]) -> None:
    eval_defaults = EVAL_PROFILE_DEFAULTS.get(str(args.eval_profile))
    if eval_defaults is None:
        raise ValueError(f"Unsupported eval profile: {args.eval_profile!r}")
    _apply_cli_profile_defaults(
        args,
        argv,
        eval_defaults,
        flag_aliases={
            "max_actions": ("--max-actions",),
            "env_horizon": ("--env-horizon",),
            "target_action_hz": ("--target-action-hz",),
            "video_fps": ("--video-fps",),
            "deadline_miss_policy": ("--deadline-miss-policy",),
        },
    )

    try:
        scheduler_defaults = resolve_realtime_scheduler_defaults(
            args.realtime_scheduler_profile,
        )
    except ValueError as error:
        raise ValueError(
            f"Unsupported realtime scheduler profile: {args.realtime_scheduler_profile!r}"
        ) from error
    _apply_cli_profile_defaults(
        args,
        argv,
        scheduler_defaults.to_override_mapping(),
        flag_aliases={
            "planner_mode": ("--planner-mode",),
            "sequence_empty_plan_policy": ("--sequence-empty-plan-policy",),
            "fallback_history_policy": ("--fallback-history-policy",),
            "startup_open_loop_chunks": ("--startup-open-loop-chunks",),
            "replan_low_watermark_actions": ("--replan-low-watermark-actions", "--periodic-replan-frames"),
        },
    )
    args.realtime_scheduler_profile = RealtimeSchedulerProfile(args.realtime_scheduler_profile)
    args.planner_mode = RealtimePlannerMode(args.planner_mode)
    args.sequence_empty_plan_policy = RealtimeEmptyPlanPolicy(args.sequence_empty_plan_policy)
    args.fallback_history_policy = FallbackHistoryPolicy(args.fallback_history_policy)


def _apply_cli_profile_defaults(
    args: argparse.Namespace,
    argv: list[str],
    defaults: dict[str, object],
    *,
    flag_aliases: dict[str, tuple[str, ...]],
) -> None:
    for attr, value in defaults.items():
        if _cli_flag_present(argv, *flag_aliases.get(attr, ())):
            continue
        setattr(args, attr, value)


def _cli_flag_present(argv: list[str], *flags: str) -> bool:
    for token in argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one trained LIBERO policy in a fixed-rate realtime sandbox across exact/joint, "
            "feature-attached and dual-expert policy architectures."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_lingbot_exact.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file, checkpoint_step_* directory, or run directory. "
        "If omitted, exact/joint variants use `backbone.transformer_subdir`; sequence-style variants infer from that directory.",
    )
    parser.add_argument(
        "--transformer-dir",
        type=str,
        default=None,
        help=(
            "Exact/joint exported-transformer override. This intentionally does "
            "not merge checkpoint resolved_config.yaml."
        ),
    )
    parser.add_argument(
        "--pretrained-model-root",
        type=str,
        default=None,
        help=(
            "Optional reference asset root override for VAE/text/tokenizer assets. "
            "Use this with --transformer-dir when comparing against an external full-model export."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply training-style config overrides after checkpoint runtime-config merging.",
    )
    parser.add_argument(
        "--merge-checkpoint-runtime-config",
        action="store_true",
        help=(
            "Opt into merging checkpoint resolved_config.yaml before rollout. "
            "Exact/joint parallel-stream realtime rollouts skip this merge by default so "
            "`--checkpoint` remains parity-compatible with exact visualization and only "
            "uses the checkpoint to locate the exported transformer."
        ),
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument(
        "--env-horizon",
        type=int,
        default=None,
        help=(
            "Override the LIBERO/robosuite environment horizon (max env steps "
            "before the episode auto-terminates). Set this to a value >= "
            "--max-actions when running long realtime rollouts at high "
            "control rates, otherwise robosuite raises 'executing action in "
            "terminated episode' once the default horizon (typically 600) is "
            "reached. None keeps the upstream LIBERO default."
        ),
    )
    parser.add_argument("--target-action-hz", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument(
        "--eval-profile",
        choices=tuple(EVAL_PROFILE_DEFAULTS),
        default="debug_short",
        help=(
            "Named rollout defaults. `debug_short` preserves the historical short sandbox defaults. "
            "`libero_10hz_full` sets the long 10 Hz LIBERO eval protocol used by sampled evals."
        ),
    )
    parser.add_argument(
        "--write-fallback-timeline-video",
        action="store_true",
        help=(
            "Also render a debug MP4 that includes fallback-history decisions and the full fallback timeline. "
            "Disabled by default because it duplicates frame materialization and video encoding work."
        ),
    )
    parser.add_argument(
        "--artifact-profile",
        type=RolloutArtifactProfile,
        choices=tuple(RolloutArtifactProfile),
        default=RolloutArtifactProfile.STANDARD,
        help=(
            "Output artifact set. `lean` writes only the summary JSON and any explicitly requested startup debug dump; "
            "`standard` preserves the historical rollout MP4, trace JSONL files, and load report; "
            "`debug` also writes the fallback-timeline MP4."
        ),
    )
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--runtime-devices", type=str, default=None)
    parser.add_argument("--runtime-prep-device", type=str, default=None)
    parser.add_argument("--runtime-output-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--reference-assets-device-policy", type=str, choices=("cpu_offload", "runtime"), default="runtime")
    parser.add_argument(
        "--video-num-inference-steps",
        type=int,
        default=None,
        help="Optional override for inference.video_num_inference_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=None,
        help="Optional override for inference.action_num_inference_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--rollout-chunk-steps",
        type=int,
        default=None,
        help="Optional override for action_decoder.rollout_chunk_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--initial-generation-action-start",
        type=int,
        default=None,
        help=(
            "Optional initial generation action/frame start for generated-video rollouts. "
            "Defaults to the exact warmup-window length."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional override for inference.guidance_scale. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--action-guidance-scale",
        type=float,
        default=None,
        help="Optional override for inference.action_guidance_scale. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--planner-mode",
        type=RealtimePlannerMode,
        choices=tuple(RealtimePlannerMode),
        default=RealtimePlannerMode.ASYNC_BUFFER,
    )
    parser.add_argument(
        "--realtime-scheduler-profile",
        type=RealtimeSchedulerProfile,
        choices=tuple(RealtimeSchedulerProfile),
        default=RealtimeSchedulerProfile.MANUAL,
        help="Named realtime scheduler defaults; explicit low-level scheduler flags still override the profile.",
    )
    parser.add_argument("--sequence-buffer-threshold", type=int, default=3)
    parser.add_argument(
        "--sequence-empty-plan-policy",
        type=RealtimeEmptyPlanPolicy,
        choices=tuple(RealtimeEmptyPlanPolicy),
        default=RealtimeEmptyPlanPolicy.FALLBACK,
        help=(
            "Exact/joint and sequence-style variants. `fallback` preserves strict fixed-rate behavior. "
            "`wait_for_replan` blocks the sim when the next action chunk is late and executes the model plan."
        ),
    )
    parser.add_argument(
        "--startup-open-loop-chunks",
        type=int,
        default=0,
        help=(
            "Exact-runtime ablation: precompute this many model open-loop chunks before the live clock starts. "
            "This avoids fallback without using future observations, but it is not observation-conditioned replanning."
        ),
    )
    parser.add_argument(
        "--exact-startup-bootstrap-padding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Exact-runtime startup mode. The default writes observation frame 0 as prefix context, generates "
            "frames 1..4, and executes the full first 16 actions. The legacy bootstrap-padding path is "
            "deprecated because it warms synthetic zero actions."
        ),
    )
    parser.add_argument(
        "--fallback-history-policy",
        type=str,
        choices=tuple(policy.value for policy in FallbackHistoryPolicy),
        default=FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY.value,
        help=(
            "Controls whether fallback-period observations/actions are allowed back into the model history "
            "used for future replans. Freeze policies keep model time fixed while simulator time advances."
        ),
    )
    parser.add_argument(
        "--replan-low-watermark-actions",
        "--periodic-replan-frames",
        dest="replan_low_watermark_actions",
        type=int,
        default=0,
        help=(
            "Exact/joint fallback-mode ablation. The legacy alias --periodic-replan-frames accepts "
            "the same value, but K is action steps, not frames. If positive, submit a planner job "
            "when the future action buffer has at most "
            "K actions left, and accept late stale chunks only when at least K future actions remain. "
            "Use with --fallback-history-policy freeze_until_clean_chunk for low-watermark retrigger tests."
        ),
    )
    parser.add_argument(
        "--deadline-miss-policy",
        type=str,
        choices=tuple(policy.value for policy in DeadlineMissPolicy),
        default=DeadlineMissPolicy.HOLD_STATE.value,
        help=(
            "`hold_state` zeroes delta-motion channels and preserves configured absolute-tail channels; "
            "for ACTION_COMMAND gripper configs with no absolute tail, it zeroes the full action. "
            "`hold_last` repeats the previous raw action. `zero` sends all zeros."
        ),
    )
    parser.add_argument("--deadline-tolerance-ms", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_realtime_validation")
    parser.add_argument("--suffix", type=str, default="sandbox")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug-startup-dump",
        action="store_true",
        help=(
            "Write a startup-only debug JSON with initial observation/input/action hashes. "
            "This is disabled by default and does not change the model forward path."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Allow historical LIBERO policy configs that do not match the current strict fixed-128, "
            "one-frame, proprio-conditioned training/eval paradigm."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _apply_realtime_cli_profiles(args, sys.argv[1:])

    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive.")
    if args.env_horizon is not None and args.env_horizon <= 0:
        raise ValueError("--env-horizon must be positive when provided.")
    if args.target_action_hz <= 0:
        raise ValueError("--target-action-hz must be positive.")
    if args.sequence_buffer_threshold < 0:
        raise ValueError("--sequence-buffer-threshold must be non-negative.")
    if args.startup_open_loop_chunks < 0:
        raise ValueError("--startup-open-loop-chunks must be non-negative.")
    if args.replan_low_watermark_actions < 0:
        raise ValueError("--replan-low-watermark-actions must be non-negative.")

    global VERBOSE
    VERBOSE = bool(args.verbose)

    seed_everywhere(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    if args.transformer_dir is not None:
        if args.checkpoint is not None:
            raise ValueError("--transformer-dir is an exact-runtime override and cannot be combined with --checkpoint.")
        if not _is_exact_parallel_runtime(config):
            raise ValueError("--transformer-dir is only supported for exact/joint parallel-stream realtime rollouts.")
        checkpoint_path = None
        object.__setattr__(
            config.backbone,
            "transformer_subdir",
            str(resolve_transformer_dir_override(args.transformer_dir)),
        )
        checkpoint_runtime_config_path = None
    else:
        checkpoint_path = _resolve_checkpoint_path_for_config(config=config, checkpoint_arg=args.checkpoint)
        should_merge_checkpoint_runtime_config = (
            bool(args.merge_checkpoint_runtime_config)
            or not _is_exact_parallel_runtime(config)
        )
        if should_merge_checkpoint_runtime_config:
            config, checkpoint_runtime_config_path = merge_runtime_config_from_checkpoint(config, checkpoint_path)
            if checkpoint_runtime_config_path is not None and VERBOSE:
                print(
                    "[realtime_sandbox] merged checkpoint runtime config "
                    f"{checkpoint_runtime_config_path} into {config_path}",
                    file=sys.stderr,
                )
        else:
            checkpoint_runtime_config_path = None
        _apply_checkpoint_backbone_override(config, checkpoint_path=checkpoint_path)
    if args.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(tuple(args.set_overrides)),
        )
    if args.pretrained_model_root is not None:
        pretrained_model_root = Path(args.pretrained_model_root).expanduser().resolve()
        if not pretrained_model_root.is_dir():
            raise FileNotFoundError(f"--pretrained-model-root must be an existing directory: {pretrained_model_root}")
        object.__setattr__(config.backbone, "pretrained_model_name_or_path", str(pretrained_model_root))
    object.__setattr__(config.backbone, "reference_assets_device_policy", args.reference_assets_device_policy)
    rollout_runtime.apply_rollout_chunk_steps_override(
        config,
        args.rollout_chunk_steps,
    )
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source="run_libero_realtime_sandbox.py",
        allow_deprecated=bool(args.allow_deprecated_libero_config),
    )

    runtime_device = exact_viz.resolve_device(args.runtime_device)
    frontend_device = exact_viz.resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = exact_viz.resolve_device(args.decode_device, fallback=frontend_device)
    runtime_devices = rollout_runtime.resolve_runtime_devices(
        args.runtime_devices,
        fallback=runtime_device,
    )
    runtime_prep_device = exact_viz.resolve_device(args.runtime_prep_device, fallback=runtime_device)
    runtime_output_device = exact_viz.resolve_device(args.runtime_output_device, fallback=runtime_device)
    fallback_history_policy = FallbackHistoryPolicy(args.fallback_history_policy)
    exact_startup_bootstrap_padding = _resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=args.exact_startup_bootstrap_padding,
        checkpoint_path=checkpoint_path,
    )

    policy_name = str(config.policy_variant.name)
    if _is_exact_parallel_runtime(config):
        summary = _run_exact_like_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            max_actions=args.max_actions,
            env_horizon=args.env_horizon,
            target_action_hz=args.target_action_hz,
            video_fps=args.video_fps,
            planner_mode=args.planner_mode,
            deadline_miss_policy=args.deadline_miss_policy,
            deadline_tolerance_ms=args.deadline_tolerance_ms,
            output_dir=Path(args.output_dir),
            suffix=args.suffix,
            seed=args.seed,
            runtime_device=runtime_device,
            frontend_device=frontend_device,
            decode_device=decode_device,
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            startup_open_loop_chunks=args.startup_open_loop_chunks,
            sequence_empty_plan_policy=args.sequence_empty_plan_policy,
            fallback_history_policy=fallback_history_policy,
            replan_low_watermark_actions=args.replan_low_watermark_actions,
            write_fallback_timeline_video=args.write_fallback_timeline_video,
            artifact_profile=args.artifact_profile,
            debug_startup_dump=args.debug_startup_dump,
            exact_startup_bootstrap_padding=exact_startup_bootstrap_padding,
        )
    elif policy_name in {"post_latent", "post_decoded"}:
        summary = _run_sequence_policy_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            rollout_label=policy_name,
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            max_actions=args.max_actions,
            env_horizon=args.env_horizon,
            target_action_hz=args.target_action_hz,
            video_fps=args.video_fps,
            planner_mode=args.planner_mode,
            deadline_miss_policy=args.deadline_miss_policy,
            deadline_tolerance_ms=args.deadline_tolerance_ms,
            output_dir=Path(args.output_dir),
            suffix=args.suffix,
            seed=args.seed,
            runtime_device=runtime_device,
            runtime_devices=runtime_devices,
            runtime_prep_device=runtime_prep_device,
            runtime_output_device=runtime_output_device,
            frontend_device=frontend_device,
            decode_device=decode_device,
            sequence_buffer_threshold=args.sequence_buffer_threshold,
            sequence_empty_plan_policy=args.sequence_empty_plan_policy,
            fallback_history_policy=fallback_history_policy,
            startup_open_loop_chunks=args.startup_open_loop_chunks,
            replan_low_watermark_actions=args.replan_low_watermark_actions,
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            initial_generation_action_start=args.initial_generation_action_start,
            write_fallback_timeline_video=args.write_fallback_timeline_video,
            artifact_profile=args.artifact_profile,
        )
    elif policy_name == "dual_expert":
        summary = _run_sequence_policy_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            rollout_label="dual_expert",
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            max_actions=args.max_actions,
            env_horizon=args.env_horizon,
            target_action_hz=args.target_action_hz,
            video_fps=args.video_fps,
            planner_mode=args.planner_mode,
            deadline_miss_policy=args.deadline_miss_policy,
            deadline_tolerance_ms=args.deadline_tolerance_ms,
            output_dir=Path(args.output_dir),
            suffix=args.suffix,
            seed=args.seed,
            runtime_device=runtime_device,
            runtime_devices=runtime_devices,
            runtime_prep_device=runtime_prep_device,
            runtime_output_device=runtime_output_device,
            frontend_device=frontend_device,
            decode_device=decode_device,
            sequence_buffer_threshold=args.sequence_buffer_threshold,
            sequence_empty_plan_policy=args.sequence_empty_plan_policy,
            fallback_history_policy=fallback_history_policy,
            startup_open_loop_chunks=args.startup_open_loop_chunks,
            replan_low_watermark_actions=args.replan_low_watermark_actions,
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            initial_generation_action_start=args.initial_generation_action_start,
            write_fallback_timeline_video=args.write_fallback_timeline_video,
            artifact_profile=args.artifact_profile,
        )
    else:
        raise ValueError(
            "The realtime sandbox currently supports exact/joint `parallel_stream`, "
            "`post_latent`, `post_decoded`, and `dual_expert`, "
            f"got policy_variant={policy_name!r}."
        )
    print(json.dumps(summary, indent=2))


def _is_exact_parallel_runtime(config) -> bool:
    return str(config.policy_variant.name) == "parallel_stream" and config.policy_variant.runtime_mode in {
        ParallelRuntimeMode.LINGBOT_EXACT,
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    }


def _resolve_exact_startup_bootstrap_padding(
    config,
    *,
    cli_value: bool | None,
    checkpoint_path: Path | None = None,
) -> bool:
    del config, checkpoint_path
    if cli_value:
        raise ValueError(
            "`--exact-startup-bootstrap-padding` is deprecated because it can expose synthetic zero actions "
            "as model context. Use the default one-observation startup contract instead."
        )
    if cli_value is not None:
        return False
    return False


def _resolve_checkpoint_path_for_config(*, config, checkpoint_arg: str | None) -> Path | None:
    if checkpoint_arg is not None:
        return runtime_checkpoints.resolve_checkpoint_file(Path(checkpoint_arg))
    transformer_subdir = getattr(config.backbone, "transformer_subdir", None)
    if transformer_subdir is None:
        return None
    try:
        return runtime_checkpoints.resolve_checkpoint_file(
            runtime_checkpoints.resolve_checkpoint_step_dir_from_transformer_dir(
                str(transformer_subdir)
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        if VERBOSE:
            print(
                "[realtime_sandbox] Failed to infer a checkpoint file from "
                f"backbone.transformer_subdir={transformer_subdir!r}: {exc}",
                file=sys.stderr,
            )
        return None


def _apply_checkpoint_backbone_override(config, *, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_step_dir = checkpoint_path.parent
    transformer_dir = checkpoint_step_dir / "transformer"
    if _is_usable_transformer_dir(transformer_dir):
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))


def _is_usable_transformer_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _apply_common_inference_overrides(
    config,
    *,
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
    guidance_scale: float | None,
    action_guidance_scale: float | None,
) -> None:
    video_steps = validate_positive_step_override(
        "video_num_inference_steps",
        video_num_inference_steps,
    )
    action_steps = validate_positive_step_override(
        "action_num_inference_steps",
        action_num_inference_steps,
    )
    if video_steps is not None:
        object.__setattr__(config.inference, "video_num_inference_steps", video_steps)
    if action_steps is not None:
        object.__setattr__(config.inference, "action_num_inference_steps", action_steps)
    if guidance_scale is not None:
        object.__setattr__(config.inference, "guidance_scale", float(guidance_scale))
    if action_guidance_scale is not None:
        object.__setattr__(config.inference, "action_guidance_scale", float(action_guidance_scale))


def _construct_realtime_libero_env(task_spec, *, env_horizon: int | None):
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    count = 0
    env = None
    while env is None and count < 5:
        try:
            kwargs: dict[str, Any] = {
                "bddl_file_name": task_spec.bddl_file_path,
                "camera_heights": 128,
                "camera_widths": 128,
            }
            if env_horizon is not None:
                kwargs["horizon"] = int(env_horizon)
            env = OffScreenRenderEnv(**kwargs)
        except Exception as exc:
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _run_exact_like_realtime_rollout(
    *,
    config,
    checkpoint_path: Path | None,
    benchmark: str,
    task_id: int,
    episode_idx: int,
    max_actions: int,
    env_horizon: int | None,
    target_action_hz: float,
    video_fps: float | None,
    planner_mode: RealtimePlannerMode | str,
    deadline_miss_policy: str,
    deadline_tolerance_ms: float,
    output_dir: Path,
    suffix: str,
    seed: int,
    runtime_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
    guidance_scale: float | None,
    action_guidance_scale: float | None,
    startup_open_loop_chunks: int,
    sequence_empty_plan_policy: RealtimeEmptyPlanPolicy | str,
    fallback_history_policy: FallbackHistoryPolicy,
    replan_low_watermark_actions: int,
    write_fallback_timeline_video: bool,
    artifact_profile: RolloutArtifactProfile | str,
    debug_startup_dump: bool,
    exact_startup_bootstrap_padding: bool,
) -> dict[str, Any]:
    planner_mode = RealtimePlannerMode(planner_mode)
    sequence_empty_plan_policy = RealtimeEmptyPlanPolicy(sequence_empty_plan_policy)
    replan_low_watermark_actions = int(replan_low_watermark_actions)
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.eval()
    runner = LingbotExactRunner(pipeline)
    realtime_runtime.apply_inference_overrides(
        runner,
        video_num_inference_steps=video_num_inference_steps,
        action_num_inference_steps=action_num_inference_steps,
        guidance_scale=guidance_scale,
        action_guidance_scale=action_guidance_scale,
    )
    runner.pipeline.visual_tower.ensure_runtime_backbone_device(
        action_dim=config.action_decoder.action_dim,
        device=runtime_device,
    )
    component_report = {
        "pipeline": "open_wam_exact",
        "policy_variant": str(config.policy_variant.name),
        "runtime_mode": str(config.policy_variant.runtime_mode),
        "checkpoint_file": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "transformer_dir": str(config.backbone.transformer_subdir),
        "runtime_device": str(runtime_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "video_steps": int(runner.policy_variant.inference_config.video_num_inference_steps),
        "action_steps": int(runner.policy_variant.inference_config.action_num_inference_steps),
        "guidance_scale": float(runner.policy_variant.inference_config.guidance_scale),
        "action_guidance_scale": float(runner.policy_variant.inference_config.action_guidance_scale),
        "video_steps_override": video_num_inference_steps,
        "action_steps_override": action_num_inference_steps,
        "guidance_scale_override": guidance_scale,
        "action_guidance_scale_override": action_guidance_scale,
        "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
        "sequence_empty_plan_policy": sequence_empty_plan_policy.value,
        "fallback_history_policy": str(fallback_history_policy),
        "replan_low_watermark_actions": int(replan_low_watermark_actions),
        "periodic_replan_frames": int(replan_low_watermark_actions),
        "startup_seed": int(seed),
        "exact_startup_bootstrap_padding": bool(exact_startup_bootstrap_padding),
    }

    task_spec, prompt = exact_viz.resolve_task_spec(benchmark, task_id)
    init_states = load_libero_task_init_states(task_spec)
    env = _construct_realtime_libero_env(task_spec, env_horizon=env_horizon)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    action_per_frame = int(config.policy_variant.action_per_frame)
    action_dim = int(config.data.action_schema.action_dim)
    fallback_absolute_tail_start = realtime_history.fallback_absolute_tail_start(config)
    action_period_s = 1.0 / float(target_action_hz)
    deadline_tolerance_s = float(deadline_tolerance_ms) / 1000.0
    max_frames = int(math.ceil(max_actions / action_per_frame))

    try:
        first_raw_obs = exact_viz.initialize_raw_observation(env, init_states[episode_idx % len(init_states)])
        first_obs = exact_viz.extract_observation(first_raw_obs)
        latest_proprio_state = exact_viz.extract_proprio_context_tensor(
            first_raw_obs,
            config=config,
            device=runtime_device,
        )
        startup_debug_report: dict[str, Any] | None = None
        startup_warmup_s = 0.0
        startup_history_video_latents: torch.Tensor | None = None
        startup_history_raw_actions: np.ndarray | None = None
        startup_history_frame_index = 0
        with torch.inference_mode():
            session = runner.reset(task_text=(prompt,))
            # Preserve the exact parallel-stream contract: reseed before each chunk.
            with realtime_runtime.isolated_torch_rng(seed, frontend_device, runtime_device):
                startup_prepare_t0 = time.perf_counter()
                if VERBOSE:
                    print(
                        "[exact_startup] prepare_frontend "
                        f"bootstrap_padding={exact_startup_bootstrap_padding}",
                        flush=True,
                    )
                initial_inputs = exact_viz.prepare_exact_runtime_inputs(
                    runner,
                    views=exact_viz.observations_to_views([first_obs], device=frontend_device),
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                )
                realtime_runtime.synchronize_devices(frontend_device, runtime_device)
                startup_prepare_s = time.perf_counter() - startup_prepare_t0
                if VERBOSE:
                    print(
                        "[exact_startup] prepared "
                        f"video_latents_shape={tuple(initial_inputs['video_latents'].shape)} "
                        f"elapsed_s={startup_prepare_s:.3f}",
                        flush=True,
                    )

                rng_before_startup_infer = rollout_artifacts.capture_torch_rng_debug_state()
                startup_history_video_latents = initial_inputs["video_latents"]
                startup_infer_session = session
                startup_infer_t0 = time.perf_counter()
                if VERBOSE:
                    print("[exact_startup] infer_first_chunk", flush=True)
                first_chunk = runner.infer_chunk(
                    session=startup_infer_session,
                    video_latents=initial_inputs["video_latents"],
                    text_context=initial_inputs["text_context"],
                    negative_text_context=initial_inputs["negative_text_context"],
                    proprio_state=latest_proprio_state,
                )
                realtime_runtime.synchronize_devices(runtime_device)
                startup_infer_s = time.perf_counter() - startup_infer_t0
                if VERBOSE:
                    print(
                        "[exact_startup] infer_done "
                        f"elapsed_s={startup_infer_s:.3f} debug={first_chunk.debug}",
                        flush=True,
                    )
                if (
                    startup_history_raw_actions is None
                    and int(first_chunk.debug.get("generation_frame_start", 0)) > startup_history_frame_index
                ):
                    startup_history_raw_actions = np.zeros((action_per_frame, action_dim), dtype=np.float32)
                if debug_startup_dump:
                    startup_debug_report = rollout_artifacts.build_libero_exact_startup_debug_report(
                        options=rollout_artifacts.LiberoExactStartupDebugOptions(
                            prompt=prompt,
                            seed=seed,
                            runtime_device=runtime_device,
                            frontend_device=frontend_device,
                            decode_device=decode_device,
                            reference_assets_device_policy=str(
                                config.backbone.reference_assets_device_policy
                            ),
                            runtime_mode=str(config.policy_variant.runtime_mode),
                            video_num_inference_steps=int(
                                config.inference.video_num_inference_steps
                            ),
                            action_num_inference_steps=int(
                                config.inference.action_num_inference_steps
                            ),
                            guidance_scale=float(config.inference.guidance_scale),
                            action_guidance_scale=float(
                                config.inference.action_guidance_scale
                            ),
                            frame_chunk_size=int(config.inference.frame_chunk_size),
                            action_per_frame=int(config.policy_variant.action_per_frame),
                            exact_startup_bootstrap_padding=exact_startup_bootstrap_padding,
                        ),
                        payload=rollout_artifacts.LiberoExactStartupDebugPayload(
                            first_observation=first_obs,
                            video_latents=initial_inputs.get("video_latents"),
                            text_context=initial_inputs.get("text_context"),
                            negative_text_context=initial_inputs.get(
                                "negative_text_context"
                            ),
                            session_text_context=getattr(session, "text_context", None),
                            session_negative_text_context=getattr(
                                session,
                                "negative_text_context",
                                None,
                            ),
                            rng_before_startup_infer=rng_before_startup_infer,
                            rng_after_startup_infer=(
                                rollout_artifacts.capture_torch_rng_debug_state()
                            ),
                            first_chunk_debug=first_chunk.debug,
                            chunk_action_pred=first_chunk.chunk_action_pred,
                            raw_chunk_action_pred=first_chunk.raw_chunk_action_pred,
                            predicted_latents=first_chunk.predicted_latents,
                        ),
                    )

        history_base_session, current_chunk_session, buffer_tail_session = realtime_runtime.resolve_exact_startup_sessions(
            config=config,
            startup_session=session,
            first_chunk=first_chunk,
            frame_chunk_size=int(config.inference.frame_chunk_size),
        )
        plan_by_action: dict[int, PlannedControlStep] = merge_future_control_steps(
            {},
            realtime_runtime.exact_chunk_to_planned_steps(
                chunk=first_chunk,
                action_per_frame=action_per_frame,
                frame_chunk_size=int(config.inference.frame_chunk_size),
                source="startup_plan",
                ready_monotonic_s=time.perf_counter(),
            ),
            next_action_to_execute=0,
        )
        pending_history: list[dict[str, Any]] = [
            realtime_runtime.build_exact_startup_conditioning_history_record(
                chunk=first_chunk,
                initial_video_latents=(
                    startup_history_video_latents
                    if startup_history_video_latents is not None
                    else initial_inputs["video_latents"]
                ),
                initial_obs=first_obs,
                action_per_frame=action_per_frame,
                frame_chunk_size=int(config.inference.frame_chunk_size),
                conditioning_frame_index=startup_history_frame_index,
                raw_actions_override=startup_history_raw_actions,
                proprio_state=latest_proprio_state,
            )
        ]
        fallback_history_state = realtime_history.FrameFallbackHistoryState(policy=fallback_history_policy)

        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
        collect_video_records = rollout_artifacts.RolloutArtifactPolicy.from_value(
            artifact_profile,
            write_fallback_timeline_video=write_fallback_timeline_video,
        ).collects_video_records
        replan_records: list[dict[str, Any]] = []
        extension_records: list[dict[str, Any]] = []
        startup_open_loop_s = 0.0
        if startup_open_loop_chunks > 0:
            startup_open_loop_t0 = time.perf_counter()
            for _ in range(int(startup_open_loop_chunks)):
                if buffer_tail_session is None:
                    break
                extension_result = realtime_runtime.run_extension_job(
                    runner=runner,
                    session=buffer_tail_session,
                    config=config,
                    runtime_device=runtime_device,
                    job_seed=realtime_runtime.job_seed_for_session(seed, buffer_tail_session),
                )
                application = realtime_runtime.apply_frame_planner_result(
                    extension_result,
                    config=config,
                    plan_by_action=plan_by_action,
                    next_action_to_execute=0,
                    pending_history=pending_history,
                    history_base_session=history_base_session,
                    current_chunk_session=current_chunk_session,
                    buffer_tail_session=buffer_tail_session,
                    replan_records=replan_records,
                    extension_records=extension_records,
                    min_future_actions_to_accept_stale_chunk=replan_low_watermark_actions,
                )
                history_base_session = application.history_base_session
                current_chunk_session = application.current_chunk_session
                buffer_tail_session = application.buffer_tail_session
                plan_by_action = application.plan_by_action
                pending_history = application.pending_history
            startup_open_loop_s = time.perf_counter() - startup_open_loop_t0
            startup_infer_s += startup_open_loop_s
        startup_infer_s += startup_warmup_s
        done = False
        current_obs = first_obs
        last_action = np.zeros((action_dim,), dtype=np.float32)
        executed_action_index = 0
        next_action_index = 0
        live_start_monotonic = time.perf_counter()
        last_action_end_monotonic = live_start_monotonic
        skipped_replan_submissions = 0
        wait_for_plan_count = 0
        wait_for_plan_total_s = 0.0
        blocking_replan_count = 0
        schedule_pause_s = 0.0
        fallback_invalidated_future_actions = 0
        fallback_invalidated_buffer_count = 0
        freeze_model_timeline_on_fallback = realtime_history.fallback_policy_freezes_model_timeline(fallback_history_policy)
        hidden_fallback_period_active = False
        hidden_fallback_period_count = 0
        periodic_replan_submit_count = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            replan_future: Future[
                realtime_runtime.FramePlannerJobResult
            ] | None = None
            replan_future_cache_snapshot: VisualRuntimeStateSnapshot | None = None
            next_frame_to_execute = 1
            next_real_frame_to_execute = 1
            while next_real_frame_to_execute <= max_frames and executed_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    replan_result = realtime_speculation.resolve_future_result(
                        replan_future,
                        runner=runner,
                        snapshot=replan_future_cache_snapshot,
                    )
                    application = realtime_runtime.apply_frame_planner_result(
                        replan_result,
                        config=config,
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        history_base_session=history_base_session,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                        min_future_actions_to_accept_stale_chunk=replan_low_watermark_actions,
                    )
                    history_base_session = application.history_base_session
                    current_chunk_session = application.current_chunk_session
                    buffer_tail_session = application.buffer_tail_session
                    plan_by_action = application.plan_by_action
                    pending_history = application.pending_history
                    realtime_speculation.restore_visual_runtime_if_rejected(
                        replan_result,
                        runner=runner,
                        snapshot=replan_future_cache_snapshot,
                    )
                    replan_future = None
                    replan_future_cache_snapshot = None

                required_action_indices = required_control_action_indices(
                    next_action_index=next_action_index,
                    max_actions=max_actions,
                    action_per_frame=action_per_frame,
                )
                wait_for_plan_s = 0.0
                if (
                    sequence_empty_plan_policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
                    and missing_control_action_indices(plan_by_action, required_action_indices)
                ):
                    wait_t0 = time.perf_counter()
                    wait_job_count = 0
                    max_wait_jobs = max(4, max_frames + 2)
                    while missing_control_action_indices(plan_by_action, required_action_indices):
                        if wait_job_count >= max_wait_jobs:
                            raise RuntimeError(
                                "Blocking exact/joint replan did not produce the next required actions "
                                f"{required_action_indices}; missing="
                                f"{missing_control_action_indices(plan_by_action, required_action_indices)}."
                            )
                        wait_job_count += 1
                        if replan_future is None:
                            blocking_replan_count += 1
                            future_buffer_depth_frames = int(
                                math.ceil(
                                    future_control_depth(
                                        plan_by_action,
                                        next_action_to_execute=next_action_index,
                                    )
                                    / action_per_frame
                                )
                            )
                            (
                                replan_future,
                                replan_future_cache_snapshot,
                            ) = realtime_runtime.submit_planner_job_with_snapshot(
                                executor=executor,
                                planner_mode=resolve_realtime_planner_mode(
                                    planner_mode=planner_mode,
                                    empty_plan_policy=sequence_empty_plan_policy,
                                    has_history=bool(pending_history),
                                ),
                                pending_history=pending_history,
                                future_buffer_depth=future_buffer_depth_frames,
                                runner=runner,
                                history_base_session=history_base_session,
                                current_chunk_session=current_chunk_session,
                                prompt=prompt,
                                config=config,
                                frontend_device=frontend_device,
                                runtime_device=runtime_device,
                                buffer_tail_session=buffer_tail_session,
                                seed_base=seed,
                            )
                        result_cache_snapshot: VisualRuntimeStateSnapshot | None = None
                        if replan_future is None:
                            result_cache_snapshot = realtime_speculation.snapshot_visual_runtime(
                                runner=runner,
                                config=config,
                                session=current_chunk_session,
                            )
                            if pending_history:
                                history_payload = [
                                    realtime_runtime.copy_history_record_for_worker(record)
                                    for record in pending_history
                                ]
                                try:
                                    result = realtime_runtime.run_replan_job(
                                        runner=runner,
                                        session=history_base_session,
                                        prompt=prompt,
                                        history_records=history_payload,
                                        config=config,
                                        frontend_device=frontend_device,
                                        runtime_device=runtime_device,
                                        job_seed=realtime_runtime.job_seed_for_session(seed, current_chunk_session),
                                    )
                                except BaseException:
                                    realtime_speculation.restore_visual_runtime(
                                        runner=runner,
                                        snapshot=result_cache_snapshot,
                                    )
                                    raise
                            elif buffer_tail_session is not None:
                                try:
                                    result = realtime_runtime.run_extension_job(
                                        runner=runner,
                                        session=buffer_tail_session,
                                        config=config,
                                        runtime_device=runtime_device,
                                        job_seed=realtime_runtime.job_seed_for_session(seed, buffer_tail_session),
                                    )
                                except BaseException:
                                    realtime_speculation.restore_visual_runtime(
                                        runner=runner,
                                        snapshot=result_cache_snapshot,
                                    )
                                    raise
                            else:
                                raise RuntimeError(
                                    "Exact/joint wait-for-replan mode has no pending future, history, or buffer "
                                    f"session for required actions {required_action_indices}."
                                )
                        else:
                            result = realtime_speculation.resolve_future_result(
                                replan_future,
                                runner=runner,
                                snapshot=replan_future_cache_snapshot,
                            )
                            replan_future = None
                            result_cache_snapshot = replan_future_cache_snapshot
                            replan_future_cache_snapshot = None
                        wait_for_plan_s = time.perf_counter() - wait_t0
                        result.trace["blocking_wait_action_index"] = int(
                            next_action_index
                        )
                        result.trace["blocking_wait_frame_index"] = int(
                            next_frame_to_execute
                        )
                        result.trace["blocking_wait_s"] = float(wait_for_plan_s)
                        application = realtime_runtime.apply_frame_planner_result(
                            result,
                            config=config,
                            plan_by_action=plan_by_action,
                            next_action_to_execute=next_action_index,
                            pending_history=pending_history,
                            history_base_session=history_base_session,
                            current_chunk_session=current_chunk_session,
                            buffer_tail_session=buffer_tail_session,
                            replan_records=replan_records,
                            extension_records=extension_records,
                            min_future_actions_to_accept_stale_chunk=replan_low_watermark_actions,
                        )
                        history_base_session = application.history_base_session
                        current_chunk_session = application.current_chunk_session
                        buffer_tail_session = application.buffer_tail_session
                        plan_by_action = application.plan_by_action
                        pending_history = application.pending_history
                        realtime_speculation.restore_visual_runtime_if_rejected(
                            result,
                            runner=runner,
                            snapshot=result_cache_snapshot,
                        )
                    wait_for_plan_count += 1
                    wait_for_plan_total_s += wait_for_plan_s

                frame_model_action_start = int(next_action_index)
                frame_actions: list[np.ndarray] = []
                frame_action_sources: list[str] = []
                frame_action_metadata: list[dict[str, Any]] = []
                missing_required_action_indices = missing_control_action_indices(
                    plan_by_action,
                    required_action_indices,
                )
                use_fallback_frame = (
                    sequence_empty_plan_policy == RealtimeEmptyPlanPolicy.FALLBACK
                    and bool(missing_required_action_indices)
                )
                if use_fallback_frame and freeze_model_timeline_on_fallback and not hidden_fallback_period_active:
                    hidden_fallback_period_active = True
                    hidden_fallback_period_count += 1
                if not use_fallback_frame and hidden_fallback_period_active:
                    hidden_fallback_period_active = False
                for action_offset in range(action_per_frame):
                    if executed_action_index + action_offset >= max_actions:
                        break
                    if use_fallback_frame:
                        raw_action = realtime_runtime.build_fallback_frame_actions(
                            action_dim=action_dim,
                            action_per_frame=1,
                            policy=deadline_miss_policy,
                            last_action=last_action,
                            preserve_absolute_tail_from=fallback_absolute_tail_start,
                        )[0]
                        source = f"fallback_{deadline_miss_policy}"
                        generation_frame_start = None
                        planner_step_index = None
                        ready_monotonic_s = None
                    else:
                        planned_step = plan_by_action.pop(frame_model_action_start + action_offset, None)
                        if planned_step is None:
                            if sequence_empty_plan_policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN:
                                raise RuntimeError(
                                    "Exact/joint wait-for-replan mode exhausted without action "
                                    f"{frame_model_action_start + action_offset}."
                                )
                            raise ValueError(f"Unsupported sequence_empty_plan_policy={sequence_empty_plan_policy!r}.")
                        if planned_step.raw_action is None:
                            raise RuntimeError("Exact/joint plan did not contain raw LIBERO actions.")
                        raw_action = np.array(planned_step.raw_action, copy=True)
                        source = str(planned_step.source)
                        generation_frame_start = planned_step.generation_frame_start
                        planner_step_index = planned_step.planner_step_index
                        ready_monotonic_s = planned_step.ready_monotonic_s
                    frame_actions.append(raw_action)
                    frame_action_sources.append(str(source))
                    frame_action_metadata.append(
                        {
                            "source": str(source),
                            "generation_frame_start": generation_frame_start,
                            "planner_step_index": planner_step_index,
                            "ready_monotonic_s": ready_monotonic_s,
                        }
                    )

                if not frame_actions:
                    break

                schedule_pause_s += wait_for_plan_s
                frame_obs_sequence: list[dict[str, np.ndarray]] = []
                for action_offset, action in enumerate(frame_actions):
                    real_action_index = int(executed_action_index)
                    model_action_index = int(frame_model_action_start + action_offset)
                    action_metadata = frame_action_metadata[action_offset]
                    generation_frame_start = action_metadata["generation_frame_start"]
                    planner_step_index = action_metadata["planner_step_index"]
                    ready_monotonic_s = action_metadata["ready_monotonic_s"]
                    source = str(action_metadata["source"])
                    scheduled_monotonic = live_start_monotonic + real_action_index * action_period_s + schedule_pause_s
                    now = time.perf_counter()
                    if now < scheduled_monotonic:
                        time.sleep(scheduled_monotonic - now)
                    actual_start_monotonic = time.perf_counter()
                    lateness_s = max(0.0, actual_start_monotonic - scheduled_monotonic)
                    obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                    action_end_monotonic = time.perf_counter()
                    env_step_s = action_end_monotonic - actual_start_monotonic
                    extracted_obs = exact_viz.extract_observation(obs)
                    latest_proprio_state = exact_viz.extract_proprio_context_tensor(
                        obs,
                        config=config,
                        device=runtime_device,
                    )
                    last_action_end_monotonic = action_end_monotonic
                    frame_obs_sequence.append(
                        {key: np.array(value, copy=True) for key, value in extracted_obs.items()}
                    )
                    last_action = np.array(action, copy=True)
                    current_obs = extracted_obs
                    generation_action_start = (
                        None
                        if generation_frame_start is None
                        else frame_index_to_action_start(generation_frame_start, action_per_frame)
                    )
                    action_record = {
                        "action_index": real_action_index,
                        "absolute_action_index": real_action_index,
                        "absolute_frame_index": int(next_real_frame_to_execute),
                        "action_offset": int(action_offset),
                        "model_action_index": model_action_index,
                        "model_frame_index": int(next_frame_to_execute),
                        "model_action_offset": int(action_offset),
                        "source": source,
                        "scheduled_start_s": float(scheduled_monotonic - live_start_monotonic),
                        "actual_start_s": float(actual_start_monotonic - live_start_monotonic),
                        "lateness_s": float(lateness_s),
                        "env_step_s": float(env_step_s),
                        "generation_action_start": generation_action_start,
                        "generation_lag_actions": (
                            None
                            if generation_action_start is None
                            else int(model_action_index - generation_action_start)
                        ),
                        "generation_frame_start": generation_frame_start,
                        "generation_lag_frames": (
                            None
                            if generation_frame_start is None
                            else int(next_frame_to_execute - generation_frame_start)
                        ),
                        "planner_step_index": planner_step_index,
                        "action": np.asarray(action, dtype=np.float32).tolist(),
                        "plan_ready_delay_s": (
                            None
                            if ready_monotonic_s is None
                            else float(actual_start_monotonic - ready_monotonic_s)
                        ),
                        "wait_for_plan_s": float(wait_for_plan_s),
                    }
                    action_records.append(action_record)
                    if collect_video_records:
                        action_video_records.append(
                            {
                                **action_record,
                                "obs": {key: np.array(value, copy=True) for key, value in extracted_obs.items()},
                            }
                        )
                    if VERBOSE and real_action_index % 50 == 0:
                        _print_stage(
                            "exact_like_action_progress",
                            action_index=real_action_index,
                            model_action_index=model_action_index,
                            source=source,
                            done=bool(done),
                        )
                    executed_action_index += 1
                    if realtime_history.action_advances_model_timeline(
                        source,
                        fallback_history_policy=fallback_history_policy,
                    ):
                        next_action_index += 1
                    if done or executed_action_index >= max_actions:
                        break

                if done or executed_action_index >= max_actions:
                    break

                history_decision = realtime_history.append_frame_history_record(
                    pending_history=pending_history,
                    state=fallback_history_state,
                    absolute_frame_index=int(next_frame_to_execute),
                    current_obs=current_obs,
                    proprio_state=latest_proprio_state,
                    frame_obs_sequence=frame_obs_sequence,
                    frame_actions=frame_actions,
                    frame_action_sources=frame_action_sources,
                    frame_chunk_size=int(config.inference.frame_chunk_size),
                )
                frame_contains_fallback = realtime_history.frame_contains_fallback_action(frame_action_sources)
                for record in action_records[-len(frame_actions) :]:
                    record["frame_history_decision"] = history_decision.value
                    record["frame_contains_fallback_action"] = bool(frame_contains_fallback)
                    record["frame_action_sources"] = [str(source) for source in frame_action_sources]
                for record in action_video_records[-len(frame_actions) :]:
                    record["frame_history_decision"] = history_decision.value
                    record["frame_contains_fallback_action"] = bool(frame_contains_fallback)
                    record["frame_action_sources"] = [str(source) for source in frame_action_sources]
                if frame_contains_fallback and not freeze_model_timeline_on_fallback:
                    fallback_invalidated_future_actions += len(plan_by_action)
                    plan_by_action = {}
                    if buffer_tail_session is not None:
                        fallback_invalidated_buffer_count += 1
                    buffer_tail_session = None

                if replan_future is not None and replan_future.done():
                    replan_result = realtime_speculation.resolve_future_result(
                        replan_future,
                        runner=runner,
                        snapshot=replan_future_cache_snapshot,
                    )
                    application = realtime_runtime.apply_frame_planner_result(
                        replan_result,
                        config=config,
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        history_base_session=history_base_session,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                        min_future_actions_to_accept_stale_chunk=replan_low_watermark_actions,
                    )
                    history_base_session = application.history_base_session
                    current_chunk_session = application.current_chunk_session
                    buffer_tail_session = application.buffer_tail_session
                    plan_by_action = application.plan_by_action
                    pending_history = application.pending_history
                    realtime_speculation.restore_visual_runtime_if_rejected(
                        replan_result,
                        runner=runner,
                        snapshot=replan_future_cache_snapshot,
                    )
                    replan_future = None
                    replan_future_cache_snapshot = None

                future_buffer_depth_frames = int(
                    math.ceil(
                        future_control_depth(plan_by_action, next_action_to_execute=next_action_index)
                        / action_per_frame
                    )
                )
                future_buffer_depth_actions = future_control_depth(
                    plan_by_action,
                    next_action_to_execute=next_action_index,
                )
                should_submit_replan = should_submit_frame_grouped_planner(
                    future_buffer_depth_actions=future_buffer_depth_actions,
                    future_buffer_depth_frames=future_buffer_depth_frames,
                    empty_plan_policy=sequence_empty_plan_policy,
                    replan_low_watermark_actions=replan_low_watermark_actions,
                )
                if replan_future is None and should_submit_replan:
                    (
                        replan_future,
                        replan_future_cache_snapshot,
                    ) = realtime_runtime.submit_planner_job_with_snapshot(
                        executor=executor,
                        planner_mode=resolve_realtime_planner_mode(
                            planner_mode=planner_mode,
                            empty_plan_policy=sequence_empty_plan_policy,
                            has_history=bool(pending_history),
                        ),
                        pending_history=pending_history,
                        future_buffer_depth=future_buffer_depth_frames,
                        runner=runner,
                        history_base_session=history_base_session,
                        current_chunk_session=current_chunk_session,
                        prompt=prompt,
                        config=config,
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        buffer_tail_session=buffer_tail_session,
                        seed_base=seed,
                    )
                    if replan_future is not None:
                        if replan_low_watermark_actions > 0:
                            periodic_replan_submit_count += 1
                else:
                    skipped_replan_submissions += 1
                next_real_frame_to_execute += 1
                if not (frame_contains_fallback and freeze_model_timeline_on_fallback):
                    next_frame_to_execute += 1

            if replan_future is not None and replan_future.done():
                replan_result = realtime_speculation.resolve_future_result(
                    replan_future,
                    runner=runner,
                    snapshot=replan_future_cache_snapshot,
                )
                application = realtime_runtime.apply_frame_planner_result(
                    replan_result,
                    config=config,
                    plan_by_action=plan_by_action,
                    next_action_to_execute=next_action_index,
                    pending_history=pending_history,
                    history_base_session=history_base_session,
                    current_chunk_session=current_chunk_session,
                    buffer_tail_session=buffer_tail_session,
                    replan_records=replan_records,
                    extension_records=extension_records,
                    min_future_actions_to_accept_stale_chunk=replan_low_watermark_actions,
                )
                history_base_session = application.history_base_session
                current_chunk_session = application.current_chunk_session
                buffer_tail_session = application.buffer_tail_session
                plan_by_action = application.plan_by_action
                pending_history = application.pending_history
                realtime_speculation.restore_visual_runtime_if_rejected(
                    replan_result,
                    runner=runner,
                    snapshot=replan_future_cache_snapshot,
                )

        live_wall_time_s = last_action_end_monotonic - live_start_monotonic if executed_action_index > 0 else 0.0
        summary = build_live_rollout_summary(
            action_records=action_records,
            replan_records=replan_records,
            target_action_hz=target_action_hz,
            live_wall_time_s=live_wall_time_s,
            startup_prepare_s=startup_prepare_s,
            startup_infer_s=startup_infer_s,
            deadline_tolerance_s=deadline_tolerance_s,
        )
        stale_replan_actions = sum(int(record.get("stale_planned_actions", 0)) for record in replan_records)
        stale_extension_actions = sum(int(record.get("stale_planned_actions", 0)) for record in extension_records)
        chunk_boundary_dropped_replan_actions = sum(
            int(record.get("chunk_boundary_dropped_actions", 0)) for record in replan_records
        )
        chunk_boundary_dropped_extension_actions = sum(
            int(record.get("chunk_boundary_dropped_actions", 0)) for record in extension_records
        )
        summary.update(
            {
                "benchmark": benchmark,
                "task_id": int(task_id),
                "prompt": prompt,
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "max_actions": int(max_actions),
                "env_horizon": None if env_horizon is None else int(env_horizon),
                "executed_actions": int(executed_action_index),
                "model_executed_actions": int(next_action_index),
                "model_executed_frames": int(max(0, next_frame_to_execute - 1)),
                "env_timestep": int(env.env.timestep),
                "seed": int(seed),
                "runtime_device": str(runtime_device),
                "frontend_device": str(frontend_device),
                "decode_device": str(decode_device),
                "checkpoint_file": None if checkpoint_path is None else str(checkpoint_path.resolve()),
                "transformer_dir": str(config.backbone.transformer_subdir),
                "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
                "video_num_inference_steps": int(runner.policy_variant.inference_config.video_num_inference_steps),
                "action_num_inference_steps": int(runner.policy_variant.inference_config.action_num_inference_steps),
                "guidance_scale": float(runner.policy_variant.inference_config.guidance_scale),
                "action_guidance_scale": float(runner.policy_variant.inference_config.action_guidance_scale),
                "planner_mode": planner_mode.value,
                "deadline_miss_policy": deadline_miss_policy,
                "fallback_absolute_tail_start": fallback_absolute_tail_start,
                "sequence_empty_plan_policy": sequence_empty_plan_policy.value,
                "fallback_history_policy": str(fallback_history_policy),
                "replan_low_watermark_actions": int(replan_low_watermark_actions),
                "periodic_replan_frames": int(replan_low_watermark_actions),
                "periodic_replan_submit_count": int(periodic_replan_submit_count),
                "startup_open_loop_chunks": int(startup_open_loop_chunks),
                "startup_open_loop_s": float(startup_open_loop_s),
                "startup_warmup_s": float(startup_warmup_s),
                "exact_startup_bootstrap_padding": bool(exact_startup_bootstrap_padding),
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "wait_for_plan_count": int(wait_for_plan_count),
                "wait_for_plan_total_s": float(wait_for_plan_total_s),
                "schedule_pause_s": float(schedule_pause_s),
                "blocking_replan_count": int(blocking_replan_count),
                "history_replan_count": len(replan_records),
                "open_loop_extension_count": len(extension_records),
                "stale_replan_planned_actions": int(stale_replan_actions),
                "stale_extension_planned_actions": int(stale_extension_actions),
                "chunk_boundary_dropped_replan_actions": int(chunk_boundary_dropped_replan_actions),
                "chunk_boundary_dropped_extension_actions": int(chunk_boundary_dropped_extension_actions),
                "fallback_invalidated_future_actions": int(fallback_invalidated_future_actions),
                "fallback_invalidated_buffer_count": int(fallback_invalidated_buffer_count),
                "hidden_fallback_period_count": int(hidden_fallback_period_count),
                "hidden_history_frames": int(fallback_history_state.hidden_history_frames),
                "hidden_history_raw_observations": int(fallback_history_state.hidden_history_raw_observations),
                "hidden_fallback_history_frames": int(fallback_history_state.hidden_fallback_frames),
                "hidden_fallback_history_raw_observations": int(
                    fallback_history_state.hidden_fallback_raw_observations
                ),
                "hidden_washout_history_frames": int(fallback_history_state.hidden_washout_frames),
                "hidden_washout_history_raw_observations": int(
                    fallback_history_state.hidden_washout_raw_observations
                ),
                "fallback_history_quarantine_count": int(fallback_history_state.fallback_quarantine_count),
                "policy_variant": str(config.policy_variant.name),
                "runtime_mode": str(config.policy_variant.runtime_mode),
            }
        )
        return _finalize_rollout_outputs(
            summary=summary,
            action_records=action_records,
            action_video_records=action_video_records,
            replan_records=replan_records,
            extension_records=extension_records,
            component_report=component_report,
            output_dir=output_dir,
            benchmark=benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            suffix=suffix,
            video_fps=video_fps or target_action_hz,
            action_per_frame=action_per_frame,
            write_fallback_timeline_video=write_fallback_timeline_video,
            artifact_profile=artifact_profile,
            startup_debug_report=startup_debug_report,
        )
    finally:
        env.close()


def _run_sequence_policy_realtime_rollout(
    *,
    config,
    checkpoint_path: Path | None,
    rollout_label: str,
    benchmark: str,
    task_id: int,
    episode_idx: int,
    max_actions: int,
    env_horizon: int | None,
    target_action_hz: float,
    video_fps: float | None,
    planner_mode: RealtimePlannerMode | str,
    deadline_miss_policy: str,
    deadline_tolerance_ms: float,
    output_dir: Path,
    suffix: str,
    seed: int,
    runtime_device: torch.device,
    runtime_devices: tuple[torch.device, ...],
    runtime_prep_device: torch.device,
    runtime_output_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    sequence_buffer_threshold: int,
    sequence_empty_plan_policy: RealtimeEmptyPlanPolicy | str,
    fallback_history_policy: FallbackHistoryPolicy,
    startup_open_loop_chunks: int,
    replan_low_watermark_actions: int,
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
    guidance_scale: float | None,
    action_guidance_scale: float | None,
    initial_generation_action_start: int | None,
    write_fallback_timeline_video: bool,
    artifact_profile: RolloutArtifactProfile | str,
) -> dict[str, Any]:
    planner_mode = RealtimePlannerMode(planner_mode)
    sequence_empty_plan_policy = RealtimeEmptyPlanPolicy(sequence_empty_plan_policy)
    _apply_common_inference_overrides(
        config,
        video_num_inference_steps=video_num_inference_steps,
        action_num_inference_steps=action_num_inference_steps,
        guidance_scale=guidance_scale,
        action_guidance_scale=action_guidance_scale,
    )
    if checkpoint_path is None:
        raise ValueError(
            f"{rollout_label} realtime rollout requires a full checkpoint via `--checkpoint` or config-backed inference."
        )
    realtime_runtime.validate_sequence_startup_open_loop_support(
        config=config,
        startup_open_loop_chunks=startup_open_loop_chunks,
    )
    _print_stage(f"{rollout_label}_build_pipeline_start")
    pipeline = build_variant_pipeline_from_config(config)
    _print_stage(f"{rollout_label}_build_pipeline_done")
    _print_stage(f"{rollout_label}_load_checkpoint_start", checkpoint=str(checkpoint_path))
    checkpoint_report = runtime_checkpoints.load_pipeline_checkpoint(
        pipeline,
        checkpoint_path,
    )
    if checkpoint_report.missing_keys:
        print(f"viz.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
    if checkpoint_report.unexpected_keys:
        print(
            "viz.checkpoint_unexpected_keys "
            f"{len(checkpoint_report.unexpected_keys)}"
        )
    _print_stage(f"{rollout_label}_load_checkpoint_done")
    _print_stage(f"{rollout_label}_move_pipeline_start", runtime_device=str(runtime_device))
    pipeline = pipeline.to(runtime_device)
    _print_stage(f"{rollout_label}_move_pipeline_done")
    dual_expert_inference_backend = None
    if str(config.policy_variant.name) == "dual_expert":
        dual_expert_inference_backend = ensure_dual_expert_inference_backend(pipeline, config)
        _print_stage(
            f"{rollout_label}_dual_expert_inference_backend",
            **dual_expert_inference_backend,
        )
    _print_stage(f"{rollout_label}_configure_runtime_devices_start")
    pipeline.eval()
    pipeline.visual_tower.configure_runtime_devices(
        runtime_devices,
        prep_device=runtime_prep_device,
        output_device=runtime_output_device,
    )
    _print_stage(f"{rollout_label}_configure_runtime_devices_done")
    runner = VariantRolloutRunner(pipeline)
    load_report = {
        "pipeline": "open_wam_variant_sequence_rollout",
        "policy_variant": str(config.policy_variant.name),
        "rollout_label": str(rollout_label),
        "checkpoint_file": str(checkpoint_path.resolve()),
        "transformer_dir": str(config.backbone.transformer_subdir),
        "runtime_device": str(runtime_device),
        "runtime_prep_device": str(runtime_prep_device),
        "runtime_output_device": str(runtime_output_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "action_device": str(runtime_device) if str(config.policy_variant.name) == "dual_expert" else None,
        "runtime_devices": [str(device) for device in runtime_devices],
        "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
        "video_steps": int(config.inference.video_num_inference_steps),
        "action_steps": int(config.inference.action_num_inference_steps),
        "guidance_scale": float(config.inference.guidance_scale),
        "action_guidance_scale": float(config.inference.action_guidance_scale),
        "video_steps_override": video_num_inference_steps,
        "action_steps_override": action_num_inference_steps,
        "guidance_scale_override": guidance_scale,
        "action_guidance_scale_override": action_guidance_scale,
        "action_horizon": int(config.data.action_schema.action_horizon),
        "sequence_empty_plan_policy": sequence_empty_plan_policy.value,
        "fallback_history_policy": str(fallback_history_policy),
        "sequence_buffer_threshold": int(sequence_buffer_threshold),
        "startup_open_loop_chunks": int(startup_open_loop_chunks),
        "replan_low_watermark_actions": int(replan_low_watermark_actions),
        "strict_dual_expert_split_cache_startup": bool(realtime_runtime.uses_strict_dual_expert_split_cache_startup(config)),
        "strict_dual_expert_one_frame_history": bool(realtime_runtime.uses_strict_dual_expert_one_frame_history(config)),
        "decoder_runtime": realtime_runtime.collect_decoder_runtime_metadata(pipeline, config),
    }
    if dual_expert_inference_backend is not None:
        load_report["dual_expert_inference_backend"] = dual_expert_inference_backend

    _print_stage(f"{rollout_label}_resolve_task_start", benchmark=benchmark, task_id=task_id)
    task_spec = resolve_libero_task_by_id(benchmark, task_id, REPO_ROOT)
    prompt = task_spec.task_language
    _print_stage(f"{rollout_label}_resolve_task_done", prompt=prompt)
    init_states = load_libero_task_init_states(task_spec, REPO_ROOT)
    _print_stage(f"{rollout_label}_load_init_states_done", num_init_states=len(init_states))
    env = _construct_realtime_libero_env(task_spec, env_horizon=env_horizon)
    _print_stage(f"{rollout_label}_construct_env_done", env_created=env is not None)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    action_period_s = 1.0 / float(target_action_hz)
    deadline_tolerance_s = float(deadline_tolerance_ms) / 1000.0
    raw_window_frames = raw_window_frames_for_latents(int(config.data.num_frames))
    startup_env_init_frames = realtime_runtime.resolve_sequence_startup_environment_frames(
        config,
        raw_window_frames=raw_window_frames,
    )
    model_obs_window_frames = realtime_runtime.resolve_sequence_model_observation_window_frames(
        config,
        raw_window_frames=raw_window_frames,
    )
    load_report["raw_window_frames"] = int(raw_window_frames)
    load_report["startup_env_init_frames"] = int(startup_env_init_frames)
    load_report["model_obs_window_frames"] = int(model_obs_window_frames)
    control_config = LiberoControlConfig()

    try:
        with torch.inference_mode():
            initial_obs_window = libero_rollout.initialize_libero_observation_window(
                env,
                init_states[episode_idx % len(init_states)],
                num_frames=startup_env_init_frames,
            )
            _print_stage(f"{rollout_label}_init_env_done", initial_window=len(initial_obs_window))
            startup_model_obs_window = realtime_runtime.build_sequence_startup_observation_window(config, initial_obs_window)
            startup_prepare_t0 = time.perf_counter()
            initial_inputs = rollout_runtime.prepare_rollout_observation_inputs(
                pipeline,
                views=libero_rollout.libero_observation_window_to_views(
                    startup_model_obs_window,
                    device=frontend_device,
                ),
                task_text=(prompt,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            realtime_runtime.synchronize_devices(frontend_device, runtime_device)
            startup_prepare_s = time.perf_counter() - startup_prepare_t0

            session = runner.reset(
                task_text=(prompt,),
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )
            startup_infer_t0 = time.perf_counter()
            startup = realtime_runtime.run_sequence_replan_job(
                runner=runner,
                session=session,
                obs_window=realtime_history.copy_observation_window(startup_model_obs_window),
                config=config,
                options=realtime_runtime.SequenceReplanJobOptions(
                    prompt=prompt,
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    generation_action_start=rollout_runtime.resolve_initial_generation_action_start(
                        initial_obs_window,
                        initial_generation_action_start=initial_generation_action_start,
                        rollout_starts_at_action_zero=rollout_runtime.uses_zero_based_generation_start(
                            config
                        ),
                    ),
                    source="startup_plan",
                ),
            )
            realtime_runtime.synchronize_devices(runtime_device)
            startup_infer_s = time.perf_counter() - startup_infer_t0
            _print_stage(
                f"{rollout_label}_startup_done",
                startup_prepare_s=float(startup_prepare_s),
                startup_infer_s=float(startup_infer_s),
            )

        session = startup.session
        next_generation_action_start = int(startup.next_generation_action_start)
        dual_expert_non_joint_sequence = realtime_runtime.uses_dual_expert_split_cache_sequence(config)
        history_base_session = session if dual_expert_non_joint_sequence else realtime_speculation.clone_session(session)
        history_base_cache_snapshot = startup.runtime_cache_snapshot
        history_generation_action_start = int(next_generation_action_start)
        buffer_tail_session = session if dual_expert_non_joint_sequence else realtime_speculation.clone_session(session)
        buffer_tail_cache_snapshot = startup.runtime_cache_snapshot
        buffer_tail_generation_action_start = int(next_generation_action_start)
        plan_by_action = merge_future_control_steps(
            {},
            startup.planned_steps,
            next_action_to_execute=0,
        )
        current_obs = realtime_history.copy_observation(initial_obs_window[-1])
        obs_window = realtime_history.copy_observation_window(initial_obs_window)
        model_obs_window = realtime_runtime.build_sequence_startup_observation_window(config, initial_obs_window)
        sequence_fallback_state = realtime_history.ActionFallbackHistoryState(policy=fallback_history_policy)
        sequence_clean_actions_required = max(1, int(config.data.action_schema.action_horizon))
        extension_records: list[dict[str, Any]] = []
        startup_open_loop_s = 0.0
        if startup_open_loop_chunks > 0:
            startup_open_loop_t0 = time.perf_counter()
            for _ in range(int(startup_open_loop_chunks)):
                extension = realtime_runtime.run_sequence_replan_job(
                    runner=runner,
                    session=(
                        buffer_tail_session
                        if dual_expert_non_joint_sequence
                        else realtime_speculation.clone_session(buffer_tail_session)
                    ),
                    obs_window=realtime_history.copy_observation_window(model_obs_window),
                    config=config,
                    options=realtime_runtime.SequenceReplanJobOptions(
                        prompt=prompt,
                        task_id=int(task_id),
                        episode_idx=int(episode_idx),
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        generation_action_start=buffer_tail_generation_action_start,
                        source="open_loop_extension",
                        reset_observation_conditioned_session=False,
                        use_observation_update=False,
                        runtime_cache_snapshot=buffer_tail_cache_snapshot,
                        preserve_rng_state=True,
                    ),
                )
                extension_records.append(extension.trace)
                buffer_tail_session = (
                    extension.session
                    if dual_expert_non_joint_sequence
                    else realtime_speculation.clone_session(extension.session)
                )
                buffer_tail_cache_snapshot = extension.runtime_cache_snapshot
                buffer_tail_generation_action_start = int(
                    extension.next_generation_action_start
                )
                session = (
                    buffer_tail_session
                    if dual_expert_non_joint_sequence
                    else realtime_speculation.clone_session(buffer_tail_session)
                )
                next_generation_action_start = int(buffer_tail_generation_action_start)
                plan_by_action = merge_future_control_steps(
                    plan_by_action,
                    extension.planned_steps,
                    next_action_to_execute=0,
                )
            startup_open_loop_s = time.perf_counter() - startup_open_loop_t0
            startup_infer_s += startup_open_loop_s
        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
        collect_video_records = rollout_artifacts.RolloutArtifactPolicy.from_value(
            artifact_profile,
            write_fallback_timeline_video=write_fallback_timeline_video,
        ).collects_video_records
        replan_records: list[dict[str, Any]] = []
        done = False
        last_action = np.zeros((int(config.data.action_schema.action_dim),), dtype=np.float32)
        executed_action_index = 0
        next_action_index = 0
        live_start_monotonic = time.perf_counter()
        last_action_end_monotonic = live_start_monotonic
        skipped_replan_submissions = 0
        wait_for_plan_count = 0
        wait_for_plan_total_s = 0.0
        blocking_replan_count = 0
        schedule_pause_s = 0.0

        with ThreadPoolExecutor(max_workers=1) as executor:
            replan_future: Future[
                realtime_runtime.SequenceReplanJobResult
            ] | None = None
            while executed_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
                    if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                        future_steps = realtime_runtime.annotate_sequence_planner_acceptance(
                            result,
                            next_action_to_execute=next_action_index,
                        )
                        replan_records.append(result.trace)
                        if future_steps:
                            if bool(result.trace.get("use_observation_update", True)):
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = drop_control_steps_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.runtime_cache_snapshot
                                history_generation_action_start = int(result.next_generation_action_start)
                                buffer_tail_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.runtime_cache_snapshot
                                buffer_tail_generation_action_start = int(result.next_generation_action_start)
                            else:
                                buffer_tail_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.runtime_cache_snapshot
                                buffer_tail_generation_action_start = int(result.next_generation_action_start)
                            plan_by_action = merge_future_control_steps(
                                plan_by_action,
                                future_steps,
                                next_action_to_execute=next_action_index,
                            )
                            session = realtime_speculation.session_reference(
                                buffer_tail_session,
                                share_session=dual_expert_non_joint_sequence,
                            )
                            next_generation_action_start = int(buffer_tail_generation_action_start)
                    else:
                        replan_records.append(result.trace)
                        session = result.session
                        next_generation_action_start = int(result.next_generation_action_start)
                        plan_by_action = merge_future_control_steps(
                            plan_by_action,
                            result.planned_steps,
                            next_action_to_execute=next_action_index,
                        )
                    replan_future = None

                planned_step = plan_by_action.pop(next_action_index, None)
                wait_for_plan_s = 0.0
                if planned_step is None:
                    if sequence_empty_plan_policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN:
                        wait_t0 = time.perf_counter()
                        if replan_future is None:
                            blocking_replan_count += 1
                            if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                                if (
                                    buffer_tail_session is not None
                                    and realtime_runtime.sequence_buffer_tail_ready_for_history_promotion(
                                        config=config,
                                        next_action_index=next_action_index,
                                        buffer_tail_generation_action_start=buffer_tail_generation_action_start,
                                        history_generation_action_start=history_generation_action_start,
                                    )
                                ):
                                    history_base_session = realtime_speculation.session_reference(
                                        buffer_tail_session,
                                        share_session=dual_expert_non_joint_sequence,
                                    )
                                    history_base_cache_snapshot = buffer_tail_cache_snapshot
                                    history_generation_action_start = int(buffer_tail_generation_action_start)
                                blocking_session = realtime_speculation.session_reference(
                                    history_base_session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                blocking_generation_action_start = int(history_generation_action_start)
                                blocking_cache_snapshot = history_base_cache_snapshot
                                blocking_condition_frame_start = realtime_runtime.resolve_sequence_condition_frame_start(
                                    config=config,
                                    generation_action_start=blocking_generation_action_start,
                                )
                            else:
                                blocking_session = session
                                blocking_generation_action_start = int(next_generation_action_start)
                                blocking_cache_snapshot = None
                                blocking_condition_frame_start = None
                            # The restored history snapshot already contains
                            # the correct DualExpert action-cache prefix. Rewinding
                            # here mutates chunk-by-chunk parity.
                            result = realtime_runtime.run_sequence_replan_job(
                                runner=runner,
                                session=blocking_session,
                                obs_window=realtime_history.copy_observation_window(model_obs_window),
                                config=config,
                                options=realtime_runtime.SequenceReplanJobOptions(
                                    prompt=prompt,
                                    task_id=int(task_id),
                                    episode_idx=int(episode_idx),
                                    frontend_device=frontend_device,
                                    runtime_device=runtime_device,
                                    generation_action_start=blocking_generation_action_start,
                                    source="blocking_replan",
                                    runtime_cache_snapshot=blocking_cache_snapshot,
                                    dual_expert_condition_frame_start=(
                                        blocking_condition_frame_start
                                    ),
                                ),
                            )
                        else:
                            result = replan_future.result()
                            replan_future = None
                        wait_for_plan_s = time.perf_counter() - wait_t0
                        wait_for_plan_count += 1
                        wait_for_plan_total_s += wait_for_plan_s
                        result.trace["blocking_wait_action_index"] = int(next_action_index)
                        result.trace["blocking_wait_s"] = float(wait_for_plan_s)
                        if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                            future_steps = realtime_runtime.annotate_sequence_planner_acceptance(
                                result,
                                next_action_to_execute=next_action_index,
                            )
                            replan_records.append(result.trace)
                            if future_steps:
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = drop_control_steps_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.runtime_cache_snapshot
                                history_generation_action_start = int(result.next_generation_action_start)
                                buffer_tail_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.runtime_cache_snapshot
                                buffer_tail_generation_action_start = int(result.next_generation_action_start)
                                plan_by_action = merge_future_control_steps(
                                    plan_by_action,
                                    future_steps,
                                    next_action_to_execute=next_action_index,
                                )
                                session = realtime_speculation.session_reference(
                                    buffer_tail_session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                next_generation_action_start = int(buffer_tail_generation_action_start)
                        else:
                            (
                                session,
                                next_generation_action_start,
                                plan_by_action,
                            ) = realtime_runtime.apply_sequence_replan_result(
                                result=result,
                                replan_records=replan_records,
                                plan_by_action=plan_by_action,
                                next_action_to_execute=next_action_index,
                            )
                        planned_step = plan_by_action.pop(next_action_index, None)
                        if planned_step is None:
                            raise RuntimeError(
                                "Blocking sequence replan did not produce the next required action "
                                f"{next_action_index}; planned={sorted(plan_by_action)}."
                            )
                    elif sequence_empty_plan_policy == RealtimeEmptyPlanPolicy.FALLBACK:
                        action = realtime_runtime.build_fallback_frame_actions(
                            action_dim=int(config.data.action_schema.action_dim),
                            action_per_frame=1,
                            policy=deadline_miss_policy,
                            last_action=last_action,
                        )[0]
                        source = f"fallback_{deadline_miss_policy}"
                        generation_action_start = None
                        planner_step_index = None
                        ready_monotonic_s = None
                    else:
                        raise ValueError(f"Unsupported sequence_empty_plan_policy={sequence_empty_plan_policy!r}.")
                if planned_step is not None:
                    action = realtime_runtime.materialize_sequence_control_action(
                        planned_step,
                        current_obs=current_obs,
                        control_config=control_config,
                        gripper_representation=str(config.data.action_target.gripper_representation),
                    )
                    source = str(planned_step.source)
                    generation_action_start = int(planned_step.generation_action_start)
                    planner_step_index = planned_step.planner_step_index
                    ready_monotonic_s = planned_step.ready_monotonic_s
                else:
                    wait_for_plan_s = 0.0

                schedule_pause_s += wait_for_plan_s
                real_action_index = int(executed_action_index)
                model_action_index = int(next_action_index)
                scheduled_monotonic = live_start_monotonic + real_action_index * action_period_s + schedule_pause_s
                now = time.perf_counter()
                if now < scheduled_monotonic:
                    time.sleep(scheduled_monotonic - now)
                actual_start_monotonic = time.perf_counter()
                lateness_s = max(0.0, actual_start_monotonic - scheduled_monotonic)
                obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                action_end_monotonic = time.perf_counter()
                env_step_s = action_end_monotonic - actual_start_monotonic
                last_action_end_monotonic = action_end_monotonic
                current_obs = libero_rollout.extract_libero_rollout_observation(obs)
                obs_window = realtime_history.append_observation_window(
                    obs_window,
                    current_obs,
                    max_window_frames=raw_window_frames,
                )
                history_append_result = realtime_history.append_action_observation(
                    model_obs_window=model_obs_window,
                    state=sequence_fallback_state,
                    current_obs=current_obs,
                    action_source=source,
                    clean_actions_required=sequence_clean_actions_required,
                    max_window_frames=model_obs_window_frames,
                )
                last_action = np.array(action, copy=True)

                action_record = {
                    "action_index": int(real_action_index),
                    "absolute_action_index": int(real_action_index),
                    "absolute_frame_index": int(real_action_index + 1),
                    "action_offset": 0,
                    "model_action_index": int(model_action_index),
                    "model_frame_index": int(model_action_index + 1),
                    "model_action_offset": 0,
                    "source": source,
                    "scheduled_start_s": float(scheduled_monotonic - live_start_monotonic),
                    "actual_start_s": float(actual_start_monotonic - live_start_monotonic),
                    "lateness_s": float(lateness_s),
                    "env_step_s": float(env_step_s),
                    "generation_action_start": generation_action_start,
                    "generation_lag_actions": (
                        None
                        if generation_action_start is None
                        else int(model_action_index - generation_action_start)
                    ),
                    "generation_frame_start": (
                        None
                        if generation_action_start is None
                        else int(generation_action_start + 1)
                    ),
                    "generation_lag_frames": (
                        None
                        if generation_action_start is None
                        else int((model_action_index + 1) - (generation_action_start + 1))
                    ),
                    "planner_step_index": planner_step_index,
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                    "plan_ready_delay_s": (
                        None
                        if ready_monotonic_s is None
                        else float(actual_start_monotonic - ready_monotonic_s)
                    ),
                    "wait_for_plan_s": float(wait_for_plan_s),
                    "history_append_result": history_append_result.value,
                }
                action_records.append(action_record)
                if collect_video_records:
                    action_video_records.append(
                        {
                            **action_record,
                            "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
                        }
                    )
                if VERBOSE and real_action_index % 50 == 0:
                    _print_stage(
                        f"{rollout_label}_action_progress",
                        action_index=int(real_action_index),
                        model_action_index=int(model_action_index),
                        source=source,
                        done=bool(done),
                    )
                executed_action_index += 1
                if realtime_history.action_advances_model_timeline(
                    source,
                    fallback_history_policy=fallback_history_policy,
                ):
                    next_action_index += 1
                if done or executed_action_index >= max_actions:
                    break

                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
                    if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                        future_steps = realtime_runtime.annotate_sequence_planner_acceptance(
                            result,
                            next_action_to_execute=next_action_index,
                        )
                        replan_records.append(result.trace)
                        if future_steps:
                            if bool(result.trace.get("use_observation_update", True)):
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = drop_control_steps_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.runtime_cache_snapshot
                                history_generation_action_start = int(result.next_generation_action_start)
                                buffer_tail_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.runtime_cache_snapshot
                                buffer_tail_generation_action_start = int(result.next_generation_action_start)
                            else:
                                buffer_tail_session = realtime_speculation.session_reference(
                                    result.session,
                                    share_session=dual_expert_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.runtime_cache_snapshot
                                buffer_tail_generation_action_start = int(result.next_generation_action_start)
                            plan_by_action = merge_future_control_steps(
                                plan_by_action,
                                future_steps,
                                next_action_to_execute=next_action_index,
                            )
                            session = realtime_speculation.session_reference(
                                buffer_tail_session,
                                share_session=dual_expert_non_joint_sequence,
                            )
                            next_generation_action_start = int(buffer_tail_generation_action_start)
                    else:
                        replan_records.append(result.trace)
                        session = result.session
                        next_generation_action_start = int(result.next_generation_action_start)
                        plan_by_action = merge_future_control_steps(
                            plan_by_action,
                            result.planned_steps,
                            next_action_to_execute=next_action_index,
                        )
                    replan_future = None

                remaining_buffer = future_control_depth(plan_by_action, next_action_to_execute=next_action_index)
                should_submit = should_submit_sequence_planner(
                    planner_mode=planner_mode,
                    future_buffer_depth_actions=remaining_buffer,
                    empty_plan_policy=sequence_empty_plan_policy,
                    sequence_buffer_threshold=sequence_buffer_threshold,
                    replan_low_watermark_actions=replan_low_watermark_actions,
                )
                if replan_future is None and should_submit:
                    obs_snapshot = realtime_history.copy_observation_window(model_obs_window)
                    if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                        if (
                            buffer_tail_session is not None
                            and realtime_runtime.sequence_buffer_tail_ready_for_history_promotion(
                                config=config,
                                next_action_index=next_action_index,
                                buffer_tail_generation_action_start=buffer_tail_generation_action_start,
                                history_generation_action_start=history_generation_action_start,
                            )
                        ):
                            history_base_session = realtime_speculation.session_reference(
                                buffer_tail_session,
                                share_session=dual_expert_non_joint_sequence,
                            )
                            history_base_cache_snapshot = buffer_tail_cache_snapshot
                            history_generation_action_start = int(buffer_tail_generation_action_start)
                        history_ready = realtime_runtime.sequence_history_replan_ready(
                            config=config,
                            next_action_index=next_action_index,
                            generation_action_start=history_generation_action_start,
                        )
                        use_observation_update = (
                            planner_mode == RealtimePlannerMode.HISTORY_ONLY
                            or (
                                planner_mode
                                in {
                                    RealtimePlannerMode.ASYNC_MIX,
                                    RealtimePlannerMode.ASYNC_HISTORY_FIRST,
                                }
                                and history_ready
                            )
                        )
                        if use_observation_update:
                            submit_session = realtime_speculation.session_reference(
                                history_base_session,
                                share_session=dual_expert_non_joint_sequence,
                            )
                            submit_generation_action_start = int(history_generation_action_start)
                            submit_cache_snapshot = history_base_cache_snapshot
                            submit_condition_frame_start = realtime_runtime.resolve_sequence_condition_frame_start(
                                config=config,
                                generation_action_start=submit_generation_action_start,
                            )
                        else:
                            submit_session = realtime_speculation.session_reference(
                                buffer_tail_session,
                                share_session=dual_expert_non_joint_sequence,
                            )
                            submit_generation_action_start = int(buffer_tail_generation_action_start)
                            submit_cache_snapshot = buffer_tail_cache_snapshot
                            submit_condition_frame_start = None
                    else:
                        use_observation_update = not realtime_runtime.should_use_sequence_open_loop_extension(
                            config=config,
                            planner_mode=planner_mode,
                            remaining_buffer_actions=remaining_buffer,
                        )
                        submit_session = session
                        submit_generation_action_start = int(next_generation_action_start)
                        submit_cache_snapshot = None
                        submit_condition_frame_start = None
                    replan_future = executor.submit(
                        realtime_runtime.run_sequence_replan_job,
                        runner=runner,
                        session=submit_session,
                        obs_window=obs_snapshot,
                        config=config,
                        options=realtime_runtime.SequenceReplanJobOptions(
                            prompt=prompt,
                            task_id=int(task_id),
                            episode_idx=int(episode_idx),
                            frontend_device=frontend_device,
                            runtime_device=runtime_device,
                            generation_action_start=submit_generation_action_start,
                            source=(
                                "history_replan"
                                if use_observation_update
                                else "open_loop_extension"
                            ),
                            reset_observation_conditioned_session=(
                                use_observation_update
                            ),
                            use_observation_update=use_observation_update,
                            runtime_cache_snapshot=submit_cache_snapshot,
                            dual_expert_condition_frame_start=submit_condition_frame_start,
                            # Async observation-conditioned DualExpert replans can be
                            # launched from a speculative buffer-tail session.
                            # Trim that future action K/V suffix to the chunk
                            # being replaced; blocking/history-only replans keep
                            # their accepted cache prefix untouched for parity.
                            dual_expert_action_cache_rewind_frame_start=(
                                realtime_runtime.resolve_sequence_action_cache_rewind_frame(
                                    config=config,
                                    planner_mode=planner_mode,
                                    use_observation_update=use_observation_update,
                                    condition_frame_start=(
                                        submit_condition_frame_start
                                    ),
                                )
                            ),
                            preserve_rng_state=not bool(use_observation_update),
                        ),
                    )
                elif replan_future is not None:
                    skipped_replan_submissions += 1

            if replan_future is not None and replan_future.done():
                result = replan_future.result()
                if realtime_runtime.uses_dual_expert_split_cache_sequence(config):
                    future_steps = realtime_runtime.annotate_sequence_planner_acceptance(
                        result,
                        next_action_to_execute=next_action_index,
                    )
                    replan_records.append(result.trace)
                    if future_steps:
                        plan_by_action = merge_future_control_steps(
                            plan_by_action,
                            future_steps,
                            next_action_to_execute=next_action_index,
                        )
                else:
                    replan_records.append(result.trace)
                    session = result.session
                    next_generation_action_start = int(result.next_generation_action_start)
                    plan_by_action = merge_future_control_steps(
                        plan_by_action,
                        result.planned_steps,
                        next_action_to_execute=next_action_index,
                    )

        live_wall_time_s = last_action_end_monotonic - live_start_monotonic if executed_action_index > 0 else 0.0
        summary = build_live_rollout_summary(
            action_records=action_records,
            replan_records=replan_records,
            target_action_hz=target_action_hz,
            live_wall_time_s=live_wall_time_s,
            startup_prepare_s=startup_prepare_s,
            startup_infer_s=startup_infer_s,
            deadline_tolerance_s=deadline_tolerance_s,
        )
        summary.update(
            {
                "benchmark": benchmark,
                "task_id": int(task_id),
                "prompt": prompt,
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "max_actions": int(max_actions),
                "env_horizon": None if env_horizon is None else int(env_horizon),
                "executed_actions": int(executed_action_index),
                "model_executed_actions": int(next_action_index),
                "env_timestep": int(env.env.timestep),
                "seed": int(seed),
                "runtime_device": str(runtime_device),
                "runtime_prep_device": str(runtime_prep_device),
                "runtime_output_device": str(runtime_output_device),
                "frontend_device": str(frontend_device),
                "decode_device": str(decode_device),
                "runtime_devices": [str(device) for device in runtime_devices],
                "checkpoint_file": str(checkpoint_path.resolve()),
                "transformer_dir": str(config.backbone.transformer_subdir),
                "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
                "video_num_inference_steps": int(config.inference.video_num_inference_steps),
                "action_num_inference_steps": int(config.inference.action_num_inference_steps),
                "guidance_scale": float(config.inference.guidance_scale),
                "action_guidance_scale": float(config.inference.action_guidance_scale),
                "raw_window_frames": int(raw_window_frames),
                "startup_env_init_frames": int(startup_env_init_frames),
                "model_obs_window_frames": int(model_obs_window_frames),
                "strict_dual_expert_one_frame_history": bool(realtime_runtime.uses_strict_dual_expert_one_frame_history(config)),
                "planner_mode": planner_mode.value,
                "sequence_buffer_threshold": int(sequence_buffer_threshold),
                "sequence_empty_plan_policy": sequence_empty_plan_policy.value,
                "fallback_history_policy": str(fallback_history_policy),
                "replan_low_watermark_actions": int(replan_low_watermark_actions),
                "periodic_replan_frames": int(replan_low_watermark_actions),
                "startup_open_loop_chunks": int(startup_open_loop_chunks),
                "startup_open_loop_s": float(startup_open_loop_s),
                "deadline_miss_policy": deadline_miss_policy,
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "wait_for_plan_count": int(wait_for_plan_count),
                "wait_for_plan_total_s": float(wait_for_plan_total_s),
                "schedule_pause_s": float(schedule_pause_s),
                "blocking_replan_count": int(blocking_replan_count),
                "history_replan_count": len(replan_records),
                "open_loop_extension_count": len(extension_records),
                "hidden_fallback_period_count": int(sequence_fallback_state.fallback_quarantine_count),
                "hidden_fallback_history_actions": int(sequence_fallback_state.hidden_fallback_actions),
                "hidden_washout_history_actions": int(sequence_fallback_state.hidden_washout_actions),
                "hidden_history_actions": int(sequence_fallback_state.hidden_history_actions),
                "policy_variant": str(config.policy_variant.name),
                "startup_plan_trace": startup.trace,
            }
        )
        return _finalize_rollout_outputs(
            summary=summary,
            action_records=action_records,
            action_video_records=action_video_records,
            replan_records=replan_records,
            extension_records=extension_records,
            component_report=load_report,
            output_dir=output_dir,
            benchmark=benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            suffix=suffix,
            video_fps=video_fps or target_action_hz,
            action_per_frame=1,
            write_fallback_timeline_video=write_fallback_timeline_video,
            artifact_profile=artifact_profile,
        )
    finally:
        env.close()



def _finalize_rollout_outputs(
    *,
    summary: dict[str, Any],
    action_records: list[dict[str, Any]],
    action_video_records: list[dict[str, Any]],
    replan_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
    component_report: dict[str, Any],
    output_dir: Path,
    benchmark: str,
    task_id: int,
    prompt: str,
    episode_idx: int,
    suffix: str,
    video_fps: float,
    action_per_frame: int,
    write_fallback_timeline_video: bool,
    artifact_profile: RolloutArtifactProfile | str,
    startup_debug_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = rollout_artifacts.persist_libero_realtime_artifacts(
        identity=rollout_artifacts.LiberoRealtimeArtifactIdentity(
            benchmark=benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            suffix=suffix,
        ),
        options=rollout_artifacts.LiberoRealtimeArtifactOptions(
            output_root=output_dir,
            video_fps=video_fps,
            action_per_frame=action_per_frame,
            policy=rollout_artifacts.RolloutArtifactPolicy.from_value(
                artifact_profile,
                write_fallback_timeline_video=write_fallback_timeline_video,
            ),
        ),
        payload=rollout_artifacts.LiberoRealtimeArtifactPayload(
            action_records=action_records,
            action_video_records=action_video_records,
            replan_records=replan_records,
            extension_records=extension_records,
            component_report=component_report,
            startup_debug_report=startup_debug_report,
        ),
        summary=summary,
    )
    return output.summary


def _print_stage(name: str, **payload: object) -> None:
    if VERBOSE:
        print(json.dumps({"stage": name, **payload}), flush=True)


if __name__ == "__main__":
    main()
