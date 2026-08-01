from __future__ import annotations

import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


_prepend_import_path(SRC_ROOT)
_prepend_import_path(SCRIPT_ROOT)

import libero_exact_realtime_common as exact_sandbox  # noqa: E402

from open_wam.configs import ActionTargetRepresentation, GripperRepresentation, ParallelRuntimeMode  # noqa: E402
from open_wam.configs.enums import (  # noqa: E402
    DeadlineMissPolicy,
    FallbackHistoryPolicy,
    RolloutArtifactProfile,
)
from open_wam.data.action_transforms import PoseSequence  # noqa: E402
from open_wam.data.latent_temporal import raw_window_frames_for_latents  # noqa: E402
from open_wam.integrations import (  # noqa: E402
    LiberoControlConfig,
    compute_osc_pose_action,
    ensure_local_libero_config,
    load_libero_task_init_states,
    resolve_libero_task_by_id,
)
from open_wam.integrations import libero_rollout  # noqa: E402
from open_wam.integrations.realtime_control import build_live_rollout_summary  # noqa: E402
from open_wam.evals import libero_rollout_artifacts as rollout_artifacts  # noqa: E402
from open_wam.evals import libero_visualization as exact_viz  # noqa: E402
from open_wam.models.common.rollout_startup import require_strict_startup_generation_frame  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.models.policy_variants.mot.runtime_routing import (  # noqa: E402
    ensure_mot_inference_backend,
    mot_config_uses_strict_rollout_parity,
    resolve_mot_sequence_actions_per_frame,
    resolve_mot_sequence_execution_action_offset,
    resolve_mot_runtime_route,
)
from open_wam.pipelines import LingbotExactRunner, VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.runtime import checkpoints as runtime_checkpoints  # noqa: E402
from open_wam.runtime import rollout as rollout_runtime  # noqa: E402
from open_wam.configs import load_experiment_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    apply_config_overrides,
    merge_runtime_config_from_checkpoint,
    parse_override_assignments,
    resolve_transformer_dir_override,
    seed_everywhere,
    validate_positive_step_override,
)
from open_wam.utils.libero_paradigm import require_current_libero_policy_paradigm  # noqa: E402

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


REALTIME_SCHEDULER_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "manual": {},
    "blocking_control": {
        "planner_mode": "history_only",
        "sequence_empty_plan_policy": "wait_for_replan",
        "fallback_history_policy": FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY.value,
        "startup_open_loop_chunks": 0,
        "replan_low_watermark_actions": 0,
    },
    "freeze_until_clean_chunk": {
        "planner_mode": "history_only",
        "sequence_empty_plan_policy": "fallback",
        "fallback_history_policy": FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK.value,
        "startup_open_loop_chunks": 0,
        "replan_low_watermark_actions": 0,
    },
    "async_history_first": {
        "planner_mode": "async_history_first",
        "sequence_empty_plan_policy": "fallback",
        "fallback_history_policy": FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK.value,
        "startup_open_loop_chunks": 1,
    },
}


@dataclass(frozen=True)
class PlannedControlStep:
    absolute_action_index: int
    generation_action_start: int
    source: str
    planner_step_index: int | None = None
    ready_monotonic_s: float | None = None
    generation_frame_start: int | None = None
    raw_action: np.ndarray | None = None
    desired_position: np.ndarray | None = None
    desired_quaternion: np.ndarray | None = None
    desired_gripper: np.ndarray | None = None


@dataclass
class ExactFallbackHistoryState:
    policy: FallbackHistoryPolicy
    quarantine_active: bool = False
    clean_frames_since_fallback: int = 0
    hidden_history_frames: int = 0
    hidden_history_raw_observations: int = 0
    hidden_fallback_frames: int = 0
    hidden_fallback_raw_observations: int = 0
    hidden_washout_frames: int = 0
    hidden_washout_raw_observations: int = 0
    fallback_quarantine_count: int = 0
    quarantine_records: list[dict[str, Any]] | None = None


@dataclass
class SequenceFallbackHistoryState:
    policy: FallbackHistoryPolicy
    quarantine_active: bool = False
    clean_actions_since_fallback: int = 0
    hidden_history_actions: int = 0
    hidden_fallback_actions: int = 0
    hidden_washout_actions: int = 0
    fallback_quarantine_count: int = 0
    quarantine_observations: list[dict[str, np.ndarray]] | None = None


class _RolloutRunnerLike(Protocol):
    def reset(
        self,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ):
        ...


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

    scheduler_defaults = REALTIME_SCHEDULER_PROFILE_DEFAULTS.get(str(args.realtime_scheduler_profile))
    if scheduler_defaults is None:
        raise ValueError(f"Unsupported realtime scheduler profile: {args.realtime_scheduler_profile!r}")
    _apply_cli_profile_defaults(
        args,
        argv,
        scheduler_defaults,
        flag_aliases={
            "planner_mode": ("--planner-mode",),
            "sequence_empty_plan_policy": ("--sequence-empty-plan-policy",),
            "fallback_history_policy": ("--fallback-history-policy",),
            "startup_open_loop_chunks": ("--startup-open-loop-chunks",),
            "replan_low_watermark_actions": ("--replan-low-watermark-actions", "--periodic-replan-frames"),
        },
    )


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
            "method-4 video-conditioned, and method-5 MoT variants."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
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
        type=str,
        choices=("history_only", "async_buffer", "async_mix", "async_history_first"),
        default="async_buffer",
    )
    parser.add_argument(
        "--realtime-scheduler-profile",
        choices=tuple(REALTIME_SCHEDULER_PROFILE_DEFAULTS),
        default="manual",
        help="Named realtime scheduler defaults; explicit low-level scheduler flags still override the profile.",
    )
    parser.add_argument("--sequence-buffer-threshold", type=int, default=3)
    parser.add_argument(
        "--sequence-empty-plan-policy",
        type=str,
        choices=("fallback", "wait_for_replan"),
        default="fallback",
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
            "Allow historical LIBERO M1/M5 configs that do not match the current strict fixed-128, "
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
            rollout_label="method4",
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
    elif policy_name == "mot":
        summary = _run_sequence_policy_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            rollout_label="method5",
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
            "`post_latent`, `post_decoded`, and `mot`, "
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


def _is_mot_non_joint_two_stream(config) -> bool:
    # Historical helper name retained for call-site locality: this means the
    # Method-1-style split-cache MoT route, not every non-joint packed coupling.
    return resolve_mot_runtime_route(config).uses_split_cache_rollout


def _uses_strict_mot_split_cache_startup(config) -> bool:
    route = resolve_mot_runtime_route(config)
    return bool(route.uses_split_cache_rollout and mot_config_uses_strict_rollout_parity(config))


def _uses_strict_mot_one_frame_history(config) -> bool:
    route = resolve_mot_runtime_route(config)
    return bool(route.is_mot and mot_config_uses_strict_rollout_parity(config))


def _sequence_startup_model_obs_window(
    config,
    initial_obs_window: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    if not initial_obs_window:
        raise ValueError("Cannot build sequence startup observation window from an empty initial window.")
    if _uses_strict_mot_one_frame_history(config):
        return [_copy_obs_record(initial_obs_window[-1])]
    return _copy_obs_window(initial_obs_window)


def _sequence_model_obs_window_frames(config, *, raw_window_frames: int) -> int:
    if _uses_strict_mot_one_frame_history(config):
        return 1
    return int(raw_window_frames)


def _sequence_startup_env_init_frames(config, *, raw_window_frames: int) -> int:
    if _uses_strict_mot_one_frame_history(config):
        return 1
    return int(raw_window_frames)


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


def _frame_contains_fallback_action(action_sources: list[str]) -> bool:
    return any(str(source).startswith("fallback_") for source in action_sources)


def _frame_source_label(action_sources: list[str]) -> str:
    if not action_sources:
        return "unknown"
    first_source = str(action_sources[0])
    if all(str(source) == first_source for source in action_sources):
        return first_source
    return "mixed"


def _fallback_history_washout_frames_required(
    policy: FallbackHistoryPolicy,
    *,
    frame_chunk_size: int,
) -> int:
    if policy is FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK:
        return int(frame_chunk_size)
    return 0


def _fallback_policy_freezes_model_timeline(policy: FallbackHistoryPolicy) -> bool:
    return policy is not FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY


def _action_advances_model_timeline(
    action_source: str,
    *,
    fallback_history_policy: FallbackHistoryPolicy,
) -> bool:
    return not (
        str(action_source).startswith("fallback_")
        and _fallback_policy_freezes_model_timeline(fallback_history_policy)
    )


def _fallback_absolute_tail_start(config) -> int | None:
    action_target = getattr(getattr(config, "data", None), "action_target", None)
    gripper_representation = getattr(action_target, "gripper_representation", None)
    if gripper_representation == GripperRepresentation.ACTION_COMMAND:
        return None
    if str(gripper_representation) == GripperRepresentation.ACTION_COMMAND.value:
        return None
    if bool(getattr(action_target, "include_gripper", False)):
        return 6
    return None


def _exact_runtime_cache_name(session) -> str | None:
    cache = getattr(getattr(session, "policy_state", None), "cache", None)
    if not isinstance(cache, dict):
        return None
    cache_name = cache.get("cache_name")
    return None if cache_name is None else str(cache_name)


def _runtime_cache_name_for_session(*, config, session) -> str | None:
    cache_name = _exact_runtime_cache_name(session)
    if cache_name is not None:
        return cache_name
    if resolve_mot_runtime_route(config).uses_split_cache_rollout:
        return "mot_non_joint_two_stream_cache"
    return None


def _exact_runtime_transformer(runner, config):
    pipeline = getattr(runner, "pipeline", None)
    visual_tower = getattr(pipeline, "visual_tower", None)
    if visual_tower is None or not hasattr(visual_tower, "get_runtime_backbone"):
        return None
    policy_variant = getattr(runner, "policy_variant", None)
    action_dim = getattr(policy_variant, "action_dim", None)
    if action_dim is None:
        action_dim = int(config.data.action_schema.action_dim)
    return visual_tower.get_runtime_backbone(action_dim=int(action_dim))


def _exact_runtime_streaming_vae(runner):
    pipeline = getattr(runner, "pipeline", None)
    visual_tower = getattr(pipeline, "visual_tower", None)
    frontend = getattr(visual_tower, "frontend", None)
    reference_assets = getattr(frontend, "reference_assets", None)
    return getattr(reference_assets, "streaming_vae", None)


def _snapshot_exact_runtime_cache(
    *,
    runner,
    config,
    session,
) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {}
    streaming_vae = _exact_runtime_streaming_vae(runner)
    if streaming_vae is not None and hasattr(streaming_vae, "feat_cache"):
        snapshot["streaming_vae_feat_cache"] = copy.deepcopy(streaming_vae.feat_cache)

    cache_name = _runtime_cache_name_for_session(config=config, session=session)
    transformer = _exact_runtime_transformer(runner, config)
    caches = getattr(transformer, "_exact_runtime_caches", None)
    if cache_name is None or not isinstance(caches, dict):
        return snapshot or None
    if cache_name not in caches:
        snapshot.update({"cache_name": cache_name, "exists": False})
        return snapshot
    snapshot.update(
        {
            "cache_name": cache_name,
            "exists": True,
            "cache_state": copy.deepcopy(caches[cache_name]),
        }
    )
    return snapshot


def _restore_exact_runtime_cache_snapshot(
    *,
    runner,
    config,
    snapshot: dict[str, Any] | None,
) -> None:
    if snapshot is None:
        return
    streaming_vae = _exact_runtime_streaming_vae(runner)
    if streaming_vae is not None and "streaming_vae_feat_cache" in snapshot:
        streaming_vae.feat_cache = copy.deepcopy(snapshot["streaming_vae_feat_cache"])

    if "cache_name" not in snapshot:
        return
    transformer = _exact_runtime_transformer(runner, config)
    caches = getattr(transformer, "_exact_runtime_caches", None)
    if not isinstance(caches, dict):
        return
    cache_name = str(snapshot["cache_name"])
    if bool(snapshot.get("exists", False)):
        caches[cache_name] = copy.deepcopy(snapshot["cache_state"])
    else:
        caches.pop(cache_name, None)


def _restore_exact_runtime_cache_if_rejected(
    result: dict[str, Any],
    *,
    runner,
    config,
    snapshot: dict[str, Any] | None,
) -> None:
    if bool(result.get("trace", {}).get("accepted_chunk", False)):
        return
    _restore_exact_runtime_cache_snapshot(runner=runner, config=config, snapshot=snapshot)


def _clone_sequence_session(session):
    return copy.deepcopy(session)


def _sequence_session_ref(session, *, share_session: bool):
    return session if share_session else _clone_sequence_session(session)


def _snapshot_sequence_runtime_cache(*, runner, config, session) -> dict[str, Any] | None:
    if resolve_mot_runtime_route(config).uses_stateful_realtime_session:
        return None
    return _snapshot_exact_runtime_cache(runner=runner, config=config, session=session)


def _snapshot_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    cuda_state = snapshot.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _resolve_exact_planner_future_result(
    future: Future[dict[str, Any]],
    *,
    runner,
    config,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        return future.result()
    except BaseException:
        _restore_exact_runtime_cache_snapshot(runner=runner, config=config, snapshot=snapshot)
        raise


def _maybe_submit_exact_planner_job_with_cache_snapshot(
    *,
    executor: ThreadPoolExecutor,
    planner_mode: str,
    pending_history: list[dict[str, Any]],
    future_buffer_depth: int,
    runner,
    history_base_session,
    current_chunk_session,
    prompt: str,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    buffer_tail_session,
    seed_base: int | None,
) -> tuple[Future[dict[str, Any]] | None, dict[str, Any] | None]:
    if not exact_sandbox._should_submit_planner_job(
        planner_mode=planner_mode,
        has_history=bool(pending_history),
        future_buffer_depth=future_buffer_depth,
        has_buffer_tail_session=buffer_tail_session is not None,
    ):
        return None, None
    # Keep expensive runtime-cache snapshots on the path that will launch a planner job.
    snapshot = _snapshot_exact_runtime_cache(
        runner=runner,
        config=config,
        session=current_chunk_session,
    )
    submitted_future = exact_sandbox._maybe_submit_planner_job(
        executor=executor,
        planner_mode=planner_mode,
        pending_history=pending_history,
        future_buffer_depth=future_buffer_depth,
        runner=runner,
        history_base_session=history_base_session,
        current_chunk_session=current_chunk_session,
        prompt=prompt,
        config=config,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
        buffer_tail_session=buffer_tail_session,
        seed_base=seed_base,
    )
    if submitted_future is None:
        return None, None
    return submitted_future, snapshot


def _record_hidden_exact_history_frame(
    state: ExactFallbackHistoryState,
    *,
    raw_observation_count: int,
    hidden_kind: str,
) -> None:
    state.hidden_history_frames += 1
    state.hidden_history_raw_observations += int(raw_observation_count)
    if hidden_kind == "fallback":
        state.hidden_fallback_frames += 1
        state.hidden_fallback_raw_observations += int(raw_observation_count)
        return
    if hidden_kind == "washout":
        state.hidden_washout_frames += 1
        state.hidden_washout_raw_observations += int(raw_observation_count)
        return
    raise ValueError(f"Unsupported hidden_kind={hidden_kind!r}.")


def _maybe_append_exact_history_record(
    *,
    pending_history: list[dict[str, Any]],
    state: ExactFallbackHistoryState,
    absolute_frame_index: int,
    current_obs: dict[str, np.ndarray],
    frame_obs_sequence: list[dict[str, np.ndarray]],
    frame_actions: list[np.ndarray],
    frame_action_sources: list[str],
    frame_chunk_size: int,
    proprio_state: np.ndarray | torch.Tensor | None = None,
) -> str:
    contains_fallback_action = _frame_contains_fallback_action(frame_action_sources)
    history_record = {
        "absolute_frame_index": int(absolute_frame_index),
        "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
        "obs_sequence": [
            {key: np.array(value, copy=True) for key, value in obs.items()}
            for obs in frame_obs_sequence
        ],
        "raw_actions": np.stack(frame_actions, axis=0).astype(np.float32),
        "source": _frame_source_label(frame_action_sources),
        "action_sources": [str(source) for source in frame_action_sources],
        "contains_fallback_action": bool(contains_fallback_action),
    }
    if proprio_state is not None:
        history_record["proprio_state"] = exact_sandbox._proprio_state_to_numpy(proprio_state)
    raw_observation_count = len(frame_obs_sequence)
    policy = state.policy
    if policy is FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY:
        pending_history.append(history_record)
        return "included"

    washout_frames_required = _fallback_history_washout_frames_required(
        policy,
        frame_chunk_size=frame_chunk_size,
    )
    if contains_fallback_action:
        if not state.quarantine_active:
            state.fallback_quarantine_count += 1
        state.quarantine_active = True
        state.clean_frames_since_fallback = 0
        state.quarantine_records = []
        _record_hidden_exact_history_frame(
            state,
            raw_observation_count=raw_observation_count,
            hidden_kind="fallback",
        )
        return "fallback"
    if not state.quarantine_active:
        pending_history.append(history_record)
        return "included"

    state.clean_frames_since_fallback += 1
    if state.quarantine_records is None:
        state.quarantine_records = []
    state.quarantine_records.append(history_record)
    _record_hidden_exact_history_frame(
        state,
        raw_observation_count=raw_observation_count,
        hidden_kind="washout",
    )
    if state.clean_frames_since_fallback >= washout_frames_required:
        pending_history.extend(state.quarantine_records)
        state.quarantine_records = []
        state.quarantine_active = False
        state.clean_frames_since_fallback = 0
    return "washout"


def _copy_obs_record(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def _copy_obs_window(obs_window: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    return [_copy_obs_record(obs) for obs in obs_window]


def _append_obs_window_record(
    obs_window: list[dict[str, np.ndarray]],
    obs: dict[str, np.ndarray],
    *,
    max_window_frames: int,
) -> list[dict[str, np.ndarray]]:
    obs_window.append(_copy_obs_record(obs))
    if len(obs_window) > int(max_window_frames):
        del obs_window[: -int(max_window_frames)]
    return obs_window


def _maybe_append_sequence_model_observation(
    *,
    model_obs_window: list[dict[str, np.ndarray]],
    state: SequenceFallbackHistoryState,
    current_obs: dict[str, np.ndarray],
    action_source: str,
    clean_actions_required: int,
    max_window_frames: int,
) -> str:
    contains_fallback_action = str(action_source).startswith("fallback_")
    if state.policy is FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY:
        _append_obs_window_record(model_obs_window, current_obs, max_window_frames=max_window_frames)
        return "included"

    if contains_fallback_action:
        if not state.quarantine_active:
            state.fallback_quarantine_count += 1
        state.quarantine_active = True
        state.clean_actions_since_fallback = 0
        state.quarantine_observations = []
        state.hidden_history_actions += 1
        state.hidden_fallback_actions += 1
        return "fallback"

    if not state.quarantine_active:
        _append_obs_window_record(model_obs_window, current_obs, max_window_frames=max_window_frames)
        return "included"

    state.clean_actions_since_fallback += 1
    if state.quarantine_observations is None:
        state.quarantine_observations = []
    state.quarantine_observations.append(_copy_obs_record(current_obs))
    state.hidden_history_actions += 1
    state.hidden_washout_actions += 1
    if state.clean_actions_since_fallback >= int(clean_actions_required):
        for obs in state.quarantine_observations:
            _append_obs_window_record(model_obs_window, obs, max_window_frames=max_window_frames)
        state.quarantine_observations = []
        state.quarantine_active = False
        state.clean_actions_since_fallback = 0
    return "washout"


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
    planner_mode: str,
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
    sequence_empty_plan_policy: str,
    fallback_history_policy: FallbackHistoryPolicy,
    replan_low_watermark_actions: int,
    write_fallback_timeline_video: bool,
    artifact_profile: RolloutArtifactProfile | str,
    debug_startup_dump: bool,
    exact_startup_bootstrap_padding: bool,
) -> dict[str, Any]:
    replan_low_watermark_actions = int(replan_low_watermark_actions)
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.eval()
    runner = LingbotExactRunner(pipeline)
    exact_sandbox._apply_inference_overrides(
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
        "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
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
    fallback_absolute_tail_start = _fallback_absolute_tail_start(config)
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
            # Preserve the exact M1 contract: reseed immediately before each chunk.
            with exact_sandbox._isolated_torch_rng(seed, frontend_device, runtime_device):
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
                exact_sandbox._synchronize_devices(frontend_device, runtime_device)
                startup_prepare_s = time.perf_counter() - startup_prepare_t0
                if VERBOSE:
                    print(
                        "[exact_startup] prepared "
                        f"video_latents_shape={tuple(initial_inputs['video_latents'].shape)} "
                        f"elapsed_s={startup_prepare_s:.3f}",
                        flush=True,
                    )

                rng_before_startup_infer = _debug_rng_state()
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
                exact_sandbox._synchronize_devices(runtime_device)
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
                    startup_debug_report = _build_exact_startup_debug_report(
                        first_obs=first_obs,
                        initial_inputs=initial_inputs,
                        session=session,
                        first_chunk=first_chunk,
                        config=config,
                        prompt=prompt,
                        seed=seed,
                        runtime_device=runtime_device,
                        frontend_device=frontend_device,
                        decode_device=decode_device,
                        rng_before_startup_infer=rng_before_startup_infer,
                        rng_after_startup_infer=_debug_rng_state(),
                        exact_startup_bootstrap_padding=exact_startup_bootstrap_padding,
                        startup_warmup_debug=None,
                    )

        history_base_session, current_chunk_session, buffer_tail_session = exact_sandbox._resolve_exact_startup_sessions(
            config=config,
            startup_session=session,
            first_chunk=first_chunk,
            frame_chunk_size=int(config.inference.frame_chunk_size),
        )
        plan_by_action: dict[int, PlannedControlStep] = _merge_future_step_actions(
            {},
            _exact_chunk_to_planned_steps(
                chunk=first_chunk,
                action_per_frame=action_per_frame,
                frame_chunk_size=int(config.inference.frame_chunk_size),
                source="startup_plan",
                ready_monotonic_s=time.perf_counter(),
            ),
            next_action_to_execute=0,
        )
        pending_history: list[dict[str, Any]] = [
            _exact_startup_conditioning_history_record(
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
        fallback_history_state = ExactFallbackHistoryState(policy=fallback_history_policy)

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
                extension_result = exact_sandbox._run_extension_job(
                    runner=runner,
                    session=buffer_tail_session,
                    config=config,
                    runtime_device=runtime_device,
                    job_seed=exact_sandbox._job_seed_for_session(seed, buffer_tail_session),
                )
                (
                    history_base_session,
                    current_chunk_session,
                    buffer_tail_session,
                    plan_by_action,
                    pending_history,
                ) = _consume_exact_future_result(
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
        freeze_model_timeline_on_fallback = _fallback_policy_freezes_model_timeline(fallback_history_policy)
        hidden_fallback_period_active = False
        hidden_fallback_period_count = 0
        periodic_replan_submit_count = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            replan_future: Future[dict[str, Any]] | None = None
            replan_future_cache_snapshot: dict[str, Any] | None = None
            next_frame_to_execute = 1
            next_real_frame_to_execute = 1
            while next_real_frame_to_execute <= max_frames and executed_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    replan_result = _resolve_exact_planner_future_result(
                        replan_future,
                        runner=runner,
                        config=config,
                        snapshot=replan_future_cache_snapshot,
                    )
                    (
                        history_base_session,
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
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
                    _restore_exact_runtime_cache_if_rejected(
                        replan_result,
                        runner=runner,
                        config=config,
                        snapshot=replan_future_cache_snapshot,
                    )
                    replan_future = None
                    replan_future_cache_snapshot = None

                required_action_indices = _required_frame_action_indices(
                    next_action_index=next_action_index,
                    max_actions=max_actions,
                    action_per_frame=action_per_frame,
                )
                wait_for_plan_s = 0.0
                if (
                    sequence_empty_plan_policy == "wait_for_replan"
                    and _missing_plan_action_indices(plan_by_action, required_action_indices)
                ):
                    wait_t0 = time.perf_counter()
                    wait_job_count = 0
                    max_wait_jobs = max(4, max_frames + 2)
                    while _missing_plan_action_indices(plan_by_action, required_action_indices):
                        if wait_job_count >= max_wait_jobs:
                            raise RuntimeError(
                                "Blocking exact/joint replan did not produce the next required actions "
                                f"{required_action_indices}; missing="
                                f"{_missing_plan_action_indices(plan_by_action, required_action_indices)}."
                            )
                        wait_job_count += 1
                        if replan_future is None:
                            blocking_replan_count += 1
                            future_buffer_depth_frames = int(
                                math.ceil(
                                    _future_buffer_depth_actions(
                                        plan_by_action,
                                        next_action_to_execute=next_action_index,
                                    )
                                    / action_per_frame
                                )
                            )
                            (
                                replan_future,
                                replan_future_cache_snapshot,
                            ) = _maybe_submit_exact_planner_job_with_cache_snapshot(
                                executor=executor,
                                planner_mode=_resolve_exact_realtime_planner_mode(
                                    planner_mode=planner_mode,
                                    sequence_empty_plan_policy=sequence_empty_plan_policy,
                                    pending_history=pending_history,
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
                        result_cache_snapshot: dict[str, Any] | None = None
                        if replan_future is None:
                            result_cache_snapshot = _snapshot_exact_runtime_cache(
                                runner=runner,
                                config=config,
                                session=current_chunk_session,
                            )
                            if pending_history:
                                history_payload = [
                                    exact_sandbox._copy_history_record_for_worker(record)
                                    for record in pending_history
                                ]
                                try:
                                    result = exact_sandbox._run_replan_job(
                                        runner=runner,
                                        session=history_base_session,
                                        prompt=prompt,
                                        history_records=history_payload,
                                        config=config,
                                        frontend_device=frontend_device,
                                        runtime_device=runtime_device,
                                        job_seed=exact_sandbox._job_seed_for_session(seed, current_chunk_session),
                                    )
                                except BaseException:
                                    _restore_exact_runtime_cache_snapshot(
                                        runner=runner,
                                        config=config,
                                        snapshot=result_cache_snapshot,
                                    )
                                    raise
                            elif buffer_tail_session is not None:
                                try:
                                    result = exact_sandbox._run_extension_job(
                                        runner=runner,
                                        session=buffer_tail_session,
                                        config=config,
                                        runtime_device=runtime_device,
                                        job_seed=exact_sandbox._job_seed_for_session(seed, buffer_tail_session),
                                    )
                                except BaseException:
                                    _restore_exact_runtime_cache_snapshot(
                                        runner=runner,
                                        config=config,
                                        snapshot=result_cache_snapshot,
                                    )
                                    raise
                            else:
                                raise RuntimeError(
                                    "Exact/joint wait-for-replan mode has no pending future, history, or buffer "
                                    f"session for required actions {required_action_indices}."
                                )
                        else:
                            result = _resolve_exact_planner_future_result(
                                replan_future,
                                runner=runner,
                                config=config,
                                snapshot=replan_future_cache_snapshot,
                            )
                            replan_future = None
                            result_cache_snapshot = replan_future_cache_snapshot
                            replan_future_cache_snapshot = None
                        wait_for_plan_s = time.perf_counter() - wait_t0
                        result["trace"]["blocking_wait_action_index"] = int(next_action_index)
                        result["trace"]["blocking_wait_frame_index"] = int(next_frame_to_execute)
                        result["trace"]["blocking_wait_s"] = float(wait_for_plan_s)
                        (
                            history_base_session,
                            current_chunk_session,
                            buffer_tail_session,
                            plan_by_action,
                            pending_history,
                        ) = _consume_exact_future_result(
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
                        _restore_exact_runtime_cache_if_rejected(
                            result,
                            runner=runner,
                            config=config,
                            snapshot=result_cache_snapshot,
                        )
                    wait_for_plan_count += 1
                    wait_for_plan_total_s += wait_for_plan_s

                frame_model_action_start = int(next_action_index)
                frame_actions: list[np.ndarray] = []
                frame_action_sources: list[str] = []
                frame_action_metadata: list[dict[str, Any]] = []
                missing_required_action_indices = _missing_plan_action_indices(plan_by_action, required_action_indices)
                use_fallback_frame = (
                    sequence_empty_plan_policy == "fallback"
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
                        raw_action = exact_sandbox._build_fallback_frame_actions(
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
                            if sequence_empty_plan_policy == "wait_for_replan":
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
                        else _frame_index_to_action_start(generation_frame_start, action_per_frame)
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
                    if _action_advances_model_timeline(
                        source,
                        fallback_history_policy=fallback_history_policy,
                    ):
                        next_action_index += 1
                    if done or executed_action_index >= max_actions:
                        break

                if done or executed_action_index >= max_actions:
                    break

                history_decision = _maybe_append_exact_history_record(
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
                frame_contains_fallback = _frame_contains_fallback_action(frame_action_sources)
                for record in action_records[-len(frame_actions) :]:
                    record["frame_history_decision"] = str(history_decision)
                    record["frame_contains_fallback_action"] = bool(frame_contains_fallback)
                    record["frame_action_sources"] = [str(source) for source in frame_action_sources]
                for record in action_video_records[-len(frame_actions) :]:
                    record["frame_history_decision"] = str(history_decision)
                    record["frame_contains_fallback_action"] = bool(frame_contains_fallback)
                    record["frame_action_sources"] = [str(source) for source in frame_action_sources]
                if frame_contains_fallback and not freeze_model_timeline_on_fallback:
                    fallback_invalidated_future_actions += len(plan_by_action)
                    plan_by_action = {}
                    if buffer_tail_session is not None:
                        fallback_invalidated_buffer_count += 1
                    buffer_tail_session = None

                if replan_future is not None and replan_future.done():
                    replan_result = _resolve_exact_planner_future_result(
                        replan_future,
                        runner=runner,
                        config=config,
                        snapshot=replan_future_cache_snapshot,
                    )
                    (
                        history_base_session,
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
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
                    _restore_exact_runtime_cache_if_rejected(
                        replan_result,
                        runner=runner,
                        config=config,
                        snapshot=replan_future_cache_snapshot,
                    )
                    replan_future = None
                    replan_future_cache_snapshot = None

                future_buffer_depth_frames = int(
                    math.ceil(
                        _future_buffer_depth_actions(plan_by_action, next_action_to_execute=next_action_index)
                        / action_per_frame
                    )
                )
                future_buffer_depth_actions = _future_buffer_depth_actions(
                    plan_by_action,
                    next_action_to_execute=next_action_index,
                )
                should_submit_replan = _should_submit_exact_realtime_planner(
                    future_buffer_depth_actions=future_buffer_depth_actions,
                    future_buffer_depth_frames=future_buffer_depth_frames,
                    sequence_empty_plan_policy=sequence_empty_plan_policy,
                    replan_low_watermark_actions=replan_low_watermark_actions,
                )
                if replan_future is None and should_submit_replan:
                    (
                        replan_future,
                        replan_future_cache_snapshot,
                    ) = _maybe_submit_exact_planner_job_with_cache_snapshot(
                        executor=executor,
                        planner_mode=_resolve_exact_realtime_planner_mode(
                            planner_mode=planner_mode,
                            sequence_empty_plan_policy=sequence_empty_plan_policy,
                            pending_history=pending_history,
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
                replan_result = _resolve_exact_planner_future_result(
                    replan_future,
                    runner=runner,
                    config=config,
                    snapshot=replan_future_cache_snapshot,
                )
                (
                    history_base_session,
                    current_chunk_session,
                    buffer_tail_session,
                    plan_by_action,
                    pending_history,
                ) = _consume_exact_future_result(
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
                _restore_exact_runtime_cache_if_rejected(
                    replan_result,
                    runner=runner,
                    config=config,
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
                "planner_mode": planner_mode,
                "deadline_miss_policy": deadline_miss_policy,
                "fallback_absolute_tail_start": fallback_absolute_tail_start,
                "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
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
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": int(len(extension_records)),
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


def _consume_exact_future_result(
    result: dict[str, Any],
    *,
    config,
    plan_by_action: dict[int, PlannedControlStep],
    next_action_to_execute: int,
    pending_history: list[dict[str, Any]],
    history_base_session,
    current_chunk_session,
    buffer_tail_session,
    replan_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
    min_future_actions_to_accept_stale_chunk: int = 0,
):
    planned_steps = _planned_frames_to_step_actions(result["planned_frames"])
    (
        mergeable_planned_steps,
        chunk_boundary_dropped_actions,
        partial_stale_chunk_accepted_actions,
    ) = _drop_partial_stale_chunk_steps(
        planned_steps,
        next_action_to_execute=next_action_to_execute,
        min_future_actions_to_accept_stale_chunk=min_future_actions_to_accept_stale_chunk,
    )
    future_planned_steps = [
        step
        for step in mergeable_planned_steps
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]
    chunk_accepted = bool(future_planned_steps)
    result["trace"]["planned_action_indices"] = [
        int(step.absolute_action_index)
        for step in planned_steps
    ]
    result["trace"]["future_planned_actions"] = int(len(future_planned_steps))
    result["trace"]["stale_planned_actions"] = int(len(planned_steps) - len(future_planned_steps))
    result["trace"]["chunk_boundary_dropped_actions"] = int(chunk_boundary_dropped_actions)
    result["trace"]["partial_stale_chunk_accepted_actions"] = int(partial_stale_chunk_accepted_actions)
    result["trace"]["accepted_chunk"] = bool(chunk_accepted)
    if result["job_kind"] == "history_replan":
        replan_records.append(result["trace"])
        if chunk_accepted:
            submitted_through_frame = int(result["submitted_through_frame"])
            pending_history = [
                record for record in pending_history if int(record["absolute_frame_index"]) > submitted_through_frame
            ]
            current_chunk_session = result["session"]
            history_base_session = exact_sandbox._resolve_next_exact_history_base_session(
                config=config,
                result=result,
                history_base_session=history_base_session,
            )
            buffer_tail_session = result["buffer_tail_session"]
    else:
        extension_records.append(result["trace"])
        if chunk_accepted:
            buffer_tail_session = result["buffer_tail_session"]
        else:
            buffer_tail_session = None
    plan_by_action = _merge_future_step_actions(
        plan_by_action,
        mergeable_planned_steps,
        next_action_to_execute=next_action_to_execute,
    )
    return history_base_session, current_chunk_session, buffer_tail_session, plan_by_action, pending_history


def _drop_partial_stale_chunk_steps(
    planned_steps: list[PlannedControlStep],
    *,
    next_action_to_execute: int,
    min_future_actions_to_accept_stale_chunk: int = 0,
) -> tuple[list[PlannedControlStep], int, int]:
    """Keep exact-runtime chunks atomic when a result arrives after its first action is stale."""

    stale_steps = [
        step
        for step in planned_steps
        if int(step.absolute_action_index) < int(next_action_to_execute)
    ]
    future_steps = [
        step
        for step in planned_steps
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]
    if stale_steps and future_steps:
        if int(min_future_actions_to_accept_stale_chunk) > 0 and len(future_steps) >= int(
            min_future_actions_to_accept_stale_chunk
        ):
            return future_steps, 0, len(future_steps)
        return [], len(future_steps), 0
    return planned_steps, 0, 0


def _required_frame_action_indices(
    *,
    next_action_index: int,
    max_actions: int,
    action_per_frame: int,
) -> list[int]:
    frame_action_count = min(int(action_per_frame), max(0, int(max_actions) - int(next_action_index)))
    return [int(next_action_index) + offset for offset in range(frame_action_count)]


def _missing_plan_action_indices(
    plan_by_action: dict[int, PlannedControlStep],
    required_action_indices: list[int],
) -> list[int]:
    return [int(action_index) for action_index in required_action_indices if int(action_index) not in plan_by_action]


def _resolve_exact_realtime_planner_mode(
    *,
    planner_mode: str,
    sequence_empty_plan_policy: str,
    pending_history: list[dict[str, Any]],
) -> str:
    if sequence_empty_plan_policy == "wait_for_replan" and pending_history:
        return "history_only"
    return str(planner_mode)


def _should_submit_exact_realtime_planner(
    *,
    future_buffer_depth_actions: int,
    future_buffer_depth_frames: int,
    sequence_empty_plan_policy: str,
    replan_low_watermark_actions: int = 0,
) -> bool:
    """Return whether to submit another planner job.

    In fallback mode, positive K is an action-step low-watermark, not a video
    frame cadence: submit once the future action buffer has at most K actions
    remaining. Blocking mode keeps its historical empty-frame-buffer behavior.
    """
    if sequence_empty_plan_policy == "wait_for_replan":
        return int(future_buffer_depth_frames) <= 0
    if int(replan_low_watermark_actions) <= 0:
        return True
    return int(future_buffer_depth_actions) <= int(replan_low_watermark_actions)


def _should_submit_sequence_realtime_planner(
    *,
    planner_mode: str,
    future_buffer_depth_actions: int,
    sequence_empty_plan_policy: str,
    sequence_buffer_threshold: int,
    replan_low_watermark_actions: int = 0,
) -> bool:
    """Return whether a sequence-style rollout should queue the next full chunk."""

    if sequence_empty_plan_policy not in {"fallback", "wait_for_replan"}:
        raise ValueError(f"Unsupported sequence_empty_plan_policy={sequence_empty_plan_policy!r}.")
    if planner_mode == "history_only":
        return int(future_buffer_depth_actions) <= 0
    if planner_mode not in {"async_buffer", "async_mix", "async_history_first"}:
        raise ValueError(f"Unsupported sequence realtime planner_mode={planner_mode!r}.")
    threshold = (
        int(replan_low_watermark_actions)
        if int(replan_low_watermark_actions) > 0
        else int(sequence_buffer_threshold)
    )
    return int(future_buffer_depth_actions) <= threshold


def _exact_chunk_to_planned_steps(
    *,
    chunk,
    action_per_frame: int,
    frame_chunk_size: int,
    source: str,
    ready_monotonic_s: float,
) -> list[PlannedControlStep]:
    if chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    generation_frame_start = int(chunk.debug.get("generation_frame_start", 1))
    require_strict_startup_generation_frame(generation_frame_start)
    generation_action_start = _frame_index_to_action_start(generation_frame_start, action_per_frame)
    planned_steps: list[PlannedControlStep] = []
    for frame_offset in range(raw_actions.shape[0]):
        absolute_frame_index = generation_frame_start + frame_offset
        for action_offset in range(raw_actions.shape[1]):
            absolute_action_index = _frame_index_to_action_start(
                absolute_frame_index,
                action_per_frame,
            ) + action_offset
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(absolute_action_index),
                    generation_action_start=int(generation_action_start),
                    generation_frame_start=int(generation_frame_start),
                    source=str(source),
                    planner_step_index=int(chunk.session.policy_state.step_index),
                    ready_monotonic_s=ready_monotonic_s,
                    raw_action=raw_actions[frame_offset, action_offset].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
    return planned_steps


def _exact_startup_conditioning_history_record(
    *,
    chunk,
    initial_video_latents: torch.Tensor,
    initial_obs: dict[str, np.ndarray],
    action_per_frame: int,
    frame_chunk_size: int,
    conditioning_frame_index: int | None = None,
    raw_actions_override: np.ndarray | None = None,
    proprio_state: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    if chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    resolved_conditioning_frame_index = (
        int(chunk.debug.get("generation_frame_start", 0))
        if conditioning_frame_index is None
        else int(conditioning_frame_index)
    )
    generation_frame_start = int(chunk.debug.get("generation_frame_start", resolved_conditioning_frame_index))
    require_strict_startup_generation_frame(generation_frame_start)
    resolved_raw_actions = (
        raw_actions[0].detach().to(dtype=torch.float32).cpu().numpy()
        if raw_actions_override is None
        else np.asarray(raw_actions_override, dtype=np.float32)
    )
    raw_actions_valid = generation_frame_start <= resolved_conditioning_frame_index
    if not raw_actions_valid:
        resolved_raw_actions = np.zeros((0, int(raw_actions.shape[-1])), dtype=np.float32)
    record = {
        "absolute_frame_index": int(resolved_conditioning_frame_index),
        "obs": {key: np.array(value, copy=True) for key, value in initial_obs.items()},
        "obs_sequence": [],
        "raw_actions": resolved_raw_actions,
        "raw_actions_valid": bool(raw_actions_valid),
        "raw_action_dim": int(raw_actions.shape[-1]),
        "video_latents": initial_video_latents.detach(),
        "source": "startup_conditioning_frame",
    }
    if proprio_state is not None:
        record["proprio_state"] = exact_sandbox._proprio_state_to_numpy(proprio_state)
    return record


def _planned_frames_to_step_actions(planned_frames: list[Any]) -> list[PlannedControlStep]:
    planned_steps: list[PlannedControlStep] = []
    for planned_frame in planned_frames:
        raw_actions = np.asarray(planned_frame.raw_actions, dtype=np.float32)
        generation_frame_start = int(planned_frame.generation_frame_start)
        generation_action_start = _frame_index_to_action_start(generation_frame_start, int(raw_actions.shape[0]))
        for action_offset in range(raw_actions.shape[0]):
            absolute_action_index = (
                _frame_index_to_action_start(int(planned_frame.absolute_frame_index), int(raw_actions.shape[0]))
                + action_offset
            )
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(absolute_action_index),
                    generation_action_start=int(generation_action_start),
                    generation_frame_start=int(generation_frame_start),
                    source=str(planned_frame.source),
                    planner_step_index=planned_frame.planner_step_index,
                    ready_monotonic_s=planned_frame.ready_monotonic_s,
                    raw_action=np.array(raw_actions[action_offset], copy=True),
                )
            )
    return planned_steps


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
    planner_mode: str,
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
    sequence_empty_plan_policy: str,
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
    _validate_mot_startup_open_loop_support(
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
    mot_inference_backend = None
    if str(config.policy_variant.name) == "mot":
        mot_inference_backend = ensure_mot_inference_backend(pipeline, config)
        _print_stage(
            f"{rollout_label}_mot_inference_backend",
            **mot_inference_backend,
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
        "action_device": str(runtime_device) if str(config.policy_variant.name) == "mot" else None,
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
        "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
        "fallback_history_policy": str(fallback_history_policy),
        "sequence_buffer_threshold": int(sequence_buffer_threshold),
        "startup_open_loop_chunks": int(startup_open_loop_chunks),
        "replan_low_watermark_actions": int(replan_low_watermark_actions),
        "strict_mot_split_cache_startup": bool(_uses_strict_mot_split_cache_startup(config)),
        "strict_mot_one_frame_history": bool(_uses_strict_mot_one_frame_history(config)),
        "decoder_runtime": _collect_decoder_runtime_metadata(pipeline, config),
    }
    if mot_inference_backend is not None:
        load_report["mot_inference_backend"] = mot_inference_backend

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
    startup_env_init_frames = _sequence_startup_env_init_frames(
        config,
        raw_window_frames=raw_window_frames,
    )
    model_obs_window_frames = _sequence_model_obs_window_frames(
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
            startup_model_obs_window = _sequence_startup_model_obs_window(config, initial_obs_window)
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
            exact_sandbox._synchronize_devices(frontend_device, runtime_device)
            startup_prepare_s = time.perf_counter() - startup_prepare_t0

            session = runner.reset(
                task_text=(prompt,),
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )
            startup_infer_t0 = time.perf_counter()
            startup = _run_sequence_replan_job(
                runner=runner,
                session=session,
                obs_window=_copy_obs_window(startup_model_obs_window),
                prompt=prompt,
                task_id=int(task_id),
                episode_idx=int(episode_idx),
                config=config,
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
            )
            exact_sandbox._synchronize_devices(runtime_device)
            startup_infer_s = time.perf_counter() - startup_infer_t0
            _print_stage(
                f"{rollout_label}_startup_done",
                startup_prepare_s=float(startup_prepare_s),
                startup_infer_s=float(startup_infer_s),
            )

        session = startup["session"]
        next_generation_action_start = int(startup["next_generation_action_start"])
        mot_non_joint_sequence = _is_mot_non_joint_two_stream(config)
        history_base_session = session if mot_non_joint_sequence else _clone_sequence_session(session)
        history_base_cache_snapshot = startup.get("runtime_cache_snapshot")
        history_generation_action_start = int(next_generation_action_start)
        buffer_tail_session = session if mot_non_joint_sequence else _clone_sequence_session(session)
        buffer_tail_cache_snapshot = startup.get("runtime_cache_snapshot")
        buffer_tail_generation_action_start = int(next_generation_action_start)
        plan_by_action = _merge_future_step_actions({}, startup["planned_steps"], next_action_to_execute=0)
        current_obs = _copy_obs_record(initial_obs_window[-1])
        obs_window = _copy_obs_window(initial_obs_window)
        model_obs_window = _sequence_startup_model_obs_window(config, initial_obs_window)
        sequence_fallback_state = SequenceFallbackHistoryState(policy=fallback_history_policy)
        sequence_clean_actions_required = max(1, int(config.data.action_schema.action_horizon))
        extension_records: list[dict[str, Any]] = []
        startup_open_loop_s = 0.0
        if startup_open_loop_chunks > 0:
            startup_open_loop_t0 = time.perf_counter()
            for _ in range(int(startup_open_loop_chunks)):
                extension = _run_sequence_replan_job(
                    runner=runner,
                    session=(
                        buffer_tail_session
                        if mot_non_joint_sequence
                        else _clone_sequence_session(buffer_tail_session)
                    ),
                    obs_window=_copy_obs_window(model_obs_window),
                    prompt=prompt,
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    config=config,
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    generation_action_start=buffer_tail_generation_action_start,
                    source="open_loop_extension",
                    reset_observation_conditioned_session=False,
                    use_observation_update=False,
                    runtime_cache_snapshot=buffer_tail_cache_snapshot,
                    preserve_rng_state=True,
                )
                extension_records.append(extension["trace"])
                buffer_tail_session = (
                    extension["session"]
                    if mot_non_joint_sequence
                    else _clone_sequence_session(extension["session"])
                )
                buffer_tail_cache_snapshot = extension.get("runtime_cache_snapshot")
                buffer_tail_generation_action_start = int(extension["next_generation_action_start"])
                session = (
                    buffer_tail_session
                    if mot_non_joint_sequence
                    else _clone_sequence_session(buffer_tail_session)
                )
                next_generation_action_start = int(buffer_tail_generation_action_start)
                plan_by_action = _merge_future_step_actions(
                    plan_by_action,
                    extension["planned_steps"],
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
            replan_future: Future[dict[str, Any]] | None = None
            while executed_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
                    if _is_mot_non_joint_two_stream(config):
                        future_steps = _annotate_sequence_planner_acceptance(
                            result,
                            next_action_to_execute=next_action_index,
                        )
                        replan_records.append(result["trace"])
                        if future_steps:
                            if bool(result["trace"].get("use_observation_update", True)):
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = _drop_sequence_future_actions_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.get("runtime_cache_snapshot")
                                history_generation_action_start = int(result["next_generation_action_start"])
                                buffer_tail_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.get("runtime_cache_snapshot")
                                buffer_tail_generation_action_start = int(result["next_generation_action_start"])
                            else:
                                buffer_tail_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.get("runtime_cache_snapshot")
                                buffer_tail_generation_action_start = int(result["next_generation_action_start"])
                            plan_by_action = _merge_future_step_actions(
                                plan_by_action,
                                future_steps,
                                next_action_to_execute=next_action_index,
                            )
                            session = _sequence_session_ref(
                                buffer_tail_session,
                                share_session=mot_non_joint_sequence,
                            )
                            next_generation_action_start = int(buffer_tail_generation_action_start)
                    else:
                        replan_records.append(result["trace"])
                        session = result["session"]
                        next_generation_action_start = int(result["next_generation_action_start"])
                        plan_by_action = _merge_future_step_actions(
                            plan_by_action,
                            result["planned_steps"],
                            next_action_to_execute=next_action_index,
                        )
                    replan_future = None

                planned_step = plan_by_action.pop(next_action_index, None)
                wait_for_plan_s = 0.0
                if planned_step is None:
                    if sequence_empty_plan_policy == "wait_for_replan":
                        wait_t0 = time.perf_counter()
                        if replan_future is None:
                            blocking_replan_count += 1
                            if _is_mot_non_joint_two_stream(config):
                                if (
                                    buffer_tail_session is not None
                                    and _sequence_buffer_tail_ready_for_history_promotion(
                                        config=config,
                                        next_action_index=next_action_index,
                                        buffer_tail_generation_action_start=buffer_tail_generation_action_start,
                                        history_generation_action_start=history_generation_action_start,
                                    )
                                ):
                                    history_base_session = _sequence_session_ref(
                                        buffer_tail_session,
                                        share_session=mot_non_joint_sequence,
                                    )
                                    history_base_cache_snapshot = buffer_tail_cache_snapshot
                                    history_generation_action_start = int(buffer_tail_generation_action_start)
                                blocking_session = _sequence_session_ref(
                                    history_base_session,
                                    share_session=mot_non_joint_sequence,
                                )
                                blocking_generation_action_start = int(history_generation_action_start)
                                blocking_cache_snapshot = history_base_cache_snapshot
                                blocking_condition_frame_start = _mot_condition_frame_start_for_generation(
                                    config=config,
                                    generation_action_start=blocking_generation_action_start,
                                )
                            else:
                                blocking_session = session
                                blocking_generation_action_start = int(next_generation_action_start)
                                blocking_cache_snapshot = None
                                blocking_condition_frame_start = None
                            # The restored history snapshot already contains
                            # the correct MoT action-cache prefix. Rewinding
                            # here mutates chunk-by-chunk parity.
                            result = _run_sequence_replan_job(
                                runner=runner,
                                session=blocking_session,
                                obs_window=_copy_obs_window(model_obs_window),
                                prompt=prompt,
                                task_id=int(task_id),
                                episode_idx=int(episode_idx),
                                config=config,
                                frontend_device=frontend_device,
                                runtime_device=runtime_device,
                                generation_action_start=blocking_generation_action_start,
                                source="blocking_replan",
                                runtime_cache_snapshot=blocking_cache_snapshot,
                                mot_condition_frame_start=blocking_condition_frame_start,
                            )
                        else:
                            result = replan_future.result()
                            replan_future = None
                        wait_for_plan_s = time.perf_counter() - wait_t0
                        wait_for_plan_count += 1
                        wait_for_plan_total_s += wait_for_plan_s
                        result["trace"]["blocking_wait_action_index"] = int(next_action_index)
                        result["trace"]["blocking_wait_s"] = float(wait_for_plan_s)
                        if _is_mot_non_joint_two_stream(config):
                            future_steps = _annotate_sequence_planner_acceptance(
                                result,
                                next_action_to_execute=next_action_index,
                            )
                            replan_records.append(result["trace"])
                            if future_steps:
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = _drop_sequence_future_actions_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.get("runtime_cache_snapshot")
                                history_generation_action_start = int(result["next_generation_action_start"])
                                buffer_tail_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.get("runtime_cache_snapshot")
                                buffer_tail_generation_action_start = int(result["next_generation_action_start"])
                                plan_by_action = _merge_future_step_actions(
                                    plan_by_action,
                                    future_steps,
                                    next_action_to_execute=next_action_index,
                                )
                                session = _sequence_session_ref(
                                    buffer_tail_session,
                                    share_session=mot_non_joint_sequence,
                                )
                                next_generation_action_start = int(buffer_tail_generation_action_start)
                        else:
                            session, next_generation_action_start, plan_by_action = _apply_sequence_replan_result(
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
                    elif sequence_empty_plan_policy == "fallback":
                        action = exact_sandbox._build_fallback_frame_actions(
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
                    action = _materialize_sequence_control_action(
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
                obs_window = _append_obs_window_record(
                    obs_window,
                    current_obs,
                    max_window_frames=raw_window_frames,
                )
                history_append_result = _maybe_append_sequence_model_observation(
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
                    "history_append_result": history_append_result,
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
                if _action_advances_model_timeline(
                    source,
                    fallback_history_policy=fallback_history_policy,
                ):
                    next_action_index += 1
                if done or executed_action_index >= max_actions:
                    break

                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
                    if _is_mot_non_joint_two_stream(config):
                        future_steps = _annotate_sequence_planner_acceptance(
                            result,
                            next_action_to_execute=next_action_index,
                        )
                        replan_records.append(result["trace"])
                        if future_steps:
                            if bool(result["trace"].get("use_observation_update", True)):
                                replace_from_action = min(int(step.absolute_action_index) for step in future_steps)
                                plan_by_action = _drop_sequence_future_actions_from(
                                    plan_by_action,
                                    replace_from_action=replace_from_action,
                                )
                                history_base_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                history_base_cache_snapshot = result.get("runtime_cache_snapshot")
                                history_generation_action_start = int(result["next_generation_action_start"])
                                buffer_tail_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.get("runtime_cache_snapshot")
                                buffer_tail_generation_action_start = int(result["next_generation_action_start"])
                            else:
                                buffer_tail_session = _sequence_session_ref(
                                    result["session"],
                                    share_session=mot_non_joint_sequence,
                                )
                                buffer_tail_cache_snapshot = result.get("runtime_cache_snapshot")
                                buffer_tail_generation_action_start = int(result["next_generation_action_start"])
                            plan_by_action = _merge_future_step_actions(
                                plan_by_action,
                                future_steps,
                                next_action_to_execute=next_action_index,
                            )
                            session = _sequence_session_ref(
                                buffer_tail_session,
                                share_session=mot_non_joint_sequence,
                            )
                            next_generation_action_start = int(buffer_tail_generation_action_start)
                    else:
                        replan_records.append(result["trace"])
                        session = result["session"]
                        next_generation_action_start = int(result["next_generation_action_start"])
                        plan_by_action = _merge_future_step_actions(
                            plan_by_action,
                            result["planned_steps"],
                            next_action_to_execute=next_action_index,
                        )
                    replan_future = None

                remaining_buffer = _future_buffer_depth_actions(plan_by_action, next_action_to_execute=next_action_index)
                if planner_mode not in {"history_only", "async_buffer", "async_mix", "async_history_first"}:
                    raise ValueError(f"Unsupported planner_mode={planner_mode!r} for policy_variant={config.policy_variant.name!r}.")
                should_submit = _should_submit_sequence_realtime_planner(
                    planner_mode=planner_mode,
                    future_buffer_depth_actions=remaining_buffer,
                    sequence_empty_plan_policy=sequence_empty_plan_policy,
                    sequence_buffer_threshold=sequence_buffer_threshold,
                    replan_low_watermark_actions=replan_low_watermark_actions,
                )
                if replan_future is None and should_submit:
                    obs_snapshot = _copy_obs_window(model_obs_window)
                    if _is_mot_non_joint_two_stream(config):
                        if (
                            buffer_tail_session is not None
                            and _sequence_buffer_tail_ready_for_history_promotion(
                                config=config,
                                next_action_index=next_action_index,
                                buffer_tail_generation_action_start=buffer_tail_generation_action_start,
                                history_generation_action_start=history_generation_action_start,
                            )
                        ):
                            history_base_session = _sequence_session_ref(
                                buffer_tail_session,
                                share_session=mot_non_joint_sequence,
                            )
                            history_base_cache_snapshot = buffer_tail_cache_snapshot
                            history_generation_action_start = int(buffer_tail_generation_action_start)
                        history_ready = _mot_history_replan_ready(
                            config=config,
                            next_action_index=next_action_index,
                            generation_action_start=history_generation_action_start,
                        )
                        use_observation_update = (
                            planner_mode == "history_only"
                            or (planner_mode in {"async_mix", "async_history_first"} and history_ready)
                        )
                        if use_observation_update:
                            submit_session = _sequence_session_ref(
                                history_base_session,
                                share_session=mot_non_joint_sequence,
                            )
                            submit_generation_action_start = int(history_generation_action_start)
                            submit_cache_snapshot = history_base_cache_snapshot
                            submit_condition_frame_start = _mot_condition_frame_start_for_generation(
                                config=config,
                                generation_action_start=submit_generation_action_start,
                            )
                        else:
                            submit_session = _sequence_session_ref(
                                buffer_tail_session,
                                share_session=mot_non_joint_sequence,
                            )
                            submit_generation_action_start = int(buffer_tail_generation_action_start)
                            submit_cache_snapshot = buffer_tail_cache_snapshot
                            submit_condition_frame_start = None
                    else:
                        use_observation_update = not _should_use_mot_open_loop_extension(
                            config=config,
                            planner_mode=planner_mode,
                            remaining_buffer_actions=remaining_buffer,
                        )
                        submit_session = session
                        submit_generation_action_start = int(next_generation_action_start)
                        submit_cache_snapshot = None
                        submit_condition_frame_start = None
                    replan_future = executor.submit(
                        _run_sequence_replan_job,
                        runner=runner,
                        session=submit_session,
                        obs_window=obs_snapshot,
                        prompt=prompt,
                        task_id=int(task_id),
                        episode_idx=int(episode_idx),
                        config=config,
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        generation_action_start=submit_generation_action_start,
                        source="history_replan" if use_observation_update else "open_loop_extension",
                        reset_observation_conditioned_session=use_observation_update,
                        use_observation_update=use_observation_update,
                        runtime_cache_snapshot=submit_cache_snapshot,
                        mot_condition_frame_start=submit_condition_frame_start,
                        # Async observation-conditioned MoT replans can be
                        # launched from a speculative buffer-tail session.
                        # Trim that future action K/V suffix to the chunk
                        # being replaced; blocking/history-only replans keep
                        # their accepted cache prefix untouched for parity.
                        mot_action_cache_rewind_frame_start=_mot_action_cache_rewind_for_sequence_submit(
                            config=config,
                            planner_mode=planner_mode,
                            use_observation_update=use_observation_update,
                            condition_frame_start=submit_condition_frame_start,
                        ),
                        preserve_rng_state=not bool(use_observation_update),
                    )
                elif replan_future is not None:
                    skipped_replan_submissions += 1

            if replan_future is not None and replan_future.done():
                result = replan_future.result()
                if _is_mot_non_joint_two_stream(config):
                    future_steps = _annotate_sequence_planner_acceptance(
                        result,
                        next_action_to_execute=next_action_index,
                    )
                    replan_records.append(result["trace"])
                    if future_steps:
                        plan_by_action = _merge_future_step_actions(
                            plan_by_action,
                            future_steps,
                            next_action_to_execute=next_action_index,
                        )
                else:
                    replan_records.append(result["trace"])
                    session = result["session"]
                    next_generation_action_start = int(result["next_generation_action_start"])
                    plan_by_action = _merge_future_step_actions(
                        plan_by_action,
                        result["planned_steps"],
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
                "strict_mot_one_frame_history": bool(_uses_strict_mot_one_frame_history(config)),
                "planner_mode": planner_mode,
                "sequence_buffer_threshold": int(sequence_buffer_threshold),
                "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
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
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": int(len(extension_records)),
                "hidden_fallback_period_count": int(sequence_fallback_state.fallback_quarantine_count),
                "hidden_fallback_history_actions": int(sequence_fallback_state.hidden_fallback_actions),
                "hidden_washout_history_actions": int(sequence_fallback_state.hidden_washout_actions),
                "hidden_history_actions": int(sequence_fallback_state.hidden_history_actions),
                "policy_variant": str(config.policy_variant.name),
                "startup_plan_trace": startup["trace"],
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


def _validate_strict_mot_split_cache_startup_inputs(
    *,
    config,
    source: str,
    generation_action_start: int,
    video_latents: object,
) -> None:
    if not _uses_strict_mot_split_cache_startup(config) or str(source) != "startup_plan":
        return
    if int(generation_action_start) != 0:
        raise ValueError(
            "M5 strict split-cache startup expects generation_action_start=0 so executable "
            f"actions start at action index 0; got {generation_action_start}."
        )
    if not isinstance(video_latents, torch.Tensor) or video_latents.ndim != 5:
        raise ValueError(
            "M5 strict split-cache startup expects tensor video_latents with shape [B, C, T, H, W], "
            f"got {type(video_latents).__name__}."
        )
    latent_context_frames = int(video_latents.shape[2])
    if latent_context_frames != 1:
        raise ValueError(
            "M5 strict split-cache startup expects exactly one latent context frame before "
            f"the first generated chunk; got {latent_context_frames}. This would break "
            "target_alignment=next_after_context parity."
        )


def _run_sequence_replan_job(
    *,
    runner: VariantRolloutRunner,
    session,
    obs_window: list[dict[str, np.ndarray]],
    prompt: str,
    task_id: int,
    episode_idx: int,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    generation_action_start: int,
    source: str,
    reset_observation_conditioned_session: bool = True,
    use_observation_update: bool = True,
    runtime_cache_snapshot: dict[str, Any] | None = None,
    mot_condition_frame_start: int | None = None,
    mot_action_cache_rewind_frame_start: int | None = None,
    preserve_rng_state: bool = False,
) -> dict[str, Any]:
    rng_snapshot = _snapshot_rng_state() if preserve_rng_state else None
    with torch.inference_mode():
        if runtime_cache_snapshot is not None:
            _restore_exact_runtime_cache_snapshot(
                runner=runner,
                config=config,
                snapshot=runtime_cache_snapshot,
            )
        prepare_t0 = time.perf_counter()
        rollout_inputs = rollout_runtime.prepare_rollout_observation_inputs(
            runner.pipeline,
            views=libero_rollout.libero_observation_window_to_views(
                obs_window,
                device=frontend_device,
            ),
            task_text=(prompt,),
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        exact_sandbox._synchronize_devices(frontend_device, runtime_device)
        prepare_s = time.perf_counter() - prepare_t0
        _validate_strict_mot_split_cache_startup_inputs(
            config=config,
            source=source,
            generation_action_start=int(generation_action_start),
            video_latents=rollout_inputs.get("video_latents"),
        )

        infer_t0 = time.perf_counter()
        inference_session = (
            _resolve_observation_conditioned_replan_session(
                runner=runner,
                session=session,
                config=config,
            )
            if reset_observation_conditioned_session
            else session
        )
        reset_session_for_replan = inference_session is not session
        infer_extra = rollout_runtime.build_sequence_rollout_infer_extra(
            config=config,
            prompt=prompt,
            generation_action_start=int(generation_action_start),
            runtime_device=runtime_device,
            task_id=int(task_id),
            episode_idx=int(episode_idx),
        )
        mot_runtime_route = resolve_mot_runtime_route(config)
        if mot_runtime_route.is_mot and mot_runtime_route.supports_realtime_history_controls:
            infer_extra["mot_skip_observation_update"] = not bool(use_observation_update)
            if mot_condition_frame_start is not None:
                infer_extra["mot_condition_frame_start"] = int(mot_condition_frame_start)
            if mot_action_cache_rewind_frame_start is not None:
                infer_extra["mot_action_cache_rewind_frame_start"] = int(mot_action_cache_rewind_frame_start)
        elif mot_runtime_route.is_mot and not bool(use_observation_update):
            raise ValueError(
                "M5 runtime route does not support split-cache open-loop controls: "
                f"{mot_runtime_route.to_report()}"
            )
        step_output = runner.infer_step(
            session=inference_session,
            context=PolicyInferContext(
                state=libero_rollout.build_libero_state_history(
                    obs_window,
                    state_horizon=int(config.data.action_schema.state_horizon),
                    state_encoding=config.data.action_target.state_encoding,
                ).unsqueeze(0).to(device=runtime_device),
                extra=infer_extra,
            ),
            video_latents=rollout_inputs["video_latents"],
            canonical_video=None,
        )
        exact_sandbox._synchronize_devices(runtime_device)
        infer_s = time.perf_counter() - infer_t0
        output_runtime_cache_snapshot = _snapshot_sequence_runtime_cache(
            runner=runner,
            config=config,
            session=step_output.session,
        )

    policy_aux = step_output.infer_output.policy_output.aux
    mot_cache_debug = policy_aux.get("mot_cache_debug")
    if not isinstance(mot_cache_debug, dict):
        mot_cache_debug = {}
    sequence_context = step_output.infer_output.policy_output.decoder_sequence_context
    video_condition_window = None if sequence_context is None else sequence_context.video_condition_window
    video_condition_metadata = {} if video_condition_window is None else dict(video_condition_window.metadata)
    predicted_latents = policy_aux.get("predicted_latents")
    action_pred, action_plan_metadata = _decoder_output_to_rollout_action_plan(
        step_output.infer_output.decoder_output
    )
    _advance_decoder_state_to_rollout_commit(step_output.session, action_plan_metadata)
    ready_monotonic_s = time.perf_counter()
    planned_steps = _sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs=obs_window[-1],
        generation_action_start=generation_action_start,
        execution_action_offset=_sequence_execution_action_offset(config),
        source=source,
        planner_step_index=(
            None
            if step_output.session.policy_state is None
            else int(step_output.session.policy_state.step_index)
        ),
        ready_monotonic_s=ready_monotonic_s,
        action_target_representation=config.data.action_target.representation,
        rotation_representation=str(config.data.action_target.rotation_representation),
    )
    next_generation_action_start = generation_action_start + int(action_pred.shape[0])
    job_result = {
        "session": step_output.session,
        "runtime_cache_snapshot": output_runtime_cache_snapshot,
        "planned_steps": planned_steps,
        "next_generation_action_start": int(next_generation_action_start),
        "trace": {
            "job_kind": "history_replan",
            "source": str(source),
            "reset_observation_conditioned_session": bool(reset_session_for_replan),
            "use_observation_update": bool(use_observation_update),
            "observed_action_index": int(max(-1, generation_action_start - 1)),
            "history_frame_count": int(len(obs_window)),
            "generation_action_start": int(generation_action_start),
            "execution_action_offset": int(_sequence_execution_action_offset(config)),
            "mot_condition_frame_start": mot_condition_frame_start,
            "mot_action_cache_rewind_frame_start": mot_action_cache_rewind_frame_start,
            "mot_runtime_route": mot_runtime_route.to_report() if mot_runtime_route.is_mot else None,
            "model_generation_frame_start": _json_scalar_from_tensor(policy_aux.get("generation_frame_start")),
            "mot_chunk_origin_frame": _json_scalar_from_tensor(mot_cache_debug.get("chunk_origin_frame")),
            "mot_current_action_frame_start": _json_scalar_from_tensor(
                mot_cache_debug.get("current_action_frame_start")
            ),
            "preserve_rng_state": bool(preserve_rng_state),
            "planned_action_ids": [int(plan.absolute_action_index) for plan in planned_steps],
            "prepare_s": float(prepare_s),
            "warmup_s": 0.0,
            "infer_s": float(infer_s),
            "total_latency_s": float(prepare_s + infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
            "video_condition_source": policy_aux.get("video_condition_source"),
            "video_condition_uses_future_ground_truth": policy_aux.get("video_condition_uses_future_ground_truth"),
            "video_condition_frame_start": video_condition_metadata.get("frame_start"),
            "video_condition_sample_seed": video_condition_metadata.get("sample_seed"),
            "video_condition_observed_prefix_anchor": video_condition_metadata.get("observed_prefix_anchor"),
            "video_condition_observed_prefix_start_index": video_condition_metadata.get("observed_prefix_start_index"),
            "predicted_video_latents_shape": (
                list(predicted_latents.shape)
                if isinstance(predicted_latents, torch.Tensor)
                else None
            ),
            **_collect_decoder_runtime_metadata(runner.pipeline, config),
            **action_plan_metadata,
            "decoder_sampled_new_chunk": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get("sampled_new_chunk")
            ),
            "decoder_num_inference_steps": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get("num_inference_steps")
            ),
            "decoder_current_action_index": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get("current_action_index")
            ),
        },
    }
    _restore_rng_state(rng_snapshot)
    return job_result


def _resolve_observation_conditioned_replan_session(
    *,
    runner: _RolloutRunnerLike,
    session,
    config,
):
    mot_runtime_route = resolve_mot_runtime_route(config)
    if not mot_runtime_route.is_mot:
        return session
    if mot_runtime_route.uses_split_cache_rollout:
        # The split-cache route owns a Method-1-style rollout cache. Live
        # replans write the real observation window into that cache, so the
        # session is the continuity boundary.
        return session
    if mot_runtime_route.uses_native_packed_rollout:
        # Native packed inference carries history in PolicyInferState rather
        # than the shared transformer's slot-pool cache.
        return session
    # Older MoT video-prefill-style modes tie the cache directly to the
    # current observation window, so rebuild them for each live replan.
    return runner.reset(
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )


def _should_use_mot_open_loop_extension(
    *,
    config,
    planner_mode: str,
    remaining_buffer_actions: int,
) -> bool:
    if not resolve_mot_runtime_route(config).supports_realtime_history_controls:
        return False
    if planner_mode not in {"async_buffer", "async_mix", "async_history_first"}:
        return False
    return int(remaining_buffer_actions) > 0


def _validate_mot_startup_open_loop_support(*, config, startup_open_loop_chunks: int):
    mot_runtime_route = resolve_mot_runtime_route(config)
    if (
        mot_runtime_route.is_mot
        and int(startup_open_loop_chunks) > 0
        and not mot_runtime_route.supports_realtime_history_controls
    ):
        raise ValueError(
            "M5 runtime route does not support startup open-loop extension because "
            "it has no split-cache observation-skip/rewind controls. Use "
            "`startup_open_loop_chunks=0`, or a split-cache M5 route such as "
            "`video_then_action` / `decoupled_same_step`. Runtime route: "
            f"{mot_runtime_route.to_report()}"
        )
    return mot_runtime_route


def _mot_action_cache_rewind_for_sequence_submit(
    *,
    config,
    planner_mode: str,
    use_observation_update: bool,
    condition_frame_start: int | None,
) -> int | None:
    if condition_frame_start is None:
        return None
    if not bool(use_observation_update):
        return None
    if not resolve_mot_runtime_route(config).supports_realtime_history_controls:
        return None
    if planner_mode not in {"async_mix", "async_history_first"}:
        return None
    return int(condition_frame_start)


def _apply_sequence_replan_result(
    *,
    result: dict[str, Any],
    replan_records: list[dict[str, Any]],
    plan_by_action: dict[int, PlannedControlStep],
    next_action_to_execute: int,
) -> tuple[Any, int, dict[int, PlannedControlStep]]:
    replan_records.append(result["trace"])
    return (
        result["session"],
        int(result["next_generation_action_start"]),
        _merge_future_step_actions(
            plan_by_action,
            result["planned_steps"],
            next_action_to_execute=next_action_to_execute,
        ),
    )


def _sequence_future_planned_steps(
    result: dict[str, Any],
    *,
    next_action_to_execute: int,
) -> list[PlannedControlStep]:
    return [
        step
        for step in result["planned_steps"]
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]


def _drop_sequence_future_actions_from(
    plan_by_action: dict[int, PlannedControlStep],
    *,
    replace_from_action: int,
) -> dict[int, PlannedControlStep]:
    return {
        int(action_index): plan
        for action_index, plan in plan_by_action.items()
        if int(action_index) < int(replace_from_action)
    }


def _annotate_sequence_planner_acceptance(
    result: dict[str, Any],
    *,
    next_action_to_execute: int,
) -> list[PlannedControlStep]:
    future_steps = _sequence_future_planned_steps(
        result,
        next_action_to_execute=next_action_to_execute,
    )
    result["trace"]["future_planned_actions"] = int(len(future_steps))
    result["trace"]["stale_planned_actions"] = int(
        len(result["planned_steps"]) - len(future_steps)
    )
    result["trace"]["accepted_chunk"] = bool(future_steps)
    return future_steps


def _mot_history_replan_ready(
    *,
    config,
    next_action_index: int,
    generation_action_start: int,
) -> bool:
    execution_start = int(generation_action_start) - _sequence_execution_action_offset(config)
    return int(next_action_index) >= int(execution_start)


def _sequence_buffer_tail_ready_for_history_promotion(
    *,
    config,
    next_action_index: int,
    buffer_tail_generation_action_start: int,
    history_generation_action_start: int,
) -> bool:
    if int(buffer_tail_generation_action_start) <= int(history_generation_action_start):
        return False
    execution_start = int(buffer_tail_generation_action_start) - _sequence_execution_action_offset(config)
    return int(next_action_index) >= int(execution_start)


def _mot_condition_frame_start_for_generation(*, config, generation_action_start: int) -> int:
    action_per_frame = _sequence_actions_per_frame(config)
    return int(generation_action_start) // int(action_per_frame)


def _decoder_output_to_rollout_action_plan(decoder_output) -> tuple[np.ndarray, dict[str, Any]]:
    action_chunk = decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu()
    current_action = decoder_output.aux.get("current_action")
    if isinstance(current_action, torch.Tensor):
        current_action_index = _resolve_decoder_current_action_index(decoder_output)
        rollout_chunk_steps = _resolve_decoder_rollout_chunk_steps(decoder_output)
        if current_action_index < 0:
            raise ValueError(f"Decoder current action index must be non-negative, got {current_action_index}.")
        commit_end_index = min(
            int(action_chunk.shape[0]),
            max(int(current_action_index) + 1, int(rollout_chunk_steps)),
        )
        planned_chunk = action_chunk[int(current_action_index) : commit_end_index]
        source = "decoder_current_action_rollout_chunk"
        if int(planned_chunk.shape[0]) == 0:
            planned_chunk = _current_action_tensor_to_chunk(current_action)
            commit_end_index = int(current_action_index) + int(planned_chunk.shape[0])
            source = "decoder_current_action"
        return planned_chunk.numpy(), {
            "action_plan_source": source,
            "decoder_rollout_chunk_steps": int(rollout_chunk_steps),
            "decoder_rollout_commit_start_index": int(current_action_index),
            "decoder_rollout_commit_end_index": int(commit_end_index),
            "decoder_rollout_committed_actions": int(planned_chunk.shape[0]),
        }
    return (
        action_chunk.numpy(),
        {
            "action_plan_source": "decoder_action_chunk",
            "decoder_rollout_chunk_steps": None,
            "decoder_rollout_commit_start_index": None,
            "decoder_rollout_commit_end_index": None,
            "decoder_rollout_committed_actions": int(action_chunk.shape[0]),
        },
    )


def _current_action_tensor_to_chunk(current_action: torch.Tensor) -> torch.Tensor:
    current_action = current_action.detach().to(dtype=torch.float32).cpu()
    if current_action.ndim == 1:
        return current_action.unsqueeze(0)
    if current_action.ndim == 2:
        return current_action[:1]
    raise ValueError(f"Expected current_action shape [D] or [B, D], got {tuple(current_action.shape)}.")


def _resolve_decoder_current_action_index(decoder_output) -> int:
    current_action_index = decoder_output.aux.get("current_action_index")
    if current_action_index is not None:
        return int(_json_scalar_from_tensor(current_action_index))
    next_state = getattr(decoder_output, "next_state", None)
    if next_state is not None and hasattr(next_state, "step_within_chunk"):
        return max(0, int(next_state.step_within_chunk) - 1)
    return 0


def _resolve_decoder_rollout_chunk_steps(decoder_output) -> int:
    rollout_chunk_steps = decoder_output.aux.get("rollout_chunk_steps")
    if rollout_chunk_steps is None:
        next_state = getattr(decoder_output, "next_state", None)
        rollout_chunk_steps = (
            getattr(next_state, "aux", {}).get("rollout_chunk_steps")
            if next_state is not None
            else None
        )
    if rollout_chunk_steps is None:
        return 1
    return max(1, int(_json_scalar_from_tensor(rollout_chunk_steps)))


def _advance_decoder_state_to_rollout_commit(session, action_plan_metadata: dict[str, Any]) -> None:
    commit_end_index = action_plan_metadata.get("decoder_rollout_commit_end_index")
    if commit_end_index is None:
        return
    policy_state = getattr(session, "policy_state", None)
    decoder_state = getattr(policy_state, "decoder_state", None)
    if decoder_state is None or not hasattr(decoder_state, "step_within_chunk"):
        return
    decoder_state.step_within_chunk = max(int(decoder_state.step_within_chunk), int(commit_end_index))


def _json_scalar_from_tensor(value) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _collect_decoder_runtime_metadata(pipeline, config) -> dict[str, Any]:
    decoder = getattr(pipeline, "action_decoder", None)
    generation_backend = getattr(decoder, "generation_backend", None)
    return {
        "action_decoder_class": None if decoder is None else decoder.__class__.__name__,
        "config_action_num_inference_steps": int(config.inference.action_num_inference_steps),
        "config_video_num_inference_steps": int(config.inference.video_num_inference_steps),
        "decoder_generation_num_sampling_steps": (
            None
            if generation_backend is None
            else int(getattr(generation_backend, "num_sampling_steps", 0) or 0)
        ),
        "decoder_rollout_chunk_steps": (
            None
            if decoder is None or not hasattr(decoder, "rollout_chunk_steps")
            else int(getattr(decoder, "rollout_chunk_steps"))
        ),
    }


def _sequence_chunk_to_planned_steps(
    *,
    action_pred: np.ndarray,
    reference_obs: dict[str, np.ndarray],
    generation_action_start: int,
    execution_action_offset: int = 0,
    source: str,
    planner_step_index: int | None,
    ready_monotonic_s: float,
    action_target_representation: ActionTargetRepresentation | str,
    rotation_representation: str,
) -> list[PlannedControlStep]:
    representation = ActionTargetRepresentation(action_target_representation)
    planned_steps: list[PlannedControlStep] = []
    execution_start = int(generation_action_start) - max(0, int(execution_action_offset))
    if representation == ActionTargetRepresentation.RAW:
        for action_offset in range(action_pred.shape[0]):
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(execution_start + action_offset),
                    generation_action_start=int(generation_action_start),
                    source=str(source),
                    planner_step_index=planner_step_index,
                    ready_monotonic_s=ready_monotonic_s,
                    raw_action=np.asarray(action_pred[action_offset], dtype=np.float32).copy(),
                    desired_position=None,
                    desired_quaternion=None,
                    desired_gripper=None,
                )
            )
        return planned_steps

    if representation != ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
        raise ValueError(f"Unsupported action target representation for sequence rollout: {representation!r}.")

    desired_pose_targets = libero_rollout.reconstruct_libero_pose_targets(
        action_pred,
        reference_observation=reference_obs,
        rotation_representation=rotation_representation,
    )
    for action_offset in range(action_pred.shape[0]):
        desired_gripper = None
        if desired_pose_targets.gripper is not None:
            desired_gripper = desired_pose_targets.gripper[action_offset].detach().to(dtype=torch.float32).cpu().numpy()
        planned_steps.append(
            PlannedControlStep(
                absolute_action_index=int(execution_start + action_offset),
                generation_action_start=int(generation_action_start),
                source=str(source),
                planner_step_index=planner_step_index,
                ready_monotonic_s=ready_monotonic_s,
                raw_action=None,
                desired_position=desired_pose_targets.position[action_offset].detach().to(dtype=torch.float32).cpu().numpy(),
                desired_quaternion=desired_pose_targets.quaternion[action_offset].detach().to(dtype=torch.float32).cpu().numpy(),
                desired_gripper=desired_gripper,
            )
        )
    return planned_steps


def _sequence_execution_action_offset(config) -> int:
    if str(config.policy_variant.name) != "mot":
        return 0
    return resolve_mot_sequence_execution_action_offset(
        config,
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
    )


def _sequence_actions_per_frame(config) -> int:
    return resolve_mot_sequence_actions_per_frame(
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
    )


def _materialize_sequence_control_action(
    planned_step: PlannedControlStep,
    *,
    current_obs: dict[str, np.ndarray],
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    if planned_step.raw_action is not None:
        return np.clip(np.asarray(planned_step.raw_action, dtype=np.float32), -1.0, 1.0)
    if planned_step.desired_position is None or planned_step.desired_quaternion is None:
        raise RuntimeError("Sequence rollout step is missing absolute pose targets.")
    desired_pose = PoseSequence(
        position=torch.from_numpy(np.asarray(planned_step.desired_position, dtype=np.float32)),
        quaternion=torch.from_numpy(np.asarray(planned_step.desired_quaternion, dtype=np.float32)),
        gripper=(
            None
            if planned_step.desired_gripper is None
            else torch.from_numpy(np.asarray(planned_step.desired_gripper, dtype=np.float32))
        ),
    )
    return compute_osc_pose_action(
        current_pose=libero_rollout.pose_from_libero_observation(current_obs),
        desired_pose=desired_pose,
        control_config=control_config,
        gripper_representation=gripper_representation,
    ).astype(np.float32)


def _merge_future_step_actions(
    existing: dict[int, PlannedControlStep],
    incoming: list[PlannedControlStep],
    *,
    next_action_to_execute: int,
) -> dict[int, PlannedControlStep]:
    merged = {
        int(action_index): plan
        for action_index, plan in existing.items()
        if int(action_index) >= int(next_action_to_execute)
    }
    for plan in incoming:
        if int(plan.absolute_action_index) < int(next_action_to_execute):
            continue
        merged[int(plan.absolute_action_index)] = plan
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def _future_buffer_depth_actions(
    plan_by_action: dict[int, PlannedControlStep],
    *,
    next_action_to_execute: int,
) -> int:
    if not plan_by_action:
        return 0
    return max(0, max(int(action_id) for action_id in plan_by_action) - int(next_action_to_execute) + 1)


def _frame_index_to_action_start(frame_index: int, action_per_frame: int) -> int:
    return max(0, int(frame_index) - 1) * int(action_per_frame)


def _debug_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _debug_array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    flat = array.reshape(-1)
    numeric = flat.astype(np.float64, copy=False) if flat.size else flat
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "sha256": _debug_sha256_bytes(array.tobytes()),
        "preview": flat[:12].tolist(),
        "mean": None if flat.size == 0 else float(numeric.mean()),
        "std": None if flat.size == 0 else float(numeric.std()),
    }


def _debug_tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach().contiguous().cpu()
    byte_tensor = tensor.view(torch.uint8)
    flat = tensor.reshape(-1)
    numeric = flat.to(dtype=torch.float32) if flat.numel() else flat
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(value.device),
        "sha256": _debug_sha256_bytes(byte_tensor.numpy().tobytes()),
        "preview": flat[:12].to(dtype=torch.float32).tolist(),
        "mean": None if flat.numel() == 0 else float(numeric.mean().item()),
        "std": None if flat.numel() == 0 else float(numeric.std(unbiased=False).item()),
    }


def _debug_rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": _debug_tensor_summary(torch.get_rng_state()),
        "torch_cuda": (
            [_debug_tensor_summary(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _debug_raw_action_grid(
    *,
    chunk,
    frame_chunk_size: int,
    action_per_frame: int,
) -> dict[str, Any] | None:
    if chunk.raw_chunk_action_pred is None:
        return None
    raw_actions = rearrange(
        chunk.raw_chunk_action_pred[0].detach().to(dtype=torch.float32).cpu(),
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    generation_frame_start = int(chunk.debug.get("generation_frame_start", 0))
    executable: list[list[float]] = []
    for frame_offset in range(raw_actions.shape[0]):
        if generation_frame_start + frame_offset < 1:
            continue
        for action_offset in range(raw_actions.shape[1]):
            executable.append([float(value) for value in raw_actions[frame_offset, action_offset].tolist()])
    return {
        "generation_frame_start": int(generation_frame_start),
        "all_gripper_by_frame": [
            [float(raw_actions[frame_offset, action_offset, 6].item()) for action_offset in range(raw_actions.shape[1])]
            for frame_offset in range(raw_actions.shape[0])
        ],
        "first_executable_actions": executable[:16],
    }


def _build_exact_startup_debug_report(
    *,
    first_obs: dict[str, np.ndarray],
    initial_inputs: dict[str, torch.Tensor | None],
    session,
    first_chunk,
    config,
    prompt: str,
    seed: int,
    runtime_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    rng_before_startup_infer: dict[str, Any],
    rng_after_startup_infer: dict[str, Any],
    exact_startup_bootstrap_padding: bool = False,
    startup_warmup_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda_device_name = None
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        device_index = torch.cuda.current_device() if runtime_device.index is None else int(runtime_device.index)
        cuda_device_name = torch.cuda.get_device_name(device_index)
    return {
        "schema_version": 1,
        "purpose": "startup_first_chunk_cross_gpu_debug",
        "prompt": str(prompt),
        "seed": int(seed),
        "torch_version": str(torch.__version__),
        "cuda_device_name": cuda_device_name,
        "runtime_device": str(runtime_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
        "runtime_mode": str(config.policy_variant.runtime_mode),
        "video_num_inference_steps": int(config.inference.video_num_inference_steps),
        "action_num_inference_steps": int(config.inference.action_num_inference_steps),
        "guidance_scale": float(config.inference.guidance_scale),
        "action_guidance_scale": float(config.inference.action_guidance_scale),
        "exact_startup_bootstrap_padding": bool(exact_startup_bootstrap_padding),
        "startup_warmup_debug": None if startup_warmup_debug is None else dict(startup_warmup_debug),
        "first_obs": {key: _debug_array_summary(value) for key, value in sorted(first_obs.items())},
        "initial_inputs": {
            "video_latents": _debug_tensor_summary(initial_inputs.get("video_latents")),
            "text_context": _debug_tensor_summary(initial_inputs.get("text_context")),
            "negative_text_context": _debug_tensor_summary(initial_inputs.get("negative_text_context")),
        },
        "session_text_context": _debug_tensor_summary(getattr(session, "text_context", None)),
        "session_negative_text_context": _debug_tensor_summary(getattr(session, "negative_text_context", None)),
        "rng_before_startup_infer": rng_before_startup_infer,
        "rng_after_startup_infer": rng_after_startup_infer,
        "first_chunk": {
            "debug": dict(first_chunk.debug),
            "chunk_action_pred": _debug_tensor_summary(first_chunk.chunk_action_pred),
            "raw_chunk_action_pred": _debug_tensor_summary(first_chunk.raw_chunk_action_pred),
            "predicted_latents": _debug_tensor_summary(first_chunk.predicted_latents),
            "raw_action_grid": _debug_raw_action_grid(
                chunk=first_chunk,
                frame_chunk_size=int(config.inference.frame_chunk_size),
                action_per_frame=int(config.policy_variant.action_per_frame),
            ),
        },
    }


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
