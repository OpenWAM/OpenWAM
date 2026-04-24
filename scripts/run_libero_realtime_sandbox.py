from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_libero_exact_realtime_sandbox as exact_sandbox  # noqa: E402
import run_libero_exact_visualization as exact_viz  # noqa: E402
import run_libero_video_sequence_visualization as video_viz  # noqa: E402

from open_wam.configs import ActionTargetRepresentation, ParallelRuntimeMode  # noqa: E402
from open_wam.integrations import LiberoControlConfig, compute_osc_pose_action  # noqa: E402
from open_wam.integrations.realtime_control import build_live_rollout_summary  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import LingbotExactRunner, VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    load_experiment_config,
    seed_everywhere,
    validate_positive_step_override,
)

VERBOSE = False


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


class _RolloutRunnerLike(Protocol):
    def reset(
        self,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ):
        ...


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one trained LIBERO policy in a fixed-rate realtime sandbox across exact/joint, "
            "method-3, method-4 video-conditioned, and method-5 MoT variants."
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
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--target-action-hz", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=None)
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
    parser.add_argument("--deadline-miss-policy", type=str, choices=("hold_last", "zero"), default="hold_last")
    parser.add_argument("--deadline-tolerance-ms", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_realtime_validation")
    parser.add_argument("--suffix", type=str, default="sandbox")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive.")
    if args.target_action_hz <= 0:
        raise ValueError("--target-action-hz must be positive.")
    if args.sequence_buffer_threshold < 0:
        raise ValueError("--sequence-buffer-threshold must be non-negative.")
    if args.startup_open_loop_chunks < 0:
        raise ValueError("--startup-open-loop-chunks must be non-negative.")

    global VERBOSE
    VERBOSE = bool(args.verbose)

    seed_everywhere(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    object.__setattr__(config.backbone, "reference_assets_device_policy", args.reference_assets_device_policy)
    video_viz._apply_rollout_chunk_steps_override(config, args.rollout_chunk_steps)
    checkpoint_path = _resolve_checkpoint_path_for_config(config=config, checkpoint_arg=args.checkpoint)
    _apply_checkpoint_backbone_override(config, checkpoint_path=checkpoint_path)

    runtime_device = exact_viz._resolve_device(args.runtime_device)
    frontend_device = exact_viz._resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = exact_viz._resolve_device(args.decode_device, fallback=frontend_device)
    runtime_devices = video_viz._resolve_runtime_devices(args.runtime_devices, fallback=runtime_device)
    runtime_prep_device = exact_viz._resolve_device(args.runtime_prep_device, fallback=runtime_device)
    runtime_output_device = exact_viz._resolve_device(args.runtime_output_device, fallback=runtime_device)

    policy_name = str(config.policy_variant.name)
    if _is_exact_parallel_runtime(config):
        summary = _run_exact_like_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            max_actions=args.max_actions,
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
        )
    elif policy_name == "video_sequence_policy":
        summary = _run_sequence_policy_realtime_rollout(
            config=config,
            checkpoint_path=checkpoint_path,
            rollout_label="method3",
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            max_actions=args.max_actions,
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            initial_generation_action_start=args.initial_generation_action_start,
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            initial_generation_action_start=args.initial_generation_action_start,
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            initial_generation_action_start=args.initial_generation_action_start,
        )
    else:
        raise ValueError(
            "The realtime sandbox currently supports exact/joint `parallel_stream`, `video_sequence_policy`, "
            "`post_latent`, `post_decoded`, and `mot`, "
            f"got policy_variant={policy_name!r}."
        )
    print(json.dumps(summary, indent=2))


def _is_exact_parallel_runtime(config) -> bool:
    return str(config.policy_variant.name) == "parallel_stream" and config.policy_variant.runtime_mode in {
        ParallelRuntimeMode.LINGBOT_EXACT,
        ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    }


def _resolve_checkpoint_path_for_config(*, config, checkpoint_arg: str | None) -> Path | None:
    if checkpoint_arg is not None:
        return video_viz._resolve_checkpoint_file(Path(checkpoint_arg))
    transformer_subdir = getattr(config.backbone, "transformer_subdir", None)
    if transformer_subdir is None:
        return None
    try:
        return video_viz._resolve_checkpoint_path_from_args_or_config(
            checkpoint_arg=None,
            transformer_subdir=str(transformer_subdir),
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
    if transformer_dir.is_dir():
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))


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

def _run_exact_like_realtime_rollout(
    *,
    config,
    checkpoint_path: Path | None,
    benchmark: str,
    task_id: int,
    episode_idx: int,
    max_actions: int,
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
) -> dict[str, Any]:
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
    }

    task_spec, prompt = exact_viz._resolve_task_spec(benchmark, task_id)
    init_states = exact_viz.load_libero_task_init_states(task_spec)
    env = exact_viz._construct_single_env(task_spec)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    action_per_frame = int(config.policy_variant.action_per_frame)
    action_dim = int(config.data.action_schema.action_dim)
    action_period_s = 1.0 / float(target_action_hz)
    deadline_tolerance_s = float(deadline_tolerance_ms) / 1000.0
    max_frames = int(math.ceil(max_actions / action_per_frame))

    try:
        first_obs = exact_viz._init_single_env(env, init_states[episode_idx % len(init_states)])
        with torch.inference_mode():
            session = runner.reset(task_text=(prompt,))
            startup_prepare_t0 = time.perf_counter()
            initial_inputs = exact_viz._prepare_exact_runtime_inputs(
                runner,
                views=exact_viz._obs_list_to_views([first_obs], config=config, device=frontend_device),
                task_text=(prompt,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            exact_sandbox._synchronize_devices(frontend_device, runtime_device)
            startup_prepare_s = time.perf_counter() - startup_prepare_t0

            startup_infer_t0 = time.perf_counter()
            first_chunk = runner.infer_chunk(
                session=session,
                video_latents=initial_inputs["video_latents"],
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )
            exact_sandbox._synchronize_devices(runtime_device)
            startup_infer_s = time.perf_counter() - startup_infer_t0

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
                initial_video_latents=initial_inputs["video_latents"],
                initial_obs=first_obs,
                action_per_frame=action_per_frame,
                frame_chunk_size=int(config.inference.frame_chunk_size),
            )
        ]

        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
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
                )
            startup_open_loop_s = time.perf_counter() - startup_open_loop_t0
            startup_infer_s += startup_open_loop_s
        done = False
        current_obs = first_obs
        last_action = np.zeros((action_dim,), dtype=np.float32)
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
            next_frame_to_execute = 1
            while next_frame_to_execute <= max_frames and next_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    (
                        history_base_session,
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
                        replan_future.result(),
                        config=config,
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        history_base_session=history_base_session,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                    )
                    replan_future = None

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
                            replan_future = exact_sandbox._maybe_submit_planner_job(
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
                        if replan_future is None:
                            if pending_history:
                                history_payload = [
                                    exact_sandbox._copy_history_record_for_worker(record)
                                    for record in pending_history
                                ]
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
                            elif buffer_tail_session is not None:
                                result = exact_sandbox._run_extension_job(
                                    runner=runner,
                                    session=buffer_tail_session,
                                    config=config,
                                    job_seed=exact_sandbox._job_seed_for_session(seed, buffer_tail_session),
                                )
                            else:
                                raise RuntimeError(
                                    "Exact/joint wait-for-replan mode has no pending future, history, or buffer "
                                    f"session for required actions {required_action_indices}."
                                )
                        else:
                            result = replan_future.result()
                            replan_future = None
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
                        )
                    wait_for_plan_count += 1
                    wait_for_plan_total_s += wait_for_plan_s

                frame_actions: list[np.ndarray] = []
                frame_source = None
                generation_frame_start = None
                planner_step_index = None
                ready_monotonic_s = None
                for action_offset in range(action_per_frame):
                    if next_action_index + action_offset >= max_actions:
                        break
                    planned_step = plan_by_action.pop(next_action_index + action_offset, None)
                    if planned_step is None:
                        if sequence_empty_plan_policy == "wait_for_replan":
                            raise RuntimeError(
                                "Exact/joint wait-for-replan mode exhausted without action "
                                f"{next_action_index + action_offset}."
                            )
                        if sequence_empty_plan_policy == "fallback":
                            raw_action = exact_sandbox._build_fallback_frame_actions(
                                action_dim=action_dim,
                                action_per_frame=1,
                                policy=deadline_miss_policy,
                                last_action=last_action,
                            )[0]
                            source = f"fallback_{deadline_miss_policy}"
                        else:
                            raise ValueError(f"Unsupported sequence_empty_plan_policy={sequence_empty_plan_policy!r}.")
                    else:
                        if planned_step.raw_action is None:
                            raise RuntimeError("Exact/joint plan did not contain raw LIBERO actions.")
                        raw_action = np.array(planned_step.raw_action, copy=True)
                        source = str(planned_step.source)
                        generation_frame_start = planned_step.generation_frame_start
                        planner_step_index = planned_step.planner_step_index
                        ready_monotonic_s = planned_step.ready_monotonic_s
                    frame_actions.append(raw_action)
                    frame_source = source if frame_source is None else frame_source

                if not frame_actions:
                    break

                schedule_pause_s += wait_for_plan_s
                frame_obs_sequence: list[dict[str, np.ndarray]] = []
                for action_offset, action in enumerate(frame_actions):
                    scheduled_monotonic = live_start_monotonic + next_action_index * action_period_s + schedule_pause_s
                    now = time.perf_counter()
                    if now < scheduled_monotonic:
                        time.sleep(scheduled_monotonic - now)
                    actual_start_monotonic = time.perf_counter()
                    lateness_s = max(0.0, actual_start_monotonic - scheduled_monotonic)
                    obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                    action_end_monotonic = time.perf_counter()
                    env_step_s = action_end_monotonic - actual_start_monotonic
                    last_action_end_monotonic = action_end_monotonic
                    extracted_obs = exact_viz._extract_obs(obs)
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
                        "action_index": int(next_action_index),
                        "absolute_action_index": int(next_action_index),
                        "absolute_frame_index": int(next_frame_to_execute),
                        "action_offset": int(action_offset),
                        "source": str(frame_source),
                        "scheduled_start_s": float(scheduled_monotonic - live_start_monotonic),
                        "actual_start_s": float(actual_start_monotonic - live_start_monotonic),
                        "lateness_s": float(lateness_s),
                        "env_step_s": float(env_step_s),
                        "generation_action_start": generation_action_start,
                        "generation_lag_actions": (
                            None
                            if generation_action_start is None
                            else int(next_action_index - generation_action_start)
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
                    action_video_records.append(
                        {
                            **action_record,
                            "obs": {key: np.array(value, copy=True) for key, value in extracted_obs.items()},
                        }
                    )
                    if VERBOSE and next_action_index % 50 == 0:
                        _print_stage(
                            "exact_like_action_progress",
                            action_index=int(next_action_index),
                            source=frame_source,
                            done=bool(done),
                        )
                    next_action_index += 1
                    if done or next_action_index >= max_actions:
                        break

                if done or next_action_index >= max_actions:
                    break

                pending_history.append(
                    {
                        "absolute_frame_index": int(next_frame_to_execute),
                        "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
                        "obs_sequence": frame_obs_sequence,
                        "raw_actions": np.stack(frame_actions, axis=0).astype(np.float32),
                    }
                )

                if replan_future is not None and replan_future.done():
                    (
                        history_base_session,
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
                        replan_future.result(),
                        config=config,
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        history_base_session=history_base_session,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                    )
                    replan_future = None

                future_buffer_depth_frames = int(
                    math.ceil(_future_buffer_depth_actions(plan_by_action, next_action_to_execute=next_action_index) / action_per_frame)
                )
                should_submit_replan = _should_submit_exact_realtime_planner(
                    future_buffer_depth_frames=future_buffer_depth_frames,
                    sequence_empty_plan_policy=sequence_empty_plan_policy,
                )
                if replan_future is None and should_submit_replan:
                    replan_future = exact_sandbox._maybe_submit_planner_job(
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
                else:
                    skipped_replan_submissions += 1
                next_frame_to_execute += 1

            if replan_future is not None and replan_future.done():
                (
                    history_base_session,
                    current_chunk_session,
                    buffer_tail_session,
                    plan_by_action,
                    pending_history,
                ) = _consume_exact_future_result(
                    replan_future.result(),
                    config=config,
                    plan_by_action=plan_by_action,
                    next_action_to_execute=next_action_index,
                    pending_history=pending_history,
                    history_base_session=history_base_session,
                    current_chunk_session=current_chunk_session,
                    buffer_tail_session=buffer_tail_session,
                    replan_records=replan_records,
                    extension_records=extension_records,
                )

        live_wall_time_s = last_action_end_monotonic - live_start_monotonic if next_action_index > 0 else 0.0
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
        summary.update(
            {
                "benchmark": benchmark,
                "task_id": int(task_id),
                "prompt": prompt,
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "max_actions": int(max_actions),
                "executed_actions": int(next_action_index),
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
                "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
                "startup_open_loop_chunks": int(startup_open_loop_chunks),
                "startup_open_loop_s": float(startup_open_loop_s),
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "wait_for_plan_count": int(wait_for_plan_count),
                "wait_for_plan_total_s": float(wait_for_plan_total_s),
                "schedule_pause_s": float(schedule_pause_s),
                "blocking_replan_count": int(blocking_replan_count),
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": int(len(extension_records)),
                "stale_replan_planned_actions": int(stale_replan_actions),
                "stale_extension_planned_actions": int(stale_extension_actions),
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
):
    if result["job_kind"] == "history_replan":
        replan_records.append(result["trace"])
        current_chunk_session = result["session"]
        history_base_session = exact_sandbox._resolve_next_exact_history_base_session(
            config=config,
            result=result,
            history_base_session=history_base_session,
        )
        buffer_tail_session = result["buffer_tail_session"]
        submitted_through_frame = int(result["submitted_through_frame"])
        pending_history = [
            record for record in pending_history if int(record["absolute_frame_index"]) > submitted_through_frame
        ]
    else:
        extension_records.append(result["trace"])
        buffer_tail_session = result["buffer_tail_session"]
    planned_steps = _planned_frames_to_step_actions(result["planned_frames"])
    future_planned_steps = [
        step
        for step in planned_steps
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]
    result["trace"]["planned_action_indices"] = [
        int(step.absolute_action_index)
        for step in planned_steps
    ]
    result["trace"]["future_planned_actions"] = int(len(future_planned_steps))
    result["trace"]["stale_planned_actions"] = int(len(planned_steps) - len(future_planned_steps))
    plan_by_action = _merge_future_step_actions(
        plan_by_action,
        planned_steps,
        next_action_to_execute=next_action_to_execute,
    )
    return history_base_session, current_chunk_session, buffer_tail_session, plan_by_action, pending_history


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
    future_buffer_depth_frames: int,
    sequence_empty_plan_policy: str,
) -> bool:
    if sequence_empty_plan_policy == "wait_for_replan":
        return int(future_buffer_depth_frames) <= 0
    return True


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
    generation_action_start = _frame_index_to_action_start(generation_frame_start, action_per_frame)
    planned_steps: list[PlannedControlStep] = []
    for frame_offset in range(raw_actions.shape[0]):
        absolute_frame_index = generation_frame_start + frame_offset
        # Exact LIBERO rollout uses frame 0 only as the startup conditioning
        # block. The first executable actions come from absolute frame 1.
        if absolute_frame_index < 1:
            continue
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
) -> dict[str, Any]:
    if chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    conditioning_frame_index = int(chunk.debug.get("generation_frame_start", 0))
    return {
        "absolute_frame_index": int(conditioning_frame_index),
        "obs": {key: np.array(value, copy=True) for key, value in initial_obs.items()},
        "obs_sequence": [],
        "raw_actions": raw_actions[0].detach().to(dtype=torch.float32).cpu().numpy(),
        "video_latents": initial_video_latents.detach(),
        "source": "startup_conditioning_frame",
    }


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
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
    guidance_scale: float | None,
    action_guidance_scale: float | None,
    initial_generation_action_start: int | None,
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
    _print_stage(f"{rollout_label}_build_pipeline_start")
    pipeline = build_variant_pipeline_from_config(config)
    _print_stage(f"{rollout_label}_build_pipeline_done")
    _print_stage(f"{rollout_label}_load_checkpoint_start", checkpoint=str(checkpoint_path))
    video_viz._load_pipeline_checkpoint(pipeline, checkpoint_path)
    _print_stage(f"{rollout_label}_load_checkpoint_done")
    _print_stage(f"{rollout_label}_move_pipeline_start", runtime_device=str(runtime_device))
    pipeline = pipeline.to(runtime_device)
    _print_stage(f"{rollout_label}_move_pipeline_done")
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
        "decoder_runtime": _collect_decoder_runtime_metadata(pipeline, config),
    }

    _print_stage(f"{rollout_label}_resolve_task_start", benchmark=benchmark, task_id=task_id)
    task_spec, prompt = video_viz._resolve_task_spec(benchmark, task_id)
    _print_stage(f"{rollout_label}_resolve_task_done", prompt=prompt)
    init_states = video_viz.load_libero_task_init_states(task_spec)
    _print_stage(f"{rollout_label}_load_init_states_done", num_init_states=len(init_states))
    env = video_viz._construct_single_env(task_spec)
    _print_stage(f"{rollout_label}_construct_env_done", env_created=env is not None)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    action_period_s = 1.0 / float(target_action_hz)
    deadline_tolerance_s = float(deadline_tolerance_ms) / 1000.0
    action_horizon = int(config.data.action_schema.action_horizon)
    raw_window_frames = video_viz._default_raw_window_frames(int(config.data.num_frames))
    control_config = LiberoControlConfig()

    try:
        with torch.inference_mode():
            initial_obs_window = video_viz._init_single_env(
                env,
                init_states[episode_idx % len(init_states)],
                num_frames=raw_window_frames,
            )
            _print_stage(f"{rollout_label}_init_env_done", initial_window=len(initial_obs_window))
            startup_prepare_t0 = time.perf_counter()
            initial_inputs = video_viz._prepare_rollout_inputs(
                pipeline,
                views=video_viz._obs_window_to_rollout_views(initial_obs_window, device=frontend_device),
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
                obs_window=[{key: np.array(value, copy=True) for key, value in obs.items()} for obs in initial_obs_window],
                prompt=prompt,
                task_id=int(task_id),
                episode_idx=int(episode_idx),
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                generation_action_start=video_viz._resolve_initial_generation_action_start(
                    initial_obs_window,
                    initial_generation_action_start=initial_generation_action_start,
                    rollout_starts_at_action_zero=video_viz._uses_zero_based_generation_start(config),
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
        plan_by_action = _merge_future_step_actions({}, startup["planned_steps"], next_action_to_execute=0)
        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
        replan_records: list[dict[str, Any]] = []
        done = False
        current_obs = {key: np.array(value, copy=True) for key, value in initial_obs_window[-1].items()}
        obs_window = [{key: np.array(value, copy=True) for key, value in obs.items()} for obs in initial_obs_window]
        last_action = np.zeros((int(config.data.action_schema.action_dim),), dtype=np.float32)
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
            while next_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
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
                            result = _run_sequence_replan_job(
                                runner=runner,
                                session=session,
                                obs_window=[
                                    {key: np.array(value, copy=True) for key, value in obs.items()}
                                    for obs in obs_window
                                ],
                                prompt=prompt,
                                task_id=int(task_id),
                                episode_idx=int(episode_idx),
                                config=config,
                                frontend_device=frontend_device,
                                runtime_device=runtime_device,
                                generation_action_start=next_generation_action_start,
                                source="blocking_replan",
                            )
                        else:
                            result = replan_future.result()
                            replan_future = None
                        wait_for_plan_s = time.perf_counter() - wait_t0
                        wait_for_plan_count += 1
                        wait_for_plan_total_s += wait_for_plan_s
                        result["trace"]["blocking_wait_action_index"] = int(next_action_index)
                        result["trace"]["blocking_wait_s"] = float(wait_for_plan_s)
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
                scheduled_monotonic = live_start_monotonic + next_action_index * action_period_s + schedule_pause_s
                now = time.perf_counter()
                if now < scheduled_monotonic:
                    time.sleep(scheduled_monotonic - now)
                actual_start_monotonic = time.perf_counter()
                lateness_s = max(0.0, actual_start_monotonic - scheduled_monotonic)
                obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                action_end_monotonic = time.perf_counter()
                env_step_s = action_end_monotonic - actual_start_monotonic
                last_action_end_monotonic = action_end_monotonic
                current_obs = video_viz._extract_obs(obs)
                obs_window.append({key: np.array(value, copy=True) for key, value in current_obs.items()})
                if len(obs_window) > raw_window_frames:
                    obs_window = obs_window[-raw_window_frames:]
                last_action = np.array(action, copy=True)

                action_record = {
                    "action_index": int(next_action_index),
                    "absolute_action_index": int(next_action_index),
                    "absolute_frame_index": int(next_action_index + 1),
                    "action_offset": 0,
                    "source": source,
                    "scheduled_start_s": float(scheduled_monotonic - live_start_monotonic),
                    "actual_start_s": float(actual_start_monotonic - live_start_monotonic),
                    "lateness_s": float(lateness_s),
                    "env_step_s": float(env_step_s),
                    "generation_action_start": generation_action_start,
                    "generation_lag_actions": (
                        None
                        if generation_action_start is None
                        else int(next_action_index - generation_action_start)
                    ),
                    "generation_frame_start": (
                        None
                        if generation_action_start is None
                        else int(generation_action_start + 1)
                    ),
                    "generation_lag_frames": (
                        None
                        if generation_action_start is None
                        else int((next_action_index + 1) - (generation_action_start + 1))
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
                action_video_records.append(
                    {
                        **action_record,
                        "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
                    }
                )
                if VERBOSE and next_action_index % 50 == 0:
                    _print_stage(
                        f"{rollout_label}_action_progress",
                        action_index=int(next_action_index),
                        source=source,
                        done=bool(done),
                    )
                next_action_index += 1
                if done or next_action_index >= max_actions:
                    break

                if replan_future is not None and replan_future.done():
                    result = replan_future.result()
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
                should_submit = False
                if planner_mode == "history_only":
                    should_submit = remaining_buffer == 0
                elif planner_mode in {"async_buffer", "async_mix", "async_history_first"}:
                    should_submit = remaining_buffer <= int(sequence_buffer_threshold)
                else:
                    raise ValueError(f"Unsupported planner_mode={planner_mode!r} for policy_variant={config.policy_variant.name!r}.")
                if replan_future is None and should_submit:
                    obs_snapshot = [
                        {key: np.array(value, copy=True) for key, value in obs.items()}
                        for obs in obs_window
                    ]
                    replan_future = executor.submit(
                        _run_sequence_replan_job,
                        runner=runner,
                        session=session,
                        obs_window=obs_snapshot,
                        prompt=prompt,
                        task_id=int(task_id),
                        episode_idx=int(episode_idx),
                        config=config,
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        generation_action_start=next_generation_action_start,
                        source="history_replan",
                    )
                elif replan_future is not None:
                    skipped_replan_submissions += 1

            if replan_future is not None and replan_future.done():
                result = replan_future.result()
                replan_records.append(result["trace"])
                session = result["session"]
                next_generation_action_start = int(result["next_generation_action_start"])
                plan_by_action = _merge_future_step_actions(
                    plan_by_action,
                    result["planned_steps"],
                    next_action_to_execute=next_action_index,
                )

        live_wall_time_s = last_action_end_monotonic - live_start_monotonic if next_action_index > 0 else 0.0
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
                "executed_actions": int(next_action_index),
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
                "video_num_inference_steps": int(config.inference.video_num_inference_steps),
                "action_num_inference_steps": int(config.inference.action_num_inference_steps),
                "guidance_scale": float(config.inference.guidance_scale),
                "action_guidance_scale": float(config.inference.action_guidance_scale),
                "planner_mode": planner_mode,
                "sequence_buffer_threshold": int(sequence_buffer_threshold),
                "sequence_empty_plan_policy": str(sequence_empty_plan_policy),
                "deadline_miss_policy": deadline_miss_policy,
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "wait_for_plan_count": int(wait_for_plan_count),
                "wait_for_plan_total_s": float(wait_for_plan_total_s),
                "schedule_pause_s": float(schedule_pause_s),
                "blocking_replan_count": int(blocking_replan_count),
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": 0,
                "policy_variant": str(config.policy_variant.name),
                "startup_plan_trace": startup["trace"],
            }
        )
        return _finalize_rollout_outputs(
            summary=summary,
            action_records=action_records,
            action_video_records=action_video_records,
            replan_records=replan_records,
            extension_records=[],
            component_report=load_report,
            output_dir=output_dir,
            benchmark=benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            suffix=suffix,
            video_fps=video_fps or target_action_hz,
            action_per_frame=1,
        )
    finally:
        env.close()


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
) -> dict[str, Any]:
    with torch.inference_mode():
        prepare_t0 = time.perf_counter()
        rollout_inputs = video_viz._prepare_rollout_inputs(
            runner.pipeline,
            views=video_viz._obs_window_to_rollout_views(obs_window, device=frontend_device),
            task_text=(prompt,),
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        exact_sandbox._synchronize_devices(frontend_device, runtime_device)
        prepare_s = time.perf_counter() - prepare_t0

        infer_t0 = time.perf_counter()
        inference_session = _resolve_observation_conditioned_replan_session(
            runner=runner,
            session=session,
            config=config,
        )
        step_output = runner.infer_step(
            session=inference_session,
            context=PolicyInferContext(
                state=video_viz._build_state_inputs_from_obs_window(
                    obs_window,
                    state_horizon=int(config.data.action_schema.state_horizon),
                    state_encoding=str(config.data.action_target.state_encoding),
                ).unsqueeze(0).to(device=runtime_device),
                extra=video_viz._build_sequence_rollout_infer_extra(
                    config=config,
                    prompt=prompt,
                    generation_action_start=int(generation_action_start),
                    runtime_device=runtime_device,
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                ),
            ),
            video_latents=rollout_inputs["video_latents"],
            canonical_video=None,
        )
        exact_sandbox._synchronize_devices(runtime_device)
        infer_s = time.perf_counter() - infer_t0

    policy_aux = step_output.infer_output.policy_output.aux
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
    return {
        "session": step_output.session,
        "planned_steps": planned_steps,
        "next_generation_action_start": int(next_generation_action_start),
        "trace": {
            "job_kind": "history_replan",
            "observed_action_index": int(max(-1, generation_action_start - 1)),
            "history_frame_count": int(len(obs_window)),
            "generation_action_start": int(generation_action_start),
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


def _resolve_observation_conditioned_replan_session(
    *,
    runner: _RolloutRunnerLike,
    session,
    config,
):
    if str(config.policy_variant.name) != "mot":
        return session
    # MoT's video-prefill cache is tied to the current observation window.
    # Rebuild it for every live replan instead of carrying the startup cache
    # across later observation-conditioned replans.
    return runner.reset(
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )


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
    source: str,
    planner_step_index: int | None,
    ready_monotonic_s: float,
    action_target_representation: ActionTargetRepresentation | str,
    rotation_representation: str,
) -> list[PlannedControlStep]:
    representation = ActionTargetRepresentation(action_target_representation)
    planned_steps: list[PlannedControlStep] = []
    if representation == ActionTargetRepresentation.RAW:
        for action_offset in range(action_pred.shape[0]):
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(generation_action_start + action_offset),
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

    desired_pose_targets = video_viz._reconstruct_chunk_pose_targets(
        action_pred,
        reference_obs=reference_obs,
        rotation_representation=rotation_representation,
    )
    for action_offset in range(action_pred.shape[0]):
        desired_gripper = None
        if desired_pose_targets.gripper is not None:
            desired_gripper = desired_pose_targets.gripper[action_offset].detach().to(dtype=torch.float32).cpu().numpy()
        planned_steps.append(
            PlannedControlStep(
                absolute_action_index=int(generation_action_start + action_offset),
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
    desired_pose = video_viz.PoseSequence(
        position=torch.from_numpy(np.asarray(planned_step.desired_position, dtype=np.float32)),
        quaternion=torch.from_numpy(np.asarray(planned_step.desired_quaternion, dtype=np.float32)),
        gripper=(
            None
            if planned_step.desired_gripper is None
            else torch.from_numpy(np.asarray(planned_step.desired_gripper, dtype=np.float32))
        ),
    )
    return compute_osc_pose_action(
        current_pose=video_viz._pose_from_obs_record(current_obs),
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
) -> dict[str, Any]:
    output_stem = exact_sandbox._build_output_stem(
        root=output_dir,
        benchmark_name=benchmark,
        task_id=task_id,
        prompt=prompt,
        episode_idx=episode_idx,
        suffix=suffix,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    video_frames = exact_sandbox._build_realtime_video_frames(
        action_video_records=action_video_records,
        target_action_hz=float(summary["target_action_hz"]),
        action_per_frame=action_per_frame,
    )
    video_path = output_stem.with_suffix(".mp4")
    imageio.mimsave(video_path, video_frames, fps=float(video_fps))
    summary["video_path"] = str(video_path.resolve())

    summary_path = output_stem.with_suffix(".json")
    action_trace_path = output_stem.with_name(f"{output_stem.stem}_actions.jsonl")
    replan_trace_path = output_stem.with_name(f"{output_stem.stem}_replans.jsonl")
    extension_trace_path = output_stem.with_name(f"{output_stem.stem}_extensions.jsonl")
    load_report_path = output_stem.with_name(f"{output_stem.stem}_load_report.json")
    summary["summary_path"] = str(summary_path.resolve())
    summary["action_trace_path"] = str(action_trace_path.resolve())
    summary["replan_trace_path"] = str(replan_trace_path.resolve())
    summary["extension_trace_path"] = str(extension_trace_path.resolve())
    summary["load_report_path"] = str(load_report_path.resolve())
    exact_sandbox._write_jsonl(action_trace_path, action_records)
    exact_sandbox._write_jsonl(replan_trace_path, replan_records)
    exact_sandbox._write_jsonl(extension_trace_path, extension_records)
    load_report_path.write_text(json.dumps(component_report, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _print_stage(name: str, **payload: object) -> None:
    if VERBOSE:
        print(json.dumps({"stage": name, **payload}), flush=True)


if __name__ == "__main__":
    main()
