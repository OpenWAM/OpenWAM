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

from open_wam.configs import ParallelRuntimeMode  # noqa: E402
from open_wam.integrations import LiberoControlConfig, compute_osc_pose_action  # noqa: E402
from open_wam.integrations.realtime_control import build_live_rollout_summary  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import LingbotExactRunner, VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402

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
    parser.add_argument("--video-num-inference-steps", type=int, default=1)
    parser.add_argument("--action-num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--planner-mode",
        type=str,
        choices=("history_only", "async_buffer"),
        default="async_buffer",
    )
    parser.add_argument("--sequence-buffer-threshold", type=int, default=3)
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

    global VERBOSE
    VERBOSE = bool(args.verbose)

    seed_everywhere(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    object.__setattr__(config.backbone, "reference_assets_device_policy", args.reference_assets_device_policy)
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
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
            video_num_inference_steps=args.video_num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
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
    video_num_inference_steps: int,
    action_num_inference_steps: int,
    guidance_scale: float,
    action_guidance_scale: float,
) -> None:
    object.__setattr__(config.inference, "video_num_inference_steps", int(video_num_inference_steps))
    object.__setattr__(config.inference, "action_num_inference_steps", int(action_num_inference_steps))
    object.__setattr__(config.inference, "guidance_scale", float(guidance_scale))
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
    video_num_inference_steps: int,
    action_num_inference_steps: int,
    guidance_scale: float,
    action_guidance_scale: float,
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
        "video_steps": int(video_num_inference_steps),
        "action_steps": int(action_num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "action_guidance_scale": float(action_guidance_scale),
        "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
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

            session = runner.reset(
                task_text=(prompt,),
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )
            startup_infer_t0 = time.perf_counter()
            first_chunk = runner.infer_chunk(
                session=session,
                video_latents=initial_inputs["video_latents"],
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )
            exact_sandbox._synchronize_devices(runtime_device)
            startup_infer_s = time.perf_counter() - startup_infer_t0

        current_chunk_session = first_chunk.session
        buffer_tail_session = exact_sandbox._session_for_next_chunk(
            first_chunk.session,
            next_frame_start=int(first_chunk.debug.get("generation_frame_start", 0)) + int(config.inference.frame_chunk_size),
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

        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
        replan_records: list[dict[str, Any]] = []
        extension_records: list[dict[str, Any]] = []
        pending_history: list[dict[str, Any]] = []
        done = False
        current_obs = first_obs
        last_action = np.zeros((action_dim,), dtype=np.float32)
        next_action_index = 0
        live_start_monotonic = time.perf_counter()
        last_action_end_monotonic = live_start_monotonic
        skipped_replan_submissions = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            replan_future: Future[dict[str, Any]] | None = None
            next_frame_to_execute = 1
            while next_frame_to_execute <= max_frames and next_action_index < max_actions and not done:
                if replan_future is not None and replan_future.done():
                    (
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
                        replan_future.result(),
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                    )
                    replan_future = None

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
                        raw_action = exact_sandbox._build_fallback_frame_actions(
                            action_dim=action_dim,
                            action_per_frame=1,
                            policy=deadline_miss_policy,
                            last_action=last_action,
                        )[0]
                        source = f"fallback_{deadline_miss_policy}"
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

                for action_offset, action in enumerate(frame_actions):
                    scheduled_monotonic = live_start_monotonic + next_action_index * action_period_s
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
                        "plan_ready_delay_s": (
                            None
                            if ready_monotonic_s is None
                            else float(actual_start_monotonic - ready_monotonic_s)
                        ),
                    }
                    action_records.append(action_record)
                    action_video_records.append(
                        {
                            **action_record,
                            "obs": {key: np.array(value, copy=True) for key, value in extracted_obs.items()},
                        }
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
                        "raw_actions": np.stack(frame_actions, axis=0).astype(np.float32),
                    }
                )

                if replan_future is not None and replan_future.done():
                    (
                        current_chunk_session,
                        buffer_tail_session,
                        plan_by_action,
                        pending_history,
                    ) = _consume_exact_future_result(
                        replan_future.result(),
                        plan_by_action=plan_by_action,
                        next_action_to_execute=next_action_index,
                        pending_history=pending_history,
                        current_chunk_session=current_chunk_session,
                        buffer_tail_session=buffer_tail_session,
                        replan_records=replan_records,
                        extension_records=extension_records,
                    )
                    replan_future = None

                future_buffer_depth_frames = int(
                    math.ceil(_future_buffer_depth_actions(plan_by_action, next_action_to_execute=next_action_index) / action_per_frame)
                )
                if replan_future is None:
                    replan_future = exact_sandbox._maybe_submit_planner_job(
                        executor=executor,
                        planner_mode=planner_mode,
                        pending_history=pending_history,
                        future_buffer_depth=future_buffer_depth_frames,
                        runner=runner,
                        current_chunk_session=current_chunk_session,
                        prompt=prompt,
                        config=config,
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        buffer_tail_session=buffer_tail_session,
                    )
                else:
                    skipped_replan_submissions += 1
                next_frame_to_execute += 1

            if replan_future is not None and replan_future.done():
                (
                    current_chunk_session,
                    buffer_tail_session,
                    plan_by_action,
                    pending_history,
                ) = _consume_exact_future_result(
                    replan_future.result(),
                    plan_by_action=plan_by_action,
                    next_action_to_execute=next_action_index,
                    pending_history=pending_history,
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
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": int(len(extension_records)),
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
    plan_by_action: dict[int, PlannedControlStep],
    next_action_to_execute: int,
    pending_history: list[dict[str, Any]],
    current_chunk_session,
    buffer_tail_session,
    replan_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
):
    if result["job_kind"] == "history_replan":
        replan_records.append(result["trace"])
        current_chunk_session = result["session"]
        buffer_tail_session = result["buffer_tail_session"]
        submitted_through_frame = int(result["submitted_through_frame"])
        pending_history = [
            record for record in pending_history if int(record["absolute_frame_index"]) > submitted_through_frame
        ]
    else:
        extension_records.append(result["trace"])
        buffer_tail_session = result["buffer_tail_session"]
    planned_steps = _planned_frames_to_step_actions(result["planned_frames"])
    plan_by_action = _merge_future_step_actions(
        plan_by_action,
        planned_steps,
        next_action_to_execute=next_action_to_execute,
    )
    return current_chunk_session, buffer_tail_session, plan_by_action, pending_history


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
        for action_offset in range(raw_actions.shape[1]):
            absolute_action_index = generation_action_start + frame_offset * action_per_frame + action_offset
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
    video_num_inference_steps: int,
    action_num_inference_steps: int,
    guidance_scale: float,
    action_guidance_scale: float,
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
        "action_horizon": int(config.data.action_schema.action_horizon),
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
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                generation_action_start=0,
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
                if planned_step is None:
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

                scheduled_monotonic = live_start_monotonic + next_action_index * action_period_s
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
                    "plan_ready_delay_s": (
                        None
                        if ready_monotonic_s is None
                        else float(actual_start_monotonic - ready_monotonic_s)
                    ),
                }
                action_records.append(action_record)
                action_video_records.append(
                    {
                        **action_record,
                        "obs": {key: np.array(value, copy=True) for key, value in current_obs.items()},
                    }
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
                elif planner_mode == "async_buffer":
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
                "planner_mode": planner_mode,
                "sequence_buffer_threshold": int(sequence_buffer_threshold),
                "deadline_miss_policy": deadline_miss_policy,
                "skipped_replan_submissions": int(skipped_replan_submissions),
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
        infer_extra = {"task_text": (prompt,)}
        if str(config.policy_variant.name) in {"post_latent", "post_decoded"}:
            infer_extra["video_condition_observed_prefix_anchor"] = "end"
        if str(config.policy_variant.name) == "mot":
            infer_extra["action_device"] = str(runtime_device)
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
                extra=infer_extra,
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
    action_pred = step_output.infer_output.decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu().numpy()
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
            "video_condition_observed_prefix_anchor": video_condition_metadata.get("observed_prefix_anchor"),
            "video_condition_observed_prefix_start_index": video_condition_metadata.get("observed_prefix_start_index"),
            "predicted_video_latents_shape": (
                list(predicted_latents.shape)
                if isinstance(predicted_latents, torch.Tensor)
                else None
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


def _sequence_chunk_to_planned_steps(
    *,
    action_pred: np.ndarray,
    reference_obs: dict[str, np.ndarray],
    generation_action_start: int,
    source: str,
    planner_step_index: int | None,
    ready_monotonic_s: float,
    rotation_representation: str,
) -> list[PlannedControlStep]:
    desired_pose_targets = video_viz._reconstruct_chunk_pose_targets(
        action_pred,
        reference_obs=reference_obs,
        rotation_representation=rotation_representation,
    )
    planned_steps: list[PlannedControlStep] = []
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
