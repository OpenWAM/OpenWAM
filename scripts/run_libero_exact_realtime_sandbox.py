from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_libero_exact_visualization as exact_viz  # noqa: E402

from open_wam.integrations.realtime_control import (  # noqa: E402
    PlannedFrameAction,
    build_live_rollout_summary,
    make_planned_frame_actions,
    merge_future_frame_actions,
)
from open_wam.models.policy_variants import PolicyInferState, RolloutCursor  # noqa: E402
from open_wam.pipelines import build_exact_runtime_runner_from_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    load_experiment_config,
    resolve_transformer_dir_override,
    seed_everywhere,
    validate_positive_step_override,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact LIBERO method-1 policy in a fixed-rate live-control sandbox."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
    )
    parser.add_argument("--transformer-dir", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--target-action-hz", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
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
        choices=("history_only", "async_buffer", "async_mix"),
        default="async_buffer",
    )
    parser.add_argument(
        "--startup-open-loop-chunks",
        type=int,
        default=0,
        help=(
            "Precompute this many model open-loop chunks before the live clock starts. "
            "This avoids fallback without future observations, but it is not observation-conditioned replanning."
        ),
    )
    parser.add_argument("--deadline-miss-policy", type=str, choices=("hold_last", "zero"), default="hold_last")
    parser.add_argument("--deadline-tolerance-ms", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_exact_realtime")
    parser.add_argument("--suffix", type=str, default="sandbox")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.target_action_hz <= 0:
        raise ValueError("--target-action-hz must be positive.")
    if args.startup_open_loop_chunks < 0:
        raise ValueError("--startup-open-loop-chunks must be non-negative.")

    seed_everywhere(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    if args.transformer_dir is not None:
        object.__setattr__(
            config.backbone,
            "transformer_subdir",
            str(resolve_transformer_dir_override(args.transformer_dir)),
        )
    object.__setattr__(config.backbone, "reference_assets_device_policy", args.reference_assets_device_policy)

    runner = build_exact_runtime_runner_from_config(config)
    runtime_device = exact_viz._resolve_device(args.runtime_device)
    frontend_device = exact_viz._resolve_device(args.frontend_device, fallback=runtime_device)
    runner.pipeline.visual_tower.ensure_runtime_backbone_device(
        action_dim=config.action_decoder.action_dim,
        device=runtime_device,
    )
    _apply_inference_overrides(
        runner,
        video_num_inference_steps=args.video_num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
        guidance_scale=args.guidance_scale,
        action_guidance_scale=args.action_guidance_scale,
    )
    component_report = exact_viz._build_open_wam_component_report(
        config,
        runner,
        runtime_device=runtime_device,
        frontend_device=frontend_device,
        decode_device=frontend_device,
    )

    task_spec, prompt = exact_viz._resolve_task_spec(args.benchmark, args.task_id)
    init_states = exact_viz.load_libero_task_init_states(task_spec)
    env = exact_viz._construct_single_env(task_spec)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    action_per_frame = int(config.policy_variant.action_per_frame)
    action_dim = int(config.data.action_schema.action_dim)
    action_period_s = 1.0 / float(args.target_action_hz)
    deadline_tolerance_s = float(args.deadline_tolerance_ms) / 1000.0

    try:
        first_obs = exact_viz._init_single_env(env, init_states[args.episode_idx % len(init_states)])
        with torch.inference_mode():
            startup_prepare_t0 = time.perf_counter()
            initial_inputs = exact_viz._prepare_exact_runtime_inputs(
                runner,
                views=exact_viz._obs_list_to_views([first_obs], config=config, device=frontend_device),
                task_text=(prompt,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            _synchronize_devices(frontend_device, runtime_device)
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
            _synchronize_devices(runtime_device)
            startup_infer_s = time.perf_counter() - startup_infer_t0

        current_chunk_session = first_chunk.session
        plan_by_frame: dict[int, PlannedFrameAction] = {}
        next_frame_to_execute = 1
        initial_plans = _chunk_to_planned_frames(
            first_chunk=first_chunk,
            frame_chunk_size=int(config.inference.frame_chunk_size),
            action_per_frame=action_per_frame,
            source="startup_plan",
            ready_monotonic_s=time.perf_counter(),
        )
        plan_by_frame = merge_future_frame_actions(
            plan_by_frame,
            initial_plans,
            next_frame_to_execute=next_frame_to_execute,
        )

        action_records: list[dict[str, Any]] = []
        action_video_records: list[dict[str, Any]] = []
        replan_records: list[dict[str, Any]] = []
        extension_records: list[dict[str, Any]] = []
        done = False
        current_obs = first_obs
        last_action = np.zeros((action_dim,), dtype=np.float32)
        next_action_index = 0
        skipped_replan_submissions = 0
        pending_history: list[dict[str, Any]] = []
        frame_chunk_size = int(config.inference.frame_chunk_size)
        buffer_tail_session = _session_for_next_chunk(
            first_chunk.session,
            next_frame_start=int(first_chunk.debug.get("generation_frame_start", 0)) + frame_chunk_size,
            frame_chunk_size=frame_chunk_size,
        )
        startup_open_loop_s = 0.0
        if args.startup_open_loop_chunks > 0:
            startup_open_loop_t0 = time.perf_counter()
            for _ in range(int(args.startup_open_loop_chunks)):
                if buffer_tail_session is None:
                    break
                extension_result = _run_extension_job(
                    runner=runner,
                    session=buffer_tail_session,
                    config=config,
                    job_seed=_job_seed_for_session(args.seed, buffer_tail_session),
                )
                extension_records.append(extension_result["trace"])
                buffer_tail_session = extension_result["buffer_tail_session"]
                plan_by_frame = merge_future_frame_actions(
                    plan_by_frame,
                    extension_result["planned_frames"],
                    next_frame_to_execute=next_frame_to_execute,
                )
            startup_open_loop_s = time.perf_counter() - startup_open_loop_t0
            startup_infer_s += startup_open_loop_s
        live_start_monotonic = time.perf_counter()
        last_action_end_monotonic = live_start_monotonic

        with ThreadPoolExecutor(max_workers=1) as executor:
            replan_future: Future[dict[str, Any]] | None = None
            while next_frame_to_execute <= int(args.max_frames) and not done:
                if replan_future is not None and replan_future.done():
                    replan_result = replan_future.result()
                    if replan_result["job_kind"] == "history_replan":
                        replan_records.append(replan_result["trace"])
                        current_chunk_session = replan_result["session"]
                        buffer_tail_session = replan_result["buffer_tail_session"]
                        submitted_through_frame = int(replan_result["submitted_through_frame"])
                        pending_history = [
                            record
                            for record in pending_history
                            if int(record["absolute_frame_index"]) > submitted_through_frame
                        ]
                    else:
                        extension_records.append(replan_result["trace"])
                        buffer_tail_session = replan_result["buffer_tail_session"]
                    plan_by_frame = merge_future_frame_actions(
                        plan_by_frame,
                        replan_result["planned_frames"],
                        next_frame_to_execute=next_frame_to_execute,
                    )
                    replan_future = None

                plan_by_frame = merge_future_frame_actions(
                    plan_by_frame,
                    (),
                    next_frame_to_execute=next_frame_to_execute,
                )
                planned_frame = plan_by_frame.pop(next_frame_to_execute, None)
                if planned_frame is None:
                    frame_actions = _build_fallback_frame_actions(
                        action_dim=action_dim,
                        action_per_frame=action_per_frame,
                        policy=args.deadline_miss_policy,
                        last_action=last_action,
                    )
                    frame_source = f"fallback_{args.deadline_miss_policy}"
                    generation_frame_start = None
                    planner_step_index = None
                    ready_monotonic_s = None
                else:
                    frame_actions = np.asarray(planned_frame.raw_actions, dtype=np.float32)
                    frame_source = str(planned_frame.source)
                    generation_frame_start = int(planned_frame.generation_frame_start)
                    planner_step_index = planned_frame.planner_step_index
                    ready_monotonic_s = planned_frame.ready_monotonic_s

                for action_offset in range(action_per_frame):
                    scheduled_monotonic = live_start_monotonic + next_action_index * action_period_s
                    now = time.perf_counter()
                    if now < scheduled_monotonic:
                        time.sleep(scheduled_monotonic - now)
                    actual_start_monotonic = time.perf_counter()
                    lateness_s = max(0.0, actual_start_monotonic - scheduled_monotonic)
                    action = frame_actions[action_offset].astype(np.float32, copy=False)
                    obs, _, done, _ = env.step(action)
                    action_end_monotonic = time.perf_counter()
                    env_step_s = action_end_monotonic - actual_start_monotonic
                    last_action_end_monotonic = action_end_monotonic
                    extracted_obs = exact_viz._extract_obs(obs)
                    last_action = np.array(action, copy=True)
                    current_obs = extracted_obs
                    action_record = {
                        "action_index": int(next_action_index),
                        "absolute_frame_index": int(next_frame_to_execute),
                        "action_offset": int(action_offset),
                        "source": frame_source,
                        "scheduled_start_s": float(scheduled_monotonic - live_start_monotonic),
                        "actual_start_s": float(actual_start_monotonic - live_start_monotonic),
                        "lateness_s": float(lateness_s),
                        "env_step_s": float(env_step_s),
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
                            "obs": {
                                key: np.array(value, copy=True)
                                for key, value in extracted_obs.items()
                            },
                        }
                    )
                    next_action_index += 1
                    if done:
                        break

                if done:
                    break

                pending_history.append(
                    {
                        "absolute_frame_index": int(next_frame_to_execute),
                        "obs": {
                            key: np.array(value, copy=True)
                            for key, value in current_obs.items()
                        },
                        "raw_actions": np.array(frame_actions, copy=True),
                    }
                )

                if replan_future is not None and replan_future.done():
                    replan_result = replan_future.result()
                    if replan_result["job_kind"] == "history_replan":
                        replan_records.append(replan_result["trace"])
                        current_chunk_session = replan_result["session"]
                        buffer_tail_session = replan_result["buffer_tail_session"]
                        submitted_through_frame = int(replan_result["submitted_through_frame"])
                        pending_history = [
                            record
                            for record in pending_history
                            if int(record["absolute_frame_index"]) > submitted_through_frame
                        ]
                    else:
                        extension_records.append(replan_result["trace"])
                        buffer_tail_session = replan_result["buffer_tail_session"]
                    plan_by_frame = merge_future_frame_actions(
                        plan_by_frame,
                        replan_result["planned_frames"],
                        next_frame_to_execute=next_frame_to_execute + 1,
                    )
                    replan_future = None

                future_buffer_depth = _future_buffer_depth(
                    plan_by_frame,
                    next_frame_to_execute=next_frame_to_execute + 1,
                )
                if replan_future is None:
                    replan_future = _maybe_submit_planner_job(
                        executor=executor,
                        planner_mode=args.planner_mode,
                        pending_history=pending_history,
                        future_buffer_depth=future_buffer_depth,
                        runner=runner,
                        current_chunk_session=current_chunk_session,
                        prompt=prompt,
                        config=config,
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        buffer_tail_session=buffer_tail_session,
                        seed_base=args.seed,
                    )
                elif replan_future is not None:
                    skipped_replan_submissions += 1
                next_frame_to_execute += 1

            if replan_future is not None and replan_future.done():
                replan_result = replan_future.result()
                if replan_result["job_kind"] == "history_replan":
                    replan_records.append(replan_result["trace"])
                else:
                    extension_records.append(replan_result["trace"])

        live_wall_time_s = last_action_end_monotonic - live_start_monotonic if next_action_index > 0 else 0.0
        summary = build_live_rollout_summary(
            action_records=action_records,
            replan_records=replan_records,
            target_action_hz=args.target_action_hz,
            live_wall_time_s=live_wall_time_s,
            startup_prepare_s=startup_prepare_s,
            startup_infer_s=startup_infer_s,
            deadline_tolerance_s=deadline_tolerance_s,
        )
        summary.update(
            {
                "benchmark": args.benchmark,
                "task_id": int(args.task_id),
                "prompt": prompt,
                "episode_idx": int(args.episode_idx),
                "success": bool(done),
                "max_frames": int(args.max_frames),
                "executed_actions": int(next_action_index),
                "executed_frames": int(len({record["absolute_frame_index"] for record in action_records})),
                "env_timestep": int(env.env.timestep),
                "seed": int(args.seed),
                "runtime_device": str(runtime_device),
                "frontend_device": str(frontend_device),
                "transformer_dir": str(config.backbone.transformer_subdir),
                "reference_assets_device_policy": str(config.backbone.reference_assets_device_policy),
                "video_num_inference_steps": int(runner.policy_variant.inference_config.video_num_inference_steps),
                "action_num_inference_steps": int(runner.policy_variant.inference_config.action_num_inference_steps),
                "guidance_scale": float(runner.policy_variant.inference_config.guidance_scale),
                "action_guidance_scale": float(runner.policy_variant.inference_config.action_guidance_scale),
                "planner_mode": args.planner_mode,
                "deadline_miss_policy": args.deadline_miss_policy,
                "startup_open_loop_chunks": int(args.startup_open_loop_chunks),
                "startup_open_loop_s": float(startup_open_loop_s),
                "skipped_replan_submissions": int(skipped_replan_submissions),
                "history_replan_count": int(len(replan_records)),
                "open_loop_extension_count": int(len(extension_records)),
            }
        )

        output_stem = _build_output_stem(
            root=Path(args.output_dir),
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            prompt=prompt,
            episode_idx=args.episode_idx,
            suffix=args.suffix,
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        video_frames = _build_realtime_video_frames(
            action_video_records=action_video_records,
            target_action_hz=args.target_action_hz,
            action_per_frame=action_per_frame,
        )
        video_path = output_stem.with_suffix(".mp4")
        imageio.mimsave(
            video_path,
            video_frames,
            fps=float(args.video_fps or args.target_action_hz),
        )
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
        _write_jsonl(action_trace_path, action_records)
        _write_jsonl(replan_trace_path, replan_records)
        _write_jsonl(extension_trace_path, extension_records)
        load_report_path.write_text(json.dumps(component_report, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _apply_inference_overrides(
    runner,
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
        object.__setattr__(
            runner.policy_variant.inference_config,
            "video_num_inference_steps",
            video_steps,
        )
    if action_steps is not None:
        object.__setattr__(
            runner.policy_variant.inference_config,
            "action_num_inference_steps",
            action_steps,
        )
    if guidance_scale is not None:
        object.__setattr__(runner.policy_variant.inference_config, "guidance_scale", float(guidance_scale))
    if action_guidance_scale is not None:
        object.__setattr__(
            runner.policy_variant.inference_config,
            "action_guidance_scale",
            float(action_guidance_scale),
        )

def _chunk_to_planned_frames(
    *,
    first_chunk,
    frame_chunk_size: int,
    action_per_frame: int,
    source: str,
    ready_monotonic_s: float,
) -> list[PlannedFrameAction]:
    if first_chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        first_chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    return make_planned_frame_actions(
        raw_actions.detach().to(dtype=torch.float32).cpu().numpy(),
        generation_frame_start=int(first_chunk.debug.get("generation_frame_start", 0)),
        source=source,
        planner_step_index=int(first_chunk.session.policy_state.step_index),
        ready_monotonic_s=ready_monotonic_s,
    )


def _future_buffer_depth(
    plan_by_frame: dict[int, PlannedFrameAction],
    *,
    next_frame_to_execute: int,
) -> int:
    if not plan_by_frame:
        return 0
    return max(0, max(int(frame_id) for frame_id in plan_by_frame) - int(next_frame_to_execute) + 1)


def _maybe_submit_planner_job(
    *,
    executor: ThreadPoolExecutor,
    planner_mode: str,
    pending_history: list[dict[str, Any]],
    future_buffer_depth: int,
    runner,
    current_chunk_session,
    prompt: str,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    buffer_tail_session,
    seed_base: int | None = None,
) -> Future[dict[str, Any]] | None:
    history_payload = [
        {
            "absolute_frame_index": int(record["absolute_frame_index"]),
            "obs": {
                key: np.array(value, copy=True)
                for key, value in record["obs"].items()
            },
            "raw_actions": np.array(record["raw_actions"], copy=True),
        }
        for record in pending_history
    ]
    if planner_mode == "history_only":
        if not history_payload:
            return None
        return executor.submit(
            _run_replan_job,
            runner=runner,
            session=current_chunk_session,
            prompt=prompt,
            history_records=history_payload,
            config=config,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            job_seed=_job_seed_for_session(seed_base, current_chunk_session),
        )
    if planner_mode == "async_buffer":
        if buffer_tail_session is not None and future_buffer_depth <= 3:
            return executor.submit(
                _run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                job_seed=_job_seed_for_session(seed_base, buffer_tail_session),
            )
        if history_payload:
            return executor.submit(
                _run_replan_job,
                runner=runner,
                session=current_chunk_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=_job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 6:
            return executor.submit(
                _run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                job_seed=_job_seed_for_session(seed_base, buffer_tail_session),
            )
        return None
    if planner_mode == "async_mix":
        if history_payload and len(history_payload) >= 2 and future_buffer_depth >= 2:
            return executor.submit(
                _run_replan_job,
                runner=runner,
                session=current_chunk_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=_job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 3:
            return executor.submit(
                _run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                job_seed=_job_seed_for_session(seed_base, buffer_tail_session),
            )
        if history_payload:
            return executor.submit(
                _run_replan_job,
                runner=runner,
                session=current_chunk_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=_job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 6:
            return executor.submit(
                _run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                job_seed=_job_seed_for_session(seed_base, buffer_tail_session),
            )
        return None
    raise ValueError(f"Unsupported planner_mode={planner_mode!r}.")


def _job_seed_for_session(seed_base: int | None, session) -> int | None:
    if seed_base is None:
        return None
    return int(seed_base) + int(session.policy_state.step_index)


def _session_for_next_chunk(
    session,
    *,
    next_frame_start: int,
    frame_chunk_size: int,
):
    next_cache = dict(session.policy_state.cache)
    next_cache["frame_start"] = int(next_frame_start)
    next_state = PolicyInferState(
        step_index=int(session.policy_state.step_index),
        cursor=RolloutCursor(
            current_start_frame=int(next_frame_start),
            block_index=int(session.policy_state.cursor.block_index + 1),
            chunk_size=int(frame_chunk_size),
        ),
        cache=next_cache,
        decoder_state=session.policy_state.decoder_state,
    )
    return type(session)(
        policy_state=next_state,
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )


def _run_replan_job(
    *,
    runner,
    session,
    prompt: str,
    history_records: list[dict[str, Any]],
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    job_seed: int | None = None,
) -> dict[str, Any]:
    if not history_records:
        raise ValueError("Realtime replan requires at least one observed history frame.")
    if job_seed is not None:
        seed_everywhere(int(job_seed))
    observed_frame_index = int(history_records[-1]["absolute_frame_index"])
    history_views = [
        {
            key: np.array(value, copy=True)
            for key, value in record["obs"].items()
        }
        for record in history_records
    ]
    action_history = np.concatenate(
        [np.asarray(record["raw_actions"], dtype=np.float32) for record in history_records],
        axis=0,
    )
    with torch.inference_mode():
        prepare_t0 = time.perf_counter()
        prepared = _prepare_history_runtime_inputs(
            runner,
            history_views=history_views,
            task_text=(prompt,),
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
            config=config,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
        )
        _synchronize_devices(frontend_device, runtime_device)
        prepare_s = time.perf_counter() - prepare_t0

        warmup_t0 = time.perf_counter()
        warmup = runner.warmup_cache(
            session=session,
            video_latents=prepared["video_latents"],
            text_context=prepared["text_context"],
            negative_text_context=prepared["negative_text_context"],
            action_history=torch.as_tensor(action_history, device=runtime_device, dtype=torch.float32).unsqueeze(0),
            action_space="raw",
        )
        _synchronize_devices(runtime_device)
        warmup_s = time.perf_counter() - warmup_t0

        infer_t0 = time.perf_counter()
        chunk = runner.infer_chunk(session=warmup.session)
        _synchronize_devices(runtime_device)
        infer_s = time.perf_counter() - infer_t0

    ready_monotonic_s = time.perf_counter()
    planned_frames = _chunk_to_planned_frames(
        first_chunk=chunk,
        frame_chunk_size=int(config.inference.frame_chunk_size),
        action_per_frame=int(config.policy_variant.action_per_frame),
        source="history_replan",
        ready_monotonic_s=ready_monotonic_s,
    )
    generation_frame_start = int(chunk.debug.get("generation_frame_start", observed_frame_index))
    return {
        "job_kind": "history_replan",
        "session": chunk.session,
        "buffer_tail_session": _session_for_next_chunk(
            chunk.session,
            next_frame_start=generation_frame_start + int(config.inference.frame_chunk_size),
            frame_chunk_size=int(config.inference.frame_chunk_size),
        ),
        "planned_frames": planned_frames,
        "trace": {
            "job_kind": "history_replan",
            "observed_frame_index": int(observed_frame_index),
            "history_frame_count": int(len(history_records)),
            "session_frame_start_before": int(session.policy_state.cache.get("frame_start", -1)),
            "session_step_before": int(session.policy_state.step_index),
            "session_step_after": int(chunk.session.policy_state.step_index),
            "generation_frame_start": generation_frame_start,
            "planned_frame_ids": [int(plan.absolute_frame_index) for plan in planned_frames],
            "prepare_s": float(prepare_s),
            "warmup_s": float(warmup_s),
            "infer_s": float(infer_s),
            "total_latency_s": float(prepare_s + warmup_s + infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
        },
        "submitted_through_frame": observed_frame_index,
    }


def _run_extension_job(
    *,
    runner,
    session,
    config,
    job_seed: int | None = None,
) -> dict[str, Any]:
    if job_seed is not None:
        seed_everywhere(int(job_seed))
    with torch.inference_mode():
        infer_t0 = time.perf_counter()
        chunk = runner.infer_chunk(session=session, advance_frame_start=True)
        _synchronize_devices(chunk.chunk_action_pred.device)
        infer_s = time.perf_counter() - infer_t0
    ready_monotonic_s = time.perf_counter()
    generation_frame_start = int(chunk.debug.get("generation_frame_start", session.policy_state.cache.get("frame_start", 0)))
    planned_frames = _chunk_to_planned_frames(
        first_chunk=chunk,
        frame_chunk_size=int(config.inference.frame_chunk_size),
        action_per_frame=int(config.policy_variant.action_per_frame),
        source="open_loop_extension",
        ready_monotonic_s=ready_monotonic_s,
    )
    return {
        "job_kind": "open_loop_extension",
        "planned_frames": planned_frames,
        "buffer_tail_session": chunk.session,
        "trace": {
            "job_kind": "open_loop_extension",
            "generation_frame_start": generation_frame_start,
            "planned_frame_ids": [int(plan.absolute_frame_index) for plan in planned_frames],
            "prepare_s": 0.0,
            "warmup_s": 0.0,
            "infer_s": float(infer_s),
            "total_latency_s": float(infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
        },
        "submitted_through_frame": None,
    }


def _prepare_history_runtime_inputs(
    runner,
    *,
    history_views: list[dict[str, np.ndarray]],
    task_text: tuple[str | None, ...] | None,
    text_context: torch.Tensor | None,
    negative_text_context: torch.Tensor | None,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
) -> dict[str, torch.Tensor | None]:
    if not history_views:
        raise ValueError("Expected at least one observed frame for history preparation.")
    if len(history_views) == 1:
        return exact_viz._prepare_exact_runtime_inputs(
            runner,
            views=exact_viz._obs_list_to_views(history_views, config=config, device=frontend_device),
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            preserve_stream_cache=False,
        )

    latent_chunks: list[torch.Tensor] = []
    resolved_text_context = text_context
    resolved_negative_text_context = negative_text_context
    for frame_obs in history_views:
        prepared = exact_viz._prepare_exact_runtime_inputs(
            runner,
            views=exact_viz._obs_list_to_views([frame_obs], config=config, device=frontend_device),
            task_text=task_text,
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            preserve_stream_cache=False,
        )
        latent_chunks.append(prepared["video_latents"])
        resolved_text_context = prepared["text_context"]
        resolved_negative_text_context = prepared["negative_text_context"]
    return {
        "video_latents": torch.cat(latent_chunks, dim=2),
        "text_context": resolved_text_context,
        "negative_text_context": resolved_negative_text_context,
    }


def _synchronize_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        if device.type != "cuda":
            continue
        key = (device.type, device.index)
        if key in seen:
            continue
        torch.cuda.synchronize(device)
        seen.add(key)


def _build_fallback_frame_actions(
    *,
    action_dim: int,
    action_per_frame: int,
    policy: str,
    last_action: np.ndarray,
) -> np.ndarray:
    if policy == "zero":
        return np.zeros((action_per_frame, action_dim), dtype=np.float32)
    repeated = np.repeat(np.asarray(last_action, dtype=np.float32)[None, :], action_per_frame, axis=0)
    if repeated.shape != (action_per_frame, action_dim):
        raise ValueError(
            "Fallback last-action shape mismatch, "
            f"expected {(action_per_frame, action_dim)}, got {tuple(repeated.shape)}."
        )
    return repeated


def _build_output_stem(
    *,
    root: Path,
    benchmark_name: str,
    task_id: int,
    prompt: str,
    episode_idx: int,
    suffix: str,
) -> Path:
    safe_prompt = _safe_path_token(prompt)
    safe_suffix = _safe_path_token(suffix)
    return root / benchmark_name / f"{task_id}_{safe_prompt}" / f"{episode_idx}_{safe_suffix}"


def _safe_path_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "run"


def _build_realtime_video_frames(
    *,
    action_video_records: list[dict[str, Any]],
    target_action_hz: float,
    action_per_frame: int,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for record in action_video_records:
        obs = record["obs"]
        agentview = np.ascontiguousarray(obs[exact_viz.LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[exact_viz.LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        titled = exact_viz._with_title(Image.fromarray(np.ascontiguousarray(row_real)), "Live LIBERO (AgentView / Wrist)")
        info_panel = Image.new("RGB", (titled.width, 108), color=(0, 0, 0))
        draw = ImageDraw.Draw(info_panel)
        header = (
            f"Action {int(record['action_index']) + 1} | "
            f"Frame {int(record['absolute_frame_index'])} [{int(record['action_offset']) + 1}/{action_per_frame}]"
        )
        source = str(record["source"])
        lag_text = (
            "fallback"
            if record["generation_lag_frames"] is None
            else str(int(record["generation_lag_frames"]))
        )
        lines = [
            header,
            (
                f"Source: {source} | Target: {target_action_hz:.1f} Hz | "
                f"Lateness: {1000.0 * float(record['lateness_s']):.1f} ms"
            ),
            (
                f"Env step: {1000.0 * float(record['env_step_s']):.1f} ms | "
                f"Generation lag: {lag_text} frame(s)"
            ),
        ]
        text_color = (255, 255, 255) if source == "policy" else (255, 180, 120)
        for index, line in enumerate(lines):
            draw.text((10, 10 + index * 28), line, fill=text_color)
        full_frame = np.vstack([np.array(titled, copy=True), np.array(info_panel, copy=True)])
        frames.append(np.ascontiguousarray(full_frame))
    return frames


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    main()
