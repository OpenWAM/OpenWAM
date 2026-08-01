"""Realtime LIBERO scheduling and observed-history execution contracts."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any

import numpy as np
import torch
from einops import rearrange

from open_wam.configs import ParallelRuntimeMode
from open_wam.configs.enums import DeadlineMissPolicy
from open_wam.evals import libero_visualization as exact_viz
from open_wam.integrations.realtime_control import (
    PlannedFrameAction,
    make_planned_frame_actions,
)
from open_wam.models.common.rollout_startup import (
    require_strict_startup_generation_frame,
)
from open_wam.models.policy_variants import PolicyInferState, RolloutCursor
from open_wam.utils import validate_positive_step_override


__all__ = [
    "apply_inference_overrides",
    "build_fallback_frame_actions",
    "copy_history_record_for_worker",
    "isolated_torch_rng",
    "job_seed_for_session",
    "maybe_submit_planner_job",
    "resolve_exact_startup_sessions",
    "resolve_next_exact_history_base_session",
    "run_extension_job",
    "run_replan_job",
    "should_submit_planner_job",
    "synchronize_devices",
]


@contextmanager
def isolated_torch_rng(seed: int | None, *devices: torch.device):
    if seed is None:
        yield
        return
    cuda_devices: list[int] = []
    for raw_device in devices:
        device = torch.device(raw_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            continue
        device_index = torch.cuda.current_device() if device.index is None else int(device.index)
        if device_index not in cuda_devices:
            cuda_devices.append(device_index)
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        yield


def apply_inference_overrides(
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
    generation_frame_start: int | None = None,
) -> list[PlannedFrameAction]:
    if first_chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        first_chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    resolved_generation_frame_start = (
        int(first_chunk.debug.get("generation_frame_start", 0))
        if generation_frame_start is None
        else int(generation_frame_start)
    )
    require_strict_startup_generation_frame(resolved_generation_frame_start)
    return make_planned_frame_actions(
        raw_actions.detach().to(dtype=torch.float32).cpu().numpy(),
        generation_frame_start=resolved_generation_frame_start,
        source=source,
        planner_step_index=int(first_chunk.session.policy_state.step_index),
        ready_monotonic_s=ready_monotonic_s,
    )


def should_submit_planner_job(
    *,
    planner_mode: str,
    has_history: bool,
    future_buffer_depth: int,
    has_buffer_tail_session: bool,
) -> bool:
    if planner_mode == "history_only":
        return has_history
    if planner_mode == "async_buffer":
        if has_buffer_tail_session and future_buffer_depth <= 3:
            return True
        if has_history:
            return True
        return has_buffer_tail_session and future_buffer_depth <= 6
    if planner_mode == "async_history_first":
        return has_history or (has_buffer_tail_session and future_buffer_depth <= 6)
    if planner_mode == "async_mix":
        if has_history and future_buffer_depth >= 2:
            return True
        if has_buffer_tail_session and future_buffer_depth <= 3:
            return True
        if has_history:
            return True
        return has_buffer_tail_session and future_buffer_depth <= 6
    raise ValueError(f"Unsupported planner_mode={planner_mode!r}.")


def maybe_submit_planner_job(
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
    seed_base: int | None = None,
) -> Future[dict[str, Any]] | None:
    if not should_submit_planner_job(
        planner_mode=planner_mode,
        has_history=bool(pending_history),
        future_buffer_depth=future_buffer_depth,
        has_buffer_tail_session=buffer_tail_session is not None,
    ):
        return None
    history_payload = [
        copy_history_record_for_worker(record)
        for record in pending_history
    ]
    if planner_mode == "history_only":
        if not history_payload:
            return None
        return executor.submit(
            run_replan_job,
            runner=runner,
            session=history_base_session,
            prompt=prompt,
            history_records=history_payload,
            config=config,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            job_seed=job_seed_for_session(seed_base, current_chunk_session),
        )
    if planner_mode == "async_buffer":
        if buffer_tail_session is not None and future_buffer_depth <= 3:
            return executor.submit(
                run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, buffer_tail_session),
            )
        if history_payload:
            return executor.submit(
                run_replan_job,
                runner=runner,
                session=history_base_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 6:
            return executor.submit(
                run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, buffer_tail_session),
            )
        return None
    if planner_mode == "async_history_first":
        if history_payload:
            return executor.submit(
                run_replan_job,
                runner=runner,
                session=history_base_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 6:
            return executor.submit(
                run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, buffer_tail_session),
            )
        return None
    if planner_mode == "async_mix":
        if history_payload and len(history_payload) >= 2 and future_buffer_depth >= 2:
            return executor.submit(
                run_replan_job,
                runner=runner,
                session=history_base_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 3:
            return executor.submit(
                run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, buffer_tail_session),
            )
        if history_payload:
            return executor.submit(
                run_replan_job,
                runner=runner,
                session=history_base_session,
                prompt=prompt,
                history_records=history_payload,
                config=config,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, current_chunk_session),
            )
        if buffer_tail_session is not None and future_buffer_depth <= 6:
            return executor.submit(
                run_extension_job,
                runner=runner,
                session=buffer_tail_session,
                config=config,
                runtime_device=runtime_device,
                job_seed=job_seed_for_session(seed_base, buffer_tail_session),
            )
        return None
    raise ValueError(f"Unsupported planner_mode={planner_mode!r}.")


def copy_history_record_for_worker(record: dict[str, Any]) -> dict[str, Any]:
    copied = {
        "absolute_frame_index": int(record["absolute_frame_index"]),
        "obs": {
            key: np.array(value, copy=True)
            for key, value in record["obs"].items()
        },
    }
    if "raw_actions" in record:
        copied["raw_actions"] = np.array(record["raw_actions"], copy=True)
    if "raw_actions_valid" in record:
        copied["raw_actions_valid"] = bool(record["raw_actions_valid"])
    if "raw_action_dim" in record:
        copied["raw_action_dim"] = int(record["raw_action_dim"])
    if "obs_sequence" in record:
        copied["obs_sequence"] = [
            {
                key: np.array(value, copy=True)
                for key, value in obs.items()
            }
            for obs in record["obs_sequence"]
        ]
    if isinstance(record.get("video_latents"), torch.Tensor):
        copied["video_latents"] = record["video_latents"].detach().clone()
    if record.get("proprio_state") is not None:
        copied["proprio_state"] = np.array(record["proprio_state"], dtype=np.float32, copy=True)
    return copied


def job_seed_for_session(seed_base: int | None, session) -> int | None:
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


def resolve_exact_startup_sessions(
    *,
    config,
    startup_session,
    first_chunk,
    frame_chunk_size: int,
):
    generation_frame_start = int(first_chunk.debug.get("generation_frame_start", 0))
    require_strict_startup_generation_frame(generation_frame_start)
    current_chunk_session = first_chunk.session
    history_base_session = _resolve_exact_startup_history_base_session(
        config=config,
        startup_session=startup_session,
        current_chunk_session=current_chunk_session,
    )
    buffer_tail_session = _session_for_next_chunk(
        current_chunk_session,
        next_frame_start=generation_frame_start + int(frame_chunk_size),
        frame_chunk_size=int(frame_chunk_size),
    )
    return history_base_session, current_chunk_session, buffer_tail_session


def _resolve_exact_startup_history_base_session(
    *,
    config,
    startup_session,
    current_chunk_session,
):
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    if runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
        return startup_session
    return current_chunk_session


def resolve_next_exact_history_base_session(
    *,
    config,
    result: dict[str, Any],
    history_base_session,
):
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    if runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
        return result.get("warmup_session", history_base_session)
    return result["session"]


def run_replan_job(
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
    observed_frame_index = int(history_records[-1]["absolute_frame_index"])
    history_frame_start = int(history_records[0]["absolute_frame_index"])
    raw_observation_count = _count_history_raw_observations(history_records)
    history_views = [
        {key: np.array(value, copy=True) for key, value in obs.items()}
        for obs in _history_records_to_obs_sequence(history_records)
    ]
    precomputed_video_latents = _history_records_to_precomputed_video_latents(history_records)
    proprio_state = _history_records_to_proprio_state(
        history_records,
        config=config,
        device=runtime_device,
    )
    action_history = _history_records_to_action_history(history_records, config=config)
    with isolated_torch_rng(job_seed, frontend_device, runtime_device), torch.inference_mode():
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
            precomputed_video_latents=precomputed_video_latents,
        )
        synchronize_devices(frontend_device, runtime_device)
        prepare_s = time.perf_counter() - prepare_t0

        warmup_t0 = time.perf_counter()
        warmup = runner.warmup_cache(
            session=session,
            video_latents=prepared["video_latents"],
            text_context=prepared["text_context"],
            negative_text_context=prepared["negative_text_context"],
            action_history=torch.as_tensor(action_history, device=runtime_device, dtype=torch.float32).unsqueeze(0),
            action_space="raw",
            frame_start_override=history_frame_start,
            proprio_state=proprio_state,
        )
        synchronize_devices(runtime_device)
        warmup_s = time.perf_counter() - warmup_t0

        infer_t0 = time.perf_counter()
        chunk = runner.infer_chunk(session=warmup.session, proprio_state=proprio_state)
        synchronize_devices(runtime_device)
        infer_s = time.perf_counter() - infer_t0

    ready_monotonic_s = time.perf_counter()
    # The exact runtime's internal frame-start metadata drifts with raw
    # observation count during cache warmup, but the realtime scheduler needs
    # the next executable chunk to remain aligned to the observed LIBERO frame.
    generation_frame_start = observed_frame_index + 1
    model_generation_frame_start = int(chunk.debug.get("generation_frame_start", generation_frame_start))
    planned_frames = _chunk_to_planned_frames(
        first_chunk=chunk,
        frame_chunk_size=int(config.inference.frame_chunk_size),
        action_per_frame=int(config.policy_variant.action_per_frame),
        source="history_replan",
        ready_monotonic_s=ready_monotonic_s,
        generation_frame_start=generation_frame_start,
    )
    return {
        "job_kind": "history_replan",
        "session": chunk.session,
        "warmup_session": warmup.session,
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
            "history_frame_start": int(history_frame_start),
            "raw_observation_count": int(raw_observation_count),
            "precomputed_video_latent_frames": (
                0
                if precomputed_video_latents is None
                else int(precomputed_video_latents.shape[2])
            ),
            "warmup_video_latent_frames": int(prepared["video_latents"].shape[2]),
            "action_history_steps": int(action_history.shape[0]),
            "session_frame_start_before": int(session.policy_state.cache.get("frame_start", -1)),
            "session_step_before": int(session.policy_state.step_index),
            "session_step_after": int(chunk.session.policy_state.step_index),
            "generation_frame_start": generation_frame_start,
            "model_generation_frame_start": model_generation_frame_start,
            "session_frame_start_after_model": int(chunk.session.policy_state.cache.get("frame_start", -1)),
            "planned_frame_ids": [int(plan.absolute_frame_index) for plan in planned_frames],
            "prepare_s": float(prepare_s),
            "warmup_s": float(warmup_s),
            "infer_s": float(infer_s),
            "total_latency_s": float(prepare_s + warmup_s + infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
        },
        "submitted_through_frame": observed_frame_index,
    }


def run_extension_job(
    *,
    runner,
    session,
    config,
    runtime_device: torch.device,
    job_seed: int | None = None,
) -> dict[str, Any]:
    with isolated_torch_rng(job_seed, runtime_device), torch.inference_mode():
        infer_t0 = time.perf_counter()
        chunk = runner.infer_chunk(session=session, advance_frame_start=True)
        synchronize_devices(chunk.chunk_action_pred.device)
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
    precomputed_video_latents: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if not history_views and precomputed_video_latents is None:
        raise ValueError("Expected observed frames or precomputed latents for history preparation.")
    prepared = None
    resolved_text_context = text_context
    resolved_negative_text_context = negative_text_context
    if history_views:
        prepared = exact_viz.prepare_exact_runtime_inputs(
            runner,
            views=exact_viz.observations_to_views(history_views, device=frontend_device),
            task_text=task_text,
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            preserve_stream_cache=True,
        )
        resolved_text_context = prepared["text_context"]
        resolved_negative_text_context = prepared["negative_text_context"]
    latent_chunks: list[torch.Tensor] = []
    if precomputed_video_latents is not None:
        latent_chunks.append(precomputed_video_latents.to(device=runtime_device))
    if prepared is not None:
        latent_chunks.append(prepared["video_latents"])
    return {
        "video_latents": torch.cat(latent_chunks, dim=2),
        "text_context": resolved_text_context,
        "negative_text_context": resolved_negative_text_context,
    }


def _history_records_to_obs_sequence(history_records: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    history_views: list[dict[str, np.ndarray]] = []
    for record in history_records:
        obs_sequence = record.get("obs_sequence")
        if obs_sequence is not None:
            history_views.extend(
                {
                    key: np.array(value, copy=True)
                    for key, value in obs.items()
                }
                for obs in obs_sequence
            )
            continue
        history_views.append(
            {
                key: np.array(value, copy=True)
                for key, value in record["obs"].items()
            }
        )
    return history_views


def _count_history_raw_observations(history_records: list[dict[str, Any]]) -> int:
    raw_count = 0
    for record in history_records:
        obs_sequence = record.get("obs_sequence")
        if obs_sequence is not None:
            raw_count += len(obs_sequence)
        elif "obs" in record and not isinstance(record.get("video_latents"), torch.Tensor):
            raw_count += 1
    return int(raw_count)


def _history_records_to_action_history(history_records: list[dict[str, Any]], *, config) -> np.ndarray:
    action_rows: list[np.ndarray] = []
    inferred_dim: int | None = None
    for record in history_records:
        if "raw_action_dim" in record:
            inferred_dim = int(record["raw_action_dim"])
        raw_actions = record.get("raw_actions")
        if raw_actions is None or record.get("raw_actions_valid") is False:
            continue
        raw_array = np.asarray(raw_actions, dtype=np.float32)
        if raw_array.ndim != 2:
            raise ValueError(f"History raw_actions must be [T, D], got {raw_array.shape}.")
        inferred_dim = int(raw_array.shape[-1])
        if int(raw_array.shape[0]) > 0:
            action_rows.append(raw_array)
    if action_rows:
        return np.concatenate(action_rows, axis=0)
    if inferred_dim is None:
        action_schema = getattr(getattr(config, "data", None), "action_schema", None)
        inferred_dim = int(getattr(action_schema, "action_dim", 0) or 0)
    if inferred_dim is None or inferred_dim <= 0:
        raise ValueError("Unable to infer raw action dimension for empty exact rollout history.")
    return np.zeros((0, int(inferred_dim)), dtype=np.float32)


def _history_records_to_precomputed_video_latents(history_records: list[dict[str, Any]]) -> torch.Tensor | None:
    latent_chunks: list[torch.Tensor] = []
    for record in history_records:
        video_latents = record.get("video_latents")
        if isinstance(video_latents, torch.Tensor):
            latent_chunks.append(video_latents.detach())
    if not latent_chunks:
        return None
    return torch.cat(latent_chunks, dim=2)


def _history_records_to_proprio_state(
    history_records: list[dict[str, Any]],
    *,
    config,
    device: torch.device,
) -> torch.Tensor | None:
    if not exact_viz.proprio_context_enabled(config):
        return None
    for record in reversed(history_records):
        proprio_state = record.get("proprio_state")
        if proprio_state is not None:
            return torch.as_tensor(proprio_state, device=device, dtype=torch.float32).reshape(1, -1)
    return None


def synchronize_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        if device.type != "cuda":
            continue
        key = (device.type, device.index)
        if key in seen:
            continue
        torch.cuda.synchronize(device)
        seen.add(key)


def build_fallback_frame_actions(
    *,
    action_dim: int,
    action_per_frame: int,
    policy: str,
    last_action: np.ndarray,
    preserve_absolute_tail_from: int | None = 6,
) -> np.ndarray:
    deadline_policy = DeadlineMissPolicy(policy)
    if deadline_policy is DeadlineMissPolicy.ZERO:
        return np.zeros((action_per_frame, action_dim), dtype=np.float32)
    if deadline_policy is DeadlineMissPolicy.HOLD_STATE:
        action = np.zeros((action_dim,), dtype=np.float32)
        last = np.asarray(last_action, dtype=np.float32)
        if last.shape != (action_dim,):
            raise ValueError(
                "Fallback last-action shape mismatch, "
                f"expected {(action_dim,)}, got {tuple(last.shape)}."
            )
        if preserve_absolute_tail_from is not None and action_dim > int(preserve_absolute_tail_from):
            action[int(preserve_absolute_tail_from) :] = last[int(preserve_absolute_tail_from) :]
        return np.repeat(action[None, :], action_per_frame, axis=0)
    repeated = np.repeat(np.asarray(last_action, dtype=np.float32)[None, :], action_per_frame, axis=0)
    if repeated.shape != (action_per_frame, action_dim):
        raise ValueError(
            "Fallback last-action shape mismatch, "
            f"expected {(action_per_frame, action_dim)}, got {tuple(repeated.shape)}."
        )
    return repeated
