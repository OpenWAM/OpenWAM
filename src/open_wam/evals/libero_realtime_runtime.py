"""Realtime LIBERO planner jobs and observed-history execution contracts."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from open_wam.configs import ActionTargetRepresentation, ExperimentConfig
from open_wam.configs.enums import (
    RealtimePlannerJob,
    RealtimePlannerMode,
)
from open_wam.evals import libero_visualization as exact_viz
from open_wam.evals import realtime_speculation
from open_wam.evals.libero_realtime_plans import (
    FramePlannerJobResult,
    FramePlannerResultApplication,
    SequenceReplanJobOptions,
    SequenceReplanJobResult,
    _chunk_to_planned_frames,
    _session_for_next_chunk,
    annotate_sequence_planner_acceptance,
    apply_frame_planner_result,
    apply_sequence_replan_result,
    build_exact_startup_conditioning_history_record,
    build_fallback_frame_actions,
    exact_chunk_to_planned_steps,
    materialize_sequence_control_action,
    resolve_exact_startup_sessions,
    resolve_next_exact_history_base_session,
    sequence_chunk_to_planned_steps,
)
from open_wam.integrations import libero_rollout
from open_wam.integrations.realtime_contracts import (
    PlannedFrameAction as PlannedFrameAction,
)
from open_wam.integrations.realtime_scheduling import select_realtime_planner_job
from open_wam.models.policy_variants import (
    PolicyInferContext,
)
from open_wam.models.policy_variants.dual_expert.runtime_routes import (
    resolve_dual_expert_runtime_route,
)
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot
from open_wam.pipelines import VariantRolloutRunner, VariantRolloutSession
from open_wam.runtime import rollout as rollout_runtime
from open_wam.utils import validate_positive_step_override

from .libero_realtime_history_inputs import (
    _count_history_raw_observations,
    _history_records_to_action_history,
    _history_records_to_obs_sequence,
    _history_records_to_precomputed_video_latents,
    _history_records_to_proprio_state,
    _prepare_history_runtime_inputs,
    copy_history_record_for_worker,
)
from .libero_realtime_sequence import (
    build_sequence_startup_observation_window,
    collect_decoder_runtime_metadata,
    resolve_observation_conditioned_replan_session,
    resolve_sequence_action_cache_rewind_frame,
    resolve_sequence_actions_per_frame,
    resolve_sequence_condition_frame_start,
    resolve_sequence_execution_action_offset,
    resolve_sequence_model_observation_window_frames,
    resolve_sequence_startup_environment_frames,
    sequence_buffer_tail_ready_for_history_promotion,
    sequence_history_replan_ready,
    should_use_sequence_open_loop_extension,
    uses_dual_expert_split_cache_sequence,
    uses_strict_dual_expert_one_frame_history,
    uses_strict_dual_expert_split_cache_startup,
    validate_sequence_startup_inputs,
    validate_sequence_startup_open_loop_support,
)

_RUNTIME_COMPATIBILITY_EXPORTS = (
    ActionTargetRepresentation,
    PlannedFrameAction,
    exact_viz,
)


__all__ = [
    "FramePlannerJobResult",
    "FramePlannerResultApplication",
    "SequenceReplanJobOptions",
    "SequenceReplanJobResult",
    "annotate_sequence_planner_acceptance",
    "apply_frame_planner_result",
    "apply_inference_overrides",
    "apply_sequence_replan_result",
    "build_exact_startup_conditioning_history_record",
    "build_fallback_frame_actions",
    "build_sequence_startup_observation_window",
    "collect_decoder_runtime_metadata",
    "copy_history_record_for_worker",
    "exact_chunk_to_planned_steps",
    "isolated_torch_rng",
    "job_seed_for_session",
    "materialize_sequence_control_action",
    "resolve_exact_startup_sessions",
    "resolve_next_exact_history_base_session",
    "resolve_observation_conditioned_replan_session",
    "resolve_sequence_action_cache_rewind_frame",
    "resolve_sequence_actions_per_frame",
    "resolve_sequence_condition_frame_start",
    "resolve_sequence_execution_action_offset",
    "resolve_sequence_model_observation_window_frames",
    "resolve_sequence_startup_environment_frames",
    "run_extension_job",
    "run_replan_job",
    "run_sequence_replan_job",
    "sequence_buffer_tail_ready_for_history_promotion",
    "sequence_chunk_to_planned_steps",
    "sequence_history_replan_ready",
    "should_use_sequence_open_loop_extension",
    "submit_planner_job_with_snapshot",
    "synchronize_devices",
    "uses_dual_expert_split_cache_sequence",
    "uses_strict_dual_expert_one_frame_history",
    "uses_strict_dual_expert_split_cache_startup",
    "validate_sequence_startup_inputs",
    "validate_sequence_startup_open_loop_support",
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
    variant = getattr(runner, "policy_variant", None)
    if variant is None:
        variant = getattr(getattr(runner, "pipeline", None), "policy_variant", None)
    if variant is None and any(
        value is not None
        for value in (video_steps, action_steps, guidance_scale, action_guidance_scale)
    ):
        raise AttributeError(
            f"{type(runner).__name__} exposes neither `policy_variant` nor "
            "`pipeline.policy_variant`, so inference overrides cannot be applied"
        )
    if video_steps is not None:
        object.__setattr__(
            variant.inference_config,
            "video_num_inference_steps",
            video_steps,
        )
    if action_steps is not None:
        object.__setattr__(
            variant.inference_config,
            "action_num_inference_steps",
            action_steps,
        )
    if guidance_scale is not None:
        object.__setattr__(variant.inference_config, "guidance_scale", float(guidance_scale))
    if action_guidance_scale is not None:
        object.__setattr__(
            variant.inference_config,
            "action_guidance_scale",
            float(action_guidance_scale),
        )


def run_sequence_replan_job(
    *,
    runner: VariantRolloutRunner,
    session: VariantRolloutSession,
    obs_window: list[dict[str, np.ndarray]],
    config: ExperimentConfig,
    options: SequenceReplanJobOptions,
) -> SequenceReplanJobResult:
    """Run one sequence-policy planner job from observed or cached history."""

    rng_snapshot = (
        realtime_speculation.snapshot_rng_state()
        if options.preserve_rng_state
        else None
    )
    with torch.inference_mode():
        if options.runtime_cache_snapshot is not None:
            realtime_speculation.restore_visual_runtime(
                runner=runner,
                snapshot=options.runtime_cache_snapshot,
            )
        prepare_t0 = time.perf_counter()
        rollout_inputs = rollout_runtime.prepare_rollout_observation_inputs(
            runner.pipeline,
            views=libero_rollout.libero_observation_window_to_views(
                obs_window,
                device=options.frontend_device,
            ),
            task_text=(options.prompt,),
            frontend_device=options.frontend_device,
            runtime_device=options.runtime_device,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        synchronize_devices(options.frontend_device, options.runtime_device)
        prepare_s = time.perf_counter() - prepare_t0
        validate_sequence_startup_inputs(
            config=config,
            source=options.source,
            generation_action_start=int(options.generation_action_start),
            video_latents=rollout_inputs.get("video_latents"),
        )

        infer_t0 = time.perf_counter()
        inference_session = (
            resolve_observation_conditioned_replan_session(
                runner=runner,
                session=session,
                config=config,
            )
            if options.reset_observation_conditioned_session
            else session
        )
        reset_session_for_replan = inference_session is not session
        infer_extra = rollout_runtime.build_sequence_rollout_infer_extra(
            policy_variant=runner.pipeline.policy_variant,
            prompt=options.prompt,
            runtime_device=options.runtime_device,
        )
        dual_expert_runtime_route = resolve_dual_expert_runtime_route(config)
        if (
            dual_expert_runtime_route.is_dual_expert
            and dual_expert_runtime_route.supports_realtime_history_controls
        ):
            infer_extra["dual_expert_skip_observation_update"] = not bool(
                options.use_observation_update
            )
            if options.dual_expert_condition_frame_start is not None:
                infer_extra["dual_expert_condition_frame_start"] = int(
                    options.dual_expert_condition_frame_start
                )
            if options.dual_expert_action_cache_rewind_frame_start is not None:
                infer_extra["dual_expert_action_cache_rewind_frame_start"] = int(
                    options.dual_expert_action_cache_rewind_frame_start
                )
        elif dual_expert_runtime_route.is_dual_expert and not bool(options.use_observation_update):
            raise ValueError(
                "Dual-expert runtime route does not support split-cache open-loop controls: "
                f"{dual_expert_runtime_route.to_report()}"
            )
        step_output = runner.infer_step(
            session=inference_session,
            context=PolicyInferContext(
                state=libero_rollout.build_libero_state_history(
                    obs_window,
                    state_horizon=int(config.data.action_schema.state_horizon),
                    state_encoding=config.data.action_target.state_encoding,
                )
                .unsqueeze(0)
                .to(device=options.runtime_device),
                extra=infer_extra,
            ),
            video_latents=rollout_inputs["video_latents"],
            canonical_video=None,
        )
        synchronize_devices(options.runtime_device)
        infer_s = time.perf_counter() - infer_t0
        output_runtime_cache_snapshot = (
            realtime_speculation.snapshot_sequence_visual_runtime(
                runner=runner,
                config=config,
                session=step_output.session,
            )
        )

    policy_aux = step_output.infer_output.policy_output.aux
    dual_expert_cache_debug = policy_aux.get("dual_expert_cache_debug")
    if not isinstance(dual_expert_cache_debug, dict):
        dual_expert_cache_debug = {}
    predicted_latents = policy_aux.get("predicted_latents")
    action_plan = runner.build_action_rollout_plan(
        step_output.infer_output.decoder_output
    )
    action_pred = action_plan.actions.numpy()
    action_plan_metadata = action_plan.to_metadata()
    runner.commit_action_rollout_plan(
        session=step_output.session,
        plan=action_plan,
    )
    ready_monotonic_s = time.perf_counter()
    planned_steps = sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs=obs_window[-1],
        generation_action_start=options.generation_action_start,
        execution_action_offset=resolve_sequence_execution_action_offset(config),
        source=options.source,
        planner_step_index=(
            None
            if step_output.session.policy_state is None
            else int(step_output.session.policy_state.step_index)
        ),
        ready_monotonic_s=ready_monotonic_s,
        action_target_representation=config.data.action_target.representation,
        rotation_representation=str(
            config.data.action_target.rotation_representation
        ),
    )
    next_generation_action_start = int(options.generation_action_start) + int(
        action_pred.shape[0]
    )
    result = SequenceReplanJobResult(
        session=step_output.session,
        runtime_cache_snapshot=output_runtime_cache_snapshot,
        planned_steps=planned_steps,
        next_generation_action_start=int(next_generation_action_start),
        trace={
            "job_kind": "history_replan",
            "source": str(options.source),
            "reset_observation_conditioned_session": bool(
                reset_session_for_replan
            ),
            "use_observation_update": bool(options.use_observation_update),
            "observed_action_index": int(
                max(-1, int(options.generation_action_start) - 1)
            ),
            "history_frame_count": len(obs_window),
            "generation_action_start": int(options.generation_action_start),
            "execution_action_offset": int(
                resolve_sequence_execution_action_offset(config)
            ),
            "dual_expert_condition_frame_start": options.dual_expert_condition_frame_start,
            "dual_expert_action_cache_rewind_frame_start": (
                options.dual_expert_action_cache_rewind_frame_start
            ),
            "dual_expert_runtime_route": (
                dual_expert_runtime_route.to_report() if dual_expert_runtime_route.is_dual_expert else None
            ),
            "model_generation_frame_start": _json_scalar_from_tensor(
                policy_aux.get("generation_frame_start")
            ),
            "dual_expert_chunk_origin_frame": _json_scalar_from_tensor(
                dual_expert_cache_debug.get("chunk_origin_frame")
            ),
            "dual_expert_current_action_frame_start": _json_scalar_from_tensor(
                dual_expert_cache_debug.get("current_action_frame_start")
            ),
            "preserve_rng_state": bool(options.preserve_rng_state),
            "planned_action_ids": [
                int(plan.absolute_action_index) for plan in planned_steps
            ],
            "prepare_s": float(prepare_s),
            "warmup_s": 0.0,
            "infer_s": float(infer_s),
            "total_latency_s": float(prepare_s + infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
            "video_condition_source": policy_aux.get("video_condition_source"),
            "video_condition_uses_future_ground_truth": policy_aux.get(
                "video_condition_uses_future_ground_truth"
            ),
            "predicted_video_latents_shape": (
                list(predicted_latents.shape)
                if isinstance(predicted_latents, torch.Tensor)
                else None
            ),
            **collect_decoder_runtime_metadata(runner.pipeline, config),
            **action_plan_metadata,
            "decoder_sampled_new_chunk": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get(
                    "sampled_new_chunk"
                )
            ),
            "decoder_num_inference_steps": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get(
                    "num_inference_steps"
                )
            ),
            "decoder_current_action_index": _json_scalar_from_tensor(
                step_output.infer_output.decoder_output.aux.get(
                    "current_action_index"
                )
            ),
        },
    )
    realtime_speculation.restore_rng_state(rng_snapshot)
    return result


def _json_scalar_from_tensor(value) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _submit_planner_job(
    *,
    selected_job: RealtimePlannerJob,
    executor: ThreadPoolExecutor,
    pending_history: list[dict[str, Any]],
    runner,
    history_base_session,
    current_chunk_session,
    prompt: str,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    buffer_tail_session,
    seed_base: int | None = None,
) -> Future[FramePlannerJobResult]:
    history_payload = [
        copy_history_record_for_worker(record)
        for record in pending_history
    ]
    if selected_job == RealtimePlannerJob.HISTORY_REPLAN:
        if not history_payload:
            raise RuntimeError("History replan selected without any observed history.")
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
    if selected_job == RealtimePlannerJob.BUFFER_EXTENSION:
        if buffer_tail_session is None:
            raise RuntimeError("Buffer extension selected without a tail session.")
        return executor.submit(
            run_extension_job,
            runner=runner,
            session=buffer_tail_session,
            config=config,
            runtime_device=runtime_device,
            job_seed=job_seed_for_session(seed_base, buffer_tail_session),
        )
    raise AssertionError(f"Unhandled realtime planner job: {selected_job!r}")


def submit_planner_job_with_snapshot(
    *,
    executor: ThreadPoolExecutor,
    planner_mode: RealtimePlannerMode | str,
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
) -> tuple[
    Future[FramePlannerJobResult] | None,
    VisualRuntimeStateSnapshot | None,
]:
    """Submit a planner branch with a rollback point for shared visual state."""

    selected_job = select_realtime_planner_job(
        planner_mode=planner_mode,
        history_count=len(pending_history),
        future_buffer_depth=future_buffer_depth,
        has_buffer_tail_session=buffer_tail_session is not None,
    )
    if selected_job is None:
        return None, None
    snapshot = realtime_speculation.snapshot_visual_runtime(
        runner=runner,
        config=config,
        session=current_chunk_session,
    )
    submitted_future = _submit_planner_job(
        selected_job=selected_job,
        executor=executor,
        pending_history=pending_history,
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
    return submitted_future, snapshot


def job_seed_for_session(seed_base: int | None, session) -> int | None:
    if seed_base is None:
        return None
    return int(seed_base) + int(session.policy_state.step_index)


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
) -> FramePlannerJobResult:
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
    return FramePlannerJobResult(
        job_kind="history_replan",
        session=chunk.session,
        warmup_session=warmup.session,
        buffer_tail_session=_session_for_next_chunk(
            chunk.session,
            next_frame_start=generation_frame_start + int(config.inference.frame_chunk_size),
            frame_chunk_size=int(config.inference.frame_chunk_size),
        ),
        planned_frames=planned_frames,
        trace={
            "job_kind": "history_replan",
            "observed_frame_index": int(observed_frame_index),
            "history_frame_count": len(history_records),
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
        submitted_through_frame=observed_frame_index,
    )


def run_extension_job(
    *,
    runner,
    session,
    config,
    runtime_device: torch.device,
    job_seed: int | None = None,
) -> FramePlannerJobResult:
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
    return FramePlannerJobResult(
        job_kind="open_loop_extension",
        planned_frames=planned_frames,
        buffer_tail_session=chunk.session,
        trace={
            "job_kind": "open_loop_extension",
            "generation_frame_start": generation_frame_start,
            "planned_frame_ids": [int(plan.absolute_frame_index) for plan in planned_frames],
            "prepare_s": 0.0,
            "warmup_s": 0.0,
            "infer_s": float(infer_s),
            "total_latency_s": float(infer_s),
            "ready_monotonic_s": float(ready_monotonic_s),
        },
        submitted_through_frame=None,
    )


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
