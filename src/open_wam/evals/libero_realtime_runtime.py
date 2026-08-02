"""Realtime LIBERO planner jobs and observed-history execution contracts."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any

import numpy as np
import torch

from open_wam.configs import ActionTargetRepresentation, ExperimentConfig
from open_wam.configs.enums import (
    RealtimePlannerJob,
    RealtimePlannerMode,
)
from open_wam.evals import libero_visualization as exact_viz
from open_wam.evals import realtime_history
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
from open_wam.integrations.realtime_contracts import PlannedFrameAction as PlannedFrameAction
from open_wam.integrations.realtime_scheduling import select_realtime_planner_job
from open_wam.models.policy_variants import (
    PolicyInferContext,
)
from open_wam.models.policy_variants.mot.rollout_geometry import (
    mot_config_uses_strict_rollout_parity,
    resolve_mot_sequence_actions_per_frame,
    resolve_mot_sequence_execution_action_offset,
)
from open_wam.models.policy_variants.mot.runtime_routes import (
    MoTRuntimeRoute,
    resolve_mot_runtime_route,
)
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot
from open_wam.pipelines import VariantRolloutRunner, VariantRolloutSession
from open_wam.runtime import rollout as rollout_runtime
from open_wam.utils import validate_positive_step_override


_PLANNER_COMPATIBILITY_EXPORTS = (
    ActionTargetRepresentation,
    PlannedFrameAction,
)


__all__ = [
    "FramePlannerJobResult",
    "FramePlannerResultApplication",
    "SequenceReplanJobOptions",
    "SequenceReplanJobResult",
    "annotate_sequence_planner_acceptance",
    "apply_inference_overrides",
    "apply_frame_planner_result",
    "apply_sequence_replan_result",
    "build_exact_startup_conditioning_history_record",
    "build_sequence_startup_observation_window",
    "build_fallback_frame_actions",
    "collect_decoder_runtime_metadata",
    "copy_history_record_for_worker",
    "exact_chunk_to_planned_steps",
    "isolated_torch_rng",
    "job_seed_for_session",
    "materialize_sequence_control_action",
    "resolve_observation_conditioned_replan_session",
    "resolve_sequence_action_cache_rewind_frame",
    "resolve_sequence_actions_per_frame",
    "resolve_sequence_condition_frame_start",
    "resolve_sequence_execution_action_offset",
    "resolve_sequence_model_observation_window_frames",
    "resolve_sequence_startup_environment_frames",
    "resolve_exact_startup_sessions",
    "resolve_next_exact_history_base_session",
    "run_extension_job",
    "run_replan_job",
    "run_sequence_replan_job",
    "sequence_buffer_tail_ready_for_history_promotion",
    "sequence_chunk_to_planned_steps",
    "sequence_history_replan_ready",
    "should_use_sequence_open_loop_extension",
    "submit_planner_job_with_snapshot",
    "synchronize_devices",
    "uses_mot_split_cache_sequence",
    "uses_strict_mot_split_cache_startup",
    "uses_strict_mot_one_frame_history",
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


def uses_mot_split_cache_sequence(config: ExperimentConfig) -> bool:
    """Return whether the sequence route shares the Method-1-style cache."""

    return resolve_mot_runtime_route(config).uses_split_cache_rollout


def uses_strict_mot_split_cache_startup(config: ExperimentConfig) -> bool:
    route = resolve_mot_runtime_route(config)
    return bool(
        route.uses_split_cache_rollout
        and mot_config_uses_strict_rollout_parity(config)
    )


def uses_strict_mot_one_frame_history(config: ExperimentConfig) -> bool:
    route = resolve_mot_runtime_route(config)
    return bool(route.is_mot and mot_config_uses_strict_rollout_parity(config))


def build_sequence_startup_observation_window(
    config: ExperimentConfig,
    initial_observation_window: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Select and copy the observed frames visible to sequence startup."""

    if not initial_observation_window:
        raise ValueError(
            "Cannot build sequence startup observation window from an empty "
            "initial window."
        )
    if uses_strict_mot_one_frame_history(config):
        return [
            realtime_history.copy_observation(initial_observation_window[-1])
        ]
    return realtime_history.copy_observation_window(initial_observation_window)


def resolve_sequence_model_observation_window_frames(
    config: ExperimentConfig,
    *,
    raw_window_frames: int,
) -> int:
    if uses_strict_mot_one_frame_history(config):
        return 1
    return int(raw_window_frames)


def resolve_sequence_startup_environment_frames(
    config: ExperimentConfig,
    *,
    raw_window_frames: int,
) -> int:
    if uses_strict_mot_one_frame_history(config):
        return 1
    return int(raw_window_frames)


def validate_sequence_startup_inputs(
    *,
    config: ExperimentConfig,
    source: str,
    generation_action_start: int,
    video_latents: object,
) -> None:
    """Fail when strict split-cache startup cannot match rollout parity."""

    if (
        not uses_strict_mot_split_cache_startup(config)
        or str(source) != "startup_plan"
    ):
        return
    if int(generation_action_start) != 0:
        raise ValueError(
            "M5 strict split-cache startup expects generation_action_start=0 so "
            "executable actions start at action index 0; "
            f"got {generation_action_start}."
        )
    if not isinstance(video_latents, torch.Tensor) or video_latents.ndim != 5:
        raise ValueError(
            "M5 strict split-cache startup expects tensor video_latents with shape "
            "[B, C, T, H, W], "
            f"got {type(video_latents).__name__}."
        )
    latent_context_frames = int(video_latents.shape[2])
    if latent_context_frames != 1:
        raise ValueError(
            "M5 strict split-cache startup expects exactly one latent context frame "
            f"before the first generated chunk; got {latent_context_frames}. This "
            "would break target_alignment=next_after_context parity."
        )


def resolve_observation_conditioned_replan_session(
    *,
    runner: VariantRolloutRunner,
    session: VariantRolloutSession,
    config: ExperimentConfig,
) -> VariantRolloutSession:
    """Resolve cache continuity for a newly observed sequence window."""

    mot_runtime_route = resolve_mot_runtime_route(config)
    if not mot_runtime_route.is_mot:
        return session
    if mot_runtime_route.uses_split_cache_rollout:
        return session
    if mot_runtime_route.uses_native_packed_rollout:
        return session
    return runner.reset(
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )


def should_use_sequence_open_loop_extension(
    *,
    config: ExperimentConfig,
    planner_mode: RealtimePlannerMode | str,
    remaining_buffer_actions: int,
) -> bool:
    """Gate sequence extension on route capability, mode, and buffered work."""

    if not resolve_mot_runtime_route(config).supports_realtime_history_controls:
        return False
    mode = RealtimePlannerMode(planner_mode)
    if mode not in {
        RealtimePlannerMode.ASYNC_BUFFER,
        RealtimePlannerMode.ASYNC_MIX,
        RealtimePlannerMode.ASYNC_HISTORY_FIRST,
    }:
        return False
    return int(remaining_buffer_actions) > 0


def validate_sequence_startup_open_loop_support(
    *,
    config: ExperimentConfig,
    startup_open_loop_chunks: int,
) -> MoTRuntimeRoute:
    """Reject open-loop startup when the selected policy route cannot support it."""

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


def resolve_sequence_action_cache_rewind_frame(
    *,
    config: ExperimentConfig,
    planner_mode: RealtimePlannerMode | str,
    use_observation_update: bool,
    condition_frame_start: int | None,
) -> int | None:
    """Resolve the speculative action-cache suffix replaced by a live replan."""

    if condition_frame_start is None:
        return None
    if not bool(use_observation_update):
        return None
    if not resolve_mot_runtime_route(config).supports_realtime_history_controls:
        return None
    if RealtimePlannerMode(planner_mode) not in {
        RealtimePlannerMode.ASYNC_MIX,
        RealtimePlannerMode.ASYNC_HISTORY_FIRST,
    }:
        return None
    return int(condition_frame_start)


def sequence_history_replan_ready(
    *,
    config: ExperimentConfig,
    next_action_index: int,
    generation_action_start: int,
) -> bool:
    execution_start = int(generation_action_start) - (
        resolve_sequence_execution_action_offset(config)
    )
    return int(next_action_index) >= int(execution_start)


def sequence_buffer_tail_ready_for_history_promotion(
    *,
    config: ExperimentConfig,
    next_action_index: int,
    buffer_tail_generation_action_start: int,
    history_generation_action_start: int,
) -> bool:
    if int(buffer_tail_generation_action_start) <= int(
        history_generation_action_start
    ):
        return False
    execution_start = int(buffer_tail_generation_action_start) - (
        resolve_sequence_execution_action_offset(config)
    )
    return int(next_action_index) >= int(execution_start)


def resolve_sequence_condition_frame_start(
    *,
    config: ExperimentConfig,
    generation_action_start: int,
) -> int:
    action_per_frame = resolve_sequence_actions_per_frame(config)
    return int(generation_action_start) // int(action_per_frame)


def collect_decoder_runtime_metadata(
    pipeline,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Collect stable decoder/runtime fields for load and replan traces."""

    decoder = getattr(pipeline, "action_decoder", None)
    generation_backend = getattr(decoder, "generation_backend", None)
    return {
        "action_decoder_class": (
            None if decoder is None else decoder.__class__.__name__
        ),
        "config_action_num_inference_steps": int(
            config.inference.action_num_inference_steps
        ),
        "config_video_num_inference_steps": int(
            config.inference.video_num_inference_steps
        ),
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


def resolve_sequence_execution_action_offset(config: ExperimentConfig) -> int:
    if str(config.policy_variant.name) != "mot":
        return 0
    return resolve_mot_sequence_execution_action_offset(
        config,
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
    )


def resolve_sequence_actions_per_frame(config: ExperimentConfig) -> int:
    return resolve_mot_sequence_actions_per_frame(
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
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
            config=config,
            prompt=options.prompt,
            generation_action_start=int(options.generation_action_start),
            runtime_device=options.runtime_device,
            task_id=int(options.task_id),
            episode_idx=int(options.episode_idx),
        )
        mot_runtime_route = resolve_mot_runtime_route(config)
        if (
            mot_runtime_route.is_mot
            and mot_runtime_route.supports_realtime_history_controls
        ):
            infer_extra["mot_skip_observation_update"] = not bool(
                options.use_observation_update
            )
            if options.mot_condition_frame_start is not None:
                infer_extra["mot_condition_frame_start"] = int(
                    options.mot_condition_frame_start
                )
            if options.mot_action_cache_rewind_frame_start is not None:
                infer_extra["mot_action_cache_rewind_frame_start"] = int(
                    options.mot_action_cache_rewind_frame_start
                )
        elif mot_runtime_route.is_mot and not bool(options.use_observation_update):
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
    mot_cache_debug = policy_aux.get("mot_cache_debug")
    if not isinstance(mot_cache_debug, dict):
        mot_cache_debug = {}
    sequence_context = (
        step_output.infer_output.policy_output.decoder_sequence_context
    )
    video_condition_window = (
        None
        if sequence_context is None
        else sequence_context.video_condition_window
    )
    video_condition_metadata = (
        {}
        if video_condition_window is None
        else dict(video_condition_window.metadata)
    )
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
            "history_frame_count": int(len(obs_window)),
            "generation_action_start": int(options.generation_action_start),
            "execution_action_offset": int(
                resolve_sequence_execution_action_offset(config)
            ),
            "mot_condition_frame_start": options.mot_condition_frame_start,
            "mot_action_cache_rewind_frame_start": (
                options.mot_action_cache_rewind_frame_start
            ),
            "mot_runtime_route": (
                mot_runtime_route.to_report() if mot_runtime_route.is_mot else None
            ),
            "model_generation_frame_start": _json_scalar_from_tensor(
                policy_aux.get("generation_frame_start")
            ),
            "mot_chunk_origin_frame": _json_scalar_from_tensor(
                mot_cache_debug.get("chunk_origin_frame")
            ),
            "mot_current_action_frame_start": _json_scalar_from_tensor(
                mot_cache_debug.get("current_action_frame_start")
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
            "video_condition_frame_start": video_condition_metadata.get(
                "frame_start"
            ),
            "video_condition_sample_seed": video_condition_metadata.get(
                "sample_seed"
            ),
            "video_condition_observed_prefix_anchor": (
                video_condition_metadata.get("observed_prefix_anchor")
            ),
            "video_condition_observed_prefix_start_index": (
                video_condition_metadata.get("observed_prefix_start_index")
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
