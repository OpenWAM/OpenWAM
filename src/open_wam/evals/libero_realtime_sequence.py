"""Sequence-policy semantics for realtime LIBERO planning."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from open_wam.configs import ExperimentConfig
from open_wam.configs.enums import RealtimePlannerMode
from open_wam.evals import realtime_history
from open_wam.models.policy_variants.dual_expert.rollout_geometry import (
    dual_expert_config_uses_strict_rollout_parity,
    resolve_dual_expert_sequence_actions_per_frame,
    resolve_dual_expert_sequence_execution_action_offset,
)
from open_wam.models.policy_variants.dual_expert.runtime_routes import (
    DualExpertRuntimeRoute,
    resolve_dual_expert_runtime_route,
)
from open_wam.pipelines import VariantRolloutRunner, VariantRolloutSession


def uses_dual_expert_split_cache_sequence(config: ExperimentConfig) -> bool:
    """Return whether the sequence route shares the parallel-stream cache."""

    return resolve_dual_expert_runtime_route(config).uses_split_cache_rollout


def uses_strict_dual_expert_split_cache_startup(config: ExperimentConfig) -> bool:
    route = resolve_dual_expert_runtime_route(config)
    return bool(
        route.uses_split_cache_rollout
        and dual_expert_config_uses_strict_rollout_parity(config)
    )


def uses_strict_dual_expert_one_frame_history(config: ExperimentConfig) -> bool:
    route = resolve_dual_expert_runtime_route(config)
    return bool(route.is_dual_expert and dual_expert_config_uses_strict_rollout_parity(config))


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
    if uses_strict_dual_expert_one_frame_history(config):
        return [
            realtime_history.copy_observation(initial_observation_window[-1])
        ]
    return realtime_history.copy_observation_window(initial_observation_window)


def resolve_sequence_model_observation_window_frames(
    config: ExperimentConfig,
    *,
    raw_window_frames: int,
) -> int:
    if uses_strict_dual_expert_one_frame_history(config):
        return 1
    return int(raw_window_frames)


def resolve_sequence_startup_environment_frames(
    config: ExperimentConfig,
    *,
    raw_window_frames: int,
) -> int:
    if uses_strict_dual_expert_one_frame_history(config):
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
        not uses_strict_dual_expert_split_cache_startup(config)
        or str(source) != "startup_plan"
    ):
        return
    if int(generation_action_start) != 0:
        raise ValueError(
            "Dual-expert strict split-cache startup expects generation_action_start=0 so "
            "executable actions start at action index 0; "
            f"got {generation_action_start}."
        )
    if not isinstance(video_latents, torch.Tensor) or video_latents.ndim != 5:
        raise ValueError(
            "Dual-expert strict split-cache startup expects tensor video_latents with shape "
            "[B, C, T, H, W], "
            f"got {type(video_latents).__name__}."
        )
    latent_context_frames = int(video_latents.shape[2])
    if latent_context_frames != 1:
        raise ValueError(
            "Dual-expert strict split-cache startup expects exactly one latent context frame "
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

    dual_expert_runtime_route = resolve_dual_expert_runtime_route(config)
    if not dual_expert_runtime_route.is_dual_expert:
        return session
    if dual_expert_runtime_route.uses_split_cache_rollout:
        return session
    if dual_expert_runtime_route.uses_native_packed_rollout:
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

    if not resolve_dual_expert_runtime_route(config).supports_realtime_history_controls:
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
) -> DualExpertRuntimeRoute:
    """Reject open-loop startup when the selected policy route cannot support it."""

    dual_expert_runtime_route = resolve_dual_expert_runtime_route(config)
    if (
        dual_expert_runtime_route.is_dual_expert
        and int(startup_open_loop_chunks) > 0
        and not dual_expert_runtime_route.supports_realtime_history_controls
    ):
        raise ValueError(
            "Dual-expert runtime route does not support startup open-loop extension because "
            "it has no split-cache observation-skip/rewind controls. Use "
            "`startup_open_loop_chunks=0`, or a split-cache dual-expert route such as "
            "`video_then_action` / `decoupled_same_step`. Runtime route: "
            f"{dual_expert_runtime_route.to_report()}"
        )
    return dual_expert_runtime_route


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
    if not resolve_dual_expert_runtime_route(config).supports_realtime_history_controls:
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
    }


def resolve_sequence_execution_action_offset(config: ExperimentConfig) -> int:
    if str(config.policy_variant.name) != "dual_expert":
        return 0
    return resolve_dual_expert_sequence_execution_action_offset(
        config,
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
    )


def resolve_sequence_actions_per_frame(config: ExperimentConfig) -> int:
    return resolve_dual_expert_sequence_actions_per_frame(
        action_horizon=int(config.data.action_schema.action_horizon),
        frame_chunk_size=int(config.inference.frame_chunk_size),
    )


__all__ = [
    "build_sequence_startup_observation_window",
    "collect_decoder_runtime_metadata",
    "resolve_observation_conditioned_replan_session",
    "resolve_sequence_action_cache_rewind_frame",
    "resolve_sequence_actions_per_frame",
    "resolve_sequence_condition_frame_start",
    "resolve_sequence_execution_action_offset",
    "resolve_sequence_model_observation_window_frames",
    "resolve_sequence_startup_environment_frames",
    "sequence_buffer_tail_ready_for_history_promotion",
    "sequence_history_replan_ready",
    "should_use_sequence_open_loop_extension",
    "uses_dual_expert_split_cache_sequence",
    "uses_strict_dual_expert_one_frame_history",
    "uses_strict_dual_expert_split_cache_startup",
    "validate_sequence_startup_inputs",
    "validate_sequence_startup_open_loop_support",
]
