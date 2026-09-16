"""DualExpert rollout chunk, history, action, and execution geometry."""

from __future__ import annotations

from typing import Protocol

from open_wam.configs import CurrentBlockCoupling
from open_wam.models.common.denoising import independently_generated_modalities
from open_wam.models.common.temporal_windows import (
    resolve_interleaved_cache_frames,
    resolve_interleaved_history_frames,
)
from open_wam.models.policy_variants.contracts import (
    PolicyInferenceOutputRequest,
    PolicyOutputModality,
    PolicyTemporalGeometry,
    PolicyVideoConditionedActionRequest,
    PolicyVideoGenerationRequest,
)




class _InferenceContextLike(Protocol):
    output_request: PolicyInferenceOutputRequest | None
    video_generation: PolicyVideoGenerationRequest | None
    video_conditioned_action: PolicyVideoConditionedActionRequest | None
    temporal_geometry: PolicyTemporalGeometry | None


def resolve_dual_expert_sequence_actions_per_frame(
    *, action_horizon: int, frame_chunk_size: int
) -> int:
    """Resolve low-level control actions represented by one generated video frame."""

    action_horizon = int(action_horizon)
    frame_chunk_size = int(frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"Expected frame_chunk_size > 0, got {frame_chunk_size}.")
    if action_horizon <= 0:
        raise ValueError(f"Expected action_horizon > 0, got {action_horizon}.")
    if action_horizon % frame_chunk_size != 0:
        raise ValueError(
            "DualExpert realtime rollout expects action_horizon to divide by inference.frame_chunk_size, "
            f"got action_horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return action_horizon // frame_chunk_size


def resolve_rollout_frame_chunk_size(
    context: _InferenceContextLike,
    *,
    default_frame_chunk_size: int,
    base_action_horizon: int,
) -> tuple[int, int, int]:
    """Resolve rollout video/action chunk geometry without changing token density."""

    base_frame_chunk_size = int(default_frame_chunk_size)
    if base_frame_chunk_size <= 0:
        raise ValueError(
            "DualExpert inference frame chunk size must be positive, "
            f"got {base_frame_chunk_size}."
        )
    base_action_horizon = int(base_action_horizon)
    if base_action_horizon <= 0:
        raise ValueError(
            f"DualExpert inference action horizon must be positive, got {base_action_horizon}."
        )
    if base_action_horizon % base_frame_chunk_size != 0:
        raise ValueError(
            "DualExpert inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
            f"got action_horizon={base_action_horizon}, frame_chunk_size={base_frame_chunk_size}."
        )
    action_tokens_per_frame = base_action_horizon // base_frame_chunk_size
    temporal_geometry = getattr(context, "temporal_geometry", None)
    fallback_frame_chunk_size = (
        base_frame_chunk_size
        if temporal_geometry is None
        else int(temporal_geometry.frame_chunk_size)
    )
    generation_request = getattr(context, "video_generation", None)
    conditioned_action_request = getattr(context, "video_conditioned_action", None)
    requested_video_frames = None
    if generation_request is not None:
        requested_video_frames = int(generation_request.frame_count)
    elif conditioned_action_request is not None:
        requested_video_frames = int(
            conditioned_action_request.generated_video.latents.shape[2]
        )
    frame_chunk_size = (
        requested_video_frames if requested_video_frames is not None
        else fallback_frame_chunk_size
    )
    if frame_chunk_size <= 0:
        raise ValueError(
            f"DualExpert rollout frame chunk size must be positive, got {frame_chunk_size}."
        )
    if frame_chunk_size > base_frame_chunk_size:
        raise ValueError(
            "DualExpert rollout frame chunk size cannot exceed the configured inference frame chunk size, "
            f"got override={frame_chunk_size}, configured={base_frame_chunk_size}."
        )
    action_horizon = frame_chunk_size * action_tokens_per_frame
    return frame_chunk_size, action_horizon, action_tokens_per_frame


def resolve_dual_expert_inference_output_request(
    context: _InferenceContextLike,
    *,
    current_block_coupling: CurrentBlockCoupling,
    native_modalities: frozenset[PolicyOutputModality] = frozenset(
        PolicyOutputModality
    ),
) -> PolicyInferenceOutputRequest:
    """Resolve output selection without conflating native and selective routes."""

    native = frozenset(
        PolicyOutputModality(modality) for modality in native_modalities
    )
    if not native:
        raise ValueError("DualExpert inference must declare a native output modality.")
    request = context.output_request or PolicyInferenceOutputRequest(native)

    if not request.modalities.issubset(native):
        raise ValueError(
            "DualExpert inference requested outputs outside the program's native "
            "contract: "
            f"native={sorted(item.value for item in native)}, "
            f"requested={sorted(item.value for item in request.modalities)}."
        )
    if request.modalities != native and not request.modalities.issubset(independently_generated_modalities(current_block_coupling)):
        raise ValueError(
            "Selective inference requires independently generated outputs; "
            f"program coupling={current_block_coupling.value!r}, requested={request.modalities}."
        )
    return request


def resolve_dual_expert_rollout_history_frames(
    *, window_size: int, frame_chunk_size: int
) -> int:
    """History frames visible to the current chunk under block-id windowing.

    Video and action chunks occupy alternating block ids, so odd attention
    windows do not expose an extra complete same-stream history chunk. That
    gives floor semantics for per-stream lookback, matching parallel-stream cache
    retention and dual-expert's packed rollout-history contract.
    """

    return resolve_interleaved_history_frames(
        window_size=window_size,
        frame_chunk_size=frame_chunk_size,
    )


def resolve_dual_expert_rollout_cache_window_frames(
    *, window_size: int, frame_chunk_size: int
) -> int:
    """Total cached clean frames to retain: visible history plus the current chunk."""

    return resolve_interleaved_cache_frames(
        window_size=window_size,
        frame_chunk_size=frame_chunk_size,
    )


__all__ = [
    "resolve_dual_expert_inference_output_request",
    "resolve_dual_expert_rollout_cache_window_frames",
    "resolve_rollout_frame_chunk_size",
    "resolve_dual_expert_rollout_history_frames",
    "resolve_dual_expert_sequence_actions_per_frame",
]
