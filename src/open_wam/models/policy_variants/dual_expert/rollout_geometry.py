"""DualExpert rollout chunk, history, action, and execution geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from open_wam.configs import CurrentBlockCoupling
from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment
from open_wam.models.common.temporal_windows import (
    resolve_interleaved_cache_frames,
    resolve_interleaved_history_frames,
)
from open_wam.models.policy_variants.contracts import (
    PolicyInferenceOutputRequest,
    PolicyOutputModality,
    PolicyVideoConditionedActionRequest,
    PolicyVideoGenerationRequest,
)

from .runtime_routes import _enum_value, resolve_dual_expert_runtime_route

DUAL_EXPERT_ACTION_ONLY_ROLLOUT_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)
DUAL_EXPERT_VIDEO_ONLY_ROLLOUT_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


class _InferenceContextLike(Protocol):
    output_request: PolicyInferenceOutputRequest | None
    video_generation: PolicyVideoGenerationRequest | None
    video_conditioned_action: PolicyVideoConditionedActionRequest | None
    extra: Mapping[str, Any]


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


def resolve_dual_expert_inference_window_size(
    context: _InferenceContextLike,
    *,
    default_window_size: int,
) -> int:
    """Resolve the positive rollout attention window from config and runtime overrides."""

    raw_override = context.extra.get("dual_expert_inference_window_size")
    generation_request = getattr(context, "video_generation", None)
    requested_window = (
        None if generation_request is None else generation_request.attention_window_size
    )
    if (
        requested_window is not None
        and raw_override is not None
        and int(requested_window) != int(raw_override)
    ):
        raise ValueError(
            "DualExpert inference-window override conflicts with the typed video "
            "request: "
            f"legacy={int(raw_override)}, typed={int(requested_window)}."
        )
    if requested_window is not None:
        resolved = int(requested_window)
    elif raw_override is not None:
        resolved = int(raw_override)
    else:
        resolved = int(default_window_size)
    if resolved <= 0:
        raise ValueError(f"DualExpert inference window size must be positive, got {resolved}.")
    return resolved


def resolve_dual_expert_rollout_frame_chunk_size(
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
    raw_override = context.extra.get("dual_expert_rollout_frame_chunk_size")
    generation_request = getattr(context, "video_generation", None)
    conditioned_action_request = getattr(context, "video_conditioned_action", None)
    requested_video_frames = None
    if generation_request is not None:
        requested_video_frames = int(generation_request.frame_count)
    elif conditioned_action_request is not None:
        requested_video_frames = int(
            conditioned_action_request.generated_video.latents.shape[2]
        )
    if (
        raw_override is not None
        and requested_video_frames is not None
        and int(raw_override) != requested_video_frames
    ):
        raise ValueError(
            "DualExpert rollout frame geometry conflicts with the typed video request: "
            f"legacy_override={int(raw_override)}, "
            f"requested_video_frames={requested_video_frames}."
        )
    if requested_video_frames is not None:
        frame_chunk_size = requested_video_frames
    elif raw_override is not None:
        frame_chunk_size = int(raw_override)
    else:
        frame_chunk_size = base_frame_chunk_size
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
    legacy_action_only = bool(
        context.extra.get("dual_expert_action_only_rollout", False)
    )
    request = getattr(context, "output_request", None)
    if request is None:
        request = (
            PolicyInferenceOutputRequest.action_only()
            if legacy_action_only
            else PolicyInferenceOutputRequest(native)
        )
    elif legacy_action_only and request != PolicyInferenceOutputRequest.action_only():
        raise ValueError(
            "The legacy `dual_expert_action_only_rollout` flag conflicts with "
            f"the typed output request {sorted(item.value for item in request.modalities)}."
        )

    if not request.modalities.issubset(native):
        raise ValueError(
            "DualExpert inference requested outputs outside the program's native "
            "contract: "
            f"native={sorted(item.value for item in native)}, "
            f"requested={sorted(item.value for item in request.modalities)}."
        )
    is_selective = request.modalities != native
    action_only = is_selective and request.modalities == frozenset(
        {PolicyOutputModality.ACTION}
    )
    video_only = is_selective and request.modalities == frozenset(
        {PolicyOutputModality.VIDEO}
    )
    if (
        action_only
        and current_block_coupling not in DUAL_EXPERT_ACTION_ONLY_ROLLOUT_COUPLINGS
    ):
        supported = ", ".join(
            (
                CurrentBlockCoupling.ACTION_THEN_VIDEO.value,
                CurrentBlockCoupling.DECOUPLED_SAME_STEP.value,
            )
        )
        raise ValueError(
            "`dual_expert_action_only_rollout` is only supported for dual-expert action-only-safe "
            f"couplings ({supported}); got current_block_coupling={current_block_coupling.value!r}."
        )
    if (
        video_only
        and current_block_coupling not in DUAL_EXPERT_VIDEO_ONLY_ROLLOUT_COUPLINGS
    ):
        raise ValueError(
            "DualExpert video-only inference requires a video-independent coupling "
            "(`video_then_action` or `decoupled_same_step`) so video denoising is "
            "complete without the omitted action stage; "
            f"got current_block_coupling={current_block_coupling.value!r}."
        )
    return request


def resolve_dual_expert_action_only_rollout(
    context: _InferenceContextLike,
    *,
    current_block_coupling: CurrentBlockCoupling,
    native_modalities: frozenset[PolicyOutputModality] = frozenset(
        PolicyOutputModality
    ),
) -> bool:
    """Compatibility resolver for the historical action-only boolean."""

    native = frozenset(
        PolicyOutputModality(modality) for modality in native_modalities
    )
    request = resolve_dual_expert_inference_output_request(
        context,
        current_block_coupling=current_block_coupling,
        native_modalities=native,
    )
    return (
        request.modalities != native
        and request.modalities == frozenset({PolicyOutputModality.ACTION})
    )


def resolve_dual_expert_sequence_execution_action_offset(
    config_or_policy_config: Any,
    *,
    action_horizon: int,
    frame_chunk_size: int,
) -> int:
    """Resolve action-index offset between model output and executable actions.

    Strict rollout-parity dual-expert emits only executable generated actions, including
    split-cache routes when the full experiment config is available. Older
    split-cache/legacy dual-expert routes can still include the observed frame's action
    group in the returned chunk, so they keep the historical one-frame
    execution reindexing behind the legacy config contract.
    """

    route = resolve_dual_expert_runtime_route(config_or_policy_config)
    if not route.is_dual_expert:
        return 0
    actions_per_frame = resolve_dual_expert_sequence_actions_per_frame(
        action_horizon=action_horizon,
        frame_chunk_size=frame_chunk_size,
    )
    if route.uses_native_packed_rollout or dual_expert_config_uses_strict_rollout_parity(
        config_or_policy_config
    ):
        return 0
    return actions_per_frame


def dual_expert_config_uses_strict_rollout_parity(config_or_policy_config: Any) -> bool:
    """Return whether the experiment data config uses strict rollout-parity targets."""

    data_config = getattr(config_or_policy_config, "data", None)
    sample_config = getattr(data_config, "sample_construction", None)
    if sample_config is None:
        return False
    return (
        _enum_value(getattr(sample_config, "target_alignment", None))
        == SampleTargetAlignment.NEXT_AFTER_CONTEXT.value
        and _enum_value(getattr(sample_config, "rollout_context_policy", None))
        == RolloutContextPolicy.ONE_FRAME.value
    )


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
    "DUAL_EXPERT_ACTION_ONLY_ROLLOUT_COUPLINGS",
    "dual_expert_config_uses_strict_rollout_parity",
    "resolve_dual_expert_action_only_rollout",
    "resolve_dual_expert_inference_output_request",
    "resolve_dual_expert_inference_window_size",
    "resolve_dual_expert_rollout_cache_window_frames",
    "resolve_dual_expert_rollout_frame_chunk_size",
    "resolve_dual_expert_rollout_history_frames",
    "resolve_dual_expert_sequence_actions_per_frame",
    "resolve_dual_expert_sequence_execution_action_offset",
]
