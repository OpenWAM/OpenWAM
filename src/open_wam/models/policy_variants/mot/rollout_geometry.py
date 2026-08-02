"""MoT rollout chunk, history, action, and execution geometry."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from open_wam.configs import CurrentBlockCoupling
from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment

from .runtime_routes import _enum_value, resolve_mot_runtime_route


MOT_ACTION_ONLY_ROLLOUT_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


class _InferenceContextLike(Protocol):
    extra: Mapping[str, Any]


def resolve_mot_sequence_actions_per_frame(
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
            "MoT realtime rollout expects action_horizon to divide by inference.frame_chunk_size, "
            f"got action_horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return action_horizon // frame_chunk_size


def resolve_mot_inference_window_size(
    context: _InferenceContextLike,
    *,
    default_window_size: int,
) -> int:
    """Resolve the positive rollout attention window from config and runtime overrides."""

    raw_override = context.extra.get("mot_inference_window_size")
    if raw_override is None:
        resolved = int(default_window_size)
    else:
        resolved = int(raw_override)
    if resolved <= 0:
        raise ValueError(f"MoT inference window size must be positive, got {resolved}.")
    return resolved


def resolve_mot_rollout_frame_chunk_size(
    context: _InferenceContextLike,
    *,
    default_frame_chunk_size: int,
    base_action_horizon: int,
) -> tuple[int, int, int]:
    """Resolve rollout video/action chunk geometry without changing token density."""

    base_frame_chunk_size = int(default_frame_chunk_size)
    if base_frame_chunk_size <= 0:
        raise ValueError(
            "MoT inference frame chunk size must be positive, "
            f"got {base_frame_chunk_size}."
        )
    base_action_horizon = int(base_action_horizon)
    if base_action_horizon <= 0:
        raise ValueError(
            f"MoT inference action horizon must be positive, got {base_action_horizon}."
        )
    if base_action_horizon % base_frame_chunk_size != 0:
        raise ValueError(
            "MoT inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
            f"got action_horizon={base_action_horizon}, frame_chunk_size={base_frame_chunk_size}."
        )
    action_tokens_per_frame = base_action_horizon // base_frame_chunk_size
    raw_override = context.extra.get("mot_rollout_frame_chunk_size")
    if raw_override is None:
        frame_chunk_size = base_frame_chunk_size
    else:
        frame_chunk_size = int(raw_override)
    if frame_chunk_size <= 0:
        raise ValueError(
            f"MoT rollout frame chunk size must be positive, got {frame_chunk_size}."
        )
    if frame_chunk_size > base_frame_chunk_size:
        raise ValueError(
            "MoT rollout frame chunk size cannot exceed the configured inference frame chunk size, "
            f"got override={frame_chunk_size}, configured={base_frame_chunk_size}."
        )
    action_horizon = frame_chunk_size * action_tokens_per_frame
    return frame_chunk_size, action_horizon, action_tokens_per_frame


def resolve_mot_action_only_rollout(
    context: _InferenceContextLike,
    *,
    current_block_coupling: CurrentBlockCoupling,
) -> bool:
    """Validate and resolve the optional action-only rollout route."""

    requested = bool(context.extra.get("mot_action_only_rollout", False))
    if requested and current_block_coupling not in MOT_ACTION_ONLY_ROLLOUT_COUPLINGS:
        supported = ", ".join(
            (
                CurrentBlockCoupling.ACTION_THEN_VIDEO.value,
                CurrentBlockCoupling.DECOUPLED_SAME_STEP.value,
            )
        )
        raise ValueError(
            "`mot_action_only_rollout` is only supported for M5 action-only-safe "
            f"couplings ({supported}); got current_block_coupling={current_block_coupling.value!r}."
        )
    return requested


def resolve_mot_sequence_execution_action_offset(
    config_or_policy_config: Any,
    *,
    action_horizon: int,
    frame_chunk_size: int,
) -> int:
    """Resolve action-index offset between model output and executable actions.

    Strict rollout-parity M5 emits only executable generated actions, including
    split-cache routes when the full experiment config is available. Older
    split-cache/legacy M5 routes can still include the observed frame's action
    group in the returned chunk, so they keep the historical one-frame
    execution reindexing behind the legacy config contract.
    """

    route = resolve_mot_runtime_route(config_or_policy_config)
    if not route.is_mot:
        return 0
    actions_per_frame = resolve_mot_sequence_actions_per_frame(
        action_horizon=action_horizon,
        frame_chunk_size=frame_chunk_size,
    )
    if route.uses_native_packed_rollout or mot_config_uses_strict_rollout_parity(
        config_or_policy_config
    ):
        return 0
    return actions_per_frame


def mot_config_uses_strict_rollout_parity(config_or_policy_config: Any) -> bool:
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


def resolve_mot_rollout_history_frames(
    *, window_size: int, frame_chunk_size: int
) -> int:
    """History frames visible to the current chunk under block-id windowing.

    Video and action chunks occupy alternating block ids, so odd attention
    windows do not expose an extra complete same-stream history chunk. That
    gives floor semantics for per-stream lookback, matching Method-1 cache
    retention and M5's fixed-128 rollout-history contract.
    """

    chunk = max(1, int(frame_chunk_size))
    window = max(1, int(window_size))
    return max(chunk, (window // 2) * chunk)


def resolve_mot_rollout_cache_window_frames(
    *, window_size: int, frame_chunk_size: int
) -> int:
    """Total cached clean frames to retain: visible history plus the current chunk."""

    chunk = max(1, int(frame_chunk_size))
    return (
        resolve_mot_rollout_history_frames(
            window_size=window_size,
            frame_chunk_size=chunk,
        )
        + chunk
    )


__all__ = [
    "MOT_ACTION_ONLY_ROLLOUT_COUPLINGS",
    "mot_config_uses_strict_rollout_parity",
    "resolve_mot_action_only_rollout",
    "resolve_mot_inference_window_size",
    "resolve_mot_rollout_cache_window_frames",
    "resolve_mot_rollout_frame_chunk_size",
    "resolve_mot_rollout_history_frames",
    "resolve_mot_sequence_actions_per_frame",
    "resolve_mot_sequence_execution_action_offset",
]
