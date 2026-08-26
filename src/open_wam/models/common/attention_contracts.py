"""Declarative attention-profile contracts and semantic normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
)

if TYPE_CHECKING:
    from open_wam.configs import HistoryStreamVisibility

try:
    from torch.nn.attention.flex_attention import BlockMask
except (
    ImportError
):  # pragma: no cover - older torch builds may not expose FlexAttention
    BlockMask = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class AttentionProfileSpec:
    """Declarative description of a reusable attention visibility profile."""

    name: str
    family: str
    backend: str


@dataclass
class PreparedAttentionProfile:
    """Backend-ready attention visibility state.

    The profile can carry either dense boolean masks, FlexAttention block masks,
    or both. Callers choose the best representation for the current runtime.
    """

    spec: AttentionProfileSpec
    self_attention_mask: torch.Tensor | None = None
    cross_attention_mask: torch.Tensor | None = None
    self_attention_block_mask: BlockMask | None = None
    cross_attention_block_mask: BlockMask | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


VIDEO_THEN_ACTION_COUPLING = "video_then_action"


JOINT_COUPLING = "joint"


ACTION_THEN_VIDEO_COUPLING = "action_then_video"


DECOUPLED_SAME_STEP_COUPLING = "decoupled_same_step"


VIDEO_NOISY_TO_ACTION_COUPLING = "video_noisy_to_action"


ACTION_NOISY_TO_VIDEO_COUPLING = "action_noisy_to_video"


HISTORY_STREAM_VISIBILITY_FULL = "full"


HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY = "video_queries_video_only"


HISTORY_STREAM_VISIBILITY_VIDEO_ONLY = "video_only"


CONDITIONAL_HISTORY_POLICY_NONE = "none"


CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY = (
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY
)


_HISTORY_STREAM_VISIBILITY_VALUES = {
    HISTORY_STREAM_VISIBILITY_FULL,
    HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY,
    HISTORY_STREAM_VISIBILITY_VIDEO_ONLY,
}


_CONDITIONAL_HISTORY_POLICY_VALUES = {
    CONDITIONAL_HISTORY_POLICY_NONE,
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
}


_CHUNKED_EXACT_PROFILE_BY_COUPLING: dict[str, str] = {
    VIDEO_THEN_ACTION_COUPLING: "chunked_temporal_exact",
    JOINT_COUPLING: "chunked_temporal_exact_joint",
    ACTION_THEN_VIDEO_COUPLING: "chunked_temporal_exact_action_then_video",
    DECOUPLED_SAME_STEP_COUPLING: "chunked_temporal_exact_decoupled_same_step",
    VIDEO_NOISY_TO_ACTION_COUPLING: "chunked_temporal_exact_video_noisy_to_action",
    ACTION_NOISY_TO_VIDEO_COUPLING: "chunked_temporal_exact_action_noisy_to_video",
}


_CHUNKED_EXACT_COUPLING_BY_PROFILE = {
    profile_name: coupling
    for coupling, profile_name in _CHUNKED_EXACT_PROFILE_BY_COUPLING.items()
}


_ATTENTION_PROFILE_ALIASES: dict[str, str] = {
    "chunked_temporal_exact": "chunked_temporal_exact",
    "chunked_temporal_exact_joint": "chunked_temporal_exact_joint",
    "chunked_temporal_exact_action_then_video": "chunked_temporal_exact_action_then_video",
    "chunked_temporal_exact_decoupled_same_step": "chunked_temporal_exact_decoupled_same_step",
    "chunked_temporal_exact_video_noisy_to_action": "chunked_temporal_exact_video_noisy_to_action",
    "chunked_temporal_exact_action_noisy_to_video": "chunked_temporal_exact_action_noisy_to_video",
    "lingbot_chunked_exact": "chunked_temporal_exact",
    "none": "none",
}


def normalize_attention_profile_name(name: str | None) -> str | None:
    if name is None:
        return None
    try:
        return _ATTENTION_PROFILE_ALIASES[name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported attention profile {name!r}. Expected one of {tuple(_ATTENTION_PROFILE_ALIASES)}."
        ) from exc


def normalize_chunked_temporal_exact_coupling(coupling: str | None) -> str:
    """Normalize packed video/action current-block coupling names."""

    if coupling is None:
        return VIDEO_THEN_ACTION_COUPLING
    value = str(getattr(coupling, "value", coupling))
    if value in _CHUNKED_EXACT_PROFILE_BY_COUPLING:
        return value
    try:
        normalized_profile = normalize_attention_profile_name(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported exact current-block coupling {coupling!r}. "
            f"Expected one of {tuple(_CHUNKED_EXACT_PROFILE_BY_COUPLING)}."
        ) from exc
    if normalized_profile in _CHUNKED_EXACT_COUPLING_BY_PROFILE:
        return _CHUNKED_EXACT_COUPLING_BY_PROFILE[normalized_profile]
    raise ValueError(
        f"Unsupported exact current-block coupling {coupling!r}. "
        f"Expected one of {tuple(_CHUNKED_EXACT_PROFILE_BY_COUPLING)}."
    )


def normalize_history_stream_visibility(
    visibility: HistoryStreamVisibility | str | None,
) -> str:
    """Normalize clean-history visibility using the public policy default."""

    if visibility is None:
        return HISTORY_STREAM_VISIBILITY_VIDEO_ONLY
    value = str(getattr(visibility, "value", visibility))
    if value in _HISTORY_STREAM_VISIBILITY_VALUES:
        return value
    raise ValueError(
        f"Unsupported history stream visibility {visibility!r}. "
        f"Expected one of {tuple(sorted(_HISTORY_STREAM_VISIBILITY_VALUES))}."
    )


def normalize_conditional_history_policy(policy: str | None) -> str:
    """Normalize the optional conditional IDM/FDM historical K/V policy."""

    if policy is None:
        return CONDITIONAL_HISTORY_POLICY_NONE
    value = str(getattr(policy, "value", policy))
    if value in _CONDITIONAL_HISTORY_POLICY_VALUES:
        return value
    raise ValueError(
        f"Unsupported conditional history policy {policy!r}. "
        f"Expected one of {tuple(sorted(_CONDITIONAL_HISTORY_POLICY_VALUES))}."
    )


def chunked_temporal_exact_profile_name_for_coupling(coupling: str | None) -> str:
    """Return the attention-profile name for an exact parallel-stream coupling mode."""

    return _CHUNKED_EXACT_PROFILE_BY_COUPLING[
        normalize_chunked_temporal_exact_coupling(coupling)
    ]


def chunked_temporal_exact_coupling_from_profile_name(name: str) -> str:
    """Return the exact parallel-stream coupling represented by an attention-profile name."""

    normalized_profile = normalize_attention_profile_name(name)
    if normalized_profile not in _CHUNKED_EXACT_COUPLING_BY_PROFILE:
        raise ValueError(f"Attention profile {name!r} is not a chunked exact profile.")
    return _CHUNKED_EXACT_COUPLING_BY_PROFILE[normalized_profile]


__all__ = [
    "ACTION_NOISY_TO_VIDEO_COUPLING",
    "ACTION_THEN_VIDEO_COUPLING",
    "CONDITIONAL_HISTORY_POLICY_NONE",
    "CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY",
    "DECOUPLED_SAME_STEP_COUPLING",
    "HISTORY_STREAM_VISIBILITY_FULL",
    "HISTORY_STREAM_VISIBILITY_VIDEO_ONLY",
    "HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY",
    "JOINT_COUPLING",
    "VIDEO_NOISY_TO_ACTION_COUPLING",
    "VIDEO_THEN_ACTION_COUPLING",
    "AttentionProfileSpec",
    "PreparedAttentionProfile",
    "chunked_temporal_exact_coupling_from_profile_name",
    "chunked_temporal_exact_profile_name_for_coupling",
    "normalize_attention_profile_name",
    "normalize_chunked_temporal_exact_coupling",
    "normalize_conditional_history_policy",
    "normalize_history_stream_visibility",
]
