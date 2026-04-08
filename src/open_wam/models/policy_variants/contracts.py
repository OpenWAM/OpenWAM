from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common import RolloutCursor


@dataclass
class PolicyTrainBatch:
    """Structured policy training inputs independent from attachment site."""

    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyPreparedInputs:
    """Prepared variant-specific inputs."""

    batch: PolicyTrainBatch
    variant_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoConditionWindowContext:
    """Typed decoder-facing local video-conditioning window."""

    local_window_tokens: torch.Tensor
    previous_context_tokens: torch.Tensor | None = None
    token_grid: Any | None = None
    source_stage: str | None = None
    input_space: str | None = None
    local_window_frames: int | None = None
    current_frame_index: int = 0
    current_action_index: int = 0
    action_chunk_anchor_mode: str | None = None
    observed_frame_count: int = 1
    previous_context_frames: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderSequenceContext:
    """Structured decoder-facing visual sequence context.

    This is the additive contract needed by future `video_sequence_policy`
    decoders. Existing simple decoders can ignore it and continue consuming
    `policy_features` only.

    `sequence_tokens` is intentionally flexible:
    - `[B, T, N, D]` for frame-token grids
    - `[B, T, D]` for already-collapsed frame sequences
    """

    sequence_tokens: torch.Tensor
    sequence_layout: dict[str, Any] = field(default_factory=dict)
    token_grid: Any | None = None
    frame_count: int | None = None
    source_stage: str | None = None
    state_sequence: torch.Tensor | None = None
    goal_features: torch.Tensor | None = None
    video_condition_window: VideoConditionWindowContext | None = None
    aux_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyTrainOutput:
    """Train-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    metrics: dict[str, torch.Tensor]
    decoder_sequence_context: DecoderSequenceContext | None = None
    owned_decoder_output: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferState:
    """Per-variant inference state."""

    step_index: int = 0
    cursor: RolloutCursor = field(default_factory=RolloutCursor)
    cache: Any = field(default_factory=dict)
    variant_state: Any | None = None
    decoder_state: Any | None = None


@dataclass
class PolicyInferContext:
    """Inputs required for one policy inference step."""

    state: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferOutput:
    """Inference-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    next_state: PolicyInferState
    decoder_sequence_context: DecoderSequenceContext | None = None
    owned_decoder_output: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)
