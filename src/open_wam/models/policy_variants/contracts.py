from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from open_wam.models.common import RolloutCursor

_DecoderArtifactT = TypeVar("_DecoderArtifactT")


@dataclass(frozen=True)
class DecoderArtifactEnvelope:
    """Typed handoff from one policy architecture to its action decoder.

    ``contract`` is an open extension identifier, while ``payload`` is owned by
    the policy/decoder pair that declares it. This keeps architecture-specific
    tensors out of the shared pipeline and replaces undocumented ``aux`` keys.
    """

    contract: str
    payload: Any

    def require(
        self,
        *,
        contract: str,
        payload_type: type[_DecoderArtifactT],
    ) -> _DecoderArtifactT:
        if self.contract != contract:
            raise ValueError(
                f"Decoder artifact contract {self.contract!r} does not match "
                f"required contract {contract!r}."
            )
        if not isinstance(self.payload, payload_type):
            raise TypeError(
                f"Decoder artifact contract {contract!r} requires payload "
                f"{payload_type.__name__}, got {type(self.payload).__name__}."
            )
        return self.payload


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

    Sequence-aware decoders can consume this additive contract. Simpler
    decoders can ignore it and continue consuming `policy_features` only.

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
    decoder_artifacts: DecoderArtifactEnvelope | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferState:
    """Per-variant inference state."""

    step_index: int = 0
    cursor: RolloutCursor = field(default_factory=RolloutCursor)
    cache: Any = field(default_factory=dict)
    variant_state: Any | None = None
    decoder_state: Any | None = None


@dataclass(frozen=True)
class PolicyObservedHistory:
    """Canonical observations committed after executing one policy chunk.

    ``video_latents`` and ``proprio_history`` describe only the newly observed
    window, not the policy's full recurrent cache. ``action_history`` contains
    the actions actually executed for this commit; a policy decides how much
    speculative action history they replace. ``observation_frame_count`` is
    the raw environment-frame count and can differ from latent time.
    """

    video_latents: torch.Tensor
    observation_frame_count: int
    action_history: torch.Tensor | None = None
    proprio_history: torch.Tensor | None = None
    inference_window_size: int | None = None
    rollout_frame_chunk_size: int | None = None


@dataclass(frozen=True)
class PolicyObservedHistoryOutput:
    """Policy state and diagnostics after reconciling real observations."""

    next_state: PolicyInferState | None
    debug: dict[str, Any] = field(default_factory=dict)


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
    decoder_artifacts: DecoderArtifactEnvelope | None = None
    aux: dict[str, Any] = field(default_factory=dict)
