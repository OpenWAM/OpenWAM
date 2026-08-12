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
class PolicyTrainOutput:
    """Train-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    metrics: dict[str, torch.Tensor]
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
    decoder_artifacts: DecoderArtifactEnvelope | None = None
    aux: dict[str, Any] = field(default_factory=dict)
