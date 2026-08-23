from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import torch

from open_wam.configs.enums import DynamicsObjective, ProprioContextMode
from open_wam.configs.policy_contracts import PolicyConditioningRequirements
from open_wam.models.common import RolloutCursor
from open_wam.models.common.dynamics_contracts import DynamicsRolloutRequest

_DecoderArtifactT = TypeVar("_DecoderArtifactT")


class PolicyGenerationActionOrigin(str, Enum):
    """Index origin used when a policy starts a generated action sequence."""

    AFTER_OBSERVATION_WINDOW = "after_observation_window"
    ZERO = "zero"


class PolicyObservationWindowSessionPolicy(str, Enum):
    """How recurrent state changes when the observed window advances."""

    REUSE = "reuse"
    REBUILD_FROM_OBSERVATION_WINDOW = "rebuild_from_observation_window"


class PolicyVisualStage(str, Enum):
    """Visual outputs a policy may request from the shared pipeline."""

    FRONTEND = "frontend"
    CORE = "core"


@dataclass(frozen=True)
class PolicyRolloutContract:
    """Variant-owned lifecycle semantics consumed by generic rollout code."""

    generation_action_origin: PolicyGenerationActionOrigin = (
        PolicyGenerationActionOrigin.AFTER_OBSERVATION_WINDOW
    )
    observation_window_session_policy: PolicyObservationWindowSessionPolicy = (
        PolicyObservationWindowSessionPolicy.REUSE
    )


@dataclass(frozen=True)
class PolicyStateDictOverlay:
    """Additional module state projected into a shared exported state dict."""

    module: torch.nn.Module
    map_key: Callable[[str], str | None]
    exclusive_target_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyModuleTopology:
    """Variant-owned module placement consumed by generic training services.

    A policy may transfer shared-backbone or action-side blocks into a packed
    owner without exposing that implementation detail to component selection,
    FSDP, or checkpoint export.
    """

    visual_runtime_modules: tuple[torch.nn.Module, ...]
    action_expert_modules: tuple[torch.nn.Module, ...] = ()
    fsdp_block_stacks: tuple[torch.nn.Module, ...] = ()
    fsdp_atomic_modules: tuple[torch.nn.Module, ...] = ()
    runtime_backbone_state_overlays: tuple[PolicyStateDictOverlay, ...] = ()


@dataclass(frozen=True)
class PolicyPipelineRequirements:
    """Policy-declared geometry and shared conditioning requirements.

    The composition factory consumes this contract without knowing which policy
    architecture produced it. Action geometry is expressed in model space and
    can differ from dataset geometry when the policy owns an action adapter.
    """

    action_dim: int
    action_horizon: int
    state_dim: int
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    dynamics_mode_context_enabled: bool = False
    source_action_channel_ids: tuple[int, ...] = ()
    accepted_source_action_shapes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proprio_context_mode",
            ProprioContextMode(self.proprio_context_mode),
        )
        if int(self.action_dim) <= 0:
            raise ValueError(
                "Pipeline requirements need a positive model action dimension, "
                f"got {self.action_dim}."
            )
        if int(self.action_horizon) < 0:
            raise ValueError(
                "Pipeline requirements need a non-negative model action horizon, "
                f"got {self.action_horizon}."
            )
        if int(self.state_dim) < 0:
            raise ValueError(
                "Pipeline requirements need a non-negative state dimension, "
                f"got {self.state_dim}."
            )
        if (
            self.proprio_context_mode != ProprioContextMode.NONE
            and int(self.state_dim) == 0
        ):
            raise ValueError(
                "Proprio conditioning requires a positive state dimension."
            )
        source_action_channel_ids = tuple(
            int(index) for index in self.source_action_channel_ids
        )
        if any(index < 0 for index in source_action_channel_ids):
            raise ValueError("Source action channel ids must be non-negative.")
        if len(set(source_action_channel_ids)) != len(source_action_channel_ids):
            raise ValueError("Source action channel ids must be unique.")
        if source_action_channel_ids and max(source_action_channel_ids) >= int(
            self.action_dim
        ):
            raise ValueError(
                "Source action channel ids must index the model action space, "
                f"got max_index={max(source_action_channel_ids)} and "
                f"action_dim={self.action_dim}."
            )
        object.__setattr__(
            self,
            "source_action_channel_ids",
            source_action_channel_ids,
        )
        accepted_source_action_shapes = tuple(
            (int(action_dim), int(action_horizon))
            for action_dim, action_horizon in self.accepted_source_action_shapes
        )
        if any(
            action_dim <= 0 or action_horizon < 0
            for action_dim, action_horizon in accepted_source_action_shapes
        ):
            raise ValueError(
                "Accepted source action shapes require a positive action dimension "
                "and non-negative horizon."
            )
        if len(set(accepted_source_action_shapes)) != len(
            accepted_source_action_shapes
        ):
            raise ValueError("Accepted source action shapes must be unique.")
        object.__setattr__(
            self,
            "accepted_source_action_shapes",
            accepted_source_action_shapes,
        )

    def validate_source_action_shape(
        self,
        *,
        action_dim: int,
        action_horizon: int,
    ) -> None:
        """Require dataset actions to match a policy-supported input shape."""

        if not self.accepted_source_action_shapes:
            return
        actual = (int(action_dim), int(action_horizon))
        if actual in self.accepted_source_action_shapes:
            return
        supported = ", ".join(
            f"(action_dim={source_dim}, action_horizon={source_horizon})"
            for source_dim, source_horizon in self.accepted_source_action_shapes
        )
        raise ValueError(
            "Dataset action geometry is not accepted by the policy input adapter: "
            f"got action_dim={actual[0]}, action_horizon={actual[1]}; "
            f"supported shapes: {supported}."
        )

    def validate_action_decoder(
        self,
        *,
        action_dim: int,
        action_horizon: int,
    ) -> None:
        """Require the decoder to consume the policy's model-space geometry."""

        actual = (int(action_dim), int(action_horizon))
        expected = (int(self.action_dim), int(self.action_horizon))
        if actual == expected:
            return
        raise ValueError(
            "Policy and action decoder model-space geometry do not match: "
            f"policy requires action_dim={expected[0]}, action_horizon={expected[1]}; "
            f"decoder declares action_dim={actual[0]}, action_horizon={actual[1]}."
        )

    def validate_visual_tower(
        self,
        *,
        action_dim: int | None,
        state_dim: int | None,
    ) -> None:
        """Require shared tower adapters to use the policy's model geometry."""

        actual = (action_dim, state_dim)
        expected = (int(self.action_dim), int(self.state_dim))
        if actual == expected:
            return
        raise ValueError(
            "Policy and visual tower model-space geometry do not match: "
            f"policy requires action_dim={expected[0]}, state_dim={expected[1]}; "
            f"tower declares action_dim={actual[0]}, state_dim={actual[1]}."
        )

    def validate_conditioning(
        self,
        configured: PolicyConditioningRequirements,
    ) -> None:
        """Require module-time needs to match pre-allocation config needs."""

        expected = PolicyConditioningRequirements(
            proprio_context_mode=self.proprio_context_mode,
            dynamics_mode_context_enabled=self.dynamics_mode_context_enabled,
        )
        if configured == expected:
            return
        raise ValueError(
            "Policy runtime conditioning requirements do not match the policy "
            "configuration used to assemble the visual tower: "
            f"config={configured!r}, runtime={expected!r}. Declare shared "
            "conditioning on the policy config before pipeline construction."
        )


@dataclass(frozen=True)
class DecoderArtifactEnvelope:
    """Typed handoff from one policy architecture to its action decoder.

    ``contract`` is an open extension identifier, while ``payload`` is owned by
    the policy/decoder pair that declares it. This keeps architecture-specific
    tensors out of the shared pipeline and replaces undocumented ``aux`` keys.
    """

    contract: str
    payload: Any
    dynamics_objective: DynamicsObjective | None = None

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
    dynamics: DynamicsRolloutRequest | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferOutput:
    """Inference-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    next_state: PolicyInferState
    decoder_artifacts: DecoderArtifactEnvelope | None = None
    aux: dict[str, Any] = field(default_factory=dict)
