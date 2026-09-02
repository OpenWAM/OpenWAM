from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import torch

from open_wam.contracts import VideoLatentSpaceIdentity
from open_wam.configs.enums import (
    DynamicsObjective,
    ProprioContextMode,
    TextConditioningMode,
)
from open_wam.configs.policy_contracts import PolicyConditioningRequirements
from open_wam.models.common import RolloutCursor
from open_wam.models.common.dynamics_contracts import DynamicsRolloutRequest
from open_wam.models.visual_tower.contracts import VisualComponentTopology

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


class PolicyOutputModality(str, Enum):
    """Model products requested from one policy inference step."""

    VIDEO = "video"
    ACTION = "action"


class PolicyRecurrentHistoryPolicy(str, Enum):
    """How a recurrent policy replaces speculative history after execution.

    Online composition executes actions inferred from a generated video and then
    receives real observations. A producer must explicitly declare whether its
    next inference call replaces the speculative video itself, or whether the
    rollout driver must reconcile the executed observations into policy state.
    ``UNSUPPORTED`` keeps ordinary inference available without claiming that a
    policy is safe to use as either stage of recurrent composition.
    """

    UNSUPPORTED = "unsupported"
    NEXT_OBSERVATION = "next_observation"
    EXPLICIT_RECONCILIATION = "explicit_reconciliation"


@dataclass(frozen=True)
class PolicyInferenceOutputRequest:
    """Architecture-independent selection of inference products.

    Policies may reject a selection when their coupling semantics require an
    omitted modality to produce the requested one. A missing request on
    ``PolicyInferContext`` preserves each policy's normal output contract.
    """

    modalities: frozenset[PolicyOutputModality]

    def __post_init__(self) -> None:
        modalities = frozenset(
            PolicyOutputModality(modality) for modality in self.modalities
        )
        if not modalities:
            raise ValueError(
                "Policy inference must request at least one output modality."
            )
        object.__setattr__(self, "modalities", modalities)

    @classmethod
    def full(cls) -> PolicyInferenceOutputRequest:
        return cls(frozenset(PolicyOutputModality))

    @classmethod
    def video_only(cls) -> PolicyInferenceOutputRequest:
        return cls(frozenset({PolicyOutputModality.VIDEO}))

    @classmethod
    def action_only(cls) -> PolicyInferenceOutputRequest:
        return cls(frozenset({PolicyOutputModality.ACTION}))

    def requests(self, modality: PolicyOutputModality) -> bool:
        return PolicyOutputModality(modality) in self.modalities


@dataclass(frozen=True)
class PolicyInferenceCapabilities:
    """Outputs a policy produces natively and can select independently.

    ``native_modalities`` describes a normal inference call with no selective
    request. ``selective_requests`` lists exact subsets that the policy can
    produce without running the omitted output stages. A composition may still
    consume a subset of the native output when no selective route exists.
    """

    native_modalities: frozenset[PolicyOutputModality]
    selective_requests: tuple[PolicyInferenceOutputRequest, ...] = ()
    recurrent_history_policy: PolicyRecurrentHistoryPolicy = (
        PolicyRecurrentHistoryPolicy.UNSUPPORTED
    )

    def __post_init__(self) -> None:
        native = frozenset(
            PolicyOutputModality(modality) for modality in self.native_modalities
        )
        if not native:
            raise ValueError("A policy must declare at least one native output modality.")
        selective = tuple(self.selective_requests)
        seen_selective: set[frozenset[PolicyOutputModality]] = set()
        for request in selective:
            if not request.modalities.issubset(native):
                raise ValueError(
                    "Selective policy outputs must be a subset of native outputs; "
                    f"native={sorted(item.value for item in native)}, "
                    f"requested={sorted(item.value for item in request.modalities)}."
                )
            if request.modalities == native:
                raise ValueError(
                    "Selective policy outputs must be a strict subset of native "
                    "outputs; omit a request when normal inference already emits "
                    f"{sorted(item.value for item in native)}."
                )
            if request.modalities in seen_selective:
                raise ValueError(
                    "Selective policy output requests must be unique; duplicate="
                    f"{sorted(item.value for item in request.modalities)}."
                )
            seen_selective.add(request.modalities)
        object.__setattr__(self, "native_modalities", native)
        object.__setattr__(self, "selective_requests", selective)
        history_policy = PolicyRecurrentHistoryPolicy(
            self.recurrent_history_policy
        )
        object.__setattr__(
            self,
            "recurrent_history_policy",
            history_policy,
        )

    def request_for(
        self,
        required_modalities: frozenset[PolicyOutputModality],
    ) -> PolicyInferenceOutputRequest | None:
        """Resolve an efficient request while permitting native supersets.

        ``None`` means the normal policy output already contains every required
        modality. This is important for coupled policies whose video is valid
        but cannot be generated independently from their action stream.
        """

        required = frozenset(
            PolicyOutputModality(modality) for modality in required_modalities
        )
        if not required:
            raise ValueError("A composition must require at least one output modality.")
        if not required.issubset(self.native_modalities):
            raise ValueError(
                "Policy does not produce every required modality; "
                f"native={sorted(item.value for item in self.native_modalities)}, "
                f"required={sorted(item.value for item in required)}."
            )
        for request in self.selective_requests:
            if request.modalities == required:
                return request
        return None


@dataclass(frozen=True)
class PolicyTemporalSpan:
    """Half-open model-frame interval owned by one inference transaction."""

    start_frame: int
    frame_count: int

    def __post_init__(self) -> None:
        if int(self.start_frame) < 0:
            raise ValueError(
                "Policy temporal spans require a non-negative start frame, "
                f"got {self.start_frame}."
            )
        if int(self.frame_count) <= 0:
            raise ValueError(
                "Policy temporal spans require a positive frame count, "
                f"got {self.frame_count}."
            )

    @property
    def end_frame(self) -> int:
        return int(self.start_frame) + int(self.frame_count)

    def prefix(self, frame_count: int) -> PolicyTemporalSpan:
        """Return an executed prefix while preserving this span's origin."""

        resolved_count = int(frame_count)
        if resolved_count <= 0 or resolved_count > int(self.frame_count):
            raise ValueError(
                "Executed temporal prefixes must contain between one and the "
                "full speculative frame count; "
                f"executed={resolved_count}, speculative={self.frame_count}."
            )
        return PolicyTemporalSpan(
            start_frame=int(self.start_frame),
            frame_count=resolved_count,
        )


@dataclass(frozen=True)
class PolicyExecutionCommit:
    """Observed prefix committed after executing a speculative model span."""

    speculative_span: PolicyTemporalSpan
    executed_frame_count: int

    def __post_init__(self) -> None:
        self.speculative_span.prefix(int(self.executed_frame_count))

    @property
    def executed_span(self) -> PolicyTemporalSpan:
        return self.speculative_span.prefix(int(self.executed_frame_count))


@dataclass(frozen=True)
class PolicyGeneratedVideo:
    """Future-only latent video emitted for downstream composition.

    Conditioning or observed-prefix frames are deliberately excluded. This
    makes a causal video-only model, a staged VTA policy, and a jointly decoded
    policy expose the same downstream handoff semantics.
    """

    latents: torch.Tensor
    frame_start: int | None = None
    latent_space_identity: VideoLatentSpaceIdentity | None = None

    def __post_init__(self) -> None:
        if self.latents.ndim != 5 or int(self.latents.shape[2]) <= 0:
            raise ValueError(
                "Generated video must be a non-empty [B, C, T, H, W] tensor, "
                f"got {tuple(self.latents.shape)}."
            )
        if self.frame_start is not None and int(self.frame_start) < 0:
            raise ValueError(
                f"Generated-video frame_start must be non-negative, got {self.frame_start}."
            )


@dataclass(frozen=True)
class PolicyVideoGenerationRequest:
    """Geometry requested from a video-producing inference stage."""

    frame_count: int
    attention_window_size: int | None = None

    def __post_init__(self) -> None:
        if int(self.frame_count) <= 0:
            raise ValueError(
                f"Video generation frame_count must be positive, got {self.frame_count}."
            )
        if self.attention_window_size is not None and int(
            self.attention_window_size
        ) <= 0:
            raise ValueError(
                "Video generation attention_window_size must be positive when "
                f"provided, got {self.attention_window_size}."
            )


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
    visual_components: VisualComponentTopology = field(
        default_factory=VisualComponentTopology
    )
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
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT
    source_action_channel_ids: tuple[int, ...] = ()
    accepted_source_action_shapes: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proprio_context_mode",
            ProprioContextMode(self.proprio_context_mode),
        )
        object.__setattr__(
            self,
            "text_conditioning_mode",
            TextConditioningMode(self.text_conditioning_mode),
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
            text_conditioning_mode=self.text_conditioning_mode,
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
    """Structured policy training inputs independent from attachment site.

    ``source_text_context`` retains the positive conditioning tensor when the
    training executor selects an unconditional context for classifier-free
    dropout. Policies that validate their conditioning source can inspect it
    without changing the effective context consumed by the visual stack.
    """

    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    source_text_context: torch.Tensor | None = None


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
    ``rollout_frame_chunk_size`` identifies the speculative request being
    reconciled; it is independent of a model's fixed internal block geometry.
    """

    video_latents: torch.Tensor
    observation_frame_count: int
    action_history: torch.Tensor | None = None
    proprio_history: torch.Tensor | None = None
    inference_window_size: int | None = None
    rollout_frame_chunk_size: int | None = None
    execution_commit: PolicyExecutionCommit | None = None


@dataclass(frozen=True)
class PolicyObservedHistoryOutput:
    """Policy state and diagnostics after reconciling real observations."""

    next_state: PolicyInferState | None
    debug: dict[str, Any] = field(default_factory=dict)
    applied: bool = False


@dataclass
class PolicyInferContext:
    """Inputs required for one policy inference step."""

    state: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    dynamics: DynamicsRolloutRequest | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Keep extensions after `extra` so historical positional construction keeps
    # binding its fourth argument to the same field.
    output_request: PolicyInferenceOutputRequest | None = None
    video_generation: PolicyVideoGenerationRequest | None = None


@dataclass
class PolicyInferOutput:
    """Inference-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    next_state: PolicyInferState
    decoder_artifacts: DecoderArtifactEnvelope | None = None
    aux: dict[str, Any] = field(default_factory=dict)
    generated_video: PolicyGeneratedVideo | None = None
    generation_frame_start: int | None = None

    def __post_init__(self) -> None:
        if self.generation_frame_start is not None and int(
            self.generation_frame_start
        ) < 0:
            raise ValueError(
                "Policy inference generation_frame_start must be non-negative, "
                f"got {self.generation_frame_start}."
            )
        video_start = (
            None if self.generated_video is None else self.generated_video.frame_start
        )
        if (
            video_start is not None
            and self.generation_frame_start is not None
            and int(video_start) != int(self.generation_frame_start)
        ):
            raise ValueError(
                "Policy inference and generated-video temporal origins differ: "
                f"output={self.generation_frame_start}, video={video_start}."
            )
