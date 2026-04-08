from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common import PreparedAttentionProfile, RegisterSequenceLayout
from open_wam.models.video_backbone.contracts import (
    CacheState,
    CacheUpdateMetadata,
    ChunkMetadata,
    ConditioningState,
    TokenGridMetadata,
)


@dataclass(frozen=True)
class VisualReadoutRequest:
    """Opt-in intermediate readout capture requested from the visual core."""

    capture_layer_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class VisualIntermediateReadout:
    """One captured intermediate visual-core layer output."""

    layer_index: int
    tokens: torch.Tensor
    token_layout: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedFeatureLayout:
    """Layout metadata for decoded visual features."""

    kind: str
    num_frames: int
    tokens_per_frame: int
    hidden_size: int


@dataclass
class VisualFrontendOutput:
    """Outputs produced by the fixed visual frontend."""

    canonical_video: torch.Tensor
    video_latents: torch.Tensor
    video_tokens: torch.Tensor
    input_source: str
    token_grid: TokenGridMetadata
    chunk: ChunkMetadata
    conditioning: ConditioningState


@dataclass
class VisualSequenceMetadata:
    """Structured runtime metadata for packed visual-core calls.

    Method 2 needs the core to know which part of the packed sequence
    corresponds to clean-prefix video tokens versus noisy video/action/state
    registers. Keeping this explicit at the contract layer lets the upcoming
    DreamZero-alignment rewrite move these semantics into the core itself.
    """

    teacher_forcing: bool = False
    clean_prefix_tokens: int = 0
    noisy_video_tokens: int = 0
    action_register_tokens: int = 0
    state_register_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisterSequenceSemantics:
    """Explicit structured-register semantics carried into the core.

    This keeps variant-specific register meaning out of ad hoc metadata dicts
    and lets the structured-core path say exactly what kind of packed sequence
    it is materializing.
    """

    sequence_family: str
    attention_style: str
    teacher_forcing_layout: str
    timestep_layout: str
    video_sequence_tokens: int
    action_register_tokens: int
    state_register_tokens: int
    current_start_frame: int
    teacher_forcing: bool
    structured_block_mode: str = "none"
    structured_time_layout: str = "generic"
    structured_frequency_mode: str = "shared_rotary_only"
    structured_teacher_forcing_layout: str = "none"
    structured_attention_kernel: str = "mask_only"
    structured_cache_kernel: str = "prefix_mask_only"


@dataclass(frozen=True)
class StructuredFrequencyBundle:
    """Explicit per-stream frequency inputs for structured block runtimes."""

    layout: str
    clean_prefix_grid_ids: torch.Tensor | None = None
    video_grid_ids: torch.Tensor | None = None
    action_grid_ids: torch.Tensor | None = None
    state_grid_ids: torch.Tensor | None = None
    shared_grid_ids: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredBlockSemantics:
    """Explicit block-facing structured runtime semantics."""

    mode: str
    teacher_forcing_enabled: bool
    clean_prefix_span: tuple[int, int]
    video_span: tuple[int, int]
    action_span: tuple[int, int]
    state_span: tuple[int, int]
    clean_prefix_length: int
    video_token_length: int
    action_register_length: int
    state_register_length: int
    current_start_frame: int
    observed_prefix_frames: int
    time_layout: str
    position_layout: str
    frequency_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredAttentionContext:
    """Unified block-facing context for structured attention runtimes."""

    mode: str
    teacher_forcing_enabled: bool
    clean_prefix_length: int
    video_token_length: int
    action_register_length: int
    state_register_length: int
    current_start_frame: int
    observed_prefix_frames: int
    num_frame_per_block: int
    num_action_per_block: int
    num_state_per_block: int
    num_video_blocks: int
    num_action_blocks: int
    num_state_blocks: int
    tokens_per_frame: int
    tokens_per_video_block: int
    frequency_mode: str
    attention_kernel: str = "mask_only"
    cache_kernel: str = "prefix_mask_only"
    rollout_phase: str = "teacher_forcing"
    action_state_index: int = 0
    cached_video_tokens: int = 0
    cached_segment_lengths: tuple[int, ...] = field(default_factory=tuple)
    clean_prefix_grid_ids: torch.Tensor | None = None
    video_grid_ids: torch.Tensor | None = None
    action_grid_ids: torch.Tensor | None = None
    state_grid_ids: torch.Tensor | None = None
    clean_prefix_freqs: torch.Tensor | None = None
    video_freqs: torch.Tensor | None = None
    action_freqs: torch.Tensor | None = None
    state_freqs: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisterSequenceComponents:
    """Structured method-2 sequence components materialized inside the core."""

    layout: RegisterSequenceLayout
    token_grid: TokenGridMetadata
    clean_video_prefix_tokens: torch.Tensor | None
    noisy_video_tokens: torch.Tensor
    action_register_tokens: torch.Tensor
    state_register_tokens: torch.Tensor
    current_start_frame: int
    video_timesteps: torch.Tensor
    action_timesteps: torch.Tensor
    state_timesteps: torch.Tensor
    semantics: RegisterSequenceSemantics


@dataclass
class VisualCoreInput:
    """Generic packed-sequence input accepted by the shared visual core."""

    tokens: torch.Tensor | None
    token_layout: Any | None = None
    position_context: torch.Tensor | None = None
    timestep_context: torch.Tensor | None = None
    grid_ids: torch.Tensor | None = None
    timestep_values: torch.Tensor | None = None
    stream_ids: torch.Tensor | None = None
    text_context: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    attention_profile: PreparedAttentionProfile | None = None
    cache_state: CacheState | None = None
    cache_update_metadata: CacheUpdateMetadata | None = None
    conditioning: ConditioningState | None = None
    readout_request: VisualReadoutRequest | None = None
    sequence_metadata: VisualSequenceMetadata | None = None
    register_components: RegisterSequenceComponents | None = None
    structured_block_semantics: StructuredBlockSemantics | None = None
    structured_frequency_bundle: StructuredFrequencyBundle | None = None
    structured_attention_context: StructuredAttentionContext | None = None


@dataclass
class VisualCoreOutput:
    """Outputs returned by the shared visual core."""

    tokens: torch.Tensor
    token_layout: Any | None
    cache_state: CacheState
    intermediate_readouts: tuple[VisualIntermediateReadout, ...] = field(default_factory=tuple)
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualDecodeOutput:
    """Decoded visual features exposed to post-decoded policies."""

    decoded_features: torch.Tensor
    feature_layout: DecodedFeatureLayout
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualStageOutputs:
    """Stageful outputs computed for one forward pass."""

    frontend: VisualFrontendOutput
    core: VisualCoreOutput | None = None
    decode: VisualDecodeOutput | None = None
