from __future__ import annotations

from dataclasses import dataclass, field

from .visual_readout import VisualReadoutConfig
from .enums import (
    ActionNormMethod,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    AttachSite,
    DecodeFeatureMode,
    ParallelCacheMode,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    PolicyVariantName,
    PoolingMode,
    RegisterLayout,
    RegisterMaskMode,
    StreamEncoderType,
    StreamInputAdapterFamily,
    StreamOutputHeadFamily,
    StructuredAttentionKernel,
    StructuredBlockMode,
    StructuredCacheKernel,
    StructuredFrequencyMode,
    StructuredTeacherForcingLayout,
    StructuredTimeLayout,
    TemporalPositionMode,
    TemporalProjection,
    VisualStateSource,
    coerce_fields,
)


@dataclass(frozen=True)
class PolicyVariantConfig:
    """Base config shared by all policy variants."""

    name: PolicyVariantName
    hidden_size: int
    attach_site: AttachSite
    visual_readout: VisualReadoutConfig | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "name": PolicyVariantName,
                "attach_site": AttachSite,
            },
        )


@dataclass(frozen=True)
class PostLatentPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.POST_LATENT
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    pooling_mode: PoolingMode = PoolingMode.PER_FRAME_MEAN
    query_count: int = 0
    temporal_projection: TemporalProjection = TemporalProjection.INTERPOLATE
    use_state_projection: bool = True
    compatibility_mode: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Post-latent policy now requires `attach_site = post_visual_core` so all variants share "
                f"the same visual backbone path, got attach_site={self.attach_site!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "pooling_mode": PoolingMode,
                "temporal_projection": TemporalProjection,
            },
        )


@dataclass(frozen=True)
class PostDecodedPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.POST_DECODED
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_DECODE
    decode_feature_mode: DecodeFeatureMode = DecodeFeatureMode.FRAME_TOKEN_SEQUENCE
    pooling_mode: PoolingMode = PoolingMode.PER_FRAME_MEAN
    temporal_projection: TemporalProjection = TemporalProjection.INTERPOLATE
    use_state_projection: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_DECODE:
            raise ValueError(
                "Post-decoded policy requires `attach_site = post_visual_decode`, "
                f"got attach_site={self.attach_site!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "decode_feature_mode": DecodeFeatureMode,
                "pooling_mode": PoolingMode,
                "temporal_projection": TemporalProjection,
            },
        )


@dataclass(frozen=True)
class VideoSequencePolicyConfig(PolicyVariantConfig):
    """Sequence-preserving post-core policy family for future method-3 decoders.

    The variant itself stays intentionally lightweight: it owns the attachment
    point and packages rich decoder-facing sequence context, while future
    sequence decoders own temporal compression, goal/state conditioning, and
    action-generation algorithms.
    """

    name: PolicyVariantName = PolicyVariantName.VIDEO_SEQUENCE_POLICY
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    temporal_projection: TemporalProjection = TemporalProjection.INTERPOLATE
    visual_state_source: VisualStateSource = VisualStateSource.DENOISED_VIDEO_TOKENS
    visual_denoise_ratio: float = 1.0
    use_state_context: bool = True
    use_goal_context: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Video-sequence policy requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "temporal_projection": TemporalProjection,
                "visual_state_source": VisualStateSource,
            },
        )
        if not (0.0 < float(self.visual_denoise_ratio) <= 1.0):
            raise ValueError(
                "Video-sequence policy requires `0 < visual_denoise_ratio <= 1`, "
                f"got visual_denoise_ratio={self.visual_denoise_ratio!r}."
            )


@dataclass(frozen=True)
class CausalVideoPredictionPolicyConfig(PolicyVariantConfig):
    """Standalone causal video-only pretraining variant."""

    name: PolicyVariantName = PolicyVariantName.CAUSAL_VIDEO_PREDICTION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Causal video prediction requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )


@dataclass(frozen=True)
class RegisterAttachedPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.REGISTER_ATTACHED
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.WITHIN_VISUAL_CORE
    num_frame_per_block: int = 1
    num_action_per_block: int = 1
    num_state_per_block: int = 1
    max_chunk_size: int = 1
    register_layout: RegisterLayout = RegisterLayout.ACTION_THEN_STATE
    mask_mode: RegisterMaskMode = RegisterMaskMode.DREAMZERO_BLOCKWISE
    use_state_encoder: bool = True
    action_encoder_type: StreamEncoderType = StreamEncoderType.MLP
    state_encoder_type: StreamEncoderType = StreamEncoderType.MLP
    couple_action_to_video_blocks: bool = True
    structured_block_mode: StructuredBlockMode = StructuredBlockMode.REGISTER_EXPLICIT
    structured_time_layout: StructuredTimeLayout = StructuredTimeLayout.VIDEO_ACTION_STATE
    structured_frequency_mode: StructuredFrequencyMode = StructuredFrequencyMode.STREAM_LOCAL
    structured_teacher_forcing_layout: StructuredTeacherForcingLayout = StructuredTeacherForcingLayout.CLEAN_PREFIX
    structured_attention_kernel: StructuredAttentionKernel = StructuredAttentionKernel.BRANCHWISE_EXPLICIT
    structured_cache_kernel: StructuredCacheKernel = StructuredCacheKernel.BRANCHWISE_ROLLOUT_EXPLICIT
    stream_input_adapter_family: StreamInputAdapterFamily = StreamInputAdapterFamily.STRUCTURED_REGISTER_STREAMS
    stream_output_head_family: StreamOutputHeadFamily = StreamOutputHeadFamily.STRUCTURED_JOINT_FLOW

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "register_layout": RegisterLayout,
                "mask_mode": RegisterMaskMode,
                "action_encoder_type": StreamEncoderType,
                "state_encoder_type": StreamEncoderType,
                "structured_block_mode": StructuredBlockMode,
                "structured_time_layout": StructuredTimeLayout,
                "structured_frequency_mode": StructuredFrequencyMode,
                "structured_teacher_forcing_layout": StructuredTeacherForcingLayout,
                "structured_attention_kernel": StructuredAttentionKernel,
                "structured_cache_kernel": StructuredCacheKernel,
                "stream_input_adapter_family": StreamInputAdapterFamily,
                "stream_output_head_family": StreamOutputHeadFamily,
            },
        )


@dataclass(frozen=True)
class ParallelStreamPolicyConfig(PolicyVariantConfig):
    name: PolicyVariantName = PolicyVariantName.PARALLEL_STREAM
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.WITHIN_VISUAL_CORE
    runtime_mode: ParallelRuntimeMode = ParallelRuntimeMode.LINGBOT_EXACT
    reference_profile: str | None = None
    frame_chunk_size: int = 2
    action_per_frame: int = 1
    attn_window: int = 8
    sequence_order: tuple[ParallelSequenceComponent, ...] = field(
        default_factory=lambda: (
            ParallelSequenceComponent.VIDEO_NOISY,
            ParallelSequenceComponent.VIDEO_CONDITION,
            ParallelSequenceComponent.ACTION_NOISY,
            ParallelSequenceComponent.ACTION_CONDITION,
        )
    )
    mask_mode: ParallelMaskMode = ParallelMaskMode.LINGBOT_CHUNKED
    cache_mode: ParallelCacheMode = ParallelCacheMode.METADATA_ONLY
    noisy_video_condition_prob: float = 0.5
    video_condition_on_action: bool = False
    video_action_condition_source: ParallelActionConditionSource = ParallelActionConditionSource.NOISY_ACTION
    video_action_attention_scope: ParallelActionAttentionScope = ParallelActionAttentionScope.BLOCK_LOCAL
    couple_action_to_video_timesteps: bool = True
    temporal_position_mode: TemporalPositionMode = TemporalPositionMode.GLOBAL_SHIFTED
    used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    inverse_used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    action_norm_method: ActionNormMethod = ActionNormMethod.NONE
    norm_q01: tuple[float, ...] = field(default_factory=tuple)
    norm_q99: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "runtime_mode": ParallelRuntimeMode,
                "mask_mode": ParallelMaskMode,
                "cache_mode": ParallelCacheMode,
                "video_action_condition_source": ParallelActionConditionSource,
                "video_action_attention_scope": ParallelActionAttentionScope,
                "temporal_position_mode": TemporalPositionMode,
                "action_norm_method": ActionNormMethod,
            },
            enum_tuple_fields={"sequence_order": ParallelSequenceComponent},
        )
