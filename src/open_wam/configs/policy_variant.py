from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    ActionChunkAnchorMode,
    ActionNormMethod,
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    AttachSite,
    DecodeFeatureMode,
    GeneralistTrainingParadigm,
    MoTConditionMode,
    MoTActionExpertInitMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    MoTRuntimeMode,
    ParallelCacheMode,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelSequenceComponent,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    ProprioContextMode,
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
    VideoConditionInputSpace,
    VideoConditionSource,
    VisualStateSource,
    coerce_fields,
)
from .variant_semantics import coerce_probability_map, default_video_action_conditioning_mode_probs
from .visual_readout import VisualReadoutConfig


def _default_joint_denoise_training_mode_probs(
    variant_profile: ParallelStreamVariantProfile,
) -> dict[JointDenoiseTrainingMode, float]:
    return default_video_action_conditioning_mode_probs(
        JointDenoiseTrainingMode,
        generalist=variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
    )


def _coerce_joint_denoise_training_mode_probs(
    raw_value: object,
    *,
    variant_profile: ParallelStreamVariantProfile | str,
) -> dict[JointDenoiseTrainingMode, float]:
    resolved_profile = ParallelStreamVariantProfile(variant_profile)
    if raw_value is None:
        return _default_joint_denoise_training_mode_probs(resolved_profile)
    return coerce_probability_map(
        raw_value,
        enum_cls=JointDenoiseTrainingMode,
        field_name="joint_denoise_training_mode_probs",
    )



def _coerce_mot_generalist_training_mode_probs(
    raw_value: object,
) -> dict[MoTGeneralistTrainingMode, float] | None:
    """Coerce an optional M5 generalist sampling distribution.

    ``None`` keeps the existing fixed ``current_block_coupling`` path. When a
    mapping is provided, missing modes default to 0 and probabilities are
    normalized to sum to one.
    """

    if raw_value is None:
        return None
    return coerce_probability_map(
        raw_value,
        enum_cls=MoTGeneralistTrainingMode,
        field_name="mot_generalist_training_mode_probs",
    )


@dataclass(frozen=True)
class PolicyVariantConfig:
    """Base config shared by all policy variants."""

    name: PolicyVariantName
    hidden_size: int
    attach_site: AttachSite

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
    video_condition_input_space: VideoConditionInputSpace = VideoConditionInputSpace.VIDEO_LATENT
    train_video_condition_source: VideoConditionSource = VideoConditionSource.LOCAL_WINDOW
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    local_video_window_frames: int = 4
    current_video_frame_index: int = 0
    visual_readout: VisualReadoutConfig | None = None

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
                "video_condition_input_space": VideoConditionInputSpace,
                "train_video_condition_source": VideoConditionSource,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
            },
        )
        if int(self.local_video_window_frames) <= 0:
            raise ValueError(
                "Post-latent policy requires `local_video_window_frames > 0`, "
                f"got local_video_window_frames={self.local_video_window_frames!r}."
            )
        if not (0 <= int(self.current_video_frame_index) < int(self.local_video_window_frames)):
            raise ValueError(
                "Post-latent policy requires `0 <= current_video_frame_index < local_video_window_frames`, "
                f"got current_video_frame_index={self.current_video_frame_index!r}, "
                f"local_video_window_frames={self.local_video_window_frames!r}."
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
    video_condition_input_space: VideoConditionInputSpace = VideoConditionInputSpace.RGB_VIDEO
    train_video_condition_source: VideoConditionSource = VideoConditionSource.LOCAL_WINDOW
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    local_video_window_frames: int = 4
    current_video_frame_index: int = 0
    visual_readout: VisualReadoutConfig | None = None

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
                "video_condition_input_space": VideoConditionInputSpace,
                "train_video_condition_source": VideoConditionSource,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
            },
        )
        if int(self.local_video_window_frames) <= 0:
            raise ValueError(
                "Post-decoded policy requires `local_video_window_frames > 0`, "
                f"got local_video_window_frames={self.local_video_window_frames!r}."
            )
        if not (0 <= int(self.current_video_frame_index) < int(self.local_video_window_frames)):
            raise ValueError(
                "Post-decoded policy requires `0 <= current_video_frame_index < local_video_window_frames`, "
                f"got current_video_frame_index={self.current_video_frame_index!r}, "
                f"local_video_window_frames={self.local_video_window_frames!r}."
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
    visual_readout: VisualReadoutConfig | None = None
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
class MoTPolicyConfig(PolicyVariantConfig):
    """Method-5 MoT scaffold config.

    The first Open-WAM version only wires the config/build surface and reserves
    the runtime modes for the later action-expert implementation stages.
    """

    name: PolicyVariantName = PolicyVariantName.MOT
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    preset: MoTPreset | None = None
    runtime_mode: MoTRuntimeMode = MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    condition_mode: MoTConditionMode = MoTConditionMode.FIRST_FRAME
    action_expert_init_mode: MoTActionExpertInitMode = MoTActionExpertInitMode.VIDEO_WEIGHT_COPY
    video_prefix_frames: int = 1
    teacher_forcing_video_noise_prob: float = 0.5
    # Probability of augmenting the ``V_clean`` copy with a light top-half
    # schedule corruption during non-joint packed training. Matches Method 1
    # ``ParallelStreamPolicyConfig.noisy_video_condition_prob`` (default 0.5).
    # When augmentation fires, the clean copy is noised with per-frame
    # timesteps sampled from ``[0.5, 1.0]`` of the schedule, simulating the
    # "past chunks were generated, not observed" regime at inference.
    noisy_video_condition_prob: float = 0.5
    num_action_layers: int = 30
    action_hidden_size: int | None = None
    action_ffn_dim: int | None = None
    video_can_attend_action: bool = True
    current_block_coupling: CurrentBlockCoupling | None = None
    use_text_conditioning: bool = True
    use_state_conditioning: bool = False
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    history_stream_visibility: ParallelHistoryStreamVisibility = ParallelHistoryStreamVisibility.FULL
    context_condition_latent_source: ParallelContextConditionLatentSource = (
        ParallelContextConditionLatentSource.VIDEO_LATENTS
    )
    # Trade forward compute for activation memory by recomputing each
    # (video, action) block pair during backward instead of storing its
    # activations. Only affects two-stream train paths that run through
    # `forward_joint_video_action_denoise`.
    use_activation_checkpointing: bool = False
    # Prefer reset-cache condition latents from latent datasets when available.
    # This keeps train-time clean video conditioning aligned with live rollout
    # observations while preserving fallback compatibility for datasets that
    # have not been augmented yet.
    use_condition_latents: bool = True
    require_condition_latents: bool = False
    parallel_sequence_contract: ParallelSequenceContract = ParallelSequenceContract.DEFAULT
    # Optional M5 generalist joint-denoise sampling distribution. ``None``
    # preserves the fixed six-mode path; a dict samples one of joint /
    # action_conditioned_video / video_conditioned_action per segment.
    mot_generalist_training_mode_probs: dict[MoTGeneralistTrainingMode, float] | None = None
    # Append a learned GJD mode token to text conditioning for M5 GJD ablations.
    # Proprio remains hidden-state per-chunk additive context, not a text token.
    # Only meaningful when `mot_generalist_training_mode_probs` is set.
    generalist_mode_text_token: bool = False
    # Canonical joint denoising synchronizes action/video noise levels by
    # sigma; index matching and independent clocks are explicit ablations.
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA
    # Deprecated compatibility shim for old configs/checkpoints. New configs
    # should set `joint_timestep_coupling` explicitly instead.
    couple_action_to_video_timesteps: bool | None = None
    generalist_training_paradigm: GeneralistTrainingParadigm = GeneralistTrainingParadigm.DEMO_ONLY

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "MoT policy requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )
        if int(self.video_prefix_frames) <= 0:
            raise ValueError(
                "MoT policy requires `video_prefix_frames > 0`, "
                f"got video_prefix_frames={self.video_prefix_frames!r}."
            )
        if not (0.0 <= float(self.teacher_forcing_video_noise_prob) <= 1.0):
            raise ValueError(
                "MoT policy requires `0 <= teacher_forcing_video_noise_prob <= 1`, "
                f"got teacher_forcing_video_noise_prob={self.teacher_forcing_video_noise_prob!r}."
            )
        if not (0.0 <= float(self.noisy_video_condition_prob) <= 1.0):
            raise ValueError(
                "MoT policy requires `0 <= noisy_video_condition_prob <= 1`, "
                f"got noisy_video_condition_prob={self.noisy_video_condition_prob!r}."
            )
        if bool(self.require_condition_latents) and not bool(self.use_condition_latents):
            raise ValueError("MoT `require_condition_latents` cannot be true when `use_condition_latents` is false.")
        if int(self.num_action_layers) <= 0:
            raise ValueError(
                "MoT policy requires `num_action_layers > 0`, "
                f"got num_action_layers={self.num_action_layers!r}."
            )
        if self.action_hidden_size is not None and int(self.action_hidden_size) <= 0:
            raise ValueError(
                "MoT policy requires `action_hidden_size > 0` when provided, "
                f"got action_hidden_size={self.action_hidden_size!r}."
            )
        if self.action_ffn_dim is not None and int(self.action_ffn_dim) <= 0:
            raise ValueError(
                "MoT policy requires `action_ffn_dim > 0` when provided, "
                f"got action_ffn_dim={self.action_ffn_dim!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "runtime_mode": MoTRuntimeMode,
                "condition_mode": MoTConditionMode,
                "action_expert_init_mode": MoTActionExpertInitMode,
                "generalist_training_paradigm": GeneralistTrainingParadigm,
                "proprio_context_mode": ProprioContextMode,
                "history_stream_visibility": ParallelHistoryStreamVisibility,
                "context_condition_latent_source": ParallelContextConditionLatentSource,
                "parallel_sequence_contract": ParallelSequenceContract,
                "joint_timestep_coupling": JointTimestepCoupling,
            },
            optional_enum_fields={
                "preset": MoTPreset,
                "current_block_coupling": CurrentBlockCoupling,
            },
            transforms={
                "mot_generalist_training_mode_probs": _coerce_mot_generalist_training_mode_probs,
            },
        )
        if self.couple_action_to_video_timesteps is not None:
            object.__setattr__(
                self,
                "joint_timestep_coupling",
                JointTimestepCoupling.MATCH_SIGMA
                if bool(self.couple_action_to_video_timesteps)
                else JointTimestepCoupling.INDEPENDENT,
            )
        if self.mot_generalist_training_mode_probs is not None:
            if self.current_block_coupling != CurrentBlockCoupling.JOINT:
                raise ValueError(
                    "`mot_generalist_training_mode_probs` requires `current_block_coupling = joint`, "
                    f"got current_block_coupling={self.current_block_coupling!r}."
                )
        if bool(self.generalist_mode_text_token) and self.mot_generalist_training_mode_probs is None:
            raise ValueError(
                "`generalist_mode_text_token = true` for MoT/M5 requires "
                "`mot_generalist_training_mode_probs` so the runtime has a sampled/forced GJD mode token."
            )
        if (
            self.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
            and self.mot_generalist_training_mode_probs is None
        ):
            raise ValueError(
                "`generalist_training_paradigm = mixed_dynamics` requires "
                "`mot_generalist_training_mode_probs` so the runtime can consume forced GJD modes."
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
    variant_profile: ParallelStreamVariantProfile = ParallelStreamVariantProfile.STANDARD
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
    current_block_coupling: CurrentBlockCoupling | None = None
    # Canonical joint denoising synchronizes action/video noise levels by
    # sigma; index matching and independent clocks are explicit ablations.
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA
    # Deprecated compatibility shim for old configs/checkpoints. New configs
    # should set `joint_timestep_coupling` explicitly instead.
    couple_action_to_video_timesteps: bool | None = None
    joint_denoise_training_mode_probs: dict[JointDenoiseTrainingMode, float] | None = None
    generalist_training_paradigm: GeneralistTrainingParadigm = GeneralistTrainingParadigm.DEMO_ONLY
    # Ablation: append one learned text-space token identifying the sampled
    # generalist mode (joint / action_conditioned_video / video_conditioned_action).
    generalist_mode_text_token: bool = False
    # When true, restrict PAST-chunk attention (both clean_to_clean and
    # noise_to_clean) so that any video-stream query (V_clean or V_noisy)
    # only sees same-stream history (V_clean), never history A_*. Action
    # queries (A_clean / A_noisy) keep full history visibility. Same-
    # chunk cross-stream visibility is unchanged across all 6 coupling
    # modes -- in particular staged modes' "current-chunk first-stage
    # clean reads" still work. The goal is to keep the video stream's
    # K/V context byte-aligned with the video-only pretrain distribution
    # at all transformer depths. Default false preserves backward compat
    # with existing checkpoints.
    preserve_video_pretrain_history: bool = False
    history_stream_visibility: ParallelHistoryStreamVisibility = ParallelHistoryStreamVisibility.FULL
    context_condition_latent_source: ParallelContextConditionLatentSource = (
        ParallelContextConditionLatentSource.VIDEO_LATENTS
    )
    use_condition_latents: bool = True
    require_condition_latents: bool = False
    parallel_sequence_contract: ParallelSequenceContract = ParallelSequenceContract.DEFAULT
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
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
                "variant_profile": ParallelStreamVariantProfile,
                "mask_mode": ParallelMaskMode,
                "cache_mode": ParallelCacheMode,
                "video_action_condition_source": ParallelActionConditionSource,
                "video_action_attention_scope": ParallelActionAttentionScope,
                "generalist_training_paradigm": GeneralistTrainingParadigm,
                "history_stream_visibility": ParallelHistoryStreamVisibility,
                "context_condition_latent_source": ParallelContextConditionLatentSource,
                "parallel_sequence_contract": ParallelSequenceContract,
                "proprio_context_mode": ProprioContextMode,
                "temporal_position_mode": TemporalPositionMode,
                "action_norm_method": ActionNormMethod,
                "joint_timestep_coupling": JointTimestepCoupling,
            },
            optional_enum_fields={"current_block_coupling": CurrentBlockCoupling},
            enum_tuple_fields={"sequence_order": ParallelSequenceComponent},
            transforms={
                "joint_denoise_training_mode_probs": lambda value: _coerce_joint_denoise_training_mode_probs(
                    value,
                    variant_profile=ParallelStreamVariantProfile(self.variant_profile),
                )
            },
        )
        if self.couple_action_to_video_timesteps is not None:
            object.__setattr__(
                self,
                "joint_timestep_coupling",
                JointTimestepCoupling.MATCH_SIGMA
                if bool(self.couple_action_to_video_timesteps)
                else JointTimestepCoupling.INDEPENDENT,
            )
        assert self.joint_denoise_training_mode_probs is not None
        if bool(self.require_condition_latents) and not bool(self.use_condition_latents):
            raise ValueError(
                "Parallel-stream `require_condition_latents` cannot be true when `use_condition_latents` is false."
            )
        if self.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING:
            conditional_generalist_modes_enabled = any(
                self.joint_denoise_training_mode_probs[mode] > 0.0
                for mode in (
                    JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
                    JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
                )
            )
            if (
                self.context_condition_latent_source
                == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            ):
                if self.parallel_sequence_contract != ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
                    raise ValueError(
                        "`context_condition_latent_source = single_frame_condition_latent` is not supported with "
                        "`variant_profile = generalist_joint_denoising`; the generalist rewrite expects full clean "
                        "video condition latents."
                    )
                if conditional_generalist_modes_enabled:
                    raise ValueError(
                        "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` supports "
                        "`variant_profile=generalist_joint_denoising` only when "
                        "`joint_denoise_training_mode_probs` is pure `joint`."
                    )
            if self.runtime_mode != ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
                raise ValueError(
                    "`variant_profile = generalist_joint_denoising` requires "
                    "`runtime_mode = lingbot_exact_action_conditioned`."
                )
            if self.current_block_coupling not in {None, CurrentBlockCoupling.JOINT}:
                raise ValueError(
                    "`variant_profile = generalist_joint_denoising` requires joint current-block coupling, "
                    f"got current_block_coupling={self.current_block_coupling!r}."
                )
            if not self.video_condition_on_action:
                raise ValueError(
                    "`variant_profile = generalist_joint_denoising` requires `video_condition_on_action = true`."
                )
        elif bool(self.generalist_mode_text_token):
            raise ValueError(
                "`generalist_mode_text_token = true` requires "
                "`variant_profile = generalist_joint_denoising`."
            )
        elif any(
            self.joint_denoise_training_mode_probs[mode] > 0.0
            for mode in (
                JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
                JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
            )
        ):
            raise ValueError(
                "Conditional joint-denoise training modes require "
                "`variant_profile = generalist_joint_denoising`."
            )
        if (
            self.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
            and self.variant_profile != ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        ):
            raise ValueError(
                "`generalist_training_paradigm = mixed_dynamics` requires "
                "`variant_profile = generalist_joint_denoising`."
            )
