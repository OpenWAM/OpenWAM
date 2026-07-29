from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import enums as config_enums
from .backbone import SharedVideoTransformerConfig
from .coercion import (
    coerce_bool as _coerce_bool,
    coerce_enum as _coerce_enum,
    coerce_enum_tuple as _coerce_enum_tuple,
    coerce_optional_enum as _coerce_optional_enum,
)
from .data import DataConfig
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
    TemporalPositionMode,
    TemporalProjection,
    VideoConditionInputSpace,
    VideoConditionSource,
    coerce_fields,
)
from .inference import InferenceConfig
from .training import TrainingConfig
from .variant_semantics import coerce_probability_map, default_video_action_conditioning_mode_probs
from .visual_readout import VisualReadoutConfig, parse_visual_readout_config


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
class ExtensionPolicyConfig(PolicyVariantConfig):
    """Config envelope for an application-owned policy variant."""

    name: PolicyVariantName = PolicyVariantName.EXTENSION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    extension_type: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != PolicyVariantName.EXTENSION:
            raise ValueError("Extension policy requires `name = extension`.")
        if not isinstance(self.extension_type, str) or not self.extension_type.strip():
            raise ValueError("Extension policy requires a non-empty `extension_type` string.")
        if self.extension_type != self.extension_type.strip():
            raise ValueError("Extension policy `extension_type` must not have surrounding whitespace.")
        if not isinstance(self.options, Mapping):
            raise ValueError("Extension policy `options` must be a mapping.")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("Extension policy `options` keys must be strings.")
        object.__setattr__(self, "options", dict(self.options))


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


def parse_policy_variant_config(
    policy_variant_raw: Mapping[str, Any],
    action_head_raw: Mapping[str, Any],
    data_config: DataConfig,
    backbone_config: SharedVideoTransformerConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> PolicyVariantConfig:
    resolved_raw = dict(policy_variant_raw)
    compatibility_mode = False
    if not resolved_raw:
        compatibility_mode = True
        resolved_raw = {
            "name": config_enums.PolicyVariantName.POST_LATENT,
            "hidden_size": action_head_raw.get("hidden_size", backbone_config.hidden_size),
            "attach_site": config_enums.AttachSite.POST_VISUAL_CORE,
            "pooling_mode": config_enums.PoolingMode.COMPAT_GLOBAL_MEAN,
            "use_state_projection": True,
            "compatibility_mode": True,
        }

    name = _coerce_enum(
        config_enums.PolicyVariantName,
        resolved_raw.get("name", config_enums.PolicyVariantName.POST_LATENT),
    )
    hidden_size = resolved_raw.get("hidden_size", backbone_config.hidden_size)
    if name == config_enums.PolicyVariantName.EXTENSION:
        return ExtensionPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            extension_type=resolved_raw.get("extension_type", ""),
            options=resolved_raw.get("options", {}),
        )
    if name == config_enums.PolicyVariantName.POST_LATENT:
        return PostLatentPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get(
                    "pooling_mode",
                    (
                        config_enums.PoolingMode.COMPAT_GLOBAL_MEAN
                        if compatibility_mode
                        else config_enums.PoolingMode.PER_FRAME_MEAN
                    ),
                ),
            ),
            query_count=resolved_raw.get("query_count", 0),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            compatibility_mode=resolved_raw.get("compatibility_mode", compatibility_mode),
            video_condition_input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("video_condition_input_space", config_enums.VideoConditionInputSpace.VIDEO_LATENT),
            ),
            train_video_condition_source=_coerce_enum(
                config_enums.VideoConditionSource,
                resolved_raw.get("train_video_condition_source", config_enums.VideoConditionSource.LOCAL_WINDOW),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get(
                    "action_chunk_anchor_mode",
                    config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE,
                ),
            ),
            local_video_window_frames=resolved_raw.get("local_video_window_frames", 4),
            current_video_frame_index=resolved_raw.get("current_video_frame_index", 0),
            visual_readout=parse_visual_readout_config(resolved_raw.get("visual_readout")),
        )
    if name == config_enums.PolicyVariantName.POST_DECODED:
        return PostDecodedPolicyConfig(
            hidden_size=hidden_size,
            decode_feature_mode=_coerce_enum(
                config_enums.DecodeFeatureMode,
                resolved_raw.get("decode_feature_mode", config_enums.DecodeFeatureMode.FRAME_TOKEN_SEQUENCE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get("pooling_mode", config_enums.PoolingMode.PER_FRAME_MEAN),
            ),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            video_condition_input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("video_condition_input_space", config_enums.VideoConditionInputSpace.RGB_VIDEO),
            ),
            train_video_condition_source=_coerce_enum(
                config_enums.VideoConditionSource,
                resolved_raw.get("train_video_condition_source", config_enums.VideoConditionSource.LOCAL_WINDOW),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get(
                    "action_chunk_anchor_mode",
                    config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE,
                ),
            ),
            local_video_window_frames=resolved_raw.get("local_video_window_frames", 4),
            current_video_frame_index=resolved_raw.get("current_video_frame_index", 0),
            visual_readout=parse_visual_readout_config(resolved_raw.get("visual_readout")),
        )
    if name == config_enums.PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
        return CausalVideoPredictionPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
        )
    if name == config_enums.PolicyVariantName.MOT:
        preset = _coerce_optional_enum(
            config_enums.MoTPreset,
            resolved_raw.get("preset"),
        )
        mot_generalist_training_mode_probs = resolved_raw.get("mot_generalist_training_mode_probs")
        mot_joint_timestep_coupling_default = (
            config_enums.JointTimestepCoupling.INDEPENDENT
            if mot_generalist_training_mode_probs is not None
            else config_enums.JointTimestepCoupling.MATCH_SIGMA
        )
        mot_defaults: dict[str, Any] = {}
        if preset == config_enums.MoTPreset.FASTWAM:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.FIRST_FRAME,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_JOINT:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.JOINT_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.FULL_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_IDM:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.5,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_NON_JOINT:
            # Method-1-non-joint-aligned two-stream MoT: both video and action
            # run through a history-clean / current-noisy split, and the mask
            # disallows same-chunk noisy-to-noisy cross-stream attention.
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM,
                "condition_mode": config_enums.MoTConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        context_condition_latent_source = _coerce_enum(
            config_enums.ParallelContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ParallelContextConditionLatentSource.VIDEO_LATENTS,
            ),
        )
        return MoTPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            preset=preset,
            runtime_mode=_coerce_enum(
                config_enums.MoTRuntimeMode,
                resolved_raw.get(
                    "runtime_mode",
                    mot_defaults.get("runtime_mode", config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE),
                ),
            ),
            condition_mode=_coerce_enum(
                config_enums.MoTConditionMode,
                resolved_raw.get(
                    "condition_mode",
                    mot_defaults.get("condition_mode", config_enums.MoTConditionMode.FIRST_FRAME),
                ),
            ),
            action_expert_init_mode=_coerce_enum(
                config_enums.MoTActionExpertInitMode,
                resolved_raw.get(
                    "action_expert_init_mode",
                    config_enums.MoTActionExpertInitMode.VIDEO_WEIGHT_COPY,
                ),
            ),
            video_prefix_frames=resolved_raw.get("video_prefix_frames", mot_defaults.get("video_prefix_frames", 1)),
            teacher_forcing_video_noise_prob=resolved_raw.get(
                "teacher_forcing_video_noise_prob",
                mot_defaults.get("teacher_forcing_video_noise_prob", 0.5),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            num_action_layers=resolved_raw.get("num_action_layers", backbone_config.num_layers),
            action_hidden_size=resolved_raw.get("action_hidden_size"),
            action_ffn_dim=resolved_raw.get("action_ffn_dim"),
            video_can_attend_action=resolved_raw.get("video_can_attend_action", True),
            current_block_coupling=(
                _coerce_enum(config_enums.CurrentBlockCoupling, resolved_raw["current_block_coupling"])
                if "current_block_coupling" in resolved_raw
                else None
            ),
            use_text_conditioning=resolved_raw.get("use_text_conditioning", True),
            use_state_conditioning=resolved_raw.get("use_state_conditioning", False),
            proprio_context_mode=_coerce_enum(
                config_enums.ProprioContextMode,
                resolved_raw.get("proprio_context_mode", config_enums.ProprioContextMode.NONE),
            ),
            history_stream_visibility=_coerce_enum(
                config_enums.ParallelHistoryStreamVisibility,
                resolved_raw.get(
                    "history_stream_visibility",
                    config_enums.ParallelHistoryStreamVisibility.FULL,
                ),
            ),
            context_condition_latent_source=context_condition_latent_source,
            use_activation_checkpointing=resolved_raw.get("use_activation_checkpointing", False),
            use_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("use_condition_latents", True))
            ),
            require_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("require_condition_latents", False))
            ),
            parallel_sequence_contract=_coerce_enum(
                config_enums.ParallelSequenceContract,
                resolved_raw.get(
                    "parallel_sequence_contract",
                    config_enums.ParallelSequenceContract.DEFAULT,
                ),
            ),
            mot_generalist_training_mode_probs=mot_generalist_training_mode_probs,
            generalist_mode_text_token=_coerce_bool(
                resolved_raw.get("generalist_mode_text_token", False),
                field_name="policy_variant.generalist_mode_text_token",
            ),
            joint_timestep_coupling=_coerce_enum(
                config_enums.JointTimestepCoupling,
                resolved_raw.get(
                    "joint_timestep_coupling",
                    mot_joint_timestep_coupling_default,
                ),
            ),
            couple_action_to_video_timesteps=resolved_raw.get("couple_action_to_video_timesteps"),
            generalist_training_paradigm=_coerce_enum(
                config_enums.GeneralistTrainingParadigm,
                resolved_raw.get(
                    "generalist_training_paradigm",
                    config_enums.GeneralistTrainingParadigm.DEMO_ONLY,
                ),
            ),
        )
    if name == config_enums.PolicyVariantName.PARALLEL_STREAM:
        default_action_per_frame = max(
            1,
            data_config.action_schema.action_horizon // max(1, data_config.num_frames),
        )
        runtime_mode = _coerce_enum(
            config_enums.ParallelRuntimeMode,
            resolved_raw.get("runtime_mode", config_enums.ParallelRuntimeMode.LINGBOT_EXACT),
        )
        current_block_coupling = (
            _coerce_enum(
                config_enums.CurrentBlockCoupling,
                resolved_raw["current_block_coupling"],
            )
            if "current_block_coupling" in resolved_raw
            else None
        )
        preserve_video_pretrain_history = bool(
            resolved_raw.get("preserve_video_pretrain_history", False)
        )
        history_stream_visibility = _coerce_enum(
            config_enums.ParallelHistoryStreamVisibility,
            resolved_raw.get(
                "history_stream_visibility",
                (
                    config_enums.ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
                    if preserve_video_pretrain_history
                    else config_enums.ParallelHistoryStreamVisibility.FULL
                ),
            ),
        )
        context_condition_latent_source = _coerce_enum(
            config_enums.ParallelContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ParallelContextConditionLatentSource.VIDEO_LATENTS,
            ),
        )
        proprio_context_mode = _coerce_enum(
            config_enums.ProprioContextMode,
            resolved_raw.get("proprio_context_mode", config_enums.ProprioContextMode.NONE),
        )
        if proprio_context_mode == config_enums.ProprioContextMode.PER_CHUNK_ADDITIVE:
            exact_runtime_mode = runtime_mode in {
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT,
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            }
            compact_runtime_mode = runtime_mode in {
                config_enums.ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
                config_enums.ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            }
            if exact_runtime_mode and current_block_coupling is None:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive requires "
                    "`policy_variant.current_block_coupling` for Method-1 chunk semantics."
                )
            if not exact_runtime_mode and not compact_runtime_mode:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive is only supported for "
                    "LingBot exact and compact current-frame Method-1 runtime modes."
                )
        use_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            else bool(resolved_raw.get("use_condition_latents", True))
        )
        require_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            else bool(resolved_raw.get("require_condition_latents", False))
        )
        sequence_order = tuple(
            resolved_raw.get(
                "sequence_order",
                (
                    config_enums.ParallelSequenceComponent.VIDEO_NOISY,
                    config_enums.ParallelSequenceComponent.VIDEO_CONDITION,
                    config_enums.ParallelSequenceComponent.ACTION_NOISY,
                    config_enums.ParallelSequenceComponent.ACTION_CONDITION,
                ),
            )
        )
        variant_profile = _coerce_enum(
            config_enums.ParallelStreamVariantProfile,
            resolved_raw.get("variant_profile", config_enums.ParallelStreamVariantProfile.STANDARD),
        )
        parallel_joint_timestep_coupling_default = (
            config_enums.JointTimestepCoupling.INDEPENDENT
            if variant_profile == config_enums.ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
            else config_enums.JointTimestepCoupling.MATCH_SIGMA
        )
        return ParallelStreamPolicyConfig(
            hidden_size=hidden_size,
            runtime_mode=runtime_mode,
            variant_profile=variant_profile,
            reference_profile=resolved_raw.get("reference_profile"),
            frame_chunk_size=resolved_raw.get("frame_chunk_size", inference_config.frame_chunk_size),
            action_per_frame=resolved_raw.get("action_per_frame", default_action_per_frame),
            attn_window=resolved_raw.get("attn_window", training_config.window_size),
            sequence_order=_coerce_enum_tuple(config_enums.ParallelSequenceComponent, sequence_order),
            mask_mode=_coerce_enum(
                config_enums.ParallelMaskMode,
                resolved_raw.get("mask_mode", config_enums.ParallelMaskMode.LINGBOT_CHUNKED),
            ),
            cache_mode=_coerce_enum(
                config_enums.ParallelCacheMode,
                resolved_raw.get("cache_mode", config_enums.ParallelCacheMode.METADATA_ONLY),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            video_condition_on_action=resolved_raw.get("video_condition_on_action", False),
            video_action_condition_source=_coerce_enum(
                config_enums.ParallelActionConditionSource,
                resolved_raw.get(
                    "video_action_condition_source",
                    config_enums.ParallelActionConditionSource.NOISY_ACTION,
                ),
            ),
            video_action_attention_scope=_coerce_enum(
                config_enums.ParallelActionAttentionScope,
                resolved_raw.get(
                    "video_action_attention_scope",
                    config_enums.ParallelActionAttentionScope.BLOCK_LOCAL,
                ),
            ),
            joint_timestep_coupling=_coerce_enum(
                config_enums.JointTimestepCoupling,
                resolved_raw.get(
                    "joint_timestep_coupling",
                    parallel_joint_timestep_coupling_default,
                ),
            ),
            couple_action_to_video_timesteps=resolved_raw.get("couple_action_to_video_timesteps"),
            joint_denoise_training_mode_probs=resolved_raw.get("joint_denoise_training_mode_probs"),
            generalist_training_paradigm=_coerce_enum(
                config_enums.GeneralistTrainingParadigm,
                resolved_raw.get(
                    "generalist_training_paradigm",
                    config_enums.GeneralistTrainingParadigm.DEMO_ONLY,
                ),
            ),
            generalist_mode_text_token=_coerce_bool(
                resolved_raw.get("generalist_mode_text_token", False),
                field_name="policy_variant.generalist_mode_text_token",
            ),
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=preserve_video_pretrain_history,
            history_stream_visibility=history_stream_visibility,
            context_condition_latent_source=context_condition_latent_source,
            use_condition_latents=use_condition_latents,
            proprio_context_mode=proprio_context_mode,
            require_condition_latents=require_condition_latents,
            parallel_sequence_contract=_coerce_enum(
                config_enums.ParallelSequenceContract,
                resolved_raw.get(
                    "parallel_sequence_contract",
                    config_enums.ParallelSequenceContract.DEFAULT,
                ),
            ),
            temporal_position_mode=_coerce_enum(
                config_enums.TemporalPositionMode,
                resolved_raw.get(
                    "temporal_position_mode",
                    config_enums.TemporalPositionMode.GLOBAL_SHIFTED,
                ),
            ),
            used_action_channel_ids=tuple(resolved_raw.get("used_action_channel_ids", ())),
            inverse_used_action_channel_ids=tuple(resolved_raw.get("inverse_used_action_channel_ids", ())),
            action_norm_method=_coerce_enum(
                config_enums.ActionNormMethod,
                resolved_raw.get("action_norm_method", config_enums.ActionNormMethod.PROFILE),
            ),
            norm_q01=tuple(resolved_raw.get("norm_q01", ())),
            norm_q99=tuple(resolved_raw.get("norm_q99", ())),
        )
    raise ValueError(f"Unsupported policy variant '{name}'.")
