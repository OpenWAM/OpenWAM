"""Typed M1/parallel-stream policy configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    ActionNormMethod,
    AttachSite,
    CurrentBlockCoupling,
    GeneralistTrainingParadigm,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelCacheMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    ParallelSequenceContract,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    ProprioContextMode,
    TemporalPositionMode,
    coerce_fields,
)
from .policy_contracts import PolicyVariantConfig
from .variant_semantics import coerce_probability_map, default_video_action_conditioning_mode_probs


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


__all__ = [
    "ParallelStreamPolicyConfig",
]
