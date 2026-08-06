"""Typed parallel-stream policy configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    ActionNormMethod,
    AttachSite,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    ParallelActionAttentionScope,
    ParallelActionConditionSource,
    ParallelCacheMode,
    ParallelMaskMode,
    ParallelRuntimeMode,
    ParallelSequenceComponent,
    ParallelStreamVariantProfile,
    PolicyVariantName,
    TemporalPositionMode,
    VideoActionSequenceContract,
    coerce_fields,
)
from .policy_compatibility import resolve_legacy_policy_field
from .policy_video_action import (
    VideoActionPolicyConfig,
    fixed_conditioning_mode_for_program,
    validate_conditional_denoising_data_paradigm,
)
from .variant_semantics import (
    coerce_probability_map,
    default_video_action_conditioning_mode_probs,
)


def _default_generalist_denoising_mode_probs(
    variant_profile: ParallelStreamVariantProfile,
) -> dict[GeneralistDenoisingMode, float]:
    return default_video_action_conditioning_mode_probs(
        GeneralistDenoisingMode,
        generalist=variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
    )


def _coerce_generalist_denoising_mode_probs(
    raw_value: object,
    *,
    variant_profile: ParallelStreamVariantProfile | str,
) -> dict[GeneralistDenoisingMode, float]:
    resolved_profile = ParallelStreamVariantProfile(variant_profile)
    if raw_value is None:
        return _default_generalist_denoising_mode_probs(resolved_profile)
    return coerce_probability_map(
        raw_value,
        enum_cls=GeneralistDenoisingMode,
        field_name="generalist_denoising_mode_probs",
    )


# Historical direct-import aliases.
_default_joint_denoise_training_mode_probs = (
    _default_generalist_denoising_mode_probs
)
_coerce_joint_denoise_training_mode_probs = _coerce_generalist_denoising_mode_probs


@dataclass(frozen=True)
class ParallelStreamPolicyConfig(VideoActionPolicyConfig):
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
    video_condition_on_action: bool = False
    video_action_condition_source: ParallelActionConditionSource = ParallelActionConditionSource.NOISY_ACTION
    video_action_attention_scope: ParallelActionAttentionScope = ParallelActionAttentionScope.BLOCK_LOCAL
    # Constructor/load alias for checkpoint-era parallel-stream configs.
    joint_denoise_training_mode_probs: dict[GeneralistDenoisingMode, float] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
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
    temporal_position_mode: TemporalPositionMode = TemporalPositionMode.GLOBAL_SHIFTED
    used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    inverse_used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    action_norm_method: ActionNormMethod = ActionNormMethod.NONE
    norm_q01: tuple[float, ...] = field(default_factory=tuple)
    norm_q99: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        super().__post_init__()
        if fixed_conditioning_mode_for_program(self.program) is not None:
            raise ValueError(
                f"`program = {self.program.value}` is currently supported only by the "
                "maintained dual_expert conditional runtime. Parallel-stream GJD remains "
                "a compatibility path with a different context contract."
            )
        resolved_generalist_probs = resolve_legacy_policy_field(
            canonical_value=self.generalist_denoising_mode_probs,
            legacy_value=self.joint_denoise_training_mode_probs,
            canonical_default=None,
            canonical_name="generalist_denoising_mode_probs",
            legacy_name="joint_denoise_training_mode_probs",
        )
        object.__setattr__(
            self,
            "generalist_denoising_mode_probs",
            resolved_generalist_probs,
        )
        coerce_fields(
            self,
            enum_fields={
                "runtime_mode": ParallelRuntimeMode,
                "variant_profile": ParallelStreamVariantProfile,
                "mask_mode": ParallelMaskMode,
                "cache_mode": ParallelCacheMode,
                "video_action_condition_source": ParallelActionConditionSource,
                "video_action_attention_scope": ParallelActionAttentionScope,
                "temporal_position_mode": TemporalPositionMode,
                "action_norm_method": ActionNormMethod,
            },
            enum_tuple_fields={"sequence_order": ParallelSequenceComponent},
            transforms={
                "generalist_denoising_mode_probs": lambda value: _coerce_generalist_denoising_mode_probs(
                    value,
                    variant_profile=ParallelStreamVariantProfile(self.variant_profile),
                )
            },
        )
        object.__setattr__(
            self,
            "joint_denoise_training_mode_probs",
            None,
        )
        assert self.generalist_denoising_mode_probs is not None
        validate_conditional_denoising_data_paradigm(
            probabilities=self.generalist_denoising_mode_probs,
            paradigm=self.generalist_training_paradigm,
        )
        if self.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING:
            conditional_generalist_modes_enabled = any(
                self.generalist_denoising_mode_probs[mode] > 0.0
                for mode in (
                    GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
                    GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
                )
            )
            if (
                self.context_condition_latent_source
                == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            ):
                if self.sequence_contract != VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
                    raise ValueError(
                        "`context_condition_latent_source = single_frame_condition_latent` is not supported with "
                        "`variant_profile = generalist_joint_denoising`; the generalist rewrite expects full clean "
                        "video condition latents."
                    )
                if conditional_generalist_modes_enabled:
                    raise ValueError(
                        "`sequence_contract=legacy_prefix_single_frame_perchunk_proprio` supports "
                        "`variant_profile=generalist_joint_denoising` only when "
                        "`generalist_denoising_mode_probs` is pure `joint`."
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
            self.generalist_denoising_mode_probs[mode] > 0.0
            for mode in (
                GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
                GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
            )
        ):
            raise ValueError(
                "Conditional joint-denoise training modes require "
                "`variant_profile = generalist_joint_denoising`."
            )
        if (
            self.generalist_training_paradigm == GeneralistTrainingParadigm.DYNAMICS_ROUTED
            and self.variant_profile != ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        ):
            raise ValueError(
                "`generalist_training_paradigm = dynamics_routed` requires "
                "`variant_profile = generalist_joint_denoising`."
            )


__all__ = [
    "ParallelStreamPolicyConfig",
]
