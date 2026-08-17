"""Typed dual-expert policy configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    AttachSite,
    CurrentBlockCoupling,
    DualExpertActionExpertInitMode,
    DualExpertConditionMode,
    DualExpertPreset,
    DualExpertRuntimeMode,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    PolicyVariantName,
    coerce_fields,
)
from .policy_compatibility import resolve_legacy_policy_field
from .policy_video_action import (
    VideoActionPolicyConfig,
    fixed_conditioning_mode_for_program,
    one_hot_conditioning_mode_probabilities,
    validate_conditional_denoising_data_paradigm,
)
from .variant_semantics import coerce_probability_map


def _coerce_generalist_denoising_mode_probs(
    raw_value: object,
) -> dict[GeneralistDenoisingMode, float] | None:
    """Coerce an optional generalist denoising distribution.

    ``None`` keeps the existing fixed ``current_block_coupling`` path. When a
    mapping is provided, missing modes default to 0 and probabilities are
    normalized to sum to one.
    """

    if raw_value is None:
        return None
    return coerce_probability_map(
        raw_value,
        enum_cls=GeneralistDenoisingMode,
        field_name="generalist_denoising_mode_probs",
    )


# Historical direct-import alias.
_coerce_mot_generalist_training_mode_probs = _coerce_generalist_denoising_mode_probs


@dataclass(frozen=True)
class DualExpertPolicyConfig(VideoActionPolicyConfig):
    """Configuration for separate video and action transformer experts."""

    name: PolicyVariantName = PolicyVariantName.DUAL_EXPERT
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    preset: DualExpertPreset | None = None
    runtime_mode: DualExpertRuntimeMode = DualExpertRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    condition_mode: DualExpertConditionMode = DualExpertConditionMode.FIRST_FRAME
    action_expert_init_mode: DualExpertActionExpertInitMode = DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY
    video_prefix_frames: int = 1
    teacher_forcing_video_noise_prob: float = 0.5
    num_action_layers: int = 30
    action_hidden_size: int | None = None
    action_ffn_dim: int | None = None
    video_can_attend_action: bool = True
    use_text_conditioning: bool = True
    use_state_conditioning: bool = False
    # Trade forward compute for activation memory by recomputing each
    # (video, action) block pair during backward instead of storing its
    # activations. Only affects two-stream train paths that run through
    # `forward_joint_video_action_denoise`.
    use_activation_checkpointing: bool = False
    # Prefer reset-cache condition latents from latent datasets when available.
    # This keeps train-time clean video conditioning aligned with live rollout
    # observations while preserving fallback compatibility for datasets that
    # have not been augmented yet.
    # Constructor/load alias for checkpoint-era `mot` configs.
    mot_generalist_training_mode_probs: dict[GeneralistDenoisingMode, float] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        resolved_generalist_probs = resolve_legacy_policy_field(
            canonical_value=self.generalist_denoising_mode_probs,
            legacy_value=self.mot_generalist_training_mode_probs,
            canonical_default=None,
            canonical_name="generalist_denoising_mode_probs",
            legacy_name="mot_generalist_training_mode_probs",
        )
        fixed_conditioning_mode = fixed_conditioning_mode_for_program(self.program)
        if fixed_conditioning_mode is not None and resolved_generalist_probs is None:
            resolved_generalist_probs = one_hot_conditioning_mode_probabilities(
                fixed_conditioning_mode
            )
        object.__setattr__(
            self,
            "generalist_denoising_mode_probs",
            resolved_generalist_probs,
        )
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "DualExpert policy requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )
        if int(self.video_prefix_frames) <= 0:
            raise ValueError(
                "DualExpert policy requires `video_prefix_frames > 0`, "
                f"got video_prefix_frames={self.video_prefix_frames!r}."
            )
        if not (0.0 <= float(self.teacher_forcing_video_noise_prob) <= 1.0):
            raise ValueError(
                "DualExpert policy requires `0 <= teacher_forcing_video_noise_prob <= 1`, "
                f"got teacher_forcing_video_noise_prob={self.teacher_forcing_video_noise_prob!r}."
            )
        if int(self.num_action_layers) <= 0:
            raise ValueError(
                "DualExpert policy requires `num_action_layers > 0`, "
                f"got num_action_layers={self.num_action_layers!r}."
            )
        if self.action_hidden_size is not None and int(self.action_hidden_size) <= 0:
            raise ValueError(
                "DualExpert policy requires `action_hidden_size > 0` when provided, "
                f"got action_hidden_size={self.action_hidden_size!r}."
            )
        if self.action_ffn_dim is not None and int(self.action_ffn_dim) <= 0:
            raise ValueError(
                "DualExpert policy requires `action_ffn_dim > 0` when provided, "
                f"got action_ffn_dim={self.action_ffn_dim!r}."
            )
        coerce_fields(
            self,
            enum_fields={
                "runtime_mode": DualExpertRuntimeMode,
                "condition_mode": DualExpertConditionMode,
                "action_expert_init_mode": DualExpertActionExpertInitMode,
            },
            optional_enum_fields={"preset": DualExpertPreset},
            transforms={
                "generalist_denoising_mode_probs": _coerce_generalist_denoising_mode_probs,
            },
        )
        object.__setattr__(
            self,
            "mot_generalist_training_mode_probs",
            None,
        )
        validate_conditional_denoising_data_paradigm(
            probabilities=self.generalist_denoising_mode_probs,
            paradigm=self.generalist_training_paradigm,
        )
        if self.generalist_denoising_mode_probs is not None:
            if self.current_block_coupling != CurrentBlockCoupling.JOINT:
                raise ValueError(
                    "`generalist_denoising_mode_probs` requires `current_block_coupling = joint`, "
                    f"got current_block_coupling={self.current_block_coupling!r}."
                )
        if fixed_conditioning_mode is not None:
            expected_probs = one_hot_conditioning_mode_probabilities(
                fixed_conditioning_mode
            )
            if self.generalist_denoising_mode_probs != expected_probs:
                raise ValueError(
                    f"`program = {self.program.value}` owns a fixed one-hot "
                    f"{fixed_conditioning_mode.value!r} conditioning mode; do not set a "
                    "conflicting `generalist_denoising_mode_probs` distribution."
                )
            if bool(self.generalist_mode_text_token):
                raise ValueError(
                    f"`program = {self.program.value}` does not use a GJD mode token; "
                    "set `generalist_mode_text_token = false`."
                )
        if bool(self.generalist_mode_text_token) and self.generalist_denoising_mode_probs is None:
            raise ValueError(
                "`generalist_mode_text_token = true` requires `generalist_denoising_mode_probs` "
                "so the runtime has a sampled or forced GJD mode token."
            )
        if (
            self.generalist_training_paradigm == GeneralistTrainingParadigm.DYNAMICS_ROUTED
            and self.generalist_denoising_mode_probs is None
        ):
            raise ValueError(
                "`generalist_training_paradigm = dynamics_routed` requires "
                "`generalist_denoising_mode_probs` so the runtime can consume forced GJD modes."
            )


__all__ = [
    "DualExpertPolicyConfig",
]
