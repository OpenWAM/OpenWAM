"""Typed M5/MoT policy configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    AttachSite,
    CurrentBlockCoupling,
    GeneralistTrainingParadigm,
    JointTimestepCoupling,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTGeneralistTrainingMode,
    MoTPreset,
    MoTRuntimeMode,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelSequenceContract,
    PolicyVariantName,
    ProprioContextMode,
    coerce_fields,
)
from .policy_contracts import PolicyVariantConfig
from .variant_semantics import coerce_probability_map


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


__all__ = [
    "MoTPolicyConfig",
]
