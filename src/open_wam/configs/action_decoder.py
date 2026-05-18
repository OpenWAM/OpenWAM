from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    ActionDecoderName,
    ActionChunkAnchorMode,
    ActionExpertInitMode,
    ActionGenerationBackendFamily,
    DiffusionNoiseSchedule,
    DiffusionSampler,
    GoalConditioningAdapterFamily,
    SequenceDenoiserFamily,
    StateSequenceAdapterFamily,
    TemporalCompressionAdapterFamily,
    VideoConditionInputSpace,
    VideoConditionTrainMode,
    coerce_fields,
)


@dataclass(frozen=True)
class ActionDecoderConfig:
    """Final action-decoder config independent from policy attachment."""

    name: ActionDecoderName
    hidden_size: int
    action_dim: int
    action_horizon: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        coerce_fields(self, enum_fields={"name": ActionDecoderName})


@dataclass(frozen=True)
class MLPActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.MLP
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class RegisterActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.REGISTER
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class DecodedFeatureActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.DECODED_FEATURE
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class VideoConditionedActionDecoderConfig(ActionDecoderConfig):
    """Generic current-action decoder over a local video-conditioned window."""

    name: ActionDecoderName = ActionDecoderName.VIDEO_CONDITIONED
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    context_dim: int = 256
    text_context_dim: int = 0
    state_dim: int = 0
    freq_dim: int = 256
    num_layers: int = 1
    num_heads: int = 8
    attention_head_dim: int = 32
    ffn_dim: int = 1024
    cross_attn_norm: bool = True
    eps: float = 1e-6
    input_space: VideoConditionInputSpace = VideoConditionInputSpace.VIDEO_LATENT
    train_mode: VideoConditionTrainMode = VideoConditionTrainMode.ROLLOUT_WINDOW_DIFFUSION
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    action_expert_init_mode: ActionExpertInitMode = ActionExpertInitMode.VIDEO_WEIGHT_COPY
    rollout_chunk_steps: int = 1
    direct_latent_channels: int = 48
    direct_rgb_patch_size: int = 16
    use_text_conditioning: bool = True
    use_state_conditioning: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "input_space": VideoConditionInputSpace,
                "train_mode": VideoConditionTrainMode,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
                "action_expert_init_mode": ActionExpertInitMode,
            },
        )
        if (
            self.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
            and self.action_chunk_anchor_mode != ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
        ):
            raise ValueError(
                "Current-frame regression mode requires `action_chunk_anchor_mode = current_plus_future`, "
                f"got action_chunk_anchor_mode={self.action_chunk_anchor_mode!r}."
            )
        if int(self.rollout_chunk_steps) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `rollout_chunk_steps > 0`, "
                f"got rollout_chunk_steps={self.rollout_chunk_steps!r}."
            )
        if int(self.direct_latent_channels) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `direct_latent_channels > 0`, "
                f"got direct_latent_channels={self.direct_latent_channels!r}."
            )
        if int(self.direct_rgb_patch_size) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `direct_rgb_patch_size > 0`, "
                f"got direct_rgb_patch_size={self.direct_rgb_patch_size!r}."
            )


@dataclass(frozen=True)
class LingbotParallelActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.LINGBOT_PARALLEL
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    recovered_osc_loss_weight: float = 0.0
    recovered_osc_position_scale: float = 0.010576533139391671
    recovered_osc_rotation_scale: float = 0.1136411594890211


@dataclass(frozen=True)
class MoTActionDecoderConfig(ActionDecoderConfig):
    """MoT decoder config for action/video flow supervision and infer packaging."""

    name: ActionDecoderName = ActionDecoderName.MOT
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class VPPActionDecoderConfig(ActionDecoderConfig):
    """Sequence-native action decoder configuration closest to VPP semantics."""

    name: ActionDecoderName = ActionDecoderName.VPP
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    temporal_compression_adapter_family: TemporalCompressionAdapterFamily = (
        TemporalCompressionAdapterFamily.TEMPORAL_LATENT_RESAMPLER_3D
    )
    sequence_denoiser_family: SequenceDenoiserFamily = SequenceDenoiserFamily.GENERIC_TRANSFORMER
    goal_conditioning_adapter_family: GoalConditioningAdapterFamily = GoalConditioningAdapterFamily.PASSTHROUGH
    state_sequence_adapter_family: StateSequenceAdapterFamily = StateSequenceAdapterFamily.LINEAR
    action_generation_backend: ActionGenerationBackendFamily = ActionGenerationBackendFamily.EDM_DIFFUSION
    diffusion_noise_schedule: DiffusionNoiseSchedule = DiffusionNoiseSchedule.EXPONENTIAL
    diffusion_sampler: DiffusionSampler = DiffusionSampler.DDIM
    num_sampling_steps: int | None = None
    rollout_chunk_steps: int | None = None
    compressed_tokens_per_frame: int = 2
    compression_depth: int = 2
    temporal_compression_max_frames: int = 32
    num_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    sigma_data: float = 0.5
    sigma_min: float = 0.001
    sigma_max: float = 80.0

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "temporal_compression_adapter_family": TemporalCompressionAdapterFamily,
                "sequence_denoiser_family": SequenceDenoiserFamily,
                "goal_conditioning_adapter_family": GoalConditioningAdapterFamily,
                "state_sequence_adapter_family": StateSequenceAdapterFamily,
                "action_generation_backend": ActionGenerationBackendFamily,
                "diffusion_noise_schedule": DiffusionNoiseSchedule,
                "diffusion_sampler": DiffusionSampler,
            },
        )
        if int(self.temporal_compression_max_frames) <= 0:
            raise ValueError(
                "VPP action decoder requires `temporal_compression_max_frames > 0`, "
                f"got {self.temporal_compression_max_frames!r}."
            )


@dataclass(frozen=True)
class VideoOnlyActionDecoderConfig(ActionDecoderConfig):
    """Video-only decoder config for future-latent supervision without action loss."""

    name: ActionDecoderName = ActionDecoderName.VIDEO_ONLY
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
