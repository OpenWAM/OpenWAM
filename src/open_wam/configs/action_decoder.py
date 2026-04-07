from __future__ import annotations

from dataclasses import dataclass

from .enums import (
    ActionDecoderName,
    ActionGenerationBackendFamily,
    DiffusionNoiseSchedule,
    DiffusionSampler,
    GoalConditioningAdapterFamily,
    SequenceDenoiserFamily,
    StateSequenceAdapterFamily,
    TemporalCompressionAdapterFamily,
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
class LingbotParallelActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.LINGBOT_PARALLEL
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


@dataclass(frozen=True)
class VideoOnlyActionDecoderConfig(ActionDecoderConfig):
    """Video-only decoder config for future-latent supervision without action loss."""

    name: ActionDecoderName = ActionDecoderName.VIDEO_ONLY
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
