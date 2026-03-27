from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyVariantConfig:
    """Base config shared by all policy variants."""

    name: str
    hidden_size: int
    attach_site: str


@dataclass(frozen=True)
class PostLatentPolicyConfig(PolicyVariantConfig):
    name: str = "post_latent"
    hidden_size: int = 256
    attach_site: str = "post_visual_core"
    pooling_mode: str = "per_frame_mean"
    query_count: int = 0
    temporal_projection: str = "interpolate"
    use_state_projection: bool = True
    compatibility_mode: bool = False


@dataclass(frozen=True)
class PostDecodedPolicyConfig(PolicyVariantConfig):
    name: str = "post_decoded"
    hidden_size: int = 256
    attach_site: str = "post_visual_decode"
    decode_feature_mode: str = "frame_token_sequence"
    pooling_mode: str = "per_frame_mean"
    temporal_projection: str = "interpolate"
    use_state_projection: bool = True


@dataclass(frozen=True)
class RegisterAttachedPolicyConfig(PolicyVariantConfig):
    name: str = "register_attached"
    hidden_size: int = 256
    attach_site: str = "within_visual_core"
    num_frame_per_block: int = 1
    num_action_per_block: int = 1
    num_state_per_block: int = 1
    max_chunk_size: int = 1
    register_layout: str = "action_then_state"
    mask_mode: str = "dreamzero_blockwise"
    use_state_encoder: bool = True
    action_encoder_type: str = "mlp"
    state_encoder_type: str = "mlp"
    couple_action_to_video_blocks: bool = True
    structured_block_mode: str = "register_explicit"
    structured_time_layout: str = "video_action_state"
    structured_frequency_mode: str = "stream_local"
    structured_teacher_forcing_layout: str = "clean_prefix"
    structured_attention_kernel: str = "branchwise_explicit"
    structured_cache_kernel: str = "branchwise_rollout_explicit"
    stream_input_adapter_family: str = "structured_register_streams"
    stream_output_head_family: str = "structured_joint_flow"


@dataclass(frozen=True)
class ParallelStreamPolicyConfig(PolicyVariantConfig):
    name: str = "parallel_stream"
    hidden_size: int = 256
    attach_site: str = "within_visual_core"
    runtime_mode: str = "lingbot_exact"
    reference_profile: str | None = None
    frame_chunk_size: int = 2
    action_per_frame: int = 1
    attn_window: int = 8
    sequence_order: tuple[str, ...] = field(
        default_factory=lambda: (
            "video_noisy",
            "video_condition",
            "action_noisy",
            "action_condition",
        )
    )
    mask_mode: str = "lingbot_chunked"
    cache_mode: str = "metadata_only"
    noisy_video_condition_prob: float = 0.5
    used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    inverse_used_action_channel_ids: tuple[int, ...] = field(default_factory=tuple)
    action_norm_method: str = "none"
    norm_q01: tuple[float, ...] = field(default_factory=tuple)
    norm_q99: tuple[float, ...] = field(default_factory=tuple)
