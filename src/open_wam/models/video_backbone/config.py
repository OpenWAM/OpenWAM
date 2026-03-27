from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LingbotCompatibleVideoBackboneConfig:
    """Stage-1 config for the protected LingBot-compatible video backbone.

    The defaults preserve LingBot geometry:

    - canonical RGB canvas: 384x320
    - latent spatial stride: 16
    - latent shape per frame: 24x20
    - latent channels: 48
    - patch size: (1, 2, 2)
    - tokens per frame: 12 * 10 = 120

    `hidden_size` matches LingBot by default but can be lowered for smoke tests.
    """

    input_channels: int = 3
    latent_channels: int = 48
    latent_stride: int = 16
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2
    # Default to the LingBot-style shared-core implementation so real variants
    # run on the same backbone family unless a smoke-test config overrides it.
    implementation: str = "lingbot_replica"
    hidden_size: int = 3072
    num_layers: int = 1
    num_heads: int = 8
    attention_head_dim: int | None = None
    mlp_ratio: int = 4
    ffn_dim: int | None = None
    text_dim: int = 4096
    freq_dim: int = 256
    cross_attn_norm: bool = True
    rope_max_seq_len: int = 1024
    latent_norm_eps: float = 1e-6
    attn_mode: str = "torch"
    pretrained_model_name_or_path: str | None = None
    transformer_subdir: str = "transformer"
    vae_subdir: str = "vae"
    text_encoder_subdir: str = "text_encoder"
    tokenizer_subdir: str = "tokenizer"
    max_text_tokens: int = 512
    load_wan_vae_frontend: bool = False
    load_text_conditioning: bool = False
    load_reference_core_weights: bool = False
    # `runtime`: keep reference VAE/text assets on the active runtime device.
    # `cpu_offload`: mirror Heng's eval server and keep them on CPU.
    reference_assets_device_policy: str = "runtime"
    # Optional override for the vendored LingBot reference model source file.
    reference_model_path: str | None = None
