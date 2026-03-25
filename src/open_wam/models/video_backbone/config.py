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
    implementation: str = "dummy"
    hidden_size: int = 3072
    num_layers: int = 0
    num_heads: int = 8
    mlp_ratio: int = 4
    ffn_dim: int | None = None
    text_dim: int = 4096
    freq_dim: int = 256
    cross_attn_norm: bool = True
    rope_max_seq_len: int = 1024
    latent_norm_eps: float = 1e-6
