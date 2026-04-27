from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from open_wam.configs.enums import (
    AttentionMode,
    BackboneImplementation,
    ExportedRuntimeActionInitMode,
    ReferenceAssetsDevicePolicy,
    ReferenceCoreInitMode,
    coerce_fields,
)


_IMPLEMENTATION_ALIASES: dict[str, BackboneImplementation] = {
    "shared_transformer": BackboneImplementation.SHARED_TRANSFORMER,
    "lingbot_replica": BackboneImplementation.SHARED_TRANSFORMER,
    "dummy": BackboneImplementation.DUMMY,
}


def normalize_backbone_implementation(name: str | BackboneImplementation) -> BackboneImplementation:
    try:
        return _IMPLEMENTATION_ALIASES[name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported backbone implementation {name!r}. "
            f"Expected one of {tuple(_IMPLEMENTATION_ALIASES)}."
        ) from exc


@dataclass(frozen=True)
class SharedVideoTransformerConfig:
    """Config for the shared video-transformer backbone family.

    The defaults preserve the protected reference geometry:

    - canonical RGB canvas: 384x320
    - latent spatial stride: 16
    - latent shape per frame: 24x20
    - latent channels: 48
    - patch size: (1, 2, 2)
    - tokens per frame: 12 * 10 = 120

    `hidden_size` matches the reference model by default but can be lowered for smoke tests.
    """

    input_channels: int = 3
    latent_channels: int = 48
    latent_stride: int = 16
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2
    # Default to the shared transformer implementation so real variants run on
    # the same backbone family unless a smoke-test config overrides it.
    implementation: BackboneImplementation = BackboneImplementation.SHARED_TRANSFORMER
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
    attn_mode: AttentionMode = AttentionMode.TORCH
    train_attn_mode: AttentionMode | None = None
    infer_attn_mode: AttentionMode | None = None
    pretrained_model_name_or_path: str | None = None
    transformer_subdir: str = "transformer"
    vae_subdir: str = "vae"
    text_encoder_subdir: str = "text_encoder"
    tokenizer_subdir: str = "tokenizer"
    max_text_tokens: int = 512
    load_wan_vae_frontend: bool = False
    load_text_conditioning: bool = False
    load_reference_core_weights: bool = False
    reference_core_init_mode: ReferenceCoreInitMode = ReferenceCoreInitMode.FULL
    # Optional secondary root used by mixed initialization modes that borrow a
    # small subset of calibrated video-core weights from a compatible checkpoint.
    reference_norm2_source_path: str | None = None
    # When loading an exported runtime backbone, choose whether action/runtime
    # modules come from that checkpoint or remain randomly initialized.
    exported_runtime_action_init_mode: ExportedRuntimeActionInitMode = (
        ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT
    )
    # `runtime`: keep reference VAE/text assets on the active runtime device.
    # `cpu_offload`: mirror Heng's eval server and keep them on CPU.
    reference_assets_device_policy: ReferenceAssetsDevicePolicy = ReferenceAssetsDevicePolicy.RUNTIME
    # Optional override for the vendored LingBot reference model source file.
    reference_model_path: str | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "attn_mode": AttentionMode,
                "exported_runtime_action_init_mode": ExportedRuntimeActionInitMode,
                "reference_assets_device_policy": ReferenceAssetsDevicePolicy,
                "reference_core_init_mode": ReferenceCoreInitMode,
            },
            optional_enum_fields={
                "train_attn_mode": AttentionMode,
                "infer_attn_mode": AttentionMode,
            },
            transforms={
                "implementation": normalize_backbone_implementation,
            },
        )


LingbotCompatibleVideoBackboneConfig = SharedVideoTransformerConfig


def resolve_stage_attention_mode(
    config: SharedVideoTransformerConfig,
    *,
    stage: Literal["train", "infer"],
    exact_runtime: bool = False,
) -> AttentionMode:
    if stage == "train":
        if config.train_attn_mode is not None:
            return config.train_attn_mode
        if exact_runtime:
            # LingBot requires FlexAttention for exact training while keeping
            # inference on torch/flash attention.
            return AttentionMode.FLEX
        return config.attn_mode
    if config.infer_attn_mode is not None:
        return config.infer_attn_mode
    return config.attn_mode
