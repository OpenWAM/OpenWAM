from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from open_wam.contracts.paths import validate_model_component_path

from .coercion import coerce_enum, coerce_optional_enum
from .enums import (
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


def normalize_backbone_implementation(
    name: str | BackboneImplementation,
) -> BackboneImplementation:
    try:
        return _IMPLEMENTATION_ALIASES[name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported backbone implementation {name!r}. "
            f"Expected one of {tuple(_IMPLEMENTATION_ALIASES)}."
        ) from exc


@dataclass(frozen=True)
class SharedVideoTransformerConfig:
    """Configuration for the shared video-transformer backbone family."""

    input_channels: int = 3
    latent_channels: int = 48
    latent_stride: int = 16
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2
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
    runtime_backbone_artifact_path: str | None = None
    transformer_subdir: str = "transformer"
    vae_subdir: str = "vae"
    text_encoder_subdir: str = "text_encoder"
    tokenizer_subdir: str = "tokenizer"
    max_text_tokens: int = 512
    load_wan_vae_frontend: bool = False
    load_text_conditioning: bool = False
    load_reference_core_weights: bool = False
    reference_core_init_mode: ReferenceCoreInitMode = ReferenceCoreInitMode.FULL
    reference_norm2_source_path: str | None = None
    exported_runtime_action_init_mode: ExportedRuntimeActionInitMode = (
        ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT
    )
    reference_assets_device_policy: ReferenceAssetsDevicePolicy = (
        ReferenceAssetsDevicePolicy.RUNTIME
    )
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
        validate_model_component_path(
            self.transformer_subdir,
            field_name="backbone.transformer_subdir",
        )


def parse_shared_video_transformer_config(
    raw_value: Mapping[str, Any] | None,
) -> SharedVideoTransformerConfig:
    """Parse the backbone section at the typed configuration boundary."""

    raw = raw_value or {}
    defaults = SharedVideoTransformerConfig()
    pretrained_model_name_or_path = raw.get("pretrained_model_name_or_path")
    load_wan_vae_frontend = raw.get("load_wan_vae_frontend")
    if load_wan_vae_frontend is None:
        load_wan_vae_frontend = pretrained_model_name_or_path is not None
    load_text_conditioning = raw.get("load_text_conditioning")
    if load_text_conditioning is None:
        load_text_conditioning = pretrained_model_name_or_path is not None
    load_reference_core_weights = raw.get("load_reference_core_weights")
    if load_reference_core_weights is None:
        load_reference_core_weights = False

    return SharedVideoTransformerConfig(
        input_channels=raw.get("input_channels", defaults.input_channels),
        latent_channels=raw.get("latent_channels", defaults.latent_channels),
        latent_stride=raw.get("latent_stride", defaults.latent_stride),
        patch_size_t=raw.get("patch_size_t", defaults.patch_size_t),
        patch_size_h=raw.get("patch_size_h", defaults.patch_size_h),
        patch_size_w=raw.get("patch_size_w", defaults.patch_size_w),
        implementation=normalize_backbone_implementation(
            raw.get("implementation", defaults.implementation)
        ),
        hidden_size=raw.get("hidden_size", defaults.hidden_size),
        num_layers=raw.get("num_layers", defaults.num_layers),
        num_heads=raw.get("num_heads", defaults.num_heads),
        attention_head_dim=raw.get("attention_head_dim"),
        mlp_ratio=raw.get("mlp_ratio", defaults.mlp_ratio),
        ffn_dim=raw.get("ffn_dim"),
        text_dim=raw.get("text_dim", defaults.text_dim),
        freq_dim=raw.get("freq_dim", defaults.freq_dim),
        cross_attn_norm=raw.get("cross_attn_norm", defaults.cross_attn_norm),
        rope_max_seq_len=raw.get("rope_max_seq_len", defaults.rope_max_seq_len),
        latent_norm_eps=raw.get("latent_norm_eps", defaults.latent_norm_eps),
        attn_mode=coerce_enum(
            AttentionMode,
            raw.get("attn_mode", defaults.attn_mode),
        ),
        train_attn_mode=coerce_optional_enum(
            AttentionMode,
            raw.get("train_attn_mode", defaults.train_attn_mode),
        ),
        infer_attn_mode=coerce_optional_enum(
            AttentionMode,
            raw.get("infer_attn_mode", defaults.infer_attn_mode),
        ),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        runtime_backbone_artifact_path=raw.get(
            "runtime_backbone_artifact_path",
            defaults.runtime_backbone_artifact_path,
        ),
        transformer_subdir=raw.get("transformer_subdir", defaults.transformer_subdir),
        vae_subdir=raw.get("vae_subdir", defaults.vae_subdir),
        text_encoder_subdir=raw.get(
            "text_encoder_subdir",
            defaults.text_encoder_subdir,
        ),
        tokenizer_subdir=raw.get("tokenizer_subdir", defaults.tokenizer_subdir),
        max_text_tokens=raw.get("max_text_tokens", defaults.max_text_tokens),
        load_wan_vae_frontend=load_wan_vae_frontend,
        load_text_conditioning=load_text_conditioning,
        load_reference_core_weights=load_reference_core_weights,
        reference_core_init_mode=coerce_enum(
            ReferenceCoreInitMode,
            raw.get("reference_core_init_mode", defaults.reference_core_init_mode),
        ),
        reference_norm2_source_path=raw.get(
            "reference_norm2_source_path",
            defaults.reference_norm2_source_path,
        ),
        exported_runtime_action_init_mode=coerce_enum(
            ExportedRuntimeActionInitMode,
            raw.get(
                "exported_runtime_action_init_mode",
                defaults.exported_runtime_action_init_mode,
            ),
        ),
        reference_assets_device_policy=coerce_enum(
            ReferenceAssetsDevicePolicy,
            raw.get(
                "reference_assets_device_policy",
                defaults.reference_assets_device_policy,
            ),
        ),
        reference_model_path=raw.get("reference_model_path"),
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
            return AttentionMode.FLEX
        return config.attn_mode
    if config.infer_attn_mode is not None:
        return config.infer_attn_mode
    return config.attn_mode


__all__ = [
    "LingbotCompatibleVideoBackboneConfig",
    "SharedVideoTransformerConfig",
    "normalize_backbone_implementation",
    "parse_shared_video_transformer_config",
    "resolve_stage_attention_mode",
]
