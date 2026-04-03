from __future__ import annotations

import torch

from open_wam.configs import ReferenceCoreInitMode
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig

from .reference_loader import load_wan_transformer_class, resolve_pretrained_component_dir


def preferred_reference_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16


def build_reference_transformer(
    backbone_config: SharedVideoTransformerConfig,
    *,
    action_dim: int,
) -> torch.nn.Module:
    model_cls = load_wan_transformer_class(backbone_config)
    preferred_dtype = preferred_reference_dtype(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    transformer_dir = resolve_pretrained_component_dir(
        backbone_config.pretrained_model_name_or_path,
        backbone_config.transformer_subdir,
    )
    if transformer_dir is not None and transformer_dir.exists():
        init_mode = getattr(backbone_config, "reference_core_init_mode", ReferenceCoreInitMode.FULL)
        if init_mode == ReferenceCoreInitMode.VIDEO_ONLY:
            # Video-only init only needs the checkpoint-native reference model
            # so we load it with its original config/action dimensions and copy
            # the shared/video weights out later.
            return model_cls.from_pretrained(
                str(transformer_dir),
                torch_dtype=preferred_dtype,
            )
        load_kwargs = {
            "torch_dtype": preferred_dtype,
            "action_dim": action_dim,
        }
        return model_cls.from_pretrained(
            str(transformer_dir),
            **load_kwargs,
        )
    attention_head_dim = backbone_config.attention_head_dim or (backbone_config.hidden_size // backbone_config.num_heads)
    return model_cls(
        patch_size=[backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w],
        num_attention_heads=backbone_config.num_heads,
        attention_head_dim=attention_head_dim,
        in_channels=backbone_config.latent_channels,
        out_channels=backbone_config.latent_channels,
        action_dim=action_dim,
        text_dim=backbone_config.text_dim,
        freq_dim=backbone_config.freq_dim,
        ffn_dim=backbone_config.ffn_dim or (backbone_config.hidden_size * backbone_config.mlp_ratio),
        num_layers=backbone_config.num_layers,
        cross_attn_norm=backbone_config.cross_attn_norm,
        eps=backbone_config.latent_norm_eps,
        rope_max_seq_len=backbone_config.rope_max_seq_len,
        attn_mode=backbone_config.attn_mode,
    )
