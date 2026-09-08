from __future__ import annotations

import torch

from open_wam.configs import ReferenceCoreInitMode
from open_wam.configs.backbone import SharedVideoTransformerConfig

from .reference_loader import load_wan_transformer_class, resolve_runtime_backbone_dir


def preferred_reference_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16


def _requested_export_failure_message(
    *,
    backbone_config: SharedVideoTransformerConfig,
    transformer_dir: object,
    reason: str,
) -> str:
    """Keep requested artifact failures distinct from intentional random init."""
    return (
        f"Requested backbone warm start is unusable: {reason}\n"
        f"  looked for: {transformer_dir}\n"
        f"  backbone.runtime_backbone_artifact_path: {backbone_config.runtime_backbone_artifact_path}\n"
        f"  backbone.pretrained_model_name_or_path: {backbone_config.pretrained_model_name_or_path}\n"
        f"  backbone.transformer_subdir: {backbone_config.transformer_subdir}\n"
        "A checkpoint saved with `trainer.export_runtime_backbone: false` may have "
        "model weights but no transformer export. Export the runtime backbone or "
        "set `backbone.runtime_backbone_artifact_path` to a complete export. "
        "Remove the requested artifact only if random initialization was intended."
    )


def build_reference_transformer(
    backbone_config: SharedVideoTransformerConfig,
    *,
    action_dim: int,
) -> torch.nn.Module:
    model_cls = load_wan_transformer_class(backbone_config)
    preferred_dtype = preferred_reference_dtype(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    transformer_dir = resolve_runtime_backbone_dir(backbone_config)
    if transformer_dir is not None and transformer_dir.exists():
        init_mode = getattr(backbone_config, "reference_core_init_mode", ReferenceCoreInitMode.FULL)
        try:
            if init_mode == ReferenceCoreInitMode.VIDEO_ONLY:
                # Retain checkpoint-native action dimensions for video-only init.
                return model_cls.from_pretrained(
                    str(transformer_dir),
                    torch_dtype=preferred_dtype,
                )
            return model_cls.from_pretrained(
                str(transformer_dir),
                torch_dtype=preferred_dtype,
                action_dim=action_dim,
            )
        except Exception as error:
            raise RuntimeError(
                _requested_export_failure_message(
                    backbone_config=backbone_config,
                    transformer_dir=transformer_dir,
                    reason=f"{type(error).__name__} while loading it -- {error}",
                )
            ) from error
    if transformer_dir is not None:
        # Covers canonical detached artifacts, historical model roots, and
        # explicit absolute component paths without reviving ambiguous resume.
        raise FileNotFoundError(
            _requested_export_failure_message(
                backbone_config=backbone_config,
                transformer_dir=transformer_dir,
                reason="the resolved transformer directory does not exist",
            )
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
