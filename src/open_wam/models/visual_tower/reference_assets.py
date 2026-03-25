from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import AutoencoderKLWan

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig

from .reference_loader import resolve_pretrained_component_dir


def _load_transformers_assets() -> tuple[type[Any], type[Any]]:
    try:
        from transformers import T5TokenizerFast, UMT5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required to load LingBot text-conditioning assets. "
            "Install it before setting `backbone.load_text_conditioning=true`."
        ) from exc
    return T5TokenizerFast, UMT5EncoderModel


def _patchify(x: torch.Tensor, patch_size: int | None) -> torch.Tensor:
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


class WanVAEStreamingWrapper:
    def __init__(self, vae_model: AutoencoderKLWan) -> None:
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for module in self.encoder.modules():
                if module.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self) -> None:
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = _patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        return self.quant_conv(out)


@dataclass
class LingbotReferenceAssets:
    config: LingbotCompatibleVideoBackboneConfig
    vae: AutoencoderKLWan | None = None
    streaming_vae: WanVAEStreamingWrapper | None = None
    text_encoder: Any | None = None
    tokenizer: Any | None = None

    @classmethod
    def maybe_load(cls, config: LingbotCompatibleVideoBackboneConfig) -> "LingbotReferenceAssets":
        assets = cls(config=config)
        pretrained_root = config.pretrained_model_name_or_path
        if pretrained_root is None:
            return assets

        if config.load_wan_vae_frontend:
            vae_dir = resolve_pretrained_component_dir(pretrained_root, config.vae_subdir)
            if vae_dir is not None and vae_dir.exists():
                assets.vae = AutoencoderKLWan.from_pretrained(str(vae_dir))
                assets.streaming_vae = WanVAEStreamingWrapper(assets.vae)

        if config.load_text_conditioning:
            tokenizer_cls, text_encoder_cls = _load_transformers_assets()
            text_encoder_dir = resolve_pretrained_component_dir(pretrained_root, config.text_encoder_subdir)
            tokenizer_dir = resolve_pretrained_component_dir(pretrained_root, config.tokenizer_subdir)
            if text_encoder_dir is not None and text_encoder_dir.exists():
                assets.text_encoder = text_encoder_cls.from_pretrained(str(text_encoder_dir))
                assets.text_encoder.eval()
                for parameter in assets.text_encoder.parameters():
                    parameter.requires_grad = False
            if tokenizer_dir is not None and tokenizer_dir.exists():
                assets.tokenizer = tokenizer_cls.from_pretrained(str(tokenizer_dir))
        if assets.vae is not None:
            assets.vae.eval()
            for parameter in assets.vae.parameters():
                parameter.requires_grad = False
        return assets

    @property
    def has_vae(self) -> bool:
        return self.vae is not None and self.streaming_vae is not None

    @property
    def has_text_encoder(self) -> bool:
        return self.text_encoder is not None and self.tokenizer is not None

    def reset_runtime_state(self) -> None:
        if self.streaming_vae is not None:
            self.streaming_vae.clear_cache()

    def encode_text(
        self,
        task_text: tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.has_text_encoder:
            return None
        prompts = [text or "" for text in (task_text or tuple())]
        if not prompts:
            return None
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.config.max_text_tokens,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        encoder_device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(encoder_device),
                attention_mask.to(encoder_device),
            ).last_hidden_state
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        return torch.stack(
            [
                torch.cat(
                    [embedding[:seq_len], embedding.new_zeros(self.config.max_text_tokens - seq_len, embedding.shape[1])],
                    dim=0,
                )
                for embedding, seq_len in zip(prompt_embeds, seq_lens.tolist(), strict=True)
            ],
            dim=0,
        )

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        reset_cache: bool = True,
    ) -> torch.Tensor:
        if not self.has_vae:
            raise RuntimeError("Wan VAE assets are not loaded for LingBot reference frontend.")

        if self._matches_robotwin_layout(placements, canonical_video):
            top = placements[0]
            left = placements[1]
            right = placements[2]
            high_video = canonical_video[
                :,
                :,
                :,
                top.top : top.top + top.height,
                top.left : top.left + top.width,
            ]
            left_video = canonical_video[
                :,
                :,
                :,
                left.top : left.top + left.height,
                left.left : left.left + left.width,
            ]
            right_video = canonical_video[
                :,
                :,
                :,
                right.top : right.top + right.height,
                right.left : right.left + right.width,
            ]
            high_latent = self._encode_chunk(high_video, reset_cache=reset_cache)
            wrist_latent_left = self._encode_chunk(left_video, reset_cache=reset_cache)
            wrist_latent_right = self._encode_chunk(right_video, reset_cache=reset_cache)
            wrist_latent = torch.cat([wrist_latent_left, wrist_latent_right], dim=-1)
            return torch.cat([high_latent, wrist_latent], dim=-2)

        return self._encode_chunk(canonical_video, reset_cache=reset_cache)

    def _encode_chunk(self, video: torch.Tensor, *, reset_cache: bool = True) -> torch.Tensor:
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        scaled = video.to(device=vae_device, dtype=vae_dtype) * 2.0 - 1.0
        if reset_cache:
            self.streaming_vae.clear_cache()
        with torch.no_grad():
            enc_out = self.streaming_vae.encode_chunk(scaled)
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
        normalized = (mu.float() - latents_mean.float()) * (1.0 / latents_std.float())
        return normalized.to(device=video.device, dtype=video.dtype)

    def _matches_robotwin_layout(
        self,
        placements: tuple[ViewPlacement, ...] | None,
        canonical_video: torch.Tensor,
    ) -> bool:
        if placements is None or len(placements) != 3:
            return False
        names = tuple(placement.name for placement in placements)
        expected_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        if names != expected_names:
            return False
        height = canonical_video.shape[-2]
        width = canonical_video.shape[-1]
        return (height, width) == (384, 320)
