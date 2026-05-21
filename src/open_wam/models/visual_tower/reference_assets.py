from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan

from open_wam.configs import ReferenceAssetsDevicePolicy
from open_wam.data.raw_video import ViewPlacement
from open_wam.models.common.video_geometry import WAN_TEMPORAL_CHUNK_SIZE, wan_safe_temporal_frame_count
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig

from .reference_loader import resolve_pretrained_component_dir
from .reference_transformer import preferred_reference_dtype


_PLACEHOLDER_PATH_PREFIXES = ("/path/to/", "/path/to", "path/to/")


def _validate_pretrained_root(
    pretrained_root: str,
    *,
    config: LingbotCompatibleVideoBackboneConfig,
) -> None:
    """Fail loud if pretrained_model_name_or_path is the sample placeholder
    (`/path/to/...`) or a non-existent path while reference asset loading is
    requested. Without this guard the loader silently leaves vae / text encoder
    as None, the frontend falls back to a randomly-initialized latentizer, and
    rollouts appear to run but produce N(0,1) noise as video_latents — which
    cascades into wildly wrong actions and a fail rollout."""
    needs_assets = bool(config.load_wan_vae_frontend) or bool(config.load_text_conditioning)
    if not needs_assets:
        return
    root_str = str(pretrained_root)
    if root_str.startswith(_PLACEHOLDER_PATH_PREFIXES):
        raise FileNotFoundError(
            f"backbone.pretrained_model_name_or_path is still the placeholder "
            f"{root_str!r}. This usually means configs/local_paths.yaml was "
            f"never edited from configs/local_paths.sample.yaml, OR the "
            f"checkpoint's resolved_config.yaml has a hard-coded path that "
            f"does not exist on this machine. Edit configs/local_paths.yaml "
            f"(set paths.models.lingbot_va_base) and/or fix the checkpoint's "
            f"resolved_config.yaml before re-running."
        )
    from pathlib import Path as _Path
    if not _Path(root_str).expanduser().exists():
        raise FileNotFoundError(
            f"backbone.pretrained_model_name_or_path={root_str!r} does not "
            f"exist on this machine. Reference assets (VAE / text encoder) "
            f"would silently be skipped, leaving the frontend's latentizer "
            f"to produce random N(0,1) latents — making the rollout look "
            f"like a soft failure (action=garbage, gripper sign random) "
            f"instead of a hard error. Either: copy the assets locally and "
            f"update configs/local_paths.yaml, or fix the checkpoint's "
            f"resolved_config.yaml to point at an existing path."
        )


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


def _wan_safe_frame_count(num_frames: int, *, cache_initialized: bool) -> int:
    return wan_safe_temporal_frame_count(num_frames, cache_initialized=cache_initialized)


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
        if x_chunk.ndim != 5:
            raise ValueError(f"Expected Wan VAE input [B,C,T,H,W], got {tuple(x_chunk.shape)}.")
        cache_initialized = any(value is not None for value in self.feat_cache)
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = _patchify(x_chunk, self.vae.config.patch_size)

        outputs: list[torch.Tensor] = []
        chunk_ranges = self._stream_chunk_ranges(int(x_chunk.shape[2]), cache_initialized=cache_initialized)
        if not chunk_ranges:
            raise ValueError(
                "Streaming Wan VAE chunks after cache warmup must contain at least one complete "
                f"{WAN_TEMPORAL_CHUNK_SIZE}-frame group; got {int(x_chunk.shape[2])} frames."
            )
        for start, end in chunk_ranges:
            feat_idx = [0]
            outputs.append(self.encoder(x_chunk[:, :, start:end], feat_cache=self.feat_cache, feat_idx=feat_idx))
        out = torch.cat(outputs, dim=2)
        return self.quant_conv(out)

    @staticmethod
    def _stream_chunk_ranges(num_frames: int, *, cache_initialized: bool) -> tuple[tuple[int, int], ...]:
        if num_frames <= 0:
            raise ValueError(f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}.")
        consumed_frames = _wan_safe_frame_count(num_frames, cache_initialized=cache_initialized)
        if cache_initialized:
            return tuple(
                (start, start + WAN_TEMPORAL_CHUNK_SIZE)
                for start in range(0, consumed_frames, WAN_TEMPORAL_CHUNK_SIZE)
            )
        ranges = [(0, 1)]
        ranges.extend(
            (start, start + WAN_TEMPORAL_CHUNK_SIZE)
            for start in range(1, consumed_frames, WAN_TEMPORAL_CHUNK_SIZE)
        )
        return tuple(ranges)


@dataclass
class LingbotReferenceAssets:
    config: LingbotCompatibleVideoBackboneConfig
    vae: AutoencoderKLWan | None = None
    streaming_vae: WanVAEStreamingWrapper | None = None
    streaming_vae_by_key: dict[str, WanVAEStreamingWrapper] = field(default_factory=dict)
    text_encoder: Any | None = None
    tokenizer: Any | None = None

    @classmethod
    def maybe_load(cls, config: LingbotCompatibleVideoBackboneConfig) -> "LingbotReferenceAssets":
        assets = cls(config=config)
        pretrained_root = config.pretrained_model_name_or_path
        if pretrained_root is None:
            return assets

        _validate_pretrained_root(pretrained_root, config=config)

        reference_dtype = torch.bfloat16

        if config.load_wan_vae_frontend:
            vae_dir = resolve_pretrained_component_dir(pretrained_root, config.vae_subdir)
            if vae_dir is None or not vae_dir.exists():
                raise FileNotFoundError(
                    f"backbone.load_wan_vae_frontend=True but the VAE component directory "
                    f"could not be resolved under pretrained_model_name_or_path="
                    f"{pretrained_root!r} (looked for subdir {config.vae_subdir!r}; "
                    f"resolved to {vae_dir}). Update configs/local_paths.yaml or the "
                    f"checkpoint's resolved_config.yaml so this path actually exists. "
                    f"Without it, the frontend silently falls back to a randomly-initialized "
                    f"latentizer producing N(0,1) noise, which makes downstream rollouts "
                    f"appear to 'run' but with wildly wrong actions."
                )
            assets.vae = AutoencoderKLWan.from_pretrained(
                str(vae_dir),
                torch_dtype=reference_dtype,
            )
            assets.streaming_vae = WanVAEStreamingWrapper(assets.vae)

        if config.load_text_conditioning:
            tokenizer_cls, text_encoder_cls = _load_transformers_assets()
            text_encoder_dir = resolve_pretrained_component_dir(pretrained_root, config.text_encoder_subdir)
            tokenizer_dir = resolve_pretrained_component_dir(pretrained_root, config.tokenizer_subdir)
            if text_encoder_dir is None or not text_encoder_dir.exists():
                raise FileNotFoundError(
                    f"backbone.load_text_conditioning=True but the text encoder directory "
                    f"could not be resolved under pretrained_model_name_or_path="
                    f"{pretrained_root!r} (looked for subdir {config.text_encoder_subdir!r}; "
                    f"resolved to {text_encoder_dir})."
                )
            if tokenizer_dir is None or not tokenizer_dir.exists():
                raise FileNotFoundError(
                    f"backbone.load_text_conditioning=True but the tokenizer directory "
                    f"could not be resolved under pretrained_model_name_or_path="
                    f"{pretrained_root!r} (looked for subdir {config.tokenizer_subdir!r}; "
                    f"resolved to {tokenizer_dir})."
                )
            assets.text_encoder = text_encoder_cls.from_pretrained(
                str(text_encoder_dir),
                torch_dtype=reference_dtype,
            )
            assets.text_encoder.eval()
            for parameter in assets.text_encoder.parameters():
                parameter.requires_grad = False
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
        for streaming_vae in self.streaming_vae_by_key.values():
            streaming_vae.clear_cache()

    def encode_text(
        self,
        task_text: tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        prompts = [text or "" for text in (task_text or tuple())]
        return self.encode_prompts(prompts, device=device, dtype=dtype)

    def encode_blank_text(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if batch_size <= 0:
            return None
        return self.encode_prompts([""] * batch_size, device=device, dtype=dtype)

    def encode_prompts(
        self,
        prompts: list[str] | tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.has_text_encoder:
            return None
        if not prompts:
            return None
        self._ensure_text_encoder_runtime_device(device)
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
        self._ensure_vae_runtime_device(canonical_video.device)

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
            high_video = self._resize_rgb_chunk(high_video, top.height, top.width)
            left_video = canonical_video[
                :,
                :,
                :,
                left.top : left.top + left.height,
                left.left : left.left + left.width,
            ]
            left_video = self._resize_rgb_chunk(left_video, left.height, left.width)
            right_video = canonical_video[
                :,
                :,
                :,
                right.top : right.top + right.height,
                right.left : right.left + right.width,
            ]
            right_video = self._resize_rgb_chunk(right_video, right.height, right.width)
            high_latent = self._encode_chunk(high_video, reset_cache=reset_cache, cache_key="robotwin:cam_high")
            wrist_latent_left = self._encode_chunk(
                left_video,
                reset_cache=reset_cache,
                cache_key="robotwin:cam_left_wrist",
            )
            wrist_latent_right = self._encode_chunk(
                right_video,
                reset_cache=reset_cache,
                cache_key="robotwin:cam_right_wrist",
            )
            wrist_latent = torch.cat([wrist_latent_left, wrist_latent_right], dim=-1)
            return torch.cat([high_latent, wrist_latent], dim=-2)

        if self._matches_libero_layout(placements, canonical_video):
            agentview = placements[0]
            wrist = placements[1]
            agentview_video = canonical_video[
                :,
                :,
                :,
                agentview.top : agentview.top + agentview.height,
                agentview.left : agentview.left + agentview.width,
            ]
            agentview_video = self._resize_rgb_chunk(agentview_video, agentview.height, agentview.width)
            wrist_video = canonical_video[
                :,
                :,
                :,
                wrist.top : wrist.top + wrist.height,
                wrist.left : wrist.left + wrist.width,
            ]
            wrist_video = self._resize_rgb_chunk(wrist_video, wrist.height, wrist.width)
            batch_size = canonical_video.shape[0]
            encoded = self._encode_chunk(
                torch.cat([agentview_video, wrist_video], dim=0),
                reset_cache=reset_cache,
            )
            agentview_latent, wrist_latent = encoded.split(batch_size, dim=0)
            return torch.cat([agentview_latent, wrist_latent], dim=-1)

        return self._encode_chunk(canonical_video, reset_cache=reset_cache)

    def _encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        # Match Heng's reference path exactly: normalize RGB to [-1, 1] in
        # float32 first, then cast to the VAE runtime dtype. Doing the math
        # directly in bf16 perturbs the conditioned first-frame latent enough
        # to break exact rollout parity.
        scaled = (video.to(device=vae_device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=vae_dtype)
        streaming_vae = self._streaming_vae_for_key(cache_key)
        if reset_cache:
            streaming_vae.clear_cache()
        with torch.no_grad():
            enc_out = streaming_vae.encode_chunk(scaled)
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        normalized = self._normalize_reference_latents(mu)
        return normalized.to(device=video.device)

    def _streaming_vae_for_key(self, cache_key: str | None) -> WanVAEStreamingWrapper:
        if self.vae is None:
            raise RuntimeError("Wan VAE assets are not loaded for LingBot reference frontend.")
        if cache_key is None:
            if self.streaming_vae is None:
                self.streaming_vae = WanVAEStreamingWrapper(self.vae)
            return self.streaming_vae
        streaming_vae = self.streaming_vae_by_key.get(cache_key)
        if streaming_vae is None or streaming_vae.vae is not self.vae:
            streaming_vae = WanVAEStreamingWrapper(self.vae)
            self.streaming_vae_by_key[cache_key] = streaming_vae
        return streaming_vae

    def _normalize_reference_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
        return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents)

    def _ensure_vae_runtime_device(self, device: torch.device) -> None:
        if self.vae is None or self.streaming_vae is None or not isinstance(self.vae, torch.nn.Module):
            return
        target_device = self._resolve_reference_runtime_device(device)
        target_dtype = self._reference_asset_runtime_dtype(self.vae, target_device=target_device)
        if not self._module_matches_runtime(self.vae, device=target_device, dtype=target_dtype):
            self.vae = self.vae.to(device=target_device, dtype=target_dtype)
            self.streaming_vae = WanVAEStreamingWrapper(self.vae)
            self.streaming_vae_by_key.clear()

    def _ensure_text_encoder_runtime_device(self, device: torch.device) -> None:
        if self.text_encoder is None or not isinstance(self.text_encoder, torch.nn.Module):
            return
        target_device = self._resolve_reference_runtime_device(device)
        target_dtype = self._reference_asset_runtime_dtype(self.text_encoder, target_device=target_device)
        if not self._module_matches_runtime(self.text_encoder, device=target_device, dtype=target_dtype):
            self.text_encoder = self.text_encoder.to(device=target_device, dtype=target_dtype)

    def _resolve_reference_runtime_device(self, device: torch.device) -> torch.device:
        policy = getattr(self.config, "reference_assets_device_policy", ReferenceAssetsDevicePolicy.RUNTIME)
        if policy == ReferenceAssetsDevicePolicy.CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _reference_asset_runtime_dtype(module: torch.nn.Module, *, target_device: torch.device) -> torch.dtype:
        try:
            current_dtype = next(module.parameters()).dtype
        except StopIteration:
            current_dtype = preferred_reference_dtype(target_device)
        # Heng keeps CPU-offloaded VAE/text assets in their checkpoint dtype
        # (bf16 for the released Wan/LingBot assets) instead of upcasting them
        # to fp32 when they live on CPU.
        if target_device.type == "cpu":
            return current_dtype
        return preferred_reference_dtype(target_device)

    @staticmethod
    def _module_matches_runtime(module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype) -> bool:
        for parameter in module.parameters():
            if parameter.device != device or parameter.dtype != dtype:
                return False
        for buffer in module.buffers():
            if buffer.device != device:
                return False
        return True

    def _resize_rgb_chunk(
        self,
        video: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        batch_size, channels, num_frames, _, _ = video.shape
        flattened = video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, video.shape[-2], video.shape[-1])
        resized = F.interpolate(
            flattened,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, num_frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)

    def _matches_robotwin_layout(
        self,
        placements: tuple[ViewPlacement, ...] | None,
        canonical_video: torch.Tensor,
    ) -> bool:
        if placements is None or len(placements) != 3:
            return False
        names = tuple(placement.canonical_name for placement in placements)
        expected_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        if names != expected_names:
            return False
        height = canonical_video.shape[-2]
        width = canonical_video.shape[-1]
        return (height, width) == (384, 320)

    def _matches_libero_layout(
        self,
        placements: tuple[ViewPlacement, ...] | None,
        canonical_video: torch.Tensor,
    ) -> bool:
        if placements is None or len(placements) != 2:
            return False
        names = tuple(placement.canonical_name for placement in placements)
        if names != ("image", "wrist_image"):
            return False
        height = canonical_video.shape[-2]
        width = canonical_video.shape[-1]
        if (height, width) != (128, 256):
            return False
        agentview, wrist = placements
        return (
            agentview.top,
            agentview.left,
            agentview.height,
            agentview.width,
            wrist.top,
            wrist.left,
            wrist.height,
            wrist.width,
        ) == (0, 0, 128, 128, 0, 128, 128, 128)
