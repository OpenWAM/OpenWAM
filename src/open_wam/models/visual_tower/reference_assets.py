from __future__ import annotations
from pathlib import Path

import copy
import importlib.metadata
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan

from open_wam.configs import ReferenceAssetsDevicePolicy
from open_wam.configs.backbone import LingbotCompatibleVideoBackboneConfig
from open_wam.contracts import (
    VideoLatentSpaceIdentity,
    ViewPlacement,
    identify_video_latent_space,
)
from open_wam.models.common.video_geometry import (
    WAN_TEMPORAL_CHUNK_SIZE,
    wan_safe_temporal_frame_count,
)

from .reference_loader import resolve_pretrained_component_dir
from .prompt_cache import OfflinePromptCache
from .reference_transformer import preferred_reference_dtype

_PLACEHOLDER_PATH_PREFIXES = ("/path/to/", "/path/to", "path/to/")
_WAN_LATENT_ENCODING_CONTRACT = "open_wam.wan_vae_latents.v1"


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
    needs_assets = bool(config.load_wan_vae_frontend) or bool(
        config.load_text_conditioning
    )
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
    return wan_safe_temporal_frame_count(
        num_frames, cache_initialized=cache_initialized
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

    def snapshot_cache(self) -> list[Any]:
        """Copy causal encoder features for speculative execution."""

        return copy.deepcopy(self.feat_cache)

    def restore_cache(self, snapshot: list[Any]) -> None:
        """Restore a previously copied causal encoder cache."""

        self.feat_cache = copy.deepcopy(snapshot)

    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        if x_chunk.ndim != 5:
            raise ValueError(
                f"Expected Wan VAE input [B,C,T,H,W], got {tuple(x_chunk.shape)}."
            )
        cache_initialized = any(value is not None for value in self.feat_cache)
        if (
            hasattr(self.vae.config, "patch_size")
            and self.vae.config.patch_size is not None
        ):
            x_chunk = _patchify(x_chunk, self.vae.config.patch_size)

        outputs: list[torch.Tensor] = []
        chunk_ranges = self._stream_chunk_ranges(
            int(x_chunk.shape[2]), cache_initialized=cache_initialized
        )
        if not chunk_ranges:
            raise ValueError(
                "Streaming Wan VAE chunks after cache warmup must contain at least one complete "
                f"{WAN_TEMPORAL_CHUNK_SIZE}-frame group; got {int(x_chunk.shape[2])} frames."
            )
        for start, end in chunk_ranges:
            feat_idx = [0]
            outputs.append(
                self.encoder(
                    x_chunk[:, :, start:end],
                    feat_cache=self.feat_cache,
                    feat_idx=feat_idx,
                )
            )
        out = torch.cat(outputs, dim=2)
        return self.quant_conv(out)

    @staticmethod
    def _stream_chunk_ranges(
        num_frames: int, *, cache_initialized: bool
    ) -> tuple[tuple[int, int], ...]:
        if num_frames <= 0:
            raise ValueError(
                f"Wan VAE encoding requires at least one frame, got num_frames={num_frames}."
            )
        consumed_frames = _wan_safe_frame_count(
            num_frames, cache_initialized=cache_initialized
        )
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


@dataclass(frozen=True)
class ReferenceAssetsRuntimeSnapshot:
    """Streaming frontend state copied independently of model parameters."""

    default_streaming_vae_cache: list[Any] | None = None
    keyed_streaming_vae_caches: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class LingbotReferenceAssets:
    config: LingbotCompatibleVideoBackboneConfig
    vae: AutoencoderKLWan | None = None
    streaming_vae: WanVAEStreamingWrapper | None = None
    streaming_vae_by_key: dict[str, WanVAEStreamingWrapper] = field(
        default_factory=dict
    )
    text_encoder: Any | None = None
    tokenizer: Any | None = None
    text_embedding_cache: dict[tuple[tuple[str, ...], str, str, int], torch.Tensor] = (
        field(default_factory=dict)
    )
    latent_space_identity: VideoLatentSpaceIdentity | None = None
    offline_prompt_cache: OfflinePromptCache | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.config.prompt_cache is not None:
            from open_wam.artifacts.resolver import ArtifactResolver

            cache = self.config.prompt_cache
            self.offline_prompt_cache = OfflinePromptCache(
                Path(cache.root), max_text_tokens=self.config.max_text_tokens,
                text_dim=self.config.text_dim,
                expected_encoder_fingerprint=cache.encoder_fingerprint,
                resolver=ArtifactResolver(cache.artifact_cache),
            )

    @classmethod
    def maybe_load(
        cls, config: LingbotCompatibleVideoBackboneConfig
    ) -> LingbotReferenceAssets:
        assets = cls(config=config)
        pretrained_root = config.pretrained_model_name_or_path
        if pretrained_root is None:
            return assets

        asset_config = (
            replace(config, load_text_conditioning=False)
            if assets.offline_prompt_cache is not None else config
        )
        _validate_pretrained_root(pretrained_root, config=asset_config)

        reference_dtype = torch.bfloat16

        if config.load_wan_vae_frontend:
            vae_dir = resolve_pretrained_component_dir(
                pretrained_root, config.vae_subdir
            )
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
            assets.latent_space_identity = identify_video_latent_space(
                vae_dir,
                encoder_family=(
                    f"{AutoencoderKLWan.__module__}.{AutoencoderKLWan.__qualname__}"
                    f"@{importlib.metadata.version('diffusers')}"
                ),
                encoding_contract=_WAN_LATENT_ENCODING_CONTRACT,
            )
            assets.vae = AutoencoderKLWan.from_pretrained(
                str(vae_dir),
                torch_dtype=reference_dtype,
            )
            assets.streaming_vae = WanVAEStreamingWrapper(assets.vae)

        if config.load_text_conditioning and assets.offline_prompt_cache is None:
            tokenizer_cls, text_encoder_cls = _load_transformers_assets()
            text_encoder_dir = resolve_pretrained_component_dir(
                pretrained_root, config.text_encoder_subdir
            )
            tokenizer_dir = resolve_pretrained_component_dir(
                pretrained_root, config.tokenizer_subdir
            )
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

    def snapshot_runtime_state(self) -> ReferenceAssetsRuntimeSnapshot | None:
        default_cache = (
            None if self.streaming_vae is None else self.streaming_vae.snapshot_cache()
        )
        keyed_caches = {
            str(cache_key): streaming_vae.snapshot_cache()
            for cache_key, streaming_vae in self.streaming_vae_by_key.items()
        }
        if default_cache is None and not keyed_caches:
            return None
        return ReferenceAssetsRuntimeSnapshot(
            default_streaming_vae_cache=default_cache,
            keyed_streaming_vae_caches=keyed_caches,
        )

    def restore_runtime_state(
        self,
        snapshot: ReferenceAssetsRuntimeSnapshot | None,
    ) -> None:
        if snapshot is None:
            return
        if snapshot.default_streaming_vae_cache is not None:
            if self.streaming_vae is None and self.vae is not None:
                self.streaming_vae = WanVAEStreamingWrapper(self.vae)
            if self.streaming_vae is not None:
                self.streaming_vae.restore_cache(snapshot.default_streaming_vae_cache)
        elif self.streaming_vae is not None:
            self.streaming_vae.clear_cache()

        snapshot_keys = set(snapshot.keyed_streaming_vae_caches)
        for cache_key in set(self.streaming_vae_by_key) - snapshot_keys:
            self.streaming_vae_by_key.pop(cache_key)
        for cache_key, cache_snapshot in snapshot.keyed_streaming_vae_caches.items():
            streaming_vae = self.streaming_vae_by_key.get(cache_key)
            if streaming_vae is None and self.vae is not None:
                streaming_vae = WanVAEStreamingWrapper(self.vae)
                self.streaming_vae_by_key[cache_key] = streaming_vae
            if streaming_vae is not None:
                streaming_vae.restore_cache(cache_snapshot)

    def encode_text(
        self,
        task_text: tuple[str | None, ...] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        prompts = [text or "" for text in (task_text or ())]
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
        if not prompts:
            return None
        normalized_prompts = tuple(str(prompt) for prompt in prompts)
        if self.offline_prompt_cache is not None:
            return self.offline_prompt_cache.encode_prompts(
                normalized_prompts, device=device, dtype=dtype
            )
        if not self.has_text_encoder:
            return None
        cache_key = (
            normalized_prompts,
            str(torch.device(device)),
            str(dtype),
            int(self.config.max_text_tokens),
        )
        cached = self.text_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        self._ensure_text_encoder_runtime_device(device)
        text_inputs = self.tokenizer(
            normalized_prompts,
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
        encoded = torch.stack(
            [
                torch.cat(
                    [
                        embedding[:seq_len],
                        embedding.new_zeros(
                            self.config.max_text_tokens - seq_len, embedding.shape[1]
                        ),
                    ],
                    dim=0,
                )
                for embedding, seq_len in zip(
                    prompt_embeds, seq_lens.tolist(), strict=True
                )
            ],
            dim=0,
        )
        self.text_embedding_cache[cache_key] = encoded
        return encoded

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        reset_cache: bool = True,
    ) -> torch.Tensor:
        if not self.has_vae:
            raise RuntimeError(
                "Wan VAE assets are not loaded for LingBot reference frontend."
            )
        self._ensure_vae_runtime_device(canonical_video.device)
        if placements and len(placements) > 1:
            return self._encode_placed_views(
                canonical_video,
                placements=placements,
                reset_cache=reset_cache,
            )
        return self._encode_chunk(canonical_video, reset_cache=reset_cache)

    def _encode_placed_views(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...],
        reset_cache: bool,
    ) -> torch.Tensor:
        """Encode each canonical view independently and restore its layout.

        ``ViewPlacement`` is the only layout authority. Equal-resolution views
        are batched through the VAE, matching the established two-view path;
        mixed-resolution views keep independent streaming caches.
        """

        canvas_height, canvas_width = (
            int(canonical_video.shape[-2]),
            int(canonical_video.shape[-1]),
        )
        view_videos = tuple(
            self._crop_placed_view(
                canonical_video,
                placement=placement,
                canvas_height=canvas_height,
                canvas_width=canvas_width,
            )
            for placement in placements
        )
        view_shapes = {
            (int(video.shape[-2]), int(video.shape[-1])) for video in view_videos
        }
        if len(view_shapes) == 1:
            batch_size = int(canonical_video.shape[0])
            encoded = self._encode_chunk(
                torch.cat(view_videos, dim=0),
                reset_cache=reset_cache,
            )
            view_latents = tuple(encoded.split(batch_size, dim=0))
        else:
            view_latents = tuple(
                self._encode_chunk(
                    video,
                    reset_cache=reset_cache,
                    cache_key=f"view:{index}:{placement.canonical_name}",
                )
                for index, (placement, video) in enumerate(
                    zip(placements, view_videos, strict=True)
                )
            )
        return self._assemble_placed_view_latents(
            view_latents,
            placements=placements,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )

    def _crop_placed_view(
        self,
        canonical_video: torch.Tensor,
        *,
        placement: ViewPlacement,
        canvas_height: int,
        canvas_width: int,
    ) -> torch.Tensor:
        top = int(placement.top)
        left = int(placement.left)
        height = int(placement.height)
        width = int(placement.width)
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            raise ValueError(f"Invalid canonical view placement: {placement!r}.")
        if top + height > canvas_height or left + width > canvas_width:
            raise ValueError(
                "Canonical view placement exceeds the RGB canvas, "
                f"got placement={placement!r}, canvas={(canvas_height, canvas_width)}."
            )
        view = canonical_video[
            :,
            :,
            :,
            top : top + height,
            left : left + width,
        ]
        return self._resize_rgb_chunk(view, height, width)

    @staticmethod
    def _assemble_placed_view_latents(
        view_latents: tuple[torch.Tensor, ...],
        *,
        placements: tuple[ViewPlacement, ...],
        canvas_height: int,
        canvas_width: int,
    ) -> torch.Tensor:
        if len(view_latents) != len(placements) or not view_latents:
            raise ValueError(
                "Placed-view latent assembly requires one latent per view."
            )
        first = view_latents[0]
        if first.ndim != 5:
            raise ValueError(
                "Placed-view VAE output must have shape [B, C, T, H, W], "
                f"got {tuple(first.shape)}."
            )
        batch_channels_time = tuple(int(value) for value in first.shape[:3])
        spatial_scales: set[tuple[int, int]] = set()
        for placement, latent in zip(placements, view_latents, strict=True):
            if tuple(int(value) for value in latent.shape[:3]) != batch_channels_time:
                raise ValueError(
                    "Placed-view VAE outputs must share batch, channel, and time dimensions."
                )
            latent_height, latent_width = (
                int(latent.shape[-2]),
                int(latent.shape[-1]),
            )
            if (
                latent_height <= 0
                or latent_width <= 0
                or int(placement.height) % latent_height
                or int(placement.width) % latent_width
            ):
                raise ValueError(
                    "View placement and latent shape do not define an integer spatial scale, "
                    f"got placement={placement!r}, latent={tuple(latent.shape)}."
                )
            spatial_scales.add(
                (
                    int(placement.height) // latent_height,
                    int(placement.width) // latent_width,
                )
            )
        if len(spatial_scales) != 1:
            raise ValueError(
                "All placed views must use the same VAE spatial scale, "
                f"got {sorted(spatial_scales)}."
            )
        scale_h, scale_w = next(iter(spatial_scales))
        if canvas_height % scale_h or canvas_width % scale_w:
            raise ValueError(
                "Canonical RGB canvas is not divisible by the VAE spatial scale, "
                f"got canvas={(canvas_height, canvas_width)}, scale={(scale_h, scale_w)}."
            )
        latent_canvas = first.new_zeros(
            (
                *batch_channels_time,
                canvas_height // scale_h,
                canvas_width // scale_w,
            )
        )
        occupied = torch.zeros(
            latent_canvas.shape[-2:],
            dtype=torch.bool,
            device=latent_canvas.device,
        )
        for placement, latent in zip(placements, view_latents, strict=True):
            if int(placement.top) % scale_h or int(placement.left) % scale_w:
                raise ValueError(
                    "View placement must align to the VAE latent grid, "
                    f"got placement={placement!r}, scale={(scale_h, scale_w)}."
                )
            top = int(placement.top) // scale_h
            left = int(placement.left) // scale_w
            bottom = top + int(latent.shape[-2])
            right = left + int(latent.shape[-1])
            if bool(occupied[top:bottom, left:right].any()):
                raise ValueError(f"Canonical view placements overlap at {placement!r}.")
            latent_canvas[..., top:bottom, left:right] = latent
            occupied[top:bottom, left:right] = True
        return latent_canvas

    def _encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        # Match the LingBot reference path exactly: normalize RGB to [-1, 1] in
        # float32 first, then cast to the VAE runtime dtype. Doing the math
        # directly in bf16 perturbs the conditioned first-frame latent enough
        # to break exact rollout parity.
        scaled = (video.to(device=vae_device, dtype=torch.float32) * 2.0 - 1.0).to(
            dtype=vae_dtype
        )
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
            raise RuntimeError(
                "Wan VAE assets are not loaded for LingBot reference frontend."
            )
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
        latents_mean = torch.tensor(
            self.vae.config.latents_mean, device=latents.device
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std, device=latents.device
        ).view(1, -1, 1, 1, 1)
        return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents)

    def _ensure_vae_runtime_device(self, device: torch.device) -> None:
        if (
            self.vae is None
            or self.streaming_vae is None
            or not isinstance(self.vae, torch.nn.Module)
        ):
            return
        target_device = self._resolve_reference_runtime_device(device)
        target_dtype = self._reference_asset_runtime_dtype(
            self.vae, target_device=target_device
        )
        if not self._module_matches_runtime(
            self.vae, device=target_device, dtype=target_dtype
        ):
            self.vae = self.vae.to(device=target_device, dtype=target_dtype)
            self.streaming_vae = WanVAEStreamingWrapper(self.vae)
            self.streaming_vae_by_key.clear()

    def _ensure_text_encoder_runtime_device(self, device: torch.device) -> None:
        if self.text_encoder is None or not isinstance(
            self.text_encoder, torch.nn.Module
        ):
            return
        target_device = self._resolve_reference_runtime_device(device)
        target_dtype = self._reference_asset_runtime_dtype(
            self.text_encoder, target_device=target_device
        )
        if not self._module_matches_runtime(
            self.text_encoder, device=target_device, dtype=target_dtype
        ):
            self.text_encoder = self.text_encoder.to(
                device=target_device, dtype=target_dtype
            )

    def _resolve_reference_runtime_device(self, device: torch.device) -> torch.device:
        policy = getattr(
            self.config,
            "reference_assets_device_policy",
            ReferenceAssetsDevicePolicy.RUNTIME,
        )
        if policy == ReferenceAssetsDevicePolicy.CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _reference_asset_runtime_dtype(
        module: torch.nn.Module, *, target_device: torch.device
    ) -> torch.dtype:
        try:
            current_dtype = next(module.parameters()).dtype
        except StopIteration:
            current_dtype = preferred_reference_dtype(target_device)
        # The LingBot runtime keeps CPU-offloaded VAE/text assets in checkpoint dtype
        # (bf16 for the released Wan/LingBot assets) instead of upcasting them
        # to fp32 when they live on CPU.
        if target_device.type == "cpu":
            return current_dtype
        return preferred_reference_dtype(target_device)

    @staticmethod
    def _module_matches_runtime(
        module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype
    ) -> bool:
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
        flattened = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, video.shape[-2], video.shape[-1]
        )
        resized = F.interpolate(
            flattened,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(
            batch_size, num_frames, channels, target_height, target_width
        ).permute(0, 2, 1, 3, 4)
