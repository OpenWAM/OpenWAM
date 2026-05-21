from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from open_wam.data.raw_video import ViewPlacement
from open_wam.configs.enums import serialize_enum_values
from open_wam.models.common.video_geometry import video_token_grid_from_latent_shape
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.video_backbone.contracts import ChunkMetadata, ConditioningState, TokenGridMetadata

from .contracts import VisualFrontendOutput
from .reference_assets import LingbotReferenceAssets


class SharedVideoFrontend(nn.Module):
    """Canonical RGB -> latent -> token visual frontend."""

    def __init__(self, config: SharedVideoTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedVideoTransformerConfig()
        self.reference_assets = LingbotReferenceAssets.maybe_load(self.config)
        self.latentizer = nn.Conv3d(
            in_channels=self.config.input_channels,
            out_channels=self.config.latent_channels,
            kernel_size=(1, self.config.latent_stride, self.config.latent_stride),
            stride=(1, self.config.latent_stride, self.config.latent_stride),
        )
        self.latent_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=self.config.latent_channels,
            eps=self.config.latent_norm_eps,
        )
        patch_dim = (
            self.config.latent_channels
            * self.config.patch_size_t
            * self.config.patch_size_h
            * self.config.patch_size_w
        )
        self.token_embed = nn.Linear(patch_dim, self.config.hidden_size)

    def forward(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        preserve_stream_cache: bool = False,
    ) -> VisualFrontendOutput:
        if canonical_video.ndim != 5:
            raise ValueError(
                "Expected canonical video of shape [B, 3, T, H, W], "
                f"got {tuple(canonical_video.shape)}"
            )
        video_latents = self.encode_video(
            canonical_video,
            placements=placements,
            reset_reference_cache=not preserve_stream_cache,
        )
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = self.reference_assets.encode_text(
                task_text,
                device=canonical_video.device,
                dtype=canonical_video.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if resolved_negative_text_context is None and resolved_text_context is not None:
            resolved_negative_text_context = self.reference_assets.encode_blank_text(
                batch_size=canonical_video.shape[0],
                device=canonical_video.device,
                dtype=canonical_video.dtype,
            )
        return self._build_output(
            canonical_video=canonical_video,
            video_latents=video_latents,
            input_source="canonical_rgb",
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
        )

    def from_video_latents(
        self,
        video_latents: torch.Tensor,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
    ) -> VisualFrontendOutput:
        if video_latents.ndim != 5:
            raise ValueError(
                "Expected shared-backbone video latents of shape [B, C, T, H, W], "
                f"got {tuple(video_latents.shape)}"
            )
        canonical = canonical_video
        if canonical is None:
            canonical = video_latents.new_zeros(
                video_latents.shape[0],
                self.config.input_channels,
                video_latents.shape[2],
                video_latents.shape[3] * self.config.latent_stride,
                video_latents.shape[4] * self.config.latent_stride,
            )
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = self.reference_assets.encode_text(
                task_text,
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if resolved_negative_text_context is None and resolved_text_context is not None:
            resolved_negative_text_context = self.reference_assets.encode_blank_text(
                batch_size=video_latents.shape[0],
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
        return self._build_output(
            canonical_video=canonical,
            video_latents=video_latents,
            input_source="video_latents",
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
        )

    def reset_runtime_state(self) -> None:
        self.reference_assets.reset_runtime_state()

    def _build_output(
        self,
        *,
        canonical_video: torch.Tensor,
        video_latents: torch.Tensor,
        input_source: str,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
    ) -> VisualFrontendOutput:
        video_tokens, token_grid = self.tokenize_video_latents(video_latents)
        metadata = {
            "backbone_config": serialize_enum_values(asdict(self.config)),
            "video_frame_mapping": self._video_frame_mapping(
                canonical_video=canonical_video,
                video_latents=video_latents,
            ),
        }
        return VisualFrontendOutput(
            canonical_video=canonical_video,
            video_latents=video_latents,
            video_tokens=video_tokens,
            input_source=str(input_source),
            token_grid=token_grid,
            chunk=ChunkMetadata(
                chunk_start_frame=0,
                chunk_num_frames=video_latents.shape[2],
                frame_stride=1,
                chunk_type="dense_video_chunk",
            ),
            conditioning=ConditioningState(
                supported=text_context is not None,
                text_context=text_context,
                negative_text_context=negative_text_context,
                first_frame_context=video_latents[:, :, :1],
                metadata=metadata,
            ),
        )

    def _video_frame_mapping(
        self,
        *,
        canonical_video: torch.Tensor,
        video_latents: torch.Tensor,
    ) -> dict[str, int | str]:
        raw_frames = int(canonical_video.shape[2])
        latent_frames = int(video_latents.shape[2])
        if raw_frames == latent_frames:
            return {
                "kind": "identity",
                "raw_frames": raw_frames,
                "latent_frames": latent_frames,
            }
        if self.reference_assets.has_vae:
            return {
                "kind": "wan_temporal_downsample",
                "raw_frames": raw_frames,
                "latent_frames": latent_frames,
            }
        return {
            "kind": "unknown_temporal_mapping",
            "raw_frames": raw_frames,
            "latent_frames": latent_frames,
        }

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        reset_reference_cache: bool = True,
    ) -> torch.Tensor:
        if self.reference_assets.has_vae:
            return self.reference_assets.encode_video(
                canonical_video,
                placements=placements,
                reset_cache=reset_reference_cache,
            )
        latents = self.latentizer(canonical_video)
        return self.latent_norm(latents)

    def tokenize_video_latents(self, video_latents: torch.Tensor) -> tuple[torch.Tensor, TokenGridMetadata]:
        batch_size, channels, num_frames, latent_height, latent_width = video_latents.shape
        patch_size = (
            self.config.patch_size_t,
            self.config.patch_size_h,
            self.config.patch_size_w,
        )
        patch_t, patch_h, patch_w = patch_size
        if num_frames % patch_t != 0 or latent_height % patch_h != 0 or latent_width % patch_w != 0:
            raise ValueError(
                "Latent tensor must be divisible by patch size. "
                f"latents={tuple(video_latents.shape)}, patch={patch_size}"
            )
        patches = (
            video_latents
            .view(
                batch_size,
                channels,
                num_frames // patch_t,
                patch_t,
                latent_height // patch_h,
                patch_h,
                latent_width // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch_size, -1, channels * patch_t * patch_h * patch_w)
        )
        patches = patches.to(dtype=self.token_embed.weight.dtype)
        tokens = self.token_embed(patches)
        token_grid = video_token_grid_from_latent_shape(
            video_latents,
            patch_size=patch_size,
        )
        return tokens, token_grid


LingbotVisualFrontend = SharedVideoFrontend
