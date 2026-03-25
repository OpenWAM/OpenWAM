from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import ChunkMetadata, ConditioningState, TokenGridMetadata

from .contracts import VisualFrontendOutput


class LingbotVisualFrontend(nn.Module):
    """Canonical RGB -> latent -> token visual frontend."""

    def __init__(self, config: LingbotCompatibleVideoBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
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

    def forward(self, canonical_video: torch.Tensor) -> VisualFrontendOutput:
        if canonical_video.ndim != 5:
            raise ValueError(
                "Expected canonical video of shape [B, 3, T, H, W], "
                f"got {tuple(canonical_video.shape)}"
            )
        video_latents = self.encode_video(canonical_video)
        video_tokens, token_grid = self.tokenize_video_latents(video_latents)
        return VisualFrontendOutput(
            canonical_video=canonical_video,
            video_latents=video_latents,
            video_tokens=video_tokens,
            token_grid=token_grid,
            chunk=ChunkMetadata(
                chunk_start_frame=0,
                chunk_num_frames=canonical_video.shape[2],
                frame_stride=1,
                chunk_type="dense_video_chunk",
            ),
            conditioning=ConditioningState(
                supported=False,
                metadata={"backbone_config": asdict(self.config)},
            ),
        )

    def encode_video(self, canonical_video: torch.Tensor) -> torch.Tensor:
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
        tokens = self.token_embed(patches)
        patches_per_frame_h = latent_height // patch_h
        patches_per_frame_w = latent_width // patch_w
        tokens_per_frame = patches_per_frame_h * patches_per_frame_w
        token_grid = TokenGridMetadata(
            num_frames=num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            patch_size=patch_size,
            patches_per_frame_h=patches_per_frame_h,
            patches_per_frame_w=patches_per_frame_w,
            tokens_per_frame=tokens_per_frame,
            sequence_length=tokens.shape[1],
        )
        return tokens, token_grid
