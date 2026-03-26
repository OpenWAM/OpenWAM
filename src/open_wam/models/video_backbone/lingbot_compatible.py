from __future__ import annotations

import torch
from torch import nn

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.visual_tower import VisualTower

from .config import LingbotCompatibleVideoBackboneConfig
from .contracts import BackboneOutput


class LingbotCompatibleVideoBackbone(nn.Module):
    """Protected stage-1 backbone boundary for the new WAM codebase.

    This module does not yet port the full LingBot transformer stack. Instead,
    it establishes the LingBot-compatible geometry and the shared output
    contract that all future action-head variants will consume.

    Stage-1 responsibilities:
    - preserve the canonical RGB -> latent -> token geometry
    - provide a stable backbone output contract
    - keep the boundary clean so future LingBot weight-loading can happen here
    """

    def __init__(self, config: LingbotCompatibleVideoBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        self.tower = VisualTower(self.config)

    def forward(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        preserve_stream_cache: bool = False,
    ) -> BackboneOutput:
        frontend_output = self.tower.run_frontend(
            canonical_video,
            placements=placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        core_output = self.tower.run_default_core(frontend_output)
        return BackboneOutput(
            canonical_video=frontend_output.canonical_video,
            video_latents=frontend_output.video_latents,
            video_tokens=core_output.tokens,
            token_grid=frontend_output.token_grid,
            chunk=frontend_output.chunk,
            cache_state=core_output.cache_state,
            conditioning=frontend_output.conditioning,
        )

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        reset_reference_cache: bool = True,
    ) -> torch.Tensor:
        return self.tower.frontend.encode_video(
            canonical_video,
            placements=placements,
            reset_reference_cache=reset_reference_cache,
        )

    def tokenize_video_latents(self, video_latents: torch.Tensor):
        return self.tower.frontend.tokenize_video_latents(video_latents)
