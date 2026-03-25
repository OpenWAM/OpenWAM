from __future__ import annotations

import torch
from torch import nn

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

    def forward(self, canonical_video: torch.Tensor) -> BackboneOutput:
        stage_outputs = self.tower.forward_default(canonical_video, include_decode=False)
        if stage_outputs.core is None:
            raise RuntimeError("Default visual-tower forward must produce core outputs.")
        return BackboneOutput(
            canonical_video=stage_outputs.frontend.canonical_video,
            video_latents=stage_outputs.frontend.video_latents,
            video_tokens=stage_outputs.core.tokens,
            token_grid=stage_outputs.frontend.token_grid,
            chunk=stage_outputs.frontend.chunk,
            cache_state=stage_outputs.core.cache_state,
            conditioning=stage_outputs.frontend.conditioning,
        )

    def encode_video(self, canonical_video: torch.Tensor) -> torch.Tensor:
        return self.tower.frontend.encode_video(canonical_video)

    def tokenize_video_latents(self, video_latents: torch.Tensor):
        return self.tower.frontend.tokenize_video_latents(video_latents)
