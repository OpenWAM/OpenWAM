from __future__ import annotations

import torch
from torch import nn

from .contracts import DecodedFeatureLayout, VisualCoreOutput, VisualDecodeOutput, VisualFrontendOutput


class VisualFeatureDecoder(nn.Module):
    """Lightweight decoded-feature adapter for post-decoded policies."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, frontend_output: VisualFrontendOutput, core_output: VisualCoreOutput) -> VisualDecodeOutput:
        tokens = self.proj(core_output.tokens)
        batch_size, seq_len, hidden_size = tokens.shape
        num_frames = frontend_output.token_grid.num_frames
        tokens_per_frame = frontend_output.token_grid.tokens_per_frame
        if seq_len == frontend_output.token_grid.sequence_length:
            decoded_features = tokens.view(batch_size, num_frames, tokens_per_frame, hidden_size)
            layout = DecodedFeatureLayout(
                kind="frame_token_sequence",
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                hidden_size=hidden_size,
            )
        else:
            decoded_features = tokens
            layout = DecodedFeatureLayout(
                kind="sequence",
                num_frames=num_frames,
                tokens_per_frame=max(1, seq_len // max(1, num_frames)),
                hidden_size=hidden_size,
            )
        return VisualDecodeOutput(
            decoded_features=decoded_features,
            feature_layout=layout,
            aux={"decoder": "lightweight_feature_projection"},
        )
