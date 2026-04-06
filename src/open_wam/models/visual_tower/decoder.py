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
        return self.forward_tokens(
            frontend_output=frontend_output,
            tokens=core_output.tokens,
            token_layout=core_output.token_layout,
        )

    def forward_tokens(
        self,
        *,
        frontend_output: VisualFrontendOutput,
        tokens: torch.Tensor,
        token_layout,
    ) -> VisualDecodeOutput:
        tokens = self.proj(tokens)
        batch_size, seq_len, hidden_size = tokens.shape
        num_frames = int(frontend_output.token_grid.num_frames)
        tokens_per_frame = int(frontend_output.token_grid.tokens_per_frame)
        expected_seq_len = int(frontend_output.token_grid.sequence_length)
        if token_layout is not None:
            if hasattr(token_layout, "num_frames"):
                num_frames = int(token_layout.num_frames)
            if hasattr(token_layout, "tokens_per_frame"):
                tokens_per_frame = int(token_layout.tokens_per_frame)
            if hasattr(token_layout, "sequence_length"):
                expected_seq_len = int(token_layout.sequence_length)
        frame_token_groups = num_frames
        if tokens_per_frame > 0 and expected_seq_len > 0 and expected_seq_len % tokens_per_frame == 0:
            frame_token_groups = expected_seq_len // tokens_per_frame
        if seq_len == expected_seq_len and tokens_per_frame > 0 and seq_len % tokens_per_frame == 0:
            decoded_features = tokens.view(batch_size, frame_token_groups, tokens_per_frame, hidden_size)
            layout = DecodedFeatureLayout(
                kind="frame_token_sequence",
                num_frames=frame_token_groups,
                tokens_per_frame=tokens_per_frame,
                hidden_size=hidden_size,
            )
        else:
            layout_num_frames = frame_token_groups if tokens_per_frame > 0 and seq_len % tokens_per_frame == 0 else num_frames
            decoded_features = tokens
            layout = DecodedFeatureLayout(
                kind="sequence",
                num_frames=layout_num_frames,
                tokens_per_frame=max(1, seq_len // max(1, layout_num_frames)),
                hidden_size=hidden_size,
            )
        return VisualDecodeOutput(
            decoded_features=decoded_features,
            feature_layout=layout,
            aux={"decoder": "lightweight_feature_projection"},
        )
