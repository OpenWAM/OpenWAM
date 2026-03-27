from __future__ import annotations

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata
from open_wam.models.policy_variants.common.positions import build_sequence_position_context, build_video_position_context

from .layout import RegisterSequenceLayout


def build_register_position_context(
    layout: RegisterSequenceLayout,
    token_grid: TokenGridMetadata,
    hidden_size: int,
    device: torch.device,
    current_start_frame: int = 0,
) -> torch.Tensor:
    position_chunks: list[torch.Tensor] = []
    if layout.has_clean_video_prefix:
        position_chunks.append(
            build_video_position_context(
                token_grid=token_grid,
                hidden_size=hidden_size,
                device=device,
                frame_offset=current_start_frame,
            )
        )
    position_chunks.append(
        build_video_position_context(
            token_grid=token_grid,
            hidden_size=hidden_size,
            device=device,
            frame_offset=current_start_frame,
        )
    )
    action_length = sum(end - start for start, end in layout.action_block_spans)
    state_length = sum(end - start for start, end in layout.state_block_spans)
    action_position = build_sequence_position_context(action_length, hidden_size, device=device, offset=0)
    state_position = build_sequence_position_context(state_length, hidden_size, device=device, offset=0)
    position_chunks.extend([action_position, state_position])
    return torch.cat(position_chunks, dim=0)
