from __future__ import annotations

import torch

from open_wam.models.policy_variants.common.positions import (
    build_action_grid_position_context,
    build_video_position_context,
)
from open_wam.models.video_backbone.contracts import TokenGridMetadata

from .packing import ParallelPackedSequenceLayout


def build_parallel_position_context(
    token_grid: TokenGridMetadata,
    layout: ParallelPackedSequenceLayout,
    hidden_size: int,
    action_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    contexts = []
    for name, (start, end) in layout.spans.items():
        if name.startswith("video"):
            contexts.append(build_video_position_context(token_grid, hidden_size, device=device))
        else:
            contexts.append(
                build_action_grid_position_context(
                    num_frames=token_grid.num_frames,
                    action_per_frame=action_per_frame,
                    hidden_size=hidden_size,
                    device=device,
                )
            )
    return torch.cat(contexts, dim=0)
