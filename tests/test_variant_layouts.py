from __future__ import annotations

import torch

from open_wam.models.policy_variants.parallel_stream.packing import (
    build_parallel_layout,
)
from open_wam.models.video_backbone.contracts import TokenGridMetadata


def _token_grid(num_frames: int = 4, tokens_per_frame: int = 120) -> TokenGridMetadata:
    return TokenGridMetadata(
        num_frames=num_frames,
        latent_height=24,
        latent_width=20,
        patch_size=(1, 2, 2),
        patches_per_frame_h=12,
        patches_per_frame_w=10,
        tokens_per_frame=tokens_per_frame,
        sequence_length=num_frames * tokens_per_frame,
    )


def test_parallel_layout_length_matches_expected_streams() -> None:
    token_grid = _token_grid()
    layout = build_parallel_layout(
        token_grid=token_grid,
        action_per_frame=2,
        frame_chunk_size=2,
        sequence_order=("video_noisy", "video_condition", "action_noisy", "action_condition"),
        device=torch.device("cpu"),
    )
    assert layout.frame_ids.numel() == 976
    assert layout.spans["action_noisy"][1] - layout.spans["action_noisy"][0] == 8
