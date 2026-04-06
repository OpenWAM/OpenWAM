from __future__ import annotations

import torch

import open_wam.configs  # noqa: F401
from open_wam.models.video_backbone.contracts import CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata
from open_wam.models.visual_tower.contracts import VisualCoreOutput, VisualFrontendOutput
from open_wam.models.visual_tower.decoder import VisualFeatureDecoder


def _build_frontend_output(*, num_frames: int, tokens_per_frame: int, sequence_length: int) -> VisualFrontendOutput:
    return VisualFrontendOutput(
        canonical_video=torch.zeros(1, 3, num_frames, 8, 8),
        video_latents=torch.zeros(1, 4, num_frames, 4, 4),
        video_tokens=torch.zeros(1, sequence_length, 16),
        token_grid=TokenGridMetadata(
            num_frames=num_frames,
            latent_height=4,
            latent_width=4,
            patch_size=(2, 2, 2),
            patches_per_frame_h=2,
            patches_per_frame_w=tokens_per_frame // 2,
            tokens_per_frame=tokens_per_frame,
            sequence_length=sequence_length,
        ),
        chunk=ChunkMetadata(chunk_start_frame=0, chunk_num_frames=num_frames, frame_stride=1, chunk_type="dense_video_chunk"),
        conditioning=ConditioningState(supported=False),
    )


def test_visual_feature_decoder_uses_sequence_length_for_temporal_patching() -> None:
    decoder = VisualFeatureDecoder(hidden_size=16)
    frontend_output = _build_frontend_output(num_frames=4, tokens_per_frame=6, sequence_length=12)
    core_output = VisualCoreOutput(
        tokens=torch.randn(2, 12, 16),
        token_layout=frontend_output.token_grid,
        cache_state=CacheState(supported=False, current_start_frame=0, cached_frames=0, chunk_size=12),
    )

    decode_output = decoder.forward(frontend_output, core_output)

    assert decode_output.decoded_features.shape == (2, 2, 6, 16)
    assert decode_output.feature_layout.kind == "frame_token_sequence"
    assert decode_output.feature_layout.num_frames == 2
    assert decode_output.feature_layout.tokens_per_frame == 6


def test_visual_feature_decoder_keeps_generic_sequence_when_length_mismatch() -> None:
    decoder = VisualFeatureDecoder(hidden_size=16)
    frontend_output = _build_frontend_output(num_frames=4, tokens_per_frame=6, sequence_length=12)
    core_output = VisualCoreOutput(
        tokens=torch.randn(2, 10, 16),
        token_layout=frontend_output.token_grid,
        cache_state=CacheState(supported=False, current_start_frame=0, cached_frames=0, chunk_size=10),
    )

    decode_output = decoder.forward(frontend_output, core_output)

    assert decode_output.decoded_features.shape == (2, 10, 16)
    assert decode_output.feature_layout.kind == "sequence"
    assert decode_output.feature_layout.num_frames == 4
