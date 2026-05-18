from __future__ import annotations

import pytest

from open_wam.configs import LatentTemporalLayout
from open_wam.data.latent_temporal import (
    latent_anchor_positions,
    latent_raw_boundaries,
    raw_window_frames_for_latents,
)


def test_wan_causal_stride4_layout() -> None:
    assert raw_window_frames_for_latents(4) == 13
    assert raw_window_frames_for_latents(4, latent_stride_frames=4) == 13
    assert latent_raw_boundaries(
        raw_frame_count=15,
        latent_num_frames=4,
        layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    ) == [0, 1, 5, 9, 13]
    assert latent_anchor_positions(
        raw_frame_count=15,
        latent_num_frames=4,
        layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    ) == [0, 4, 8, 12]


def test_raw_window_frames_separates_latent_stride_from_legacy_action_alias() -> None:
    assert raw_window_frames_for_latents(4, latent_stride_frames=2) == 7
    assert raw_window_frames_for_latents(4, action_per_frame=2) == 7


def test_equal_bucket_legacy_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="equal_bucket_legacy.*deprecated and unsupported"):
        raw_window_frames_for_latents(4, layout=LatentTemporalLayout.EQUAL_BUCKET_LEGACY)

    with pytest.raises(ValueError, match="equal_bucket_legacy.*deprecated and unsupported"):
        latent_raw_boundaries(
            raw_frame_count=15,
            latent_num_frames=4,
            layout="equal_bucket_legacy",
        )
