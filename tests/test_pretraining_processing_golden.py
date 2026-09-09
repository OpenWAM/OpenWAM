"""Exact preprocessing numerics recorded from #65 before the module moves."""

import json
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file

from open_wam.data.preparation import frames
from open_wam.data.preparation.multiview import layout
from open_wam.models.visual_tower import vae_encoding
from tests.characterization.pretraining_processing import capture


@pytest.mark.parametrize(
    "height,width,expected",
    [(100, 100, "small"), (101, 101, "large"), (10000, 10000, "large"),
     (100, 200, "wide"), (200, 100, "tall")],
)
def test_resize_bin_aspect_ratio_and_pixel_tiers(height, width, expected):
    from open_wam.configs import MixedVideoResizeBinConfig
    from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin

    bins = (
        MixedVideoResizeBinConfig("large", 1, 1, 256, 256),
        MixedVideoResizeBinConfig("small", 1, 1, 128, 128, max_pixels=10000),
        MixedVideoResizeBinConfig("wide", 2, 1, 128, 256),
        MixedVideoResizeBinConfig("tall", 1, 2, 256, 128),
    )
    selected = select_mixed_video_resize_bin(bins, source_height=height, source_width=width)
    assert selected.name == expected


def test_rgb_frame_selection_layouts_and_vae_inputs_match_original_pr():
    root = Path(__file__).parent / "fixtures/pretraining"
    expected = load_file(root / "processing.safetensors")
    actual, layouts = capture(frames, layout, vae_encoding)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
    expected_layouts = json.loads((root / "layouts.json").read_text())
    for value in layouts.values():
        # acf7a2a1 renamed the artifact label; pixel geometry stays frozen.
        assert value["policy"] == "rgb_multi_view_65k_portable_v2"
        value["policy"] = "rgb_mosaic_65k_portable_v2"
    assert json.loads(json.dumps(layouts)) == expected_layouts
