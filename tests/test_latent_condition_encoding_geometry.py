"""Regression coverage for latent condition encoding geometry."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from .test_augment_lerobot_latents_with_single_frame_condition import _load_script_module


pytestmark = pytest.mark.unit


class _Reader:
    def __init__(self, frames, *, error=StopIteration):
        self.frames = frames
        self.error = error
        self.calls = []
        self.closed = False

    def get_data(self, index):
        self.calls.append(index)
        if index not in self.frames:
            raise self.error(index)
        return self.frames[index]

    def get_meta_data(self):
        raise AssertionError("EOF fallback must not request PyAV metadata")

    def close(self):
        self.closed = True


class _Assets:
    def __init__(self, module, *, invalid_geometry=False):
        self.module = module
        self.invalid_geometry = invalid_geometry
        self.resize_calls = []
        self.encoded_videos = []

    def _resize_rgb_chunk(self, video, height, width):
        self.resize_calls.append((height, width))
        return self.module.LingbotReferenceAssets._resize_rgb_chunk(self, video, height, width)

    def encode_video(self, video, *, placements, reset_cache):
        assert placements is None and reset_cache is True
        self.encoded_videos.append(video.clone())
        if self.invalid_geometry:
            return torch.zeros(video.shape[0], 2, 1, 1, 1)
        return video[:, :2, :, ::2, ::2]


@pytest.mark.parametrize(
    ("metadata", "expected_size", "resize_count"),
    [
        ({"video_height": 4, "video_width": 8}, (4, 8), 2),
        ({"video_height": 6, "video_width": 10}, (6, 10), 0),
        ({}, (6, 10), 0),
        ({"video_height": None, "video_width": 0}, (6, 10), 0),
        ({"video_height": 4}, (4, 10), 2),
    ],
)
def test_condition_encoder_matches_payload_resolution_before_vae(
    monkeypatch, metadata, expected_size, resize_count
):
    module = _load_script_module()
    frame = torch.arange(6 * 10 * 3, dtype=torch.uint8).reshape(6, 10, 3)
    frames = {index: frame + index for index in range(9)}
    reader = _Reader(frames)
    monkeypatch.setattr(module.imageio, "get_reader", lambda path: reader)
    assets = _Assets(module)
    height, width = expected_size
    payload = {"latent_num_frames": 3, "latent_height": height // 2,
               "latent_width": width // 2, "frame_ids": list(range(9)), **metadata}

    encoded = module._encode_condition_latents(
        payload=payload, video_path=Path("fixture.mp4"), assets=assets,
        device=torch.device("cpu"), output_dtype=torch.float64, batch_size=2,
    )

    assert reader.calls == [1, 5, 8] and reader.closed
    assert assets.resize_calls == [expected_size] * resize_count
    expected = torch.stack([frames[index].permute(2, 0, 1).float() / 255.0 for index in reader.calls])
    if resize_count:
        expected = F.interpolate(expected, size=expected_size, mode="bilinear", align_corners=False)
    observed_video = torch.cat(assets.encoded_videos, dim=0)
    torch.testing.assert_close(observed_video, expected.unsqueeze(2), rtol=0, atol=0)
    expected_latents = expected[:, :2, ::2, ::2].permute(0, 2, 3, 1).reshape(-1, 2)
    assert encoded.dtype is torch.float64 and encoded.device.type == "cpu"
    torch.testing.assert_close(encoded, expected_latents.to(torch.float64), rtol=0, atol=0)


def test_condition_encoder_keeps_geometry_guard_and_closes_reader(monkeypatch):
    module = _load_script_module()
    reader = _Reader({0: torch.zeros(6, 10, 3, dtype=torch.uint8)})
    monkeypatch.setattr(module.imageio, "get_reader", lambda path: reader)
    with pytest.raises(ValueError, match="geometry does not match payload"):
        module._encode_condition_latents(
            payload={"latent_num_frames": 1, "latent_height": 2, "latent_width": 4,
                     "video_height": 4, "video_width": 8, "frame_ids": [0]},
            video_path=Path("fixture.mp4"), assets=_Assets(module, invalid_geometry=True),
            device=torch.device("cpu"), output_dtype=torch.float32, batch_size=1,
        )
    assert reader.closed


@pytest.mark.parametrize("error", [IndexError, StopIteration])
def test_eof_falls_back_to_nearest_readable_frame_without_metadata(error):
    module = _load_script_module()
    frame = object()
    reader = _Reader({4: frame}, error=error)
    assert module._read_video_frame(reader, 6) is frame
    assert reader.calls == [6, 5, 4]


@pytest.mark.parametrize("readable_index", [8, 7])
def test_eof_backoff_is_exactly_bounded_to_32_previous_frames(readable_index):
    module = _load_script_module()
    frame = object()
    reader = _Reader({readable_index: frame})
    if readable_index == 8:
        assert module._read_video_frame(reader, 40) is frame
    else:
        with pytest.raises(RuntimeError, match="within 32 frames"):
            module._read_video_frame(reader, 40)
    assert reader.calls == list(range(40, 7, -1))


@pytest.mark.parametrize("requested", [0, 1])
def test_eof_backoff_never_reads_a_negative_frame(requested):
    module = _load_script_module()
    reader = _Reader({})
    with pytest.raises(RuntimeError, match="no readable frame"):
        module._read_video_frame(reader, requested)
    assert reader.calls == list(range(requested, -1, -1))


def test_non_eof_errors_are_not_swallowed():
    module = _load_script_module()
    reader = _Reader({}, error=TypeError)
    with pytest.raises(TypeError):
        module._read_video_frame(reader, 10)
    assert reader.calls == [10]


def test_readable_requested_frame_does_not_trigger_fallback():
    module = _load_script_module()
    frame = object()
    reader = _Reader({3: frame})
    assert module._read_video_frame(reader, 3) is frame
    assert reader.calls == [3]
