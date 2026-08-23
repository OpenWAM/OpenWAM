from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.common.video_geometry import (
    wan_raw_frame_count_to_latent_count,
    wan_safe_temporal_frame_count,
)
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower.reference_assets import (
    LingbotReferenceAssets,
    WanVAEStreamingWrapper,
)


class _RecordingWanEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | None],
        feat_idx: list[int],
    ) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        feat_cache[feat_idx[0]] = x[:, :, -1:].detach().clone()
        feat_idx[0] += 1
        return x


def _recording_wan_wrapper() -> tuple[WanVAEStreamingWrapper, _RecordingWanEncoder]:
    encoder = _RecordingWanEncoder()
    vae = SimpleNamespace(
        config=SimpleNamespace(patch_size=None),
        encoder=encoder,
        quant_conv=torch.nn.Identity(),
        _cached_conv_counts={"encoder": 1},
    )
    return WanVAEStreamingWrapper(vae), encoder


def test_reference_video_scaling_matches_float32_then_cast_behavior() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=SimpleNamespace(
            parameters=lambda: iter(
                (torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),)
            ),
        ),
        streaming_vae=object(),
    )
    video = torch.tensor([[[[[0.5019608]]]]], dtype=torch.float32)

    param = next(assets.vae.parameters())
    scaled = (video.to(device=param.device, dtype=torch.float32) * 2.0 - 1.0).to(
        dtype=param.dtype
    )
    expected = torch.tensor([[[[[0.00392157]]]]], dtype=torch.float32).to(
        dtype=param.dtype
    )
    wrong = video.to(device=param.device, dtype=param.dtype) * 2.0 - 1.0

    assert torch.equal(scaled, expected)
    assert not torch.equal(scaled, wrong)


def test_wan_temporal_counts_match_diffusers_chunking() -> None:
    fresh_counts = [
        wan_safe_temporal_frame_count(frames, cache_initialized=False)
        for frames in range(1, 19)
    ]
    streaming_counts = [
        wan_safe_temporal_frame_count(frames, cache_initialized=True)
        for frames in range(1, 10)
    ]
    latent_counts = [
        wan_raw_frame_count_to_latent_count(frames) for frames in range(1, 19)
    ]

    assert fresh_counts == [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13, 17, 17]
    assert streaming_counts == [0, 0, 0, 4, 4, 4, 4, 8, 8]
    assert latent_counts == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]


def test_wan_streaming_wrapper_matches_fresh_diffusers_windows() -> None:
    wrapper, encoder = _recording_wan_wrapper()
    video = (
        torch.arange(16, dtype=torch.float32)
        .view(1, 1, 16, 1, 1)
        .expand(1, 3, 16, 1, 1)
    )

    _ = wrapper.encode_chunk(video)
    encoded_input = torch.cat(encoder.calls, dim=2)

    assert [call.shape[2] for call in encoder.calls] == [1, 4, 4, 4]
    assert encoded_input.shape[2] == 13
    assert torch.equal(encoded_input, video[:, :, :13])


def test_wan_streaming_wrapper_uses_only_complete_existing_cache_groups() -> None:
    wrapper, encoder = _recording_wan_wrapper()
    wrapper.encode_chunk(torch.zeros(1, 3, 1, 1, 1))
    encoder.calls.clear()
    video = (
        torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1).expand(1, 3, 7, 1, 1)
    )

    _ = wrapper.encode_chunk(video)
    encoded_input = torch.cat(encoder.calls, dim=2)

    assert [call.shape[2] for call in encoder.calls] == [4]
    assert encoded_input.shape[2] == 4
    assert torch.equal(encoded_input, video[:, :, :4])


def test_reference_assets_runtime_snapshot_restores_default_and_keyed_caches() -> None:
    default_wrapper, _ = _recording_wan_wrapper()
    keyed_wrapper, _ = _recording_wan_wrapper()
    new_wrapper, _ = _recording_wan_wrapper()
    default_wrapper.feat_cache[0] = torch.tensor([1.0])
    keyed_wrapper.feat_cache[0] = torch.tensor([2.0])
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=default_wrapper.vae,
        streaming_vae=default_wrapper,
        streaming_vae_by_key={"camera:a": keyed_wrapper},
    )

    snapshot = assets.snapshot_runtime_state()
    assert snapshot is not None
    default_wrapper.feat_cache[0][0] = 9.0
    keyed_wrapper.feat_cache[0][0] = 8.0
    new_wrapper.feat_cache[0] = torch.tensor([7.0])
    assets.streaming_vae_by_key["camera:new"] = new_wrapper

    assets.restore_runtime_state(snapshot)

    assert float(default_wrapper.feat_cache[0][0]) == 1.0
    assert float(keyed_wrapper.feat_cache[0][0]) == 2.0
    assert "camera:new" not in assets.streaming_vae_by_key


def test_equal_resolution_layout_encodes_views_without_camera_name_rules() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=object(),  # mark VAE assets as present for this layout-only unit test
        streaming_vae=object(),
    )

    def fake_encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        del reset_cache, cache_key
        batch_size, _, num_frames, height, width = video.shape
        means = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        return means.expand(
            batch_size, 48, num_frames, height // 16, width // 16
        ).clone()

    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]

    canonical_video = torch.zeros(1, 3, 2, 128, 256, dtype=torch.float32)
    canonical_video[:, :, :, :, :128] = 1.0
    canonical_video[:, :, :, :, 128:] = 3.0
    placements = (
        ViewPlacement(
            source_name="front",
            canonical_name="front",
            top=0,
            left=0,
            height=128,
            width=128,
        ),
        ViewPlacement(
            source_name="hand",
            canonical_name="hand",
            top=0,
            left=128,
            height=128,
            width=128,
        ),
    )

    encoded = assets.encode_video(
        canonical_video, placements=placements, reset_cache=True
    )

    assert encoded.shape == (1, 48, 2, 8, 16)
    assert torch.allclose(encoded[..., :8], torch.ones_like(encoded[..., :8]))
    assert torch.allclose(encoded[..., 8:], torch.full_like(encoded[..., 8:], 3.0))


def test_mixed_resolution_layout_uses_independent_streaming_cache_keys() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=object(),
        streaming_vae=object(),
    )
    calls: list[tuple[str | None, bool]] = []

    def fake_encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        calls.append((cache_key, reset_cache))
        batch_size, _, num_frames, height, width = video.shape
        means = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        return means.expand(
            batch_size, 48, num_frames, height // 16, width // 16
        ).clone()

    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]

    canonical_video = torch.zeros(1, 3, 2, 384, 320, dtype=torch.float32)
    canonical_video[:, :, :, :256, :] = 1.0
    canonical_video[:, :, :, 256:, :160] = 2.0
    canonical_video[:, :, :, 256:, 160:] = 4.0
    placements = (
        ViewPlacement(
            source_name="cam_high",
            canonical_name="cam_high",
            top=0,
            left=0,
            height=256,
            width=320,
        ),
        ViewPlacement(
            source_name="cam_left_wrist",
            canonical_name="cam_left_wrist",
            top=256,
            left=0,
            height=128,
            width=160,
        ),
        ViewPlacement(
            source_name="cam_right_wrist",
            canonical_name="cam_right_wrist",
            top=256,
            left=160,
            height=128,
            width=160,
        ),
    )

    encoded = assets.encode_video(
        canonical_video, placements=placements, reset_cache=False
    )

    assert calls == [
        ("view:0:cam_high", False),
        ("view:1:cam_left_wrist", False),
        ("view:2:cam_right_wrist", False),
    ]
    assert encoded.shape == (1, 48, 2, 24, 20)
    assert torch.allclose(encoded[..., :16, :], torch.ones_like(encoded[..., :16, :]))
    assert torch.allclose(
        encoded[..., 16:, :10], torch.full_like(encoded[..., 16:, :10], 2.0)
    )
    assert torch.allclose(
        encoded[..., 16:, 10:], torch.full_like(encoded[..., 16:, 10:], 4.0)
    )


def test_three_view_layout_preserves_centered_row_and_empty_regions() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=object(),
        streaming_vae=object(),
    )
    encoded_batch_sizes: list[int] = []

    def fake_encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        del reset_cache, cache_key
        encoded_batch_sizes.append(int(video.shape[0]))
        batch_size, _, num_frames, height, width = video.shape
        means = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        return means.expand(
            batch_size,
            48,
            num_frames,
            height // 16,
            width // 16,
        ).clone()

    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]
    canonical_video = torch.zeros(1, 3, 2, 256, 256, dtype=torch.float32)
    canonical_video[..., :128, :128] = 1.0
    canonical_video[..., :128, 128:] = 2.0
    canonical_video[..., 128:, 64:192] = 3.0
    placements = (
        ViewPlacement("a", "a", top=0, left=0, height=128, width=128),
        ViewPlacement("b", "b", top=0, left=128, height=128, width=128),
        ViewPlacement("c", "c", top=128, left=64, height=128, width=128),
    )

    encoded = assets.encode_video(canonical_video, placements=placements)

    assert encoded_batch_sizes == [3]
    assert encoded.shape == (1, 48, 2, 16, 16)
    assert torch.allclose(encoded[..., :8, :8], torch.ones_like(encoded[..., :8, :8]))
    assert torch.allclose(
        encoded[..., :8, 8:], torch.full_like(encoded[..., :8, 8:], 2.0)
    )
    assert torch.allclose(
        encoded[..., 8:, 4:12], torch.full_like(encoded[..., 8:, 4:12], 3.0)
    )
    assert torch.count_nonzero(encoded[..., 8:, :4]) == 0
    assert torch.count_nonzero(encoded[..., 8:, 12:]) == 0


def test_reference_latent_normalization_uses_float32_stats_before_casting_back() -> (
    None
):
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=SimpleNamespace(
            config=SimpleNamespace(
                latents_mean=[0.123456789],
                latents_std=[0.987654321],
            )
        ),
        streaming_vae=object(),
    )
    latents = torch.tensor([[[[[1.1]]]]], dtype=torch.bfloat16)

    normalized = assets._normalize_reference_latents(latents)
    expected = (
        (latents.float() - torch.tensor([0.123456789]).view(1, 1, 1, 1, 1))
        * (1.0 / torch.tensor([0.987654321]).view(1, 1, 1, 1, 1))
    ).to(latents)
    wrong = (
        (
            latents.float()
            - torch.tensor([0.123456789], dtype=latents.dtype)
            .float()
            .view(1, 1, 1, 1, 1)
        )
        * (
            1.0
            / torch.tensor([0.987654321], dtype=latents.dtype)
            .float()
            .view(1, 1, 1, 1, 1)
        )
    ).to(latents)

    assert torch.equal(normalized, expected)
    assert not torch.equal(normalized, wrong)


def test_encode_video_moves_reference_vae_to_runtime_device() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=object(),
        streaming_vae=object(),
    )
    recorded_devices: list[torch.device] = []

    def fake_ensure_vae_runtime_device(self, device: torch.device) -> None:
        recorded_devices.append(torch.device(device))

    def fake_encode_chunk(
        self,
        video: torch.Tensor,
        *,
        reset_cache: bool = True,
        cache_key: str | None = None,
    ) -> torch.Tensor:
        del reset_cache, cache_key
        batch_size, _, num_frames, height, width = video.shape
        return torch.zeros(
            batch_size,
            48,
            num_frames,
            height // 16,
            width // 16,
            dtype=video.dtype,
            device=video.device,
        )

    assets._ensure_vae_runtime_device = MethodType(
        fake_ensure_vae_runtime_device, assets
    )  # type: ignore[method-assign]
    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]

    canonical_video = torch.zeros(1, 3, 1, 128, 256, dtype=torch.float32)
    placements = (
        ViewPlacement(
            source_name="image",
            canonical_name="image",
            top=0,
            left=0,
            height=128,
            width=128,
        ),
        ViewPlacement(
            source_name="wrist_image",
            canonical_name="wrist_image",
            top=0,
            left=128,
            height=128,
            width=128,
        ),
    )

    encoded = assets.encode_video(
        canonical_video, placements=placements, reset_cache=True
    )

    assert recorded_devices == [canonical_video.device]
    assert encoded.device == canonical_video.device
