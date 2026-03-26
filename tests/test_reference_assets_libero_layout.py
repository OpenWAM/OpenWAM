from __future__ import annotations

from types import SimpleNamespace
from types import MethodType

import torch

from open_wam.data.raw_video import ViewPlacement
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets


def test_reference_video_scaling_matches_float32_then_cast_behavior() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=SimpleNamespace(
            parameters=lambda: iter((torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),)),
        ),
        streaming_vae=object(),
    )
    video = torch.tensor([[[[[0.5019608]]]]], dtype=torch.float32)

    param = next(assets.vae.parameters())
    scaled = (video.to(device=param.device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=param.dtype)
    expected = torch.tensor([[[[[0.00392157]]]]], dtype=torch.float32).to(dtype=param.dtype)
    wrong = video.to(device=param.device, dtype=param.dtype) * 2.0 - 1.0

    assert torch.equal(scaled, expected)
    assert not torch.equal(scaled, wrong)


def test_libero_layout_encodes_views_separately_and_concatenates_latents() -> None:
    assets = LingbotReferenceAssets(
        config=LingbotCompatibleVideoBackboneConfig(),
        vae=object(),  # mark VAE assets as present for this layout-only unit test
        streaming_vae=object(),
    )

    def fake_encode_chunk(self, video: torch.Tensor, *, reset_cache: bool = True) -> torch.Tensor:
        del reset_cache
        batch_size, _, num_frames, height, width = video.shape
        means = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        return means.expand(batch_size, 48, num_frames, height // 16, width // 16).clone()

    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]

    canonical_video = torch.zeros(1, 3, 2, 128, 256, dtype=torch.float32)
    canonical_video[:, :, :, :, :128] = 1.0
    canonical_video[:, :, :, :, 128:] = 3.0
    placements = (
        ViewPlacement(name="image", top=0, left=0, height=128, width=128),
        ViewPlacement(name="wrist_image", top=0, left=128, height=128, width=128),
    )

    encoded = assets.encode_video(canonical_video, placements=placements, reset_cache=True)

    assert encoded.shape == (1, 48, 2, 8, 16)
    assert torch.allclose(encoded[..., :8], torch.ones_like(encoded[..., :8]))
    assert torch.allclose(encoded[..., 8:], torch.full_like(encoded[..., 8:], 3.0))


def test_reference_latent_normalization_uses_float32_stats_before_casting_back() -> None:
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
        (latents.float() - torch.tensor([0.123456789], dtype=latents.dtype).float().view(1, 1, 1, 1, 1))
        * (1.0 / torch.tensor([0.987654321], dtype=latents.dtype).float().view(1, 1, 1, 1, 1))
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

    def fake_encode_chunk(self, video: torch.Tensor, *, reset_cache: bool = True) -> torch.Tensor:
        del reset_cache
        batch_size, _, num_frames, height, width = video.shape
        return torch.zeros(batch_size, 48, num_frames, height // 16, width // 16, dtype=video.dtype, device=video.device)

    assets._ensure_vae_runtime_device = MethodType(fake_ensure_vae_runtime_device, assets)  # type: ignore[method-assign]
    assets._encode_chunk = MethodType(fake_encode_chunk, assets)  # type: ignore[method-assign]

    canonical_video = torch.zeros(1, 3, 1, 128, 256, dtype=torch.float32)
    placements = (
        ViewPlacement(name="image", top=0, left=0, height=128, width=128),
        ViewPlacement(name="wrist_image", top=0, left=128, height=128, width=128),
    )

    encoded = assets.encode_video(canonical_video, placements=placements, reset_cache=True)

    assert recorded_devices == [canonical_video.device]
    assert encoded.device == canonical_video.device
