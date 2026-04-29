from __future__ import annotations

import torch

from open_wam.models.common import build_chunked_temporal_exact_attention_profile
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def _small_core() -> SharedVideoTransformerCore:
    config = SharedVideoTransformerConfig(
        latent_channels=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        text_dim=8,
        freq_dim=4,
    )
    return SharedVideoTransformerCore(config=config, action_dim=2)


def test_cached_optional_tensor_reuses_per_device_copy() -> None:
    tensor = torch.ones(2, 3)
    cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}

    first = SharedVideoTransformerCore._cached_optional_tensor(
        tensor,
        cache=cache,
        name="tensor",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = SharedVideoTransformerCore._cached_optional_tensor(
        tensor,
        cache=cache,
        name="tensor",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first is second
    assert len(cache) == 1


def test_attention_profile_cache_reuses_device_specific_profile() -> None:
    core = _small_core()
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 4, 2, 1, 1),
        action_shape=(1, 2, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=4,
        patch_size=core.patch_size,
        text_token_count=3,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )
    cache = {}

    first = core._cached_attention_profile(profile, cache=cache, device=torch.device("cpu"))
    second = core._cached_attention_profile(profile, cache=cache, device=torch.device("cpu"))

    assert first is second
    assert first is not None
    assert first.self_attention_mask is not None
    assert first.self_attention_mask.device.type == "cpu"
