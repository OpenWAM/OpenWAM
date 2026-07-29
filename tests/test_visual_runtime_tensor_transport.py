from __future__ import annotations

import torch

from open_wam.models.common import (
    SlotPoolLayerState,
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower import replica_core as replica_core_module
from open_wam.models.visual_tower import select_split_segments
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore
from open_wam.models.visual_tower.runtime_tensor_transport import (
    cached_attention_profile,
    cached_optional_tensor,
    move_attention_profile,
    move_optional_tensor,
    move_slot_pool_layer_state,
)


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


def test_runtime_transport_compatibility_exports_preserve_identity() -> None:
    assert SharedVideoTransformerCore._move_optional_tensor is move_optional_tensor
    assert SharedVideoTransformerCore._cached_optional_tensor is cached_optional_tensor
    assert SharedVideoTransformerCore._move_slot_pool_layer_state is move_slot_pool_layer_state
    assert replica_core_module._select_split_segments is select_split_segments


def test_cached_optional_tensor_reuses_per_device_copy() -> None:
    tensor = torch.ones(2, 3)
    cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}

    first = cached_optional_tensor(
        tensor,
        cache=cache,
        name="tensor",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = cached_optional_tensor(
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

    first = cached_attention_profile(
        profile,
        cache=cache,
        patch_size=core.patch_size,
        device=torch.device("cpu"),
    )
    second = core._cached_attention_profile(
        profile,
        cache=cache,
        device=torch.device("cpu"),
    )

    assert first is second
    assert first is not None
    assert first.self_attention_mask is not None
    assert first.self_attention_mask.device.type == "cpu"
    wrapped = core._move_attention_profile(profile, device=torch.device("cpu"))
    direct = move_attention_profile(
        profile,
        patch_size=core.patch_size,
        device=torch.device("cpu"),
    )
    assert wrapped is not None and direct is not None
    assert wrapped.spec == direct.spec
    assert torch.equal(wrapped.self_attention_mask, direct.self_attention_mask)
    assert torch.equal(wrapped.cross_attention_mask, direct.cross_attention_mask)


def test_move_slot_pool_layer_state_preserves_object_identity() -> None:
    state = SlotPoolLayerState(
        key=torch.ones(1, 2, 3),
        value=torch.zeros(1, 2, 3),
        slot_ids=torch.arange(2),
        slot_mask=torch.ones(2, dtype=torch.bool),
    )

    moved = move_slot_pool_layer_state(state, device=torch.device("cpu"))

    assert moved is state
    assert state.key is not None and state.key.device.type == "cpu"
    assert state.value is not None and state.value.device.type == "cpu"
