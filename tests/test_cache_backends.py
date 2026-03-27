from __future__ import annotations

import torch

from open_wam.models.common import (
    clear_cache_backend_payload,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    update_slot_pool_layer_state,
)
from open_wam.models.policy_variants.parallel_stream.reference_runtime import prepare_reference_single_stream_input
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def test_slot_pool_backend_materializes_and_clears_predicted_entries() -> None:
    payload = init_cache_backend_payload(
        "slot_pool_exact",
        num_layers=1,
        total_tokens=6,
        num_heads=2,
        head_dim=4,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layer_state = payload.layer_states[0]

    update_slot_pool_layer_state(
        layer_state,
        key=torch.randn(1, 2, 2, 4),
        value=torch.randn(1, 2, 2, 4),
        is_pred=False,
    )
    entries = materialize_cache_backend_entries(payload)
    assert len(entries) == 1
    assert entries[0].key is not None
    assert entries[0].key.shape == (1, 2, 2, 4)
    assert entries[0].metadata["cached_tokens"] == 2

    update_slot_pool_layer_state(
        layer_state,
        key=torch.randn(1, 1, 2, 4),
        value=torch.randn(1, 1, 2, 4),
        is_pred=True,
    )
    entries = materialize_cache_backend_entries(payload)
    assert entries[0].key is not None
    assert entries[0].key.shape[2] == 3
    assert torch.equal(entries[0].metadata["prediction_mask"], torch.tensor([False, False, True]))

    cleared = clear_cache_backend_payload(payload, clear_predictions_only=True)
    entries = materialize_cache_backend_entries(cleared)
    assert entries[0].key is not None
    assert entries[0].key.shape[2] == 2
    assert torch.equal(entries[0].metadata["prediction_mask"], torch.tensor([False, False]))


def test_exact_replica_core_uses_slot_pool_cache_backend() -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    core = SharedVideoTransformerCore(backbone_config, action_dim=4).to(dtype=torch.bfloat16)
    cache_name = "slot_pool_exact"
    core.create_empty_cache(
        cache_name,
        attn_window=4,
        latent_token_per_chunk=4,
        action_token_per_chunk=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        batch_size=1,
    )

    cache_state = core._exact_runtime_caches[cache_name]
    assert cache_state.backend_name == "slot_pool_exact"
    assert cache_state.backend_payload is not None

    text_emb = torch.zeros(1, backbone_config.max_text_tokens, backbone_config.text_dim, dtype=torch.bfloat16)
    video_latents = torch.randn(1, backbone_config.latent_channels, 1, 4, 4, dtype=torch.bfloat16)

    committed_input = prepare_reference_single_stream_input(
        latents=video_latents,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=0,
        backbone_config=backbone_config,
        action_mode=False,
    )
    core(committed_input, update_cache=2, cache_name=cache_name, action_mode=False)
    committed_state = core._exact_runtime_caches[cache_name]
    assert committed_state.self_attention_kv[0].key is not None
    assert committed_state.self_attention_kv[0].key.shape[2] == 4

    predicted_input = prepare_reference_single_stream_input(
        latents=video_latents,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=1,
        backbone_config=backbone_config,
        action_mode=False,
    )
    core(predicted_input, update_cache=1, cache_name=cache_name, action_mode=False)
    predicted_state = core._exact_runtime_caches[cache_name]
    assert predicted_state.self_attention_kv[0].key is not None
    assert predicted_state.self_attention_kv[0].key.shape[2] == 8

    core.clear_pred_cache(cache_name)
    cleared_state = core._exact_runtime_caches[cache_name]
    assert cleared_state.self_attention_kv[0].key is not None
    assert cleared_state.self_attention_kv[0].key.shape[2] == 4
