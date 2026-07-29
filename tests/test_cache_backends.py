from __future__ import annotations

import pytest
import torch

from open_wam.configs.enums import CurrentBlockCoupling
from open_wam.models.common import (
    AttentionProfileSpec,
    apply_attention_backend,
    clear_cache_backend_payload,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    merge_attention_cache_entries,
    packed_slot_pool_query_sequence_ids,
    prepend_cached_prefix_mask,
    PreparedAttentionProfile,
    prepare_sdpa_mask,
    resolve_slot_pool_prefix_visibility,
    retained_slot_pool_indices_for_current_write,
    update_slot_pool_layer_state,
)
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    _build_joint_clean_cache_attention_mask,
    prepare_reference_single_stream_input,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower import replica_core as replica_core_module
from open_wam.models.visual_tower.replica_core import (
    SharedTransformerAttention,
    SharedVideoTransformerCore,
)


def test_cache_policy_public_and_compatibility_exports_preserve_identity() -> None:
    assert replica_core_module._prepare_sdpa_mask is prepare_sdpa_mask
    assert replica_core_module._prepend_cached_prefix_mask is prepend_cached_prefix_mask
    assert replica_core_module._resolve_slot_pool_prefix_visibility is resolve_slot_pool_prefix_visibility
    assert replica_core_module._packed_slot_pool_query_sequence_ids is packed_slot_pool_query_sequence_ids
    assert (
        replica_core_module._retained_slot_pool_indices_for_current_write
        is retained_slot_pool_indices_for_current_write
    )
    assert replica_core_module._merge_attention_cache_entries is merge_attention_cache_entries


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
        stream_ids=torch.tensor([0, 0]),
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
        stream_ids=torch.tensor([1]),
    )
    entries = materialize_cache_backend_entries(payload)
    assert entries[0].key is not None
    assert entries[0].key.shape[2] == 3
    assert torch.equal(entries[0].metadata["prediction_mask"], torch.tensor([False, False, True]))
    assert torch.equal(entries[0].metadata["stream_ids"], torch.tensor([0, 0, 1]))

    cleared = clear_cache_backend_payload(payload, clear_predictions_only=True)
    entries = materialize_cache_backend_entries(cleared)
    assert entries[0].key is not None
    assert entries[0].key.shape[2] == 2
    assert torch.equal(entries[0].metadata["prediction_mask"], torch.tensor([False, False]))
    assert torch.equal(entries[0].metadata["stream_ids"], torch.tensor([0, 0]))


def test_slot_pool_prefix_visibility_preserves_video_pretrain_history() -> None:
    current_mask = torch.ones(3, 2, dtype=torch.bool)

    resolved = resolve_slot_pool_prefix_visibility(
        current_mask,
        prefix_len=2,
        prefix_visibility_mode="preserve_video_pretrain_history",
        query_stream_ids=torch.tensor([0, 1, -1]),
        cached_prefix_stream_ids=torch.tensor([0, 1]),
    )

    assert resolved is not None
    expected_prefix = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, False],
        ]
    )
    assert torch.equal(resolved[:, :2], expected_prefix)
    assert torch.equal(resolved[:, 2:], current_mask)


def test_slot_pool_prefix_visibility_allows_staged_current_action_tail() -> None:
    current_mask = torch.ones(3, 2, dtype=torch.bool)

    resolved = resolve_slot_pool_prefix_visibility(
        current_mask,
        prefix_len=3,
        prefix_visibility_mode="preserve_video_pretrain_history",
        query_stream_ids=torch.tensor([0, 1, -1]),
        cached_prefix_stream_ids=torch.tensor([0, 1, 1]),
        allow_video_query_to_action_prefix_tail_tokens=1,
    )

    assert resolved is not None
    expected_prefix = torch.tensor(
        [
            [True, False, True],
            [True, True, True],
            [False, False, False],
        ]
    )
    assert torch.equal(resolved[:, :3], expected_prefix)
    assert torch.equal(resolved[:, 3:], current_mask)


def test_slot_pool_prefix_visibility_uses_packed_sequence_ids() -> None:
    current_mask = torch.ones(3, 2, dtype=torch.bool)

    resolved = resolve_slot_pool_prefix_visibility(
        current_mask,
        prefix_len=4,
        prefix_visibility_mode="full_history",
        query_sequence_ids=torch.tensor([0, 1, 0]),
        cached_prefix_sequence_ids=torch.tensor([0, 0, 1, 1]),
    )

    assert resolved is not None
    expected_prefix = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
            [True, True, False, False],
        ]
    )
    assert torch.equal(resolved[:, :4], expected_prefix)
    assert torch.equal(resolved[:, 4:], current_mask)


def test_packed_slot_pool_query_sequence_ids_matches_exact_flattened_layout() -> None:
    profile = PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="chunked_temporal_exact_joint",
            family="chunked_exact",
            backend="torch",
        ),
        metadata={"batch_size": 3},
    )
    video_tokens_per_component = 2
    action_tokens_per_component = 1
    stream_ids = torch.cat(
        [
            torch.zeros(2 * 3 * video_tokens_per_component, dtype=torch.long),
            torch.ones(2 * 3 * action_tokens_per_component, dtype=torch.long),
            torch.full((2,), -1, dtype=torch.long),
        ],
        dim=0,
    )

    sequence_ids = packed_slot_pool_query_sequence_ids(
        attention_profile=profile,
        query_stream_ids=stream_ids,
        query_len=int(stream_ids.numel()),
        cache_batch_size=3,
        device=torch.device("cpu"),
    )

    assert sequence_ids is not None
    assert torch.equal(
        sequence_ids,
        torch.tensor(
            [
                0,
                0,
                1,
                1,
                2,
                2,
                0,
                0,
                1,
                1,
                2,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                -1,
                -1,
            ]
        ),
    )


def test_packed_slot_pool_query_sequence_ids_rejects_misaligned_stream_runs() -> None:
    profile = PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="chunked_temporal_exact_joint",
            family="chunked_exact",
            backend="torch",
        ),
        metadata={"batch_size": 3},
    )

    with pytest.raises(ValueError, match="stream run"):
        packed_slot_pool_query_sequence_ids(
            attention_profile=profile,
            query_stream_ids=torch.zeros(7, dtype=torch.long),
            query_len=7,
            cache_batch_size=3,
            device=torch.device("cpu"),
        )


def test_attention_backend_prefers_dense_mask_over_block_mask() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    attention_mask = torch.ones(2, 2, dtype=torch.bool)

    output = apply_attention_backend(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        block_mask=object(),
    )

    assert output.shape == query.shape


def test_slot_pool_update_zero_does_not_evict_persistent_history() -> None:
    payload = init_cache_backend_payload(
        "slot_pool_exact",
        num_layers=1,
        total_tokens=2,
        num_heads=1,
        head_dim=8,
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layer_state = payload.layer_states[0]
    update_slot_pool_layer_state(
        layer_state,
        key=torch.randn(2, 2, 1, 8),
        value=torch.randn(2, 2, 1, 8),
        is_pred=False,
        stream_ids=torch.tensor([0, 1]),
    )
    before_slot_mask = layer_state.slot_mask.clone()
    before_slot_ids = layer_state.slot_ids.clone()
    before_stream_ids = layer_state.stream_ids.clone()
    before_key = layer_state.key.clone()
    before_value = layer_state.value.clone()

    attention = SharedTransformerAttention(dim=8, heads=1, dim_head=8, eps=1e-6)
    hidden = torch.randn(1, 8, 8)
    attention_profile = PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="test_packed_exact",
            family="test",
            backend="torch",
        ),
        metadata={"batch_size": 2},
    )
    output, _ = attention(
        hidden,
        hidden,
        hidden,
        attention_mask=torch.ones(8, 8, dtype=torch.bool),
        attention_profile=attention_profile,
        cache_backend_name="slot_pool_exact",
        cache_backend_state=layer_state,
        cache_backend_update_mode=0,
        cache_backend_stream_ids=torch.zeros(8, dtype=torch.long),
    )

    assert output.shape == hidden.shape
    assert torch.equal(layer_state.slot_mask, before_slot_mask)
    assert torch.equal(layer_state.slot_ids, before_slot_ids)
    assert torch.equal(layer_state.stream_ids, before_stream_ids)
    assert torch.equal(layer_state.key, before_key)
    assert torch.equal(layer_state.value, before_value)


def test_slot_pool_update_write_attends_after_non_mutating_eviction(monkeypatch) -> None:
    payload = init_cache_backend_payload(
        "slot_pool_exact",
        num_layers=1,
        total_tokens=2,
        num_heads=1,
        head_dim=8,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layer_state = payload.layer_states[0]
    update_slot_pool_layer_state(
        layer_state,
        key=torch.randn(1, 2, 1, 8),
        value=torch.randn(1, 2, 1, 8),
        is_pred=False,
        stream_ids=torch.tensor([0, 1]),
    )
    captured: dict[str, tuple[int, ...]] = {}

    def fake_apply_attention_backend(
        *,
        query,
        key,
        value,
        attention_mask=None,
        block_mask=None,
        kernel_options=None,
    ):
        del value, block_mask, kernel_options
        captured["query_shape"] = tuple(query.shape)
        captured["key_shape"] = tuple(key.shape)
        captured["mask_shape"] = tuple(attention_mask.shape) if attention_mask is not None else ()
        return torch.zeros_like(query)

    monkeypatch.setattr(replica_core_module, "apply_attention_backend", fake_apply_attention_backend)

    attention = SharedTransformerAttention(dim=8, heads=1, dim_head=8, eps=1e-6)
    hidden = torch.randn(1, 1, 8)
    output, _ = attention(
        hidden,
        hidden,
        hidden,
        attention_mask=torch.ones(1, 1, dtype=torch.bool),
        cache_backend_name="slot_pool_exact",
        cache_backend_state=layer_state,
        cache_backend_update_mode=2,
        cache_backend_stream_ids=torch.zeros(1, dtype=torch.long),
    )

    assert output.shape == hidden.shape
    assert captured["query_shape"] == (1, 1, 1, 8)
    assert captured["key_shape"] == (1, 1, 2, 8)
    assert captured["mask_shape"] == (1, 1, 1, 2)
    assert int(layer_state.slot_mask.sum().item()) == 2
    valid = layer_state.slot_mask.nonzero(as_tuple=False).squeeze(-1)
    ordered = valid[torch.argsort(layer_state.slot_ids[valid], stable=True)]
    assert torch.equal(layer_state.stream_ids[ordered], torch.tensor([1, 0]))


def test_joint_clean_cache_commit_mask_matches_preserved_history_rule() -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    latents = torch.randn(1, backbone_config.latent_channels, 1, 1, 1)
    actions = torch.randn(1, 4, 1, 1, 1)

    mask = _build_joint_clean_cache_attention_mask(
        latents=latents,
        actions=actions,
        text_token_count=1,
        backbone_config=backbone_config,
        chunk_size=1,
        window_size=4,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        preserve_video_pretrain_history=True,
    )

    assert mask.shape == (2, 2)
    assert bool(mask[0, 0].item()) is True
    assert bool(mask[0, 1].item()) is False
    assert bool(mask[1, 0].item()) is True
    assert bool(mask[1, 1].item()) is True


def test_joint_clean_cache_commit_mask_counts_action_width() -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    latents = torch.randn(1, backbone_config.latent_channels, 1, 1, 1)
    actions = torch.randn(1, 4, 1, 2, 3)

    mask = _build_joint_clean_cache_attention_mask(
        latents=latents,
        actions=actions,
        text_token_count=1,
        backbone_config=backbone_config,
        chunk_size=1,
        window_size=4,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        preserve_video_pretrain_history=True,
    )

    assert mask.shape == (7, 7)


def test_joint_clean_cache_commit_mask_is_batch_local() -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    latents = torch.randn(2, backbone_config.latent_channels, 1, 1, 1)
    actions = torch.randn(2, 4, 1, 1, 1)

    mask = _build_joint_clean_cache_attention_mask(
        latents=latents,
        actions=actions,
        text_token_count=1,
        backbone_config=backbone_config,
        chunk_size=1,
        window_size=4,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        preserve_video_pretrain_history=True,
    )

    assert mask.shape == (2, 2)


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
