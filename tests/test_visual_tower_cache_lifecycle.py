from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from open_wam.models.common import RolloutCursor
from open_wam.models.common.cache_backends import SlotPoolCachePayload
from open_wam.models.video_backbone.config import (
    LingbotCompatibleVideoBackboneConfig,
    SharedVideoTransformerConfig,
)
from open_wam.models.video_backbone.contracts import (
    AttentionCacheEntry,
    CacheBranchState,
    CacheState,
    CacheUpdateMetadata,
)
from open_wam.models.visual_tower import RuntimeCacheLifecycle, VisualTower
from open_wam.models.visual_tower.runtime_programs import RuntimeStepOutput


def test_visual_tower_advance_runtime_cache_state_updates_cursor_metadata() -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=2)
    cache = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="test_stage",
        payload={"tokens_per_frame": 3},
        max_cached_frames=4,
    )

    next_cache = tower.advance_runtime_cache_state(
        cache,
        next_cursor=RolloutCursor(current_start_frame=2, block_index=1, chunk_size=2),
        payload_updates={"cache_name": "unit_test"},
        tokens_per_frame=3,
    )

    assert next_cache.current_start_frame == 2
    assert next_cache.cached_frames == 2
    assert next_cache.payload["block_index"] == 1
    assert next_cache.payload["cache_name"] == "unit_test"
    assert next_cache.payload["tokens_per_frame"] == 3
    assert next_cache.update_metadata.current_start_frame == 2
    assert next_cache.update_metadata.max_cached_frames == 4


def test_visual_tower_truncate_runtime_cache_state_applies_shared_retention_policy() -> (
    None
):
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    entry = AttentionCacheEntry(
        key=torch.randn(1, 2, 6, 4),
        value=torch.randn(1, 2, 6, 4),
        metadata={"sequence_length": 6},
    )
    cache = CacheState(
        supported=True,
        current_start_frame=0,
        cached_frames=3,
        chunk_size=1,
        capability="self_attn_plus_cross_attn",
        payload={"tokens_per_frame": 2},
        self_attention_kv=(entry,),
        cross_attention_kv=tuple(),
        update_metadata=CacheUpdateMetadata(
            current_start_frame=0,
            max_cached_frames=2,
        ),
    )

    truncated = tower.truncate_runtime_cache_state(cache)

    assert truncated.cached_frames == 2
    assert truncated.self_attention_kv[0].key is not None
    assert truncated.self_attention_kv[0].key.shape[2] == 4
    assert truncated.self_attention_kv[0].value is not None
    assert truncated.self_attention_kv[0].value.shape[2] == 4


def test_runtime_cache_lifecycle_is_plain_immutable_policy() -> None:
    lifecycle = RuntimeCacheLifecycle(
        capability="self_attn_plus_cross_attn",
        num_layers=2,
    )

    assert not isinstance(lifecycle, nn.Module)
    with pytest.raises(FrozenInstanceError):
        setattr(lifecycle, "num_layers", 3)


def test_visual_tower_cache_default_bound_is_distinct_from_explicit_unbounded() -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=2)
    )
    cursor = RolloutCursor(current_start_frame=3, block_index=1, chunk_size=4)

    default_bound = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="default_bound",
        backend_name="slot_pool_exact",
    )
    unbounded = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="unbounded",
        backend_name="slot_pool_exact",
        max_cached_frames=None,
    )

    assert default_bound.update_metadata.max_cached_frames == 4
    assert unbounded.update_metadata.max_cached_frames is None
    assert isinstance(default_bound.backend_payload, SlotPoolCachePayload)
    assert len(default_bound.backend_payload.layer_states) == 2

    tower.core.blocks = nn.ModuleList()
    after_ownership_transfer = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="after_transfer",
        backend_name="slot_pool_exact",
    )
    assert isinstance(after_ownership_transfer.backend_payload, SlotPoolCachePayload)
    assert after_ownership_transfer.backend_payload.layer_states == tuple()


def test_visual_tower_resolve_cache_preserves_existing_state_identity() -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=2)
    existing = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="existing",
        payload={"owner": "caller"},
    )

    resolved = tower.resolve_runtime_cache_state(
        existing,
        cursor=RolloutCursor(current_start_frame=8, block_index=4, chunk_size=2),
        stage="ignored",
        payload={"owner": "replacement"},
        max_cached_frames=None,
    )

    assert resolved is existing


def test_visual_tower_cache_update_metadata_preserves_policy_and_allows_overrides() -> (
    None
):
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    cache = CacheState(
        supported=True,
        current_start_frame=1,
        cached_frames=3,
        chunk_size=2,
        update_metadata=CacheUpdateMetadata(
            current_start_frame=1,
            update_kv_cache=True,
            update_cross_attention_cache=True,
            cfg_mode="separate_branches",
            max_cached_frames=7,
            sink_frames=2,
            local_attn_window=3,
            cache_branch="conditioned",
        ),
    )

    inherited = tower.build_runtime_cache_update_metadata(
        cache,
        current_start_frame=4,
    )
    overridden = tower.build_runtime_cache_update_metadata(
        cache,
        current_start_frame=6,
        update_kv_cache=True,
        update_cross_attention_cache=False,
        cfg_mode="none",
        cache_branch="default",
    )

    assert inherited == CacheUpdateMetadata(
        current_start_frame=4,
        update_kv_cache=False,
        update_cross_attention_cache=True,
        cfg_mode="separate_branches",
        max_cached_frames=7,
        sink_frames=2,
        local_attn_window=3,
        cache_branch="conditioned",
    )
    assert overridden == CacheUpdateMetadata(
        current_start_frame=6,
        update_kv_cache=True,
        update_cross_attention_cache=False,
        cfg_mode="none",
        max_cached_frames=7,
        sink_frames=2,
        local_attn_window=3,
        cache_branch="default",
    )


def test_visual_tower_cache_branch_retention_keeps_sink_and_local_tail() -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    values = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    root_self = AttentionCacheEntry(
        key=values,
        value=values + 10,
        metadata={"sequence_length": 6},
    )
    branch_self = AttentionCacheEntry(
        key=values + 20,
        value=values + 30,
        metadata={"sequence_length": 6},
    )
    cross = AttentionCacheEntry(
        key=values + 40,
        value=values + 50,
        metadata={"sequence_length": 6},
    )
    cache = CacheState(
        supported=True,
        current_start_frame=0,
        cached_frames=6,
        chunk_size=1,
        capability="self_attn_plus_cross_attn",
        payload={"tokens_per_frame": 1},
        self_attention_kv=(root_self,),
        cross_attention_kv=(cross,),
        update_metadata=CacheUpdateMetadata(
            max_cached_frames=4,
            sink_frames=1,
            local_attn_window=2,
        ),
        branch_states={
            "conditioned": CacheBranchState(
                self_attention_kv=(branch_self,),
                cross_attention_kv=(cross,),
            )
        },
    )

    truncated = tower.truncate_runtime_cache_state(cache)

    assert truncated.cached_frames == 3
    assert truncated.self_attention_kv[0].key is not None
    assert truncated.self_attention_kv[0].key.flatten().tolist() == [0.0, 4.0, 5.0]
    branch_entry = truncated.branch_states["conditioned"].self_attention_kv[0]
    assert branch_entry.key is not None
    assert branch_entry.key.flatten().tolist() == [20.0, 24.0, 25.0]
    assert truncated.cross_attention_kv[0] is cross
    assert truncated.branch_states["conditioned"].cross_attention_kv[0] is cross
    assert truncated.self_attention_kv[0].metadata["sequence_length"] == 3


def test_visual_tower_cache_branches_advance_and_clear_preserve_policy() -> None:
    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1)
    )
    cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=2)
    cache = tower.init_runtime_cache_state(
        cursor=cursor,
        stage="rollout",
        payload={"session": "unit"},
        cfg_mode="separate_branches",
        update_kv_cache=True,
        update_cross_attention_cache=True,
        max_cached_frames=9,
        sink_frames=1,
        local_attn_window=4,
    )
    branched = tower.ensure_runtime_cache_branches(
        cache,
        branch_names=("default", "conditioned", "unconditioned"),
    )
    advanced = tower.advance_runtime_cache_state(
        branched,
        next_cursor=RolloutCursor(
            current_start_frame=2,
            block_index=1,
            chunk_size=2,
        ),
        payload_updates={"phase": "next"},
        cached_frames_increment=1,
    )
    cleared = tower.clear_runtime_cache_state(
        advanced,
        cursor=RolloutCursor(
            current_start_frame=0,
            block_index=0,
            chunk_size=2,
        ),
        stage="reset",
        payload={"reason": "test"},
    )

    assert set(branched.branch_states) == {"conditioned", "unconditioned"}
    assert advanced.cached_frames == 1
    assert advanced.payload["phase"] == "next"
    assert cleared.cached_frames == 0
    assert cleared.current_start_frame == 0
    assert cleared.payload["stage"] == "reset"
    assert cleared.payload["reason"] == "test"
    assert cleared.self_attention_kv == tuple()
    assert cleared.cross_attention_kv == tuple()
    assert set(cleared.branch_states) == {"conditioned", "unconditioned"}
    assert all(
        branch.self_attention_kv == tuple() and branch.cross_attention_kv == tuple()
        for branch in cleared.branch_states.values()
    )
    assert cleared.update_metadata == CacheUpdateMetadata(
        current_start_frame=0,
        update_kv_cache=False,
        update_cross_attention_cache=False,
        cfg_mode="separate_branches",
        max_cached_frames=9,
        sink_frames=1,
        local_attn_window=4,
    )


def test_prefill_exact_video_cache_materializes_self_attention_kv_for_single_stream_runtime() -> (
    None
):
    tower = VisualTower(
        SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            max_text_tokens=4,
            load_wan_vae_frontend=False,
            load_text_conditioning=False,
            load_reference_core_weights=False,
        ),
        action_dim=4,
    )
    observed_prefix = torch.randn(1, 48, 2, 2, 2)
    text_context = torch.zeros(1, 4, 16)

    cache = tower.prefill_exact_video_cache(
        observed_prefix=observed_prefix,
        text_context=text_context,
        frame_start=0,
        cache_name="unit_test_prefill",
    )

    assert len(cache.self_attention_kv) == 2
    assert cache.self_attention_kv[0].key is not None
    assert cache.self_attention_kv[0].value is not None
    assert cache.self_attention_kv[0].key.shape[2] == 2
    assert cache.self_attention_kv[0].value.shape[2] == 2
    assert not cache.self_attention_kv[0].key.requires_grad


def test_exact_video_cache_prefill_accepts_attention_mask_and_trainable_cache() -> None:
    tower = VisualTower(
        SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            max_text_tokens=4,
            load_wan_vae_frontend=False,
            load_text_conditioning=False,
            load_reference_core_weights=False,
        ),
        action_dim=4,
    )
    observed_prefix = torch.randn(1, 48, 2, 2, 2, requires_grad=True)
    text_context = torch.zeros(1, 4, 16)
    attention_mask = torch.tril(torch.ones(2, 2, dtype=torch.bool))

    cache = tower.prefill_exact_video_cache(
        observed_prefix=observed_prefix,
        text_context=text_context,
        frame_start=0,
        cache_name="unit_test_masked_prefill",
        attention_mask=attention_mask,
        detach_cache=False,
    )
    flow_pred = tower.predict_video_flow(
        noisy_latents=observed_prefix.detach(),
        timesteps=torch.zeros(1, 2),
        text_context=text_context,
        attention_mask=attention_mask,
    )

    assert cache.self_attention_kv[0].key is not None
    assert cache.self_attention_kv[0].key.requires_grad
    assert flow_pred.shape == observed_prefix.shape


def test_packed_exact_video_forward_reuses_positions_per_copy_and_returns_kv(
    monkeypatch,
) -> None:
    tower = VisualTower(
        SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            max_text_tokens=4,
            load_wan_vae_frontend=False,
            load_text_conditioning=False,
            load_reference_core_weights=False,
        ),
        action_dim=4,
    )
    video_latents = torch.randn(1, 48, 4, 2, 2, requires_grad=True)
    timesteps = torch.zeros(1, 4)
    attention_mask = torch.ones(4, 4, dtype=torch.bool)
    captured = {}

    def fake_execute_runtime_step(step_input):
        captured["payload"] = step_input.payload
        captured["cache_name"] = step_input.cache_name
        entry = AttentionCacheEntry(
            key=torch.randn(1, 4, 4, 8, requires_grad=True),
            value=torch.randn(1, 4, 4, 8, requires_grad=True),
        )
        return RuntimeStepOutput(
            tokens=torch.zeros(1, 4, 192),
            cache_state=CacheState(
                supported=True,
                current_start_frame=7,
                cached_frames=4,
                chunk_size=4,
                self_attention_kv=(entry,),
                update_metadata=CacheUpdateMetadata(
                    current_start_frame=7,
                    update_kv_cache=True,
                ),
            ),
        )

    monkeypatch.setattr(tower, "execute_runtime_step", fake_execute_runtime_step)

    flow_pred, kv = tower.run_packed_exact_video_forward(
        video_latents=video_latents,
        timesteps=timesteps,
        text_context=None,
        frame_start=7,
        attention_mask=attention_mask,
        cache_name="unit_test_packed",
        packed_copies=2,
        detach_cache=False,
    )

    grid_id = captured["payload"]["grid_id"]
    cache = tower.core._exact_runtime_caches["unit_test_packed"]
    assert captured["cache_name"] == "unit_test_packed"
    assert captured["payload"]["attention_mask"] is attention_mask
    assert grid_id.dtype == torch.float32
    assert torch.equal(grid_id[:, :, :2], grid_id[:, :, 2:])
    assert grid_id[0, 0].tolist() == [7, 8, 7, 8]
    assert cache.payload["packed_copies"] == 2
    assert cache.payload["detach_self_attention_cache"] is False
    assert flow_pred.shape == video_latents.shape
    assert len(kv) == 1
    assert kv[0].key is not None
    assert kv[0].key.requires_grad
