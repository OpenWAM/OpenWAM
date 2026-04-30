from __future__ import annotations

import torch

from open_wam.models.common import RolloutCursor
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig, SharedVideoTransformerConfig
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata
from open_wam.models.visual_tower import VisualTower
from open_wam.models.visual_tower.runtime_programs import RuntimeStepOutput


def test_visual_tower_advance_runtime_cache_state_updates_cursor_metadata() -> None:
    tower = VisualTower(LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1))
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


def test_visual_tower_truncate_runtime_cache_state_applies_shared_retention_policy() -> None:
    tower = VisualTower(LingbotCompatibleVideoBackboneConfig(implementation="dummy", num_layers=1))
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


def test_prefill_exact_video_cache_materializes_self_attention_kv_for_single_stream_runtime() -> None:
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


def test_packed_exact_video_forward_reuses_positions_per_copy_and_returns_kv(monkeypatch) -> None:
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
