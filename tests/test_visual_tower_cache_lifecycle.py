from __future__ import annotations

import torch

from open_wam.models.common import RolloutCursor
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata
from open_wam.models.visual_tower import VisualTower


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
