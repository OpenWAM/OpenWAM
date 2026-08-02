"""Backbone-agnostic runtime cache-state lifecycle policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.models.common import RolloutCursor
from open_wam.models.common.cache_backend_contracts import resolve_cache_backend_spec
from open_wam.models.common.cache_backend_lifecycle import (
    clear_cache_backend_payload,
    init_cache_backend_payload,
)
from open_wam.models.video_backbone.contracts import (
    AttentionCacheEntry,
    CacheBranchState,
    CacheState,
    CacheUpdateMetadata,
)

_MAX_CACHED_FRAMES_UNSET = object()


@dataclass(frozen=True, slots=True)
class RuntimeCacheLifecycle:
    """Apply runtime cache initialization, retention, and reset semantics.

    The lifecycle owns no modules or tensors. A visual tower supplies its
    current cache capability and layer count, while this object applies the
    shared ``CacheState`` contract.
    """

    capability: str
    num_layers: int

    def init_state(
        self,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        backend_name: str = "merged_prefix",
        backend_payload=None,
        backend_init_kwargs: dict[str, Any] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        backend_spec = resolve_cache_backend_spec(backend_name)
        resolved_max_cached_frames = (
            cursor.chunk_size
            if max_cached_frames is _MAX_CACHED_FRAMES_UNSET
            else max_cached_frames
        )
        resolved_payload = {"stage": stage, "block_index": cursor.block_index}
        if payload is not None:
            resolved_payload.update(payload)
        resolved_backend_payload = backend_payload
        if resolved_backend_payload is None:
            resolved_backend_payload = init_cache_backend_payload(
                backend_spec.name,
                num_layers=self.num_layers,
                **(backend_init_kwargs or {}),
                metadata={"stage": stage, "block_index": cursor.block_index},
            )
        return CacheState(
            supported=self.capability != "none",
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=self.capability,
            backend_name=backend_spec.name,
            backend_payload=resolved_backend_payload,
            payload=resolved_payload,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=update_kv_cache,
                update_cross_attention_cache=update_cross_attention_cache,
                cfg_mode=cfg_mode,
                max_cached_frames=resolved_max_cached_frames,
                sink_frames=sink_frames,
                local_attn_window=local_attn_window,
            ),
        )

    def resolve_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        backend_name: str = "merged_prefix",
        backend_payload=None,
        backend_init_kwargs: dict[str, Any] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        """Return an existing cache or initialize one for the rollout step."""

        if isinstance(cache_state, CacheState):
            return cache_state
        return self.init_state(
            cursor=cursor,
            stage=stage,
            payload=payload,
            backend_name=backend_name,
            backend_payload=backend_payload,
            backend_init_kwargs=backend_init_kwargs,
            cfg_mode=cfg_mode,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=update_cross_attention_cache,
            max_cached_frames=max_cached_frames,
            sink_frames=sink_frames,
            local_attn_window=local_attn_window,
        )

    def build_update_metadata(
        self,
        cache_state: CacheState,
        *,
        current_start_frame: int,
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool | None = None,
        cfg_mode: str | None = None,
        cache_branch: str | None = None,
    ) -> CacheUpdateMetadata:
        """Build one cache-update instruction from the runtime state."""

        previous_metadata = cache_state.update_metadata
        return CacheUpdateMetadata(
            current_start_frame=current_start_frame,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=(
                previous_metadata.update_cross_attention_cache
                if update_cross_attention_cache is None
                else update_cross_attention_cache
            ),
            cfg_mode=previous_metadata.cfg_mode if cfg_mode is None else cfg_mode,
            max_cached_frames=previous_metadata.max_cached_frames,
            sink_frames=previous_metadata.sink_frames,
            local_attn_window=previous_metadata.local_attn_window,
            cache_branch=(
                previous_metadata.cache_branch
                if cache_branch is None
                else cache_branch
            ),
        )

    def ensure_branches(
        self,
        cache_state: CacheState,
        *,
        branch_names: tuple[str, ...],
    ) -> CacheState:
        """Ensure named cache branches exist on a runtime cache."""

        next_branch_states = dict(cache_state.branch_states)
        for branch_name in branch_names:
            if branch_name == "default" or branch_name in next_branch_states:
                continue
            next_branch_states[branch_name] = CacheBranchState(
                backend_name=cache_state.backend_name,
                backend_payload=clear_cache_backend_payload(
                    cache_state.backend_payload
                ),
                payload={**cache_state.payload, "cache_branch": branch_name},
                self_attention_kv=tuple(),
                cross_attention_kv=tuple(),
            )
        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=cache_state.self_attention_kv,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
            branch_states=next_branch_states,
        )

    def truncate_state(
        self,
        cache_state: CacheState,
        *,
        tokens_per_frame: int | None = None,
    ) -> CacheState:
        """Apply the configured retention policy to a cache state."""

        if not cache_state.supported:
            return cache_state
        if cache_state.backend_name != "merged_prefix":
            return cache_state

        resolved_tokens_per_frame = tokens_per_frame
        if resolved_tokens_per_frame is None:
            payload_tokens_per_frame = cache_state.payload.get("tokens_per_frame")
            if (
                isinstance(payload_tokens_per_frame, int)
                and payload_tokens_per_frame > 0
            ):
                resolved_tokens_per_frame = payload_tokens_per_frame
        if resolved_tokens_per_frame is None or resolved_tokens_per_frame <= 0:
            return cache_state

        metadata = cache_state.update_metadata
        max_cached_frames = metadata.max_cached_frames
        sink_frames = max(0, metadata.sink_frames)
        local_attn_window = metadata.local_attn_window
        if max_cached_frames is None and local_attn_window is None:
            return cache_state

        sink_tokens = sink_frames * resolved_tokens_per_frame
        local_window_tokens = (
            None
            if local_attn_window is None
            else max(0, local_attn_window) * resolved_tokens_per_frame
        )
        max_cached_tokens = (
            None
            if max_cached_frames is None
            else max(0, max_cached_frames) * resolved_tokens_per_frame
        )

        truncated_self_attention = tuple(
            self._truncate_attention_cache_entry(
                entry,
                max_cached_tokens=max_cached_tokens,
                sink_tokens=sink_tokens,
                local_window_tokens=local_window_tokens,
            )
            for entry in cache_state.self_attention_kv
        )
        truncated_cross_attention = tuple(cache_state.cross_attention_kv)
        truncated_branch_states = {
            branch_name: CacheBranchState(
                backend_name=branch_state.backend_name,
                backend_payload=branch_state.backend_payload,
                payload=dict(branch_state.payload),
                self_attention_kv=tuple(
                    self._truncate_attention_cache_entry(
                        entry,
                        max_cached_tokens=max_cached_tokens,
                        sink_tokens=sink_tokens,
                        local_window_tokens=local_window_tokens,
                    )
                    for entry in branch_state.self_attention_kv
                ),
                cross_attention_kv=tuple(branch_state.cross_attention_kv),
            )
            for branch_name, branch_state in cache_state.branch_states.items()
        }

        retained_frame_cap = cache_state.cached_frames
        if max_cached_frames is not None:
            retained_frame_cap = min(retained_frame_cap, max_cached_frames)
        if local_attn_window is not None:
            retained_frame_cap = min(
                retained_frame_cap,
                sink_frames + max(0, local_attn_window),
            )

        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=retained_frame_cap,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=truncated_self_attention,
            cross_attention_kv=truncated_cross_attention,
            update_metadata=cache_state.update_metadata,
            branch_states=truncated_branch_states,
        )

    def advance_state(
        self,
        cache_state: CacheState,
        *,
        next_cursor: RolloutCursor,
        payload_updates: dict[str, object] | None = None,
        tokens_per_frame: int | None = None,
        cached_frames_increment: int | None = None,
    ) -> CacheState:
        """Advance one cache state to the next rollout cursor."""

        increment = (
            next_cursor.chunk_size
            if cached_frames_increment is None
            else cached_frames_increment
        )
        next_payload = dict(cache_state.payload)
        next_payload["block_index"] = next_cursor.block_index
        if tokens_per_frame is not None:
            next_payload["tokens_per_frame"] = tokens_per_frame
        if payload_updates is not None:
            next_payload.update(payload_updates)

        next_cache_state = CacheState(
            supported=cache_state.supported,
            current_start_frame=next_cursor.current_start_frame,
            cached_frames=cache_state.cached_frames + increment,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=next_payload,
            self_attention_kv=cache_state.self_attention_kv,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=next_cursor.current_start_frame,
                update_kv_cache=cache_state.update_metadata.update_kv_cache,
                update_cross_attention_cache=(
                    cache_state.update_metadata.update_cross_attention_cache
                ),
                cfg_mode=cache_state.update_metadata.cfg_mode,
                max_cached_frames=cache_state.update_metadata.max_cached_frames,
                sink_frames=cache_state.update_metadata.sink_frames,
                local_attn_window=cache_state.update_metadata.local_attn_window,
                cache_branch=cache_state.update_metadata.cache_branch,
            ),
            branch_states=dict(cache_state.branch_states),
        )
        return self.truncate_state(
            next_cache_state,
            tokens_per_frame=tokens_per_frame,
        )

    def clear_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> CacheState:
        """Clear cached tensors while preserving the cache policy."""

        resolved_cache = self.resolve_state(
            cache_state,
            cursor=cursor,
            stage=stage or "runtime_reset",
            payload=payload,
        )
        next_payload = dict(resolved_cache.payload)
        if payload is not None:
            next_payload.update(payload)
        if stage is not None:
            next_payload["stage"] = stage
        return CacheState(
            supported=resolved_cache.supported,
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=resolved_cache.capability,
            backend_name=resolved_cache.backend_name,
            backend_payload=clear_cache_backend_payload(
                resolved_cache.backend_payload
            ),
            payload=next_payload,
            self_attention_kv=tuple(),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=False,
                update_cross_attention_cache=False,
                cfg_mode=resolved_cache.update_metadata.cfg_mode,
                max_cached_frames=resolved_cache.update_metadata.max_cached_frames,
                sink_frames=resolved_cache.update_metadata.sink_frames,
                local_attn_window=resolved_cache.update_metadata.local_attn_window,
                cache_branch=resolved_cache.update_metadata.cache_branch,
            ),
            branch_states={
                branch_name: CacheBranchState(
                    backend_name=branch_state.backend_name,
                    backend_payload=clear_cache_backend_payload(
                        branch_state.backend_payload
                    ),
                    payload=dict(branch_state.payload),
                    self_attention_kv=tuple(),
                    cross_attention_kv=tuple(),
                )
                for branch_name, branch_state in resolved_cache.branch_states.items()
            },
        )

    def _truncate_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        *,
        max_cached_tokens: int | None,
        sink_tokens: int,
        local_window_tokens: int | None,
    ) -> AttentionCacheEntry:
        if entry.key is None or entry.value is None:
            return entry
        sequence_length = entry.key.shape[2]
        if sequence_length == 0:
            return entry

        if (
            max_cached_tokens is not None
            and sequence_length <= max_cached_tokens
        ):
            return entry

        total_tokens = sequence_length
        target_local_tokens = local_window_tokens
        if max_cached_tokens is not None:
            if sink_tokens >= max_cached_tokens:
                keep_indices = torch.arange(
                    min(max_cached_tokens, total_tokens),
                    device=entry.key.device,
                )
                return self._slice_attention_cache_entry(entry, keep_indices)
            tail_budget = max(0, max_cached_tokens - sink_tokens)
            if target_local_tokens is None:
                target_local_tokens = tail_budget
            else:
                target_local_tokens = min(target_local_tokens, tail_budget)

        if target_local_tokens is None:
            if max_cached_tokens is None:
                return entry
            keep_indices = torch.arange(
                total_tokens - max_cached_tokens,
                total_tokens,
                device=entry.key.device,
            )
            return self._slice_attention_cache_entry(entry, keep_indices)

        sink_tokens = min(sink_tokens, total_tokens)
        remaining_tokens = max(0, total_tokens - sink_tokens)
        target_local_tokens = min(target_local_tokens, remaining_tokens)
        if sink_tokens + target_local_tokens >= total_tokens:
            return entry

        head_indices = (
            torch.arange(sink_tokens, device=entry.key.device)
            if sink_tokens > 0
            else torch.empty(
                0,
                dtype=torch.long,
                device=entry.key.device,
            )
        )
        tail_indices = torch.arange(
            total_tokens - target_local_tokens,
            total_tokens,
            device=entry.key.device,
        )
        keep_indices = torch.cat((head_indices, tail_indices), dim=0)
        return self._slice_attention_cache_entry(entry, keep_indices)

    def _slice_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        keep_indices: torch.Tensor,
    ) -> AttentionCacheEntry:
        key = entry.key.index_select(2, keep_indices)
        value = entry.value.index_select(2, keep_indices)
        next_metadata = dict(entry.metadata)
        next_metadata["sequence_length"] = int(key.shape[2])
        return AttentionCacheEntry(
            key=key,
            value=value,
            metadata=next_metadata,
        )
