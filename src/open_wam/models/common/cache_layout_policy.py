"""Parameter-free layout and retention policy for runtime cache prefixes."""

from __future__ import annotations

import math

import torch

from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.video_backbone.contracts import AttentionCacheEntry

from .cache_backend_contracts import (
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    SlotPoolLayerState,
)

__all__ = [
    "merge_attention_cache_entries",
    "packed_slot_pool_query_sequence_ids",
    "prepend_cached_prefix_mask",
    "prepare_sdpa_mask",
    "resolve_slot_pool_prefix_visibility",
    "retained_slot_pool_indices_for_current_write",
]


def prepare_sdpa_mask(
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize a dense attention mask to SDPA's batch/head layout."""

    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        return attention_mask[None, None, :, :].to(device=device)
    if attention_mask.ndim == 3:
        return attention_mask[:, None, :, :].to(device=device)
    if attention_mask.ndim == 4:
        return attention_mask.to(device=device)
    raise ValueError(
        "Expected attention mask with shape [seq, seq], [B, seq, seq], or [B, H, seq, seq], "
        f"got {tuple(attention_mask.shape)}"
    )


def prepend_cached_prefix_mask(
    attention_mask: torch.Tensor | None,
    *,
    cached_prefix_visibility: torch.Tensor | None,
    prefix_len: int,
    cached_segment_lengths: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    """Prepend cached-token visibility to a current-token attention mask."""

    if attention_mask is None or cached_prefix_visibility is None or prefix_len <= 0:
        return attention_mask
    visibility = cached_prefix_visibility
    visibility_width = int(visibility.shape[-1])
    if visibility_width != prefix_len:
        segment_lengths = tuple(
            int(length) for length in (cached_segment_lengths or ())
        )
        if not segment_lengths:
            segment_lengths = (visibility_width,)
        prefix_chunks: list[torch.Tensor] = []
        source_offset = 0
        remaining_prefix = prefix_len
        for segment_length in segment_lengths:
            if segment_length <= 0 or remaining_prefix <= 0:
                continue
            take = min(segment_length, remaining_prefix)
            if source_offset >= visibility_width:
                source_offset = 0
            source_end = min(source_offset + take, visibility_width)
            chunk = visibility[..., source_offset:source_end]
            if chunk.shape[-1] < take:
                # When the current visibility span is narrower than the total
                # cached prefix, repeat the source pattern across cached
                # segments. This keeps the mask width aligned with merged cache
                # entries produced by repeated warmup passes.
                repeat_factor = math.ceil(take / max(chunk.shape[-1], 1))
                repeats = [1] * chunk.ndim
                repeats[-1] = repeat_factor
                chunk = chunk.repeat(*repeats)[..., :take]
            prefix_chunks.append(chunk)
            source_offset = (source_offset + take) % max(visibility_width, 1)
            remaining_prefix -= take
        if remaining_prefix > 0:
            repeat_factor = math.ceil(remaining_prefix / max(visibility_width, 1))
            repeats = [1] * visibility.ndim
            repeats[-1] = repeat_factor
            tail = visibility.repeat(*repeats)[..., :remaining_prefix]
            prefix_chunks.append(tail)
        visibility = torch.cat(prefix_chunks, dim=-1)
    if attention_mask.ndim == 2:
        if visibility.ndim == 3:
            visibility = visibility[0]
        prefix = visibility
        return torch.cat(
            [prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1
        )
    if attention_mask.ndim == 3:
        prefix = visibility
        return torch.cat(
            [prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1
        )
    if attention_mask.ndim == 4:
        prefix = visibility[:, None, :, :].expand(
            -1,
            attention_mask.shape[1],
            -1,
            prefix_len,
        )
        return torch.cat(
            [prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1
        )
    raise ValueError(
        "Expected attention mask with shape [seq, seq], [B, seq, seq], or [B, H, seq, seq], "
        f"got {tuple(attention_mask.shape)}"
    )


def resolve_slot_pool_prefix_visibility(
    attention_mask: torch.Tensor | None,
    *,
    prefix_len: int,
    prefix_visibility_mode: str,
    query_stream_ids: torch.Tensor | None = None,
    cached_prefix_stream_ids: torch.Tensor | None = None,
    query_sequence_ids: torch.Tensor | None = None,
    cached_prefix_sequence_ids: torch.Tensor | None = None,
    allow_video_query_to_action_prefix_tail_tokens: int = 0,
) -> torch.Tensor | None:
    """Resolve slot-pool stream and packed-sequence visibility."""

    if attention_mask is None or prefix_len <= 0:
        return attention_mask

    def _normalize_stream_ids(
        stream_ids: torch.Tensor | None,
        *,
        expected_len: int,
        label: str,
    ) -> torch.Tensor:
        if stream_ids is None:
            raise ValueError(
                f"Slot-pool prefix_visibility_mode={prefix_visibility_mode!r} requires `{label}`."
            )
        if stream_ids.ndim == 2:
            if stream_ids.shape[0] != 1:
                raise ValueError(
                    f"Slot-pool `{label}` must be rank-1 or batch-shared rank-2, "
                    f"got shape {tuple(stream_ids.shape)}."
                )
            stream_ids = stream_ids.squeeze(0)
        if stream_ids.ndim != 1 or int(stream_ids.shape[0]) != expected_len:
            raise ValueError(
                f"Slot-pool `{label}` must have length {expected_len}, "
                f"got shape {tuple(stream_ids.shape)}."
            )
        return stream_ids.to(device=attention_mask.device, dtype=torch.long)

    if prefix_visibility_mode == "full_history":
        cached_prefix_visibility_2d = torch.ones(
            attention_mask.shape[-2],
            prefix_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
    elif prefix_visibility_mode == "video_queries_video_only":
        q_stream = _normalize_stream_ids(
            query_stream_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_stream_ids",
        )
        kv_stream = _normalize_stream_ids(
            cached_prefix_stream_ids,
            expected_len=prefix_len,
            label="cached_prefix_stream_ids",
        )
        valid_streams = (q_stream[:, None] >= 0) & (kv_stream[None, :] >= 0)
        cached_prefix_visibility_2d = (
            (q_stream[:, None] == kv_stream[None, :]) | (q_stream[:, None] == 1)
        ) & valid_streams
        tail_tokens = max(
            0, min(int(allow_video_query_to_action_prefix_tail_tokens), int(prefix_len))
        )
        if tail_tokens > 0:
            tail_positions = torch.arange(prefix_len, device=attention_mask.device) >= (
                prefix_len - tail_tokens
            )
            # Staged action-then-video commits the current clean action before
            # denoising current video. Training permits that same-current-chunk
            # action context while still hiding older action history from video.
            cached_prefix_visibility_2d = cached_prefix_visibility_2d | (
                (q_stream[:, None] == 0)
                & (kv_stream[None, :] == 1)
                & tail_positions[None, :]
                & valid_streams
            )
        cached_prefix_visibility_2d = cached_prefix_visibility_2d.to(
            dtype=attention_mask.dtype
        )
    elif prefix_visibility_mode == "video_history_only":
        q_stream = _normalize_stream_ids(
            query_stream_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_stream_ids",
        )
        kv_stream = _normalize_stream_ids(
            cached_prefix_stream_ids,
            expected_len=prefix_len,
            label="cached_prefix_stream_ids",
        )
        valid_streams = (q_stream[:, None] >= 0) & (kv_stream[None, :] >= 0)
        cached_prefix_visibility_2d = (kv_stream[None, :] == 0) & valid_streams
        tail_tokens = max(
            0, min(int(allow_video_query_to_action_prefix_tail_tokens), int(prefix_len))
        )
        if tail_tokens > 0:
            tail_positions = torch.arange(prefix_len, device=attention_mask.device) >= (
                prefix_len - tail_tokens
            )
            cached_prefix_visibility_2d = cached_prefix_visibility_2d | (
                (q_stream[:, None] == 0)
                & (kv_stream[None, :] == 1)
                & tail_positions[None, :]
                & valid_streams
            )
        cached_prefix_visibility_2d = cached_prefix_visibility_2d.to(
            dtype=attention_mask.dtype
        )
    else:
        raise ValueError(
            f"Unsupported slot-pool prefix_visibility_mode {prefix_visibility_mode!r}."
        )
    if query_sequence_ids is not None or cached_prefix_sequence_ids is not None:
        q_seq = _normalize_stream_ids(
            query_sequence_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_sequence_ids",
        )
        kv_seq = _normalize_stream_ids(
            cached_prefix_sequence_ids,
            expected_len=prefix_len,
            label="cached_prefix_sequence_ids",
        )
        same_sequence = (
            (q_seq[:, None] == kv_seq[None, :])
            & (q_seq[:, None] >= 0)
            & (kv_seq[None, :] >= 0)
        )
        if cached_prefix_visibility_2d.dtype == torch.bool:
            cached_prefix_visibility_2d = cached_prefix_visibility_2d & same_sequence
        else:
            cached_prefix_visibility_2d = (
                cached_prefix_visibility_2d
                * same_sequence.to(dtype=cached_prefix_visibility_2d.dtype)
            )

    if attention_mask.ndim == 2:
        cached_prefix_visibility = cached_prefix_visibility_2d
    elif attention_mask.ndim == 3:
        cached_prefix_visibility = cached_prefix_visibility_2d[None].expand(
            attention_mask.shape[0],
            -1,
            -1,
        )
    elif attention_mask.ndim == 4:
        cached_prefix_visibility = cached_prefix_visibility_2d[None, None].expand(
            attention_mask.shape[0],
            attention_mask.shape[1],
            -1,
            -1,
        )
    else:  # pragma: no cover - defensive guard
        raise ValueError(
            "Unsupported attention mask rank while resolving slot-pool prefix visibility: "
            f"{tuple(attention_mask.shape)}"
        )
    return prepend_cached_prefix_mask(
        attention_mask,
        cached_prefix_visibility=cached_prefix_visibility,
        prefix_len=prefix_len,
    )


def packed_slot_pool_query_sequence_ids(
    *,
    attention_profile: PreparedAttentionProfile | None,
    query_stream_ids: torch.Tensor | None,
    query_len: int,
    cache_batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return sequence ids for exact joint current tokens packed into batch 1."""

    if attention_profile is None:
        return None
    profile_batch_size = int(attention_profile.metadata.get("batch_size", 0))
    if profile_batch_size != int(cache_batch_size) or profile_batch_size <= 1:
        return None
    if query_stream_ids is None:
        return None
    stream_ids = query_stream_ids.to(device=device, dtype=torch.long)
    if stream_ids.ndim == 2:
        if int(stream_ids.shape[0]) != 1:
            raise ValueError(
                "Packed slot-pool query stream ids must be rank-1 or batch-shared rank-2, "
                f"got shape {tuple(stream_ids.shape)}."
            )
        stream_ids = stream_ids.squeeze(0)
    if stream_ids.ndim != 1 or int(stream_ids.shape[0]) != int(query_len):
        raise ValueError(
            "Packed slot-pool query stream ids must have one value per current KV token, "
            f"got shape {tuple(stream_ids.shape)} for query_len={int(query_len)}."
        )

    sequence_parts: list[torch.Tensor] = []
    offset = 0
    while offset < int(query_len):
        stream_value = int(stream_ids[offset].item())
        run_end = offset + 1
        while (
            run_end < int(query_len) and int(stream_ids[run_end].item()) == stream_value
        ):
            run_end += 1
        run_length = run_end - offset
        if stream_value < 0:
            sequence_parts.append(
                torch.full((run_length,), -1, device=device, dtype=torch.long)
            )
        else:
            if run_length % (2 * profile_batch_size) != 0:
                raise ValueError(
                    "Packed exact slot-pool stream run must contain noisy+condition components "
                    "for every packed sequence row, "
                    f"got run_length={run_length}, packed_batch={profile_batch_size}."
                )
            tokens_per_component = run_length // (2 * profile_batch_size)
            component_ids = torch.arange(
                profile_batch_size, device=device, dtype=torch.long
            ).repeat_interleave(tokens_per_component)
            sequence_parts.append(torch.cat([component_ids, component_ids], dim=0))
        offset = run_end
    return torch.cat(sequence_parts, dim=0)


def retained_slot_pool_indices_for_current_write(
    layer_state: SlotPoolLayerState,
    *,
    valid: torch.Tensor,
    current_token_count: int,
    update_mode: int,
) -> torch.Tensor:
    """Return the prefix slots visible after a non-mutating slot allocation."""

    if (
        int(update_mode) == 0
        or int(current_token_count) <= 0
        or int(valid.numel()) == 0
    ):
        return valid
    if layer_state.slot_mask is None or layer_state.slot_ids is None:
        raise ValueError(
            "Slot-pool backend requires initialized `slot_mask` and `slot_ids` tensors."
        )
    if bool(
        layer_state.metadata.get(
            SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION, False
        )
    ):
        return valid
    free_count = int(layer_state.slot_mask.numel()) - int(valid.numel())
    evict_count = max(0, int(current_token_count) - free_count)
    if evict_count <= 0:
        return valid
    if evict_count >= int(valid.numel()):
        return valid.new_empty((0,), dtype=valid.dtype)
    slot_ids = layer_state.slot_ids[valid]
    order = torch.argsort(slot_ids, stable=True)
    return valid[order[evict_count:]]


def merge_attention_cache_entries(
    existing: AttentionCacheEntry | None,
    new_entry: AttentionCacheEntry | None,
    *,
    max_tokens: int | None,
) -> AttentionCacheEntry:
    """Merge and optionally tail-truncate two prefix-cache entries."""

    if new_entry is None or new_entry.key is None or new_entry.value is None:
        return existing if existing is not None else AttentionCacheEntry()
    if existing is not None and existing.key is not None and existing.value is not None:
        key = torch.cat([existing.key, new_entry.key], dim=2)
        value = torch.cat([existing.value, new_entry.value], dim=2)
        metadata = dict(existing.metadata)
    else:
        key = new_entry.key
        value = new_entry.value
        metadata = {}
    existing_segments = metadata.get("segment_token_lengths")
    if existing_segments is None:
        existing_segments_tuple: tuple[int, ...] = tuple()
        if existing is not None and existing.key is not None:
            existing_segments_tuple = (int(existing.key.shape[2]),)
    else:
        existing_segments_tuple = tuple(int(length) for length in existing_segments)
    new_segments = new_entry.metadata.get("segment_token_lengths")
    if new_segments is None:
        new_segments_tuple = (int(new_entry.key.shape[2]),)
    else:
        new_segments_tuple = tuple(int(length) for length in new_segments)
    segment_token_lengths = existing_segments_tuple + new_segments_tuple
    if max_tokens is not None and key.shape[2] > max_tokens:
        trimmed_segments: list[int] = []
        remaining = max_tokens
        for segment_length in reversed(segment_token_lengths):
            if remaining <= 0:
                break
            take = min(segment_length, remaining)
            trimmed_segments.append(take)
            remaining -= take
        segment_token_lengths = tuple(reversed(trimmed_segments))
        key = key[:, :, -max_tokens:, :]
        value = value[:, :, -max_tokens:, :]
    metadata.update(new_entry.metadata)
    metadata["cached_tokens"] = int(key.shape[2])
    metadata["segment_token_lengths"] = segment_token_lengths
    return AttentionCacheEntry(key=key, value=value, metadata=metadata)
