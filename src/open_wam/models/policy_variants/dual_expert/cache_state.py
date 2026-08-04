"""Typed cache-state movement and retention operations for DualExpert rollout."""

from __future__ import annotations

import torch

from .contracts import (
    DualExpertActionCache,
    DualExpertActionLayerCache,
    DualExpertRuntimeState,
    DualExpertVideoCache,
    DualExpertVideoLayerCache,
)


def move_dual_expert_video_cache(
    video_cache: DualExpertVideoCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> DualExpertVideoCache:
    target_device = torch.device(device)
    moved_layers: list[DualExpertVideoLayerCache] = []
    for layer in video_cache.layers:
        moved_layers.append(
            DualExpertVideoLayerCache(
                key=layer.key.to(device=target_device, dtype=dtype if dtype is not None else layer.key.dtype),
                value=layer.value.to(device=target_device, dtype=dtype if dtype is not None else layer.value.dtype),
            )
        )
    return DualExpertVideoCache(
        layers=tuple(moved_layers),
        video_seq_len=video_cache.video_seq_len,
    )


def trim_dual_expert_video_cache_tail(
    video_cache: DualExpertVideoCache,
    *,
    max_video_seq_len: int,
) -> DualExpertVideoCache:
    if max_video_seq_len <= 0:
        raise ValueError(
            "DualExpert video cache tail trim requires `max_video_seq_len > 0`, "
            f"got max_video_seq_len={max_video_seq_len}."
        )
    if video_cache.video_seq_len <= max_video_seq_len:
        return video_cache
    trim_start = int(video_cache.video_seq_len - max_video_seq_len)
    trimmed_layers: list[DualExpertVideoLayerCache] = []
    for layer in video_cache.layers:
        trimmed_layers.append(
            DualExpertVideoLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return DualExpertVideoCache(
        layers=tuple(trimmed_layers),
        video_seq_len=int(max_video_seq_len),
    )


def move_dual_expert_action_cache(
    action_cache: DualExpertActionCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> DualExpertActionCache:
    target_device = torch.device(device)
    moved_layers: list[DualExpertActionLayerCache] = []
    for layer in action_cache.layers:
        moved_layers.append(
            DualExpertActionLayerCache(
                key=layer.key.to(
                    device=target_device,
                    dtype=dtype if dtype is not None else layer.key.dtype,
                ),
                value=layer.value.to(
                    device=target_device,
                    dtype=dtype if dtype is not None else layer.value.dtype,
                ),
            )
        )
    return DualExpertActionCache(
        layers=tuple(moved_layers),
        action_seq_len=action_cache.action_seq_len,
    )


def append_dual_expert_action_cache(
    base_cache: DualExpertActionCache,
    appended_cache: DualExpertActionCache,
) -> DualExpertActionCache:
    if len(base_cache.layers) != len(appended_cache.layers):
        raise ValueError(
            "Cannot append DualExpert action caches with different layer counts, "
            f"got base_layers={len(base_cache.layers)}, appended_layers={len(appended_cache.layers)}."
        )
    merged_layers: list[DualExpertActionLayerCache] = []
    for base_layer, appended_layer in zip(base_cache.layers, appended_cache.layers, strict=True):
        merged_layers.append(
            DualExpertActionLayerCache(
                key=torch.cat([base_layer.key, appended_layer.key], dim=2),
                value=torch.cat([base_layer.value, appended_layer.value], dim=2),
            )
        )
    return DualExpertActionCache(
        layers=tuple(merged_layers),
        action_seq_len=int(base_cache.action_seq_len + appended_cache.action_seq_len),
    )


def trim_dual_expert_action_cache_tail(
    action_cache: DualExpertActionCache,
    *,
    max_action_seq_len: int,
) -> DualExpertActionCache:
    """Drop the oldest action K/V tokens so the cache stays bounded.

    Mirrors `trim_dual_expert_video_cache_tail`. parallel-stream-aligned inference uses
    this to keep the action lookback window in sync with the video cache's
    sliding window (which the shared transformer's reference cache backend
    enforces via `attn_window`). Without this trim the action cache grows
    unboundedly while video stays capped, which puts the action expert
    well past its training-time `window_size` distribution after ~10
    chunks and corrupts late-rollout action predictions.
    """

    if max_action_seq_len <= 0:
        raise ValueError(
            "DualExpert action cache tail trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trim_start = int(action_cache.action_seq_len - max_action_seq_len)
    trimmed_layers: list[DualExpertActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            DualExpertActionLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return DualExpertActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def trim_dual_expert_action_cache_prefix(
    action_cache: DualExpertActionCache,
    *,
    max_action_seq_len: int,
) -> DualExpertActionCache:
    """Keep the oldest action K/V tokens when rewinding a speculative tail."""

    if max_action_seq_len <= 0:
        raise ValueError(
            "DualExpert action cache prefix trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trimmed_layers: list[DualExpertActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            DualExpertActionLayerCache(
                key=layer.key[:, :, :max_action_seq_len, :].contiguous(),
                value=layer.value[:, :, :max_action_seq_len, :].contiguous(),
            )
        )
    return DualExpertActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def rewind_dual_expert_runtime_action_cache_to_frame(
    runtime_state: DualExpertRuntimeState,
    *,
    absolute_frame_start: int,
    action_tokens_per_frame: int,
) -> None:
    """Discard a speculative action-cache suffix after an environment rewind."""

    if action_tokens_per_frame <= 0:
        raise ValueError(
            "DualExpert action-cache rewind requires positive action_tokens_per_frame, "
            f"got {action_tokens_per_frame}."
        )
    target_frame = int(absolute_frame_start)
    action_cache = runtime_state.action_cache
    if action_cache is None:
        runtime_state.action_cache_start_frame = target_frame
        return
    if action_cache.action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "DualExpert action cache length must be frame-aligned before rewind, "
            f"got action_seq_len={action_cache.action_seq_len}, "
            f"action_tokens_per_frame={action_tokens_per_frame}."
        )
    cache_start_frame = int(runtime_state.action_cache_start_frame)
    keep_frames = target_frame - cache_start_frame
    if keep_frames <= 0:
        runtime_state.action_cache = None
        runtime_state.action_cache_start_frame = target_frame
        return
    cached_frames = action_cache.action_seq_len // action_tokens_per_frame
    if keep_frames >= cached_frames:
        return
    runtime_state.action_cache = trim_dual_expert_action_cache_prefix(
        action_cache,
        max_action_seq_len=int(keep_frames * action_tokens_per_frame),
    )
