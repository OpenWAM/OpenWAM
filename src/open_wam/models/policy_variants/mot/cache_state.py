"""Typed cache-state movement and retention operations for MoT rollout."""

from __future__ import annotations

import torch

from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTRuntimeState,
    MoTVideoCache,
    MoTVideoLayerCache,
)


def move_mot_video_cache(
    video_cache: MoTVideoCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> MoTVideoCache:
    target_device = torch.device(device)
    moved_layers: list[MoTVideoLayerCache] = []
    for layer in video_cache.layers:
        moved_layers.append(
            MoTVideoLayerCache(
                key=layer.key.to(device=target_device, dtype=dtype if dtype is not None else layer.key.dtype),
                value=layer.value.to(device=target_device, dtype=dtype if dtype is not None else layer.value.dtype),
            )
        )
    return MoTVideoCache(
        layers=tuple(moved_layers),
        video_seq_len=video_cache.video_seq_len,
    )


def trim_mot_video_cache_tail(
    video_cache: MoTVideoCache,
    *,
    max_video_seq_len: int,
) -> MoTVideoCache:
    if max_video_seq_len <= 0:
        raise ValueError(
            "MoT video cache tail trim requires `max_video_seq_len > 0`, "
            f"got max_video_seq_len={max_video_seq_len}."
        )
    if video_cache.video_seq_len <= max_video_seq_len:
        return video_cache
    trim_start = int(video_cache.video_seq_len - max_video_seq_len)
    trimmed_layers: list[MoTVideoLayerCache] = []
    for layer in video_cache.layers:
        trimmed_layers.append(
            MoTVideoLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return MoTVideoCache(
        layers=tuple(trimmed_layers),
        video_seq_len=int(max_video_seq_len),
    )


def move_mot_action_cache(
    action_cache: MoTActionCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> MoTActionCache:
    target_device = torch.device(device)
    moved_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        moved_layers.append(
            MoTActionLayerCache(
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
    return MoTActionCache(
        layers=tuple(moved_layers),
        action_seq_len=action_cache.action_seq_len,
    )


def append_mot_action_cache(
    base_cache: MoTActionCache,
    appended_cache: MoTActionCache,
) -> MoTActionCache:
    if len(base_cache.layers) != len(appended_cache.layers):
        raise ValueError(
            "Cannot append MoT action caches with different layer counts, "
            f"got base_layers={len(base_cache.layers)}, appended_layers={len(appended_cache.layers)}."
        )
    merged_layers: list[MoTActionLayerCache] = []
    for base_layer, appended_layer in zip(base_cache.layers, appended_cache.layers, strict=True):
        merged_layers.append(
            MoTActionLayerCache(
                key=torch.cat([base_layer.key, appended_layer.key], dim=2),
                value=torch.cat([base_layer.value, appended_layer.value], dim=2),
            )
        )
    return MoTActionCache(
        layers=tuple(merged_layers),
        action_seq_len=int(base_cache.action_seq_len + appended_cache.action_seq_len),
    )


def trim_mot_action_cache_tail(
    action_cache: MoTActionCache,
    *,
    max_action_seq_len: int,
) -> MoTActionCache:
    """Drop the oldest action K/V tokens so the cache stays bounded.

    Mirrors `trim_mot_video_cache_tail`. Method-1-aligned inference uses
    this to keep the action lookback window in sync with the video cache's
    sliding window (which the shared transformer's reference cache backend
    enforces via `attn_window`). Without this trim the action cache grows
    unboundedly while video stays capped, which puts the action expert
    well past its training-time `window_size` distribution after ~10
    chunks and corrupts late-rollout action predictions.
    """

    if max_action_seq_len <= 0:
        raise ValueError(
            "MoT action cache tail trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trim_start = int(action_cache.action_seq_len - max_action_seq_len)
    trimmed_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            MoTActionLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return MoTActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def trim_mot_action_cache_prefix(
    action_cache: MoTActionCache,
    *,
    max_action_seq_len: int,
) -> MoTActionCache:
    """Keep the oldest action K/V tokens when rewinding a speculative tail."""

    if max_action_seq_len <= 0:
        raise ValueError(
            "MoT action cache prefix trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trimmed_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            MoTActionLayerCache(
                key=layer.key[:, :, :max_action_seq_len, :].contiguous(),
                value=layer.value[:, :, :max_action_seq_len, :].contiguous(),
            )
        )
    return MoTActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def rewind_mot_runtime_action_cache_to_frame(
    runtime_state: MoTRuntimeState,
    *,
    absolute_frame_start: int,
    action_tokens_per_frame: int,
) -> None:
    """Discard a speculative action-cache suffix after an environment rewind."""

    if action_tokens_per_frame <= 0:
        raise ValueError(
            "MoT action-cache rewind requires positive action_tokens_per_frame, "
            f"got {action_tokens_per_frame}."
        )
    target_frame = int(absolute_frame_start)
    action_cache = runtime_state.action_cache
    if action_cache is None:
        runtime_state.action_cache_start_frame = target_frame
        return
    if action_cache.action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "MoT action cache length must be frame-aligned before rewind, "
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
    runtime_state.action_cache = trim_mot_action_cache_prefix(
        action_cache,
        max_action_seq_len=int(keep_frames * action_tokens_per_frame),
    )
