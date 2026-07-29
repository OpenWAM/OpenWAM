"""Per-device tensor transport for shared visual runtime execution."""

from __future__ import annotations

from dataclasses import replace

import torch

from open_wam.models.common import (
    PreparedAttentionProfile,
    SlotPoolLayerState,
    build_chunked_temporal_exact_attention_profile,
)


def move_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Move an optional tensor, casting only floating-point values."""

    if tensor is None:
        return None
    kwargs = {"device": device}
    if dtype is not None and tensor.is_floating_point():
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


def cached_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor],
    name: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Memoize one tensor copy per logical name, device, and floating dtype."""

    if tensor is None:
        return None
    dtype_key = dtype if dtype is not None and tensor.is_floating_point() else None
    cache_key = (name, torch.device(device), dtype_key)
    cached = cache.get(cache_key)
    if cached is None:
        cached = move_optional_tensor(tensor, device=torch.device(device), dtype=dtype)
        cache[cache_key] = cached
    return cached


def move_attention_profile(
    profile: PreparedAttentionProfile | None,
    *,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> PreparedAttentionProfile | None:
    """Materialize a prepared attention profile on one runtime device."""

    if profile is None:
        return None
    device = torch.device(device)
    has_block_masks = (
        profile.self_attention_block_mask is not None
        or profile.cross_attention_block_mask is not None
    )
    if has_block_masks:
        metadata = profile.metadata
        required_keys = (
            "latent_shape",
            "action_shape",
            "padded_length",
            "chunk_size",
            "window_size",
            "text_token_count",
        )
        if all(key in metadata for key in required_keys) and (
            "current_block_coupling" in metadata
            or "allow_joint_noisy_block_attention" in metadata
        ):
            current_block_coupling = metadata.get("current_block_coupling")
            if current_block_coupling is None:
                current_block_coupling = (
                    "joint"
                    if bool(metadata["allow_joint_noisy_block_attention"])
                    else "video_then_action"
                )
            return build_chunked_temporal_exact_attention_profile(
                latent_shape=tuple(int(v) for v in metadata["latent_shape"]),
                action_shape=tuple(int(v) for v in metadata["action_shape"]),
                padded_length=int(metadata["padded_length"]),
                chunk_size=int(metadata["chunk_size"]),
                window_size=int(metadata["window_size"]),
                patch_size=patch_size,
                text_token_count=int(metadata["text_token_count"]),
                base_text_token_count=(
                    None
                    if "base_text_token_count" not in metadata
                    else int(metadata["base_text_token_count"])
                ),
                proprio_context_token_count=int(metadata.get("proprio_context_token_count", 0)),
                chunk_origin_frame=int(metadata.get("chunk_origin_frame", 0)),
                prefix_condition_frames=int(metadata.get("prefix_condition_frames", 0)),
                singleton_chunk_frame=(
                    None
                    if metadata.get("singleton_chunk_frame") is None
                    else int(metadata["singleton_chunk_frame"])
                ),
                action_context_mask=(
                    torch.tensor(
                        metadata["action_context_valid_tokens"],
                        device=device,
                        dtype=torch.bool,
                    )[None, :]
                    if metadata.get("action_context_valid_tokens") is not None
                    else None
                ),
                device=device,
                build_dense_masks=(
                    profile.self_attention_mask is not None
                    or profile.cross_attention_mask is not None
                ),
                build_flex_masks=True,
                current_block_coupling=str(current_block_coupling),
                preserve_video_pretrain_history=bool(
                    metadata.get("preserve_video_pretrain_history", False)
                ),
                history_stream_visibility=metadata.get("history_stream_visibility"),
                conditional_history_policy=metadata.get("conditional_history_policy"),
            )
        if profile.self_attention_mask is None and profile.cross_attention_mask is None:
            return profile
        return replace(
            profile,
            self_attention_mask=move_optional_tensor(profile.self_attention_mask, device=device),
            cross_attention_mask=move_optional_tensor(profile.cross_attention_mask, device=device),
            self_attention_block_mask=None,
            cross_attention_block_mask=None,
        )
    return replace(
        profile,
        self_attention_mask=move_optional_tensor(profile.self_attention_mask, device=device),
        cross_attention_mask=move_optional_tensor(profile.cross_attention_mask, device=device),
    )


def cached_attention_profile(
    profile: PreparedAttentionProfile | None,
    *,
    cache: dict[torch.device, PreparedAttentionProfile | None],
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> PreparedAttentionProfile | None:
    """Memoize one prepared attention profile per runtime device."""

    device = torch.device(device)
    if device not in cache:
        cache[device] = move_attention_profile(
            profile,
            patch_size=patch_size,
            device=device,
        )
    return cache[device]


def move_slot_pool_layer_state(
    layer_state: SlotPoolLayerState | None,
    *,
    device: torch.device,
) -> SlotPoolLayerState | None:
    """Move every tensor owned by a mutable slot-pool layer state."""

    if layer_state is None:
        return None
    for name in ("key", "value", "slot_ids", "stream_ids", "slot_mask", "prediction_mask"):
        tensor = getattr(layer_state, name)
        if tensor is not None and tensor.device != device:
            setattr(layer_state, name, tensor.to(device=device))
    return layer_state


__all__ = [
    "cached_attention_profile",
    "cached_optional_tensor",
    "move_attention_profile",
    "move_optional_tensor",
    "move_slot_pool_layer_state",
]
