from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:  # pragma: no cover - older torch builds may not expose FlexAttention
    BlockMask = Any  # type: ignore[misc,assignment]
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]


_COMPILED_FLEX_ATTENTION = None
_COMPILED_CREATE_BLOCK_MASK = None


def _resolve_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        return None
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


def _resolve_compiled_create_block_mask():
    global _COMPILED_CREATE_BLOCK_MASK
    if create_block_mask is None:
        return None
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


@dataclass(frozen=True)
class AttentionProfileSpec:
    """Declarative description of a reusable attention visibility profile."""

    name: str
    family: str
    backend: str


@dataclass
class PreparedAttentionProfile:
    """Backend-ready attention visibility state.

    The profile can carry either dense boolean masks, FlexAttention block masks,
    or both. Callers choose the best representation for the current runtime.
    """

    spec: AttentionProfileSpec
    self_attention_mask: torch.Tensor | None = None
    cross_attention_mask: torch.Tensor | None = None
    self_attention_block_mask: BlockMask | None = None
    cross_attention_block_mask: BlockMask | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_ATTENTION_PROFILE_ALIASES: dict[str, str] = {
    "chunked_temporal_exact": "chunked_temporal_exact",
    "lingbot_chunked_exact": "chunked_temporal_exact",
    "none": "none",
}


def normalize_attention_profile_name(name: str | None) -> str | None:
    if name is None:
        return None
    try:
        return _ATTENTION_PROFILE_ALIASES[name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported attention profile {name!r}. Expected one of {tuple(_ATTENTION_PROFILE_ALIASES)}."
        ) from exc


def resolve_attention_profile_backend(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> str:
    if profile is None:
        return "none"
    if prefer_flex and device.type == "cuda":
        if is_cross_attention and profile.cross_attention_block_mask is not None:
            return "lingbot_flex"
        if not is_cross_attention and profile.self_attention_block_mask is not None:
            return "lingbot_flex"
    return "sdpa" if (
        (is_cross_attention and profile.cross_attention_mask is not None)
        or (not is_cross_attention and profile.self_attention_mask is not None)
    ) else "none"


def select_attention_profile_mask(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> tuple[torch.Tensor | None, BlockMask | None]:
    backend = resolve_attention_profile_backend(
        profile,
        device=device,
        prefer_flex=prefer_flex,
        is_cross_attention=is_cross_attention,
    )
    if profile is None or backend == "none":
        return None, None
    if backend == "lingbot_flex":
        return (
            None,
            profile.cross_attention_block_mask if is_cross_attention else profile.self_attention_block_mask,
        )
    return (
        (profile.cross_attention_mask if is_cross_attention else profile.self_attention_mask),
        None,
    )


def apply_attention_backend(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    block_mask: BlockMask | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    if block_mask is not None:
        if flex_attention is None:
            raise RuntimeError("FlexAttention is not available in this torch build.")
        compiled_flex_attention = _resolve_compiled_flex_attention()
        if compiled_flex_attention is not None:
            return compiled_flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)
        return flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)


def build_chunked_temporal_exact_attention_profile(
    *,
    latent_shape: tuple[int, int, int, int, int],
    action_shape: tuple[int, int, int, int, int],
    padded_length: int,
    chunk_size: int,
    window_size: int,
    patch_size: tuple[int, int, int],
    text_token_count: int,
    device: torch.device,
    build_dense_masks: bool = False,
    build_flex_masks: bool = False,
) -> PreparedAttentionProfile:
    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    _, _, action_frames, action_height, action_width = action_shape
    patch_t, patch_h, patch_w = patch_size

    latent_seq_id = (
        torch.arange(batch_size, device=device)[:, None, None, None]
        .expand(-1, latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w)
        .flatten()
    )
    action_seq_id = (
        torch.arange(batch_size, device=device)[:, None, None, None]
        .expand(-1, action_frames, action_height, action_width)
        .flatten()
    )
    seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

    latent_frame_id = (
        torch.arange(latent_frames // patch_t, device=device)[None, :, None, None]
        .expand(batch_size, -1, latent_height // patch_h, latent_width // patch_w)[None]
        .flatten()
    )
    action_frame_id = (
        torch.arange(action_frames, device=device)[None, :, None, None]
        .expand(batch_size, -1, action_height, action_width)[None]
        .flatten()
    )
    frame_ids = torch.cat(
        [latent_frame_id // chunk_size * 2] * 2
        + [action_frame_id // chunk_size * 2 + 1] * 2
    )
    noise_ids = torch.cat(
        [
            torch.zeros_like(latent_frame_id),
            torch.ones_like(latent_frame_id),
            torch.zeros_like(action_frame_id),
            torch.ones_like(action_frame_id),
        ]
    )

    if padded_length > 0:
        seq_ids = torch.nn.functional.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = torch.nn.functional.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = torch.nn.functional.pad(noise_ids, (0, padded_length), value=-1)

    text_seq_ids = torch.arange(batch_size, device=device)[:, None].expand(-1, text_token_count).flatten()

    self_attention_mask = None
    cross_attention_mask = None
    if build_dense_masks:
        q_seq = seq_ids[:, None]
        kv_seq = seq_ids[None, :]
        q_frame = frame_ids[:, None]
        kv_frame = frame_ids[None, :]
        q_noise = noise_ids[:, None]
        kv_noise = noise_ids[None, :]

        same_seq = (q_seq == kv_seq) & (q_seq >= 0) & (kv_seq >= 0)
        clean_to_clean = (q_noise == 1) & (kv_noise == 1) & (kv_frame <= q_frame)
        noise_to_clean = (q_noise == 0) & (kv_noise == 1) & (kv_frame < q_frame)
        noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_frame == q_frame)
        within_window = (q_frame - kv_frame).abs() <= int(window_size)
        self_attention_mask = same_seq & within_window & (clean_to_clean | noise_to_clean | noise_to_noise)
        cross_attention_mask = (
            (seq_ids[:, None] == text_seq_ids[None, :])
            & (seq_ids[:, None] >= 0)
            & (text_seq_ids[None, :] >= 0)
        )

    self_attention_block_mask = None
    cross_attention_block_mask = None
    if build_flex_masks and create_block_mask is not None:
        seq_ids_flex = seq_ids.to(device=device, dtype=torch.long)
        frame_ids_flex = frame_ids.to(device=device, dtype=torch.long)
        noise_ids_flex = noise_ids.to(device=device, dtype=torch.long)
        text_seq_ids_flex = text_seq_ids.to(device=device, dtype=torch.long)

        def self_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            same_seq = (
                (seq_ids_flex[q_idx] == seq_ids_flex[kv_idx])
                & (seq_ids_flex[q_idx] >= 0)
                & (seq_ids_flex[kv_idx] >= 0)
            )
            clean_to_clean = (
                (noise_ids_flex[q_idx] == 1)
                & (noise_ids_flex[kv_idx] == 1)
                & (frame_ids_flex[kv_idx] <= frame_ids_flex[q_idx])
            )
            noise_to_clean = (
                (noise_ids_flex[q_idx] == 0)
                & (noise_ids_flex[kv_idx] == 1)
                & (frame_ids_flex[kv_idx] < frame_ids_flex[q_idx])
            )
            noise_to_noise = (
                (noise_ids_flex[q_idx] == 0)
                & (noise_ids_flex[kv_idx] == 0)
                & (frame_ids_flex[kv_idx] == frame_ids_flex[q_idx])
            )
            within_window = (frame_ids_flex[q_idx] - frame_ids_flex[kv_idx]).abs() <= int(window_size)
            return same_seq & within_window & (clean_to_clean | noise_to_clean | noise_to_noise)

        def cross_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return (
                (seq_ids_flex[q_idx] == text_seq_ids_flex[kv_idx])
                & (seq_ids_flex[q_idx] >= 0)
                & (text_seq_ids_flex[kv_idx] >= 0)
            )

        total_seq_len = int(seq_ids.numel())
        total_text_len = int(text_seq_ids.numel())
        compiled_create_block_mask = _resolve_compiled_create_block_mask()
        block_mask_builder = compiled_create_block_mask or create_block_mask
        self_attention_block_mask = block_mask_builder(
            self_mask_mod,
            1,
            1,
            total_seq_len,
            total_seq_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )
        cross_attention_block_mask = block_mask_builder(
            cross_mask_mod,
            1,
            1,
            total_seq_len,
            total_text_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )

    return PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="chunked_temporal_exact",
            family="chunked_exact",
            backend="flex_or_sdpa",
        ),
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        self_attention_block_mask=self_attention_block_mask,
        cross_attention_block_mask=cross_attention_block_mask,
        metadata={
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "window_size": int(window_size),
            "latent_shape": tuple(int(v) for v in latent_shape),
            "action_shape": tuple(int(v) for v in action_shape),
            "padded_length": int(padded_length),
            "text_token_count": int(text_token_count),
        },
    )


def build_lingbot_chunked_exact_attention_profile(**kwargs) -> PreparedAttentionProfile:
    return build_chunked_temporal_exact_attention_profile(**kwargs)
