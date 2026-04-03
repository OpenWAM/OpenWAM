from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata


@dataclass(frozen=True)
class RegisterSequenceLayout:
    """DreamZero-style video plus register layout over one packed sequence."""

    clean_video_span: tuple[int, int]
    noisy_video_span: tuple[int, int]
    first_noisy_frame_span: tuple[int, int]
    noisy_video_block_spans: tuple[tuple[int, int], ...]
    action_block_spans: tuple[tuple[int, int], ...]
    state_block_spans: tuple[tuple[int, int], ...]
    clean_video_sequence_length: int
    noisy_video_sequence_length: int
    total_sequence_length: int
    num_image_blocks: int
    num_action_blocks: int
    num_state_blocks: int
    tokens_per_frame: int
    tokens_per_image_block: int
    num_video_frames: int
    has_clean_video_prefix: bool


def build_register_sequence_layout(
    token_grid: TokenGridMetadata,
    action_horizon: int,
    state_horizon: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    *,
    include_clean_video_prefix: bool,
    include_register_tokens: bool = True,
    require_matching_block_counts: bool = True,
) -> RegisterSequenceLayout:
    if token_grid.num_frames < 1:
        raise ValueError("Register-attached variant requires at least one frame.")
    if (token_grid.num_frames - 1) % num_frame_per_block != 0:
        raise ValueError(
            "Expected `(num_frames - 1)` to be divisible by `num_frame_per_block`, "
            f"got num_frames={token_grid.num_frames}, num_frame_per_block={num_frame_per_block}"
        )
    if include_register_tokens and action_horizon % num_action_per_block != 0:
        raise ValueError(
            "Expected `action_horizon` to be divisible by `num_action_per_block`, "
            f"got action_horizon={action_horizon}, num_action_per_block={num_action_per_block}"
        )
    if include_register_tokens and state_horizon % num_state_per_block != 0:
        raise ValueError(
            "Expected `state_horizon` to be divisible by `num_state_per_block`, "
            f"got state_horizon={state_horizon}, num_state_per_block={num_state_per_block}"
        )
    num_image_blocks = (token_grid.num_frames - 1) // num_frame_per_block
    num_action_blocks = action_horizon // num_action_per_block if include_register_tokens else 0
    num_state_blocks = state_horizon // num_state_per_block if include_register_tokens else 0
    if (
        include_register_tokens
        and require_matching_block_counts
        and (num_image_blocks != num_action_blocks or num_image_blocks != num_state_blocks)
    ):
        raise ValueError(
            "Expected image, action, and state block counts to match, "
            f"got image={num_image_blocks}, action={num_action_blocks}, state={num_state_blocks}"
        )

    tokens_per_frame = token_grid.tokens_per_frame
    clean_video_length = token_grid.sequence_length if include_clean_video_prefix else 0
    clean_video_span = (0, clean_video_length)
    cursor = clean_video_length

    first_noisy_frame_span = (cursor, cursor + tokens_per_frame)
    cursor += tokens_per_frame
    noisy_video_block_spans: list[tuple[int, int]] = []
    for _ in range(num_image_blocks):
        block_tokens = num_frame_per_block * tokens_per_frame
        noisy_video_block_spans.append((cursor, cursor + block_tokens))
        cursor += block_tokens
    noisy_video_span = (first_noisy_frame_span[0], cursor)
    noisy_video_sequence_length = noisy_video_span[1] - noisy_video_span[0]

    action_block_spans: list[tuple[int, int]] = []
    register_cursor = cursor
    if include_register_tokens:
        for _ in range(num_action_blocks):
            action_block_spans.append((register_cursor, register_cursor + num_action_per_block))
            register_cursor += num_action_per_block

    state_block_spans: list[tuple[int, int]] = []
    if include_register_tokens:
        for _ in range(num_state_blocks):
            state_block_spans.append((register_cursor, register_cursor + num_state_per_block))
            register_cursor += num_state_per_block

    return RegisterSequenceLayout(
        clean_video_span=clean_video_span,
        noisy_video_span=noisy_video_span,
        first_noisy_frame_span=first_noisy_frame_span,
        noisy_video_block_spans=tuple(noisy_video_block_spans),
        action_block_spans=tuple(action_block_spans),
        state_block_spans=tuple(state_block_spans),
        clean_video_sequence_length=clean_video_length,
        noisy_video_sequence_length=noisy_video_sequence_length,
        total_sequence_length=register_cursor,
        num_image_blocks=num_image_blocks,
        num_action_blocks=num_action_blocks,
        num_state_blocks=num_state_blocks,
        tokens_per_frame=tokens_per_frame,
        tokens_per_image_block=num_frame_per_block * tokens_per_frame,
        num_video_frames=token_grid.num_frames,
        has_clean_video_prefix=include_clean_video_prefix,
    )


def build_register_attention_mask(
    layout: RegisterSequenceLayout,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    seq_len = layout.total_sequence_length
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    clean_start, clean_end = layout.clean_video_span
    first_noisy_start, first_noisy_end = layout.first_noisy_frame_span

    if layout.has_clean_video_prefix:
        mask[clean_start:clean_end, clean_start:clean_end] = torch.tril(
            torch.ones(clean_end - clean_start, clean_end - clean_start, device=device, dtype=torch.bool)
        )
        mask[first_noisy_start:first_noisy_end, first_noisy_start:first_noisy_end] = True
    else:
        mask[first_noisy_start:first_noisy_end, first_noisy_start:first_noisy_end] = True

    for block_index, image_span in enumerate(layout.noisy_video_block_spans):
        row_start, row_end = image_span
        if layout.has_clean_video_prefix:
            clean_context_end = clean_start + layout.tokens_per_frame + block_index * layout.tokens_per_image_block
            mask[row_start:row_end, clean_start:clean_context_end] = True
        else:
            mask[row_start:row_end, first_noisy_start:first_noisy_end] = True
            for previous_span in layout.noisy_video_block_spans[:block_index]:
                mask[row_start:row_end, previous_span[0]:previous_span[1]] = True
        mask[row_start:row_end, row_start:row_end] = True
        if block_index < len(layout.action_block_spans):
            action_span = layout.action_block_spans[block_index]
            mask[row_start:row_end, action_span[0]:action_span[1]] = True
        if block_index < len(layout.state_block_spans):
            state_span = layout.state_block_spans[block_index]
            mask[row_start:row_end, state_span[0]:state_span[1]] = True

    for block_index, action_span in enumerate(layout.action_block_spans):
        row_start, row_end = action_span
        if layout.has_clean_video_prefix:
            clean_context_end = clean_start + layout.tokens_per_frame + block_index * layout.tokens_per_image_block
            mask[row_start:row_end, clean_start:clean_context_end] = True
        else:
            mask[row_start:row_end, first_noisy_start:first_noisy_end] = True
            for previous_span in layout.noisy_video_block_spans[:block_index]:
                mask[row_start:row_end, previous_span[0]:previous_span[1]] = True
        if block_index < len(layout.noisy_video_block_spans):
            noisy_image_span = layout.noisy_video_block_spans[block_index]
            mask[row_start:row_end, noisy_image_span[0]:noisy_image_span[1]] = True
        mask[row_start:row_end, row_start:row_end] = True
        if block_index < len(layout.state_block_spans):
            state_span = layout.state_block_spans[block_index]
            mask[row_start:row_end, state_span[0]:state_span[1]] = True

    for state_span in layout.state_block_spans:
        row_start, row_end = state_span
        mask[row_start:row_end, row_start:row_end] = True

    return mask[None, :, :].expand(batch_size, -1, -1)


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.float()
    if dim <= 0:
        return torch.zeros(*values.shape, 0, device=values.device, dtype=values.dtype)
    half_dim = max(1, dim // 2)
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=values.device, dtype=values.dtype)
    exponent = exponent / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    angles = values[..., None] * freqs
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dim:
        pad = torch.zeros(*embedding.shape[:-1], dim - embedding.shape[-1], device=embedding.device, dtype=embedding.dtype)
        embedding = torch.cat([embedding, pad], dim=-1)
    return embedding[..., :dim]


def _build_sequence_position_context(length: int, hidden_size: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    positions = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
    return _sinusoidal_embedding(positions, hidden_size)


def _build_video_position_context(
    token_grid: TokenGridMetadata,
    hidden_size: int,
    device: torch.device,
    frame_offset: int = 0,
) -> torch.Tensor:
    num_frames = token_grid.num_frames
    tokens_per_frame = token_grid.tokens_per_frame
    frame_ids = torch.arange(frame_offset, frame_offset + num_frames, device=device, dtype=torch.float32)
    frame_ids = frame_ids.repeat_interleave(tokens_per_frame)

    patch_h = token_grid.patches_per_frame_h
    patch_w = token_grid.patches_per_frame_w
    h_ids = torch.arange(patch_h, device=device, dtype=torch.float32).repeat_interleave(patch_w)
    w_ids = torch.arange(patch_w, device=device, dtype=torch.float32).repeat(patch_h)
    h_ids = h_ids.repeat(num_frames)
    w_ids = w_ids.repeat(num_frames)

    frame_dim = hidden_size // 3
    h_dim = hidden_size // 3
    w_dim = hidden_size - frame_dim - h_dim
    return torch.cat(
        [
            _sinusoidal_embedding(frame_ids, frame_dim),
            _sinusoidal_embedding(h_ids, h_dim),
            _sinusoidal_embedding(w_ids, w_dim),
        ],
        dim=-1,
    )


def build_register_position_context(
    layout: RegisterSequenceLayout,
    token_grid: TokenGridMetadata,
    hidden_size: int,
    device: torch.device,
    current_start_frame: int = 0,
) -> torch.Tensor:
    position_chunks: list[torch.Tensor] = []
    if layout.has_clean_video_prefix:
        position_chunks.append(
            _build_video_position_context(
                token_grid=token_grid,
                hidden_size=hidden_size,
                device=device,
                frame_offset=current_start_frame,
            )
        )
    position_chunks.append(
        _build_video_position_context(
            token_grid=token_grid,
            hidden_size=hidden_size,
            device=device,
            frame_offset=current_start_frame,
        )
    )
    action_length = sum(end - start for start, end in layout.action_block_spans)
    state_length = sum(end - start for start, end in layout.state_block_spans)
    action_position = _build_sequence_position_context(action_length, hidden_size, device=device, offset=0)
    state_position = _build_sequence_position_context(state_length, hidden_size, device=device, offset=0)
    position_chunks.extend([action_position, state_position])
    return torch.cat(position_chunks, dim=0)
