from __future__ import annotations

import math

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings for arbitrary scalar positions."""

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


def build_sequence_position_context(length: int, hidden_size: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    positions = torch.arange(offset, offset + length, device=device, dtype=torch.float32)
    return sinusoidal_embedding(positions, hidden_size)


def build_video_position_context(
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
            sinusoidal_embedding(frame_ids, frame_dim),
            sinusoidal_embedding(h_ids, h_dim),
            sinusoidal_embedding(w_ids, w_dim),
        ],
        dim=-1,
    )


def build_action_grid_position_context(
    num_frames: int,
    action_per_frame: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    frame_ids = torch.arange(num_frames, device=device, dtype=torch.float32).repeat_interleave(action_per_frame)
    row_ids = torch.arange(action_per_frame, device=device, dtype=torch.float32).repeat(num_frames)
    frame_dim = hidden_size // 2
    row_dim = hidden_size - frame_dim
    return torch.cat(
        [
            sinusoidal_embedding(frame_ids, frame_dim),
            sinusoidal_embedding(row_ids, row_dim),
        ],
        dim=-1,
    )
