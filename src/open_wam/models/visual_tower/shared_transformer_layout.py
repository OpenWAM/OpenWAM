"""Parameter-free tensor layout helpers for shared transformer execution."""

from __future__ import annotations

import torch
from einops import rearrange


def select_chunk_slices(tensor: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
    """Split the chunk axis into cloned per-chunk tensors."""

    chunked = rearrange(tensor, "b l n c -> b n l c").contiguous()
    if int(chunked.shape[1]) != count:
        raise ValueError(
            f"Expected chunk axis length {count}, got {tuple(chunked.shape)}."
        )
    return tuple(chunked[:, index, :, :].clone() for index in range(count))


def select_split_segments(
    tensor: torch.Tensor, lengths: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    offset = 0
    segments: list[torch.Tensor] = []
    for length in lengths:
        segments.append(tensor.narrow(1, offset, length).clone())
        offset += length
    return tuple(segments)


__all__ = ["select_chunk_slices", "select_split_segments"]
