from __future__ import annotations

import torch


def pack_temporal_sequence(
    *,
    sequence: torch.Tensor,
    target_dim: int,
    target_length: int,
    left_pad: bool = False,
    sequence_name: str = "sequence",
    truncate_to_target_length: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad one ``[T, D]`` sequence and build its element-validity mask.

    Dataset adapters own source decoding and decide whether overlong sequences
    may be truncated. Callers should enable truncation unless they already
    guarantee ``T <= target_length``. This shared transform owns the
    model-facing float32 tensor layout. Left padding aligns short state
    histories to their newest timestep; right padding is the normal
    future-target layout.
    """

    if sequence.ndim != 2:
        raise ValueError(
            f"Expected {sequence_name} tensor with shape [T, D], got {tuple(sequence.shape)}."
        )

    raw_dim = sequence.shape[-1]
    if raw_dim > target_dim:
        raise ValueError(
            f"Raw {sequence_name} dim {raw_dim} exceeds configured target dim {target_dim}."
        )

    output = torch.zeros(target_length, target_dim, dtype=torch.float32)
    mask = torch.zeros(target_length, target_dim, dtype=torch.float32)
    packed = sequence[:target_length] if truncate_to_target_length else sequence
    start_index = target_length - len(packed) if left_pad else 0
    for index, values in enumerate(packed):
        output[start_index + index, :raw_dim] = values
        mask[start_index + index, :raw_dim] = 1.0
    return output, mask


__all__ = ["pack_temporal_sequence"]
