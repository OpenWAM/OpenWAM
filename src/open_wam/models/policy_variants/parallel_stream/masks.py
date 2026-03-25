from __future__ import annotations

import torch

from .packing import ParallelPackedSequenceLayout


def build_parallel_attention_mask(
    layout: ParallelPackedSequenceLayout,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    frame_ids = layout.frame_ids
    chunk_ids = layout.chunk_ids
    noise_ids = layout.noise_ids
    seq_len = frame_ids.shape[0]
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    for query_index in range(seq_len):
        query_noise = int(noise_ids[query_index])
        query_chunk = int(chunk_ids[query_index])
        for key_index in range(seq_len):
            key_noise = int(noise_ids[key_index])
            key_chunk = int(chunk_ids[key_index])
            allow = False
            if query_noise == 0 and key_noise == 0:
                allow = key_chunk <= query_chunk
            elif query_noise == 1 and key_noise == 0:
                allow = key_chunk < query_chunk
            elif query_noise == 1 and key_noise == 1:
                allow = key_chunk == query_chunk
            if allow:
                mask[query_index, key_index] = True
    return mask[None, :, :].expand(batch_size, -1, -1)
