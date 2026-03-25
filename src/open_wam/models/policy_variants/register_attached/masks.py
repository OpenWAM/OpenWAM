from __future__ import annotations

import torch

from .layout import RegisterSequenceLayout


def build_register_attention_mask(
    layout: RegisterSequenceLayout,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    seq_len = layout.total_sequence_length
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    first_start, first_end = layout.first_frame_span
    for block_index, image_span in enumerate(layout.video_block_spans):
        row_start, row_end = image_span
        mask[row_start:row_end, first_start:first_end] = True
        for previous_span in layout.video_block_spans[: block_index + 1]:
            mask[row_start:row_end, previous_span[0]:previous_span[1]] = True
        action_span = layout.action_block_spans[block_index]
        state_span = layout.state_block_spans[block_index]
        mask[row_start:row_end, action_span[0]:action_span[1]] = True
        mask[row_start:row_end, state_span[0]:state_span[1]] = True

    for block_index, action_span in enumerate(layout.action_block_spans):
        row_start, row_end = action_span
        mask[row_start:row_end, first_start:first_end] = True
        for previous_span in layout.video_block_spans[: block_index + 1]:
            mask[row_start:row_end, previous_span[0]:previous_span[1]] = True
        mask[row_start:row_end, action_span[0]:action_span[1]] = True
        state_span = layout.state_block_spans[block_index]
        mask[row_start:row_end, state_span[0]:state_span[1]] = True

    return mask[None, :, :].expand(batch_size, -1, -1)
