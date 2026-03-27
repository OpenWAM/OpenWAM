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

    clean_start, clean_end = layout.clean_video_span
    first_noisy_start, first_noisy_end = layout.first_noisy_frame_span

    if layout.has_clean_video_prefix:
        # DreamZero teacher forcing keeps a full clean-video prefix. The clean
        # branch stays causal over clean frames only and never attends into the
        # noisy half. The first noisy frame remains self-only.
        mask[clean_start:clean_end, clean_start:clean_end] = torch.tril(
            torch.ones(clean_end - clean_start, clean_end - clean_start, device=device, dtype=torch.bool)
        )
        mask[first_noisy_start:first_noisy_end, first_noisy_start:first_noisy_end] = True
    else:
        # In inference there is no clean prefix, so the first noisy frame acts
        # as the observed conditioning frame and only self-attends.
        mask[first_noisy_start:first_noisy_end, first_noisy_start:first_noisy_end] = True

    # The layout stores future image blocks after the first noisy frame.
    for block_index, image_span in enumerate(layout.noisy_video_block_spans):
        row_start, row_end = image_span
        if layout.has_clean_video_prefix:
            clean_context_end = clean_start + layout.tokens_per_frame + block_index * layout.tokens_per_image_block
            mask[row_start:row_end, clean_start:clean_context_end] = True
        else:
            # DreamZero inference approximates cache behavior by exposing the
            # first observed frame plus past noisy image blocks.
            mask[row_start:row_end, first_noisy_start:first_noisy_end] = True
            for previous_span in layout.noisy_video_block_spans[:block_index]:
                mask[row_start:row_end, previous_span[0]:previous_span[1]] = True
        mask[row_start:row_end, row_start:row_end] = True
        action_span = layout.action_block_spans[block_index]
        state_span = layout.state_block_spans[block_index]
        mask[row_start:row_end, action_span[0]:action_span[1]] = True
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
        state_span = layout.state_block_spans[block_index]
        mask[row_start:row_end, state_span[0]:state_span[1]] = True

    # State tokens are conditioning registers; DreamZero keeps them local to
    # their own block rather than letting them aggregate the entire history.
    for state_span in layout.state_block_spans:
        row_start, row_end = state_span
        mask[row_start:row_end, row_start:row_end] = True

    return mask[None, :, :].expand(batch_size, -1, -1)
