from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata


@dataclass(frozen=True)
class ParallelPackedSequenceLayout:
    """Packed layout for the parallel-stream variant.

    All fields are flattened over the final packed token axis `S_total`.

    For a typical sequence order
    `[video_noisy, video_condition, action_noisy, action_condition]`:

    - each video span has length `T_video = num_frames * tokens_per_frame`
    - each action span has length `T_action = num_frames * action_per_frame`
    - total packed length is `2 * T_video + 2 * T_action`
    """

    spans: dict[str, tuple[int, int]]
    frame_ids: torch.Tensor
    chunk_ids: torch.Tensor
    noise_ids: torch.Tensor
    modality_ids: torch.Tensor


def action_tokens_to_frame_major(
    action_tokens: torch.Tensor,
    num_frames: int,
    action_per_frame: int,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = action_tokens.shape
    expected = num_frames * action_per_frame
    if seq_len != expected:
        raise ValueError(
            f"Expected action token sequence length {expected}, got {seq_len}."
        )
    return action_tokens.view(batch_size, num_frames, action_per_frame, hidden_size)


def build_parallel_layout(
    token_grid: TokenGridMetadata,
    action_per_frame: int,
    frame_chunk_size: int,
    sequence_order: tuple[str, ...],
    device: torch.device,
) -> ParallelPackedSequenceLayout:
    video_length = token_grid.sequence_length
    action_length = token_grid.num_frames * action_per_frame
    spans: dict[str, tuple[int, int]] = {}
    frame_ids: list[torch.Tensor] = []
    chunk_ids: list[torch.Tensor] = []
    noise_ids: list[torch.Tensor] = []
    modality_ids: list[torch.Tensor] = []
    cursor = 0

    # Video tokens are already flattened frame-major by the frontend:
    # `[frame0 patch0..patchN, frame1 patch0..patchN, ...]`.
    # `video_frame_ids` therefore has shape `[T_video]`.
    video_frame_ids = torch.arange(token_grid.num_frames, device=device, dtype=torch.long).repeat_interleave(
        token_grid.tokens_per_frame
    )
    video_chunk_ids = (video_frame_ids // frame_chunk_size) * 2
    # Action tokens are constructed as one short per-frame sequence of length
    # `action_per_frame`, so `action_frame_ids` has shape `[T_action]`.
    action_frame_ids = torch.arange(token_grid.num_frames, device=device, dtype=torch.long).repeat_interleave(
        action_per_frame
    )
    action_chunk_ids = (action_frame_ids // frame_chunk_size) * 2 + 1

    for name in sequence_order:
        if name.startswith("video"):
            length = video_length
            current_frame_ids = video_frame_ids
            current_chunk_ids = video_chunk_ids
            modality_id = 0
        elif name.startswith("action"):
            length = action_length
            current_frame_ids = action_frame_ids
            current_chunk_ids = action_chunk_ids
            modality_id = 1
        else:
            raise ValueError(f"Unsupported packed stream '{name}'.")
        spans[name] = (cursor, cursor + length)
        frame_ids.append(current_frame_ids)
        chunk_ids.append(current_chunk_ids)
        noise_ids.append(torch.full((length,), int("noisy" in name), device=device, dtype=torch.long))
        modality_ids.append(torch.full((length,), modality_id, device=device, dtype=torch.long))
        cursor += length

    return ParallelPackedSequenceLayout(
        spans=spans,
        frame_ids=torch.cat(frame_ids, dim=0),
        chunk_ids=torch.cat(chunk_ids, dim=0),
        noise_ids=torch.cat(noise_ids, dim=0),
        modality_ids=torch.cat(modality_ids, dim=0),
    )
