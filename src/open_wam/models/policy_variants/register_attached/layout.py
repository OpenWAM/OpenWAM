from __future__ import annotations

from dataclasses import dataclass

from open_wam.models.video_backbone.contracts import TokenGridMetadata


@dataclass(frozen=True)
class RegisterSequenceLayout:
    """Video-plus-register layout used by the register-attached variant.

    The packed sequence is laid out as:

    - optional clean video prefix tokens used only for teacher forcing
    - noisy video tokens, where the first noisy frame stays special and the
      remaining frames are grouped into DreamZero-style image blocks
    - action-register blocks
    - state-register blocks

    All spans are over the flattened packed axis `S_total`.
    """

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
) -> RegisterSequenceLayout:
    if token_grid.num_frames < 1:
        raise ValueError("Register-attached variant requires at least one frame.")
    if (token_grid.num_frames - 1) % num_frame_per_block != 0:
        raise ValueError(
            "Expected `(num_frames - 1)` to be divisible by `num_frame_per_block`, "
            f"got num_frames={token_grid.num_frames}, num_frame_per_block={num_frame_per_block}"
        )
    if action_horizon % num_action_per_block != 0:
        raise ValueError(
            "Expected `action_horizon` to be divisible by `num_action_per_block`, "
            f"got action_horizon={action_horizon}, num_action_per_block={num_action_per_block}"
        )
    if state_horizon % num_state_per_block != 0:
        raise ValueError(
            "Expected `state_horizon` to be divisible by `num_state_per_block`, "
            f"got state_horizon={state_horizon}, num_state_per_block={num_state_per_block}"
        )
    num_image_blocks = (token_grid.num_frames - 1) // num_frame_per_block
    num_action_blocks = action_horizon // num_action_per_block
    num_state_blocks = state_horizon // num_state_per_block
    if num_image_blocks != num_action_blocks or num_image_blocks != num_state_blocks:
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
        # Each image block represents `num_frame_per_block` future frames, with
        # `tokens_per_frame` flattened patch tokens per frame.
        block_tokens = num_frame_per_block * tokens_per_frame
        noisy_video_block_spans.append((cursor, cursor + block_tokens))
        cursor += block_tokens
    noisy_video_span = (first_noisy_frame_span[0], cursor)
    noisy_video_sequence_length = noisy_video_span[1] - noisy_video_span[0]

    action_block_spans: list[tuple[int, int]] = []
    register_cursor = cursor
    for _ in range(num_action_blocks):
        # Action registers stay in 1D sequence space, so one block contributes
        # `num_action_per_block` learned register slots.
        action_block_spans.append((register_cursor, register_cursor + num_action_per_block))
        register_cursor += num_action_per_block

    state_block_spans: list[tuple[int, int]] = []
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
