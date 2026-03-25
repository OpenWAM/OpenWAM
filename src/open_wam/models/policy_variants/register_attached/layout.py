from __future__ import annotations

from dataclasses import dataclass

from open_wam.models.video_backbone.contracts import TokenGridMetadata


@dataclass(frozen=True)
class RegisterSequenceLayout:
    """Video-plus-register layout used by the register-attached variant.

    The packed sequence is laid out as:

    - first observed frame tokens
    - blockwise future video tokens
    - action-register blocks
    - state-register blocks

    All spans are over the flattened packed axis `S_total`.
    """

    first_frame_span: tuple[int, int]
    video_block_spans: tuple[tuple[int, int], ...]
    action_block_spans: tuple[tuple[int, int], ...]
    state_block_spans: tuple[tuple[int, int], ...]
    video_sequence_length: int
    total_sequence_length: int
    num_image_blocks: int
    num_action_blocks: int
    num_state_blocks: int


def build_register_sequence_layout(
    token_grid: TokenGridMetadata,
    action_horizon: int,
    state_horizon: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
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
    first_frame_span = (0, tokens_per_frame)
    video_block_spans: list[tuple[int, int]] = []
    cursor = tokens_per_frame
    for _ in range(num_image_blocks):
        # Each image block represents `num_frame_per_block` future frames, with
        # `tokens_per_frame` flattened patch tokens per frame.
        block_tokens = num_frame_per_block * tokens_per_frame
        video_block_spans.append((cursor, cursor + block_tokens))
        cursor += block_tokens
    video_sequence_length = cursor

    action_block_spans: list[tuple[int, int]] = []
    register_cursor = video_sequence_length
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
        first_frame_span=first_frame_span,
        video_block_spans=tuple(video_block_spans),
        action_block_spans=tuple(action_block_spans),
        state_block_spans=tuple(state_block_spans),
        video_sequence_length=video_sequence_length,
        total_sequence_length=register_cursor,
        num_image_blocks=num_image_blocks,
        num_action_blocks=num_action_blocks,
        num_state_blocks=num_state_blocks,
    )
