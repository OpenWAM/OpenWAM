from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RolloutCursor:
    """Minimal rollout position metadata shared across variants and towers."""

    current_start_frame: int = 0
    block_index: int = 0
    chunk_size: int = 1


def advance_rollout_cursor(cursor: RolloutCursor) -> RolloutCursor:
    """Advance one rollout cursor by its chunk size."""

    return RolloutCursor(
        current_start_frame=cursor.current_start_frame + cursor.chunk_size,
        block_index=cursor.block_index + 1,
        chunk_size=cursor.chunk_size,
    )
