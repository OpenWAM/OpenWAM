"""The cached action mask must honour history_stream_visibility.

Training builds its mask through chunked_attention_visibility, where the
setting gates the *history* branch: under VIDEO_ONLY an action query cannot
attend past-chunk actions. The cached inference mask used by the staged
video_then_action and decoupled rollouts ignored the setting entirely, so a
checkpoint trained with `history_stream_visibility: video_only` was rolled out
with its whole past_action cache visible -- tokens it never saw in training.
The packed path already honoured it, so the two inference paths also disagreed
with each other on the same config field.

The assertions are deliberately two-sided: past_action must close under
VIDEO_ONLY, and video plus the fresh action tokens must stay open under it.
A filter applied to same-chunk pairs instead of history pairs would satisfy
the first and fail the second.
"""

from __future__ import annotations

import pytest
import torch

from open_wam.configs import CurrentBlockCoupling, HistoryStreamVisibility
from open_wam.models.policy_variants.dual_expert.attention import (
    build_dual_expert_inference_action_attention_mask,
)

VIDEO_SEQ_LEN = 160
PAST_ACTION_SEQ_LEN = 16
CURRENT_ACTION_SEQ_LEN = 16


def _fresh_action_visibility(
    history_stream_visibility: HistoryStreamVisibility | str,
    *,
    current_block_coupling: CurrentBlockCoupling,
) -> dict[str, torch.Tensor]:
    """Return every fresh-action query row, partitioned by key segment."""

    mask = build_dual_expert_inference_action_attention_mask(
        video_seq_len=VIDEO_SEQ_LEN,
        past_action_seq_len=PAST_ACTION_SEQ_LEN,
        current_action_seq_len=CURRENT_ACTION_SEQ_LEN,
        video_tokens_per_frame=32,
        action_tokens_per_frame=4,
        chunk_size_frames=4,
        window_size_frames=30,
        device=torch.device("cpu"),
        video_frame_start=0,
        past_action_frame_start=1,
        # a later chunk, so the past-action cache holds genuinely earlier chunks
        current_action_frame_start=5,
        chunk_origin_frame=5,
        current_block_coupling=current_block_coupling,
        history_stream_visibility=history_stream_visibility,
    )
    fresh_start = VIDEO_SEQ_LEN + PAST_ACTION_SEQ_LEN
    fresh_queries = mask[fresh_start:]
    return {
        "video": fresh_queries[:, :VIDEO_SEQ_LEN],
        "past_action": fresh_queries[:, VIDEO_SEQ_LEN:fresh_start],
        "current_action": fresh_queries[:, fresh_start:],
    }


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    ],
)
def test_video_only_closes_past_action_and_leaves_video_and_self_open(
    current_block_coupling: CurrentBlockCoupling,
) -> None:
    visibility = _fresh_action_visibility(
        HistoryStreamVisibility.VIDEO_ONLY,
        current_block_coupling=current_block_coupling,
    )
    assert not visibility["past_action"].any()
    # The filter constrains history only; clean video history and the fresh
    # action tokens themselves remain visible to every fresh-action query.
    assert visibility["video"].all()
    assert visibility["current_action"].all()


def test_full_keeps_past_action_visible() -> None:
    visibility = _fresh_action_visibility(
        HistoryStreamVisibility.FULL,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )
    assert all(segment.all() for segment in visibility.values())


def test_video_queries_video_only_does_not_restrict_action_queries() -> None:
    # Compare only the fresh-action rows consumed by split-cache action
    # inference. The training law exempts those queries from this mode.
    restricted = _fresh_action_visibility(
        HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )
    full = _fresh_action_visibility(
        HistoryStreamVisibility.FULL,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )
    for segment_name in restricted:
        torch.testing.assert_close(restricted[segment_name], full[segment_name])


def test_unknown_history_stream_visibility_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected one of.*video_only"):
        _fresh_action_visibility(
            "not_a_mode",
            current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        )
