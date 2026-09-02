from __future__ import annotations

import pytest
import torch

from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.models.policy_variants.observed_video_history import (
    ObservedVideoHistoryState,
)


def test_observed_video_history_commits_only_executed_real_prefix() -> None:
    initial = torch.zeros(1, 3, 1, 2, 2)
    state = ObservedVideoHistoryState.initialize(
        initial,
        start_frame=0,
        model_frame_chunk_size=4,
        attention_window_size=30,
    )
    pending_state, speculative_span = state.begin_generation(
        frame_count=4,
        attention_window_size=30,
    )
    assert speculative_span == PolicyTemporalSpan(start_frame=1, frame_count=4)

    real_prefix = torch.ones(1, 3, 3, 2, 2)
    committed = pending_state.commit_observations(
        PolicyObservedHistory(
            video_latents=real_prefix,
            observation_frame_count=12,
            inference_window_size=30,
            execution_commit=PolicyExecutionCommit(
                speculative_span=speculative_span,
                executed_frame_count=3,
            ),
        )
    )

    assert committed.observed_span == PolicyTemporalSpan(start_frame=0, frame_count=4)
    assert committed.pending_span is None
    torch.testing.assert_close(
        committed.video_latents,
        torch.cat([initial, real_prefix], dim=2),
    )
    _, next_span = committed.begin_generation(
        frame_count=4,
        attention_window_size=30,
    )
    assert next_span == PolicyTemporalSpan(start_frame=4, frame_count=4)


def test_observed_video_history_rejects_unreconciled_or_incompatible_updates() -> None:
    state = ObservedVideoHistoryState.initialize(
        torch.zeros(1, 3, 1, 2, 2),
        start_frame=0,
        model_frame_chunk_size=4,
        attention_window_size=30,
    )
    pending_state, speculative_span = state.begin_generation(
        frame_count=4,
        attention_window_size=30,
    )

    with pytest.raises(RuntimeError, match="reconciled"):
        pending_state.begin_generation(frame_count=4, attention_window_size=30)
    with pytest.raises(ValueError, match="cannot change attention windows"):
        state.begin_generation(frame_count=4, attention_window_size=64)
    with pytest.raises(ValueError, match="does not match the pending generation"):
        pending_state.commit_observations(
            PolicyObservedHistory(
                video_latents=torch.ones(1, 3, 3, 2, 2),
                observation_frame_count=12,
                execution_commit=PolicyExecutionCommit(
                    speculative_span=PolicyTemporalSpan(
                        start_frame=2,
                        frame_count=4,
                    ),
                    executed_frame_count=3,
                ),
            )
        )
    assert speculative_span == PolicyTemporalSpan(start_frame=1, frame_count=4)

    with pytest.raises(ValueError, match="report the pending generation length"):
        pending_state.commit_observations(
            PolicyObservedHistory(
                video_latents=torch.ones(1, 3, 4, 2, 2),
                observation_frame_count=16,
                rollout_frame_chunk_size=2,
                execution_commit=PolicyExecutionCommit(
                    speculative_span=speculative_span,
                    executed_frame_count=4,
                ),
            )
        )


def test_observed_video_history_advances_full_chunks_from_one_frame_startup() -> None:
    state = ObservedVideoHistoryState.initialize(
        torch.zeros(1, 3, 1, 2, 2),
        start_frame=0,
        model_frame_chunk_size=4,
        attention_window_size=30,
    )

    generated_starts: list[int] = []
    for _ in range(3):
        pending, span = state.begin_generation(
            frame_count=4,
            attention_window_size=30,
        )
        generated_starts.append(int(span.start_frame))
        state = pending.commit_observations(
            PolicyObservedHistory(
                video_latents=torch.zeros(1, 3, 4, 2, 2),
                observation_frame_count=16,
                inference_window_size=30,
                execution_commit=PolicyExecutionCommit(
                    speculative_span=span,
                    executed_frame_count=4,
                ),
            )
        )

    assert generated_starts == [1, 5, 9]
    assert state.observed_span == PolicyTemporalSpan(start_frame=0, frame_count=13)


def test_observed_video_history_bounds_w30_like_vta() -> None:
    state = ObservedVideoHistoryState.initialize(
        torch.zeros(1, 1, 1, 1, 1),
        start_frame=0,
        model_frame_chunk_size=4,
        attention_window_size=30,
    )

    for _ in range(20):
        pending, span = state.begin_generation(
            frame_count=4,
            attention_window_size=30,
        )
        real_frames = torch.arange(
            span.start_frame,
            span.end_frame,
            dtype=torch.float32,
        ).reshape(1, 1, 4, 1, 1)
        state = pending.commit_observations(
            PolicyObservedHistory(
                video_latents=real_frames,
                observation_frame_count=16,
                inference_window_size=30,
                rollout_frame_chunk_size=4,
                execution_commit=PolicyExecutionCommit(
                    speculative_span=span,
                    executed_frame_count=4,
                ),
            )
        )

    # W30 exposes 15 four-frame history chunks. The extra oldest frame is the
    # external clean condition used by causal video prediction.
    assert state.video_latents.shape[2] == 61
    assert state.observed_span == PolicyTemporalSpan(start_frame=20, frame_count=61)
    assert state.model_frame_start == 20
    assert state.chunk_origin_frame == 0
    torch.testing.assert_close(
        state.video_latents.flatten(),
        torch.arange(20, 81, dtype=torch.float32),
    )


def test_observed_video_history_preserves_partial_chunk_phase_when_trimmed() -> None:
    state = ObservedVideoHistoryState.initialize(
        torch.arange(65, dtype=torch.float32).reshape(1, 1, 65, 1, 1),
        start_frame=0,
        model_frame_start=-1,
        model_frame_chunk_size=4,
        attention_window_size=30,
    )
    assert state.observed_span == PolicyTemporalSpan(start_frame=4, frame_count=61)
    assert state.model_frame_start == 3
    assert state.chunk_origin_frame == 0

    pending, span = state.begin_generation(
        frame_count=4,
        attention_window_size=30,
    )
    committed = pending.commit_observations(
        PolicyObservedHistory(
            video_latents=torch.arange(65, 68, dtype=torch.float32).reshape(
                1, 1, 3, 1, 1
            ),
            observation_frame_count=12,
            inference_window_size=30,
            rollout_frame_chunk_size=4,
            execution_commit=PolicyExecutionCommit(
                speculative_span=span,
                executed_frame_count=3,
            ),
        )
    )

    # The three-frame interrupted current chunk remains present until inference
    # removes and reconstructs it. The preceding complete W30 history stays full.
    assert committed.observed_span == PolicyTemporalSpan(start_frame=4, frame_count=64)
    assert committed.model_frame_start == 3
    assert committed.chunk_origin_frame == 0
    _, next_span = committed.begin_generation(
        frame_count=4,
        attention_window_size=30,
    )
    assert next_span == PolicyTemporalSpan(start_frame=68, frame_count=4)
