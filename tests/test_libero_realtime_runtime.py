from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.evals import libero_realtime_runtime
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot


def test_realtime_runtime_rejects_frame_zero_startup_actions() -> None:
    chunk = SimpleNamespace(
        raw_chunk_action_pred=torch.zeros(1, 16, 1),
        debug={"generation_frame_start": 0},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=0)),
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        libero_realtime_runtime._chunk_to_planned_frames(
            first_chunk=chunk,
            frame_chunk_size=4,
            action_per_frame=4,
            source="startup_plan",
            ready_monotonic_s=0.0,
        )


def test_submit_planner_job_returns_visual_rollback_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_future = object()
    expected_snapshot = VisualRuntimeStateSnapshot(frontend_state=("cache",))
    snapshot_calls: list[dict[str, object]] = []

    def snapshot_visual_runtime(**kwargs):
        snapshot_calls.append(kwargs)
        return expected_snapshot

    monkeypatch.setattr(
        libero_realtime_runtime.realtime_speculation,
        "snapshot_visual_runtime",
        snapshot_visual_runtime,
    )
    monkeypatch.setattr(
        libero_realtime_runtime,
        "maybe_submit_planner_job",
        lambda **_: expected_future,
    )
    runner = SimpleNamespace()
    config = SimpleNamespace()
    session = SimpleNamespace()

    future, snapshot = libero_realtime_runtime.submit_planner_job_with_snapshot(
        executor=SimpleNamespace(),
        planner_mode="history_only",
        pending_history=[{"frame": 1}],
        future_buffer_depth=0,
        runner=runner,
        history_base_session=SimpleNamespace(),
        current_chunk_session=session,
        prompt="task",
        config=config,
        frontend_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
        buffer_tail_session=None,
        seed_base=7,
    )

    assert future is expected_future
    assert snapshot is expected_snapshot
    assert snapshot_calls == [
        {
            "runner": runner,
            "config": config,
            "session": session,
        }
    ]
