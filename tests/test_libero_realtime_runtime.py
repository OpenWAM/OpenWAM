from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object]]] = []

        def submit(self, fn, **kwargs):
            self.calls.append((fn, kwargs))
            return expected_future

    monkeypatch.setattr(
        libero_realtime_runtime.realtime_speculation,
        "snapshot_visual_runtime",
        snapshot_visual_runtime,
    )
    runner = SimpleNamespace()
    config = SimpleNamespace()
    session = SimpleNamespace(policy_state=SimpleNamespace(step_index=3))
    history_session = SimpleNamespace(name="history")
    executor = RecordingExecutor()

    future, snapshot = libero_realtime_runtime.submit_planner_job_with_snapshot(
        executor=executor,
        planner_mode="history_only",
        pending_history=[
            {
                "absolute_frame_index": 1,
                "obs": {"image": np.asarray([1], dtype=np.uint8)},
            }
        ],
        future_buffer_depth=0,
        runner=runner,
        history_base_session=history_session,
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
    assert len(executor.calls) == 1
    submitted_fn, submitted_kwargs = executor.calls[0]
    assert submitted_fn is libero_realtime_runtime.run_replan_job
    assert submitted_kwargs["session"] is history_session
    assert submitted_kwargs["job_seed"] == 10
    assert len(submitted_kwargs["history_records"]) == 1
    assert snapshot_calls == [
        {
            "runner": runner,
            "config": config,
            "session": session,
        }
    ]
