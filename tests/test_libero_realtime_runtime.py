from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.evals import libero_realtime_runtime
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot


def test_frame_planner_handoff_contracts_are_frozen_and_explicit() -> None:
    session = SimpleNamespace(name="session")
    result = libero_realtime_runtime.FramePlannerJobResult(
        job_kind="history_replan",
        planned_frames=[],
        buffer_tail_session=None,
        trace={"accepted_chunk": False},
        submitted_through_frame=3,
        session=session,
        warmup_session=None,
    )
    application = libero_realtime_runtime.FramePlannerResultApplication(
        history_base_session=session,
        current_chunk_session=session,
        buffer_tail_session=None,
        plan_by_action={},
        pending_history=[],
    )

    assert result.session is session
    assert result.submitted_through_frame == 3
    assert application.current_chunk_session is session
    with pytest.raises(FrozenInstanceError):
        result.job_kind = "open_loop_extension"
    with pytest.raises(FrozenInstanceError):
        application.pending_history = []


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


def test_frame_extension_job_returns_typed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=4,
            cache={"frame_start": 1},
        )
    )
    next_session = SimpleNamespace(
        policy_state=SimpleNamespace(step_index=5),
    )
    raw_actions = torch.arange(28, dtype=torch.float32).reshape(1, 4, 7)

    class Runner:
        def infer_chunk(self, *, session, advance_frame_start):
            assert advance_frame_start is True
            return SimpleNamespace(
                session=next_session,
                chunk_action_pred=raw_actions,
                raw_chunk_action_pred=raw_actions,
                debug={"generation_frame_start": 1},
            )

    monkeypatch.setattr(
        libero_realtime_runtime,
        "synchronize_devices",
        lambda *devices: None,
    )
    result = libero_realtime_runtime.run_extension_job(
        runner=Runner(),
        session=session,
        config=SimpleNamespace(
            inference=SimpleNamespace(frame_chunk_size=2),
            policy_variant=SimpleNamespace(action_per_frame=2),
        ),
        runtime_device=torch.device("cpu"),
    )

    assert isinstance(result, libero_realtime_runtime.FramePlannerJobResult)
    assert result.job_kind == "open_loop_extension"
    assert result.buffer_tail_session is next_session
    assert result.session is None
    assert result.warmup_session is None
    assert result.submitted_through_frame is None
    assert [frame.absolute_frame_index for frame in result.planned_frames] == [1, 2]
    assert result.trace["planned_frame_ids"] == [1, 2]


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
