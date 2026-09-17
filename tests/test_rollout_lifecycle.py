"""Clock-free publication laws; model execution is covered by test_rollout_engine."""

import os
import subprocess
import sys
from concurrent.futures import Future
from threading import Event

import pytest

from open_wam.runtime.control import (
    ControlCommand,
    ControlTransition,
    RolloutTerminationReason,
)
from open_wam.runtime.planner_executor import PlannerExecutor
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.runtime.rollout_lifecycle import RolloutLifecycle


def _steps(start, count):
    return tuple(
        PlannedControlStep(
            absolute_action_index=i,
            generation_action_start=start,
            source="test",
        )
        for i in range(start, start + count)
    )


@pytest.mark.parametrize("density", [1, 2, 4])
def test_lifecycle_publication_and_partial_terminal_history(density):
    initial = RolloutLifecycle.start("initial", "obs0")
    state = initial.publish(
        session="predicted",
        steps=_steps(0, 4 * density),
        base_revision=0,
        observation_end=None,
    )
    for i in range(density):
        state = state.executed(
            ControlCommand(i, i + 10), ControlTransition(f"obs{i + 1}")
        )
    state = state.publish(
        session="reconciled+predicted",
        steps=_steps(density, 4 * density),
        base_revision=1,
        observation_end=density,
    )
    assert state.revision == 2
    assert state.observed_control_end == density
    assert state.next_control_index == density
    assert dict(state.observations) == {density: f"obs{density}"}
    assert not state.commands
    transition = ControlTransition("terminal", done=True, info={"timeout": True})
    final = state.executed(ControlCommand("native", "source"), transition)
    assert final.observed_control_end == density
    assert final.next_control_index == density + 1
    assert final.termination.reason is RolloutTerminationReason.ENV_TERMINAL
    assert final.termination.transition is transition
    assert final.commands[density].source_action == "source"
    assert dict(initial.observations) == {0: "obs0"}
    with pytest.raises(TypeError):
        final.plan[0] = _steps(0, 1)[0]
    with pytest.raises(ValueError, match="after rollout termination"):
        final.executed(ControlCommand(0, 0), transition)
    assert final.terminate(RolloutTerminationReason.ERROR, error="unused") is final
    assert (
        final.publish(
            session="discarded",
            steps=_steps(density + 1, 4),
            base_revision=2,
            observation_end=density + 1,
        )
        is final
    )


@pytest.mark.parametrize("offset", range(9))
@pytest.mark.parametrize("revision", [0, 1])
def test_candidate_is_atomic_and_revision_scoped(offset, revision):
    state = RolloutLifecycle.start("initial", 0).publish(
        session="predicted",
        steps=_steps(0, 8),
        base_revision=0,
        observation_end=None,
    )
    for i in range(offset):
        state = state.executed(ControlCommand(i, i), ControlTransition(i + 1))
    next_state = state.publish(
        session="candidate",
        steps=_steps(0, 8),
        base_revision=revision,
        observation_end=0,
    )
    assert (next_state is not state) is (offset == 0 and revision == 1)


def test_lifecycle_cannot_commit_an_unobserved_interval():
    state = RolloutLifecycle.start("initial", 0)
    with pytest.raises(ValueError, match="unobserved"):
        state.publish(
            session="candidate",
            steps=_steps(2, 4),
            base_revision=0,
            observation_end=2,
        )


def test_observation_only_commit_never_accepts_stale_or_unexecuted_history():
    state = RolloutLifecycle.start("initial", 0).publish(
        session="prediction", steps=_steps(0, 2), base_revision=0, observation_end=None,
    )
    state = state.executed(ControlCommand(0, 0), ControlTransition(1))
    with pytest.raises(ValueError, match="fully executed"):
        state.commit_observed(session="observed", observation_end=1, base_revision=1)
    state = state.executed(ControlCommand(1, 1), ControlTransition(2))
    assert state.commit_observed(session="stale", observation_end=2, base_revision=0) is state
    with pytest.raises(ValueError, match="fully executed"):
        state.commit_observed(session="unobserved", observation_end=3, base_revision=1)
    committed = state.commit_observed(session="observed", observation_end=2, base_revision=1)
    assert committed.session == "observed"
    assert committed.observed_control_end == 2 and committed.revision == 2
    assert dict(committed.observations) == {2: 2} and not committed.commands
    assert state.observed_control_end == 0 and state.session == "prediction"
    with pytest.raises(ValueError, match="advance"):
        committed.commit_observed(session="duplicate", observation_end=2, base_revision=2)
    terminal = state.terminate(RolloutTerminationReason.MAX_ACTIONS)
    assert terminal.commit_observed(session="late", observation_end=2, base_revision=1) is terminal
    # Ordinary candidate publication still refuses an empty executable suffix.
    assert state.publish(session="empty", steps=(), base_revision=1, observation_end=2) is state


def test_lifecycle_has_no_model_or_scheduler_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from open_wam.runtime.rollout_lifecycle import RolloutLifecycle; "
            "import sys; assert 'torch' not in sys.modules; "
            "assert 'concurrent.futures.thread' not in sys.modules",
        ],
        env=os.environ.copy(),
        check=True,
    )


def test_engine_depends_on_planning_contracts_not_model_implementations():
    subprocess.run(
        [sys.executable, "-c",
         "from open_wam.runtime.rollout_engine import RolloutEngine; import sys; "
         "assert 'torch' not in sys.modules; "
         "assert 'open_wam.runtime.policy_planner' not in sys.modules"],
        env=os.environ.copy(), check=True,
    )


def test_planner_propagates_live_failures_and_serializes_work():
    started, release = Event(), Event()

    def fail(_):
        started.set()
        assert release.wait(5)
        raise ValueError("live failure")

    planner = PlannerExecutor()
    try:
        planner.submit(fail, None)
        assert started.wait(5)
        with pytest.raises(RuntimeError, match="one planner"):
            planner.submit(fail, None)
        release.set()
        with pytest.raises(ValueError, match="live failure"):
            planner.take()
        assert not planner.pending
    finally:
        release.set()
        receipt = planner.close()
    assert receipt.error is None  # Already raised to the active caller.


def test_planner_cancels_work_that_has_not_started(monkeypatch):
    import open_wam.runtime.planner_executor as module

    future = Future()
    shutdown = []

    class QueuedExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, payload):
            return future

        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(module, "ThreadPoolExecutor", QueuedExecutor)
    planner = PlannerExecutor()
    planner.submit(lambda _: pytest.fail("cancelled job ran"), None)
    receipt = planner.close()
    assert receipt.cancelled and receipt.error is None
    assert future.cancelled()
    assert shutdown == [{"wait": True, "cancel_futures": True}]
