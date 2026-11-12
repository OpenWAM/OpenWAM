"""A completed plan is eligible until its first control has actually executed."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import open_wam.runtime.planner_executor as executor_module
import open_wam.runtime.rollout_engine as engine_module
import pytest
from open_wam.configs.enums import RealtimeEmptyPlanPolicy, RealtimePlannerMode
from open_wam.runtime.control import ControlCommand, ControlTransition
from open_wam.runtime.planning_contracts import PlannerResult
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions
from open_wam.runtime.rollout_receipts import PlannerReceipt
from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Planner:
    temporal = ResolvedRolloutTemporalContract(1, 1, 1, 4, 8)
    supports_async = True
    supports_speculative_continuation = False

    def __init__(self, clock, startup_length=4):
        self.clock = clock
        self.startup_length = startup_length
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        start = request.end
        count = 4 if request.reconcile else self.startup_length
        value = 100.0 if request.reconcile else 10.0
        steps = tuple(
            PlannedControlStep(
                absolute_action_index=start + offset,
                generation_action_start=start,
                source=request.source,
                raw_action=np.array([value + offset]),
            )
            for offset in range(count)
        )
        return PlannerResult(
            session=request.source,
            steps=steps,
            observation_end=request.end,
            reconciled=request.reconcile,
            base_revision=request.base_revision,
            receipt=PlannerReceipt(
                source=request.source,
                use_observation_update=request.reconcile,
                observation_action_start=request.start,
                observation_action_end=request.end,
                model_generation_frame_start=start + 1,
                generation_action_start=start,
                planned_action_ids=tuple(step.absolute_action_index for step in steps),
                policy_action_shape=(count, 1),
                prepare_s=0.0,
                infer_s=0.0,
                total_latency_s=0.0,
                ready_monotonic_s=self.clock.now,
            ),
        )


class Adapter:
    def __init__(self):
        self.commands = []
        self.materializations = 0

    def materialize(self, step, observation):
        self.materializations += 1
        return ControlCommand(step.raw_action, step.raw_action)

    def fallback(self, last_action, observation):
        self.materializations += 1
        return ControlCommand(np.array([-9.0]), np.array([-9.0]))

    def step(self, action):
        self.commands.append(float(action[0]))
        return ControlTransition(observation=len(self.commands))


def controlled_worker(monkeypatch, clock, first_delay):
    """Keep readiness deterministic; the native PlannerExecutor owns the Future."""

    class ScheduledFuture(Future):
        def __init__(self, fn, request, delay):
            super().__init__()
            self.fn, self.request = fn, request
            self.ready_at = clock.now + delay

        def done(self):
            if not super().done() and clock.now >= self.ready_at:
                if self.set_running_or_notify_cancel():
                    try:
                        self.set_result(self.fn(self.request))
                    except BaseException as error:
                        self.set_exception(error)
            return super().done()

        def result(self, timeout=None):
            if not self.done():
                clock.now = self.ready_at
                self.done()
            return super().result(timeout)

    class Worker:
        def __init__(self, **kwargs):
            self.submissions = 0

        def submit(self, fn, request):
            delay = first_delay if self.submissions == 0 else 100.0
            self.submissions += 1
            return ScheduledFuture(fn, request, delay)

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(executor_module, "ThreadPoolExecutor", Worker)


def run_case(monkeypatch, *, delay, startup_length, hz=1.0, blocking=False):
    clock = Clock()
    monkeypatch.setattr(engine_module, "time", clock)
    controlled_worker(monkeypatch, clock, delay)
    planner, adapter = Planner(clock, startup_length), Adapter()
    options = RolloutOptions(
        max_actions=3,
        target_action_hz=hz,
        planner_mode=(
            RealtimePlannerMode.HISTORY_ONLY
            if blocking
            else RealtimePlannerMode.ASYNC_HISTORY_FIRST
        ),
        empty_plan_policy=(
            RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
            if blocking
            else RealtimeEmptyPlanPolicy.FALLBACK
        ),
    )
    result = RolloutEngine(planner, adapter, options).run(0, "initial", step=adapter.step)
    assert adapter.materializations == 3
    assert [row.command.action.item() for row in result.actions] == adapter.commands
    assert [row.command.source_action.item() for row in result.actions] == adapter.commands
    return result, adapter, planner


@pytest.mark.parametrize("startup_length", [1, 4], ids=["empty_queue", "queued_controls"])
@pytest.mark.parametrize("delay", [0.0, 0.25, 0.5, 1.0, 1.25, 2.0])
def test_plan_readiness_is_checked_at_the_dispatch_boundary(monkeypatch, startup_length, delay):
    result, adapter, planner = run_case(monkeypatch, delay=delay, startup_length=startup_length)
    expected_second = 100.0 if delay <= 1.0 else (-9.0 if startup_length == 1 else 11.0)
    assert adapter.commands[1] == expected_second
    assert [row.actual_start_s for row in result.actions] == [0.0, 1.0, 2.0]
    assert [row.wait_for_plan_s for row in result.actions] == [0.0, 0.0, 0.0]
    if delay <= 1.0:
        first = result.replans[0]
        assert first.accepted and first.stale_planned_actions == 0
        assert first.acceptance_action_index == 1
        assert result.actions[1].source == "history_replan"
        assert planner.requests[1].observations == (0, 1)
        assert planner.requests[1].actions[0].item() == 10.0
    else:
        assert not any(row.accepted for row in result.replans)
        assert result.lifecycle.observed_control_end == 0
        assert adapter.commands == (
            [10.0, -9.0, -9.0] if startup_length == 1 else [10.0, 11.0, 12.0]
        )


@pytest.mark.parametrize("delay", [0.0, 0.5])
def test_unpaced_execution_keeps_its_existing_readiness_rule(monkeypatch, delay):
    result, adapter, _ = run_case(monkeypatch, delay=delay, startup_length=4, hz=None)
    assert adapter.commands[1] == (100.0 if delay == 0.0 else 11.0)
    assert [row.actual_start_s for row in result.actions] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("startup_length", [1, 4])
def test_blocking_execution_keeps_actions_and_schedule(monkeypatch, startup_length):
    result, adapter, _ = run_case(
        monkeypatch, delay=0.5, startup_length=startup_length, blocking=True
    )
    assert adapter.commands == ([10.0, 100.0, 101.0] if startup_length == 1 else [10.0, 11.0, 12.0])
    assert [row.actual_start_s for row in result.actions] == [0.0, 1.0, 2.0]


def test_native_worker_finishing_during_pacing_can_supply_the_next_control(monkeypatch):
    """Real worker and Future; events control completion instead of wall-clock luck."""
    started, release = Event(), Event()
    futures = []
    clock = Clock()

    class Worker(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            futures.append(future)
            return future

    class DelayedPlanner(Planner):
        def plan(self, request):
            if request.reconcile:
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release the planner.")
            return super().plan(request)

    def sleep(seconds):
        clock.sleep(seconds)
        if seconds > 0 and futures:
            assert started.wait(5)
            release.set()
            futures[0].result(timeout=5)

    monkeypatch.setattr(executor_module, "ThreadPoolExecutor", Worker)
    monkeypatch.setattr(
        engine_module, "time", SimpleNamespace(perf_counter=clock.perf_counter, sleep=sleep)
    )
    planner, adapter = DelayedPlanner(clock), Adapter()
    try:
        result = RolloutEngine(
            planner,
            adapter,
            RolloutOptions(
                max_actions=2,
                target_action_hz=1.0,
                planner_mode=RealtimePlannerMode.ASYNC_HISTORY_FIRST,
                empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
            ),
        ).run(0, "initial", step=adapter.step)
    finally:
        release.set()
    assert adapter.commands == [10.0, 100.0]
    assert adapter.materializations == 2
    assert result.replans[0].accepted
    assert result.replans[0].acceptance_action_index == 1
    assert result.lifecycle.observed_control_end == 1
    assert result.lifecycle.commands[1].source_action.item() == 100.0
    assert result.planner_teardown.error is None
