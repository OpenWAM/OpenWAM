"""Control-loop contracts exercised through both real, reduced policy backends."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
import torch

from open_wam.runtime.policy_planner import PolicyPlanner
from open_wam.configs import VideoActionProgram
from open_wam.configs.enums import RealtimeEmptyPlanPolicy
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.pipelines import VariantRolloutRunner
from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions
from open_wam.runtime.control import ControlCommand, ControlTransition
from open_wam.runtime.control import RolloutTerminationReason
from tests.test_unified_policy_inference import pipeline_for


class LatentEnvironment:
    """Only simulator I/O is synthetic; preparation, inference and commits are real."""

    def __init__(self, runner, *, delay=False):
        self.runner = runner
        self.text = torch.ones(1, 3, 16)
        self.executed = []
        self.commits = []
        self.origins = []
        self.delay = delay
        self.release = Event()
        self.started = Event()
        self.prepares = 0
        self.snapshots = []

    def prepare(self, observations, session):
        self.prepares += 1
        self.snapshots.append(session)
        if self.delay and self.prepares == 2:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("Control thread did not advance during planning.")
        video = torch.tensor(observations[::2], dtype=torch.float32)[
            None, None, :, None, None
        ]
        video = video.expand(1, 48, -1, 4, 4)
        visual = self.runner.pipeline.prepare_visual_outputs_from_latents(
            video, text_context=self.text
        )
        return visual, PolicyInferContext(state=torch.ones(1, 1, 4))

    def observed_history(self, observations, actions, visual, span, session):
        actual = torch.from_numpy(np.stack(actions)).unsqueeze(0)
        self.commits.append((span, actual.clone()))
        assert len(observations) == span.frame_count * 2 + 1
        assert visual.frontend.video_latents.shape[2] == span.frame_count + 1
        return PolicyObservedHistory(
            video_latents=visual.frontend.video_latents[:, :, 1:],
            action_history=actual,
            proprio_history=torch.ones(1, span.frame_count, 4),
            observation_frame_count=span.frame_count * 2,
            execution_commit=PolicyExecutionCommit(
                PolicyTemporalSpan(
                    span.start_frame,
                    session.policy_state.cursor.current_start_frame - span.start_frame,
                ),
                span.frame_count,
            ),
        )

    def controls(self, plan, observation, action_start, source, ready_at, *, step_index):
        self.origins.append(plan.frame_span.start_frame)
        return [
            PlannedControlStep(
                absolute_action_index=action_start + index,
                generation_action_start=action_start,
                source=source,
                ready_monotonic_s=ready_at,
                raw_action=action.numpy(),
            )
            for index, action in enumerate(plan.actions)
        ]

    def materialize(self, step, observation):
        return ControlCommand(step.raw_action, step.raw_action)

    def fallback(self, last_action, observation):
        action = np.full(4, 17.0, dtype=np.float32)
        return ControlCommand(action, action)

    def step(self, action):
        self.executed.append(action.copy())
        if len(self.executed) == 6:
            self.release.set()
        return ControlTransition(observation=float(len(self.executed)))

    def synchronize(self):
        pass


@pytest.mark.parametrize("max_actions,max_plans,prefix,executed,committed", [
    (20, 2, None, 8, 8), (20, 2, 2, 4, 4), (3, 2, 2, 3, 2),
    (20, 0, None, 0, 0), (0, 3, None, 0, 0),
])
def test_blocking_plan_budget_reconciles_without_extra_prediction(
    max_actions, max_plans, prefix, executed, committed,
):
    runner = VariantRolloutRunner(pipeline_for("dual_expert", VideoActionProgram.VIDEO_THEN_ACTION))
    adapter = LatentEnvironment(runner)
    result = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter,
        RolloutOptions(max_actions=max_actions, max_plans=max_plans,
                       execute_prefix_actions=prefix, target_action_hz=None),
    ).run(0., runner.reset(text_context=adapter.text), step=adapter.step)
    assert len(result.actions) == executed
    assert result.lifecycle.observed_control_end == committed
    assert len(adapter.origins) == (2 if executed else 0)
    assert result.termination.reason is (
        RolloutTerminationReason.MAX_ACTIONS if max_actions <= executed else RolloutTerminationReason.MAX_PLANS
    )
    if committed:
        last_span, last_actions = adapter.commits[-1]
        assert last_span.start_frame + last_span.frame_count == 1 + committed // 2
        assert torch.equal(last_actions[0], torch.from_numpy(np.stack(adapter.executed[committed-len(last_actions[0]):committed])))


@pytest.mark.parametrize("stateful,budget", [(True, None), (False, 2)])
def test_serial_planner_contract_rejects_background_scheduling(stateful, budget):
    runner = VariantRolloutRunner(pipeline_for("dual_expert", VideoActionProgram.JOINT))
    adapter = LatentEnvironment(runner)
    planner = PolicyPlanner(runner, adapter)
    planner.supports_async = not stateful
    with pytest.raises(ValueError, match="require blocking"):
        RolloutEngine(planner, adapter, RolloutOptions(
            max_actions=8, max_plans=budget, empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        ))
    assert adapter.prepares == 0


def test_precontrol_termination_does_not_record_an_unexecuted_action():
    runner = VariantRolloutRunner(pipeline_for("dual_expert", VideoActionProgram.JOINT))
    adapter = LatentEnvironment(runner)
    result = RolloutEngine(PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=8)).run(
        0., runner.reset(), step=adapter.step,
        termination_check=lambda: RolloutTerminationReason.ENV_TERMINAL,
    )
    assert result.startup is not None
    assert not result.actions and not adapter.executed
    assert result.termination.reason is RolloutTerminationReason.ENV_TERMINAL


@pytest.mark.parametrize("done,success", [(True, False), (True, True), (False, True)])
def test_terminal_failure_and_task_success_are_distinct(done, success):
    class TerminalEnvironment(LatentEnvironment):
        def step(self, action):
            transition = super().step(action)
            return replace(
                transition,
                done=done,
                success=success,
                reward=-2.0,
                info={"limit": done},
            )

    runner = VariantRolloutRunner(
        pipeline_for("dual_expert", VideoActionProgram.VIDEO_THEN_ACTION)
    )
    adapter = TerminalEnvironment(runner)
    result = RolloutEngine(PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=4)).run(
        0.0, runner.reset(), step=adapter.step
    )
    assert len(result.actions) == 1
    assert result.success is success
    assert result.actions[0].transition.reward == -2.0
    assert result.termination.reason is (
        RolloutTerminationReason.SUCCESS if success else RolloutTerminationReason.ENV_TERMINAL
    )
    assert result.termination.control_count == 1
    assert result.termination.transition.info == {"limit": done}


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("extensions,prefix", [(0, None), (2, None), (0, 2)])
def test_startup_extensions_and_observed_replan(architecture, extensions, prefix):
    pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    runner = VariantRolloutRunner(pipeline)
    adapter = LatentEnvironment(runner)
    options = RolloutOptions(
        max_actions=16,
        target_action_hz=10000,
        startup_open_loop_chunks=extensions,
        execute_prefix_actions=prefix,
    )
    initial = runner.reset(text_context=adapter.text)
    result = RolloutEngine(PolicyPlanner(runner, adapter), adapter, options).run(
        0.0, initial, step=adapter.step
    )
    assert initial.policy_state is None
    assert len(result.actions) == 16
    assert not any(row.source.startswith("fallback") for row in result.actions)
    assert adapter.origins[: extensions + 1] == list(range(1, 2 * extensions + 2, 2))
    assert adapter.commits
    for span, actions in adapter.commits:
        offset = (span.start_frame - 1) * 2
        torch.testing.assert_close(
            actions[0],
            torch.from_numpy(
                np.stack(adapter.executed[offset : offset + span.frame_count * 2])
            ),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_delayed_planner_commits_actual_fallback_controls(architecture):
    pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    runner = VariantRolloutRunner(pipeline)
    adapter = LatentEnvironment(runner, delay=True)
    options = RolloutOptions(
        max_actions=20,
        target_action_hz=100,
        empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
    )
    result = RolloutEngine(PolicyPlanner(runner, adapter), adapter, options).run(
        0.0, runner.reset(text_context=adapter.text), step=adapter.step
    )
    assert adapter.started.is_set()
    assert any(row.source.startswith("fallback") for row in result.actions)
    assert any(torch.any(actions == 17) for _, actions in adapter.commits)
    for span, actions in adapter.commits:
        offset = (span.start_frame - 1) * 2
        torch.testing.assert_close(
            actions[0],
            torch.from_numpy(
                np.stack(adapter.executed[offset : offset + span.frame_count * 2])
            ),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("delay_controls", range(4))
def test_delayed_candidates_are_atomic_at_every_control_offset(
    architecture, delay_controls, monkeypatch
):
    """Control the scheduler clock, but run actual prepare/reconcile/denoising."""
    from concurrent.futures import Future
    import open_wam.runtime.planner_executor as runtime

    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )
    adapter = LatentEnvironment(runner)

    class ScheduledFuture(Future):
        def __init__(self, fn, payload):
            super().__init__()
            self.fn, self.payload = fn, payload
            self.ready = len(adapter.executed) + delay_controls

        def done(self):
            if not super().done() and len(adapter.executed) >= self.ready:
                self.result()
            return super().done()

        def result(self, timeout=None):
            if not super().done():
                self.set_result(self.fn(self.payload))
            return super().result(timeout)

    class ControlledScheduler:
        def __init__(self, **kwargs):
            pass

        def shutdown(self, **kwargs):
            pass

        def submit(self, fn, payload):
            return ScheduledFuture(fn, payload)

    monkeypatch.setattr(runtime, "ThreadPoolExecutor", ControlledScheduler)
    result = RolloutEngine(
        PolicyPlanner(runner, adapter),
        adapter,
        RolloutOptions(
            max_actions=16,
            target_action_hz=None,
            empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        ),
    ).run(0.0, runner.reset(text_context=adapter.text), step=adapter.step)
    candidate = result.replans[0]
    assert candidate.acceptance_action_index == 4 + delay_controls
    assert candidate.accepted is (delay_controls == 0)
    assert candidate.stale_planned_actions == delay_controls
    assert candidate.rejected_future_actions == (
        4 - delay_controls if delay_controls else 0
    )
    for trace in result.replans:
        if trace.accepted:
            assert trace.stale_planned_actions == 0


def test_execution_prefix_requires_whole_model_frames():
    runner = VariantRolloutRunner(
        pipeline_for("dual_expert", VideoActionProgram.VIDEO_THEN_ACTION)
    )
    adapter = LatentEnvironment(runner)
    with pytest.raises(ValueError, match="model-frame boundary"):
        RolloutEngine(
            PolicyPlanner(runner, adapter),
            adapter,
            RolloutOptions(max_actions=8, execute_prefix_actions=3),
        )


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_partial_plan_cannot_seed_an_open_loop_extension(architecture):
    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )
    adapter = LatentEnvironment(runner)
    with pytest.raises(ValueError, match="complete executable prediction"):
        RolloutEngine(
            PolicyPlanner(runner, adapter),
            adapter,
            RolloutOptions(
                max_actions=8,
                execute_prefix_actions=2,
                startup_open_loop_chunks=1,
            ),
        )


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_unused_planner_candidate_is_not_published_after_episode_end(architecture):
    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )

    class EndingEnvironment(LatentEnvironment):
        def fallback(self, last_action, observation):
            last_action[:] = 99
            return super().fallback(last_action, observation)

        def step(self, action):
            transition = super().step(action)
            return replace(
                transition,
                done=len(self.executed) == 6,
                success=len(self.executed) == 6,
            )

    adapter = EndingEnvironment(runner, delay=True)
    result = RolloutEngine(
        PolicyPlanner(runner, adapter),
        adapter,
        RolloutOptions(
            max_actions=20,
            target_action_hz=100,
            empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        ),
    ).run(0.0, runner.reset(text_context=adapter.text), step=adapter.step)
    assert result.success
    assert len(result.actions) == 6
    assert result.session is adapter.snapshots[1]
    assert result.session.policy_state.observed_frame_end == 1
    assert result.session.policy_state.cursor.current_start_frame == 3
    assert result.replans == []
    for index, command in result.lifecycle.commands.items():
        np.testing.assert_array_equal(command.action, adapter.executed[index])


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("terminal", [False, True])
def test_unused_planner_failure_cannot_replace_terminal_or_horizon_result(
    architecture, terminal
):
    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )

    class FailingPlannerEnvironment(LatentEnvironment):
        def prepare(self, *args):
            prepared = super().prepare(*args)
            if self.prepares == 2:
                raise RuntimeError("unused speculative failure")
            return prepared

        def step(self, action):
            transition = super().step(action)
            if len(self.executed) == 5:
                assert self.started.wait(5)
                self.release.set()
                return replace(transition, done=terminal, success=terminal, info={"last": 5})
            return transition

    adapter = FailingPlannerEnvironment(runner, delay=True)
    result = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter, RolloutOptions(
            max_actions=5, target_action_hz=None,
            empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        )
    ).run(0.0, runner.reset(text_context=adapter.text), step=adapter.step)
    assert result.success is terminal
    assert result.termination.reason is (
        RolloutTerminationReason.SUCCESS if terminal else RolloutTerminationReason.MAX_ACTIONS
    )
    assert result.termination.transition.info == {"last": 5}
    assert result.planner_teardown.error == "RuntimeError: unused speculative failure"
    assert result.replans == []
    assert result.session is adapter.snapshots[1]
    assert result.lifecycle.observed_control_end == 0
    assert result.lifecycle.next_control_index == 5
    assert set(result.lifecycle.commands) == set(range(5))
    assert set(result.lifecycle.observations) == set(range(6))


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_closing_external_stream_drains_without_publishing_or_raising(architecture):
    from threading import Timer

    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )

    class FailingEnvironment(LatentEnvironment):
        def prepare(self, *args):
            prepared = super().prepare(*args)
            if self.prepares == 2:
                raise RuntimeError("discarded on reset")
            return prepared

    adapter = FailingEnvironment(runner, delay=True)
    stream = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter, RolloutOptions(
            max_actions=20, target_action_hz=None,
            empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        )
    ).control_stream(0.0, runner.reset(text_context=adapter.text))
    action = next(stream)
    for _ in range(4):
        action = stream.send(adapter.step(action))
    assert adapter.started.wait(5)
    release = Timer(0.05, adapter.release.set)
    release.start()
    try:
        result = stream.close()
        assert adapter.release.is_set()
        assert stream.close() is result
        assert result.termination.reason is RolloutTerminationReason.CANCELLED
        assert result.termination.control_count == 4
        assert result.planner_teardown.error == "RuntimeError: discarded on reset"
        assert result.planner_teardown.drain_time_s > 0
        assert result.replans == []
        assert result.session is adapter.snapshots[1]
        assert result.lifecycle.observed_control_end == 0
    finally:
        adapter.release.set()
        release.join()
    # A new call on the SAME pipeline is safe after the drain.
    adapter.delay = False
    result = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=1, target_action_hz=None)
    ).run(4.0, runner.reset(text_context=adapter.text), step=adapter.step)
    assert len(result.actions) == 1


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("outcome", ["unstarted", "startup_error", "live_error", "terminal"])
def test_external_stream_close_retains_the_actual_outcome(architecture, outcome):
    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )

    class Environment(LatentEnvironment):
        def prepare(self, *args):
            if outcome == "startup_error":
                raise RuntimeError("startup failed")
            if outcome == "live_error" and self.prepares == 1:
                raise RuntimeError("live planner failed")
            return super().prepare(*args)

    adapter = Environment(runner)
    session = runner.reset(text_context=adapter.text)
    stream = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=8, target_action_hz=None)
    ).control_stream(0.0, session)
    assert iter(stream) is stream
    assert adapter.prepares == 0
    completed = None
    if outcome == "startup_error":
        with pytest.raises(RuntimeError, match="startup failed"):
            next(stream)
    elif outcome == "live_error":
        action = next(stream)
        with pytest.raises(RuntimeError, match="live planner failed"):
            for _ in range(8):
                action = stream.send(adapter.step(action))
    elif outcome == "terminal":
        action = next(stream)
        transition = replace(adapter.step(action), done=True, success=True)
        with pytest.raises(StopIteration) as stopped:
            stream.send(transition)
        completed = stopped.value.value
    result = stream.close()
    assert stream.close() is result
    if outcome == "terminal":
        assert result is completed
        assert result.termination.reason is RolloutTerminationReason.SUCCESS
        assert result.termination.transition is transition
        assert result.planner_teardown.error is None
    elif outcome == "live_error":
        assert result.termination.reason is RolloutTerminationReason.ERROR
        assert result.termination.error == "RuntimeError: live planner failed"
        assert result.actions
        assert result.planner_teardown.error is None
    else:
        assert result.session is session
        assert result.planner_teardown is None
        assert result.actions == []
        if outcome == "startup_error":
            assert result.termination.reason is RolloutTerminationReason.ERROR
            assert result.termination.error == "RuntimeError: startup failed"
        else:
            assert result.termination.reason is RolloutTerminationReason.CANCELLED


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_adapter_cannot_relabel_prediction_controls(architecture):
    runner = VariantRolloutRunner(
        pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    )

    class RelabeledEnvironment(LatentEnvironment):
        def controls(self, *args, **kwargs):
            return [
                replace(step, absolute_action_index=step.absolute_action_index + 1)
                for step in super().controls(*args, **kwargs)
            ]

    adapter = RelabeledEnvironment(runner)
    session = runner.reset(text_context=adapter.text)
    with pytest.raises(ValueError, match="contiguous prediction control indices"):
        RolloutEngine(PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=8)).run(
            0.0, session, step=adapter.step
        )
    assert adapter.executed == []
    assert session.policy_state is None


def test_engine_trace_renders_through_the_libero_artifact_boundary():
    from open_wam.evals.libero_rollout_artifact_rendering import (
        build_libero_realtime_video_frames,
        build_libero_fallback_timeline_video_frames,
    )
    from open_wam.integrations import LIBERO_ROLLOUT_VIEW_KEYS

    runner = VariantRolloutRunner(
        pipeline_for("dual_expert", VideoActionProgram.VIDEO_THEN_ACTION)
    )
    adapter = LatentEnvironment(runner)
    result = RolloutEngine(
        PolicyPlanner(runner, adapter), adapter, RolloutOptions(max_actions=4, target_action_hz=10000)
    ).run(0.0, runner.reset(text_context=adapter.text), step=adapter.step)
    observations = {
        key: np.full((32, 32, 3), 127, dtype=np.uint8)
        for key in LIBERO_ROLLOUT_VIEW_KEYS
    }
    records = [dict(record.to_record(), obs=observations) for record in result.actions]
    for render in (
        build_libero_realtime_video_frames,
        build_libero_fallback_timeline_video_frames,
    ):
        frames = render(
            action_video_records=records, target_action_hz=10.0, action_per_frame=2
        )
        assert len(frames) == 4
        assert all(frame.ndim == 3 and frame.any() for frame in frames)
