"""Benchmark-neutral control loop over immutable policy sessions.

There is one planner worker. It returns a candidate session; only the control
thread may publish that session. Rejected candidates never require semantic
rollback. Background planning requires request-isolated frontend preparation.
A serial streaming encoder may participate only through a planner that rejects
asynchronous scheduling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar
from collections.abc import Callable, Generator

import numpy as np
from open_wam.configs.enums import (
    RealtimeEmptyPlanPolicy,
    RealtimePlannerMode,
    RealtimePlannerJob,
    coerce_fields,
)
from open_wam.runtime.planning_contracts import (
    RolloutPlanner,
    ControlAdapter,
    PlannerRequest,
    PlannerResult,
)
from open_wam.runtime.rollout_receipts import ExecutedControlReceipt, PlannerReceipt
from open_wam.runtime.control import (
    ControlCommand,
    ControlTransition,
    RolloutTermination,
    RolloutTerminationReason,
)
from open_wam.runtime.rollout_lifecycle import RolloutLifecycle
from open_wam.runtime.planner_executor import PlannerExecutor, PlannerTeardown
from open_wam.runtime.realtime_scheduling import (
    should_submit_sequence_planner,
    select_realtime_planner_job,
)

Observation = TypeVar("Observation")
Session = TypeVar("Session")


@dataclass(frozen=True)
class RolloutOptions:
    max_actions: int
    target_action_hz: float | None = 10.0
    planner_mode: RealtimePlannerMode = RealtimePlannerMode.HISTORY_ONLY
    empty_plan_policy: RealtimeEmptyPlanPolicy = RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
    buffer_threshold: int = 16
    replan_low_watermark_actions: int = 0
    startup_open_loop_chunks: int = 0
    execute_prefix_actions: int | None = None
    max_plans: int | None = None

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "planner_mode": RealtimePlannerMode,
                "empty_plan_policy": RealtimeEmptyPlanPolicy,
            },
        )
        if self.max_actions < 0 or (
            self.target_action_hz is not None and self.target_action_hz <= 0
        ):
            raise ValueError(
                "Control horizon must be nonnegative and frequency positive."
            )
        if self.max_plans is not None and self.max_plans < 0:
            raise ValueError("Plan budgets cannot be negative.")
        if (
            min(
                self.buffer_threshold,
                self.replan_low_watermark_actions,
                self.startup_open_loop_chunks,
            )
            < 0
        ):
            raise ValueError(
                "Buffer thresholds and startup extensions cannot be negative."
            )
        if self.execute_prefix_actions is not None and self.execute_prefix_actions <= 0:
            raise ValueError("Executed plan prefixes must be positive.")


@dataclass
class RolloutResult(Generic[Session, Observation]):
    lifecycle: RolloutLifecycle[Session, Observation]
    actions: list[ExecutedControlReceipt[Observation]] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    replans: list[PlannerReceipt] = field(default_factory=list)
    extensions: list[PlannerReceipt] = field(default_factory=list)
    startup: PlannerReceipt | None = None
    live_wall_time_s: float = 0.0
    planner_teardown: PlannerTeardown | None = None

    @property
    def session(self) -> Session:
        return self.lifecycle.session

    @property
    def termination(self) -> RolloutTermination[Observation] | None:
        return self.lifecycle.termination

    @property
    def success(self) -> bool:
        return self.termination is not None and self.termination.success


class RolloutControlStream(Generic[Session, Observation]):
    """Externally driven controls with a retained, drain-safe completion result.

    Like a generator, next()/send() yield controls and completion raises
    StopIteration(result). close() also returns that result, including teardown.
    """

    def __init__(
        self,
        stream: Generator[
            np.ndarray,
            ControlTransition[Observation] | None,
            RolloutResult[Session, Observation],
        ],
        result: RolloutResult[Session, Observation],
    ) -> None:
        self._stream, self._result = stream, result

    def __iter__(self) -> RolloutControlStream[Session, Observation]:
        return self

    def __next__(self) -> np.ndarray:
        return self.send(None)

    def send(self, transition: ControlTransition[Observation] | None) -> np.ndarray:
        try:
            return self._stream.send(transition)
        except StopIteration:
            raise
        except Exception as exc:
            self._result.lifecycle = self._result.lifecycle.terminate(
                RolloutTerminationReason.ERROR, error=f"{type(exc).__name__}: {exc}"
            )
            raise

    def close(self) -> RolloutResult[Session, Observation]:
        """Discard unpublished work, wait for the worker, and retain its receipt."""
        self._result.lifecycle = self._result.lifecycle.terminate(
            RolloutTerminationReason.CANCELLED
        )
        self._stream.close()
        return self._result


class RolloutEngine(Generic[Session, Observation]):
    """One control lifecycle for any planner implementing the transaction contracts."""

    def __init__(
        self,
        planner: RolloutPlanner[Session, Observation],
        adapter: ControlAdapter[Observation],
        options: RolloutOptions,
    ) -> None:
        self.adapter, self.options = adapter, options
        self.temporal, self.planner = planner.temporal, planner
        self.blocking = (
            options.planner_mode is RealtimePlannerMode.HISTORY_ONLY
            and options.empty_plan_policy is RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
        )
        if not self.blocking and (
            not planner.supports_async or options.max_plans is not None
        ):
            raise ValueError(
                "Stateful planners and plan budgets require blocking execution."
            )
        prefix = options.execute_prefix_actions
        if prefix is not None and self.temporal.control_offset(prefix):
            raise ValueError("Execution prefixes must end on a model-frame boundary.")
        horizon = self.temporal.prediction_controls
        self.can_extend = planner.supports_speculative_continuation and (
            prefix is None or prefix >= horizon
        )
        if options.startup_open_loop_chunks and not self.can_extend:
            raise ValueError(
                "Open-loop extensions require a complete executable prediction and a continuable policy."
            )
        if options.max_plans is not None and options.startup_open_loop_chunks:
            raise ValueError("Plan budgets cannot be combined with startup extensions.")

    def run(
        self,
        initial_observation: Observation,
        session: Session,
        *,
        step: Callable[[np.ndarray], ControlTransition[Observation]],
        termination_check: Callable[[], RolloutTerminationReason | None] | None = None,
    ) -> RolloutResult[Session, Observation]:
        stream = self.control_stream(
            initial_observation, session, termination_check=termination_check
        )
        try:
            try:
                action = next(stream)
            except StopIteration as stopped:
                return stopped.value
            while True:
                transition = step(action)
                try:
                    action = stream.send(transition)
                except StopIteration as stopped:
                    return stopped.value
        finally:
            stream.close()

    def control_stream(
        self,
        initial_observation: Observation,
        session: Session,
        *,
        termination_check: Callable[[], RolloutTerminationReason | None] | None = None,
    ) -> RolloutControlStream[Session, Observation]:
        """Yield controls and consume transitions without owning the environment.

        Blocking drivers call run(); external APIs drive the same lifecycle
        with send(). close() discards unpublished work and returns its receipt.
        No model work is performed until next()/send().
        """
        result = RolloutResult(
            lifecycle=RolloutLifecycle.start(session, initial_observation)
        )
        return RolloutControlStream(
            self._control_stream(initial_observation, result, termination_check), result
        )

    def _control_stream(
        self,
        initial_observation: Observation,
        result: RolloutResult[Session, Observation],
        termination_check: Callable[[], RolloutTerminationReason | None] | None,
    ) -> Generator[
        np.ndarray,
        ControlTransition[Observation] | None,
        RolloutResult[Session, Observation],
    ]:
        options = self.options
        if options.max_actions == 0 or options.max_plans == 0:
            result.lifecycle = result.lifecycle.terminate(
                RolloutTerminationReason.MAX_ACTIONS
                if options.max_actions == 0
                else RolloutTerminationReason.MAX_PLANS
            )
            return result

        def plan(
            payload: PlannerRequest[Session, Observation],
        ) -> PlannerResult[Session]:
            candidate = self.planner.plan(payload)
            if options.execute_prefix_actions is not None:
                candidate = replace(
                    candidate, steps=candidate.steps[: options.execute_prefix_actions]
                )
                candidate = replace(
                    candidate,
                    receipt=replace(
                        candidate.receipt,
                        planned_action_ids=tuple(
                            s.absolute_action_index for s in candidate.steps
                        ),
                    ),
                )
            return candidate

        first = plan(
            PlannerRequest(
                result.session,
                (initial_observation,),
                (),
                0,
                0,
                False,
                "startup_plan",
                0,
            )
        )
        result.lifecycle = result.lifecycle.publish(
            session=first.session,
            steps=first.steps,
            base_revision=0,
            observation_end=None,
        )
        result.startup = first.receipt
        plan_count = 1
        current = initial_observation
        last_action = None
        for _ in range(options.startup_open_loop_chunks):
            extension = plan(
                PlannerRequest(
                    result.session,
                    (current,),
                    (),
                    0,
                    0,
                    False,
                    "open_loop_extension",
                    result.lifecycle.revision,
                )
            )
            result.lifecycle = result.lifecycle.publish(
                session=extension.session,
                steps=extension.steps,
                base_revision=extension.base_revision,
                observation_end=None,
            )
            result.extensions.append(extension.receipt)
            plan_count += 1

        def request(
            *, blocking: bool = False
        ) -> PlannerRequest[Session, Observation] | None:
            state = result.lifecycle
            cursor = state.next_control_index
            end = self.temporal.complete_control_end(cursor)
            committed_end = state.observed_control_end
            has_history = end > committed_end and all(
                i in state.observations for i in range(committed_end, end + 1)
            )
            job = select_realtime_planner_job(
                planner_mode=RealtimePlannerMode.HISTORY_ONLY
                if blocking
                else options.planner_mode,
                history_count=(end - committed_end) // self.temporal.controls_per_frame
                if has_history
                else 0,
                future_buffer_depth=len(state.plan) // self.temporal.controls_per_frame,
                has_buffer_tail_session=self.can_extend,
            )
            if job is RealtimePlannerJob.HISTORY_REPLAN:
                # The tail contains the latest speculative span, including startup extensions.
                return PlannerRequest(
                    state.session,
                    tuple(state.observations[i] for i in range(committed_end, end + 1)),
                    tuple(
                        state.commands[i].source_action
                        for i in range(committed_end, end)
                    ),
                    committed_end,
                    end,
                    True,
                    "history_replan",
                    state.revision,
                )
            if job is RealtimePlannerJob.BUFFER_EXTENSION:
                return PlannerRequest(
                    state.session,
                    (current,),
                    (),
                    end,
                    end,
                    False,
                    "open_loop_extension",
                    state.revision,
                )
            return None

        def accept(candidate: PlannerResult[Session]) -> None:
            nonlocal plan_count
            previous = result.lifecycle
            cursor = previous.next_control_index
            result.lifecycle = previous.publish(
                session=candidate.session,
                steps=candidate.steps,
                base_revision=candidate.base_revision,
                observation_end=candidate.observation_end
                if candidate.reconciled
                else None,
            )
            accepted = result.lifecycle is not previous
            plan_count += int(accepted)
            receipt = replace(
                candidate.receipt,
                accepted=accepted,
                base_revision=candidate.base_revision,
                accepted_revision=result.lifecycle.revision,
                acceptance_action_index=cursor,
                stale_planned_actions=sum(
                    s.absolute_action_index < cursor for s in candidate.steps
                ),
                rejected_future_actions=0
                if accepted
                else sum(s.absolute_action_index >= cursor for s in candidate.steps),
            )
            (result.replans if candidate.reconciled else result.extensions).append(
                receipt
            )

        start_time, pause = time.perf_counter(), 0.0
        worker = PlannerExecutor[PlannerResult[Session]]()
        try:
            for real_index in range(options.max_actions):
                cursor = result.lifecycle.next_control_index
                if worker.ready:
                    accept(worker.take())
                step = result.lifecycle.plan.get(cursor)
                waited = 0.0
                while (
                    step is None
                    and options.empty_plan_policy
                    is RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
                ):
                    began = time.perf_counter()
                    if not worker.pending:
                        payload = request(blocking=True)
                        if payload is None:
                            raise RuntimeError(
                                "No complete observed interval is available for replanning."
                            )
                        candidate = plan(payload)
                    else:
                        candidate = worker.take()
                    waited += time.perf_counter() - began
                    accept(candidate)
                    step = result.lifecycle.plan.get(cursor)
                fallback = step is None
                reason = None if termination_check is None else termination_check()
                if reason is not None:
                    result.lifecycle = result.lifecycle.terminate(reason)
                    break
                control = (
                    self.adapter.fallback(last_action, current)
                    if fallback
                    else self.adapter.materialize(step, current)
                )
                # Own the receipt even if an environment mutates its input buffer.
                control = ControlCommand(
                    np.array(control.action, copy=True),
                    np.array(control.source_action, copy=True),
                )
                action = control.action.copy()
                pause += waited
                scheduled = (
                    start_time + real_index / options.target_action_hz + pause
                    if options.target_action_hz is not None
                    else time.perf_counter()
                )
                time.sleep(max(0.0, scheduled - time.perf_counter()))
                began = time.perf_counter()
                transition = yield action
                current = transition.observation
                ended = time.perf_counter()
                last_action = control.action.copy()
                source = "fallback_control" if fallback else step.source
                receipt = ExecutedControlReceipt(
                    index=real_index,
                    command=control,
                    transition=transition,
                    source=source,
                    temporal=self.temporal,
                    scheduled_start_s=scheduled - start_time,
                    actual_start_s=began - start_time,
                    env_step_s=ended - began,
                    wait_for_plan_s=waited,
                    generation_action_start=None
                    if fallback
                    else step.generation_action_start,
                    planner_step_index=None if fallback else step.planner_step_index,
                )
                result.actions.append(receipt)
                result.observations.append(current)
                result.lifecycle = result.lifecycle.executed(control, transition)
                cursor = result.lifecycle.next_control_index
                if result.termination is not None:
                    break
                if cursor >= options.max_actions:
                    result.lifecycle = result.lifecycle.terminate(
                        RolloutTerminationReason.MAX_ACTIONS, transition=transition
                    )
                    break
                if (
                    not result.lifecycle.plan
                    and options.max_plans is not None
                    and plan_count >= options.max_plans
                ):
                    payload = request(blocking=True)
                    if payload is None:
                        raise RuntimeError(
                            "Plan budgets must end on a complete observed interval."
                        )
                    observed = self.planner.observe(payload)
                    result.lifecycle = result.lifecycle.commit_observed(
                        session=observed,
                        observation_end=payload.end,
                        base_revision=payload.base_revision,
                    ).terminate(
                        RolloutTerminationReason.MAX_PLANS, transition=transition
                    )
                    break
                if (
                    not self.blocking
                    and not worker.pending
                    and should_submit_sequence_planner(
                        planner_mode=options.planner_mode,
                        future_buffer_depth_actions=max(
                            0,
                            max(result.lifecycle.plan, default=cursor - 1) - cursor + 1,
                        ),
                        empty_plan_policy=options.empty_plan_policy,
                        sequence_buffer_threshold=options.buffer_threshold,
                        replan_low_watermark_actions=options.replan_low_watermark_actions,
                    )
                ):
                    payload = request()
                    if payload is not None:
                        worker.submit(plan, payload)
        finally:
            result.live_wall_time_s = time.perf_counter() - start_time
            result.planner_teardown = worker.close()
        return result
