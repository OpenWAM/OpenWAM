"""Model transaction ownership for the shared control lifecycle."""

from __future__ import annotations

import time
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch

from open_wam.models.action_decoders import ActionDecoderRolloutPlan
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.pipelines import VariantRolloutRunner, VariantRolloutSession
from .planning_contracts import ControlAdapter, PlannerRequest, PlannerResult
from .realtime_contracts import PlannedControlStep
from .rollout_receipts import PlannerReceipt
from .rollout_temporal import ResolvedRolloutTemporalContract

Observation = TypeVar("Observation")


class RolloutAdapter(ControlAdapter[Observation], Protocol[Observation]):
    """Environment I/O and representation conversion, never model scheduling."""

    def prepare(
        self,
        observations: tuple[Observation, ...],
        session: VariantRolloutSession,
    ) -> tuple[VisualStageOutputs, PolicyInferContext]: ...

    def observed_history(
        self,
        observations: tuple[Observation, ...],
        actions: tuple[np.ndarray, ...],
        visual: VisualStageOutputs,
        span: PolicyTemporalSpan,
        session: VariantRolloutSession,
    ) -> PolicyObservedHistory: ...

    def controls(
        self,
        plan: ActionDecoderRolloutPlan,
        observation: Observation,
        action_start: int,
        source: str,
        ready_at: float,
        *,
        step_index: int,
    ) -> list[PlannedControlStep]: ...

    def synchronize(self) -> None: ...


class PolicyPlanner(Generic[Observation]):
    """Prepare, reconcile and infer once; never publish or drive an environment."""

    def __init__(
        self,
        runner: VariantRolloutRunner,
        adapter: RolloutAdapter[Observation],
    ) -> None:
        self.runner, self.adapter = runner, adapter
        self.temporal = ResolvedRolloutTemporalContract.from_pipeline(runner.pipeline)
        self.supports_speculative_continuation = runner.pipeline.policy_variant.rollout_contract.supports_speculative_continuation
        self.supports_async = True

    def _prepare(
        self, request: PlannerRequest[VariantRolloutSession, Observation]
    ) -> tuple[VariantRolloutSession, VisualStageOutputs, PolicyInferContext]:
        session = request.session
        visual, context = self.adapter.prepare(request.observations, session)
        self.adapter.synchronize()
        if request.reconcile:
            history = self.adapter.observed_history(
                request.observations,
                request.actions,
                visual,
                self.temporal.observed_span(request.start, request.end),
                session,
            )
            update = self.runner.reconcile_observed_history(
                session=session, history=history
            )
            if not update.applied:
                raise RuntimeError("Policy refused the observed execution interval.")
            session = update.session
        return session, visual, context

    @torch.inference_mode()
    def observe(
        self, request: PlannerRequest[VariantRolloutSession, Observation]
    ) -> VariantRolloutSession:
        return self._prepare(request)[0]

    def plan(
        self, request: PlannerRequest[VariantRolloutSession, Observation]
    ) -> PlannerResult[VariantRolloutSession]:
        observations = request.observations
        start, end = request.start, request.end
        reconcile, source, base_revision = (
            request.reconcile,
            request.source,
            request.base_revision,
        )
        with torch.inference_mode():
            began = time.perf_counter()
            session, visual, context = self._prepare(request)
            prepared_at = time.perf_counter()
            step = self.runner.infer_prepared_step(
                session=session, context=context, visual_outputs=visual
            )
            self.adapter.synchronize()
            ready_at = time.perf_counter()
            generated_span = step.infer_output.policy_output.generated_span
            if generated_span is None or step.action_plan is None:
                raise RuntimeError(
                    "Policy must publish a generated span and decoded actions."
                )
            origin = generated_span.start_frame
            action_start = self.temporal.control_for_frame(origin)
            if reconcile and action_start != end:
                raise RuntimeError(
                    "Generated plan does not follow its observed history."
                )
            steps = self.adapter.controls(
                step.action_plan,
                observations[-1],
                action_start,
                source,
                ready_at,
                step_index=step.session.policy_state.step_index,
            )
            if (
                len(steps)
                != generated_span.frame_count * self.temporal.controls_per_frame
            ):
                raise ValueError(
                    "Control plan length must match its generated model-frame span."
                )
            if [item.absolute_action_index for item in steps] != list(
                range(action_start, action_start + len(steps))
            ):
                raise ValueError(
                    "The adapter must preserve contiguous prediction control indices."
                )
            return PlannerResult(
                session=step.session,
                steps=tuple(steps),
                observation_end=end,
                reconciled=reconcile,
                base_revision=base_revision,
                receipt=PlannerReceipt(
                    source=source,
                    use_observation_update=reconcile,
                    observation_action_start=start,
                    observation_action_end=end,
                    model_generation_frame_start=origin,
                    generation_action_start=action_start,
                    planned_action_ids=tuple(s.absolute_action_index for s in steps),
                    policy_action_shape=tuple(step.action_plan.actions.shape),
                    prepare_s=prepared_at - began,
                    infer_s=ready_at - prepared_at,
                    total_latency_s=ready_at - began,
                    ready_monotonic_s=ready_at,
                ),
            )
