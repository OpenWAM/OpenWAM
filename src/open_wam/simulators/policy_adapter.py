"""Normalized simulator observations and controls for the policy lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from open_wam.configs import DataConfig
from open_wam.configs.enums import ProprioContextMode
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.pipelines import VariantRolloutRunner
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.runtime.rollout import prepare_rollout_observation_inputs
from .contracts import SimulatorObservation
from open_wam.runtime.control import ControlCommand
from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract


@dataclass
class SimulatorPolicyAdapter:
    """Representation boundary, independent of environment stepping and scheduling."""

    runner: VariantRolloutRunner
    data: DataConfig
    device: torch.device
    control: Callable[[np.ndarray], ControlCommand]

    def __post_init__(self) -> None:
        self.temporal = ResolvedRolloutTemporalContract.from_pipeline(self.runner.pipeline)

    def _state(self, observations, *, horizon):
        schema = self.data.action_schema
        requirements = self.runner.pipeline.policy_variant.pipeline_requirements(
            default_action_dim=schema.action_dim,
            default_action_horizon=schema.action_horizon,
            default_state_dim=schema.state_dim,
        )
        if requirements.proprio_context_mode is ProprioContextMode.NONE:
            return None
        return _state_history_tensor(
            tuple(obs.state for obs in observations),
            state_dim=requirements.state_dim,
            state_horizon=horizon,
            device=self.device,
        )

    def prepare(self, observations: tuple[SimulatorObservation, ...], session):
        state = self._state(observations, horizon=self.data.action_schema.state_horizon)
        # Preserve the exact anchor + executed interval. Padding RGB here would
        # create fictitious recurrent history and shift the VAE's time anchors.
        views = {
            name: torch.as_tensor(np.stack([obs.views[name] for obs in observations]))
            .unsqueeze(0)
            .to(self.device)
            for name in self.data.camera_names
        }
        inputs = prepare_rollout_observation_inputs(
            self.runner.pipeline,
            views=views,
            task_text=session.task_text,
            frontend_device=self.device,
            runtime_device=self.device,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        visual = self.runner.pipeline.prepare_visual_outputs_from_latents(
            inputs["video_latents"],
            task_text=session.task_text,
            text_context=inputs["text_context"],
            negative_text_context=inputs["negative_text_context"],
        )
        return visual, PolicyInferContext(state=state, task_text=session.task_text)

    def observed_history(self, observations, actions, visual, span, session):
        density = self.temporal.controls_per_frame
        self.temporal.validate_observations(
            span, observations=len(observations), actions=len(actions),
            latents=visual.frontend.video_latents.shape[2],
        )
        proprio = self._state(observations[density::density], horizon=span.frame_count)
        return PolicyObservedHistory(
            video_latents=visual.frontend.video_latents[:, :, 1:],
            observation_frame_count=len(actions),
            action_history=self.runner.pipeline.action_adapter.to_model(
                torch.as_tensor(
                    np.stack(actions), device=self.device, dtype=torch.float32
                )
            ).unsqueeze(0),
            proprio_history=None if proprio is None else proprio[0],
            execution_commit=PolicyExecutionCommit(
                PolicyTemporalSpan(
                    span.start_frame,
                    session.policy_state.cursor.current_start_frame - span.start_frame,
                ),
                span.frame_count,
            ),
        )

    def controls(self, plan, observation, action_start, source, ready_at, *, step_index):
        actions = self.runner.pipeline.action_adapter.to_source(
            plan.actions
        )
        return [
            PlannedControlStep(
                absolute_action_index=action_start + i,
                generation_action_start=action_start,
                source=source,
                planner_step_index=step_index,
                ready_monotonic_s=ready_at,
                raw_action=action,
            )
            for i, action in enumerate(actions.numpy())
        ]

    def materialize(self, step, observation):
        if step.raw_action is None:
            raise ValueError(
                "The normalized simulator adapter requires a dataset-source action."
            )
        return self.control(step.raw_action)

    def fallback(self, last_action, observation):
        raise ValueError(
            "Generic simulator rollouts wait for a plan; fallback requires a robot-specific controller."
        )

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _state_history_tensor(
    history: tuple[np.ndarray | None, ...],
    *,
    state_dim: int,
    state_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate model-ready states without changing their temporal or feature axes."""

    if state_horizon <= 0:
        raise ValueError(
            "Proprio-conditioned policies require a positive state horizon."
        )
    if not history:
        raise ValueError("Proprio-conditioned policies require observed state.")
    states = []
    for index, value in enumerate(history):
        if value is None:
            raise ValueError(f"Missing required proprio state at observation {index}.")
        state = np.asarray(value, dtype=np.float32)
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"Proprio state at observation {index} must be finite with shape "
                f"({state_dim},), got {state.shape}. The simulator adapter must "
                "provide state in the configured model-ready encoding."
            )
        states.append(state)
    # A short startup history repeats its known anchor, never a missing row.
    states = [states[0]] * max(0, state_horizon - len(states)) + states
    return torch.as_tensor(
        np.stack(states[-state_horizon:]), dtype=torch.float32, device=device
    ).unsqueeze(0)
