"""LIBERO observation and control adapter for the generic rollout engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from open_wam.configs import ActionTargetRepresentation, ExperimentConfig
from open_wam.configs.enums import DeadlineMissPolicy
from open_wam.integrations.libero_control_plan import (
    build_fallback_frame_actions,
    materialize_sequence_control_action,
    sequence_chunk_to_planned_steps,
)
from open_wam.integrations import libero_rollout
from open_wam.integrations.simulator_configs import LiberoControlConfig
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.pipelines import (
    VariantRolloutRunner,
    VariantRolloutSession,
)
from open_wam.models.action_decoders import ActionDecoderRolloutPlan
from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract
from open_wam.runtime.rollout import prepare_rollout_observation_inputs
from open_wam.runtime.control import ControlCommand, ControlTransition


@dataclass
class LiberoRolloutAdapter:
    runner: VariantRolloutRunner
    config: ExperimentConfig
    environment: object
    prompt: str
    frontend_device: torch.device
    runtime_device: torch.device
    deadline_miss_policy: DeadlineMissPolicy = DeadlineMissPolicy.HOLD_STATE

    def __post_init__(self) -> None:
        self.validate_config(self.config)
        self.temporal = ResolvedRolloutTemporalContract.from_pipeline(self.runner.pipeline)

    @staticmethod
    def validate_config(config: ExperimentConfig) -> None:
        if (
            config.data.action_target.representation
            is not ActionTargetRepresentation.RAW
        ):
            raise ValueError(
                "Realtime reconciliation requires an invertible executed-control "
                "representation. LIBERO pose targets cannot be reconstructed from "
                "OSC controls; use a raw-action checkpoint."
            )

    def prepare(self, observations, session):
        inputs = prepare_rollout_observation_inputs(
            self.runner.pipeline,
            views=libero_rollout.libero_observation_window_to_views(
                list(observations), device=self.frontend_device
            ),
            task_text=(self.prompt,),
            frontend_device=self.frontend_device,
            runtime_device=self.runtime_device,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        visual = self.runner.pipeline.prepare_visual_outputs_from_latents(
            inputs["video_latents"],
            task_text=(self.prompt,),
            text_context=inputs["text_context"],
            negative_text_context=inputs["negative_text_context"],
        )
        state = None
        if self.config.data.action_schema.state_horizon > 0:
            state = (
                libero_rollout.build_libero_state_history(
                    list(observations),
                    state_horizon=self.config.data.action_schema.state_horizon,
                    state_encoding=self.config.data.action_target.state_encoding,
                )
                .unsqueeze(0)
                .to(self.runtime_device)
            )
        return visual, PolicyInferContext(state=state, task_text=(self.prompt,))

    def observed_history(
        self,
        observations,
        actions,
        visual: VisualStageOutputs,
        span: PolicyTemporalSpan,
        session: VariantRolloutSession,
    ) -> PolicyObservedHistory:
        density = self.temporal.controls_per_frame
        self.temporal.validate_observations(
            span, observations=len(observations), actions=len(actions),
            latents=visual.frontend.video_latents.shape[2],
        )
        model_actions = self.runner.pipeline.action_adapter.to_model(
            torch.as_tensor(
                np.stack(actions), device=self.runtime_device, dtype=torch.float32
            ),
        ).unsqueeze(0)
        proprio = None
        if self.config.data.action_schema.state_horizon > 0:
            proprio = libero_rollout.build_libero_state_history(
                list(observations[density::density]),
                state_horizon=span.frame_count,
                state_encoding=self.config.data.action_target.state_encoding,
            )
        cursor = session.policy_state.cursor.current_start_frame
        return PolicyObservedHistory(
            video_latents=visual.frontend.video_latents[:, :, 1:],
            observation_frame_count=span.frame_count * density,
            action_history=model_actions,
            proprio_history=proprio,
            execution_commit=PolicyExecutionCommit(
                PolicyTemporalSpan(span.start_frame, cursor - span.start_frame),
                span.frame_count,
            ),
        )

    def controls(
        self,
        plan: ActionDecoderRolloutPlan,
        observation,
        action_start,
        source,
        ready_at,
        *,
        step_index,
    ):
        actions = self.runner.pipeline.action_adapter.to_source(plan.actions)
        return sequence_chunk_to_planned_steps(
            action_pred=actions.numpy(),
            reference_obs=observation,
            generation_action_start=action_start,
            source=source,
            planner_step_index=step_index,
            ready_monotonic_s=ready_at,
            action_target_representation=self.config.data.action_target.representation,
            rotation_representation=self.config.data.action_target.rotation_representation,
        )

    def materialize(self, step, observation):
        action = materialize_sequence_control_action(
            step,
            current_obs=observation,
            control_config=LiberoControlConfig(),
            gripper_representation=self.config.data.action_target.gripper_representation,
        )
        return ControlCommand(action=action, source_action=action)

    def fallback(self, last_action, observation):
        action = build_fallback_frame_actions(
            action_dim=7,
            action_per_frame=1,
            policy=self.deadline_miss_policy,
            last_action=np.zeros(7, dtype=np.float32)
            if last_action is None
            else last_action,
        )[0]
        return ControlCommand(action=action, source_action=action)

    def step(self, action):
        obs, reward, done, info = self.environment.step(action.astype(np.float32, copy=False))
        return ControlTransition(
            observation=libero_rollout.extract_libero_rollout_observation(obs),
            reward=reward, done=bool(done), success=bool(self.environment.check_success()),
            info=info,
        )

    def synchronize(self):
        for device in set((self.frontend_device, self.runtime_device)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
