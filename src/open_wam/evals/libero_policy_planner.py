"""Blocking LIBERO model transactions for the shared rollout lifecycle.

The planner owns frontend state and composition, never simulator execution.
Sessions keep producer/consumer histories separate and retain only the bounded
observation window and the prepared streaming interval needed by the next call.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from open_wam.configs import DynamicsObjective
from open_wam.evals.libero_episode_artifacts import (
    LiberoEpisodeArtifacts,
    _summarize_policy_debug,
)
from open_wam.evals.libero_policy_composition import (
    VideoActionComposition,
    infer_video_conditioned_action,
)
from open_wam.evals.libero_policy_inputs import (
    _build_infer_context,
    _prepare_policy_visual_outputs,
    _select_model_obs_window,
)
from open_wam.evals.libero_policy_runtime import (
    LiberoPolicyRuntime,
    PolicyActionRoute,
    _frame_chunk_size,
    print_rollout_event,
)
from open_wam.evals.libero_rollout_artifacts import extract_predicted_latents
from open_wam.integrations import (
    build_libero_state_history,
    libero_observation_window_to_views,
)
from open_wam.models.common.rollout_history import (
    build_executed_action_history_tensor,
    resolve_execute_action_steps,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyExecutionCommit,
    PolicyObservedHistory,
    PolicyRecurrentHistoryPolicy,
    PolicyTemporalSpan,
    PolicyVideoGenerationRequest,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.pipelines import VariantRolloutSession, require_generated_video
from open_wam.runtime.planning_contracts import PlannerRequest, PlannerResult
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.runtime.rollout_receipts import PlannerReceipt
from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract
from open_wam.utils import seed_everywhere
from open_wam.utils.seeding import preserve_rng_state

if TYPE_CHECKING:
    from open_wam.evals.libero_policy_rollout import LiberoPolicyEpisodeOptions

Observation = dict[str, np.ndarray]


@dataclass(frozen=True)
class LiberoPlannerSession:
    producer: VariantRolloutSession
    consumer: VariantRolloutSession | None
    frame_window: tuple[Observation, ...]
    chunk_index: int = 0
    speculative_span: PolicyTemporalSpan | None = None
    streaming_observations: tuple[Observation, ...] = ()
    producer_visual: VisualStageOutputs | None = None
    consumer_visual: VisualStageOutputs | None = None


class LiberoPolicyPlanner:
    """Serial streaming-frontend transaction; reusable across policy architectures."""

    supports_async = False
    supports_speculative_continuation = False

    def __init__(
        self,
        resources: LiberoPolicyRuntime,
        options: LiberoPolicyEpisodeOptions,
        artifacts: LiberoEpisodeArtifacts,
        *,
        prompt: str,
        initial_timestep: int,
        composition: VideoActionComposition | None,
    ) -> None:
        self.resources, self.options, self.artifacts = resources, options, artifacts
        self.prompt, self.initial_timestep, self.composition = (
            prompt,
            initial_timestep,
            composition,
        )
        self.consumer = None if composition is None else composition.runtime
        self.action_runtime = resources if self.consumer is None else self.consumer
        self.temporal = ResolvedRolloutTemporalContract.from_pipeline(
            self.action_runtime.pipeline
        )

    def start(self, observations: list[Observation]) -> LiberoPlannerSession:
        producer = self.resources.runner.reset(task_text=(self.prompt,))
        consumer = (
            None
            if self.consumer is None
            else self.consumer.runner.reset(task_text=(self.prompt,))
        )
        if self.resources.use_lingbot_streaming_vae:
            self.resources.pipeline.visual_tower.reset_runtime_state()
            if self.consumer is not None:
                self.consumer.pipeline.visual_tower.reset_runtime_state()
        return LiberoPlannerSession(
            producer,
            consumer,
            tuple(LiberoEpisodeArtifacts.copy_observation(obs) for obs in observations),
        )

    def _encode(
        self,
        runtime: LiberoPolicyRuntime,
        session: VariantRolloutSession,
        observations: list[Observation] | tuple[Observation, ...],
        *,
        streaming: bool,
        preserve: bool = False,
        isolate_rng: bool = False,
    ) -> VisualStageOutputs:
        views = libero_observation_window_to_views(
            observations, device=runtime.frontend_device
        )
        with preserve_rng_state() if isolate_rng else nullcontext():
            return _prepare_policy_visual_outputs(
                runtime.pipeline,
                views=views,
                task_text=(self.prompt,),
                frontend_device=runtime.frontend_device,
                runtime_device=runtime.runtime_device,
                use_streaming_frontend=streaming,
                preserve_stream_cache=preserve,
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
            )

    def observe(
        self, request: PlannerRequest[LiberoPlannerSession, Observation]
    ) -> LiberoPlannerSession:
        state = request.session
        observations = request.observations[1:]
        if not observations or len(observations) != len(request.actions):
            raise ValueError(
                "Observed intervals require an anchor and each executed control observation."
            )
        span = state.speculative_span
        if span is None:
            raise ValueError("History reconciliation requires a preceding prediction.")
        commit = _build_execution_commit(
            generation_frame_start=span.start_frame,
            speculative_frame_count=span.frame_count,
            executed_action_count=len(request.actions),
            action_per_frame=self.temporal.controls_per_frame,
            terminal=False,
        )
        action_history = build_executed_action_history_tensor(
            request.actions,
            action_per_frame=self.temporal.controls_per_frame,
            action_dim=request.actions[0].shape[-1],
        )
        streaming = self.resources.use_lingbot_streaming_vae
        producer_visual = consumer_visual = None
        if streaming:
            producer_visual = self._encode(
                self.resources,
                state.producer,
                observations,
                streaming=True,
                preserve=True,
            )
            self.artifacts.event(
                state.chunk_index - 1,
                "lingbot_streaming_vae_update",
                real_obs_frames=len(observations),
                real_latent_frames=int(producer_visual.frontend.video_latents.shape[2]),
            )
            if self.consumer is not None:
                consumer_visual = self._encode(
                    self.consumer,
                    state.consumer,
                    observations,
                    streaming=True,
                    preserve=True,
                    isolate_rng=True,
                )

        def reconcile(runtime, session, visual, history_policy, phase, *, required):
            if (
                history_policy
                is not PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
            ):
                return session
            if visual is None:
                visual = self._encode(runtime, session, observations, streaming=False)
            history = runtime.runner.reconcile_observed_history(
                session=session,
                history=PolicyObservedHistory(
                    video_latents=visual.frontend.video_latents,
                    observation_frame_count=len(observations),
                    action_history=runtime.pipeline.action_adapter.to_model(
                        action_history
                    )
                    if runtime.pipeline.action_adapter is not None
                    else None,
                    proprio_history=build_libero_state_history(
                        observations,
                        state_horizon=len(observations),
                        state_encoding=runtime.config.data.action_target.state_encoding,
                    ),
                    execution_commit=commit,
                ),
            )
            if required and not history.applied:
                raise RuntimeError(
                    "Composed policy declared explicit reconciliation but did not apply it."
                )
            self.artifacts.event(state.chunk_index - 1, phase, **history.debug)
            return history.session

        producer_policy = (
            self.composition.producer_plan.recurrent_history_policy
            if self.composition is not None
            else self.resources.pipeline.policy_variant.inference_capabilities.recurrent_history_policy
        )
        producer = reconcile(
            self.resources,
            state.producer,
            producer_visual,
            producer_policy,
            "packed_history_warmup",
            required=self.composition is not None,
        )
        consumer = state.consumer
        if self.consumer is not None:
            consumer = reconcile(
                self.consumer,
                consumer,
                consumer_visual,
                self.composition.consumer_plan.recurrent_history_policy,
                "video_action_composition_packed_history_warmup",
                required=True,
            )
        return replace(
            state,
            producer=producer,
            consumer=consumer,
            frame_window=(state.frame_window + observations)[
                -self.resources.raw_window_frames :
            ],
            streaming_observations=observations if streaming else (),
            producer_visual=producer_visual,
            consumer_visual=consumer_visual,
        )

    def plan(
        self, request: PlannerRequest[LiberoPlannerSession, Observation]
    ) -> PlannerResult[LiberoPlannerSession]:
        began = time.perf_counter()
        state = self.observe(request) if request.reconcile else request.session
        # Preserve the old order: observation encoding/commits precede chunk reseeding.
        if self.options.seed is not None:
            seed_everywhere(self.options.seed + state.chunk_index)
        return self._predict(request, state, began)

    @torch.inference_mode()
    def _predict(
        self,
        request: PlannerRequest[LiberoPlannerSession, Observation],
        state: LiberoPlannerSession,
        began: float,
    ) -> PlannerResult[LiberoPlannerSession]:
        args, resources = self.options, self.resources
        config, runner = resources.config, resources.runner
        consumer_runtime = self.consumer
        action_config, action_pipeline = (
            self.action_runtime.config,
            self.action_runtime.pipeline,
        )
        video_action_composition = self.composition
        uses_composition = video_action_composition is not None
        action_route = PolicyActionRoute(args.policy_action_route)
        session, action_consumer_session = state.producer, state.consumer
        chunk_count, seed, prompt = state.chunk_index, args.seed, self.prompt
        frame_window, log_coordinates = state.frame_window, self.artifacts.coordinates
        _print_log = print_rollout_event
        _print_log(
            "stage",
            {
                "name": "chunk_prepare_start",
                **log_coordinates,
                "chunk_index": chunk_count,
                "env_timestep": self.initial_timestep + request.end,
            },
        )
        if resources.use_lingbot_streaming_vae and chunk_count > 0:
            model_obs_window = list(state.streaming_observations)
            visual_outputs, consumer_visual_outputs = (
                state.producer_visual,
                state.consumer_visual,
            )
            if visual_outputs is None or not model_obs_window:
                raise RuntimeError(
                    "Streaming planning requires a committed observed interval."
                )
            frontend_path = "lingbot_streaming_vae"
        else:
            model_obs_window = _select_model_obs_window(
                list(frame_window),
                chunk_index=chunk_count,
                startup_model_obs_frames=resources.startup_model_obs_frames,
            )
            visual_outputs = self._encode(
                resources,
                session,
                model_obs_window,
                streaming=chunk_count == 0 or resources.use_lingbot_streaming_vae,
            )
            consumer_visual_outputs = (
                None
                if consumer_runtime is None
                else self._encode(
                    consumer_runtime,
                    action_consumer_session,
                    model_obs_window,
                    streaming=chunk_count == 0
                    or consumer_runtime.use_lingbot_streaming_vae,
                    isolate_rng=True,
                )
            )
            frontend_path = (
                "lingbot_streaming_vae_init"
                if resources.use_lingbot_streaming_vae
                else ("streaming" if chunk_count == 0 else "offline")
            )
        prepared_at = time.perf_counter()
        _print_log(
            "stage",
            {
                "name": "chunk_infer_start",
                **log_coordinates,
                "chunk_index": int(chunk_count),
                "env_timestep": int((self.initial_timestep + request.end)),
                "model_obs_frames": len(model_obs_window),
                "video_latent_frames": int(
                    visual_outputs.frontend.video_latents.shape[2]
                ),
                "frontend_path": frontend_path,
            },
        )
        infer_context = _build_infer_context(
            prompt,
            action_device=resources.action_device,
            model_obs_window=model_obs_window,
            config=config,
            runtime_device=resources.runtime_device,
            inference_window_size=args.inference_window_size,
            rollout_frame_chunk_size=args.rollout_frame_chunk_size,
            action_only_rollout=bool(args.action_only_rollout),
            output_request=(
                video_action_composition.producer_plan.output_request
                if uses_composition
                else None
            ),
            video_generation=(
                PolicyVideoGenerationRequest(
                    frame_count=(
                        int(args.rollout_frame_chunk_size)
                        if args.rollout_frame_chunk_size is not None
                        else _frame_chunk_size(action_config)
                    ),
                )
                if uses_composition
                else None
            ),
        )
        pre_infer_policy_state = None
        if (
            action_route is PolicyActionRoute.JOINT_VIDEO_THEN_IDM
            and session.policy_state is not None
        ):
            pre_infer_policy_state = session.policy_state
        infer_session = runner.reset(
            task_text=session.task_text,
            text_context=session.text_context,
            negative_text_context=session.negative_text_context,
        )
        infer_session = replace(
            infer_session,
            policy_state=(
                None if args.reset_policy_state_each_chunk else session.policy_state
            ),
        )
        step_output = runner.infer_prepared_step(
            session=infer_session,
            context=infer_context,
            visual_outputs=visual_outputs,
        )
        primary_step_output = step_output
        primary_infer_output = step_output.infer_output
        infer_output = primary_infer_output
        route_predicted_latents = extract_predicted_latents(infer_output)
        generated_video = (
            require_generated_video(
                primary_infer_output,
                request=infer_context.video_generation,
            )
            if uses_composition
            else None
        )
        if generated_video is not None:
            route_predicted_latents = generated_video.latents
        action_consumer_inference_seed = None
        if action_route is PolicyActionRoute.JOINT_VIDEO_THEN_IDM:
            if (
                not isinstance(route_predicted_latents, torch.Tensor)
                or int(route_predicted_latents.shape[2]) <= 0
            ):
                raise RuntimeError(
                    "joint_video_then_idm route requires joint rollout to produce a non-empty predicted video chunk."
                )
            idm_context = _build_infer_context(
                prompt,
                action_device=resources.action_device,
                model_obs_window=model_obs_window,
                config=config,
                runtime_device=resources.runtime_device,
                inference_window_size=args.inference_window_size,
                rollout_frame_chunk_size=args.rollout_frame_chunk_size,
                action_only_rollout=False,
            )
            idm_context = replace(
                idm_context,
                dynamics=DynamicsRolloutRequest(
                    objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                    clean_video=route_predicted_latents.detach().to(
                        device=resources.runtime_device,
                        dtype=route_predicted_latents.dtype,
                    ),
                ),
            )
            idm_session = runner.reset(
                task_text=session.task_text,
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
            )
            idm_session = replace(
                idm_session,
                policy_state=(
                    None
                    if args.reset_policy_state_each_chunk
                    else pre_infer_policy_state
                ),
            )
            step_output = runner.infer_prepared_step(
                session=idm_session,
                context=idm_context,
                visual_outputs=visual_outputs,
            )
            infer_output = step_output.infer_output
        elif uses_composition:
            if generated_video is None:
                raise RuntimeError(
                    "Video/action composition requires its producer stage "
                    "to publish a non-empty generated-video chunk."
                )
            if (
                video_action_composition is None
                or consumer_runtime is None
                or action_consumer_session is None
                or consumer_visual_outputs is None
            ):
                raise RuntimeError(
                    "The composed action route was selected without complete "
                    "consumer runtime state."
                )
            consumer_step = infer_video_conditioned_action(
                video_action_composition,
                session=action_consumer_session,
                visual_outputs=consumer_visual_outputs,
                model_obs_window=model_obs_window,
                prompt=prompt,
                generated_video=generated_video,
                inference_window_size=(args.inference_window_size),
                reset_policy_state=bool(args.reset_policy_state_each_chunk),
                rollout_seed=seed,
                chunk_index=chunk_count,
                producer_rng_device=resources.runtime_device,
            )
            step_output = consumer_step.rollout
            infer_output = step_output.infer_output
            action_consumer_inference_seed = consumer_step.inference_seed
        _print_log(
            "stage",
            {
                "name": "chunk_infer_done",
                **log_coordinates,
                "chunk_index": int(chunk_count),
                "env_timestep": int((self.initial_timestep + request.end)),
                "policy_action_route": str(action_route.value),
            },
        )
        if uses_composition:
            session = primary_step_output.session
            action_consumer_session = step_output.session
        else:
            session = step_output.session
        if step_output.action_plan is None:
            raise RuntimeError("The action stage did not publish an executable plan.")
        actions = action_pipeline.action_adapter.to_source(
            step_output.action_plan.actions
        ).numpy()
        configured_frame_chunk_size = _frame_chunk_size(action_config)
        configured_action_per_frame = (
            action_pipeline.policy_variant.rollout_contract.action_tokens_per_frame
        )
        action_per_frame = configured_action_per_frame
        if int(actions.shape[0]) % int(action_per_frame) != 0:
            raise ValueError(
                "Policy action output length must be divisible by configured action_per_frame, "
                f"got action_shape={actions.shape}, action_per_frame={action_per_frame}."
            )
        frame_chunk_size = int(actions.shape[0]) // int(action_per_frame)
        execute_action_steps = resolve_execute_action_steps(
            args.execute_action_steps,
            execute_frame_chunk_size=args.execute_frame_chunk_size,
            action_horizon=int(actions.shape[0]),
            action_per_frame=action_per_frame,
        )
        predicted_latents = extract_predicted_latents(infer_output)
        if action_route in {
            PolicyActionRoute.JOINT_VIDEO_THEN_IDM,
            PolicyActionRoute.GENERATED_VIDEO_THEN_ACTION,
        } and isinstance(route_predicted_latents, torch.Tensor):
            predicted_latents = route_predicted_latents
        policy_debug = _summarize_policy_debug(infer_output.policy_output)
        chunk_log = {
            **log_coordinates,
            "chunk_index": chunk_count,
            "phase": "infer",
            "env_timestep_before": int((self.initial_timestep + request.end)),
            "window_size": len(frame_window),
            "model_obs_frames": len(model_obs_window),
            "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
            "frontend_path": frontend_path,
            "action_shape": list(actions.shape),
            "execute_action_steps": int(execute_action_steps),
            "configured_frame_chunk_size": int(configured_frame_chunk_size),
            "rollout_frame_chunk_size": int(frame_chunk_size),
            "execute_frame_chunk_size": int(execute_action_steps // action_per_frame),
            "predicted_latents_shape": None
            if not isinstance(predicted_latents, torch.Tensor)
            else list(predicted_latents.shape),
            "policy_action_route": action_route.value,
            "first_action_preview": [float(v) for v in actions[0].tolist()],
            "policy_debug": policy_debug,
        }
        if uses_composition:
            chunk_log["video_action_composition"] = {
                "action_consumer_inference_seed": action_consumer_inference_seed,
                "producer_policy_debug": _summarize_policy_debug(
                    primary_infer_output.policy_output
                ),
            }

        self.artifacts.inferred(chunk_log, predicted_latents)
        origin = infer_output.policy_output.generation_frame_start
        if origin is None:
            raise RuntimeError(
                "Policy rollout did not publish its typed generation frame origin."
            )
        ready_at = time.perf_counter()
        steps = tuple(
            PlannedControlStep(
                absolute_action_index=request.end + index,
                generation_action_start=request.end,
                generation_frame_start=int(origin),
                source=request.source,
                planner_step_index=chunk_count,
                ready_monotonic_s=ready_at,
                raw_action=action.copy(),
            )
            for index, action in enumerate(actions[:execute_action_steps])
        )
        next_state = replace(
            state,
            producer=session,
            consumer=action_consumer_session,
            chunk_index=chunk_count + 1,
            speculative_span=PolicyTemporalSpan(int(origin), frame_chunk_size),
            producer_visual=None,
            consumer_visual=None,
            streaming_observations=(),
        )
        return PlannerResult(
            session=next_state,
            steps=steps,
            observation_end=request.end,
            reconciled=request.reconcile,
            base_revision=request.base_revision,
            receipt=PlannerReceipt(
                source=request.source,
                use_observation_update=request.reconcile,
                observation_action_start=request.start,
                observation_action_end=request.end,
                model_generation_frame_start=int(origin),
                generation_action_start=request.end,
                planned_action_ids=tuple(step.absolute_action_index for step in steps),
                policy_action_shape=tuple(actions.shape),
                prepare_s=prepared_at - began,
                infer_s=ready_at - prepared_at,
                total_latency_s=ready_at - began,
                ready_monotonic_s=ready_at,
            ),
        )


def _build_execution_commit(
    *,
    generation_frame_start: int,
    speculative_frame_count: int,
    executed_action_count: int,
    action_per_frame: int,
    terminal: bool,
) -> PolicyExecutionCommit | None:
    """Describe the model-frame interval actually committed by execution."""

    if int(executed_action_count) <= 0 or terminal:
        return None
    if int(action_per_frame) <= 0:
        raise ValueError(
            f"Execution commits require action_per_frame > 0, got {action_per_frame}."
        )
    if int(executed_action_count) % int(action_per_frame) != 0:
        raise ValueError(
            "Non-terminal execution must commit complete model-frame action "
            "groups; "
            f"executed_actions={executed_action_count}, "
            f"action_per_frame={action_per_frame}."
        )
    speculative_span = PolicyTemporalSpan(
        start_frame=int(generation_frame_start),
        frame_count=int(speculative_frame_count),
    )
    return PolicyExecutionCommit(
        speculative_span=speculative_span,
        executed_frame_count=int(executed_action_count) // int(action_per_frame),
    )
