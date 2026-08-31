"""Checkpoint-backed DualExpert/GJD execution for LIBERO benchmark adapters.

This module consumes a loaded runtime and owns the exact per-episode policy
loop. Checkpoint/config/device composition lives in ``libero_dual_expert_runtime``;
LIBERO observation and environment mechanics live in ``open_wam.integrations``.
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_wam.configs import DynamicsObjective
from open_wam.evals.libero_dual_expert_composition import (
    ExternalIdmComposition,
    build_composed_component_report,
    infer_external_idm_action,
)
from open_wam.evals.libero_dual_expert_inputs import (
    _build_infer_context,
    _prepare_dual_expert_visual_outputs,
    _select_model_obs_window,
)
from open_wam.evals.libero_dual_expert_runtime import (
    CURRENT_FRONTEND_ENCODE_MODE,
    DEPRECATED_FRONTEND_ENCODE_MODE,
    DUAL_EXPERT_ACTION_ROUTES,
    DUAL_EXPERT_GJD_ACTION_ROUTES,
    DualExpertActionRoute,
    DualExpertLiberoLoadOptions,
    DualExpertLiberoRuntime,
    _action_per_frame,
    _frame_chunk_size,
    load_dual_expert_libero_runtime,
    print_rollout_event,
    uses_video_action_composition,
)
from open_wam.evals.libero_rollout_artifacts import (
    LiberoRolloutArtifactIdentity,
    LiberoRolloutArtifactOptions,
    LiberoRolloutArtifactPayload,
    append_predicted_latent_chunk,
    extract_predicted_latents,
    persist_libero_rollout_artifacts,
)
from open_wam.integrations import (
    LIBERO_ROLLOUT_VIEW_KEYS,
    LiberoTaskSpec,
    build_libero_offscreen_env,
    build_libero_state_history,
    extract_libero_rollout_observation,
    initialize_libero_observation_window,
    libero_observation_window_to_views,
    load_libero_task_init_states,
    resolve_libero_task_by_id,
)
from open_wam.models.common.rollout_history import (
    build_executed_action_history_tensor as _build_shared_executed_action_history_tensor,
)
from open_wam.models.common.rollout_history import (
    resolve_execute_action_steps as _resolve_shared_execute_action_steps,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyExecutionCommit,
    PolicyRecurrentHistoryPolicy,
    PolicyTemporalSpan,
    PolicyVideoGenerationRequest,
)
from open_wam.models.policy_variants.contracts import PolicyInferOutput
from open_wam.models.policy_variants.dual_expert.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertInferArtifacts,
)
from open_wam.pipelines import require_generated_video
from open_wam.utils import seed_everywhere

REPO_ROOT = Path(__file__).resolve().parents[3]

LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS
_build_executed_action_history_tensor = _build_shared_executed_action_history_tensor
_extract_obs = extract_libero_rollout_observation
_init_single_env = initialize_libero_observation_window
_obs_list_to_views = libero_observation_window_to_views
_resolve_execute_action_steps = _resolve_shared_execute_action_steps
_print_log = print_rollout_event
__all__ = [
    "CURRENT_FRONTEND_ENCODE_MODE",
    "DEPRECATED_FRONTEND_ENCODE_MODE",
    "DUAL_EXPERT_ACTION_ROUTES",
    "DUAL_EXPERT_GJD_ACTION_ROUTES",
    "DualExpertLiberoEpisodeOptions",
    "DualExpertLiberoLoadOptions",
    "DualExpertLiberoRuntime",
    "DualExpertLiberoTaskResources",
    "construct_dual_expert_libero_env",
    "load_dual_expert_libero_runtime",
    "print_rollout_event",
    "resolve_dual_expert_libero_task_resources",
    "run_dual_expert_libero_episode",
]


@dataclass(frozen=True)
class DualExpertLiberoTaskResources:
    """Resolved simulator task metadata and its fixed initialization states."""

    task_spec: LiberoTaskSpec
    prompt: str
    init_states: Any


@dataclass(frozen=True)
class DualExpertLiberoEpisodeOptions:
    """Policy, simulator, and artifact choices for one loaded-model episode."""

    benchmark: str
    task_id: int
    episode_idx: int
    max_timestep: int
    max_chunks: int | None
    execute_action_steps: int | None
    execute_frame_chunk_size: int | None
    dual_expert_rollout_frame_chunk_size: int | None
    dual_expert_inference_window_size: int | None
    dual_expert_action_only_rollout: bool
    dual_expert_gjd_action_route: str
    reset_policy_state_each_chunk: bool
    max_imagined_latent_frames: int | None
    output_dir: str | Path
    suffix: str
    video_fps: float
    seed: int | None
    save_rollout_video: bool = False
    skip_comparison_video: bool = False


def _raw_env_done(env: object) -> bool:
    """Return robosuite's terminal flag behind LIBERO's success-only wrapper."""
    raw_env = getattr(env, "env", env)
    return bool(getattr(raw_env, "done", False))


def resolve_dual_expert_libero_task_resources(
    benchmark_name: str,
    task_id: int,
) -> DualExpertLiberoTaskResources:
    """Resolve task metadata and initialization states for one benchmark task."""

    task_spec, prompt = _resolve_task_spec(benchmark_name, task_id)
    return DualExpertLiberoTaskResources(
        task_spec=task_spec,
        prompt=prompt,
        init_states=load_libero_task_init_states(task_spec),
    )


def construct_dual_expert_libero_env(task_spec: LiberoTaskSpec) -> Any:
    """Construct the exact 128px offscreen environment with bounded retries."""

    return _construct_single_env(task_spec)


def run_dual_expert_libero_episode(
    args: DualExpertLiberoEpisodeOptions,
    resources: DualExpertLiberoRuntime,
    task_resources: DualExpertLiberoTaskResources,
    env: Any,
    *,
    include_episode_coordinates: bool,
    close_env_after_rollout: bool,
    external_idm: ExternalIdmComposition | None = None,
) -> dict[str, object]:
    """Execute one exact dual-expert/GJD LIBERO episode with loaded resources."""

    if env is None:
        raise RuntimeError(
            "Failed to construct LIBERO OffScreenRenderEnv after 5 retries."
        )
    task_id = int(args.task_id)
    episode_idx = int(args.episode_idx)
    seed = args.seed
    prompt = task_resources.prompt
    init_states = task_resources.init_states
    log_coordinates = (
        {"task_id": task_id, "episode_idx": episode_idx}
        if include_episode_coordinates
        else {}
    )

    def chunk_log_label(chunk_index: int) -> str:
        if include_episode_coordinates:
            return f"task_{task_id}_episode_{episode_idx}_chunk_{chunk_index}"
        return f"chunk_{chunk_index}"

    config = resources.config
    pipeline = resources.pipeline
    runner = resources.runner
    use_lingbot_streaming_vae = bool(resources.use_lingbot_streaming_vae)
    action_route = DualExpertActionRoute(args.dual_expert_gjd_action_route)
    uses_external_idm = uses_video_action_composition(action_route)
    if uses_external_idm != (external_idm is not None):
        raise ValueError(
            "The generated-video external-IDM route and loaded external composition "
            "must be supplied together."
        )
    external_runtime = None if external_idm is None else external_idm.runtime
    action_config = config if external_runtime is None else external_runtime.config
    if external_runtime is not None and (
        bool(external_runtime.use_lingbot_streaming_vae) != use_lingbot_streaming_vae
    ):
        raise ValueError(
            "Primary and external IDM runtimes must use the same frontend encode mode."
        )

    try:
        _print_log(
            "stage", {"name": "init_env_rollout_start", "episode_idx": int(episode_idx)}
        )
        initial_obs_window = _init_single_env(
            env,
            init_states[episode_idx % len(init_states)],
            num_frames=resources.startup_model_obs_frames,
            init_steps=resources.startup_env_init_steps,
        )
        _print_log(
            "stage",
            {
                "name": "init_env_rollout_done",
                "initial_window": len(initial_obs_window),
                "startup_model_obs_frames": int(resources.startup_model_obs_frames),
                "startup_env_init_steps": int(resources.startup_env_init_steps),
                "startup_env_steps_executed": int(
                    max(
                        resources.startup_env_init_steps,
                        resources.startup_model_obs_frames,
                    )
                ),
            },
        )
        frame_window: deque[dict[str, np.ndarray]] = deque(
            maxlen=resources.raw_window_frames
        )
        for obs in initial_obs_window:
            frame_window.append(
                {key: np.array(value, copy=True) for key, value in obs.items()}
            )

        predicted_latent_chunks: list[torch.Tensor] = []
        rollout_frames: list[dict[str, np.ndarray]] = [
            {key: np.array(value, copy=True) for key, value in obs.items()}
            for obs in list(frame_window)
        ]
        action_trace: list[np.ndarray] = []
        chunk_logs: list[dict[str, object]] = []
        done = False
        terminal = False
        chunk_count = 0
        session = runner.reset(task_text=(prompt,))
        external_idm_session = (
            None
            if external_runtime is None
            else external_runtime.runner.reset(task_text=(prompt,))
        )
        streaming_next_visual_outputs = None
        external_streaming_next_visual_outputs = None
        streaming_next_obs_window: list[dict[str, np.ndarray]] | None = None
        if use_lingbot_streaming_vae:
            pipeline.visual_tower.reset_runtime_state()
            if external_runtime is not None:
                external_runtime.pipeline.visual_tower.reset_runtime_state()

        while env.env.timestep < args.max_timestep and not done and not terminal:
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break
            if seed is not None:
                seed_everywhere(seed + chunk_count)

            with torch.inference_mode():
                _print_log(
                    "stage",
                    {
                        "name": "chunk_prepare_start",
                        **log_coordinates,
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                    },
                )
                if use_lingbot_streaming_vae and chunk_count > 0:
                    if (
                        streaming_next_visual_outputs is None
                        or streaming_next_obs_window is None
                    ):
                        raise RuntimeError(
                            "LingBot streaming VAE rollout expected encoded observations from the previous "
                            f"environment chunk before chunk_index={chunk_count}."
                        )
                    model_obs_window = streaming_next_obs_window
                    visual_outputs = streaming_next_visual_outputs
                    streaming_next_visual_outputs = None
                    streaming_next_obs_window = None
                    frontend_path = "lingbot_streaming_vae"
                else:
                    model_obs_window = _select_model_obs_window(
                        list(frame_window),
                        chunk_index=chunk_count,
                        startup_model_obs_frames=resources.startup_model_obs_frames,
                    )
                    views = _obs_list_to_views(
                        model_obs_window, device=resources.frontend_device
                    )
                    visual_outputs = _prepare_dual_expert_visual_outputs(
                        pipeline,
                        views=views,
                        task_text=(prompt,),
                        frontend_device=resources.frontend_device,
                        runtime_device=resources.runtime_device,
                        use_streaming_frontend=chunk_count == 0
                        or use_lingbot_streaming_vae,
                        preserve_stream_cache=False,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                    frontend_path = (
                        "lingbot_streaming_vae_init"
                        if use_lingbot_streaming_vae
                        else ("streaming" if chunk_count == 0 else "offline")
                    )
                external_visual_outputs = None
                if external_runtime is not None:
                    if use_lingbot_streaming_vae and chunk_count > 0:
                        if external_streaming_next_visual_outputs is None:
                            raise RuntimeError(
                                "External IDM streaming VAE expected encoded "
                                "observations from the previous chunk."
                            )
                        external_visual_outputs = external_streaming_next_visual_outputs
                        external_streaming_next_visual_outputs = None
                    else:
                        if external_idm_session is None:
                            raise RuntimeError(
                                "External IDM session was not initialized."
                            )
                        external_views = _obs_list_to_views(
                            model_obs_window,
                            device=external_runtime.frontend_device,
                        )
                        external_visual_outputs = _prepare_dual_expert_visual_outputs(
                            external_runtime.pipeline,
                            views=external_views,
                            task_text=(prompt,),
                            frontend_device=external_runtime.frontend_device,
                            runtime_device=external_runtime.runtime_device,
                            use_streaming_frontend=(
                                chunk_count == 0
                                or external_runtime.use_lingbot_streaming_vae
                            ),
                            preserve_stream_cache=False,
                            text_context=external_idm_session.text_context,
                            negative_text_context=(
                                external_idm_session.negative_text_context
                            ),
                        )
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_start",
                        **log_coordinates,
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
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
                    dual_expert_inference_window_size=args.dual_expert_inference_window_size,
                    dual_expert_rollout_frame_chunk_size=args.dual_expert_rollout_frame_chunk_size,
                    dual_expert_action_only_rollout=bool(
                        args.dual_expert_action_only_rollout
                    ),
                    output_request=(
                        external_idm.producer_plan.output_request
                        if uses_external_idm
                        else None
                    ),
                    video_generation=(
                        PolicyVideoGenerationRequest(
                            frame_count=(
                                int(args.dual_expert_rollout_frame_chunk_size)
                                if args.dual_expert_rollout_frame_chunk_size
                                is not None
                                else _frame_chunk_size(action_config)
                            )
                        )
                        if uses_external_idm
                        else None
                    ),
                )
                pre_infer_policy_state = None
                if not uses_external_idm and session.policy_state is not None:
                    pre_infer_policy_state = copy.deepcopy(session.policy_state)
                infer_session = runner.reset(
                    task_text=session.task_text,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                )
                infer_session.policy_state = (
                    None if args.reset_policy_state_each_chunk else session.policy_state
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
                    if uses_external_idm
                    else None
                )
                if generated_video is not None:
                    route_predicted_latents = generated_video.latents
                external_idm_inference_seed = None
                if action_route is DualExpertActionRoute.JOINT_VIDEO_THEN_IDM:
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
                        dual_expert_inference_window_size=args.dual_expert_inference_window_size,
                        dual_expert_rollout_frame_chunk_size=args.dual_expert_rollout_frame_chunk_size,
                        dual_expert_action_only_rollout=False,
                    )
                    idm_context.dynamics = DynamicsRolloutRequest(
                        objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                        clean_video=route_predicted_latents.detach().to(
                            device=resources.runtime_device,
                            dtype=route_predicted_latents.dtype,
                        ),
                    )
                    idm_session = runner.reset(
                        task_text=session.task_text,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                    idm_session.policy_state = (
                        None
                        if args.reset_policy_state_each_chunk
                        else pre_infer_policy_state
                    )
                    step_output = runner.infer_prepared_step(
                        session=idm_session,
                        context=idm_context,
                        visual_outputs=visual_outputs,
                    )
                    infer_output = step_output.infer_output
                elif uses_external_idm:
                    if (
                        generated_video is None
                    ):
                        raise RuntimeError(
                            "External IDM composition requires its producer stage "
                            "to publish a non-empty generated-video chunk."
                        )
                    if (
                        external_idm is None
                        or external_runtime is None
                        or external_idm_session is None
                        or external_visual_outputs is None
                    ):
                        raise RuntimeError(
                            "External IDM route was selected without complete runtime state."
                        )
                    external_step = infer_external_idm_action(
                        external_idm,
                        session=external_idm_session,
                        visual_outputs=external_visual_outputs,
                        model_obs_window=model_obs_window,
                        prompt=prompt,
                        generated_video=generated_video,
                        inference_window_size=(args.dual_expert_inference_window_size),
                        reset_policy_state=bool(args.reset_policy_state_each_chunk),
                        rollout_seed=seed,
                        chunk_index=chunk_count,
                    )
                    step_output = external_step.rollout
                    infer_output = step_output.infer_output
                    external_idm_inference_seed = external_step.inference_seed
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_done",
                        **log_coordinates,
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "dual_expert_gjd_action_route": str(action_route.value),
                    },
                )
            if uses_external_idm:
                session = primary_step_output.session
                external_idm_session = step_output.session
            else:
                session = step_output.session
            actions = (
                infer_output.decoder_output.action_pred[0]
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            configured_frame_chunk_size = _frame_chunk_size(action_config)
            configured_action_per_frame = _action_per_frame(action_config)
            action_per_frame = configured_action_per_frame
            if int(actions.shape[0]) % int(action_per_frame) != 0:
                raise ValueError(
                    "DualExpert action output length must be divisible by configured action_per_frame, "
                    f"got action_shape={actions.shape}, action_per_frame={action_per_frame}."
                )
            frame_chunk_size = int(actions.shape[0]) // int(action_per_frame)
            execute_action_steps = _resolve_execute_action_steps(
                args.execute_action_steps,
                execute_frame_chunk_size=args.execute_frame_chunk_size,
                action_horizon=int(actions.shape[0]),
                action_per_frame=action_per_frame,
            )
            frame_actions = actions.reshape(
                frame_chunk_size, action_per_frame, actions.shape[-1]
            )
            predicted_latents = extract_predicted_latents(infer_output)
            if action_route in {
                DualExpertActionRoute.JOINT_VIDEO_THEN_IDM,
                DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION,
            } and isinstance(route_predicted_latents, torch.Tensor):
                predicted_latents = route_predicted_latents
            if not args.skip_comparison_video and isinstance(
                predicted_latents, torch.Tensor
            ):
                append_predicted_latent_chunk(
                    predicted_latent_chunks,
                    predicted_latents,
                    max_imagined_latent_frames=args.max_imagined_latent_frames,
                )

            policy_debug = _summarize_policy_debug(infer_output.policy_output)
            chunk_log = {
                **log_coordinates,
                "chunk_index": chunk_count,
                "phase": "infer",
                "env_timestep_before": int(env.env.timestep),
                "window_size": len(frame_window),
                "model_obs_frames": len(model_obs_window),
                "video_latent_frames": int(
                    visual_outputs.frontend.video_latents.shape[2]
                ),
                "frontend_path": frontend_path,
                "action_shape": list(actions.shape),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(
                    execute_action_steps // action_per_frame
                ),
                "predicted_latents_shape": None
                if not isinstance(predicted_latents, torch.Tensor)
                else list(predicted_latents.shape),
                "dual_expert_gjd_action_route": action_route.value,
                "external_idm_inference_seed": external_idm_inference_seed,
                "first_action_preview": [float(v) for v in actions[0].tolist()],
                "policy_debug": policy_debug,
                "primary_policy_debug": (
                    _summarize_policy_debug(primary_infer_output.policy_output)
                    if uses_external_idm
                    else None
                ),
            }
            _print_log(chunk_log_label(chunk_count), chunk_log)
            chunk_logs.append(chunk_log)

            real_future_frames: list[dict[str, np.ndarray]] = []
            executed_actions = 0
            executed_control_actions: list[np.ndarray] = []
            executed_obs_frames: list[dict[str, np.ndarray]] = []
            raw_generation_frame_start = (
                infer_output.policy_output.generation_frame_start
            )
            if raw_generation_frame_start is None:
                raise RuntimeError(
                    "DualExpert rollout did not publish its typed generation "
                    "frame origin."
                )
            generation_frame_start = int(raw_generation_frame_start)
            start_frame_group = (
                1 if chunk_count == 0 and generation_frame_start <= 0 else 0
            )
            max_action_index = min(int(execute_action_steps), int(actions.shape[0]))
            for frame_group in range(start_frame_group, frame_actions.shape[0]):
                for action_offset, action in enumerate(frame_actions[frame_group]):
                    absolute_action_index = (
                        frame_group * action_per_frame + action_offset
                    )
                    if absolute_action_index >= max_action_index:
                        break
                    if _raw_env_done(env) or env.env.timestep >= args.max_timestep:
                        terminal = True
                        break
                    control_action = np.clip(
                        action.astype(np.float32, copy=False), -1.0, 1.0
                    )
                    executed_control_actions.append(np.array(control_action, copy=True))
                    action_trace.append(np.array(control_action, copy=True))
                    obs, _, step_success, _ = env.step(control_action)
                    done = bool(done or step_success)
                    executed_actions += 1
                    extracted = _extract_obs(obs)
                    extracted_record = {
                        key: np.array(value, copy=True)
                        for key, value in extracted.items()
                    }
                    rollout_frames.append(
                        {
                            key: np.array(value, copy=True)
                            for key, value in extracted_record.items()
                        }
                    )
                    frame_window.append(
                        {
                            key: np.array(value, copy=True)
                            for key, value in extracted_record.items()
                        }
                    )
                    executed_obs_frames.append(
                        {
                            key: np.array(value, copy=True)
                            for key, value in extracted_record.items()
                        }
                    )
                    real_future_frames.append(
                        {
                            key: np.array(value, copy=True)
                            for key, value in extracted_record.items()
                        }
                    )
                    terminal = bool(
                        done
                        or _raw_env_done(env)
                        or env.env.timestep >= args.max_timestep
                    )
                    if terminal:
                        break
                if terminal:
                    break
                next_frame_group_first_action = (frame_group + 1) * action_per_frame
                if next_frame_group_first_action >= max_action_index:
                    break
            terminal = bool(
                done or _raw_env_done(env) or env.env.timestep >= args.max_timestep
            )
            execution_commit = _build_execution_commit(
                generation_frame_start=generation_frame_start,
                speculative_frame_count=frame_chunk_size,
                executed_action_count=executed_actions,
                action_per_frame=action_per_frame,
                start_frame_group=start_frame_group,
                terminal=terminal,
            )

            chunk_result_log = {
                **log_coordinates,
                "chunk_index": chunk_count,
                "phase": "env_rollout",
                "env_timestep_after": int(env.env.timestep),
                "executed_actions": int(executed_actions),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(
                    execute_action_steps // action_per_frame
                ),
                "start_frame_group": int(start_frame_group),
                "done_after_chunk": bool(terminal),
                "success_after_chunk": bool(done),
            }
            _print_log(chunk_log_label(chunk_count), chunk_result_log)
            chunk_logs.append(chunk_result_log)

            warmup_action_history = _build_executed_action_history_tensor(
                executed_control_actions,
                start_frame_group=start_frame_group,
                action_per_frame=action_per_frame,
                action_dim=actions.shape[-1],
            )
            if (
                use_lingbot_streaming_vae
                and executed_obs_frames
                and not terminal
                and env.env.timestep < args.max_timestep
            ):
                streaming_views = _obs_list_to_views(
                    executed_obs_frames, device=resources.frontend_device
                )
                streaming_next_visual_outputs = _prepare_dual_expert_visual_outputs(
                    pipeline,
                    views=streaming_views,
                    task_text=(prompt,),
                    frontend_device=resources.frontend_device,
                    runtime_device=resources.runtime_device,
                    use_streaming_frontend=True,
                    preserve_stream_cache=True,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                )
                streaming_next_obs_window = [
                    {key: np.array(value, copy=True) for key, value in obs.items()}
                    for obs in executed_obs_frames
                ]
                streaming_update_log = {
                    **log_coordinates,
                    "chunk_index": chunk_count,
                    "phase": "lingbot_streaming_vae_update",
                    "real_obs_frames": len(streaming_next_obs_window),
                    "real_latent_frames": int(
                        streaming_next_visual_outputs.frontend.video_latents.shape[2]
                    ),
                }
                _print_log(chunk_log_label(chunk_count), streaming_update_log)
                chunk_logs.append(streaming_update_log)
                if external_runtime is not None:
                    if external_idm_session is None:
                        raise RuntimeError(
                            "External IDM session is missing during frontend update."
                        )
                    external_streaming_views = _obs_list_to_views(
                        executed_obs_frames,
                        device=external_runtime.frontend_device,
                    )
                    external_streaming_next_visual_outputs = (
                        _prepare_dual_expert_visual_outputs(
                            external_runtime.pipeline,
                            views=external_streaming_views,
                            task_text=(prompt,),
                            frontend_device=external_runtime.frontend_device,
                            runtime_device=external_runtime.runtime_device,
                            use_streaming_frontend=True,
                            preserve_stream_cache=True,
                            text_context=external_idm_session.text_context,
                            negative_text_context=(
                                external_idm_session.negative_text_context
                            ),
                        )
                    )

            primary_history_infer_output = (
                primary_infer_output if uses_external_idm else infer_output
            )
            if (
                (
                    executed_obs_frames
                    if use_lingbot_streaming_vae
                    else real_future_frames
                )
                and warmup_action_history is not None
                and not terminal
                and env.env.timestep < args.max_timestep
                and (
                    (
                        external_idm is not None
                        and external_idm.producer_plan.recurrent_history_policy
                        is PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
                    )
                    or (
                        external_idm is None
                        and "dual_expert_packed_history_debug"
                        in primary_history_infer_output.policy_output.aux
                    )
                )
            ):
                if use_lingbot_streaming_vae:
                    if (
                        streaming_next_visual_outputs is None
                        or streaming_next_obs_window is None
                    ):
                        raise RuntimeError(
                            "Streaming VAE packed warmup expected pre-encoded next observations."
                        )
                    warmup_outputs = streaming_next_visual_outputs
                    warmup_obs_window = streaming_next_obs_window
                else:
                    warmup_obs_window = real_future_frames
                    warmup_views = _obs_list_to_views(
                        warmup_obs_window,
                        device=resources.frontend_device,
                    )
                    warmup_outputs = _prepare_dual_expert_visual_outputs(
                        pipeline,
                        views=warmup_views,
                        task_text=(prompt,),
                        frontend_device=resources.frontend_device,
                        runtime_device=resources.runtime_device,
                        use_streaming_frontend=False,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                proprio_history = build_libero_state_history(
                    warmup_obs_window,
                    state_horizon=len(warmup_obs_window),
                    state_encoding=config.data.action_target.state_encoding,
                )
                history_output = runner.reconcile_observed_history(
                    session=session,
                    visual_outputs=warmup_outputs,
                    observation_frame_count=len(warmup_obs_window),
                    action_history=warmup_action_history,
                    proprio_history=proprio_history,
                    inference_window_size=args.dual_expert_inference_window_size,
                    rollout_frame_chunk_size=args.dual_expert_rollout_frame_chunk_size,
                    execution_commit=execution_commit,
                )
                if uses_external_idm and not history_output.applied:
                    raise RuntimeError(
                        "The video producer declared explicit observed-history "
                        "reconciliation but did not apply it."
                    )
                session = history_output.session
                warmup_debug = history_output.debug
                warmup_log = {
                    **log_coordinates,
                    "chunk_index": chunk_count,
                    "phase": "packed_history_warmup",
                    **warmup_debug,
                }
                _print_log(chunk_log_label(chunk_count), warmup_log)
                chunk_logs.append(warmup_log)

            if (
                external_runtime is not None
                and external_idm_session is not None
                and (
                    executed_obs_frames
                    if use_lingbot_streaming_vae
                    else real_future_frames
                )
                and warmup_action_history is not None
                and not terminal
                and env.env.timestep < args.max_timestep
            ):
                if use_lingbot_streaming_vae:
                    if (
                        external_streaming_next_visual_outputs is None
                        or streaming_next_obs_window is None
                    ):
                        raise RuntimeError(
                            "External IDM streaming history reconciliation requires "
                            "pre-encoded observations."
                        )
                    external_warmup_outputs = external_streaming_next_visual_outputs
                    external_warmup_obs = streaming_next_obs_window
                else:
                    external_warmup_obs = real_future_frames
                    external_warmup_views = _obs_list_to_views(
                        external_warmup_obs,
                        device=external_runtime.frontend_device,
                    )
                    external_warmup_outputs = _prepare_dual_expert_visual_outputs(
                        external_runtime.pipeline,
                        views=external_warmup_views,
                        task_text=(prompt,),
                        frontend_device=external_runtime.frontend_device,
                        runtime_device=external_runtime.runtime_device,
                        use_streaming_frontend=False,
                        preserve_stream_cache=False,
                        text_context=external_idm_session.text_context,
                        negative_text_context=(
                            external_idm_session.negative_text_context
                        ),
                    )
                external_proprio_history = build_libero_state_history(
                    external_warmup_obs,
                    state_horizon=len(external_warmup_obs),
                    state_encoding=(
                        external_runtime.config.data.action_target.state_encoding
                    ),
                )
                external_history = external_runtime.runner.reconcile_observed_history(
                    session=external_idm_session,
                    visual_outputs=external_warmup_outputs,
                    observation_frame_count=len(external_warmup_obs),
                    action_history=warmup_action_history,
                    proprio_history=external_proprio_history,
                    inference_window_size=args.dual_expert_inference_window_size,
                    rollout_frame_chunk_size=(
                        args.dual_expert_rollout_frame_chunk_size
                    ),
                    execution_commit=execution_commit,
                )
                if not external_history.applied:
                    raise RuntimeError(
                        "The video-conditioned action consumer declared explicit "
                        "observed-history reconciliation but did not apply it."
                    )
                external_idm_session = external_history.session
                external_warmup_log = {
                    **log_coordinates,
                    "chunk_index": chunk_count,
                    "phase": "external_idm_packed_history_warmup",
                    **external_history.debug,
                }
                _print_log(chunk_log_label(chunk_count), external_warmup_log)
                chunk_logs.append(external_warmup_log)

            chunk_count += 1

        policy_config = config.policy_variant
        policy_program = getattr(policy_config, "program", None)
        policy_condition_mode = getattr(policy_config, "condition_mode", None)
        consumer_checkpoint_file = (
            None
            if external_runtime is None
            else str(external_runtime.checkpoint_path.resolve())
        )
        summary = {
            "benchmark": args.benchmark,
            "task_id": task_id,
            "prompt": prompt,
            "episode_idx": episode_idx,
            "success": bool(done),
            "terminal": bool(terminal),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": seed,
            "video_path": None,
            "comparison_video_path": None,
            "rollout_video_path": None,
            "pipeline": (
                "open_wam_policy_video_action_composition"
                if uses_external_idm
                else "open_wam_dual_expert"
            ),
            "program": (
                None
                if policy_program is None
                else getattr(policy_program, "value", str(policy_program))
            ),
            "condition_mode": (
                None
                if policy_condition_mode is None
                else str(policy_condition_mode)
            ),
            "startup_model_obs_frames": int(resources.startup_model_obs_frames),
            "startup_env_init_steps": int(resources.startup_env_init_steps),
            "startup_env_steps_executed": int(
                max(
                    resources.startup_env_init_steps, resources.startup_model_obs_frames
                )
            ),
            "execute_action_steps": None
            if args.execute_action_steps is None
            else int(args.execute_action_steps),
            "execute_frame_chunk_size": (
                None
                if args.execute_frame_chunk_size is None
                else int(args.execute_frame_chunk_size)
            ),
            "action_count": len(action_trace),
            "checkpoint_file": str(resources.checkpoint_path.resolve()),
            "action_consumer_checkpoint_file": consumer_checkpoint_file,
            "external_idm_checkpoint_file": consumer_checkpoint_file,
            "action_route": action_route.value,
            "dual_expert_gjd_action_route": action_route.value,
        }
        artifact_output = persist_libero_rollout_artifacts(
            pipeline=pipeline,
            identity=LiberoRolloutArtifactIdentity(
                benchmark=args.benchmark,
                task_id=task_id,
                prompt=prompt,
                episode_idx=episode_idx,
                success=bool(done),
                suffix=args.suffix,
            ),
            options=LiberoRolloutArtifactOptions(
                output_root=Path(args.output_dir),
                video_fps=args.video_fps,
                save_rollout_video=args.save_rollout_video,
                skip_comparison_video=args.skip_comparison_video,
            ),
            payload=LiberoRolloutArtifactPayload(
                real_observations=rollout_frames,
                predicted_latent_chunks=predicted_latent_chunks,
                action_trace=action_trace,
                chunk_events=chunk_logs,
                component_report=build_composed_component_report(
                    resources,
                    external_idm,
                ),
            ),
            summary=summary,
            decode_device=resources.decode_device,
        )
        summary = artifact_output.summary
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if close_env_after_rollout:
            env.close()


def _build_execution_commit(
    *,
    generation_frame_start: int,
    speculative_frame_count: int,
    executed_action_count: int,
    action_per_frame: int,
    start_frame_group: int,
    terminal: bool,
) -> PolicyExecutionCommit | None:
    """Describe the model-frame interval actually committed by execution."""

    if int(executed_action_count) <= 0 or terminal or int(start_frame_group) != 0:
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


def _resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    task_spec = resolve_libero_task_by_id(
        benchmark_name,
        task_id,
        project_root=REPO_ROOT,
    )
    return task_spec, task_spec.task_language


def _construct_single_env(task_spec: LiberoTaskSpec):
    count = 0
    env = None
    while env is None and count < 5:
        try:
            env = build_libero_offscreen_env(
                task_spec,
                camera_height=128,
                camera_width=128,
                horizon=1000,
                ignore_done=False,
                project_root=REPO_ROOT,
            )
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _summarize_policy_debug(policy_output: PolicyInferOutput) -> dict[str, object]:
    """Keep rollout logs readable by replacing large tensors with metadata."""

    summary: dict[str, object] = {}
    for key, value in policy_output.aux.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            continue
        if isinstance(value, dict):
            summary[key] = value
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            summary[key] = value
            continue
        summary[key] = type(value).__name__

    envelope = policy_output.decoder_artifacts
    if envelope is not None and envelope.contract == DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT:
        artifacts = envelope.require(
            contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
            payload_type=DualExpertInferArtifacts,
        )
        summary["dual_expert_infer_artifacts"] = {
            "action_pred_shape": list(artifacts.action_pred.shape),
            "predicted_latents_shape": (
                list(artifacts.predicted_latents.shape)
                if isinstance(artifacts.predicted_latents, torch.Tensor)
                else None
            ),
            "condition_mode": str(artifacts.condition_mode),
            "program": str(artifacts.program),
        }
    elif envelope is not None:
        summary["decoder_artifact_contract"] = envelope.contract
        summary["decoder_artifact_payload_type"] = type(envelope.payload).__name__
    if policy_output.generated_video is not None:
        summary["generated_video"] = {
            "shape": list(policy_output.generated_video.latents.shape),
            "frame_start": policy_output.generated_video.frame_start,
        }
    return summary
