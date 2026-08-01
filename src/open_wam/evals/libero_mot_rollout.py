"""Checkpoint-backed MoT/GJD execution for LIBERO benchmark adapters.

This module consumes a loaded runtime and owns the exact per-episode policy
loop. Checkpoint/config/device composition lives in ``libero_mot_runtime``;
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

from open_wam.evals.libero_mot_runtime import (
    CURRENT_FRONTEND_ENCODE_MODE,
    DEPRECATED_FRONTEND_ENCODE_MODE,
    LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES,
    MOT_GJD_ACTION_ROUTES,
    OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES,
    MotLiberoLoadOptions,
    MotLiberoRuntime,
    _action_per_frame,
    _frame_chunk_size,
    _maybe_merge_checkpoint_runtime_config,
    _require_current_frontend_encode_mode,
    _validate_live_sim_mot_generalist_rollout_mode,
    load_mot_libero_runtime,
    print_rollout_event,
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
    resolve_execute_action_steps as _resolve_shared_execute_action_steps,
)
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.utils import seed_everywhere

REPO_ROOT = Path(__file__).resolve().parents[3]

LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS
_build_executed_action_history_tensor = (
    _build_shared_executed_action_history_tensor
)
_extract_obs = extract_libero_rollout_observation
_init_single_env = initialize_libero_observation_window
_obs_list_to_views = libero_observation_window_to_views
_resolve_execute_action_steps = _resolve_shared_execute_action_steps
_print_log = print_rollout_event
_MOT_RUNTIME_COMPATIBILITY_EXPORTS = (
    LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES,
    OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES,
    _maybe_merge_checkpoint_runtime_config,
    _require_current_frontend_encode_mode,
)

__all__ = [
    "CURRENT_FRONTEND_ENCODE_MODE",
    "DEPRECATED_FRONTEND_ENCODE_MODE",
    "MOT_GJD_ACTION_ROUTES",
    "MotLiberoEpisodeOptions",
    "MotLiberoLoadOptions",
    "MotLiberoRuntime",
    "MotLiberoTaskResources",
    "construct_mot_libero_env",
    "load_mot_libero_runtime",
    "print_rollout_event",
    "resolve_mot_libero_task_resources",
    "run_mot_libero_episode",
]


@dataclass(frozen=True)
class MotLiberoTaskResources:
    """Resolved simulator task metadata and its fixed initialization states."""

    task_spec: LiberoTaskSpec
    prompt: str
    init_states: Any


@dataclass(frozen=True)
class MotLiberoEpisodeOptions:
    """Policy, simulator, and artifact choices for one loaded-model episode."""

    benchmark: str
    task_id: int
    episode_idx: int
    max_timestep: int
    max_chunks: int | None
    execute_action_steps: int | None
    execute_frame_chunk_size: int | None
    mot_rollout_frame_chunk_size: int | None
    mot_inference_window_size: int | None
    mot_action_only_rollout: bool
    mot_generalist_rollout_mode: str | None
    mot_gjd_action_route: str
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


def resolve_mot_libero_task_resources(
    benchmark_name: str,
    task_id: int,
) -> MotLiberoTaskResources:
    """Resolve task metadata and initialization states for one benchmark task."""

    task_spec, prompt = _resolve_task_spec(benchmark_name, task_id)
    return MotLiberoTaskResources(
        task_spec=task_spec,
        prompt=prompt,
        init_states=load_libero_task_init_states(task_spec),
    )


def construct_mot_libero_env(task_spec: LiberoTaskSpec) -> Any:
    """Construct the exact 128px offscreen environment with bounded retries."""

    return _construct_single_env(task_spec)


def run_mot_libero_episode(
    args: MotLiberoEpisodeOptions,
    resources: MotLiberoRuntime,
    task_resources: MotLiberoTaskResources,
    env: Any,
    *,
    include_episode_coordinates: bool,
    close_env_after_rollout: bool,
) -> dict[str, object]:
    """Execute one exact M5/GJD LIBERO episode with already-loaded resources."""

    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")
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

    try:
        _print_log("stage", {"name": "init_env_rollout_start", "episode_idx": int(episode_idx)})
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
                "startup_env_steps_executed": int(max(resources.startup_env_init_steps, resources.startup_model_obs_frames)),
            },
        )
        frame_window: deque[dict[str, np.ndarray]] = deque(maxlen=resources.raw_window_frames)
        for obs in initial_obs_window:
            frame_window.append({key: np.array(value, copy=True) for key, value in obs.items()})

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
        streaming_next_visual_outputs = None
        streaming_next_obs_window: list[dict[str, np.ndarray]] | None = None
        if use_lingbot_streaming_vae:
            pipeline.visual_tower.reset_runtime_state()

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
                    if streaming_next_visual_outputs is None or streaming_next_obs_window is None:
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
                    views = _obs_list_to_views(model_obs_window, device=resources.frontend_device)
                    visual_outputs = _prepare_mot_visual_outputs(
                        pipeline,
                        views=views,
                        task_text=(prompt,),
                        frontend_device=resources.frontend_device,
                        runtime_device=resources.runtime_device,
                        use_streaming_frontend=chunk_count == 0 or use_lingbot_streaming_vae,
                        preserve_stream_cache=False,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                    frontend_path = "lingbot_streaming_vae_init" if use_lingbot_streaming_vae else (
                        "streaming" if chunk_count == 0 else "offline"
                    )
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_start",
                        **log_coordinates,
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "model_obs_frames": int(len(model_obs_window)),
                        "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
                        "frontend_path": frontend_path,
                    },
                )
                infer_context = _build_infer_context(
                    prompt,
                    action_device=resources.action_device,
                    model_obs_window=model_obs_window,
                    config=config,
                    runtime_device=resources.runtime_device,
                    mot_inference_window_size=args.mot_inference_window_size,
                    mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    mot_action_only_rollout=bool(args.mot_action_only_rollout),
                    mot_generalist_rollout_mode=args.mot_generalist_rollout_mode,
                )
                pre_infer_policy_state = (
                    None
                    if session.policy_state is None
                    else copy.deepcopy(session.policy_state)
                )
                infer_session = runner.reset(
                    task_text=session.task_text,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                )
                infer_session.policy_state = (
                    None
                    if args.reset_policy_state_each_chunk
                    else session.policy_state
                )
                step_output = runner.infer_prepared_step(
                    session=infer_session,
                    context=infer_context,
                    visual_outputs=visual_outputs,
                )
                infer_output = step_output.infer_output
                route_predicted_latents = extract_predicted_latents(infer_output)
                if args.mot_gjd_action_route == "joint_video_then_idm":
                    if args.mot_generalist_rollout_mode not in (
                        None,
                        "joint",
                        "vanilla_joint_rollout",
                    ):
                        raise ValueError(
                            "`joint_video_then_idm` must start from joint GJD rollout; "
                            f"got mot_generalist_rollout_mode={args.mot_generalist_rollout_mode!r}."
                        )
                    if not isinstance(route_predicted_latents, torch.Tensor) or int(route_predicted_latents.shape[2]) <= 0:
                        raise RuntimeError(
                            "joint_video_then_idm route requires joint rollout to produce a non-empty predicted video chunk."
                        )
                    idm_context = _build_infer_context(
                        prompt,
                        action_device=resources.action_device,
                        model_obs_window=model_obs_window,
                        config=config,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                        mot_action_only_rollout=False,
                        mot_generalist_rollout_mode=None,
                    )
                    idm_context.extra["action_conditioning_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_generalist_rollout_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_video_condition_latents"] = route_predicted_latents.detach().to(
                        device=resources.runtime_device,
                        dtype=route_predicted_latents.dtype,
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
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_done",
                        **log_coordinates,
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "mot_gjd_action_route": str(args.mot_gjd_action_route),
                    },
                )
            session = step_output.session
            actions = infer_output.decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu().numpy()
            configured_frame_chunk_size = _frame_chunk_size(config)
            configured_action_per_frame = _action_per_frame(config)
            action_per_frame = configured_action_per_frame
            if int(actions.shape[0]) % int(action_per_frame) != 0:
                raise ValueError(
                    "MoT action output length must be divisible by configured action_per_frame, "
                    f"got action_shape={actions.shape}, action_per_frame={action_per_frame}."
                )
            frame_chunk_size = int(actions.shape[0]) // int(action_per_frame)
            execute_action_steps = _resolve_execute_action_steps(
                args.execute_action_steps,
                execute_frame_chunk_size=args.execute_frame_chunk_size,
                action_horizon=int(actions.shape[0]),
                action_per_frame=action_per_frame,
            )
            frame_actions = actions.reshape(frame_chunk_size, action_per_frame, actions.shape[-1])
            predicted_latents = extract_predicted_latents(infer_output)
            if args.mot_gjd_action_route == "joint_video_then_idm" and isinstance(route_predicted_latents, torch.Tensor):
                predicted_latents = route_predicted_latents
            if not args.skip_comparison_video and isinstance(predicted_latents, torch.Tensor):
                append_predicted_latent_chunk(
                    predicted_latent_chunks,
                    predicted_latents,
                    max_imagined_latent_frames=args.max_imagined_latent_frames,
                )

            chunk_log = {
                **log_coordinates,
                "chunk_index": chunk_count,
                "phase": "infer",
                "env_timestep_before": int(env.env.timestep),
                "window_size": len(frame_window),
                "model_obs_frames": len(model_obs_window),
                "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
                "frontend_path": frontend_path,
                "action_shape": list(actions.shape),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(execute_action_steps // action_per_frame),
                "predicted_latents_shape": None if not isinstance(predicted_latents, torch.Tensor) else list(predicted_latents.shape),
                "mot_gjd_action_route": str(args.mot_gjd_action_route),
                "first_action_preview": [float(v) for v in actions[0].tolist()],
                "policy_debug": _summarize_policy_debug(infer_output.policy_output.aux),
            }
            _print_log(chunk_log_label(chunk_count), chunk_log)
            chunk_logs.append(chunk_log)

            real_future_frames: list[dict[str, np.ndarray]] = []
            executed_actions = 0
            executed_control_actions: list[np.ndarray] = []
            executed_obs_frames: list[dict[str, np.ndarray]] = []
            policy_debug = _summarize_policy_debug(infer_output.policy_output.aux)
            generation_frame_start = int(
                policy_debug.get("generation_frame_start", 0)
                if "generation_frame_start" in policy_debug
                else policy_debug.get("mot_cache_debug", {}).get("current_action_frame_start", 0)
            )
            start_frame_group = 1 if chunk_count == 0 and generation_frame_start <= 0 else 0
            max_action_index = min(int(execute_action_steps), int(actions.shape[0]))
            for frame_group in range(start_frame_group, frame_actions.shape[0]):
                for action_offset, action in enumerate(frame_actions[frame_group]):
                    absolute_action_index = frame_group * action_per_frame + action_offset
                    if absolute_action_index >= max_action_index:
                        break
                    if _raw_env_done(env) or env.env.timestep >= args.max_timestep:
                        terminal = True
                        break
                    control_action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
                    executed_control_actions.append(np.array(control_action, copy=True))
                    action_trace.append(np.array(control_action, copy=True))
                    obs, _, step_success, _ = env.step(control_action)
                    done = bool(done or step_success)
                    executed_actions += 1
                    extracted = _extract_obs(obs)
                    extracted_record = {key: np.array(value, copy=True) for key, value in extracted.items()}
                    rollout_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    frame_window.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    executed_obs_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    real_future_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    terminal = bool(done or _raw_env_done(env) or env.env.timestep >= args.max_timestep)
                    if terminal:
                        break
                if terminal:
                    break
                next_frame_group_first_action = (frame_group + 1) * action_per_frame
                if next_frame_group_first_action >= max_action_index:
                    break
            terminal = bool(done or _raw_env_done(env) or env.env.timestep >= args.max_timestep)

            chunk_result_log = {
                **log_coordinates,
                "chunk_index": chunk_count,
                "phase": "env_rollout",
                "env_timestep_after": int(env.env.timestep),
                "executed_actions": int(executed_actions),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(execute_action_steps // action_per_frame),
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
                streaming_views = _obs_list_to_views(executed_obs_frames, device=resources.frontend_device)
                streaming_next_visual_outputs = _prepare_mot_visual_outputs(
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
                    "real_obs_frames": int(len(streaming_next_obs_window)),
                    "real_latent_frames": int(streaming_next_visual_outputs.frontend.video_latents.shape[2]),
                }
                _print_log(chunk_log_label(chunk_count), streaming_update_log)
                chunk_logs.append(streaming_update_log)

            if (
                (executed_obs_frames if use_lingbot_streaming_vae else real_future_frames)
                and warmup_action_history is not None
                and not terminal
                and env.env.timestep < args.max_timestep
                and "mot_packed_history_debug" in infer_output.policy_output.aux
            ):
                if use_lingbot_streaming_vae:
                    if streaming_next_visual_outputs is None or streaming_next_obs_window is None:
                        raise RuntimeError("Streaming VAE packed warmup expected pre-encoded next observations.")
                    warmup_outputs = streaming_next_visual_outputs
                    warmup_obs_window = streaming_next_obs_window
                else:
                    warmup_obs_window = real_future_frames
                    warmup_views = _obs_list_to_views(
                        warmup_obs_window,
                        device=resources.frontend_device,
                    )
                    warmup_outputs = _prepare_mot_visual_outputs(
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
                    inference_window_size=args.mot_inference_window_size,
                    rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
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

            chunk_count += 1

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
            "pipeline": "open_wam_mot",
            "runtime_mode": str(config.policy_variant.runtime_mode),
            "condition_mode": str(config.policy_variant.condition_mode),
            "startup_model_obs_frames": int(resources.startup_model_obs_frames),
            "startup_env_init_steps": int(resources.startup_env_init_steps),
            "startup_env_steps_executed": int(max(resources.startup_env_init_steps, resources.startup_model_obs_frames)),
            "execute_action_steps": None if args.execute_action_steps is None else int(args.execute_action_steps),
            "execute_frame_chunk_size": (
                None if args.execute_frame_chunk_size is None else int(args.execute_frame_chunk_size)
            ),
            "action_count": len(action_trace),
            "checkpoint_file": str(resources.checkpoint_path.resolve()),
            "mot_gjd_action_route": str(args.mot_gjd_action_route),
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
                component_report=resources.component_report,
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




def _build_infer_context(
    prompt: str,
    *,
    action_device: torch.device,
    model_obs_window: list[dict[str, np.ndarray]],
    config,
    runtime_device: torch.device,
    mot_inference_window_size: int | None,
    mot_action_only_rollout: bool,
    mot_generalist_rollout_mode: str | None,
    mot_rollout_frame_chunk_size: int | None = None,
):
    extra: dict[str, object] = {"task_text": (prompt,), "action_device": str(action_device)}
    if mot_inference_window_size is not None:
        extra["mot_inference_window_size"] = int(mot_inference_window_size)
    if mot_rollout_frame_chunk_size is not None:
        extra["mot_rollout_frame_chunk_size"] = int(mot_rollout_frame_chunk_size)
    if mot_action_only_rollout:
        extra["mot_action_only_rollout"] = True
    if mot_generalist_rollout_mode is not None:
        _validate_live_sim_mot_generalist_rollout_mode(mot_generalist_rollout_mode)
        extra["action_conditioning_mode"] = str(mot_generalist_rollout_mode)
        extra["mot_generalist_rollout_mode"] = str(mot_generalist_rollout_mode)
    return PolicyInferContext(
        state=build_libero_state_history(
            model_obs_window,
            state_horizon=int(config.data.action_schema.state_horizon),
            state_encoding=config.data.action_target.state_encoding,
        )
        .unsqueeze(0)
        .to(device=runtime_device),
        extra=extra,
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


def _select_model_obs_window(
    frame_window: list[dict[str, np.ndarray]],
    *,
    chunk_index: int,
    startup_model_obs_frames: int,
) -> list[dict[str, np.ndarray]]:
    if not frame_window:
        raise ValueError("MoT visualization requires at least one observation frame.")
    if chunk_index == 0:
        if startup_model_obs_frames <= 0:
            raise ValueError(
                f"Expected positive startup_model_obs_frames, got {startup_model_obs_frames}."
            )
        if startup_model_obs_frames > len(frame_window):
            raise ValueError(
                "startup_model_obs_frames cannot exceed the available startup window, "
                f"got startup_model_obs_frames={startup_model_obs_frames}, window={len(frame_window)}."
            )
        return list(frame_window[-startup_model_obs_frames:])
    return list(frame_window)


def _prepare_mot_visual_outputs(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    use_streaming_frontend: bool,
    preserve_stream_cache: bool = False,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    if use_streaming_frontend:
        canonical_batch = pipeline.canonicalize(views)
        canonical_video = canonical_batch.video.to(device=frontend_device)
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype
        return pipeline.prepare_visual_outputs_from_latents(
            frontend_output.video_latents.to(device=runtime_device, dtype=runtime_dtype),
            task_text=task_text,
            text_context=(
                None
                if frontend_output.conditioning.text_context is None
                else frontend_output.conditioning.text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            negative_text_context=(
                None
                if frontend_output.conditioning.negative_text_context is None
                else frontend_output.conditioning.negative_text_context.to(
                    device=runtime_device,
                    dtype=runtime_dtype,
                )
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )
    return _prepare_visual_outputs_offline(
        pipeline,
        views=views,
        task_text=task_text,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _prepare_visual_outputs_offline(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    canonical_batch = pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend = pipeline.visual_tower.frontend
    assets = frontend.reference_assets
    runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype

    if assets.has_vae:
        video_latents = _encode_video_window_offline(
            assets,
            canonical_video=canonical_video,
            placements=canonical_batch.placements,
            device=frontend_device,
        ).to(device=runtime_device, dtype=runtime_dtype)
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = assets.encode_text(
                task_text,
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if resolved_negative_text_context is None and resolved_text_context is not None:
            resolved_negative_text_context = assets.encode_blank_text(
                batch_size=canonical_video.shape[0],
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        return pipeline.prepare_visual_outputs_from_latents(
            video_latents,
            task_text=task_text,
            text_context=(
                None
                if resolved_text_context is None
                else resolved_text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            negative_text_context=(
                None
                if resolved_negative_text_context is None
                else resolved_negative_text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )

    return pipeline.prepare_visual_outputs(
        views,
        task_text=task_text,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _encode_video_window_offline(
    assets,
    *,
    canonical_video: torch.Tensor,
    placements,
    device: torch.device,
) -> torch.Tensor:
    if not assets.has_vae:
        raise RuntimeError("Wan VAE assets are not loaded for offline video encoding.")
    del device
    return assets.encode_video(canonical_video, placements=placements, reset_cache=True)



def _summarize_policy_debug(aux: dict[str, object]) -> dict[str, object]:
    """Keep rollout logs readable by replacing large tensors with metadata."""

    summary: dict[str, object] = {}
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            continue
        if key == "mot_infer_artifacts" and hasattr(value, "action_pred"):
            action_pred = getattr(value, "action_pred", None)
            predicted_latents = getattr(value, "predicted_latents", None)
            summary[key] = {
                "action_pred_shape": (
                    list(action_pred.shape) if isinstance(action_pred, torch.Tensor) else None
                ),
                "predicted_latents_shape": (
                    list(predicted_latents.shape) if isinstance(predicted_latents, torch.Tensor) else None
                ),
                "condition_mode": str(getattr(value, "condition_mode", "")),
                "runtime_mode": str(getattr(value, "runtime_mode", "")),
            }
            continue
        if isinstance(value, dict):
            summary[key] = value
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            summary[key] = value
            continue
        summary[key] = type(value).__name__
    return summary
