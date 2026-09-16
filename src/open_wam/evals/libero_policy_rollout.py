"""Checkpoint-backed Policy execution for LIBERO benchmark adapters.

This module assembles the shared RolloutEngine, a LIBERO driver, and an artifact
sink. Model transactions live in libero_policy_planner. Checkpoint/config/device composition lives in ``libero_policy_runtime``;
LIBERO observation and environment mechanics live in ``open_wam.integrations``.
"""

from __future__ import annotations


import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from open_wam.configs import LiberoRendererProfile
from open_wam.evals.libero_policy_composition import (
    VideoActionComposition,
    build_composed_component_report,
)
from open_wam.evals.libero_episode_artifacts import LiberoEpisodeArtifacts
from open_wam.evals.libero_policy_planner import LiberoPolicyPlanner
from open_wam.runtime.control import ControlCommand, ControlTransition, RolloutTerminationReason
from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions
from open_wam.evals.libero_policy_runtime import (
    CURRENT_FRONTEND_ENCODE_MODE,
    DEPRECATED_FRONTEND_ENCODE_MODE,
    POLICY_ACTION_ROUTES,
    GJD_ACTION_ROUTES,
    PolicyActionRoute,
    LiberoPolicyLoadOptions,
    LiberoPolicyRuntime,
    load_libero_policy_runtime,
    print_rollout_event,
    uses_video_action_composition,
)
from open_wam.evals.libero_rollout_artifacts import (
    LiberoRolloutArtifactIdentity,
    LiberoRolloutArtifactOptions,
    LiberoRolloutArtifactPayload,
    persist_libero_rollout_artifacts,
)
from open_wam.integrations import (
    LIBERO_ROLLOUT_VIEW_KEYS,
    LiberoTaskSpec,
    activate_libero_renderer,
    build_libero_offscreen_env,
    extract_libero_rollout_observation,
    initialize_libero_observation_window,
    load_libero_task_init_states,
    resolve_libero_task_by_id,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS
_extract_obs = extract_libero_rollout_observation
_init_single_env = initialize_libero_observation_window
_print_log = print_rollout_event
__all__ = [
    "CURRENT_FRONTEND_ENCODE_MODE",
    "DEPRECATED_FRONTEND_ENCODE_MODE",
    "POLICY_ACTION_ROUTES",
    "GJD_ACTION_ROUTES",
    "LiberoPolicyEpisodeOptions",
    "LiberoPolicyLoadOptions",
    "LiberoPolicyRuntime",
    "LiberoPolicyTaskResources",
    "construct_libero_policy_env",
    "load_libero_policy_runtime",
    "print_rollout_event",
    "resolve_libero_policy_task_resources",
    "run_libero_policy_episode",
]


@dataclass(frozen=True)
class LiberoPolicyTaskResources:
    """Resolved simulator task metadata and its fixed initialization states."""

    task_spec: LiberoTaskSpec
    prompt: str
    init_states: Any


@dataclass(frozen=True)
class LiberoPolicyEpisodeOptions:
    """Policy, simulator, and artifact choices for one loaded-model episode."""

    benchmark: str
    task_id: int
    episode_idx: int
    max_timestep: int
    max_chunks: int | None
    execute_action_steps: int | None
    execute_frame_chunk_size: int | None
    rollout_frame_chunk_size: int | None
    inference_window_size: int | None
    action_only_rollout: bool
    policy_action_route: str
    reset_policy_state_each_chunk: bool
    max_imagined_latent_frames: int | None
    output_dir: str | Path
    suffix: str
    video_fps: float
    seed: int | None
    save_rollout_video: bool = False
    skip_comparison_video: bool = False
    renderer_profile: LiberoRendererProfile = LiberoRendererProfile.ONLINE_ROLLOUT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "renderer_profile",
            LiberoRendererProfile(self.renderer_profile),
        )


def _raw_env_done(env: object) -> bool:
    """Return robosuite's terminal flag behind LIBERO's success-only wrapper."""
    raw_env = getattr(env, "env", env)
    return bool(getattr(raw_env, "done", False))


def resolve_libero_policy_task_resources(
    benchmark_name: str,
    task_id: int,
    *,
    renderer_profile: LiberoRendererProfile | str = (
        LiberoRendererProfile.ONLINE_ROLLOUT
    ),
) -> LiberoPolicyTaskResources:
    """Resolve task metadata and initialization states for one benchmark task."""

    activate_libero_renderer(renderer_profile)
    task_spec, prompt = _resolve_task_spec(benchmark_name, task_id)
    return LiberoPolicyTaskResources(
        task_spec=task_spec,
        prompt=prompt,
        init_states=load_libero_task_init_states(task_spec),
    )


def construct_libero_policy_env(
    task_spec: LiberoTaskSpec,
    *,
    renderer_profile: LiberoRendererProfile | str = (
        LiberoRendererProfile.ONLINE_ROLLOUT
    ),
) -> Any:
    """Construct the exact 128px offscreen environment with bounded retries."""

    return _construct_single_env(task_spec, renderer_profile=renderer_profile)


class LiberoPolicyDriver:
    """LIBERO control conversion and simulator I/O, with no model lifecycle."""

    def __init__(self, env, artifacts: LiberoEpisodeArtifacts, *, max_timestep: int):
        self.env, self.artifacts, self.max_timestep = env, artifacts, max_timestep

    def materialize(self, step, observation) -> ControlCommand:
        action = np.clip(step.raw_action.astype(np.float32, copy=False), -1.0, 1.0)
        return ControlCommand(action=action, source_action=action)

    def fallback(self, last_action, observation) -> ControlCommand:
        raise RuntimeError("Blocking LIBERO requires a ready policy plan.")

    def termination_check(self) -> RolloutTerminationReason | None:
        if _raw_env_done(self.env):
            return RolloutTerminationReason.ENV_TERMINAL
        if self.env.env.timestep >= self.max_timestep:
            return RolloutTerminationReason.MAX_ACTIONS
        return None

    def step(self, action: np.ndarray) -> ControlTransition[dict[str, np.ndarray]]:
        executed_action = action.copy()
        observation, reward, success, info = self.env.step(action)
        observation = self.artifacts.copy_observation(_extract_obs(observation))
        done = bool(success or _raw_env_done(self.env))
        # Budget exhaustion closes artifacts, but the engine owns its stop reason.
        terminal = done or self.env.env.timestep >= self.max_timestep
        self.artifacts.executed(
            executed_action, observation, timestep=self.env.env.timestep, terminal=terminal, success=success,
        )
        return ControlTransition(observation, reward=reward, done=done, success=bool(success), info=info)


def run_libero_policy_episode(
    args: LiberoPolicyEpisodeOptions,
    resources: LiberoPolicyRuntime,
    task_resources: LiberoPolicyTaskResources,
    env: Any,
    *,
    include_episode_coordinates: bool,
    close_env_after_rollout: bool,
    video_action_composition: VideoActionComposition | None = None,
) -> dict[str, object]:
    """Execute one exact policy/GJD LIBERO episode with loaded resources."""

    if env is None:
        raise RuntimeError(
            "Failed to construct LIBERO OffScreenRenderEnv after 5 retries."
        )
    renderer_config = activate_libero_renderer(args.renderer_profile)
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

    config = resources.config
    pipeline = resources.pipeline
    use_lingbot_streaming_vae = bool(resources.use_lingbot_streaming_vae)
    action_route = PolicyActionRoute(args.policy_action_route)
    uses_composition = uses_video_action_composition(action_route)
    if uses_composition != (video_action_composition is not None):
        raise ValueError(
            "The generated-video action route and loaded composition "
            "must be supplied together."
        )
    consumer_runtime = (
        None if video_action_composition is None else video_action_composition.runtime
    )
    if consumer_runtime is not None and (
        bool(consumer_runtime.use_lingbot_streaming_vae) != use_lingbot_streaming_vae
    ):
        raise ValueError(
            "Video producer and action consumer must use the same frontend encode mode."
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
        initial_obs_window = initial_obs_window[-resources.raw_window_frames:]
        artifacts = LiberoEpisodeArtifacts(
            initial_obs_window, coordinates=log_coordinates,
            skip_comparison_video=args.skip_comparison_video,
            max_imagined_latent_frames=args.max_imagined_latent_frames,
        )
        driver = LiberoPolicyDriver(env, artifacts, max_timestep=args.max_timestep)
        planner = LiberoPolicyPlanner(
            resources, args, artifacts, prompt=prompt,
            initial_timestep=int(env.env.timestep), composition=video_action_composition,
        )
        initial_session = planner.start(initial_obs_window)
        result = RolloutEngine(
            planner, driver, RolloutOptions(
                max_actions=max(0, args.max_timestep - int(env.env.timestep)),
                max_plans=args.max_chunks, target_action_hz=None,
            ),
        ).run(
            initial_obs_window[-1], initial_session, step=driver.step,
            termination_check=driver.termination_check,
        )
        done = result.success
        terminal = result.session.chunk_index > 0 and bool(
            done or _raw_env_done(env) or env.env.timestep >= args.max_timestep
        )
        artifacts.finish(timestep=env.env.timestep, terminal=terminal, success=done)
        chunk_count = result.session.chunk_index
        rollout_frames, predicted_latent_chunks = artifacts.observations, artifacts.predictions
        action_trace, chunk_logs = artifacts.actions, artifacts.events

        policy_config = config.policy_variant
        policy_program = getattr(policy_config, "program", None)
        policy_condition_mode = getattr(policy_config, "condition_mode", None)
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
                if uses_composition
                else "open_wam_policy"
            ),
            "program": (
                None
                if policy_program is None
                else getattr(policy_program, "value", str(policy_program))
            ),
            "condition_mode": (
                None if policy_condition_mode is None else str(policy_condition_mode)
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
            "action_route": action_route.value,
            "policy_action_route": action_route.value,
            "libero_renderer": renderer_config.to_dict(),
        }
        if consumer_runtime is not None:
            summary["video_action_composition"] = {
                "action_consumer_checkpoint_file": str(
                    consumer_runtime.checkpoint_path.resolve()
                )
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
                    video_action_composition,
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


def _resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    task_spec = resolve_libero_task_by_id(
        benchmark_name,
        task_id,
        project_root=REPO_ROOT,
    )
    return task_spec, task_spec.task_language


def _construct_single_env(
    task_spec: LiberoTaskSpec,
    *,
    renderer_profile: LiberoRendererProfile | str,
):
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
                renderer_profile=renderer_profile,
            )
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env
