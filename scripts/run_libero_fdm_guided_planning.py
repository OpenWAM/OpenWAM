from __future__ import annotations

"""Run pi0-fast policy rollouts with optional Open-WAM FDM-guided planning.

The script is intentionally staged:
1. `--policy gaussian --dynamics dummy --dry-run` verifies the planner loop.
2. `--policy pi0fast --planner baseline` evaluates the small policy directly.
3. `--policy pi0fast --planner fdm_guided` uses the same policy sampler but
   routes candidate action chunks through a forward-dynamics model before
   executing the first chunk of the best imagined branch.
"""

import argparse
from dataclasses import dataclass, replace
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import imageio.v2 as imageio
import numpy as np

from open_wam.planning import ActionChunk, FdmGuidedRecedingHorizonPlanner, PlannerConfig, PlanningContext
from open_wam.planning.contracts import CandidateTrajectory, PlanningResult
from open_wam.planning.dummy_dynamics import ActionTintDynamics
from open_wam.planning.evaluators import (
    ActionMagnitudeEvaluator,
    ConstantEvaluator,
    GoalDeltaAlignmentEvaluator,
    GoalImageL2Evaluator,
)
from open_wam.planning.openwam_fdm import OpenWamActionConditionedFdm
from open_wam.planning.pi0fast import Pi0FastBatchMapping, Pi0FastPolicySampler
from open_wam.planning.samplers import GaussianActionSampler
from open_wam.planning.uva_fdm import UvaLiberoActionConditionedFdm
from open_wam.planning.vlm_evaluators import GeminiVlmCandidateRerankEvaluator, GeminiVlmRerankConfig
from open_wam.simulators import EpisodeSpec, SimulatorObservation


@dataclass(frozen=True)
class _ReplayInitState:
    init_state_index: int | None = None
    dataset_episode_index: int | None = None


def main() -> None:
    args = _parse_args()
    _load_local_env_file(Path(args.gemini_env_file))
    output_dir = Path(args.output_dir).expanduser().resolve() / (args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _build_policy(args)
    dynamics = _build_dynamics(args)
    _validate_planner_runtime_contract(args, dynamics)
    evaluator = _build_evaluator(args, output_dir=output_dir)
    planner_config = PlannerConfig(
        num_policy_samples=args.num_policy_samples,
        beam_width=args.beam_width,
        chunk_action_steps=args.chunk_action_steps,
        min_plan_action_steps=args.min_plan_action_steps,
        max_plan_chunks=args.max_plan_chunks,
        execute_action_steps=args.execute_action_steps,
        policy_temperature=args.policy_temperature,
        include_policy_prior_candidate=bool(args.include_policy_prior_candidate),
        policy_prior_temperature=float(args.policy_prior_temperature),
        policy_prior_abstain_margin=float(args.policy_prior_abstain_margin),
        seed=args.seed,
    )
    planner = FdmGuidedRecedingHorizonPlanner(
        policy=policy,
        dynamics=dynamics,
        evaluator=evaluator,
        config=planner_config,
    )

    if args.planner == "replay_actions":
        summary = _run_libero_action_replay(args, output_dir=output_dir)
    elif args.dry_run:
        summary = _run_dry_rollout(args, planner=planner, output_dir=output_dir)
    elif args.planner == "full_imagined_vlm_receding":
        summary = _run_libero_full_imagined_vlm_rollouts(args, planner=planner, policy=policy, output_dir=output_dir)
    elif args.planner == "imagined_open_loop":
        summary = _run_libero_imagined_rollouts(args, planner=planner, policy=policy, output_dir=output_dir)
    else:
        summary = _run_libero_rollouts(args, planner=planner, policy=policy, output_dir=output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _run_dry_rollout(
    args: argparse.Namespace,
    *,
    planner: FdmGuidedRecedingHorizonPlanner,
    output_dir: Path,
) -> dict[str, Any]:
    context = PlanningContext(
        views={"agentview_image": np.zeros((64, 64, 3), dtype=np.uint8)},
        state=np.zeros(args.state_dim, dtype=np.float32),
        task_text="dry run",
    )
    result = planner.plan(context, seed=args.seed)
    return {
        "mode": "dry_run",
        "output_dir": str(output_dir),
        "selected_score": result.selected.score,
        "candidate_count": len(result.candidates),
        "first_chunk_shape": list(result.first_action_chunk.actions.shape),
        "planned_action_steps": result.metadata["planned_action_steps"],
    }


def _run_libero_rollouts(
    args: argparse.Namespace,
    *,
    planner: FdmGuidedRecedingHorizonPlanner,
    policy: Any,
    output_dir: Path,
) -> dict[str, Any]:
    from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig, ensure_local_libero_config

    ensure_local_libero_config(Path.cwd())
    records: list[dict[str, Any]] = []
    rollout_schedule = _resolve_rollout_schedule(args)
    goal_provider = _build_goal_provider(args, output_dir=output_dir)
    for rollout_index, (task_id, episode_idx) in enumerate(rollout_schedule):
        replay_init = _resolve_replay_init_state(args, task_id=task_id, episode_idx=episode_idx)
        adapter = LiberoBenchmarkAdapter(
            LiberoEnvConfig(
                benchmark_name=args.benchmark,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                horizon=args.max_env_steps + 16,
                ignore_done=True,
                init_state_index=replay_init.init_state_index,
            ),
            project_root=Path.cwd(),
        )
        frames: list[np.ndarray] = []
        try:
            if hasattr(policy, "reset"):
                policy.reset()
            observation = adapter.reset(
                EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=args.seed + rollout_index)
            )
            startup_noop_steps = int(args.startup_noop_steps)
            for _ in range(startup_noop_steps):
                transition = adapter.step(_libero_noop_action(action_dim=args.action_dim))
                observation = transition.observation
            static_target_goal = (
                goal_provider(
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_text=observation.task_text,
                    rollout_index=rollout_index,
                )
                if goal_provider is not None
                else None
            )
            success = False
            env_steps = 0
            executed_actions: list[np.ndarray] = []
            replan_records: list[dict[str, Any]] = []
            while env_steps < int(args.max_env_steps):
                context = _planning_context_from_observation(observation)
                context = _warm_openwam_context_for_planning(args, planner.dynamics, context)
                target_goal = static_target_goal
                if args.evaluator == "demo_future_l2" and goal_provider is not None:
                    target_goal = goal_provider(
                        task_id=task_id,
                        episode_idx=episode_idx,
                        task_text=observation.task_text,
                        rollout_index=rollout_index,
                        frame_index=env_steps + int(args.demo_future_offset_steps),
                    )
                planner_goal = _planner_goal_for_observation(args, observation, target_goal)
                if args.planner == "baseline":
                    action_chunk = policy.sample_action_chunks(
                        context,
                        num_samples=1,
                        chunk_action_steps=args.chunk_action_steps,
                        temperature=args.policy_temperature,
                        seed=args.seed + rollout_index * 100_000 + env_steps,
                    )[0]
                    selected_score = None
                    candidate_count = 1
                    candidate_action_diversity = None
                    candidate_dump_path = None
                else:
                    result = planner.plan(
                        context,
                        goal=planner_goal,
                        seed=args.seed + rollout_index * 100_000 + env_steps,
                    )
                    action_chunk = result.first_action_chunk
                    selected_score = result.selected.score
                    candidate_count = int(result.metadata.get("evaluated_candidate_count", len(result.candidates)))
                    final_beam_count = int(result.metadata.get("final_beam_count", len(result.candidates)))
                    candidate_action_diversity = _candidate_first_chunk_action_diversity(result.candidates)
                    candidate_dump_path = (
                        _dump_planner_candidate_actions(
                            output_dir=output_dir,
                            rollout_index=rollout_index,
                            replan_index=len(replan_records),
                            result=result,
                        )
                        if args.dump_planner_candidates
                        else None
                    )
                steps_to_execute = min(int(args.execute_action_steps), int(action_chunk.actions.shape[0]))
                replan_record = {
                    "env_step": env_steps,
                    "selected_score": selected_score,
                    "candidate_count": candidate_count,
                    "executed_steps": steps_to_execute,
                    "selected_action_stats": _action_array_stats(action_chunk.actions[:steps_to_execute]),
                }
                if args.planner != "baseline":
                    replan_record["final_beam_count"] = final_beam_count
                    replan_record["candidate_action_diversity"] = candidate_action_diversity
                    if candidate_dump_path is not None:
                        replan_record["candidate_dump_path"] = str(candidate_dump_path)
                replan_records.append(replan_record)
                for action in action_chunk.actions[:steps_to_execute]:
                    executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                    transition = adapter.step(np.asarray(action, dtype=np.float32))
                    observation = transition.observation
                    frame = adapter.render_frame(observation)
                    if frame is not None:
                        frames.append(np.asarray(frame, dtype=np.uint8))
                    env_steps += 1
                    success = bool(transition.success or transition.done)
                    if success or env_steps >= int(args.max_env_steps):
                        break
                if success:
                    break
            video_path = None
            if frames and args.save_video:
                video_path = output_dir / f"rollout_{rollout_index:03d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.video_fps)
            action_trace_path = _save_executed_action_trace(
                output_dir=output_dir,
                rollout_index=rollout_index,
                actions=executed_actions,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            records.append(
                {
                    "rollout_index": rollout_index,
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "success": bool(success),
                    "env_steps": int(env_steps),
                    "startup_noop_steps": int(startup_noop_steps),
                    "resolved_init_state_index": replay_init.init_state_index,
                    "replay_dataset_episode_index": replay_init.dataset_episode_index,
                    "video_path": None if video_path is None else str(video_path),
                    "action_trace_path": None if action_trace_path is None else str(action_trace_path),
                    "replans": replan_records,
                }
            )
        finally:
            adapter.close()
    successes = sum(1 for record in records if record["success"])
    return {
        "mode": "libero",
        "planner": args.planner,
        "policy": args.policy,
        "dynamics": args.dynamics,
        "episodes": len(records),
        "successes": successes,
        "success_rate": None if not records else successes / len(records),
        "rollout_schedule": [
            {"task_id": int(task_id), "episode_idx": int(episode_idx)}
            for task_id, episode_idx in rollout_schedule
        ],
        "output_dir": str(output_dir),
        "records": records,
    }


def _run_libero_full_imagined_vlm_rollouts(
    args: argparse.Namespace,
    *,
    planner: FdmGuidedRecedingHorizonPlanner,
    policy: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Receding-horizon rollout where Gemini ranks full imagined trajectories."""

    from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig, ensure_local_libero_config

    ensure_local_libero_config(Path.cwd())
    if not isinstance(planner.dynamics, OpenWamActionConditionedFdm):
        raise ValueError("--planner full_imagined_vlm_receding currently requires --dynamics openwam_gjd.")
    rollout_schedule = _resolve_rollout_schedule(args)
    records: list[dict[str, Any]] = []
    goal_provider = _build_goal_provider(args, output_dir=output_dir)
    fixed_candidate_actions = _load_optional_action_trace(args.fixed_candidate_action_trace)

    for rollout_index, (task_id, episode_idx) in enumerate(rollout_schedule):
        replay_init = _resolve_replay_init_state(args, task_id=task_id, episode_idx=episode_idx)

        def _adapter_factory() -> Any:
            return LiberoBenchmarkAdapter(
                LiberoEnvConfig(
                    benchmark_name=args.benchmark,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    horizon=max(int(args.max_env_steps) + 16, int(args.min_plan_action_steps) + 16),
                    ignore_done=True,
                    init_state_index=replay_init.init_state_index,
                ),
                project_root=Path.cwd(),
            )

        def _proprio_adapter_factory() -> Any:
            return LiberoBenchmarkAdapter(
                LiberoEnvConfig(
                    benchmark_name=args.benchmark,
                    env_backend="control",
                    use_camera_obs=False,
                    has_offscreen_renderer=False,
                    camera_obs_keys=(),
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    horizon=max(int(args.max_env_steps) + 16, int(args.min_plan_action_steps) + 16),
                    ignore_done=True,
                    init_state_index=replay_init.init_state_index,
                ),
                project_root=Path.cwd(),
            )

        adapter = _adapter_factory()
        frames: list[np.ndarray] = []
        replan_records: list[dict[str, Any]] = []
        executed_actions: list[np.ndarray] = []
        success = False
        env_steps = 0
        try:
            if hasattr(policy, "reset"):
                policy.reset()
            seed = int(args.seed) + int(rollout_index)
            observation = adapter.reset(EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed))
            for _ in range(int(args.startup_noop_steps)):
                transition = adapter.step(_libero_noop_action(action_dim=args.action_dim))
                observation = transition.observation
            if args.save_video:
                frames.append(
                    _planning_context_canvas(
                        _planning_context_from_observation(observation),
                        args=args,
                        view_transform="none",
                    )
                )
            termination_reason = "max_env_steps"
            static_target_goal = (
                goal_provider(
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_text=observation.task_text,
                    rollout_index=rollout_index,
                )
                if goal_provider is not None
                else None
            )
            context_propagator = _LiberoSimulatorProprioPropagator(
                adapter_factory=_proprio_adapter_factory,
                action_dim=int(args.action_dim),
            )
            while env_steps < int(args.max_env_steps):
                if (
                    args.max_replans_per_rollout is not None
                    and len(replan_records) >= int(args.max_replans_per_rollout)
                ):
                    termination_reason = "max_replans_per_rollout"
                    break
                context = _planning_context_from_observation(
                    observation,
                    adapter=adapter,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    seed=seed,
                )
                context = _warm_openwam_context_for_planning(args, planner.dynamics, context)
                planner_goal = _planner_goal_for_observation(args, observation, static_target_goal)
                if isinstance(planner_goal, dict):
                    planner_goal = dict(planner_goal)
                    planner_goal["planning_scope"] = "full_trajectory"
                    planner_goal["execute_first_chunk_only"] = True
                elif planner_goal is None:
                    planner_goal = {
                        "task_text": observation.task_text,
                        "planning_scope": "full_trajectory",
                        "execute_first_chunk_only": True,
                    }
                planning_error: str | None = None
                fallback_record: dict[str, Any] | None = None
                result: PlanningResult | None = None
                plan_config = planner.config
                if bool(args.plan_to_env_horizon):
                    remaining_steps = max(1, int(args.max_env_steps) - int(env_steps))
                    plan_config = replace(
                        planner.config,
                        min_plan_action_steps=remaining_steps,
                        max_plan_chunks=int(np.ceil(remaining_steps / int(planner.config.chunk_action_steps))),
                    )
                try:
                    result = _plan_full_imagined_trajectories(
                        policy=policy,
                        dynamics=planner.dynamics,
                        evaluator=planner.evaluator,
                        context_propagator=context_propagator,
                        context_rewarmer=lambda branch_context: _warm_openwam_context_for_planning(
                            args,
                            planner.dynamics,
                            branch_context,
                        ),
                        context=context,
                        goal=planner_goal,
                        config=plan_config,
                        seed=int(args.seed) + rollout_index * 100_000 + env_steps,
                        fixed_candidate_actions=fixed_candidate_actions,
                        fixed_candidate_start_step=env_steps,
                        future_policy_temperature=args.full_imagined_future_policy_temperature,
                    )
                    action_chunk = result.first_action_chunk
                except RuntimeError as exc:
                    planning_error = _exception_summary(exc)
                    if "All full-imagined branches failed" not in str(exc):
                        raise
                    action_chunk, fallback_record = _full_imagined_planning_failure_fallback(
                        args,
                        policy=policy,
                        context=context,
                        seed=int(args.seed) + rollout_index * 100_000 + env_steps,
                        reason=planning_error,
                    )
                replan_dir = output_dir / f"rollout_{rollout_index:03d}_replan_{len(replan_records):03d}"
                if result is not None and bool(args.save_full_imagined_candidates):
                    _save_full_imagined_candidate_artifacts(
                        replan_dir,
                        result=result,
                        dynamics=planner.dynamics,
                        video_fps=float(args.video_fps),
                    )
                prefix_imagined_manifest = None
                if result is not None and bool(args.save_replan_prefix_imagined_videos):
                    prefix_imagined_manifest = _save_replan_prefix_imagined_videos(
                        replan_dir / "real_prefix_plus_imagined",
                        real_prefix_frames=frames,
                        result=result,
                        dynamics=planner.dynamics,
                        remaining_frames=max(0, int(args.max_env_steps) - int(env_steps)),
                        video_fps=float(args.video_fps),
                    )
                steps_to_execute = min(
                    int(args.execute_action_steps),
                    int(action_chunk.actions.shape[0]),
                    int(args.max_env_steps) - int(env_steps),
                )
                replan_record = {
                    "env_step": int(env_steps),
                    "selected_score": None if result is None else float(result.selected.score),
                    "candidate_count": 0 if result is None else int(len(result.candidates)),
                    "planned_chunks": 0 if result is None else int(result.metadata["planned_chunks"]),
                    "planned_action_steps": int(action_chunk.actions.shape[0])
                    if result is None
                    else int(result.metadata["planned_action_steps"]),
                    "executed_steps": int(steps_to_execute),
                    "selected_action_stats": _action_array_stats(action_chunk.actions[:steps_to_execute]),
                    "candidate_artifact_dir": (
                        str(replan_dir)
                        if result is not None and bool(args.save_full_imagined_candidates)
                        else None
                    ),
                }
                if planning_error is not None:
                    replan_record["planning_error"] = planning_error
                if fallback_record is not None:
                    replan_record["fallback"] = fallback_record
                if result is not None:
                    replan_record["selected_candidate_role"] = str(
                        action_chunk.metadata.get("candidate_role", "unknown")
                    )
                    replan_record["selected_candidate_index"] = action_chunk.metadata.get(
                        "full_imagined_candidate_index"
                    )
                if prefix_imagined_manifest is not None:
                    replan_record["real_prefix_plus_imagined_manifest"] = str(prefix_imagined_manifest)
                replan_records.append(replan_record)
                for action in action_chunk.actions[:steps_to_execute]:
                    executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                    transition = adapter.step(np.asarray(action, dtype=np.float32))
                    observation = transition.observation
                    if args.save_video:
                        frames.append(
                            _planning_context_canvas(
                                _planning_context_from_observation(observation),
                                args=args,
                                view_transform="none",
                            )
                        )
                    env_steps += 1
                    success = bool(transition.success or transition.done)
                    if success or env_steps >= int(args.max_env_steps):
                        termination_reason = "success" if success else "max_env_steps"
                        break
                if success:
                    break
            video_path = None
            if frames and args.save_video:
                video_path = output_dir / f"rollout_{rollout_index:03d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.video_fps)
            action_trace_path = _save_executed_action_trace(
                output_dir=output_dir,
                rollout_index=rollout_index,
                actions=executed_actions,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            records.append(
                {
                    "rollout_index": int(rollout_index),
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "success": bool(success),
                    "env_steps": int(env_steps),
                    "termination_reason": termination_reason,
                    "startup_noop_steps": int(args.startup_noop_steps),
                    "resolved_init_state_index": replay_init.init_state_index,
                    "replay_dataset_episode_index": replay_init.dataset_episode_index,
                    "video_path": None if video_path is None else str(video_path),
                    "action_trace_path": None if action_trace_path is None else str(action_trace_path),
                    "replans": replan_records,
                }
            )
        finally:
            adapter.close()

    successes = sum(1 for record in records if record["success"])
    return {
        "mode": "libero_full_imagined_vlm_receding",
        "planner": args.planner,
        "policy": args.policy,
        "dynamics": args.dynamics,
        "evaluator": args.evaluator,
        "episodes": len(records),
        "successes": int(successes),
        "success_rate": None if not records else successes / len(records),
        "rollout_schedule": [
            {"task_id": int(task_id), "episode_idx": int(episode_idx)}
            for task_id, episode_idx in rollout_schedule
        ],
        "output_dir": str(output_dir),
        "records": records,
    }


def _plan_full_imagined_trajectories(
    *,
    policy: Any,
    dynamics: Any,
    evaluator: Any,
    context_propagator: Any,
    context_rewarmer: Callable[[PlanningContext], PlanningContext] | None = None,
    context: PlanningContext,
    goal: Any | None,
    config: PlannerConfig,
    seed: int,
    fixed_candidate_actions: np.ndarray | None = None,
    fixed_candidate_start_step: int = 0,
    future_policy_temperature: float | None = None,
) -> PlanningResult:
    candidates: list[CandidateTrajectory] = []
    candidate_specs: list[dict[str, Any]] = []
    if fixed_candidate_actions is not None:
        candidate_specs.append(
            {
                "candidate_role": "fixed_action_trace",
                "temperature": 0.0,
                "fixed_action_trace": fixed_candidate_actions,
                "fixed_candidate_start_step": int(fixed_candidate_start_step),
            }
        )
    if bool(config.include_policy_prior_candidate):
        candidate_specs.append(
            {
                "candidate_role": "policy_prior",
                "temperature": float(config.policy_prior_temperature),
            }
        )
    candidate_specs.extend(
        {
            "candidate_role": "full_imagined_policy_sample",
            "temperature": float(config.policy_temperature),
        }
        for _ in range(int(config.num_policy_samples))
    )
    preserve_live_policy_state = _full_imagined_uses_native_prior_only(candidate_specs, config)
    for candidate_index, candidate_spec in enumerate(candidate_specs):
        uses_policy = str(candidate_spec.get("candidate_role")) != "fixed_action_trace"
        policy_snapshot = None
        if uses_policy and not preserve_live_policy_state:
            policy_snapshot = _snapshot_policy_sampler_state(policy)
            if hasattr(policy, "reset"):
                policy.reset()
        branch_propagator_started = False
        try:
            begin_branch = getattr(context_propagator, "begin_branch", None)
            if callable(begin_branch):
                begin_branch(context)
                branch_propagator_started = True
            branch: CandidateTrajectory | None = CandidateTrajectory(context=context)
            branch_context = context
            chunk_index = 0
            while branch is not None and chunk_index < int(config.max_plan_chunks) and branch.action_count < int(config.min_plan_action_steps):
                branch_seed = int(seed) + candidate_index * 1_000_003 + chunk_index * 10_007
                chunk_temperature = _full_imagined_chunk_temperature(
                    candidate_spec,
                    chunk_index=chunk_index,
                    future_policy_temperature=future_policy_temperature,
                )
                try:
                    action_chunks = _sample_full_imagined_candidate_chunk(
                        policy=policy,
                        context=branch_context,
                        candidate_spec=candidate_spec,
                        chunk_action_steps=int(config.chunk_action_steps),
                        branch_action_count=branch.action_count,
                        temperature=float(chunk_temperature),
                        seed=branch_seed,
                    )
                except Exception as exc:
                    if not branch.action_chunks:
                        print(
                            "[warn] dropping full-imagined candidate "
                            f"{candidate_index}: policy failed before first chunk: {_exception_summary(exc)}",
                            flush=True,
                        )
                        branch = None
                    else:
                        print(
                            "[warn] truncating full-imagined candidate "
                            f"{candidate_index} at chunk {chunk_index}: policy failed: {_exception_summary(exc)}",
                            flush=True,
                        )
                        branch = _mark_truncated_branch(
                            branch,
                            candidate_index=candidate_index,
                            chunk_index=chunk_index,
                            reason=f"policy_error:{_exception_summary(exc)}",
                        )
                    break
                if not action_chunks:
                    if not branch.action_chunks:
                        branch = None
                    else:
                        branch = _mark_truncated_branch(
                            branch,
                            candidate_index=candidate_index,
                            chunk_index=chunk_index,
                            reason="policy_empty_action_chunk",
                        )
                    break
                action_chunk = _with_full_trajectory_metadata(
                    action_chunks[0],
                    candidate_index=candidate_index,
                    chunk_index=chunk_index,
                    candidate_role=str(candidate_spec["candidate_role"]),
                    sampling_temperature=float(chunk_temperature),
                )
                try:
                    prediction = dynamics.predict(
                        branch_context,
                        action_chunk,
                        seed=branch_seed + 1,
                    )
                except Exception as exc:
                    if not branch.action_chunks:
                        print(
                            "[warn] dropping full-imagined candidate "
                            f"{candidate_index}: dynamics failed before first video: {_exception_summary(exc)}",
                            flush=True,
                        )
                        branch = None
                    else:
                        print(
                            "[warn] truncating full-imagined candidate "
                            f"{candidate_index} at chunk {chunk_index}: dynamics failed: {_exception_summary(exc)}",
                            flush=True,
                        )
                        branch = _mark_truncated_branch(
                            branch,
                            candidate_index=candidate_index,
                            chunk_index=chunk_index,
                            reason=f"dynamics_error:{_exception_summary(exc)}",
                        )
                    break
                action_steps = int(action_chunk.actions.shape[0])
                needs_future_context = (
                    chunk_index + 1 < int(config.max_plan_chunks)
                    and branch.action_count + action_steps < int(config.min_plan_action_steps)
                )
                next_context = prediction.next_context
                if needs_future_context and context_propagator is not None:
                    next_context = context_propagator.propagate(
                        prediction.next_context,
                        action_chunk,
                        seed=branch_seed + 2,
                    )
                if (
                    needs_future_context
                    and context_rewarmer is not None
                    and _requires_short_horizon_fdm_rewarm(dynamics, action_chunk)
                ):
                    next_context = context_rewarmer(
                        next_context.with_prediction(
                            predicted_video=next_context.predicted_video,
                            metadata_updates={
                                "short_horizon_fdm_rewarm": True,
                                "short_horizon_policy_action_steps": int(action_chunk.actions.shape[0]),
                                "short_horizon_internal_action_steps": int(dynamics.expected_action_steps_per_chunk()),
                            },
                        )
                    )
                if next_context is not prediction.next_context:
                    prediction = type(prediction)(
                        predicted_video=prediction.predicted_video,
                        next_context=next_context,
                        metadata=prediction.metadata,
                    )
                metadata_updates: dict[str, Any] = {
                    "candidate_index": int(candidate_index),
                    "last_chunk_index": int(chunk_index),
                }
                stitched_latents_key = _openwam_predicted_latents_key(dynamics)
                if stitched_latents_key is not None and stitched_latents_key in prediction.metadata:
                    metadata_updates[stitched_latents_key] = (
                        *tuple(branch.metadata.get(stitched_latents_key, ())),
                        prediction.metadata[stitched_latents_key],
                    )
                branch = branch.append(
                    action_chunk=action_chunk,
                    prediction=prediction,
                    score=0.0,
                    metadata_updates=metadata_updates,
                )
                branch_context = branch.context
                chunk_index += 1
            if branch is None or not branch.action_chunks:
                continue
            candidates.append(branch)
        finally:
            end_branch = getattr(context_propagator, "end_branch", None)
            if branch_propagator_started and callable(end_branch):
                end_branch()
            _restore_policy_sampler_state(policy, policy_snapshot)
    if not candidates:
        raise RuntimeError("All full-imagined branches failed before producing an executable action chunk.")
    scores = _score_full_imagined_candidates(evaluator, candidates, goal=goal)
    scored_candidates = [
        CandidateTrajectory(
            context=candidate.context,
            action_chunks=candidate.action_chunks,
            predicted_videos=candidate.predicted_videos,
            score=float(score),
            metadata=candidate.metadata,
        )
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    scored_candidates.sort(key=lambda item: item.score, reverse=True)
    selected = scored_candidates[0]
    return PlanningResult(
        selected=selected,
        candidates=tuple(scored_candidates),
        first_action_chunk=selected.action_chunks[0],
        metadata={
            "planned_chunks": len(selected.action_chunks),
            "planned_action_steps": selected.action_count,
            "execute_action_steps": min(int(config.execute_action_steps), int(selected.action_chunks[0].actions.shape[0])),
            "evaluated_candidate_count": int(len(scored_candidates)),
            "final_beam_count": int(len(scored_candidates)),
        },
    )


def _full_imagined_uses_native_prior_only(candidate_specs: list[dict[str, Any]], config: PlannerConfig) -> bool:
    """Return true when planning should preserve live policy processor state.

    LeRobot PI0-family policies have stateful preprocessors and an internal
    action queue. In the single deterministic-prior / one-chunk control path,
    the planner's first chunk should be exactly the native policy chunk that
    would be executed without FDM ranking. Resetting inside each replan breaks
    that native contract after the first chunk.
    """

    return (
        int(config.num_policy_samples) == 0
        and int(config.max_plan_chunks) == 1
        and len(candidate_specs) == 1
        and str(candidate_specs[0].get("candidate_role")) == "policy_prior"
    )


def _full_imagined_chunk_temperature(
    candidate_spec: Mapping[str, Any],
    *,
    chunk_index: int,
    future_policy_temperature: float | None,
) -> float:
    """Return the sampling temperature for one imagined branch chunk.

    If `future_policy_temperature` is set, stochastic candidates only use their
    configured temperature for the first executable chunk. Later imagined
    chunks use the supplied future temperature, typically 0.0, so the branch
    score asks: "if we execute this sampled first chunk and then return to the
    policy prior, how good does the future look?"
    """

    base_temperature = float(candidate_spec["temperature"])
    if future_policy_temperature is None or int(chunk_index) <= 0:
        return base_temperature
    role = str(candidate_spec.get("candidate_role", ""))
    if role in {"full_imagined_policy_sample", "policy_prior"}:
        return float(future_policy_temperature)
    return base_temperature


def _sample_full_imagined_candidate_chunk(
    *,
    policy: Any,
    context: PlanningContext,
    candidate_spec: Mapping[str, Any],
    chunk_action_steps: int,
    branch_action_count: int,
    temperature: float,
    seed: int,
) -> list[ActionChunk]:
    role = str(candidate_spec.get("candidate_role", ""))
    if role == "fixed_action_trace":
        trace = np.asarray(candidate_spec.get("fixed_action_trace"), dtype=np.float32)
        if trace.ndim != 2:
            raise ValueError(f"fixed_action_trace must have shape [T,D], got {trace.shape}.")
        start = int(candidate_spec.get("fixed_candidate_start_step", 0)) + int(branch_action_count)
        end = start + int(chunk_action_steps)
        if start < 0 or start >= int(trace.shape[0]):
            return []
        actions = trace[start : min(end, int(trace.shape[0]))]
        if int(actions.shape[0]) <= 0:
            return []
        metadata = {
            "candidate_role": role,
            "fixed_trace_start_step": int(start),
            "fixed_trace_end_step": int(start + int(actions.shape[0])),
            "requested_action_steps": int(chunk_action_steps),
        }
        return [ActionChunk(actions=actions, metadata=metadata)]

    return list(
        policy.sample_action_chunks(
            context,
            num_samples=1,
            chunk_action_steps=int(chunk_action_steps),
            temperature=float(temperature),
            seed=int(seed),
        )
    )


def _snapshot_policy_sampler_state(policy: Any) -> Any | None:
    snapshot_fn = getattr(policy, "snapshot_state", None)
    if not callable(snapshot_fn):
        return None
    return snapshot_fn()


def _restore_policy_sampler_state(policy: Any, snapshot: Any | None) -> None:
    if snapshot is None:
        return
    restore_fn = getattr(policy, "restore_state", None)
    if callable(restore_fn):
        restore_fn(snapshot)


def _full_imagined_planning_failure_fallback(
    args: argparse.Namespace,
    *,
    policy: Any,
    context: PlanningContext,
    seed: int,
    reason: str,
) -> tuple[ActionChunk, dict[str, Any]]:
    fallback_mode = str(args.full_imagined_failure_fallback)
    if fallback_mode not in {"policy_prior", "noop", "error"}:
        raise ValueError(f"Unsupported --full-imagined-failure-fallback={fallback_mode!r}.")
    if fallback_mode == "error":
        raise RuntimeError(reason)
    if fallback_mode == "policy_prior":
        try:
            if hasattr(policy, "reset"):
                policy.reset()
            chunks = policy.sample_action_chunks(
                context,
                num_samples=1,
                chunk_action_steps=int(args.chunk_action_steps),
                temperature=float(args.policy_prior_temperature),
                seed=int(seed),
            )
            if chunks:
                chunk = chunks[0]
                metadata = dict(chunk.metadata or {})
                metadata.update(
                    {
                        "candidate_role": "policy_prior_fallback",
                        "fallback_reason": reason,
                    }
                )
                return (
                    ActionChunk(
                        actions=chunk.actions,
                        logprob=chunk.logprob,
                        sampler_score=chunk.sampler_score,
                        metadata=metadata,
                    ),
                    {
                        "mode": "policy_prior",
                        "reason": reason,
                        "action_steps": int(chunk.actions.shape[0]),
                    },
                )
        except Exception as exc:
            reason = f"{reason}; policy_prior_fallback_failed={_exception_summary(exc)}"
            fallback_mode = "noop"
    if fallback_mode == "noop":
        action = _libero_noop_action(action_dim=int(args.action_dim))
        actions = np.repeat(action[None], int(args.execute_action_steps), axis=0).astype(np.float32, copy=False)
        return (
            ActionChunk(
                actions=actions,
                metadata={
                    "candidate_role": "noop_fallback",
                    "fallback_reason": reason,
                },
            ),
            {
                "mode": "noop",
                "reason": reason,
                "action_steps": int(actions.shape[0]),
            },
        )
    raise AssertionError(f"Unhandled fallback mode {fallback_mode!r}.")


def _requires_short_horizon_fdm_rewarm(dynamics: Any, action_chunk: ActionChunk) -> bool:
    expected_fn = getattr(dynamics, "expected_action_steps_per_chunk", None)
    if not callable(expected_fn):
        return False
    if not bool(getattr(dynamics, "crop_decoded_video_to_action_steps", False)):
        return False
    expected = int(expected_fn())
    actual = int(action_chunk.actions.shape[0])
    return 0 < actual < expected


def _mark_truncated_branch(
    branch: CandidateTrajectory,
    *,
    candidate_index: int,
    chunk_index: int,
    reason: str,
) -> CandidateTrajectory:
    metadata = dict(branch.metadata)
    metadata.update(
        {
            "candidate_index": int(candidate_index),
            "truncated": True,
            "truncation_chunk_index": int(chunk_index),
            "truncation_reason": str(reason),
        }
    )
    return CandidateTrajectory(
        context=branch.context,
        action_chunks=branch.action_chunks,
        predicted_videos=branch.predicted_videos,
        score=branch.score,
        metadata=metadata,
    )


def _exception_summary(exc: BaseException, *, max_chars: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= int(max_chars) else text[: int(max_chars) - 3] + "..."


def _with_full_trajectory_metadata(
    action_chunk: ActionChunk,
    *,
    candidate_index: int,
    chunk_index: int,
    candidate_role: str = "full_imagined_policy_sample",
    sampling_temperature: float | None = None,
) -> ActionChunk:
    metadata = dict(action_chunk.metadata or {})
    metadata.update(
        {
            "candidate_role": str(candidate_role),
            "full_imagined_candidate_index": int(candidate_index),
            "full_imagined_chunk_index": int(chunk_index),
        }
    )
    if sampling_temperature is not None:
        metadata["full_imagined_sampling_temperature"] = float(sampling_temperature)
    return ActionChunk(
        actions=action_chunk.actions,
        logprob=action_chunk.logprob,
        sampler_score=action_chunk.sampler_score,
        metadata=metadata,
    )


def _score_full_imagined_candidates(evaluator: Any, candidates: list[CandidateTrajectory], *, goal: Any | None) -> list[float]:
    batch_scorer = getattr(evaluator, "score_candidates", None)
    if callable(batch_scorer):
        scores = list(batch_scorer(candidates, goal=goal))
    else:
        scores = [float(evaluator.score(candidate, goal=goal)) for candidate in candidates]
    if len(scores) != len(candidates):
        raise RuntimeError(f"Expected {len(candidates)} full-trajectory scores, got {len(scores)}.")
    return [float(score) for score in scores]


def _save_full_imagined_candidate_artifacts(
    output_dir: Path,
    *,
    result: PlanningResult,
    dynamics: Any | None = None,
    video_fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "selected_index": _candidate_index(result.selected),
        "candidate_count": int(len(result.candidates)),
        "candidates": [],
    }
    for rank, candidate in enumerate(result.candidates):
        candidate_index = _candidate_index(candidate)
        video = _candidate_debug_video(
            candidate,
            dynamics=dynamics,
            max_frames=int(candidate.action_count),
        )
        video_path = None
        if video is not None and int(video.shape[0]) > 0:
            video_path = output_dir / f"rank{rank:02d}_candidate{candidate_index:02d}.mp4"
            imageio.mimsave(video_path, video, fps=float(video_fps))
        action_path = output_dir / f"rank{rank:02d}_candidate{candidate_index:02d}_actions.npz"
        np.savez_compressed(action_path, actions=candidate.actions)
        manifest["candidates"].append(
            {
                "rank": int(rank),
                "candidate_index": int(candidate_index),
                "score": float(candidate.score),
                "action_steps": int(candidate.action_count),
                "action_path": str(action_path),
                "video_path": None if video_path is None else str(video_path),
                "predicted_video_frames": 0 if video is None else int(video.shape[0]),
            }
        )
    (output_dir / "full_imagined_candidates.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_replan_prefix_imagined_videos(
    output_dir: Path,
    *,
    real_prefix_frames: list[np.ndarray],
    result: PlanningResult,
    dynamics: Any | None = None,
    remaining_frames: int,
    video_fps: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _normalize_prefix_video(real_prefix_frames)
    manifest: dict[str, Any] = {
        "selected_index": _candidate_index(result.selected),
        "candidate_count": int(len(result.candidates)),
        "real_prefix_frames": int(prefix.shape[0]),
        "remaining_frames": int(max(0, remaining_frames)),
        "candidates": [],
    }
    for rank, candidate in enumerate(result.candidates):
        candidate_index = _candidate_index(candidate)
        suffix = _candidate_debug_video(
            candidate,
            dynamics=dynamics,
            max_frames=int(max(0, remaining_frames)),
        )
        suffix_frames = 0
        combined_path = None
        if suffix is not None and int(suffix.shape[0]) > 0:
            suffix_frames = int(suffix.shape[0])
            suffix = _resize_video_to_frame_shape(suffix, target_shape=prefix.shape[1:])
            combined = np.concatenate([prefix, suffix], axis=0)
            combined_path = output_dir / f"rank{rank:02d}_candidate{candidate_index:02d}_real_plus_imagined.mp4"
            imageio.mimsave(combined_path, combined, fps=float(video_fps))
        manifest["candidates"].append(
            {
                "rank": int(rank),
                "candidate_index": int(candidate_index),
                "score": float(candidate.score),
                "action_steps": int(candidate.action_count),
                "imagined_suffix_frames": int(suffix_frames),
                "total_video_frames": int(prefix.shape[0] + suffix_frames),
                "video_path": None if combined_path is None else str(combined_path),
            }
        )
    manifest_path = output_dir / "real_prefix_plus_imagined_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _candidate_index(candidate: CandidateTrajectory) -> int:
    value = candidate.metadata.get("candidate_index") if candidate.metadata else None
    if value is not None:
        return int(value)
    for chunk in candidate.action_chunks:
        value = chunk.metadata.get("full_imagined_candidate_index") if chunk.metadata else None
        if value is not None:
            return int(value)
    return -1


def _run_libero_imagined_rollouts(
    args: argparse.Namespace,
    *,
    planner: FdmGuidedRecedingHorizonPlanner,
    policy: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate full imagined LIBERO trajectories without executing them online.

    The learned FDM owns visual predictions and cache updates. A copied LIBERO
    simulator state is used only to advance robot proprio under the candidate
    actions; predicted images are never replaced with simulator renderings.
    """

    from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig, ensure_local_libero_config

    ensure_local_libero_config(Path.cwd())
    records: list[dict[str, Any]] = []
    rollout_schedule = _resolve_rollout_schedule(args)

    for rollout_index, (task_id, episode_idx) in enumerate(rollout_schedule):
        seed = int(args.seed) + int(rollout_index)
        replay_init = _resolve_replay_init_state(args, task_id=task_id, episode_idx=episode_idx)

        def _adapter_factory() -> Any:
            return LiberoBenchmarkAdapter(
                LiberoEnvConfig(
                    benchmark_name=args.benchmark,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    horizon=max(int(args.max_env_steps) + 16, int(args.chunk_action_steps) + 16),
                    ignore_done=True,
                    init_state_index=replay_init.init_state_index,
                ),
                project_root=Path.cwd(),
            )

        def _proprio_adapter_factory() -> Any:
            return LiberoBenchmarkAdapter(
                LiberoEnvConfig(
                    benchmark_name=args.benchmark,
                    env_backend="control",
                    use_camera_obs=False,
                    has_offscreen_renderer=False,
                    camera_obs_keys=(),
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    horizon=max(int(args.max_env_steps) + 16, int(args.chunk_action_steps) + 16),
                    ignore_done=True,
                    init_state_index=replay_init.init_state_index,
                ),
                project_root=Path.cwd(),
            )

        adapter = _adapter_factory()
        imagined_planner = FdmGuidedRecedingHorizonPlanner(
            policy=planner.policy,
            prior_policy=planner.prior_policy,
            dynamics=planner.dynamics,
            evaluator=planner.evaluator,
            context_propagator=_LiberoSimulatorProprioPropagator(
                adapter_factory=_proprio_adapter_factory,
                action_dim=int(args.action_dim),
            ),
            config=planner.config,
        )
        try:
            if hasattr(policy, "reset"):
                policy.reset()
            observation = adapter.reset(EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed))
            startup_noop_steps = int(args.startup_noop_steps)
            for _ in range(startup_noop_steps):
                transition = adapter.step(_libero_noop_action(action_dim=args.action_dim))
                observation = transition.observation
            context = _planning_context_from_observation(
                observation,
                adapter=adapter,
                task_id=task_id,
                episode_idx=episode_idx,
                seed=seed,
            )
            initial_context_canvas = _planning_context_canvas(
                context,
                args=args,
                view_transform="none",
            )
            initial_context_path = output_dir / f"initial_context_{rollout_index:03d}.png"
            imageio.imwrite(initial_context_path, initial_context_canvas)
            context = _warm_openwam_context_for_planning(args, planner.dynamics, context)
            result = imagined_planner.plan(
                context,
                goal=None,
                seed=int(args.seed) + rollout_index * 100_000,
            )
            max_steps = int(args.max_env_steps)
            actions = np.asarray(result.selected.actions, dtype=np.float32)[:max_steps]
            predicted_video = _candidate_debug_video(
                result.selected,
                dynamics=planner.dynamics,
                max_frames=max_steps,
            )
            proprio_history = _context_proprio_history(result.selected.context)[: actions.shape[0] + 1]
            trace_path = output_dir / f"imagined_rollout_{rollout_index:03d}.npz"
            np.savez_compressed(
                trace_path,
                actions=actions,
                proprio_states=proprio_history,
                predicted_video=np.zeros((0,), dtype=np.uint8) if predicted_video is None else predicted_video,
            )
            video_path = None
            video_with_context_path = None
            if predicted_video is not None and int(predicted_video.shape[0]) > 0 and args.save_video:
                video_path = output_dir / f"imagined_rollout_{rollout_index:03d}.mp4"
                imageio.mimsave(video_path, predicted_video, fps=args.video_fps)
                video_with_context = _prepend_context_canvas_to_video(
                    initial_context_canvas,
                    predicted_video,
                )
                video_with_context_path = output_dir / f"imagined_rollout_with_context_{rollout_index:03d}.mp4"
                imageio.mimsave(video_with_context_path, video_with_context, fps=args.video_fps)
            records.append(
                {
                    "rollout_index": int(rollout_index),
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "planned_action_steps": int(result.metadata["planned_action_steps"]),
                    "saved_action_steps": int(actions.shape[0]),
                    "planned_chunks": int(result.metadata["planned_chunks"]),
                    "predicted_video_frames": 0 if predicted_video is None else int(predicted_video.shape[0]),
                    "proprio_state_steps": int(proprio_history.shape[0]),
                    "startup_noop_steps": int(startup_noop_steps),
                    "resolved_init_state_index": replay_init.init_state_index,
                    "replay_dataset_episode_index": replay_init.dataset_episode_index,
                    "trace_path": str(trace_path),
                    "initial_context_path": str(initial_context_path),
                    "video_path": None if video_path is None else str(video_path),
                    "video_with_context_path": None if video_with_context_path is None else str(video_with_context_path),
                    "final_score": float(result.selected.score),
                    "candidate_count": int(result.metadata.get("evaluated_candidate_count", len(result.candidates))),
                }
            )
        finally:
            adapter.close()
    return {
        "mode": "libero_imagined_open_loop",
        "planner": args.planner,
        "policy": args.policy,
        "dynamics": args.dynamics,
        "episodes": len(records),
        "requested_action_steps": int(args.max_env_steps),
        "rollout_schedule": [
            {"task_id": int(task_id), "episode_idx": int(episode_idx)}
            for task_id, episode_idx in rollout_schedule
        ],
        "output_dir": str(output_dir),
        "records": records,
    }


def _run_libero_action_replay(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    """Replay saved action traces in LIBERO and render true simulator frames."""

    from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig, ensure_local_libero_config

    if args.action_trace is None:
        raise ValueError("--planner replay_actions requires --action-trace.")
    action_trace_path = Path(args.action_trace).expanduser().resolve()
    if not action_trace_path.is_file():
        raise FileNotFoundError(f"Action trace not found: {action_trace_path}")
    with np.load(action_trace_path) as trace:
        if "actions" not in trace:
            raise KeyError(f"Action trace {action_trace_path} does not contain an 'actions' array.")
        all_actions = np.asarray(trace["actions"], dtype=np.float32)
    if all_actions.ndim != 2:
        raise ValueError(f"Expected actions [T,D] in {action_trace_path}, got {all_actions.shape}.")
    if int(all_actions.shape[1]) != int(args.action_dim):
        raise ValueError(
            f"Action trace dim {int(all_actions.shape[1])} does not match --action-dim {int(args.action_dim)}."
        )

    ensure_local_libero_config(Path.cwd())
    records: list[dict[str, Any]] = []
    rollout_schedule = _resolve_rollout_schedule(args)
    max_steps = min(int(args.max_env_steps), int(all_actions.shape[0]))
    actions = all_actions[:max_steps]

    for rollout_index, (task_id, episode_idx) in enumerate(rollout_schedule):
        seed = int(args.seed) + int(rollout_index)
        replay_init = _resolve_replay_init_state(args, task_id=task_id, episode_idx=episode_idx)
        adapter = LiberoBenchmarkAdapter(
            LiberoEnvConfig(
                benchmark_name=args.benchmark,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                horizon=max_steps + 16,
                ignore_done=True,
                init_state_index=replay_init.init_state_index,
            ),
            project_root=Path.cwd(),
        )
        frames: list[np.ndarray] = []
        success = False
        success_step: int | None = None
        try:
            observation = adapter.reset(EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed))
            startup_noop_steps = int(args.startup_noop_steps)
            for _ in range(startup_noop_steps):
                transition = adapter.step(_libero_noop_action(action_dim=args.action_dim))
                observation = transition.observation
            if args.save_video:
                frames.append(
                    _planning_context_canvas(
                        _planning_context_from_observation(observation),
                        args=args,
                        view_transform="none",
                    )
                )
            for step_index, action in enumerate(actions):
                transition = adapter.step(np.asarray(action, dtype=np.float32))
                observation = transition.observation
                if args.save_video:
                    frames.append(
                        _planning_context_canvas(
                            _planning_context_from_observation(observation),
                            args=args,
                            view_transform="none",
                        )
                    )
                if bool(transition.success or transition.done):
                    success = True
                    success_step = int(step_index + 1)
                    if not bool(args.replay_continue_after_success):
                        break
            video_path = None
            if frames and args.save_video:
                video_path = output_dir / f"sim_replay_{rollout_index:03d}.mp4"
                imageio.mimsave(video_path, frames, fps=args.video_fps)
            records.append(
                {
                    "rollout_index": int(rollout_index),
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "action_trace_path": str(action_trace_path),
                    "available_action_steps": int(all_actions.shape[0]),
                    "requested_action_steps": int(args.max_env_steps),
                    "executed_action_steps": int(step_index + 1 if len(actions) else 0),
                    "success": bool(success),
                    "success_step": success_step,
                    "startup_noop_steps": int(startup_noop_steps),
                    "resolved_init_state_index": replay_init.init_state_index,
                    "replay_dataset_episode_index": replay_init.dataset_episode_index,
                    "video_path": None if video_path is None else str(video_path),
                    "action_stats": _action_array_stats(actions),
                }
            )
        finally:
            adapter.close()

    successes = sum(1 for record in records if record["success"])
    return {
        "mode": "libero_action_replay",
        "planner": args.planner,
        "episodes": len(records),
        "successes": int(successes),
        "success_rate": None if not records else float(successes / len(records)),
        "rollout_schedule": [
            {"task_id": int(task_id), "episode_idx": int(episode_idx)}
            for task_id, episode_idx in rollout_schedule
        ],
        "output_dir": str(output_dir),
        "records": records,
    }


def _warm_openwam_context_for_planning(
    args: argparse.Namespace,
    dynamics: Any,
    context: PlanningContext,
) -> PlanningContext:
    if not isinstance(dynamics, OpenWamActionConditionedFdm):
        return context
    return dynamics.warm_context_from_views(
        context,
        view_aliases=_openwam_view_aliases(args),
        action_dim=args.action_dim,
        action_conditioning_mode=args.openwam_fdm_mode,
    )


def _openwam_view_aliases(args: argparse.Namespace) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if args.openwam_agentview_camera_name:
        aliases[str(args.openwam_agentview_camera_name)] = str(args.agentview_key)
    if args.openwam_wrist_camera_name:
        aliases[str(args.openwam_wrist_camera_name)] = str(args.wrist_key)
    return aliases


def _action_array_stats(actions: np.ndarray) -> dict[str, Any]:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected action array [T,D], got {array.shape}.")
    if int(array.shape[0]) == 0:
        return {
            "steps": 0,
            "dim": int(array.shape[1]),
            "mean_l2": 0.0,
            "max_l2": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "temporal_delta_mean_l2": 0.0,
            "temporal_delta_max_l2": 0.0,
        }
    l2 = np.linalg.norm(array, axis=1)
    if int(array.shape[0]) > 1:
        temporal_delta = np.linalg.norm(np.diff(array, axis=0), axis=1)
        delta_mean = float(temporal_delta.mean())
        delta_max = float(temporal_delta.max())
    else:
        delta_mean = 0.0
        delta_max = 0.0
    return {
        "steps": int(array.shape[0]),
        "dim": int(array.shape[1]),
        "mean_l2": float(l2.mean()),
        "max_l2": float(l2.max()),
        "mean_abs": float(np.abs(array).mean()),
        "max_abs": float(np.abs(array).max()),
        "temporal_delta_mean_l2": delta_mean,
        "temporal_delta_max_l2": delta_max,
    }


def _candidate_first_chunk_action_diversity(candidates: Any) -> dict[str, Any]:
    chunks = [
        np.asarray(candidate.action_chunks[0].actions, dtype=np.float32)
        for candidate in candidates
        if getattr(candidate, "action_chunks", ())
    ]
    if not chunks:
        return {"candidate_count": 0}
    first_shape = chunks[0].shape
    if any(chunk.shape != first_shape for chunk in chunks):
        return {
            "candidate_count": len(chunks),
            "action_shape": list(first_shape),
            "pairwise_comparable": False,
        }
    stack = np.stack(chunks, axis=0)
    pairwise_step_l2: list[float] = []
    pairwise_chunk_l2: list[float] = []
    for left in range(int(stack.shape[0])):
        for right in range(left + 1, int(stack.shape[0])):
            step_diffs = np.linalg.norm(stack[left] - stack[right], axis=1)
            pairwise_step_l2.extend(float(value) for value in step_diffs)
            pairwise_chunk_l2.append(float(np.linalg.norm(stack[left] - stack[right])))
    if pairwise_step_l2:
        mean_step = float(np.mean(pairwise_step_l2))
        max_step = float(np.max(pairwise_step_l2))
        mean_chunk = float(np.mean(pairwise_chunk_l2))
        max_chunk = float(np.max(pairwise_chunk_l2))
    else:
        mean_step = max_step = mean_chunk = max_chunk = 0.0
    return {
        "candidate_count": int(stack.shape[0]),
        "action_shape": [int(dim) for dim in stack.shape[1:]],
        "pairwise_comparable": True,
        "mean_pairwise_step_l2": mean_step,
        "max_pairwise_step_l2": max_step,
        "mean_pairwise_chunk_l2": mean_chunk,
        "max_pairwise_chunk_l2": max_chunk,
        "per_candidate_mean_l2": [float(value) for value in np.linalg.norm(stack, axis=2).mean(axis=1)],
    }


def _dump_planner_candidate_actions(
    *,
    output_dir: Path,
    rollout_index: int,
    replan_index: int,
    result: Any,
) -> Path | None:
    chunks = [
        np.asarray(candidate.action_chunks[0].actions, dtype=np.float32)
        for candidate in result.candidates
        if getattr(candidate, "action_chunks", ())
    ]
    if not chunks:
        return None
    first_shape = chunks[0].shape
    if any(chunk.shape != first_shape for chunk in chunks):
        return None
    path = output_dir / f"candidate_actions_rollout{int(rollout_index):03d}_replan{int(replan_index):04d}.npz"
    scores = np.asarray([float(candidate.score) for candidate in result.candidates], dtype=np.float32)
    np.savez_compressed(
        path,
        first_action_chunks=np.stack(chunks, axis=0),
        scores=scores,
        selected_first_action_chunk=np.asarray(result.first_action_chunk.actions, dtype=np.float32),
    )
    return path


def _load_optional_action_trace(path_value: str | None) -> np.ndarray | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fixed candidate action trace does not exist: {path}")
    with np.load(path) as payload:
        if "actions" not in payload:
            raise KeyError(f"Action trace {path} does not contain an 'actions' array.")
        actions = np.asarray(payload["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Action trace {path} must have shape [T,D], got {actions.shape}.")
    return actions


def _save_executed_action_trace(
    *,
    output_dir: Path,
    rollout_index: int,
    actions: list[np.ndarray],
    task_id: int,
    episode_idx: int,
) -> Path | None:
    if not actions:
        return None
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Executed action trace must have shape [T,D], got {array.shape}.")
    path = output_dir / f"executed_actions_rollout_{int(rollout_index):03d}.npz"
    np.savez_compressed(
        path,
        actions=array,
        task_id=np.asarray(int(task_id), dtype=np.int64),
        episode_idx=np.asarray(int(episode_idx), dtype=np.int64),
    )
    return path


def _libero_noop_action(*, action_dim: int) -> np.ndarray:
    """Return the LIBERO reset-settling no-op used by LeRobot's LIBERO wrapper."""

    dim = int(action_dim)
    if dim <= 0:
        raise ValueError(f"action_dim must be positive, got {dim}.")
    action = np.zeros(dim, dtype=np.float32)
    if dim >= 7:
        action[6] = -1.0
    return action


def _resolve_rollout_schedule(args: argparse.Namespace) -> list[tuple[int, int]]:
    task_ids = _parse_optional_int_list(args.task_ids)
    episode_indices = _parse_optional_int_list(args.episode_indices)
    if task_ids is None and episode_indices is None:
        return [(int(args.task_id), int(args.episode_idx) + offset) for offset in range(int(args.episodes))]

    resolved_task_ids = task_ids if task_ids is not None else [int(args.task_id)]
    resolved_episode_indices = episode_indices if episode_indices is not None else [int(args.episode_idx)]
    return [(task_id, episode_idx) for episode_idx in resolved_episode_indices for task_id in resolved_task_ids]


def _parse_optional_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
    parsed = [int(item) for item in items if item]
    if not parsed:
        raise ValueError("Expected at least one integer in comma-separated list.")
    return parsed


def _load_local_env_file(path: Path) -> None:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _resolve_replay_init_state(args: argparse.Namespace, *, task_id: int, episode_idx: int) -> _ReplayInitState:
    if not bool(getattr(args, "use_replay_init_state", False)):
        return _ReplayInitState()
    replay_status_path = getattr(args, "replay_status_path", None)
    if replay_status_path is None:
        raise ValueError("--use-replay-init-state requires --replay-status-path.")
    path = Path(replay_status_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Replay-status file not found: {path}")
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("upstream_task_id", -1)) != int(task_id):
                continue
            if int(row.get("task_local_episode_idx", -1)) != int(episode_idx):
                continue
            matches.append(row)
    if not matches:
        raise ValueError(
            "No replay-status row matched "
            f"task_id={int(task_id)}, task_local_episode_idx={int(episode_idx)} in {path}."
        )
    row = matches[0]
    init_state = row.get("resolved_init_state_index")
    if init_state is None:
        init_state = row.get("primary_init_state_index")
    if init_state is None:
        raise ValueError(
            "Matched replay-status row does not contain resolved_init_state_index or primary_init_state_index: "
            f"task_id={int(task_id)}, task_local_episode_idx={int(episode_idx)}."
        )
    return _ReplayInitState(
        init_state_index=int(init_state),
        dataset_episode_index=(
            None if row.get("dataset_episode_index") is None else int(row["dataset_episode_index"])
        ),
    )


def _planning_context_from_observation(
    observation: SimulatorObservation,
    *,
    adapter: Any | None = None,
    task_id: int | None = None,
    episode_idx: int | None = None,
    seed: int | None = None,
) -> PlanningContext:
    views = {key: np.asarray(value, dtype=np.uint8) for key, value in observation.views.items()}
    fallback_state = None if observation.state is None else np.asarray(observation.state, dtype=np.float32)
    policy_state = _policy_state_from_observation(observation, fallback=fallback_state)
    openwam_proprio_state = _openwam_proprio_state_from_observation(observation, fallback=fallback_state)
    metadata: dict[str, Any] = {
        "observation_metadata": dict(observation.metadata or {}),
        "openwam_proprio_state": openwam_proprio_state,
    }
    if adapter is not None:
        metadata["libero_simulator_state"] = _get_libero_sim_state(adapter)
    if task_id is not None:
        metadata["libero_task_id"] = int(task_id)
    if episode_idx is not None:
        metadata["libero_episode_idx"] = int(episode_idx)
    if seed is not None:
        metadata["libero_seed"] = int(seed)
    if openwam_proprio_state is not None:
        metadata["sim_proprio_step_history"] = [np.asarray(openwam_proprio_state, dtype=np.float32).copy()]
    return PlanningContext(
        views=views,
        state=policy_state,
        task_text=observation.task_text,
        metadata=metadata,
    )


class _LiberoSimulatorProprioPropagator:
    """Advance branch proprio by replaying actions in a copied LIBERO state.

    This intentionally consumes only robot proprio from the simulator copy.
    Visual context remains the learned FDM prediction stored in `context.views`,
    so branch planning does not peek at rendered future environment images.
    """

    def __init__(self, *, adapter_factory: Callable[[], Any], action_dim: int) -> None:
        self.adapter_factory = adapter_factory
        self.action_dim = int(action_dim)
        self._branch_adapter: Any | None = None

    def begin_branch(self, context: PlanningContext) -> None:
        """Initialize one simulator copy for sequential propagation in a branch."""

        if self._branch_adapter is not None:
            raise RuntimeError("A LIBERO proprio propagation branch is already active.")
        metadata = dict(context.metadata)
        simulator_state = metadata.get("libero_simulator_state")
        task_id = metadata.get("libero_task_id")
        episode_idx = metadata.get("libero_episode_idx")
        rollout_seed = metadata.get("libero_seed")
        if simulator_state is None:
            raise ValueError("LIBERO proprio propagation requires context metadata 'libero_simulator_state'.")
        if task_id is None or episode_idx is None:
            raise ValueError(
                "LIBERO proprio propagation requires context metadata 'libero_task_id' and 'libero_episode_idx'."
            )
        adapter = self.adapter_factory()
        try:
            adapter.reset(
                EpisodeSpec(
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    seed=None if rollout_seed is None else int(rollout_seed),
                )
            )
            _restore_libero_sim_state(adapter, np.asarray(simulator_state, dtype=np.float64))
        except Exception:
            adapter.close()
            raise
        self._branch_adapter = adapter

    def end_branch(self) -> None:
        """Close the branch-local simulator copy, if one is active."""

        adapter = self._branch_adapter
        self._branch_adapter = None
        if adapter is not None:
            adapter.close()

    def propagate(
        self,
        context: PlanningContext,
        action_chunk: ActionChunk,
        *,
        seed: int | None = None,
    ) -> PlanningContext:
        del seed
        metadata = dict(context.metadata)
        simulator_state = metadata.get("libero_simulator_state")
        task_id = metadata.get("libero_task_id")
        episode_idx = metadata.get("libero_episode_idx")
        rollout_seed = metadata.get("libero_seed")
        if simulator_state is None:
            raise ValueError("LIBERO proprio propagation requires context metadata 'libero_simulator_state'.")
        if task_id is None or episode_idx is None:
            raise ValueError(
                "LIBERO proprio propagation requires context metadata 'libero_task_id' and 'libero_episode_idx'."
            )
        step_proprios: list[np.ndarray] = []
        observation: SimulatorObservation | None = None
        actions = np.asarray(action_chunk.actions, dtype=np.float32)
        if int(actions.shape[1]) != self.action_dim:
            raise ValueError(
                f"LIBERO proprio propagation expected action_dim={self.action_dim}, got {int(actions.shape[1])}."
            )
        branch_adapter = self._branch_adapter
        if branch_adapter is not None:
            for action in actions:
                transition = branch_adapter.step(action)
                observation = transition.observation
                fallback = None if observation.state is None else np.asarray(observation.state, dtype=np.float32)
                proprio = _openwam_proprio_state_from_observation(observation, fallback=fallback)
                if proprio is not None:
                    step_proprios.append(np.asarray(proprio, dtype=np.float32).copy())
            next_simulator_state = _get_libero_sim_state(branch_adapter)
            if observation is None:
                raise RuntimeError("LIBERO proprio propagation produced no observation.")
            return self._context_after_propagation(
                context,
                observation=observation,
                next_simulator_state=next_simulator_state,
                step_proprios=step_proprios,
                action_steps=int(action_chunk.actions.shape[0]),
            )
        adapter = self.adapter_factory()
        try:
            adapter.reset(
                EpisodeSpec(
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    seed=None if rollout_seed is None else int(rollout_seed),
                )
            )
            observation = _restore_libero_sim_state(adapter, np.asarray(simulator_state, dtype=np.float64))
            for action in actions:
                transition = adapter.step(action)
                observation = transition.observation
                fallback = None if observation.state is None else np.asarray(observation.state, dtype=np.float32)
                proprio = _openwam_proprio_state_from_observation(observation, fallback=fallback)
                if proprio is not None:
                    step_proprios.append(np.asarray(proprio, dtype=np.float32).copy())
            next_simulator_state = _get_libero_sim_state(adapter)
        finally:
            adapter.close()
        if observation is None:
            raise RuntimeError("LIBERO proprio propagation produced no observation.")
        return self._context_after_propagation(
            context,
            observation=observation,
            next_simulator_state=next_simulator_state,
            step_proprios=step_proprios,
            action_steps=int(action_chunk.actions.shape[0]),
        )

    def _context_after_propagation(
        self,
        context: PlanningContext,
        *,
        observation: SimulatorObservation,
        next_simulator_state: np.ndarray,
        step_proprios: list[np.ndarray],
        action_steps: int,
    ) -> PlanningContext:
        metadata = dict(context.metadata)
        fallback_state = None if observation.state is None else np.asarray(observation.state, dtype=np.float32)
        next_proprio = _openwam_proprio_state_from_observation(observation, fallback=fallback_state)
        next_policy_state = _policy_state_from_observation(observation, fallback=fallback_state)
        next_openwam_proprio = None if next_proprio is None else np.asarray(next_proprio, dtype=np.float32)
        previous_history = list(metadata.get("sim_proprio_step_history") or [])
        history = previous_history + step_proprios
        propagated_steps = int(metadata.get("sim_propagated_action_steps", 0)) + int(action_steps)
        return context.with_prediction(
            predicted_video=context.predicted_video,
            state=next_policy_state,
            metadata_updates={
                "openwam_proprio_state": next_openwam_proprio,
                "libero_simulator_state": next_simulator_state,
                "sim_proprio_step_history": history,
                "sim_propagated_action_steps": propagated_steps,
                "sim_propagator_debug": {
                    "propagated_action_steps": int(action_steps),
                    "policy_state_dim": None if next_policy_state is None else int(next_policy_state.shape[0]),
                    "openwam_proprio_dim": None
                    if next_openwam_proprio is None
                    else int(next_openwam_proprio.shape[0]),
                },
            },
        )


def _get_libero_sim_state(adapter: Any) -> np.ndarray:
    env = getattr(adapter, "_env", None)
    if env is None:
        raise RuntimeError("LIBERO adapter must be reset before reading simulator state.")
    if hasattr(env, "get_sim_state"):
        return np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()


def _restore_libero_sim_state(adapter: Any, simulator_state: np.ndarray) -> SimulatorObservation:
    env = getattr(adapter, "_env", None)
    if env is None:
        raise RuntimeError("LIBERO adapter must be reset before restoring simulator state.")
    state = np.asarray(simulator_state, dtype=np.float64).copy()
    if hasattr(env, "regenerate_obs_from_state"):
        obs = env.regenerate_obs_from_state(state)
    else:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        obs = env._get_observations()
    adapter._last_obs = obs
    return adapter._normalize_observation(obs)


def _concatenate_predicted_videos(predicted_videos: Any) -> np.ndarray | None:
    chunks = [np.asarray(video, dtype=np.uint8) for video in predicted_videos if video is not None]
    if not chunks:
        return None
    if any(chunk.ndim != 4 for chunk in chunks):
        raise ValueError(f"Predicted videos must have shape [T,H,W,3], got {[chunk.shape for chunk in chunks]}.")
    return np.concatenate(chunks, axis=0)


def _candidate_debug_video(
    candidate: CandidateTrajectory,
    *,
    dynamics: Any | None,
    max_frames: int | None = None,
) -> np.ndarray | None:
    if max_frames is not None and int(max_frames) <= 0:
        return None
    stitched = _decode_stitched_openwam_candidate_video(candidate, dynamics=dynamics, max_frames=max_frames)
    if stitched is not None:
        return stitched
    video = _concatenate_predicted_videos(candidate.predicted_videos)
    if video is None:
        return None
    if max_frames is not None:
        video = video[: int(max(0, max_frames))]
    return video


def _decode_stitched_openwam_candidate_video(
    candidate: CandidateTrajectory,
    *,
    dynamics: Any | None,
    max_frames: int | None,
) -> np.ndarray | None:
    if not isinstance(dynamics, OpenWamActionConditionedFdm):
        return None
    latents_key = _openwam_predicted_latents_key(dynamics)
    if latents_key is None:
        return None
    latent_chunks = tuple(candidate.metadata.get(latents_key, ()) if candidate.metadata else ())
    if not latent_chunks:
        return None
    try:
        import torch

        tensors = []
        for latent_chunk in latent_chunks:
            tensor = latent_chunk if isinstance(latent_chunk, torch.Tensor) else torch.as_tensor(latent_chunk)
            if tensor.ndim != 5:
                raise ValueError(f"Expected OpenWAM FDM latent chunk [B,C,F,H,W], got {tuple(tensor.shape)}.")
            tensors.append(tensor.detach().cpu())
        latents = torch.cat(tensors, dim=2)
        video = dynamics.decode_latent_sequence_for_video(
            candidate.context,
            latents,
            max_frames=None if max_frames is None else int(max(0, max_frames)),
        )
        return None if video is None else np.asarray(video, dtype=np.uint8)
    except Exception as exc:
        print(
            "[warn] failed to decode stitched OpenWAM FDM latents for debug video; "
            f"falling back to per-chunk decoded videos: {_exception_summary(exc)}",
            flush=True,
        )
        return None


def _openwam_predicted_latents_key(dynamics: Any) -> str | None:
    if not isinstance(dynamics, OpenWamActionConditionedFdm):
        return None
    keys = getattr(dynamics, "keys", None)
    key = getattr(keys, "predicted_latents", None)
    return None if key is None else str(key)


def _normalize_prefix_video(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot save prefix+imagined video without at least one real prefix frame.")
    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    first = arrays[0]
    if first.ndim != 3 or int(first.shape[-1]) != 3:
        raise ValueError(f"Expected real prefix frame [H,W,3], got {first.shape}.")
    normalized = []
    for frame in arrays:
        if frame.ndim != 3 or int(frame.shape[-1]) != 3:
            raise ValueError(f"Expected real prefix frame [H,W,3], got {frame.shape}.")
        if frame.shape != first.shape:
            frame = _resize_rgb_nearest(frame, height=int(first.shape[0]), width=int(first.shape[1]))
        normalized.append(np.ascontiguousarray(frame))
    return np.stack(normalized, axis=0)


def _resize_video_to_frame_shape(video: np.ndarray, *, target_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(video, dtype=np.uint8)
    if array.ndim != 4 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {array.shape}.")
    target_h, target_w, target_c = (int(value) for value in target_shape)
    if target_c != 3:
        raise ValueError(f"Expected target RGB shape [H,W,3], got {target_shape}.")
    if int(array.shape[1]) == target_h and int(array.shape[2]) == target_w:
        return np.ascontiguousarray(array)
    return np.stack(
        [_resize_rgb_nearest(frame, height=target_h, width=target_w) for frame in array],
        axis=0,
    )


def _context_proprio_history(context: PlanningContext) -> np.ndarray:
    history = list(context.metadata.get("sim_proprio_step_history") or [])
    if not history:
        proprio = context.metadata.get("openwam_proprio_state")
        if proprio is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(proprio, dtype=np.float32).reshape(1, -1)
    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    dim = int(arrays[0].shape[0])
    if any(int(array.shape[0]) != dim for array in arrays):
        raise ValueError("Context proprio history contains mixed state dimensions.")
    return np.stack(arrays, axis=0)


def _planning_context_canvas(
    context: PlanningContext,
    *,
    args: argparse.Namespace,
    view_transform: str | None = "none",
) -> np.ndarray:
    view_keys = [str(args.agentview_key)]
    wrist_key = str(args.wrist_key)
    if wrist_key in context.views and wrist_key != view_keys[0]:
        view_keys.append(wrist_key)
    frames: list[np.ndarray] = []
    for key in view_keys:
        if key not in context.views:
            continue
        frames.append(
            _resize_rgb_nearest(
                _transform_planning_rgb_frame(
                    np.asarray(context.views[key], dtype=np.uint8),
                    transform=view_transform,
                ),
                height=int(args.camera_height),
                width=int(args.camera_width),
            )
        )
    if not frames:
        raise ValueError("Cannot build planning context canvas: no configured views are present.")
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def _prepend_context_canvas_to_video(context_canvas: np.ndarray, predicted_video: np.ndarray) -> np.ndarray:
    video = np.asarray(predicted_video, dtype=np.uint8)
    context = np.asarray(context_canvas, dtype=np.uint8)
    if video.ndim != 4 or int(video.shape[-1]) != 3:
        raise ValueError(f"Expected predicted video [T,H,W,3], got {video.shape}.")
    if context.ndim != 3 or int(context.shape[-1]) != 3:
        raise ValueError(f"Expected context canvas [H,W,3], got {context.shape}.")
    if context.shape != video.shape[1:]:
        context = _resize_rgb_nearest(context, height=int(video.shape[1]), width=int(video.shape[2]))
    return np.concatenate([context[None], video], axis=0)


def _transform_planning_rgb_frame(frame: np.ndarray, *, transform: str | None) -> np.ndarray:
    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3 or int(value.shape[-1]) != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {value.shape}.")
    if transform in {None, "none", "identity"}:
        return value
    if transform in {"vertical_flip", "flip_ud"}:
        return np.ascontiguousarray(value[::-1])
    raise ValueError(f"Unsupported planning RGB transform {transform!r}.")


def _openwam_proprio_state_from_observation(
    observation: SimulatorObservation,
    *,
    fallback: np.ndarray | None,
) -> np.ndarray | None:
    """Return the proprio vector expected by Open-WAM GJD FDM when raw LIBERO state is available."""

    raw = observation.raw
    if isinstance(raw, Mapping) and {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }.issubset(raw):
        return _libero_eef_axisangle_gripper_2d_state(raw)
    return fallback


def _policy_state_from_observation(
    observation: SimulatorObservation,
    *,
    fallback: np.ndarray | None,
) -> np.ndarray | None:
    """Return the raw state expected by the action policy adapter.

    LeRobot's LIBERO PI0-family processors consume EEF position, EEF rotation
    as axis-angle, and gripper qpos. The shared LIBERO simulator adapter exposes
    joint qpos as its generic `SimulatorObservation.state`, so derive the
    policy-facing state from raw LIBERO fields whenever possible.
    """

    raw = observation.raw
    if isinstance(raw, Mapping) and {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }.issubset(raw):
        return _libero_eef_axisangle_gripper_2d_state(raw)
    return fallback


def _libero_eef_axisangle_gripper_2d_state(raw: Mapping[str, Any]) -> np.ndarray:
    eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] != 3:
        raise ValueError(f"Expected LIBERO robot0_eef_pos dim 3, got {eef_pos.shape[0]}.")
    if eef_quat.shape[0] != 4:
        raise ValueError(f"Expected LIBERO robot0_eef_quat dim 4, got {eef_quat.shape[0]}.")
    if gripper_qpos.shape[0] != 2:
        raise ValueError(f"Expected LIBERO robot0_gripper_qpos dim 2, got {gripper_qpos.shape[0]}.")

    import torch

    from open_wam.data.action_transforms import quaternion_to_axis_angle

    axis_angle = (
        quaternion_to_axis_angle(torch.from_numpy(eef_quat).to(dtype=torch.float32).unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=0).astype(np.float32, copy=False)


def _build_policy(args: argparse.Namespace) -> Any:
    if args.policy == "gaussian":
        return GaussianActionSampler(action_dim=args.action_dim, std=args.gaussian_std, clip=args.action_clip)
    if args.policy in {"pi0fast", "pi0"}:
        model_path = args.pi0_model if args.policy == "pi0" else args.pi0fast_model
        return Pi0FastPolicySampler.from_pretrained(
            model_path,
            batch_mapping=Pi0FastBatchMapping(
                image_keys={
                    args.pi0fast_base_image_key: args.agentview_key,
                    args.pi0fast_wrist_image_key: args.wrist_key,
                },
                image_transform=args.pi0fast_image_transform,
                state_key=args.pi0fast_state_key,
                task_key=args.pi0fast_task_key,
                device=args.policy_device,
                state_dim=args.pi0fast_state_dim,
            ),
            policy_family=args.policy,
            action_clip=args.action_clip,
            short_horizon_strategy=args.pi0fast_short_horizon_strategy,
            action_selection_mode=args.pi0fast_action_selection_mode,
            compile_model=args.policy_compile_model,
        )
    raise ValueError(f"Unsupported policy: {args.policy}")


def _build_dynamics(args: argparse.Namespace) -> Any:
    if args.dynamics == "dummy":
        return ActionTintDynamics(view_key=args.agentview_key, video_frames=args.fdm_video_frames)
    if args.dynamics == "openwam_gjd":
        return _build_openwam_gjd_dynamics(args)
    if args.dynamics == "uva_libero_fdm":
        return _build_uva_libero_dynamics(args)
    raise ValueError(f"Unsupported dynamics: {args.dynamics}")


def _build_openwam_gjd_dynamics(args: argparse.Namespace) -> OpenWamActionConditionedFdm:
    if args.openwam_checkpoint is None:
        raise ValueError("--openwam-checkpoint is required when --dynamics openwam_gjd.")
    import torch

    from open_wam.ablations.joint_denoising_fdm.cli import (
        _build_fdm_rollout_for_config,
        _repair_runtime_config_for_local_eval,
        _resolve_runtime_dtype,
    )
    from open_wam.ablations.joint_denoising_fdm.types import FdmAblationMode
    from open_wam.ablations.joint_denoising_fdm.visualization import decode_latent_video
    from open_wam.utils import (
        load_experiment_config,
        merge_runtime_config_from_checkpoint,
        resolve_checkpoint_file,
        resolve_transformer_dir_override,
    )
    from open_wam.utils.config_overrides import apply_config_overrides, parse_override_assignments

    config_path = Path(args.openwam_config).expanduser().resolve()
    checkpoint_file = resolve_checkpoint_file(args.openwam_checkpoint)
    checkpoint_dir = checkpoint_file.parent
    transformer_dir = resolve_transformer_dir_override(checkpoint_dir)
    base_config = load_experiment_config(config_path)
    config, _ = merge_runtime_config_from_checkpoint(base_config, checkpoint_file)
    config = _repair_runtime_config_for_local_eval(
        config=config,
        base_config=base_config,
        transformer_dir=transformer_dir,
        dataset_root=args.dataset_root,
        empty_text_embedding_path=args.empty_text_embedding_path,
        reference_assets_device_policy=args.reference_assets_device_policy,
        video_num_inference_steps=args.video_num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
    )
    if args.set_overrides:
        config = apply_config_overrides(config, parse_override_assignments(tuple(args.set_overrides)))

    runtime_device = torch.device(args.runtime_device)
    decode_device = torch.device(args.decode_device or args.runtime_device)
    rollout = _build_fdm_rollout_for_config(
        config=config,
        checkpoint_file=checkpoint_file,
        runtime_device=runtime_device,
        runtime_dtype=_resolve_runtime_dtype(args.runtime_dtype),
    )

    def _decode(latents):
        return decode_latent_video(rollout.runner.pipeline, latents, decode_device=decode_device)

    return OpenWamActionConditionedFdm(
        rollout=rollout,
        mode=FdmAblationMode(args.openwam_fdm_mode),
        decode_video_fn=_decode,
        predicted_view_key=args.openwam_predicted_view_key,
        drop_text_conditioning=bool(args.openwam_drop_text_conditioning),
        project_prediction_to_context_view=bool(args.openwam_project_prediction_to_context_view),
        return_canonical_prediction_video=bool(args.openwam_return_canonical_prediction_video),
        return_context_canvas_prediction_video=bool(args.openwam_return_context_canvas_prediction_video),
        context_canvas_view_keys=(str(args.agentview_key), str(args.wrist_key)),
        predicted_view_aliases=_openwam_view_aliases(args),
        input_view_transform=args.openwam_input_view_transform,
    )


def _build_uva_libero_dynamics(args: argparse.Namespace) -> UvaLiberoActionConditionedFdm:
    return UvaLiberoActionConditionedFdm(
        uva_root=args.uva_root,
        checkpoint=args.uva_checkpoint,
        dataset_dir=args.uva_dataset_dir,
        output_dir=Path(args.uva_runtime_output_dir or (Path(args.output_dir).expanduser() / "_uva_runtime")),
        runtime_device=args.runtime_device,
        view_key=args.agentview_key,
        action_space=args.uva_action_space,
        short_action_horizon_strategy=args.uva_short_action_horizon_strategy,
        predicted_view_key=args.uva_predicted_view_key,
    )


def _build_evaluator(args: argparse.Namespace, *, output_dir: Path) -> Any:
    if args.evaluator == "constant":
        return ConstantEvaluator()
    if args.evaluator == "action_magnitude":
        return ActionMagnitudeEvaluator(action_l2_weight=args.action_l2_weight)
    if args.evaluator in {"goal_image_l2", "demo_future_l2"}:
        return GoalImageL2Evaluator(weight=args.goal_image_weight)
    if args.evaluator == "goal_delta_alignment":
        return GoalDeltaAlignmentEvaluator(
            alignment_weight=args.goal_delta_alignment_weight,
            background_penalty_weight=args.goal_delta_background_penalty_weight,
            change_threshold=args.goal_delta_change_threshold,
        )
    if args.evaluator == "gemini_vlm":
        if not os.environ.get(args.gemini_api_key_env):
            raise ValueError(
                f"--evaluator gemini_vlm requires an API key in ${args.gemini_api_key_env}. "
                "Set the env var before launching so the runner fails before expensive FDM inference."
            )
        return GeminiVlmCandidateRerankEvaluator(
            GeminiVlmRerankConfig(
                output_dir=output_dir / "gemini_vlm",
                api_key_env=args.gemini_api_key_env,
                model=args.gemini_model,
                video_fps=args.gemini_candidate_video_fps,
                request_timeout_seconds=args.gemini_timeout_seconds,
                max_candidates=args.gemini_max_candidates,
                delete_uploaded_files=args.gemini_delete_uploaded_files,
                policy_prior_hint_mode=args.gemini_prior_hint_mode,
            )
        )
    raise ValueError(f"Unsupported evaluator: {args.evaluator}")


def _validate_planner_runtime_contract(args: argparse.Namespace, dynamics: Any) -> None:
    planner_mode = str(getattr(args, "planner", "baseline"))
    future_temperature = getattr(args, "full_imagined_future_policy_temperature", None)
    if future_temperature is not None and float(future_temperature) < 0.0:
        raise ValueError("--full-imagined-future-policy-temperature must be non-negative.")
    if planner_mode == "imagined_open_loop":
        if int(args.min_plan_action_steps) < int(args.max_env_steps):
            raise ValueError(
                f"--planner {planner_mode} generates full imagined trajectories. "
                "Set --min-plan-action-steps >= --max-env-steps so planning covers the requested horizon."
            )
        if int(args.max_plan_chunks) * int(args.chunk_action_steps) < int(args.max_env_steps):
            raise ValueError(
                f"--planner {planner_mode} has insufficient chunk budget: "
                f"max_plan_chunks * chunk_action_steps = "
                f"{int(args.max_plan_chunks) * int(args.chunk_action_steps)} < max_env_steps={int(args.max_env_steps)}."
            )
    if isinstance(dynamics, UvaLiberoActionConditionedFdm):
        if int(args.chunk_action_steps) > int(dynamics.internal_horizon - dynamics.action_target_start):
            raise ValueError(
                "--dynamics uva_libero_fdm cannot score policy chunks longer than its UVA target slots: "
                f"chunk_action_steps={int(args.chunk_action_steps)}, "
                f"target_slots={int(dynamics.internal_horizon - dynamics.action_target_start)}."
            )
        if int(args.max_plan_chunks) != 1:
            raise ValueError(
                "--dynamics uva_libero_fdm currently supports one-chunk selector diagnostics only. "
                "Set --max-plan-chunks 1 so repeated-current UVA context is not compounded."
            )
        if int(args.execute_action_steps) > int(args.chunk_action_steps):
            raise ValueError(
                "--dynamics uva_libero_fdm requires --execute-action-steps <= --chunk-action-steps."
            )
        return
    if not isinstance(dynamics, OpenWamActionConditionedFdm):
        return
    expected_actions = dynamics.expected_action_steps_per_chunk()
    chunk_action_steps = int(args.chunk_action_steps)
    execute_action_steps = int(args.execute_action_steps)
    if planner_mode == "imagined_open_loop" and chunk_action_steps != expected_actions:
        raise ValueError(
            f"--planner {planner_mode} with Open-WAM FDM requires the planner action chunk to match "
            "the FDM chunk exactly so video cache, action horizon, and propagated proprio advance together: "
            f"expected {expected_actions}, got {chunk_action_steps}."
        )
    if chunk_action_steps > expected_actions:
        raise ValueError(
            "--dynamics openwam_gjd cannot score policy chunks longer than the Open-WAM FDM chunk horizon: "
            f"frame_chunk_size={int(dynamics.rollout.frame_chunk_size)}, "
            f"action_per_frame={int(dynamics.rollout.action_per_frame)}, expected {expected_actions}, "
            f"got {chunk_action_steps}."
        )
    if execute_action_steps > chunk_action_steps:
        raise ValueError(
            "--dynamics openwam_gjd requires --execute-action-steps <= --chunk-action-steps "
            "so execution never consumes internally padded FDM-only actions, "
            f"got execute={execute_action_steps}, chunk={chunk_action_steps}."
        )
    if chunk_action_steps < expected_actions:
        if planner_mode == "full_imagined_vlm_receding":
            if not bool(getattr(dynamics, "crop_decoded_video_to_action_steps", False)):
                raise ValueError(
                    "--planner full_imagined_vlm_receding with a shorter Open-WAM policy chunk requires "
                    "cropping decoded FDM videos to the real action horizon."
                )
            return
        if int(args.max_plan_chunks) != 1:
            raise ValueError(
                "Open-WAM FDM can internally pad a shorter policy chunk for one-chunk receding-horizon ranking, "
                "but multi-chunk imagined planning would advance the FDM cache beyond the policy/control horizon. "
                f"Set --max-plan-chunks 1 for chunk_action_steps={chunk_action_steps} < FDM horizon {expected_actions}."
            )
        if int(args.min_plan_action_steps) > chunk_action_steps:
            raise ValueError(
                "When Open-WAM FDM pads a shorter policy chunk, --min-plan-action-steps must fit in the real "
                f"policy chunk. Set --min-plan-action-steps <= {chunk_action_steps}; "
                f"got {int(args.min_plan_action_steps)}."
            )


def _build_goal_provider(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> Callable[..., np.ndarray] | None:
    if args.evaluator not in {"goal_image_l2", "goal_delta_alignment", "demo_future_l2", "gemini_vlm"}:
        return None
    if args.goal_image is not None:
        goal = np.asarray(imageio.imread(Path(args.goal_image).expanduser()), dtype=np.uint8)

        def _fixed_goal(**_: Any) -> np.ndarray:
            return goal

        return _fixed_goal
    dataset_root = args.goal_image_dataset_root or args.dataset_root
    if dataset_root is None:
        if args.evaluator == "gemini_vlm":
            return None
        raise ValueError(
            "--evaluator goal_image_l2 requires --goal-image or --goal-image-dataset-root "
            "(or --dataset-root when using Open-WAM FDM)."
        )
    resolver = _LeRobotFinalFrameGoalProvider(
        dataset_root=Path(dataset_root).expanduser(),
        camera_name=args.goal_image_camera_name,
        frame_index=int(args.goal_image_frame_index),
        output_dir=output_dir,
        reference_policy=getattr(args, "goal_image_reference_policy", "matched_episode"),
        reference_episode_offset=int(getattr(args, "goal_image_reference_episode_offset", 1)),
    )
    return resolver


def _planner_goal_for_observation(
    args: argparse.Namespace,
    observation: SimulatorObservation,
    target_goal: np.ndarray | None,
) -> Any | None:
    if target_goal is None:
        return None
    if args.evaluator == "goal_delta_alignment":
        view_keys = _resolve_goal_current_view_keys(args)
        target = np.asarray(target_goal, dtype=np.uint8)
        if bool(getattr(args, "openwam_return_context_canvas_prediction_video", False)):
            target = _resize_rgb_nearest(
                target,
                height=int(args.camera_height),
                width=int(args.camera_width) * len(view_keys),
            )
        return {
            "current": _current_canvas_from_observation(
                observation,
                target_shape=target.shape,
                view_keys=view_keys,
            ),
            "target": target,
        }
    if args.evaluator == "gemini_vlm":
        goal: dict[str, Any] = {
            "task_text": observation.task_text,
            "demo_video_path": args.gemini_demo_video,
        }
        if target_goal is not None:
            goal["target"] = target_goal
            goal["current"] = _current_canvas_from_observation(
                observation,
                target_shape=np.asarray(target_goal).shape,
                view_keys=_resolve_goal_current_view_keys(args),
            )
        return goal
    return target_goal


def _resolve_goal_current_view_keys(args: argparse.Namespace) -> tuple[str, ...]:
    if args.goal_image_current_view_keys:
        return _parse_camera_names(args.goal_image_current_view_keys)
    target_cameras = _parse_camera_names(args.goal_image_camera_name)
    if len(target_cameras) <= 1:
        return (str(args.agentview_key),)
    return (str(args.agentview_key), str(args.wrist_key))


def _current_canvas_from_observation(
    observation: SimulatorObservation,
    *,
    target_shape: tuple[int, ...],
    view_keys: tuple[str, ...],
) -> np.ndarray:
    if len(target_shape) != 3 or int(target_shape[-1]) != 3:
        raise ValueError(f"Target goal canvas must have shape [H,W,3], got {target_shape}.")
    if not view_keys:
        raise ValueError("At least one current view key is required.")
    target_height = int(target_shape[0])
    target_width = int(target_shape[1])
    if target_width % len(view_keys) != 0:
        raise ValueError(
            "Target goal canvas width must be divisible by the number of current view keys, "
            f"got width={target_width}, view_keys={view_keys}."
        )
    per_view_width = target_width // len(view_keys)
    frames: list[np.ndarray] = []
    for key in view_keys:
        if key not in observation.views:
            raise KeyError(f"Observation has no view {key!r}; available={sorted(observation.views)}.")
        frames.append(
            _resize_rgb_nearest(
                np.asarray(observation.views[key], dtype=np.uint8),
                height=target_height,
                width=per_view_width,
            )
        )
    return np.concatenate(frames, axis=1)


def _resize_rgb_nearest(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}.")
    if int(array.shape[0]) == int(height) and int(array.shape[1]) == int(width):
        return array
    y_indices = np.linspace(0, int(array.shape[0]) - 1, int(height)).round().astype(np.int64)
    x_indices = np.linspace(0, int(array.shape[1]) - 1, int(width)).round().astype(np.int64)
    return np.ascontiguousarray(array[y_indices][:, x_indices])


class _LeRobotFinalFrameGoalProvider:
    def __init__(
        self,
        *,
        dataset_root: Path,
        camera_name: str,
        frame_index: int,
        output_dir: Path,
        reference_policy: str = "matched_episode",
        reference_episode_offset: int = 1,
    ) -> None:
        self.dataset_root = dataset_root
        self.camera_names = _parse_camera_names(camera_name)
        self.frame_index = int(frame_index)
        self.output_dir = output_dir
        self.reference_policy = str(reference_policy)
        self.reference_episode_offset = int(reference_episode_offset)
        if self.reference_policy not in {"matched_episode", "next_episode"}:
            raise ValueError(
                "reference_policy must be one of {'matched_episode', 'next_episode'}, "
                f"got {self.reference_policy!r}."
            )
        self._episodes_by_task = _load_lerobot_episodes_by_task(dataset_root)
        self._frame_cache: dict[tuple[int, str, int], np.ndarray] = {}

    def __call__(
        self,
        *,
        task_id: int,
        episode_idx: int,
        task_text: str | None,
        rollout_index: int,
        frame_index: int | None = None,
    ) -> np.ndarray:
        if not task_text:
            raise ValueError("Dataset goal-image resolution requires a rollout task text.")
        episodes = self._episodes_by_task.get(str(task_text))
        if not episodes:
            raise ValueError(f"No LeRobot dataset episodes found for task text: {task_text!r}.")
        source_local_index = int(episode_idx)
        if source_local_index < 0 or source_local_index >= len(episodes):
            raise ValueError(
                f"Episode index {episode_idx} is out of range for task {task_text!r}; "
                f"available dataset episodes: {len(episodes)}."
            )
        local_index = self._resolve_reference_local_index(
            source_local_index=source_local_index,
            episode_count=len(episodes),
        )
        dataset_episode_index = int(episodes[local_index]["episode_index"])
        resolved_frame_index = self.frame_index if frame_index is None else int(frame_index)
        frames = [
            self._read_cached_frame(
                dataset_episode_index=dataset_episode_index,
                camera_name=camera_name,
                frame_index=resolved_frame_index,
            )
            for camera_name in self.camera_names
        ]
        goal = _concatenate_goal_frames(frames)
        reference_suffix = "" if local_index == source_local_index else f"_refep{int(local_index)}"
        goal_path = (
            self.output_dir
            / f"goal_rollout_{int(rollout_index):03d}_task{int(task_id)}_ep{int(episode_idx)}{reference_suffix}.png"
        )
        imageio.imwrite(goal_path, goal)
        return goal

    def _resolve_reference_local_index(self, *, source_local_index: int, episode_count: int) -> int:
        if self.reference_policy == "matched_episode":
            return int(source_local_index)
        if episode_count < 2:
            raise ValueError(
                "goal-image reference_policy=next_episode requires at least two successful "
                "dataset episodes for the task."
            )
        offset = int(self.reference_episode_offset) % int(episode_count)
        if offset == 0:
            offset = 1
        local_index = (int(source_local_index) + offset) % int(episode_count)
        if local_index == int(source_local_index):
            local_index = (local_index + 1) % int(episode_count)
        return int(local_index)

    def _read_cached_frame(self, *, dataset_episode_index: int, camera_name: str, frame_index: int) -> np.ndarray:
        cache_key = (int(dataset_episode_index), str(camera_name), int(frame_index))
        if cache_key not in self._frame_cache:
            self._frame_cache[cache_key] = _read_video_frame(
                self.dataset_root
                / "videos"
                / "chunk-000"
                / camera_name
                / f"episode_{dataset_episode_index:06d}.mp4",
                frame_index=int(frame_index),
            )
        return self._frame_cache[cache_key]


def _parse_camera_names(camera_name: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(camera_name).split(",") if item.strip())
    if not names:
        raise ValueError("At least one goal-image camera name is required.")
    return names


def _concatenate_goal_frames(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("At least one goal frame is required.")
    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if len(arrays) == 1:
        return arrays[0]
    first_shape = arrays[0].shape
    for index, array in enumerate(arrays):
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Goal frame {index} must have shape [H,W,3], got {array.shape}.")
        if array.shape[0] != first_shape[0] or array.shape[2] != first_shape[2]:
            raise ValueError(
                "All goal frames must have matching height/channels for horizontal concatenation, "
                f"got first={first_shape}, frame{index}={array.shape}."
            )
    return np.concatenate(arrays, axis=1)


def _load_lerobot_episodes_by_task(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"LeRobot episodes metadata not found: {episodes_path}")
    episodes_by_task: dict[str, list[dict[str, Any]]] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for task in record.get("tasks", []):
                episodes_by_task.setdefault(str(task), []).append(record)
    for records in episodes_by_task.values():
        records.sort(key=lambda item: int(item["episode_index"]))
    return episodes_by_task


def _read_video_frame(video_path: Path, *, frame_index: int) -> np.ndarray:
    if not video_path.is_file():
        raise FileNotFoundError(f"Goal-image video not found: {video_path}")
    reader = imageio.get_reader(video_path)
    try:
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = None
        if int(frame_index) < 0:
            if frame_count is not None:
                resolved_index = frame_count + int(frame_index)
            else:
                frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
                if not frames:
                    raise ValueError(f"Goal-image video has no frames: {video_path}")
                return frames[int(frame_index)]
        else:
            resolved_index = int(frame_index)
        if frame_count is not None:
            resolved_index = int(np.clip(resolved_index, 0, frame_count - 1))
        return np.asarray(reader.get_data(resolved_index), dtype=np.uint8)
    finally:
        reader.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=("baseline", "fdm_guided", "imagined_open_loop", "full_imagined_vlm_receding", "replay_actions"),
        default="baseline",
    )
    parser.add_argument("--policy", choices=("gaussian", "pi0fast", "pi0"), default="gaussian")
    parser.add_argument("--dynamics", choices=("dummy", "openwam_gjd", "uva_libero_fdm"), default="dummy")
    parser.add_argument(
        "--evaluator",
        choices=(
            "constant",
            "action_magnitude",
            "goal_image_l2",
            "goal_delta_alignment",
            "demo_future_l2",
            "gemini_vlm",
        ),
        default="constant",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/libero_fdm_guided_planning")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument(
        "--use-replay-init-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resolve LIBERO init_state_index from replay-status metadata using "
            "(upstream_task_id, task_local_episode_idx). Use this when comparing "
            "against the LeRobot training/demo episode with the same task-local index."
        ),
    )
    parser.add_argument(
        "--replay-status-path",
        default=None,
        help="Replay-status JSONL used by --use-replay-init-state.",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task ids. When set, the runner sweeps these ids with --episode-indices.",
    )
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Optional comma-separated episode/init-state indices. When set, the runner sweeps these with --task-ids.",
    )
    parser.add_argument("--max-env-steps", type=int, default=300)
    parser.add_argument(
        "--action-trace",
        default=None,
        help="NPZ trace containing an `actions` array, used by --planner replay_actions.",
    )
    parser.add_argument(
        "--replay-continue-after-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When replaying a fixed action trace, keep rendering after the task success condition is reached.",
    )
    parser.add_argument(
        "--startup-noop-steps",
        type=int,
        default=0,
        help=(
            "Optional LIBERO reset-settling no-op steps before policy control. "
            "LeRobot's pi0-fast LIBERO wrapper defaults to 10."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-policy-samples", type=int, default=4)
    parser.add_argument(
        "--fixed-candidate-action-trace",
        default=None,
        help=(
            "Optional NPZ with an `actions` array [T,D]. In full-imagined mode, the trace slice starting at the "
            "current env step is added as a normal candidate. This is a debugging/control hook for verifying "
            "the planner when the candidate set is known to contain a successful chunk."
        ),
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--chunk-action-steps", type=int, default=10)
    parser.add_argument("--min-plan-action-steps", type=int, default=30)
    parser.add_argument("--max-plan-chunks", type=int, default=3)
    parser.add_argument("--execute-action-steps", type=int, default=10)
    parser.add_argument(
        "--max-replans-per-rollout",
        type=int,
        default=None,
        help=(
            "Debug cap for receding-horizon rollouts. When set, stop after this many replans even if the "
            "environment horizon has not been reached."
        ),
    )
    parser.add_argument("--policy-temperature", type=float, default=0.8)
    parser.add_argument(
        "--include-policy-prior-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include one deterministic policy-prior candidate alongside sampled candidates.",
    )
    parser.add_argument(
        "--policy-prior-temperature",
        type=float,
        default=0.0,
        help="Temperature for the optional policy-prior candidate.",
    )
    parser.add_argument(
        "--policy-prior-abstain-margin",
        type=float,
        default=0.0,
        help=(
            "Margin used by the generic planner to keep the policy prior when it is close to the best candidate. "
            "Full-imagined VLM mode records the value but lets the VLM scores decide."
        ),
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--gaussian-std", type=float, default=0.2)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--fdm-video-frames", type=int, default=4)
    parser.add_argument("--goal-image", default=None)
    parser.add_argument(
        "--goal-image-dataset-root",
        default=None,
        help=(
            "Optional LeRobot dataset root used by --evaluator goal_image_l2. "
            "The runner resolves the rollout task text and episode index to a dataset video final frame."
        ),
    )
    parser.add_argument(
        "--goal-image-camera-name",
        default="observation.images.agentview_rgb,observation.images.eye_in_hand_rgb",
        help=(
            "Dataset camera name or comma-separated camera names for goal-image L2. "
            "Multiple cameras are concatenated horizontally to match Open-WAM's canonical canvas."
        ),
    )
    parser.add_argument(
        "--goal-image-current-view-keys",
        default=None,
        help=(
            "Optional comma-separated live observation view keys used by goal_delta_alignment. "
            "Defaults to --agentview-key and --wrist-key for two-camera goals."
        ),
    )
    parser.add_argument(
        "--goal-image-frame-index",
        type=int,
        default=-1,
        help="Frame index within the selected reference dataset video. -1 selects the final frame.",
    )
    parser.add_argument(
        "--goal-image-reference-policy",
        choices=("matched_episode", "next_episode"),
        default="matched_episode",
        help=(
            "Which successful demo supplies the visual goal. Use next_episode for "
            "paper-facing selector diagnostics so the goal image is not from the same rollout."
        ),
    )
    parser.add_argument(
        "--goal-image-reference-episode-offset",
        type=int,
        default=1,
        help="Task-local episode offset used by --goal-image-reference-policy next_episode.",
    )
    parser.add_argument("--goal-image-weight", type=float, default=1.0)
    parser.add_argument("--goal-delta-alignment-weight", type=float, default=1.0)
    parser.add_argument("--goal-delta-background-penalty-weight", type=float, default=0.05)
    parser.add_argument("--goal-delta-change-threshold", type=float, default=8.0)
    parser.add_argument(
        "--demo-future-offset-steps",
        type=int,
        default=16,
        help=(
            "For --evaluator demo_future_l2, compare the predicted FDM chunk against the "
            "matched demonstration canvas at env_step + this offset. This is a diagnostic/oracle evaluator."
        ),
    )
    parser.add_argument("--action-l2-weight", type=float, default=1.0)
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument(
        "--gemini-env-file",
        default=".local/gemini.env",
        help="Optional local env file containing GEMINI_API_KEY. Values are loaded only if not already set.",
    )
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    parser.add_argument(
        "--gemini-demo-video",
        default=None,
        help="Optional successful demonstration MP4 included in Gemini VLM reranking prompts.",
    )
    parser.add_argument("--gemini-candidate-video-fps", type=float, default=4.0)
    parser.add_argument("--gemini-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--gemini-max-candidates", type=int, default=8)
    parser.add_argument(
        "--gemini-prior-hint-mode",
        choices=("conservative", "neutral", "blind"),
        default="conservative",
        help=(
            "Controls whether Gemini sees and privileges the deterministic policy-prior candidate. "
            "Use blind for offline selector benchmarks."
        ),
    )
    parser.add_argument(
        "--gemini-delete-uploaded-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete Gemini Files API uploads after each rerank response.",
    )
    parser.add_argument(
        "--save-full-imagined-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --planner full_imagined_vlm_receding, save per-replan imagined candidate videos/actions.",
    )
    parser.add_argument(
        "--save-replan-prefix-imagined-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --planner full_imagined_vlm_receding, also save per-replan videos that concatenate the "
            "real executed prefix with each candidate's imagined suffix."
        ),
    )
    parser.add_argument(
        "--plan-to-env-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For full-imagined receding-horizon planning, dynamically shorten each replan horizon to the "
            "remaining environment budget so prefix+imagined videos have max-env-steps frames."
        ),
    )
    parser.add_argument(
        "--full-imagined-failure-fallback",
        choices=("policy_prior", "noop", "error"),
        default="policy_prior",
        help=(
            "Fallback when all full-imagined candidates fail before producing an executable action. "
            "`policy_prior` retries deterministic policy action on the real observation and then no-ops if that fails."
        ),
    )
    parser.add_argument(
        "--full-imagined-future-policy-temperature",
        type=float,
        default=None,
        help=(
            "For full-imagined planning, optionally override the policy sampling temperature after "
            "the first branch chunk. Set to 0.0 to score a sampled first action chunk followed by "
            "deterministic policy-prior continuation."
        ),
    )
    parser.add_argument("--pi0fast-model", default="lerobot/pi0fast-libero")
    parser.add_argument(
        "--pi0-model",
        default="lerobot/pi0_libero_base",
        help="LeRobot PI0 checkpoint used when --policy pi0.",
    )
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument(
        "--policy-compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override LeRobot PI0-family config.compile_model. Use --no-policy-compile-model "
            "for repeated short-horizon planning calls if torch.compile/CUDAGraph is unstable."
        ),
    )
    parser.add_argument("--pi0fast-base-image-key", default="observation.images.image")
    parser.add_argument("--pi0fast-wrist-image-key", default="observation.images.image2")
    parser.add_argument(
        "--pi0fast-image-transform",
        choices=("libero_180", "none"),
        default="libero_180",
        help="Image transform before LeRobot pi0-fast preprocessing. LIBERO eval uses a 180-degree flip.",
    )
    parser.add_argument("--pi0fast-state-key", default="observation.state")
    parser.add_argument(
        "--pi0fast-state-dim",
        type=int,
        default=8,
        help=(
            "Raw observation.state dimension passed into the saved LeRobot pi0-fast preprocessor. "
            "`lerobot/pi0fast-libero` uses 8D LIBERO proprio before its internal tokenizer/max-state handling."
        ),
    )
    parser.add_argument("--pi0fast-task-key", default="task")
    parser.add_argument(
        "--pi0fast-short-horizon-strategy",
        choices=("repeat_last", "error"),
        default="repeat_last",
        help=(
            "How to adapt pi0-fast's native 10-step LIBERO chunks when the planner requests a longer horizon. "
            "For Open-WAM FDM guidance, prefer native 10-step Pi0-FAST chunks; the FDM adapter pads internally "
            "for scoring and crops predicted video back to the real policy horizon."
        ),
    )
    parser.add_argument(
        "--pi0fast-action-selection-mode",
        choices=("select_action", "predict_chunk"),
        default="select_action",
        help=(
            "`select_action` matches LeRobot eval and uses pi0-fast's internal action queue. "
            "`predict_chunk` returns explicit chunks for planner-side horizon adaptation."
        ),
    )
    parser.add_argument("--agentview-key", default="agentview_image")
    parser.add_argument("--wrist-key", default="robot0_eye_in_hand_image")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument(
        "--openwam-config",
        default="configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml",
        help="Experiment YAML for --dynamics openwam_gjd.",
    )
    parser.add_argument("--openwam-checkpoint", default=None, help="Checkpoint directory or model_state.pt for OpenWAM GJD FDM.")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root override for OpenWAM config repair.")
    parser.add_argument("--empty-text-embedding-path", default=None)
    parser.add_argument("--reference-assets-device-policy", default=None)
    parser.add_argument("--video-num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument("--decode-device", default=None)
    parser.add_argument("--runtime-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument(
        "--openwam-fdm-mode",
        choices=("forced_action_joint_fdm", "clean_action_feedback"),
        default="forced_action_joint_fdm",
        help="OpenWAM action-conditioned FDM mode used for imagined branches.",
    )
    parser.add_argument(
        "--openwam-drop-text-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop task text for FDM modes to match GJD conditional-dynamics training semantics.",
    )
    parser.add_argument("--openwam-predicted-view-key", default="agentview_image")
    parser.add_argument(
        "--openwam-input-view-transform",
        choices=("none", "vertical_flip"),
        default="vertical_flip",
        help=(
            "Transform simulator RGB views before Open-WAM FDM encoding. LIBERO simulator frames are "
            "vertical-flipped relative to the LeRobot/Open-WAM latent convention, so the runner defaults "
            "to vertical_flip. Projected policy-facing views receive the inverse transform."
        ),
    )
    parser.add_argument(
        "--openwam-project-prediction-to-context-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Project Open-WAM decoded FDM output to the current --openwam-predicted-view-key frame shape. "
            "Use this for strict single-view backend comparisons; default preserves the decoded canvas."
        ),
    )
    parser.add_argument(
        "--openwam-return-canonical-prediction-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When projection is enabled, keep returning the canonical decoded Open-WAM video for evaluator/debug "
            "outputs while still updating policy context with projected views."
        ),
    )
    parser.add_argument(
        "--openwam-return-context-canvas-prediction-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When projection is enabled, return a human/policy-facing canvas built from projected context views "
            "for evaluator/debug outputs."
        ),
    )
    parser.add_argument("--openwam-agentview-camera-name", default="observation.images.agentview_rgb")
    parser.add_argument("--openwam-wrist-camera-name", default="observation.images.eye_in_hand_rgb")
    parser.add_argument(
        "--uva-root",
        default="previous_works/unified_video_action",
        help="Path to the UVA repo clone. Loaded only when --dynamics uva_libero_fdm.",
    )
    parser.add_argument("--uva-checkpoint", default="checkpoints/libero10.ckpt")
    parser.add_argument("--uva-dataset-dir", default="data/libero_10")
    parser.add_argument(
        "--uva-runtime-output-dir",
        default=None,
        help="Optional scratch directory for UVA runtime output. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--uva-action-space",
        choices=("openwam_raw7_delta_osc_libero_uva_frame", "openwam_raw7_delta_osc"),
        default="openwam_raw7_delta_osc_libero_uva_frame",
        help="Adapter used to convert planner raw7 OSC actions into UVA's native action window.",
    )
    parser.add_argument(
        "--uva-short-action-horizon-strategy",
        choices=("repeat_last", "error"),
        default="repeat_last",
        help="How to pad policy chunks shorter than UVA's 17 target slots for FDM-only scoring.",
    )
    parser.add_argument("--uva-predicted-view-key", default="agentview_image")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Repeatable OpenWAM config override, applied after checkpoint/runtime repair.",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument(
        "--dump-planner-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save first-chunk candidate action arrays and scores for each FDM-guided replan.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
