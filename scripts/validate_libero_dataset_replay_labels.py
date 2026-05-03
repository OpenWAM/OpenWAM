#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.integrations.libero_env import (  # noqa: E402
    LiberoTaskSpec,
    ensure_local_libero_config,
    load_libero_task_init_states,
    resolve_libero_task,
)


DEFAULT_DATASET_ROOT = Path("/path/to/private-resource")
DEFAULT_DIAGNOSTIC_ROOT = Path("/path/to/private-resource")
DEFAULT_LIBERO_REPO_ROOT = Path("/path/to/private-resource")
DEFAULT_SUBSETS = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")
REPLAY_SCHEMA_VERSION = 1
LIBERO_DEMOS_PER_TASK = 50


@dataclass(frozen=True)
class EpisodeSpec:
    subset: str
    dataset_root: str
    dataset_episode_index: int
    metadata_task_index: int | None
    task_text: str
    task_local_episode_idx: int
    task_text_occurrence_index: int
    recorded_length: int
    parquet_path: str


@dataclass(frozen=True)
class ReplayAttempt:
    init_state_index: int
    reset_seed: int
    success_during_actions: bool
    success_step: int | None
    success_after_grace: bool
    grace_success_step: int | None
    env_timestep: int | None
    wall_time_s: float
    error: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay local LeRobot LIBERO action sequences in upstream LIBERO and write dataset-local "
            "replay-status metadata candidates."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--libero-repo-root", type=Path, default=DEFAULT_LIBERO_REPO_ROOT)
    parser.add_argument("--subsets", type=str, default=",".join(DEFAULT_SUBSETS))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--collect-run", type=str, default=None)
    parser.add_argument("--install-dataset-meta", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-assignment",
        choices=("strided", "contiguous"),
        default="strided",
        help="How episodes are assigned to shards. Contiguous sharding improves cache locality for task-block replay.",
    )
    parser.add_argument("--limit-per-subset", type=int, default=None)
    parser.add_argument("--episode-indices", type=str, default=None)
    parser.add_argument(
        "--init-state-strategy",
        choices=("task_local_rank", "state_match", "search_on_failure"),
        default="state_match",
        help=(
            "`state_match` ranks LIBERO init states by proximity to the dataset timestep-0 robot state, then "
            "replays only the closest candidates. `task_local_rank` validates the simpler metadata-rank mapping. "
            "`search_on_failure` is a legacy loose search fallback and should not be used for canonical labels."
        ),
    )
    parser.add_argument("--max-search-init-states", type=int, default=None)
    parser.add_argument("--state-match-top-k", type=int, default=8)
    parser.add_argument("--state-match-position-scale-m", type=float, default=0.02)
    parser.add_argument("--state-match-rotation-scale-rad", type=float, default=0.10)
    parser.add_argument("--state-match-gripper-scale", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--post-action-grace-steps", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--env-horizon", type=int, default=5000)
    parser.add_argument(
        "--max-cached-task-envs",
        type=int,
        default=1,
        help="Maximum live LIBERO task envs per shard process. Keep this low for large multi-subset runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mujoco-gl", type=str, default="osmesa")
    parser.add_argument("--pyopengl-platform", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subsets = parse_subset_selector(args.subsets)
    os.environ["LIBERO_REPO_ROOT"] = str(args.libero_repo_root)
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform or args.mujoco_gl

    if args.collect_run is not None:
        collect_run(args=args, subsets=subsets, run_id=args.collect_run)
        return

    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count.")

    run_id = args.run_id or f"libero_replay_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.diagnostic_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_environment_report(run_root / "environment.json", args=args, run_id=run_id)

    all_results: list[dict[str, Any]] = []
    for subset in subsets:
        subset_results = run_subset(args=args, run_id=run_id, subset=subset, run_root=run_root)
        all_results.extend(subset_results)

    summary = summarize_rows(all_results)
    summary.update(
        {
            "run_id": run_id,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "subsets": subsets,
            "diagnostic_root": str(run_root),
        }
    )
    print(json.dumps(summary, indent=2))


def run_subset(*, args: argparse.Namespace, run_id: str, subset: str, run_root: Path) -> list[dict[str, Any]]:
    dataset_root = args.dataset_root / subset
    episodes = load_episode_specs(dataset_root, subset=subset)
    if args.episode_indices:
        requested = set(parse_int_selector(args.episode_indices))
        episodes = [episode for episode in episodes if episode.dataset_episode_index in requested]
    if args.limit_per_subset is not None:
        episodes = episodes[: args.limit_per_subset]

    shard_episodes = select_shard_episodes(
        episodes,
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
        shard_assignment=str(args.shard_assignment),
    )
    subset_root = run_root / subset
    subset_root.mkdir(parents=True, exist_ok=True)
    output_path = subset_root / f"replay_status.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.jsonl"
    completed = load_completed_episode_indices(output_path) if args.resume else set()

    print(
        json.dumps(
            {
                "event": "subset_start",
                "subset": subset,
                "episodes": len(episodes),
                "shard_episodes": len(shard_episodes),
                "completed": len(completed),
                "output_path": str(output_path),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return []

    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs.env_wrapper import ControlEnv  # type: ignore

    task_cache: dict[str, dict[str, Any]] = {}
    env_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as output:
        try:
            for episode in shard_episodes:
                if episode.dataset_episode_index in completed:
                    continue
                row = replay_episode(
                    episode,
                    args=args,
                    run_id=run_id,
                    task_cache=task_cache,
                    env_cache=env_cache,
                    env_cls=ControlEnv,
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                rows.append(row)
                prune_env_cache(
                    env_cache,
                    keep_task_key=str(row.get("task_cache_key") or episode.task_text),
                    max_cached_task_envs=args.max_cached_task_envs,
                )
                print(
                    json.dumps(
                        {
                            "event": "episode_done",
                            "subset": subset,
                            "episode": episode.dataset_episode_index,
                            "status": row["replay_status"],
                            "primary_success": row.get("primary_success_during_actions"),
                            "resolved_init": row.get("resolved_init_state_index"),
                            "attempts": row.get("init_state_attempt_count"),
                        }
                    ),
                    flush=True,
                )
        finally:
            for env in env_cache.values():
                env.close()

    summary_path = subset_root / f"summary.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json"
    summary_path.write_text(json.dumps(summarize_rows(rows), indent=2), encoding="utf-8")
    return rows


def prune_env_cache(env_cache: dict[str, Any], *, keep_task_key: str, max_cached_task_envs: int) -> None:
    if max_cached_task_envs <= 0:
        max_cached_task_envs = 1
    while len(env_cache) > max_cached_task_envs:
        evict_key = next(key for key in env_cache if key != keep_task_key)
        env_cache.pop(evict_key).close()


def select_shard_episodes(
    episodes: list[EpisodeSpec],
    *,
    shard_count: int,
    shard_index: int,
    shard_assignment: str,
) -> list[EpisodeSpec]:
    if shard_assignment == "strided":
        return [episode for ordinal, episode in enumerate(episodes) if ordinal % shard_count == shard_index]
    if shard_assignment == "contiguous":
        start = len(episodes) * shard_index // shard_count
        stop = len(episodes) * (shard_index + 1) // shard_count
        return episodes[start:stop]
    raise ValueError(f"Unknown shard assignment: {shard_assignment!r}")


def replay_episode(
    episode: EpisodeSpec,
    *,
    args: argparse.Namespace,
    run_id: str,
    task_cache: dict[str, dict[str, Any]],
    env_cache: dict[str, Any],
    env_cls: Any,
) -> dict[str, Any]:
    row_base = {
        "subset": episode.subset,
        "dataset_root": episode.dataset_root,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "replay_run_id": run_id,
        "dataset_episode_index": episode.dataset_episode_index,
        "metadata_task_index": episode.metadata_task_index,
        "task_text": episode.task_text,
        "upstream_benchmark": episode.subset,
        "task_local_episode_idx": episode.task_local_episode_idx,
        "task_text_occurrence_index": episode.task_text_occurrence_index,
        "recorded_length": episode.recorded_length,
        "parquet_path": episode.parquet_path,
        "warmup_steps": args.warmup_steps,
        "post_action_grace_steps": args.post_action_grace_steps,
        "init_state_strategy": args.init_state_strategy,
        "state_match_position_scale_m": args.state_match_position_scale_m,
        "state_match_rotation_scale_rad": args.state_match_rotation_scale_rad,
        "state_match_gripper_scale": args.state_match_gripper_scale,
        "env_control_freq": args.control_freq,
        "mujoco_gl": args.mujoco_gl,
        "pyopengl_platform": args.pyopengl_platform or args.mujoco_gl,
        "simulator_env_id": "libero:ControlEnv",
        "use_camera_obs": False,
    }
    t0 = time.perf_counter()
    try:
        actions, dataset_initial_state = read_episode_actions_and_initial_state(Path(episode.parquet_path))
        task_info = get_task_info(
            episode,
            args=args,
            task_cache=task_cache,
            env_cache=env_cache,
            env_cls=env_cls,
            dataset_initial_state=dataset_initial_state,
        )
        init_states = task_info["init_states"]
        task_local_rank_init = int(episode.task_local_episode_idx % len(init_states))
        state_match_candidates: list[dict[str, Any]] = []
        if args.init_state_strategy == "state_match":
            state_match_candidates = rank_init_states_by_robot_state(
                dataset_initial_state=dataset_initial_state,
                candidate_robot_states=task_info["init_state_robot_states"],
                top_k=args.state_match_top_k,
                position_scale_m=args.state_match_position_scale_m,
                rotation_scale_rad=args.state_match_rotation_scale_rad,
                gripper_scale=args.state_match_gripper_scale,
            )
            candidate_indices = [int(candidate["init_state_index"]) for candidate in state_match_candidates]
        else:
            candidate_indices = build_init_state_candidate_order(
                primary_init=task_local_rank_init,
                task_id=int(task_info["task_spec"].task_id),
                init_state_count=len(init_states),
                strategy=args.init_state_strategy,
                max_search_init_states=args.max_search_init_states,
            )
        if not candidate_indices:
            raise RuntimeError("No candidate init states were selected for replay.")

        attempts: list[ReplayAttempt] = []
        selected_attempt: ReplayAttempt | None = None
        for init_state_index in candidate_indices:
            attempt = run_replay_attempt(
                task_info["env"],
                init_states[int(init_state_index)],
                actions,
                init_state_index=int(init_state_index),
                reset_seed=episode_seed(args.seed, episode.dataset_episode_index, int(init_state_index)),
                warmup_steps=args.warmup_steps,
                post_action_grace_steps=args.post_action_grace_steps,
            )
            attempts.append(attempt)
            if attempt.success_during_actions:
                selected_attempt = attempt
                break
            if args.init_state_strategy == "task_local_rank":
                break

        primary_attempt = attempts[0]
        selected_attempt = selected_attempt or primary_attempt
        replay_success = bool(selected_attempt.success_during_actions)
        return {
            **row_base,
            "replay_status": "success" if replay_success else "failure",
            "upstream_task_id": int(task_info["task_spec"].task_id),
            "upstream_task_name": str(task_info["task_spec"].task_name),
            "task_cache_key": str(task_info["task_cache_key"]),
            "task_resolution_strategy": str(task_info["task_resolution_strategy"]),
            "task_resolution_candidates": task_info["task_resolution_candidates"],
            "dataset_initial_robot_state": dataset_initial_state.astype(float).tolist(),
            "task_local_rank_init_state_index": int(task_local_rank_init),
            "state_match_top_k": int(args.state_match_top_k) if args.init_state_strategy == "state_match" else None,
            "state_match_candidates": state_match_candidates,
            "state_match_best_init_state_index": (
                int(state_match_candidates[0]["init_state_index"]) if state_match_candidates else None
            ),
            "state_match_best_score": float(state_match_candidates[0]["score"]) if state_match_candidates else None,
            "primary_init_state_index": int(candidate_indices[0]),
            "primary_success_during_actions": bool(primary_attempt.success_during_actions),
            "primary_success_step": primary_attempt.success_step,
            "resolved_init_state_index": int(selected_attempt.init_state_index),
            "attempted_init_state_indices": [int(attempt.init_state_index) for attempt in attempts],
            "init_state_attempt_count": len(attempts),
            "replayed_actions": int(actions.shape[0]),
            "success_during_actions": replay_success,
            "success_step": selected_attempt.success_step if replay_success else None,
            "success_after_grace": bool(selected_attempt.success_after_grace),
            "failure": not replay_success,
            "failure_reason": None if replay_success else "no_success_for_init_state_strategy",
            "error": None,
            "attempts": [asdict(attempt) for attempt in attempts],
            "wall_time_s": round(time.perf_counter() - t0, 6),
        }
    except Exception as exc:
        return {
            **row_base,
            "replay_status": "error",
            "failure": None,
            "failure_reason": "validator_error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_time_s": round(time.perf_counter() - t0, 6),
        }


def get_task_info(
    episode: EpisodeSpec,
    *,
    args: argparse.Namespace,
    task_cache: dict[str, dict[str, Any]],
    env_cache: dict[str, Any],
    env_cls: Any,
    dataset_initial_state: np.ndarray,
) -> dict[str, Any]:
    task_spec, resolution_strategy, resolution_candidates = resolve_episode_task_spec(
        episode,
        args=args,
        task_cache=task_cache,
        env_cache=env_cache,
        env_cls=env_cls,
        dataset_initial_state=dataset_initial_state,
    )
    task_cache_key = libero_task_cache_key(task_spec)
    if task_cache_key not in env_cache:
        env_cache[task_cache_key] = env_cls(
            bddl_file_name=task_spec.bddl_file_path,
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=False,
            horizon=int(args.env_horizon),
            ignore_done=False,
            control_freq=int(args.control_freq),
        )
    cache_entry = task_cache[task_cache_key]
    if args.init_state_strategy == "state_match" and "init_state_robot_states" not in cache_entry:
        cache_entry["init_state_robot_states"] = compute_init_state_robot_states(
            env_cache[task_cache_key],
            cache_entry["init_states"],
            base_seed=int(args.seed) + int(cache_entry["task_spec"].task_id) * 100_003,
        )
    return {
        **cache_entry,
        "env": env_cache[task_cache_key],
        "task_cache_key": task_cache_key,
        "task_resolution_strategy": resolution_strategy,
        "task_resolution_candidates": resolution_candidates,
    }


def resolve_episode_task_spec(
    episode: EpisodeSpec,
    *,
    args: argparse.Namespace,
    task_cache: dict[str, dict[str, Any]],
    env_cache: dict[str, Any],
    env_cls: Any,
    dataset_initial_state: np.ndarray,
) -> tuple[LiberoTaskSpec, str, list[dict[str, Any]]]:
    candidates = resolve_libero_task_candidates(episode.task_text, REPO_ROOT, benchmark_name=episode.subset)
    if len(candidates) == 1:
        task_spec = candidates[0]
        ensure_task_cache_entry(task_cache, task_spec)
        return task_spec, "unique_language", [
            task_resolution_candidate_row(task_spec, rank=0, selected=True)
        ]

    occurrence_index = int(episode.task_text_occurrence_index)
    if occurrence_index < len(candidates):
        task_spec = candidates[occurrence_index]
        ensure_task_cache_entry(task_cache, task_spec)
        return task_spec, "task_text_occurrence", [
            task_resolution_candidate_row(task_spec, rank=index, selected=index == occurrence_index)
            for index, task_spec in enumerate(candidates)
        ]
    if args.init_state_strategy != "state_match":
        raise ValueError(
            f"Task text {episode.task_text!r} has {len(candidates)} upstream matches, "
            f"but occurrence index {occurrence_index} is out of range."
        )

    # Fallback only. The LeRobot export should preserve 50 demonstrations per
    # upstream task, but if that invariant is broken, choose the scene whose init
    # states best match the dataset's first robot state.
    scored_candidates: list[dict[str, Any]] = []
    for candidate_rank, task_spec in enumerate(candidates):
        task_cache_key = libero_task_cache_key(task_spec)
        ensure_task_cache_entry(task_cache, task_spec)
        if task_cache_key not in env_cache:
            env_cache[task_cache_key] = env_cls(
                bddl_file_name=task_spec.bddl_file_path,
                use_camera_obs=False,
                has_offscreen_renderer=False,
                has_renderer=False,
                horizon=int(args.env_horizon),
                ignore_done=False,
                control_freq=int(args.control_freq),
            )
        cache_entry = task_cache[task_cache_key]
        if "init_state_robot_states" not in cache_entry:
            cache_entry["init_state_robot_states"] = compute_init_state_robot_states(
                env_cache[task_cache_key],
                cache_entry["init_states"],
                base_seed=int(args.seed) + int(task_spec.task_id) * 100_003,
            )
        best_match = rank_init_states_by_robot_state(
            dataset_initial_state=dataset_initial_state,
            candidate_robot_states=cache_entry["init_state_robot_states"],
            top_k=1,
            position_scale_m=args.state_match_position_scale_m,
            rotation_scale_rad=args.state_match_rotation_scale_rad,
            gripper_scale=args.state_match_gripper_scale,
        )[0]
        scored_candidates.append(
            {
                **task_resolution_candidate_row(task_spec, rank=candidate_rank, selected=False),
                "best_init_state_index": int(best_match["init_state_index"]),
                "best_score": float(best_match["score"]),
                "best_position_error_m": float(best_match["position_error_m"]),
                "best_rotation_error_deg": float(best_match["rotation_error_deg"]),
                "best_gripper_error": float(best_match["gripper_error"]),
            }
        )

    selected_index = min(range(len(scored_candidates)), key=lambda index: float(scored_candidates[index]["best_score"]))
    scored_candidates[selected_index]["selected"] = True
    task_spec = candidates[selected_index]
    return task_spec, "state_match_language_disambiguation_fallback", scored_candidates


def ensure_task_cache_entry(task_cache: dict[str, dict[str, Any]], task_spec: LiberoTaskSpec) -> None:
    task_cache_key = libero_task_cache_key(task_spec)
    if task_cache_key in task_cache:
        return
    task_cache[task_cache_key] = {
        "task_spec": task_spec,
        "init_states": load_libero_task_init_states(task_spec, REPO_ROOT),
    }


def libero_task_cache_key(task_spec: LiberoTaskSpec) -> str:
    return f"{task_spec.benchmark_name}:{task_spec.task_id}:{task_spec.task_name}"


def task_resolution_candidate_row(task_spec: LiberoTaskSpec, *, rank: int, selected: bool) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "benchmark_name": task_spec.benchmark_name,
        "task_id": int(task_spec.task_id),
        "task_name": task_spec.task_name,
        "problem_folder": task_spec.problem_folder,
        "language": task_spec.task_language,
        "selected": bool(selected),
    }


def resolve_libero_task_candidates(
    task_text: str,
    project_root: Path | None = None,
    *,
    benchmark_name: str | None = None,
) -> list[LiberoTaskSpec]:
    try:
        return [resolve_libero_task(task_text, project_root, benchmark_name=benchmark_name)]
    except ValueError as exc:
        if "matched multiple LIBERO tasks" not in str(exc):
            raise

    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    normalized_task_text = normalize_task_text(task_text)
    benchmark_classes = benchmark.get_benchmark_dict()
    if benchmark_name is not None:
        try:
            benchmark_items = ((benchmark_name, benchmark_classes[benchmark_name]),)
        except KeyError as exc:
            available = ", ".join(sorted(benchmark_classes))
            raise ValueError(
                f"Unknown LIBERO benchmark {benchmark_name!r}; available benchmarks: {available}"
            ) from exc
    else:
        benchmark_items = tuple(benchmark_classes.items())

    raw_matches: list[LiberoTaskSpec] = []
    for current_benchmark_name, benchmark_class in benchmark_items:
        try:
            benchmark_instance = benchmark_class()
        except Exception:
            continue
        for task_id in range(benchmark_instance.get_num_tasks()):
            task = benchmark_instance.get_task(task_id)
            if normalize_task_text(task.language) != normalized_task_text:
                continue
            raw_matches.append(
                LiberoTaskSpec(
                    benchmark_name=current_benchmark_name,
                    task_id=task_id,
                    task_name=task.name,
                    task_language=task.language,
                    problem_folder=task.problem_folder,
                    bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
                    init_states_path="",
                )
            )

    if not raw_matches:
        raise ValueError(f"Could not resolve LIBERO task text: {task_text!r}")

    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return [
        LiberoTaskSpec(
            benchmark_name=match.benchmark_name,
            task_id=match.task_id,
            task_name=match.task_name,
            task_language=match.task_language,
            problem_folder=match.problem_folder,
            bddl_file_path=match.bddl_file_path,
            init_states_path=str(Path(config["init_states"]) / match.problem_folder / f"{match.task_name}.pruned_init"),
        )
        for match in raw_matches
    ]


def normalize_task_text(task_text: str) -> str:
    return " ".join(task_text.lower().strip().split())


def compute_init_state_robot_states(env: Any, init_states: Any, *, base_seed: int) -> np.ndarray:
    robot_states: list[np.ndarray] = []
    for init_state_index, init_state in enumerate(init_states):
        reset_seed = int(base_seed) + int(init_state_index) * 9176
        random.seed(reset_seed)
        np.random.seed(reset_seed % (2**32 - 1))
        env.seed(reset_seed)
        env.reset()
        obs = env.set_init_state(init_state)
        robot_states.append(robot_state_from_obs(obs))
    return np.stack(robot_states, axis=0).astype(np.float32)


def rank_init_states_by_robot_state(
    *,
    dataset_initial_state: np.ndarray,
    candidate_robot_states: np.ndarray,
    top_k: int,
    position_scale_m: float,
    rotation_scale_rad: float,
    gripper_scale: float,
) -> list[dict[str, Any]]:
    top_k = max(1, int(top_k))
    dataset_state = np.asarray(dataset_initial_state, dtype=np.float32)
    candidate_states = np.asarray(candidate_robot_states, dtype=np.float32)
    dataset_quaternion = axis_angle_to_quaternion_np(dataset_state[3:6])
    candidate_quaternions = np.stack(
        [axis_angle_to_quaternion_np(candidate_state[3:6]) for candidate_state in candidate_states],
        axis=0,
    )
    position_errors = np.linalg.norm(candidate_states[:, 0:3] - dataset_state[0:3], axis=1)
    rotation_errors = quaternion_angular_error_rad_np(candidate_quaternions, dataset_quaternion)
    gripper_errors = np.linalg.norm(candidate_states[:, 6:] - dataset_state[6:], axis=1)
    scores = (
        position_errors / max(float(position_scale_m), 1e-8)
        + rotation_errors / max(float(rotation_scale_rad), 1e-8)
        + gripper_errors / max(float(gripper_scale), 1e-8)
    )
    order = np.argsort(scores, kind="stable")[:top_k]
    return [
        {
            "rank": int(rank),
            "init_state_index": int(index),
            "score": float(scores[index]),
            "position_error_m": float(position_errors[index]),
            "rotation_error_rad": float(rotation_errors[index]),
            "rotation_error_deg": float(np.rad2deg(rotation_errors[index])),
            "gripper_error": float(gripper_errors[index]),
            "candidate_robot_state": candidate_states[index].astype(float).tolist(),
        }
        for rank, index in enumerate(order)
    ]


def robot_state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    quaternion = normalize_quaternion_np(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1))
    axis_angle = quaternion_to_axis_angle_np(quaternion)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    return np.concatenate([position, axis_angle, gripper], axis=0).astype(np.float32)


def axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis_angle / angle
    half_angle = 0.5 * angle
    quaternion = np.concatenate([axis * np.sin(half_angle), [np.cos(half_angle)]])
    return normalize_quaternion_np(quaternion)


def quaternion_to_axis_angle_np(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion_np(quaternion)
    xyz = quaternion[0:3]
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, w)
    return (xyz / sin_half * angle).astype(np.float32)


def normalize_quaternion_np(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def quaternion_angular_error_rad_np(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs_xyzw, dtype=np.float64)
    rhs = normalize_quaternion_np(rhs_xyzw)
    lhs_norm = np.linalg.norm(lhs, axis=-1, keepdims=True)
    lhs = lhs / np.clip(lhs_norm, 1e-12, None)
    dots = np.abs(np.sum(lhs * rhs.reshape(1, 4), axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def run_replay_attempt(
    env: Any,
    init_state: Any,
    actions: np.ndarray,
    *,
    init_state_index: int,
    reset_seed: int,
    warmup_steps: int,
    post_action_grace_steps: int,
) -> ReplayAttempt:
    t0 = time.perf_counter()
    try:
        random.seed(reset_seed)
        np.random.seed(reset_seed % (2**32 - 1))
        env.seed(reset_seed)
        env.reset()
        env.set_init_state(init_state)
        zero_action = np.zeros((actions.shape[1],), dtype=np.float32)
        for _ in range(int(warmup_steps)):
            _, _, done, _ = env.step(zero_action)
            if done:
                return ReplayAttempt(
                    init_state_index=init_state_index,
                    reset_seed=reset_seed,
                    success_during_actions=True,
                    success_step=0,
                    success_after_grace=False,
                    grace_success_step=None,
                    env_timestep=int(getattr(env.env, "timestep", -1)),
                    wall_time_s=round(time.perf_counter() - t0, 6),
                )

        for action_index, action in enumerate(actions, start=1):
            _, _, done, _ = env.step(action.astype(np.float32, copy=False))
            if done:
                return ReplayAttempt(
                    init_state_index=init_state_index,
                    reset_seed=reset_seed,
                    success_during_actions=True,
                    success_step=action_index,
                    success_after_grace=False,
                    grace_success_step=None,
                    env_timestep=int(getattr(env.env, "timestep", -1)),
                    wall_time_s=round(time.perf_counter() - t0, 6),
                )

        for grace_index in range(1, int(post_action_grace_steps) + 1):
            _, _, done, _ = env.step(zero_action)
            if done:
                return ReplayAttempt(
                    init_state_index=init_state_index,
                    reset_seed=reset_seed,
                    success_during_actions=False,
                    success_step=None,
                    success_after_grace=True,
                    grace_success_step=grace_index,
                    env_timestep=int(getattr(env.env, "timestep", -1)),
                    wall_time_s=round(time.perf_counter() - t0, 6),
                )

        return ReplayAttempt(
            init_state_index=init_state_index,
            reset_seed=reset_seed,
            success_during_actions=False,
            success_step=None,
            success_after_grace=False,
            grace_success_step=None,
            env_timestep=int(getattr(env.env, "timestep", -1)),
            wall_time_s=round(time.perf_counter() - t0, 6),
        )
    except Exception as exc:
        return ReplayAttempt(
            init_state_index=init_state_index,
            reset_seed=reset_seed,
            success_during_actions=False,
            success_step=None,
            success_after_grace=False,
            grace_success_step=None,
            env_timestep=int(getattr(env.env, "timestep", -1)) if hasattr(env, "env") else None,
            wall_time_s=round(time.perf_counter() - t0, 6),
            error=repr(exc),
        )


def build_init_state_candidate_order(
    *,
    primary_init: int,
    task_id: int,
    init_state_count: int,
    strategy: str,
    max_search_init_states: int | None,
) -> list[int]:
    if strategy == "task_local_rank":
        return [primary_init]
    candidates = [primary_init]
    for index in (task_id, *range(init_state_count)):
        if index < 0 or index >= init_state_count or index in candidates:
            continue
        candidates.append(index)
    if max_search_init_states is not None:
        candidates = candidates[: max(1, int(max_search_init_states))]
    return candidates


def episode_seed(base_seed: int, episode_index: int, init_state_index: int) -> int:
    return int(base_seed) + int(episode_index) * 1009 + int(init_state_index) * 9176


def load_episode_specs(dataset_root: Path, *, subset: str) -> list[EpisodeSpec]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    }
    task_ranks: dict[str, int] = defaultdict(int)
    episodes: list[EpisodeSpec] = []
    for row in sorted(read_jsonl(dataset_root / "meta" / "episodes.jsonl"), key=lambda item: int(item["episode_index"])):
        episode_index = int(row["episode_index"])
        task_text = str((row.get("tasks") or [tasks.get(int(row.get("task_index", -1)), "")])[0])
        task_rank = task_ranks[task_text]
        task_ranks[task_text] += 1
        # LIBERO stores one task block per 50 pruned init states in these LeRobot exports.
        # Duplicate language strings can span multiple upstream scene tasks, so keep the
        # occurrence block separate from the language-level rank.
        task_text_occurrence_index = task_rank // LIBERO_DEMOS_PER_TASK
        episodes.append(
            EpisodeSpec(
                subset=subset,
                dataset_root=str(dataset_root),
                dataset_episode_index=episode_index,
                metadata_task_index=task_index_for_text(tasks, task_text),
                task_text=task_text,
                task_local_episode_idx=task_rank,
                task_text_occurrence_index=task_text_occurrence_index,
                recorded_length=int(row.get("length", 0)),
                parquet_path=str(resolve_episode_parquet_path(dataset_root, info=info, episode_index=episode_index)),
            )
        )
    return episodes


def task_index_for_text(tasks: dict[int, str], task_text: str) -> int | None:
    for task_index, current_text in tasks.items():
        if current_text == task_text:
            return int(task_index)
    return None


def resolve_episode_parquet_path(dataset_root: Path, *, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    return dataset_root / data_path.format(
        episode_chunk=int(episode_index) // chunks_size,
        episode_index=int(episode_index),
    )


def read_episode_actions_and_initial_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D action sequence in {path}, got shape {actions.shape}.")
    initial_state = np.asarray(table.column("observation.state").to_pylist()[0], dtype=np.float32)
    if initial_state.ndim != 1 or initial_state.shape[0] < 8:
        raise ValueError(f"Expected initial robot state with at least 8 dims in {path}, got shape {initial_state.shape}.")
    return actions, initial_state[0:8]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def load_completed_episode_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed: set[int] = set()
    for row in read_jsonl(path):
        completed.add(int(row["dataset_episode_index"]))
    return completed


def collect_run(*, args: argparse.Namespace, subsets: list[str], run_id: str) -> None:
    run_root = args.diagnostic_root / run_id
    environment = {}
    environment_path = run_root / "environment.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    collect_summary: dict[str, Any] = {"run_id": run_id, "subsets": {}}
    for subset in subsets:
        dataset_root = args.dataset_root / subset
        expected_count = len(load_episode_specs(dataset_root, subset=subset))
        rows_by_episode: dict[int, dict[str, Any]] = {}
        for shard_path in sorted((run_root / subset).glob("replay_status.shard_*.jsonl")):
            for row in read_jsonl(shard_path):
                rows_by_episode[int(row["dataset_episode_index"])] = row
        rows = [rows_by_episode[index] for index in sorted(rows_by_episode)]
        complete = len(rows) == expected_count
        summary = summarize_rows(rows)
        summary.update({"expected_episodes": expected_count, "complete": complete})
        collect_summary["subsets"][subset] = summary
        (run_root / subset).mkdir(parents=True, exist_ok=True)
        ((run_root / subset) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.install_dataset_meta:
            if not complete and not args.allow_incomplete:
                raise RuntimeError(
                    f"Refusing to install incomplete replay labels for {subset}: "
                    f"{len(rows)} / {expected_count} rows."
                )
            install_dataset_metadata(dataset_root=dataset_root, rows=rows, summary=summary, environment=environment)
    print(json.dumps(collect_summary, indent=2))


def install_dataset_metadata(
    *,
    dataset_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    status_path = meta_root / "replay_status.jsonl"
    status_tmp = meta_root / f"{status_path.name}.tmp"
    with status_tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    status_tmp.replace(status_path)
    (meta_root / "replay_status.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (meta_root / "replay_status.environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("replay_status")) for row in rows)
    primary_successes = sum(1 for row in rows if row.get("primary_success_during_actions") is True)
    searched_successes = sum(
        1
        for row in rows
        if row.get("replay_status") == "success" and row.get("primary_success_during_actions") is False
    )
    return {
        "total_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "success_rate": (status_counts.get("success", 0) / len(rows)) if rows else 0.0,
        "primary_successes": primary_successes,
        "searched_successes": searched_successes,
        "failure_episode_indices": [
            int(row["dataset_episode_index"]) for row in rows if row.get("replay_status") == "failure"
        ],
        "error_episode_indices": [
            int(row["dataset_episode_index"]) for row in rows if row.get("replay_status") == "error"
        ],
    }


def write_environment_report(path: Path, *, args: argparse.Namespace, run_id: str) -> None:
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version,
        "repo_root": str(REPO_ROOT),
        "open_wam_git_commit": run_command(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT),
        "open_wam_git_status": run_command(("git", "status", "--short"), cwd=REPO_ROOT),
        "libero_repo_root": str(args.libero_repo_root),
        "libero_git_commit": run_command(("git", "rev-parse", "HEAD"), cwd=args.libero_repo_root),
        "libero_git_status": run_command(("git", "status", "--short"), cwd=args.libero_repo_root),
        "numpy": getattr(np, "__version__", None),
        "pyarrow": getattr(pq, "__version__", None),
        "mujoco_gl": args.mujoco_gl,
        "pyopengl_platform": args.pyopengl_platform or args.mujoco_gl,
        "command": sys.argv,
    }
    write_text_atomic(path, json.dumps(report, indent=2) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def run_command(command: tuple[str, ...], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def parse_subset_selector(raw: str) -> list[str]:
    subsets = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not subsets:
        raise ValueError("--subsets did not select any dataset subsets.")
    return subsets


def parse_int_selector(value: str) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for raw_piece in value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        if ":" in piece:
            parts = piece.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid integer range selector {piece!r}.")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step == 0:
                raise ValueError(f"Invalid integer range selector {piece!r}: step cannot be zero.")
            values = range(start, stop, step)
        else:
            values = (int(piece),)
        for item in values:
            if item < 0:
                raise ValueError(f"Negative indices are not supported: {item}")
            if item in seen:
                continue
            seen.add(item)
            selected.append(item)
    return selected


if __name__ == "__main__":
    main()
