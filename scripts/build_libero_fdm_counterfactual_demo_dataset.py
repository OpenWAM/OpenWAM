#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import imageio.v2 as imageio

from open_wam.ablations.joint_denoising_fdm.branches import (
    BRANCH_PRESETS,
    apply_action_branch,
    branch_metadata,
    branch_seed_offset,
    expand_branch_names,
)


DEFAULT_REPLAY_STATUS = (
    "/path/to/private-resource"
    "libero_replay_metadata_20260502_final_top12_plus_top50/libero_10/replay_status.jsonl"
)
DEFAULT_OUTPUT_DIR = "/path/to/private-resource"
DEFAULT_BRANCHES = BRANCH_PRESETS["training_10"]
LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


@dataclass(frozen=True)
class SourceEpisode:
    dataset_episode_index: int
    task_id: int
    task_text: str
    init_state_index: int
    parquet_path: Path


@dataclass(frozen=True)
class ContextArtifact:
    context_id: int
    episode: SourceEpisode
    t0_frame: int
    requested_t0_fraction: float
    total_video_frames: int
    context_start_frame: int
    context_path: Path
    action_context_shape: tuple[int, ...]
    context_rgb_shape: tuple[int, ...]
    simulator_state: np.ndarray


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    if args.pyopengl_platform:
        os.environ.setdefault("PYOPENGL_PLATFORM", args.pyopengl_platform)

    branches = expand_branch_names(args.branches)
    t0_fractions = tuple(float(value) for value in _parse_csv_tuple(args.t0_fractions))
    output_root = _resolve_output_root(args)

    replay_rows = _read_jsonl(Path(args.replay_status_path).expanduser())
    excluded_episode_indices = _load_excluded_episode_indices(args.exclude_source_dataset_root)
    source_episodes = _select_source_episodes(
        replay_rows,
        task_ids=_parse_int_csv(args.task_ids),
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        excluded_episode_indices=excluded_episode_indices,
    )
    expected_transitions = len(source_episodes) * len(t0_fractions) * len(branches)
    if expected_transitions != int(args.target_transitions):
        raise ValueError(
            "This demo builder intentionally creates an exact rectangular matrix. "
            f"episodes={len(source_episodes)}, t0s={len(t0_fractions)}, branches={len(branches)} "
            f"produce {expected_transitions} transitions, not --target-transitions={args.target_transitions}."
        )

    manifest = {
        "dataset_kind": "libero10_counterfactual_fdm_demo",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "benchmark": args.benchmark,
        "replay_status_path": str(Path(args.replay_status_path).expanduser().resolve()),
        "target_transitions": int(args.target_transitions),
        "actual_transitions": 0,
        "task_ids": list(_parse_int_csv(args.task_ids)),
        "episodes_per_task": int(args.episodes_per_task),
        "selected_episode_count": len(source_episodes),
        "segment_frames": int(args.segment_frames),
        "horizon_frames": int(args.horizon_frames),
        "context_window_frames": int(args.context_window_frames),
        "action_per_frame": int(args.action_per_frame),
        "target_raw_frames": _decoded_raw_frames_for_latents(
            int(args.horizon_frames),
            action_per_frame=int(args.action_per_frame),
        ),
        "context_raw_frames": _raw_window_frames_for_latents(
            int(args.context_window_frames),
            action_per_frame=int(args.action_per_frame),
        ),
        "camera_height": int(args.camera_height),
        "camera_width": int(args.camera_width),
        "branches": list(branches),
        "branch_metadata": {branch: branch_metadata(branch) for branch in branches},
        "t0_fractions": list(t0_fractions),
        "seed": int(args.seed),
        "excluded_source_dataset_roots": [str(Path(path).expanduser()) for path in args.exclude_source_dataset_root],
        "excluded_source_episode_count": len(excluded_episode_indices),
        "text_conditioning_recommendation": "drop text during future FDM denoising; keep text only for policy baselines",
        "source_episodes": [_episode_to_row(episode) for episode in source_episodes],
    }
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "plan_only",
                    "output_root": str(output_root),
                    "manifest_preview": manifest,
                },
                indent=2,
                default=str,
            )
        )
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists. Pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "contexts").mkdir(parents=True, exist_ok=True)
    (output_root / "samples").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "previews").mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", manifest)

    from open_wam.integrations import (
        build_libero_offscreen_env,
        ensure_local_libero_config,
        load_libero_task_init_states,
        resolve_libero_task,
    )

    ensure_local_libero_config(Path.cwd())

    context_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    preview_count = 0
    sample_id = 0
    context_id = 0
    task_to_episodes: dict[int, list[SourceEpisode]] = defaultdict(list)
    for episode in source_episodes:
        task_to_episodes[episode.task_id].append(episode)

    for task_id in sorted(task_to_episodes):
        episodes = task_to_episodes[task_id]
        task_spec = resolve_libero_task(episodes[0].task_text, project_root=Path.cwd(), benchmark_name=args.benchmark)
        init_states = load_libero_task_init_states(task_spec, project_root=Path.cwd())
        env = build_libero_offscreen_env(
            task_spec,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            horizon=args.env_horizon,
            ignore_done=True,
            project_root=Path.cwd(),
        )
        try:
            for episode in episodes:
                actions = _read_actions(episode.parquet_path)
                t0_frames = _select_t0_frames(
                    total_video_frames=actions.shape[0] // int(args.action_per_frame),
                    horizon_frames=args.horizon_frames,
                    context_window_frames=args.context_window_frames,
                    t0_fractions=t0_fractions,
                )
                init_state = init_states[int(episode.init_state_index) % len(init_states)]
                total_video_frames = actions.shape[0] // int(args.action_per_frame)
                for requested_t0_fraction, t0_frame in t0_frames:
                    context = _render_context_at_t0(
                        env=env,
                        init_state=init_state,
                        episode=episode,
                        actions=actions,
                        t0_frame=t0_frame,
                        requested_t0_fraction=requested_t0_fraction,
                        total_video_frames=total_video_frames,
                        context_window_frames=args.context_window_frames,
                        action_per_frame=args.action_per_frame,
                        output_root=output_root,
                        context_id=context_id,
                    )
                    context_rows.append(_context_to_row(context))

                    t0_action_index = int(t0_frame * args.action_per_frame)
                    future_action_count = int(args.horizon_frames * args.action_per_frame)
                    target_raw_frames = _decoded_raw_frames_for_latents(
                        args.horizon_frames,
                        action_per_frame=args.action_per_frame,
                    )
                    demo_future_actions = actions[t0_action_index : t0_action_index + future_action_count].copy()
                    gt_target_rgb: np.ndarray | None = None
                    for branch_name in branches:
                        branch_seed = int(args.seed + episode.dataset_episode_index * 1009 + t0_frame * 37 + branch_seed_offset(branch_name))
                        future_actions = apply_action_branch(
                            demo_future_actions,
                            branch_name=branch_name,
                            seed=branch_seed,
                        )
                        target_rgb = _render_future_from_state(
                            env=env,
                            flattened_state=context.simulator_state,
                            future_actions=future_actions,
                            target_raw_frames=target_raw_frames,
                        )
                        if branch_name == "gt":
                            gt_target_rgb = target_rgb
                        action_delta = _action_delta_stats(future_actions, demo_future_actions)
                        target_motion = _target_motion_stats(target_rgb)
                        target_delta = (
                            _target_delta_stats(target_rgb, gt_target_rgb)
                            if gt_target_rgb is not None
                            else {"target_vs_gt_rgb_mse": None, "target_vs_gt_rgb_abs_mean": None}
                        )
                        sample_path = output_root / "samples" / f"sample_{sample_id:06d}.npz"
                        np.savez_compressed(
                            sample_path,
                            target_rgb=target_rgb,
                            future_actions=future_actions.astype(np.float32, copy=False),
                        )
                        preview_path = None
                        if preview_count < args.preview_count:
                            preview_path = output_root / "previews" / f"sample_{sample_id:06d}_{branch_name}.mp4"
                            _write_rgb_video(preview_path, target_rgb, fps=args.video_fps)
                            preview_count += 1
                        sample_rows.append(
                            {
                                "sample_id": sample_id,
                                "context_id": context.context_id,
                                **_episode_to_row(episode),
                                "t0_frame": int(t0_frame),
                                "requested_t0_fraction": float(requested_t0_fraction),
                                "effective_t0_fraction": float(t0_frame / max(1, total_video_frames)),
                                "context_start_frame": int(context.context_start_frame),
                                "branch": branch_name,
                                **{
                                    f"branch_{key}": value
                                    for key, value in branch_metadata(branch_name).items()
                                    if key != "name"
                                },
                                "branch_seed": branch_seed,
                                "horizon_frames": int(args.horizon_frames),
                                "target_raw_frames": int(target_raw_frames),
                                "future_action_count": int(future_action_count),
                                **action_delta,
                                **target_motion,
                                **target_delta,
                                "sample_path": _relative_to(sample_path, output_root),
                                "context_path": _relative_to(context.context_path, output_root),
                                "preview_path": None if preview_path is None else _relative_to(preview_path, output_root),
                                "text_conditioning": "drop_for_fdm_future_denoising",
                            }
                        )
                        sample_id += 1
                        _write_progress(
                            output_root,
                            completed_transitions=sample_id,
                            target_transitions=args.target_transitions,
                            completed_contexts=len(context_rows),
                        )
                    context_id += 1
        finally:
            env.close()

    manifest["actual_transitions"] = len(sample_rows)
    manifest["actual_contexts"] = len(context_rows)
    manifest["preview_count"] = preview_count
    manifest["task_transition_counts"] = dict(sorted(Counter(row["task_id"] for row in sample_rows).items()))
    manifest["task_episode_counts"] = dict(sorted(Counter(episode.task_id for episode in source_episodes).items()))
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "metadata" / "contexts.jsonl", context_rows)
    _write_jsonl(output_root / "metadata" / "transitions.jsonl", sample_rows)
    _write_csv(output_root / "metadata" / "transitions.csv", sample_rows)
    _write_json(
        output_root / "summary.json",
        {
            "output_root": str(output_root),
            "transition_count": len(sample_rows),
            "context_count": len(context_rows),
            "preview_count": preview_count,
            "task_transition_counts": manifest["task_transition_counts"],
            "task_episode_counts": manifest["task_episode_counts"],
            "task_ids": manifest["task_ids"],
            "branches": list(branches),
            "t0_fractions": list(t0_fractions),
            "manifest": str(output_root / "manifest.json"),
            "transitions_jsonl": str(output_root / "metadata" / "transitions.jsonl"),
            "contexts_jsonl": str(output_root / "metadata" / "contexts.jsonl"),
        },
    )
    print(json.dumps(json.loads((output_root / "summary.json").read_text(encoding="utf-8")), indent=2))


def _render_context_at_t0(
    *,
    env: Any,
    init_state: Any,
    episode: SourceEpisode,
    actions: np.ndarray,
    t0_frame: int,
    requested_t0_fraction: float,
    total_video_frames: int,
    context_window_frames: int,
    action_per_frame: int,
    output_root: Path,
    context_id: int,
) -> ContextArtifact:
    context_start_frame = max(0, int(t0_frame) - int(context_window_frames))
    context_action_start = int(context_start_frame * action_per_frame)
    t0_action_index = int(t0_frame * action_per_frame)
    context_raw_frames = _raw_window_frames_for_latents(
        int(t0_frame - context_start_frame),
        action_per_frame=action_per_frame,
    )
    obs = env.reset()
    obs = env.set_init_state(init_state)
    for action_index in range(context_action_start):
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))

    context_frames: list[np.ndarray] = []
    for raw_offset in range(context_raw_frames):
        action_index = context_action_start + raw_offset
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))
        context_frames.append(_compose_obs_rgb(_extract_obs(obs)))

    for action_index in range(context_action_start + context_raw_frames, t0_action_index):
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))

    flattened_state = env.sim.get_state().flatten().copy()
    context_rgb = np.stack(context_frames, axis=0).astype(np.uint8, copy=False)
    action_context = actions[context_action_start:t0_action_index].astype(np.float32, copy=False)
    context_path = output_root / "contexts" / f"context_{context_id:06d}.npz"
    np.savez_compressed(
        context_path,
        context_rgb=context_rgb,
        action_context=action_context,
        simulator_state=flattened_state,
    )
    return ContextArtifact(
        context_id=context_id,
        episode=episode,
        t0_frame=int(t0_frame),
        requested_t0_fraction=float(requested_t0_fraction),
        total_video_frames=int(total_video_frames),
        context_start_frame=context_start_frame,
        context_path=context_path,
        action_context_shape=tuple(action_context.shape),
        context_rgb_shape=tuple(context_rgb.shape),
        simulator_state=flattened_state,
    )


def _render_future_from_state(
    *,
    env: Any,
    flattened_state: np.ndarray,
    future_actions: np.ndarray,
    target_raw_frames: int,
) -> np.ndarray:
    env.sim.set_state_from_flattened(flattened_state.copy())
    env.sim.forward()
    target_frames: list[np.ndarray] = []
    for action in future_actions[:target_raw_frames]:
        obs, _, _, _ = env.step(action.astype(np.float32, copy=False))
        target_frames.append(_compose_obs_rgb(_extract_obs(obs)))
    return np.stack(target_frames, axis=0).astype(np.uint8, copy=False)


def _select_source_episodes(
    replay_rows: list[dict[str, Any]],
    *,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    seed: int,
    excluded_episode_indices: set[int] | None = None,
) -> list[SourceEpisode]:
    rng = np.random.default_rng(int(seed))
    excluded_episode_indices = excluded_episode_indices or set()
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        if row.get("replay_status") != "success" or row.get("failure"):
            continue
        dataset_episode_index_value = row.get("dataset_episode_index")
        if dataset_episode_index_value is None:
            continue
        dataset_episode_index = int(dataset_episode_index_value)
        if dataset_episode_index in excluded_episode_indices:
            continue
        task_id = int(row.get("metadata_task_index") if row.get("metadata_task_index") is not None else -1)
        if task_id < 0:
            continue
        init_state_index = row.get("resolved_init_state_index")
        if init_state_index is None:
            init_state_index = row.get("primary_init_state_index")
        parquet_path_value = str(row.get("parquet_path", ""))
        if init_state_index is None or not parquet_path_value:
            continue
        by_task[task_id].append(row)

    selected: list[SourceEpisode] = []
    for task_id in task_ids:
        rows = sorted(by_task.get(task_id, ()), key=lambda item: int(item["dataset_episode_index"]))
        if len(rows) < episodes_per_task:
            raise ValueError(f"Task {task_id} has only {len(rows)} usable successful rows; need {episodes_per_task}.")
        indices = np.sort(rng.choice(len(rows), size=episodes_per_task, replace=False))
        for index in indices:
            row = rows[int(index)]
            init_state_index = row.get("resolved_init_state_index")
            if init_state_index is None:
                init_state_index = row.get("primary_init_state_index")
            selected.append(
                SourceEpisode(
                    dataset_episode_index=int(row["dataset_episode_index"]),
                    task_id=task_id,
                    task_text=str(row["task_text"]),
                    init_state_index=int(init_state_index),
                    parquet_path=Path(str(row["parquet_path"])),
                )
            )
    selected.sort(key=lambda episode: (episode.task_id, episode.dataset_episode_index))
    return selected


def _load_excluded_episode_indices(dataset_roots: list[str]) -> set[int]:
    excluded: set[int] = set()
    for root_value in dataset_roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Excluded dataset root does not exist: {root}")
        manifest_paths = _find_dataset_manifests(root)
        if not manifest_paths:
            raise FileNotFoundError(f"No manifest.json files found under excluded dataset root: {root}")
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for episode in manifest.get("source_episodes", ()):
                excluded.add(int(episode["dataset_episode_index"]))
    return excluded


def _find_dataset_manifests(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if (root / "manifest.json").is_file():
        candidates.append(root / "manifest.json")

    aggregate_summary = root / "aggregate_summary.json"
    if aggregate_summary.is_file():
        summary = json.loads(aggregate_summary.read_text(encoding="utf-8"))
        for shard in summary.get("shards", ()):
            manifest_value = shard.get("manifest")
            if not manifest_value:
                continue
            manifest_path = Path(str(manifest_value)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = root / manifest_path
            candidates.append(manifest_path)

    candidates.extend(sorted(root.glob("shard_tasks_*/manifest.json")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _select_t0_frames(
    *,
    total_video_frames: int,
    horizon_frames: int,
    context_window_frames: int,
    t0_fractions: tuple[float, ...],
) -> tuple[tuple[float, int], ...]:
    min_t0 = max(1, int(context_window_frames))
    max_t0 = int(total_video_frames) - int(horizon_frames) - 1
    if max_t0 < min_t0:
        raise ValueError(
            f"Episode too short for context={context_window_frames}, horizon={horizon_frames}: "
            f"total_video_frames={total_video_frames}."
        )
    selected: list[tuple[float, int]] = []
    used_frames: set[int] = set()
    for fraction in t0_fractions:
        value = int(round(total_video_frames * float(fraction)))
        value = min(max(min_t0, value), max_t0)
        if value in used_frames:
            for candidate in range(min_t0, max_t0 + 1):
                if candidate not in used_frames:
                    value = candidate
                    break
        selected.append((float(fraction), int(value)))
        used_frames.add(int(value))
    return tuple(selected)


def _read_actions(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["action"])
    return np.asarray(table.column("action").to_pylist(), dtype=np.float32)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _action_delta_stats(future_actions: np.ndarray, demo_future_actions: np.ndarray) -> dict[str, float]:
    future = np.asarray(future_actions, dtype=np.float32)
    demo = np.asarray(demo_future_actions, dtype=np.float32)
    rows = min(future.shape[0], demo.shape[0])
    channels = min(future.shape[1], demo.shape[1])
    if rows <= 0 or channels <= 0:
        return {
            "action_delta_abs_mean": 0.0,
            "action_delta_l2_mean": 0.0,
            "action_delta_l2_max": 0.0,
        }
    delta = future[:rows, :channels] - demo[:rows, :channels]
    l2 = np.linalg.norm(delta, axis=1)
    return {
        "action_delta_abs_mean": float(np.mean(np.abs(delta))),
        "action_delta_l2_mean": float(np.mean(l2)),
        "action_delta_l2_max": float(np.max(l2)),
    }


def _target_motion_stats(target_rgb: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(target_rgb, dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.0001:
        rgb = rgb / 255.0
    if rgb.shape[0] <= 1:
        interframe = np.asarray([0.0], dtype=np.float32)
    else:
        interframe = np.mean(np.abs(rgb[1:] - rgb[:-1]), axis=(1, 2, 3))
    endpoint = float(np.mean(np.abs(rgb[-1] - rgb[0]))) if rgb.shape[0] else 0.0
    return {
        "target_interframe_abs_mean": float(np.mean(interframe)),
        "target_interframe_abs_max": float(np.max(interframe)),
        "target_endpoint_abs_mean": endpoint,
    }


def _target_delta_stats(target_rgb: np.ndarray, gt_target_rgb: np.ndarray) -> dict[str, float]:
    target = np.asarray(target_rgb, dtype=np.float32)
    gt = np.asarray(gt_target_rgb, dtype=np.float32)
    frames = min(target.shape[0], gt.shape[0])
    if frames <= 0:
        return {"target_vs_gt_rgb_mse": 0.0, "target_vs_gt_rgb_abs_mean": 0.0}
    target = target[:frames]
    gt = gt[:frames]
    if target.size and float(np.nanmax(target)) > 1.0001:
        target = target / 255.0
    if gt.size and float(np.nanmax(gt)) > 1.0001:
        gt = gt / 255.0
    delta = target - gt
    return {
        "target_vs_gt_rgb_mse": float(np.mean(np.square(delta))),
        "target_vs_gt_rgb_abs_mean": float(np.mean(np.abs(delta))),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_progress(
    output_root: Path,
    *,
    completed_transitions: int,
    target_transitions: int,
    completed_contexts: int,
) -> None:
    _write_json(
        output_root / "latest_progress.json",
        {
            "completed_transitions": int(completed_transitions),
            "target_transitions": int(target_transitions),
            "completed_contexts": int(completed_contexts),
        },
    )


def _write_rgb_video(path: Path, rgb: np.ndarray, *, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [_as_uint8(frame) for frame in rgb], fps=float(fps), macro_block_size=1)


def _extract_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def _compose_obs_rgb(obs: dict[str, np.ndarray]) -> np.ndarray:
    left = _as_uint8(obs[LIBERO_OBS_KEYS[0]])
    right = _as_uint8(obs[LIBERO_OBS_KEYS[1]])
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Expected matching camera shapes, got {left.shape} and {right.shape}.")
    return np.concatenate([left, right], axis=1)


def _as_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    if array.size and float(np.nanmax(array)) <= 1.0001:
        return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _raw_window_frames_for_latents(latent_frames: int, *, action_per_frame: int) -> int:
    if latent_frames <= 0:
        raise ValueError(f"Expected positive latent frame count, got {latent_frames}.")
    return int(action_per_frame) * int(latent_frames) - 1


def _decoded_raw_frames_for_latents(latent_frames: int, *, action_per_frame: int) -> int:
    if latent_frames <= 0:
        raise ValueError(f"Expected positive latent frame count, got {latent_frames}.")
    return int(action_per_frame) * (int(latent_frames) - 1) + 1


def _episode_to_row(episode: SourceEpisode) -> dict[str, Any]:
    return {
        "dataset_episode_index": int(episode.dataset_episode_index),
        "task_id": int(episode.task_id),
        "task_text": episode.task_text,
        "init_state_index": int(episode.init_state_index),
        "parquet_path": str(episode.parquet_path),
    }


def _context_to_row(context: ContextArtifact) -> dict[str, Any]:
    return {
        "context_id": int(context.context_id),
        **_episode_to_row(context.episode),
        "t0_frame": int(context.t0_frame),
        "requested_t0_fraction": float(context.requested_t0_fraction),
        "effective_t0_fraction": float(context.t0_frame / max(1, context.total_video_frames)),
        "total_video_frames": int(context.total_video_frames),
        "context_start_frame": int(context.context_start_frame),
        "context_path": f"contexts/{context.context_path.name}",
        "action_context_shape": list(context.action_context_shape),
        "context_rgb_shape": list(context.context_rgb_shape),
    }


def _relative_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one task id.")
    return parsed


def _resolve_output_root(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now().strftime("libero10_fdm_counterfactual_demo_%Y%m%d_%H%M%S")
    return Path(args.output_dir).expanduser().resolve() / run_id


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LIBERO-10 counterfactual FDM demo dataset.")
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--replay-status-path", default=DEFAULT_REPLAY_STATUS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="libero10_fdm_counterfactual_demo_10000_h16_ctx16_seed0")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--target-transitions", type=int, default=10000)
    parser.add_argument("--episodes-per-task", type=int, default=25)
    parser.add_argument("--t0-fractions", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--branches", default=",".join(DEFAULT_BRANCHES))
    parser.add_argument(
        "--exclude-source-dataset-root",
        action="append",
        default=[],
        help=(
            "Dataset root whose manifest source episodes must be excluded from sampling. "
            "May be passed multiple times to build disjoint train/validation/test splits."
        ),
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=128,
        help="Post-#101 fixed training segment length used to derive the default future horizon.",
    )
    parser.add_argument(
        "--horizon-frames",
        type=int,
        default=None,
        help="Future latent frames to render. Defaults to segment_frames - context_window_frames.",
    )
    parser.add_argument("--context-window-frames", type=int, default=16)
    parser.add_argument("--action-per-frame", type=int, default=4)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--env-horizon", type=int, default=5000)
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--pyopengl-platform", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.target_transitions <= 0:
        parser.error("--target-transitions must be positive.")
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive.")
    if args.segment_frames <= 0 or args.context_window_frames <= 0:
        parser.error("--segment-frames and --context-window-frames must be positive.")
    if args.horizon_frames is None:
        args.horizon_frames = int(args.segment_frames) - int(args.context_window_frames)
    if args.horizon_frames <= 0:
        parser.error("--horizon-frames must be positive.")
    return args


if __name__ == "__main__":
    main()
