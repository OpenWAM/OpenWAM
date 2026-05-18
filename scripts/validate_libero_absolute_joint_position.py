from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from open_wam.integrations.libero_env import (
    absolute_joint_position_to_libero_joint_delta_action,
    build_libero_offscreen_env,
    extract_joint_positions_from_obs,
    infer_task_local_episode_rank,
    load_libero_task_init_states,
    resolve_libero_joint_delta_limit,
    resolve_libero_task,
)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_episode_rows(dataset_root, episode_index=args.episode_idx)
    if not rows:
        raise ValueError(f"Episode {args.episode_idx} has no rows under {dataset_root}.")
    episode_records = _load_episode_records(dataset_root)
    task_text = _episode_task_text(episode_records, episode_index=args.episode_idx)
    task_spec = resolve_libero_task(task_text, benchmark_name=args.benchmark)
    init_states = load_libero_task_init_states(task_spec)
    init_state_index = infer_task_local_episode_rank(
        episode_records,
        episode_index=args.episode_idx,
        task_text=task_text,
    )
    init_state_index = int(np.clip(init_state_index, 0, len(init_states) - 1))

    max_steps = min(int(args.max_steps), len(rows))
    record_frames = not args.no_video
    raw_replay = _rollout_raw_osc_actions(
        task_spec=task_spec,
        init_state=init_states[init_state_index],
        rows=rows[:max_steps],
        camera_key=args.camera_key,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        init_warmup_steps=args.init_warmup_steps,
        record_frames=record_frames,
        progress_interval=args.progress_interval,
        stop_on_success=args.stop_on_success,
    )
    absolute_replay = None
    if not args.raw_only:
        absolute_replay = _rollout_absolute_joint_targets(
            task_spec=task_spec,
            init_state=init_states[init_state_index],
            target_joint_positions=raw_replay["joint_positions"],
            raw_actions=raw_replay["raw_actions"],
            camera_key=args.camera_key,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            substeps_per_target=args.substeps_per_target,
            init_warmup_steps=args.init_warmup_steps,
            record_frames=record_frames,
            progress_interval=args.progress_interval,
            stop_on_success=args.stop_on_success,
        )

    summary = {
        "benchmark": args.benchmark,
        "task_id": task_spec.task_id,
        "task_text": task_text,
        "episode_idx": args.episode_idx,
        "init_state_index": init_state_index,
        "max_steps": max_steps,
        "substeps_per_target": args.substeps_per_target,
        "init_warmup_steps": args.init_warmup_steps,
        "raw_osc_steps": int(raw_replay["steps"]),
        "raw_osc_success": raw_replay["success"],
    }
    if absolute_replay is not None:
        summary.update(
            {
                "absolute_joint_steps": int(absolute_replay["steps"]),
                "absolute_joint_success": absolute_replay["success"],
                "absolute_joint_mean_qpos_l2": float(np.mean(absolute_replay["qpos_l2_errors"])),
                "absolute_joint_max_qpos_l2": float(np.max(absolute_replay["qpos_l2_errors"])),
                "absolute_joint_mean_qpos_linf": float(np.mean(absolute_replay["qpos_linf_errors"])),
                "absolute_joint_max_qpos_linf": float(np.max(absolute_replay["qpos_linf_errors"])),
                "joint_delta_limit_rad": absolute_replay["joint_delta_limit"].tolist(),
            }
        )
    file_stem = f"libero_abs_joint_ep{args.episode_idx}_steps{max_steps}_substeps{args.substeps_per_target}"
    summary_path = output_dir / f"{file_stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    video_path = None
    if not args.no_video and absolute_replay is not None:
        video_path = output_dir / f"{file_stem}.mp4"
        _write_side_by_side_video(
            video_path,
            left_frames=raw_replay["frames"],
            right_frames=absolute_replay["frames"],
            fps=args.video_fps,
        )

    print(json.dumps({**summary, "summary_path": str(summary_path), "video_path": str(video_path)}, indent=2))


def _rollout_raw_osc_actions(
    *,
    task_spec: Any,
    init_state: Any,
    rows: list[dict[str, Any]],
    camera_key: str,
    camera_height: int,
    camera_width: int,
    init_warmup_steps: int,
    record_frames: bool,
    progress_interval: int,
    stop_on_success: bool,
) -> dict[str, Any]:
    env = build_libero_offscreen_env(
        task_spec,
        controller="OSC_POSE",
        camera_height=camera_height,
        camera_width=camera_width,
        horizon=max(5000, len(rows) + 32),
        ignore_done=True,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(init_state)
        obs = _warmup_env(env, obs=obs, action_dim=7, steps=init_warmup_steps)
        joint_positions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        success = False
        for step_index, row in enumerate(rows):
            action = _row_action(row)
            obs, _, _, _ = env.step(action)
            joint_positions.append(extract_joint_positions_from_obs(obs))
            raw_actions.append(action)
            if record_frames and camera_key in obs:
                frames.append(np.asarray(obs[camera_key], dtype=np.uint8))
            success = success or bool(env.check_success())
            _print_progress("raw_osc", step_index + 1, len(rows), progress_interval=progress_interval)
            if success and stop_on_success:
                break
        return {
            "joint_positions": np.stack(joint_positions, axis=0),
            "raw_actions": np.stack(raw_actions, axis=0),
            "frames": frames,
            "steps": len(joint_positions),
            "success": success,
        }
    finally:
        env.close()


def _rollout_absolute_joint_targets(
    *,
    task_spec: Any,
    init_state: Any,
    target_joint_positions: np.ndarray,
    raw_actions: np.ndarray,
    camera_key: str,
    camera_height: int,
    camera_width: int,
    substeps_per_target: int,
    init_warmup_steps: int,
    record_frames: bool,
    progress_interval: int,
    stop_on_success: bool,
) -> dict[str, Any]:
    env = build_libero_offscreen_env(
        task_spec,
        controller="JOINT_POSITION",
        camera_height=camera_height,
        camera_width=camera_width,
        horizon=max(5000, int(target_joint_positions.shape[0] * max(1, substeps_per_target) + 32)),
        ignore_done=True,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(init_state)
        obs = _warmup_env(env, obs=obs, action_dim=8, steps=init_warmup_steps)
        joint_delta_limit = resolve_libero_joint_delta_limit(env, joint_dim=target_joint_positions.shape[-1])
        frames: list[np.ndarray] = []
        qpos_l2_errors: list[float] = []
        qpos_linf_errors: list[float] = []
        success = False
        for step_index, (target_qpos, raw_action) in enumerate(zip(target_joint_positions, raw_actions, strict=True)):
            for _ in range(max(1, int(substeps_per_target))):
                current_qpos = extract_joint_positions_from_obs(obs)
                env_action = absolute_joint_position_to_libero_joint_delta_action(
                    target_joint_positions=target_qpos,
                    current_joint_positions=current_qpos,
                    gripper_command=float(raw_action[-1]),
                    joint_delta_limit_rad=joint_delta_limit,
                )
                obs, _, _, _ = env.step(env_action)
            current_qpos = extract_joint_positions_from_obs(obs)
            error = current_qpos - np.asarray(target_qpos, dtype=np.float32)
            qpos_l2_errors.append(float(np.linalg.norm(error)))
            qpos_linf_errors.append(float(np.max(np.abs(error))))
            if record_frames and camera_key in obs:
                frames.append(np.asarray(obs[camera_key], dtype=np.uint8))
            success = success or bool(env.check_success())
            _print_progress(
                "absolute_joint",
                step_index + 1,
                int(target_joint_positions.shape[0]),
                progress_interval=progress_interval,
            )
            if success and stop_on_success:
                break
        return {
            "frames": frames,
            "qpos_l2_errors": np.asarray(qpos_l2_errors, dtype=np.float32),
            "qpos_linf_errors": np.asarray(qpos_linf_errors, dtype=np.float32),
            "joint_delta_limit": joint_delta_limit,
            "steps": len(qpos_l2_errors),
            "success": success,
        }
    finally:
        env.close()


def _warmup_env(env: Any, *, obs: dict[str, Any], action_dim: int, steps: int) -> dict[str, Any]:
    """Mirror LIBERO exact rollout initialization by stepping zero actions."""

    for _ in range(max(0, int(steps))):
        obs, _, _, _ = env.step(np.zeros(action_dim, dtype=np.float32))
    return obs


def _print_progress(phase: str, step: int, total: int, *, progress_interval: int) -> None:
    if progress_interval <= 0:
        return
    if step == 1 or step == total or step % progress_interval == 0:
        print(f"[{phase}] step={step}/{total}", file=sys.stderr, flush=True)


def _load_episode_rows(dataset_root: Path, *, episode_index: int) -> list[dict[str, Any]]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_path = info["data_path"].format(
        episode_chunk=int(episode_index) // int(info.get("chunks_size", 1000)),
        episode_index=int(episode_index),
    )
    table = pq.read_table(dataset_root / data_path)
    return table.to_pylist()


def _load_episode_records(dataset_root: Path) -> list[Any]:
    records: list[Any] = []
    with (dataset_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(SimpleNamespace(**payload))
    return records


def _episode_task_text(records: list[Any], *, episode_index: int) -> str:
    for record in records:
        if int(record.episode_index) == int(episode_index):
            if not record.tasks:
                raise ValueError(f"Episode {episode_index} has no task text.")
            return str(record.tasks[0])
    raise ValueError(f"Episode {episode_index} is not listed in meta/episodes.jsonl.")


def _row_action(row: dict[str, Any]) -> np.ndarray:
    if "action" in row:
        return np.asarray(row["action"], dtype=np.float32).reshape(-1)
    if "actions" in row:
        return np.asarray(row["actions"], dtype=np.float32).reshape(-1)
    raise KeyError("Episode row does not expose `action` or `actions`.")


def _write_side_by_side_video(
    path: Path,
    *,
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    fps: int,
) -> None:
    if not left_frames or not right_frames:
        return
    import imageio.v2 as imageio

    count = min(len(left_frames), len(right_frames))
    frames = []
    for left, right in zip(left_frames[:count], right_frames[:count], strict=True):
        if left.shape != right.shape:
            raise ValueError(f"Video frame shapes differ: {left.shape} vs {right.shape}.")
        frames.append(np.concatenate([left, right], axis=1))
    imageio.mimsave(path, frames, fps=int(fps))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LIBERO absolute joint-position replay by generating qpos targets from an OSC expert replay "
            "and replaying those targets through the JOINT_POSITION controller."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("OPENWAM_LIBERO_LEROBOT_ROOT"),
        required=os.environ.get("OPENWAM_LIBERO_LEROBOT_ROOT") is None,
        help="Local LeRobot LIBERO root, e.g. .../libero_heng/libero_10.",
    )
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--substeps-per-target", type=int, default=1)
    parser.add_argument("--init-warmup-steps", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", default="outputs/libero_absolute_joint_position_validation")
    return parser.parse_args()


if __name__ == "__main__":
    main()
