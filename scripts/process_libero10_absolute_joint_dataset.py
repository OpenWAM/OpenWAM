from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

from open_wam.configs import GripperRepresentation, LiberoAbsoluteJointExecutionMode
from open_wam.data.action_transforms import normalize_joint_positions
from open_wam.integrations.libero_env import LiberoBenchmarkAdapter, LiberoEnvConfig
from open_wam.simulators import EpisodeSpec
from open_wam.utils import load_experiment_config

from open_wam.integrations.libero_env import (
    absolute_joint_position_to_libero_joint_delta_action,
    ensure_local_libero_config,
    extract_joint_positions_from_obs,
    load_libero_task_init_states,
    resolve_libero_joint_delta_limit,
    resolve_libero_task_by_id,
    step_libero_absolute_joint_position_goal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_libero_absolute_joint_position import (
    _load_episode_records,
    _load_episode_rows,
    _row_action,
)


@dataclass(frozen=True)
class AbsoluteReplayProfile:
    name: str
    tracking_mode: str
    control_frequency_mode: str
    gripper_substep_policy: str
    gripper_tracking_mode: str
    gripper_target_delay_steps: int
    joint_substep_policy: str
    joint_settle_l2_tolerance: float | None
    joint_kp: float | None
    disable_interpolator: bool
    rollout_adapter_execution_mode: str = "normalized_delta"


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    replay_status_path = Path(args.replay_status_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    episodes_dir = output_root / "episodes"
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    replay_rows = _load_replay_status_rows(replay_status_path)
    selected_rows = [
        row for row in replay_rows if row.get("replay_status") == "success"
    ]
    selected_rows = _filter_episode_indices(selected_rows, args.episode_indices)
    if args.episode_start is not None:
        selected_rows = [
            row for row in selected_rows if int(row["dataset_episode_index"]) >= int(args.episode_start)
        ]
    if args.episode_end is not None:
        selected_rows = [
            row for row in selected_rows if int(row["dataset_episode_index"]) <= int(args.episode_end)
        ]
    if args.limit is not None:
        selected_rows = selected_rows[: int(args.limit)]

    episode_records = _load_episode_records(dataset_root)
    rollout_config = load_experiment_config(_resolve_repo_path(args.rollout_config))
    substeps_sweep = _parse_int_csv(args.substeps_sweep)
    absolute_profiles = _build_absolute_replay_profiles(args)
    task_cache: dict[int, tuple[Any, Any]] = {}
    integrated_delta_scale = _parse_integrated_delta_scale(args.integrated_delta_scale)
    raw_replay_cache: dict[int, dict[str, Any]] = {}
    if args.absolute_joint_target_source == "integrated_delta" and integrated_delta_scale is None:
        integrated_delta_scale = _estimate_integrated_delta_scale(
            selected_rows=selected_rows,
            dataset_root=dataset_root,
            episode_records=episode_records,
            task_cache=task_cache,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            warmup_override=args.init_warmup_steps,
            raw_stop_on_success=args.raw_stop_on_success,
            raw_cache=raw_replay_cache,
        )
        print(
            json.dumps(
                {
                    "event": "integrated_delta_scale_estimated",
                    "scale": integrated_delta_scale,
                    "episodes": len(raw_replay_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if integrated_delta_scale is None:
        integrated_delta_scale = 0.05
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.json"
    if args.summarize_manifest_only:
        records_by_episode = _load_latest_manifest_records(manifest_path)
        manifest_records = [
            records_by_episode[int(row["dataset_episode_index"])]
            for row in selected_rows
            if int(row["dataset_episode_index"]) in records_by_episode
        ]
        _write_summary(
            summary_path,
            records=manifest_records,
            total_selected=len(selected_rows),
            replay_status_path=replay_status_path,
            dataset_root=dataset_root,
            output_root=output_root,
            started_at=time.time(),
        )
        missing_episode_indices = [
            int(row["dataset_episode_index"])
            for row in selected_rows
            if int(row["dataset_episode_index"]) not in records_by_episode
        ]
        print(
            json.dumps(
                {
                    "event": "summary_written",
                    "summary_path": str(summary_path),
                    "manifest_records": len(records_by_episode),
                    "selected_records_summarized": len(manifest_records),
                    "missing_selected_records": len(missing_episode_indices),
                    "missing_episode_indices": missing_episode_indices[:20],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    completed = _load_completed_manifest(manifest_path) if args.resume else {}
    manifest_records: list[dict[str, Any]] = []
    started_at = time.time()

    print(
        json.dumps(
            {
                "event": "start",
                "dataset_root": str(dataset_root),
                "replay_status_path": str(replay_status_path),
                "output_root": str(output_root),
                "selected_metadata_success_rows": len(selected_rows),
                "substeps_sweep": substeps_sweep,
                "absolute_profiles": [profile.__dict__ for profile in absolute_profiles],
                "absolute_success_grace_steps": int(args.absolute_success_grace_steps),
                "rollout_config": str(_resolve_repo_path(args.rollout_config)),
                "resume_completed": len(completed),
            }
        ),
        flush=True,
    )

    for ordinal, row in enumerate(selected_rows, start=1):
        episode_index = int(row["dataset_episode_index"])
        sidecar_path = episodes_dir / f"episode_{episode_index:06d}.npz"
        if args.resume and episode_index in completed and sidecar_path.is_file():
            manifest_records.append(completed[episode_index])
            _print_event(
                "skip_completed",
                ordinal=ordinal,
                total=len(selected_rows),
                episode_index=episode_index,
                sidecar_path=str(sidecar_path),
            )
            continue

        record = _process_one_episode(
            row=row,
            dataset_root=dataset_root,
            episode_records=episode_records,
            sidecar_path=sidecar_path,
            task_cache=task_cache,
            substeps_sweep=substeps_sweep,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            camera_key=args.camera_key,
            raw_stop_on_success=args.raw_stop_on_success,
            absolute_stop_on_success=args.absolute_stop_on_success,
            absolute_profiles=absolute_profiles,
            absolute_success_grace_steps=args.absolute_success_grace_steps,
            rollout_data_config=rollout_config.data,
            absolute_joint_target_source=args.absolute_joint_target_source,
            integrated_delta_scale=float(integrated_delta_scale),
            raw_replay_cache=raw_replay_cache,
            warmup_override=args.init_warmup_steps,
        )
        manifest_records.append(record)
        _append_jsonl(manifest_path, record)
        _write_summary(
            summary_path,
            records=manifest_records,
            total_selected=len(selected_rows),
            replay_status_path=replay_status_path,
            dataset_root=dataset_root,
            output_root=output_root,
            started_at=started_at,
        )
        _print_event(
            "episode_done",
            ordinal=ordinal,
            total=len(selected_rows),
            episode_index=episode_index,
            status=record["conversion_status"],
            raw_success=record.get("raw_osc_success"),
            absolute_success=record.get("absolute_joint_success"),
            selected_substeps=record.get("selected_substeps_per_target"),
        )

    _write_summary(
        summary_path,
        records=manifest_records,
        total_selected=len(selected_rows),
        replay_status_path=replay_status_path,
        dataset_root=dataset_root,
        output_root=output_root,
        started_at=started_at,
    )


def _process_one_episode(
    *,
    row: dict[str, Any],
    dataset_root: Path,
    episode_records: list[Any],
    sidecar_path: Path,
    task_cache: dict[int, tuple[Any, Any]],
    substeps_sweep: list[int],
    camera_height: int,
    camera_width: int,
    camera_key: str,
    raw_stop_on_success: bool,
    absolute_stop_on_success: bool,
    absolute_profiles: list[AbsoluteReplayProfile],
    absolute_success_grace_steps: int,
    rollout_data_config: Any,
    absolute_joint_target_source: str,
    integrated_delta_scale: float,
    raw_replay_cache: dict[int, dict[str, Any]],
    warmup_override: int | None,
) -> dict[str, Any]:
    episode_index = int(row["dataset_episode_index"])
    task_id = int(row["upstream_task_id"])
    init_state_index = int(row["resolved_init_state_index"])
    warmup_steps = int(row.get("warmup_steps", 0) if warmup_override is None else warmup_override)
    raw_control_freq = int(row.get("env_control_freq", 20) or 20)
    rows = _load_episode_rows(dataset_root, episode_index=episode_index)
    benchmark = str(row.get("upstream_benchmark", "libero_10"))
    task_spec, init_states = _task_resources(task_cache, benchmark=benchmark, task_id=task_id)
    if init_state_index < 0 or init_state_index >= len(init_states):
        return _base_record(row, sidecar_path=sidecar_path, warmup_steps=warmup_steps) | {
            "conversion_status": "error",
            "failure_reason": "resolved_init_state_index_out_of_range",
            "init_state_count": len(init_states),
        }
    reset_seed = _resolved_reset_seed(row)

    if episode_index in raw_replay_cache:
        raw = raw_replay_cache[episode_index]
    else:
        try:
            raw = _rollout_raw_control_env(
                task_spec=task_spec,
                init_state=init_states[init_state_index],
                rows=rows,
                camera_height=camera_height,
                camera_width=camera_width,
                init_warmup_steps=warmup_steps,
                reset_seed=reset_seed,
                control_freq=raw_control_freq,
                stop_on_success=raw_stop_on_success,
            )
        except Exception as exc:
            return _base_record(row, sidecar_path=sidecar_path, warmup_steps=warmup_steps) | {
                "conversion_status": "raw_replay_error",
                "failure_reason": repr(exc),
            }

    base = _base_record(row, sidecar_path=sidecar_path, warmup_steps=warmup_steps) | {
        "raw_osc_success": bool(raw["success"]),
        "raw_osc_steps": int(raw["steps"]),
        "raw_recorded_qpos_steps": int(raw["joint_positions"].shape[0]),
        "reset_seed": int(reset_seed),
        "raw_control_freq": int(raw_control_freq),
    }
    if not bool(raw["success"]):
        return base | {
            "conversion_status": "raw_replay_failed",
            "failure_reason": "metadata_success_row_failed_raw_replay",
        }

    target_joint_positions = _build_joint_target_positions(
        raw=raw,
        target_source=absolute_joint_target_source,
        integrated_delta_scale=integrated_delta_scale,
    )
    target_fit_errors = target_joint_positions - raw["joint_positions"]
    target_fit_l2_errors = np.linalg.norm(target_fit_errors, axis=1).astype(np.float32)
    target_fit_linf_errors = np.max(np.abs(target_fit_errors), axis=1).astype(np.float32)
    absolute_joint_actions = _build_absolute_joint_actions(
        joint_positions_after_action=target_joint_positions,
        gripper_positions_after_action=raw["gripper_positions"],
        raw_osc_actions=raw["raw_actions"],
        rollout_data_config=rollout_data_config,
    )

    attempts: list[dict[str, Any]] = []
    selected_abs: dict[str, Any] | None = None
    selected_substeps: int | None = None
    selected_profile: AbsoluteReplayProfile | None = None
    for profile in absolute_profiles:
        for substeps in substeps_sweep:
            try:
                absolute = _rollout_absolute_control_env(
                    task_spec=task_spec,
                    init_state=init_states[init_state_index],
                    benchmark_name=str(row.get("upstream_benchmark", "libero_10")),
                    task_id=task_id,
                    init_state_index=init_state_index,
                    target_joint_positions=target_joint_positions,
                    reference_joint_positions=raw["joint_positions"],
                    target_gripper_positions=raw["gripper_positions"],
                    raw_actions=raw["raw_actions"],
                    rollout_data_config=rollout_data_config,
                    camera_height=camera_height,
                    camera_width=camera_width,
                    substeps_per_target=int(substeps),
                    init_warmup_steps=warmup_steps,
                    reset_seed=reset_seed,
                    raw_control_freq=raw_control_freq,
                    control_frequency_mode=profile.control_frequency_mode,
                    tracking_mode=profile.tracking_mode,
                    rollout_adapter_execution_mode=profile.rollout_adapter_execution_mode,
                    gripper_substep_policy=profile.gripper_substep_policy,
                    gripper_tracking_mode=profile.gripper_tracking_mode,
                    gripper_target_delay_steps=profile.gripper_target_delay_steps,
                    joint_substep_policy=profile.joint_substep_policy,
                    joint_settle_l2_tolerance=profile.joint_settle_l2_tolerance,
                    joint_kp=profile.joint_kp,
                    disable_interpolator=profile.disable_interpolator,
                    integrated_delta_scale=integrated_delta_scale,
                    stop_on_success=absolute_stop_on_success,
                    success_grace_steps=absolute_success_grace_steps,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "absolute_profile": profile.name,
                        "substeps_per_target": int(substeps),
                        "success": False,
                        "error": repr(exc),
                    }
                )
                continue

            attempt = {
                "absolute_profile": profile.name,
                "substeps_per_target": int(substeps),
                "absolute_control_freq": (
                    None if absolute["absolute_control_freq"] is None else int(absolute["absolute_control_freq"])
                ),
                "absolute_control_frequency_mode": str(absolute["absolute_control_frequency_mode"]),
                "absolute_tracking_mode": str(absolute["absolute_tracking_mode"]),
                "rollout_adapter_execution_mode": absolute.get("rollout_adapter_execution_mode"),
                "absolute_gripper_substep_policy": str(absolute["absolute_gripper_substep_policy"]),
                "absolute_gripper_tracking_mode": str(absolute["absolute_gripper_tracking_mode"]),
                "absolute_gripper_target_delay_steps": int(absolute["absolute_gripper_target_delay_steps"]),
                "absolute_joint_substep_policy": str(absolute["absolute_joint_substep_policy"]),
                "absolute_joint_settle_l2_tolerance": absolute["absolute_joint_settle_l2_tolerance"],
                "absolute_joint_kp": absolute["absolute_joint_kp"],
                "absolute_disable_interpolator": bool(absolute["absolute_disable_interpolator"]),
                "success": bool(absolute["success"]),
                "steps": int(absolute["steps"]),
                "success_grace_steps_taken": int(absolute["success_grace_steps_taken"]),
                "mean_qpos_l2": _safe_mean(absolute["qpos_l2_errors"]),
                "max_qpos_l2": _safe_max(absolute["qpos_l2_errors"]),
                "mean_qpos_linf": _safe_mean(absolute["qpos_linf_errors"]),
                "max_qpos_linf": _safe_max(absolute["qpos_linf_errors"]),
                "target_source": absolute_joint_target_source,
                "integrated_delta_scale": float(integrated_delta_scale),
                "mean_target_fit_l2": _safe_mean(target_fit_l2_errors),
                "max_target_fit_l2": _safe_max(target_fit_l2_errors),
            }
            attempts.append(attempt)
            if bool(absolute["success"]):
                selected_abs = absolute
                selected_substeps = int(substeps)
                selected_profile = profile
                break
        if selected_abs is not None:
            break

    if selected_abs is None or selected_substeps is None or selected_profile is None:
        return base | {
            "conversion_status": "absolute_joint_replay_failed",
            "absolute_joint_success": False,
            "absolute_joint_attempts": attempts,
            "failure_reason": "no_substeps_setting_reached_success",
        }

    _write_sidecar(
        sidecar_path,
        joint_positions_after_action=target_joint_positions,
        gripper_positions_after_action=raw["gripper_positions"],
        raw_osc_actions=raw["raw_actions"],
        absolute_joint_actions=absolute_joint_actions,
        qpos_l2_errors=selected_abs["qpos_l2_errors"],
        qpos_linf_errors=selected_abs["qpos_linf_errors"],
    )
    return base | {
        "conversion_status": "success",
        "absolute_joint_success": True,
        "absolute_joint_target_source": absolute_joint_target_source,
        "integrated_delta_scale": float(integrated_delta_scale),
        "integrated_delta_mean_target_fit_l2": _safe_mean(target_fit_l2_errors),
        "integrated_delta_max_target_fit_l2": _safe_max(target_fit_l2_errors),
        "integrated_delta_mean_target_fit_linf": _safe_mean(target_fit_linf_errors),
        "integrated_delta_max_target_fit_linf": _safe_max(target_fit_linf_errors),
        "selected_absolute_profile": selected_profile.name,
        "selected_substeps_per_target": selected_substeps,
        "selected_absolute_control_freq": (
            None if selected_abs["absolute_control_freq"] is None else int(selected_abs["absolute_control_freq"])
        ),
        "selected_success_grace_steps_taken": int(selected_abs["success_grace_steps_taken"]),
        "absolute_joint_steps": int(selected_abs["steps"]),
        "absolute_joint_action_dim": int(absolute_joint_actions.shape[1]),
        "absolute_joint_mean_qpos_l2": _safe_mean(selected_abs["qpos_l2_errors"]),
        "absolute_joint_max_qpos_l2": _safe_max(selected_abs["qpos_l2_errors"]),
        "absolute_joint_mean_qpos_linf": _safe_mean(selected_abs["qpos_linf_errors"]),
        "absolute_joint_max_qpos_linf": _safe_max(selected_abs["qpos_linf_errors"]),
        "absolute_joint_attempts": attempts,
        "sidecar_path": str(sidecar_path),
    }


def _base_record(row: dict[str, Any], *, sidecar_path: Path, warmup_steps: int) -> dict[str, Any]:
    return {
        "dataset_episode_index": int(row["dataset_episode_index"]),
        "task_text": row.get("task_text"),
        "upstream_benchmark": row.get("upstream_benchmark", "libero_10"),
        "upstream_task_id": int(row["upstream_task_id"]),
        "upstream_task_name": row.get("upstream_task_name"),
        "resolved_init_state_index": int(row["resolved_init_state_index"]),
        "source_replay_status": row.get("replay_status"),
        "source_success_step": row.get("success_step"),
        "source_recorded_length": row.get("recorded_length"),
        "warmup_steps": int(warmup_steps),
        "sidecar_path": str(sidecar_path),
    }


def _task_resources(task_cache: dict[int, tuple[Any, Any]], *, benchmark: str, task_id: int) -> tuple[Any, Any]:
    cache_key = hash((benchmark, int(task_id)))
    if cache_key not in task_cache:
        task_spec = resolve_libero_task_by_id(benchmark, task_id)
        init_states = load_libero_task_init_states(task_spec)
        task_cache[cache_key] = (task_spec, init_states)
    return task_cache[cache_key]


def _rollout_raw_control_env(
    *,
    task_spec: Any,
    init_state: Any,
    rows: list[dict[str, Any]],
    camera_height: int,
    camera_width: int,
    init_warmup_steps: int,
    reset_seed: int,
    control_freq: int,
    stop_on_success: bool,
) -> dict[str, Any]:
    env = _build_control_env(
        task_spec,
        controller="OSC_POSE",
        camera_height=camera_height,
        camera_width=camera_width,
        control_freq=control_freq,
    )
    try:
        obs = _reset_seeded_env(env, init_state=init_state, reset_seed=reset_seed)
        zero_action = np.zeros(7, dtype=np.float32)
        success = False
        for _ in range(max(0, int(init_warmup_steps))):
            obs, _, done, _ = env.step(zero_action)
            if _env_reached_success(env, done=bool(done)):
                success = True
                break

        initial_joint_positions = extract_joint_positions_from_obs(obs)
        joint_positions: list[np.ndarray] = []
        gripper_positions: list[np.ndarray] = []
        raw_actions: list[np.ndarray] = []
        if not success:
            for row in rows:
                action = _row_action(row)
                obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                joint_positions.append(extract_joint_positions_from_obs(obs))
                gripper_positions.append(_extract_gripper_positions_from_obs(obs))
                raw_actions.append(action.astype(np.float32, copy=False))
                if _env_reached_success(env, done=bool(done)):
                    success = True
                    if stop_on_success:
                        break
        if not joint_positions:
            joint_positions.append(extract_joint_positions_from_obs(obs))
            gripper_positions.append(_extract_gripper_positions_from_obs(obs))
            raw_actions.append(zero_action)
        return {
            "joint_positions": np.stack(joint_positions, axis=0).astype(np.float32),
            "gripper_positions": np.stack(gripper_positions, axis=0).astype(np.float32),
            "raw_actions": np.stack(raw_actions, axis=0).astype(np.float32),
            "initial_joint_positions": initial_joint_positions.astype(np.float32),
            "steps": len(joint_positions),
            "success": bool(success),
        }
    finally:
        env.close()


def _rollout_absolute_control_env(
    *,
    task_spec: Any,
    init_state: Any,
    benchmark_name: str,
    task_id: int,
    init_state_index: int,
    target_joint_positions: np.ndarray,
    reference_joint_positions: np.ndarray,
    target_gripper_positions: np.ndarray,
    raw_actions: np.ndarray,
    rollout_data_config: Any,
    camera_height: int,
    camera_width: int,
    substeps_per_target: int,
    init_warmup_steps: int,
    reset_seed: int,
    raw_control_freq: int,
    control_frequency_mode: str,
    tracking_mode: str,
    rollout_adapter_execution_mode: str,
    gripper_substep_policy: str,
    gripper_tracking_mode: str,
    gripper_target_delay_steps: int,
    joint_substep_policy: str,
    joint_settle_l2_tolerance: float | None,
    joint_kp: float | None,
    disable_interpolator: bool,
    integrated_delta_scale: float,
    stop_on_success: bool,
    success_grace_steps: int,
) -> dict[str, Any]:
    substeps = max(1, int(substeps_per_target))
    if tracking_mode == "rollout_adapter":
        if int(success_grace_steps) != 0:
            raise ValueError("rollout_adapter replay must use --absolute-success-grace-steps=0.")
        execution_mode = LiberoAbsoluteJointExecutionMode(rollout_adapter_execution_mode)
        if control_frequency_mode == "raw":
            absolute_control_freq = int(raw_control_freq)
        elif control_frequency_mode == "scaled_by_substeps":
            absolute_control_freq = int(raw_control_freq) * substeps
        else:
            raise ValueError(f"Unknown absolute control frequency mode: {control_frequency_mode!r}.")
        return _rollout_adapter_absolute_control_env(
            benchmark_name=benchmark_name,
            task_id=task_id,
            init_state_index=init_state_index,
            target_joint_positions=target_joint_positions,
            reference_joint_positions=reference_joint_positions,
            target_gripper_positions=target_gripper_positions,
            raw_actions=raw_actions,
            rollout_data_config=rollout_data_config,
            camera_height=camera_height,
            camera_width=camera_width,
            reset_seed=reset_seed,
            substeps_per_target=substeps,
            absolute_control_freq=absolute_control_freq,
            control_frequency_mode=control_frequency_mode,
            execution_mode=execution_mode,
            gripper_substep_policy=gripper_substep_policy,
            joint_kp=joint_kp,
            disable_interpolator=disable_interpolator,
            integrated_delta_scale=integrated_delta_scale,
            stop_on_success=stop_on_success,
        )
    if control_frequency_mode == "raw":
        absolute_control_freq = int(raw_control_freq)
    elif control_frequency_mode == "scaled_by_substeps":
        absolute_control_freq = int(raw_control_freq) * substeps
    else:
        raise ValueError(f"Unknown absolute control frequency mode: {control_frequency_mode!r}.")
    env = _build_control_env(
        task_spec,
        controller="JOINT_POSITION",
        camera_height=camera_height,
        camera_width=camera_width,
        control_freq=absolute_control_freq,
        horizon=max(5000, int(target_joint_positions.shape[0]) * substeps + max(128, int(init_warmup_steps))),
    )
    try:
        obs = _reset_seeded_env(env, init_state=init_state, reset_seed=reset_seed)
        if joint_kp is not None:
            _set_joint_position_controller_gain(env, kp=float(joint_kp))
        if disable_interpolator:
            _disable_joint_position_controller_interpolator(env)
        zero_action = np.zeros(8, dtype=np.float32)
        success = False
        for _ in range(max(0, int(init_warmup_steps))):
            obs, _, done, _ = env.step(zero_action)
            if _env_reached_success(env, done=bool(done)):
                success = True
                break

        joint_delta_limit = resolve_libero_joint_delta_limit(env, joint_dim=target_joint_positions.shape[-1])
        qpos_l2_errors: list[float] = []
        qpos_linf_errors: list[float] = []
        if not success:
            previous_target_qpos = np.asarray(extract_joint_positions_from_obs(obs), dtype=np.float32)
            last_target_qpos: np.ndarray | None = None
            last_target_gripper_qpos: np.ndarray | None = None
            last_raw_action: np.ndarray | None = None
            for target_index, (target_qpos, reference_qpos, target_gripper_qpos, raw_action) in enumerate(
                zip(
                    target_joint_positions,
                    reference_joint_positions,
                    target_gripper_positions,
                    raw_actions,
                    strict=True,
                )
            ):
                last_target_qpos = np.asarray(target_qpos, dtype=np.float32)
                last_target_gripper_qpos = np.asarray(target_gripper_qpos, dtype=np.float32)
                if int(gripper_target_delay_steps) > 0:
                    gripper_target_index = max(0, int(target_index) - int(gripper_target_delay_steps))
                    command_target_gripper_qpos = np.asarray(
                        target_gripper_positions[gripper_target_index],
                        dtype=np.float32,
                    )
                else:
                    command_target_gripper_qpos = last_target_gripper_qpos
                last_raw_action = np.asarray(raw_action, dtype=np.float32)
                done = False
                for substep_index in range(substeps):
                    command_qpos = _joint_position_command_for_substep(
                        previous_target_positions=previous_target_qpos,
                        target_joint_positions=last_target_qpos,
                        substep_index=substep_index,
                        substeps=substeps,
                        policy=joint_substep_policy,
                    )
                    gripper_command = _gripper_command_for_target(
                        obs,
                        raw_action_command=float(raw_action[-1]),
                        target_gripper_positions=command_target_gripper_qpos,
                        substep_index=substep_index,
                        substeps=substeps,
                        substep_policy=gripper_substep_policy,
                        tracking_mode=gripper_tracking_mode,
                    )
                    if tracking_mode == "normalized_delta":
                        current_qpos = extract_joint_positions_from_obs(obs)
                        env_action = absolute_joint_position_to_libero_joint_delta_action(
                            target_joint_positions=command_qpos,
                            current_joint_positions=current_qpos,
                            gripper_command=gripper_command,
                            joint_delta_limit_rad=joint_delta_limit,
                        )
                        obs, _, done, _ = env.step(env_action)
                    elif tracking_mode == "direct_goal":
                        obs, _, done, _ = step_libero_absolute_joint_position_goal(
                            env,
                            target_joint_positions=command_qpos,
                            gripper_command=gripper_command,
                        )
                    else:
                        raise ValueError(f"Unknown absolute tracking mode: {tracking_mode!r}.")
                    if joint_settle_l2_tolerance is not None:
                        current_qpos = extract_joint_positions_from_obs(obs)
                        if float(np.linalg.norm(current_qpos - last_target_qpos)) <= float(joint_settle_l2_tolerance):
                            break
                    if _env_reached_success(env, done=bool(done)):
                        break
                current_qpos = extract_joint_positions_from_obs(obs)
                error = current_qpos - np.asarray(reference_qpos, dtype=np.float32)
                qpos_l2_errors.append(float(np.linalg.norm(error)))
                qpos_linf_errors.append(float(np.max(np.abs(error))))
                previous_target_qpos = last_target_qpos
                if _env_reached_success(env, done=bool(done)):
                    success = True
                    if stop_on_success:
                        break
            grace_steps_taken = 0
            if (
                not success
                and int(success_grace_steps) > 0
                and last_target_qpos is not None
                and last_target_gripper_qpos is not None
                and last_raw_action is not None
            ):
                for grace_step_index in range(int(success_grace_steps)):
                    gripper_command = _gripper_command_for_target(
                        obs,
                        raw_action_command=float(last_raw_action[-1]),
                        target_gripper_positions=last_target_gripper_qpos,
                        substep_index=0,
                        substeps=1,
                        substep_policy="repeat",
                        tracking_mode=gripper_tracking_mode,
                    )
                    if tracking_mode == "normalized_delta":
                        current_qpos = extract_joint_positions_from_obs(obs)
                        env_action = absolute_joint_position_to_libero_joint_delta_action(
                            target_joint_positions=last_target_qpos,
                            current_joint_positions=current_qpos,
                            gripper_command=gripper_command,
                            joint_delta_limit_rad=joint_delta_limit,
                        )
                        obs, _, done, _ = env.step(env_action)
                    elif tracking_mode == "direct_goal":
                        obs, _, done, _ = step_libero_absolute_joint_position_goal(
                            env,
                            target_joint_positions=last_target_qpos,
                            gripper_command=gripper_command,
                        )
                    else:
                        raise ValueError(f"Unknown absolute tracking mode: {tracking_mode!r}.")
                    grace_steps_taken = grace_step_index + 1
                    if _env_reached_success(env, done=bool(done)):
                        success = True
                        break
        else:
            grace_steps_taken = 0
        return {
            "qpos_l2_errors": np.asarray(qpos_l2_errors, dtype=np.float32),
            "qpos_linf_errors": np.asarray(qpos_linf_errors, dtype=np.float32),
            "joint_delta_limit": joint_delta_limit,
            "steps": len(qpos_l2_errors),
            "success_grace_steps_taken": int(grace_steps_taken),
            "success": bool(success),
            "absolute_control_freq": int(absolute_control_freq),
            "absolute_control_frequency_mode": control_frequency_mode,
            "absolute_tracking_mode": tracking_mode,
            "absolute_gripper_substep_policy": gripper_substep_policy,
            "absolute_gripper_tracking_mode": gripper_tracking_mode,
            "absolute_gripper_target_delay_steps": int(gripper_target_delay_steps),
            "absolute_joint_substep_policy": joint_substep_policy,
            "absolute_joint_settle_l2_tolerance": (
                None if joint_settle_l2_tolerance is None else float(joint_settle_l2_tolerance)
            ),
            "absolute_joint_kp": None if joint_kp is None else float(joint_kp),
            "absolute_disable_interpolator": bool(disable_interpolator),
        }
    finally:
        env.close()


def _rollout_adapter_absolute_control_env(
    *,
    benchmark_name: str,
    task_id: int,
    init_state_index: int,
    target_joint_positions: np.ndarray,
    reference_joint_positions: np.ndarray,
    target_gripper_positions: np.ndarray,
    raw_actions: np.ndarray,
    rollout_data_config: Any,
    camera_height: int,
    camera_width: int,
    reset_seed: int,
    substeps_per_target: int,
    absolute_control_freq: int,
    control_frequency_mode: str,
    execution_mode: LiberoAbsoluteJointExecutionMode,
    gripper_substep_policy: str,
    joint_kp: float | None,
    disable_interpolator: bool,
    integrated_delta_scale: float,
    stop_on_success: bool,
) -> dict[str, Any]:
    """Replay GT absolute targets through the same adapter path used by inference."""

    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            benchmark_name=benchmark_name,
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            camera_height=camera_height,
            camera_width=camera_width,
            horizon=max(5000, int(target_joint_positions.shape[0]) * max(1, int(substeps_per_target)) + 128),
            ignore_done=False,
            control_freq=int(absolute_control_freq),
            init_state_index=int(init_state_index),
            absolute_joint_execution_mode=execution_mode,
            absolute_joint_substeps_per_target=int(substeps_per_target),
            absolute_joint_gripper_substep_policy=gripper_substep_policy,
            absolute_joint_kp=joint_kp,
            absolute_joint_disable_interpolator=bool(disable_interpolator),
            absolute_joint_delta_integration_scale=float(integrated_delta_scale),
        )
    )
    qpos_l2_errors: list[float] = []
    qpos_linf_errors: list[float] = []
    env_action_values: list[np.ndarray] = []
    gripper_representation = GripperRepresentation(rollout_data_config.action_target.gripper_representation)
    adapter_gripper_tracking_mode = (
        "raw_action_command"
        if gripper_representation == GripperRepresentation.ACTION_COMMAND
        else f"measured_{gripper_representation.value}_qpos"
    )
    success = False
    try:
        adapter.reset(EpisodeSpec(task_id=int(task_id), episode_idx=0, seed=int(reset_seed)))
        for target_qpos, reference_qpos, target_gripper_qpos, raw_action in zip(
            target_joint_positions,
            reference_joint_positions,
            target_gripper_positions,
            raw_actions,
            strict=True,
        ):
            normalized_target = normalize_joint_positions(
                torch.as_tensor(target_qpos, dtype=torch.float32).unsqueeze(0),
                normalization=rollout_data_config.action_target.joint_position_normalization,
            )[0].detach().cpu().numpy()
            if gripper_representation == GripperRepresentation.ACTION_COMMAND:
                gripper_tail = np.asarray([float(raw_action[-1])], dtype=np.float32)
            elif gripper_representation == GripperRepresentation.FIRST_CHANNEL:
                gripper_tail = np.asarray(target_gripper_qpos, dtype=np.float32).reshape(-1)[:1]
            elif gripper_representation == GripperRepresentation.ALL_CHANNELS:
                gripper_tail = np.asarray(target_gripper_qpos, dtype=np.float32).reshape(-1)
            else:
                raise ValueError(f"Unsupported gripper representation: {gripper_representation}")
            model_action = np.concatenate(
                [
                    normalized_target.astype(np.float32, copy=False),
                    gripper_tail.astype(np.float32, copy=False),
                ],
                axis=0,
            )
            env_action = adapter.action_from_model_action(model_action, data_config=rollout_data_config)
            env_action_values.append(np.asarray(env_action, dtype=np.float32))
            transition = adapter.step(env_action.astype(np.float32, copy=False))
            current_qpos = extract_joint_positions_from_obs(transition.observation.raw)
            error = current_qpos - np.asarray(reference_qpos, dtype=np.float32)
            qpos_l2_errors.append(float(np.linalg.norm(error)))
            qpos_linf_errors.append(float(np.max(np.abs(error))))
            if bool(transition.success):
                success = True
                if stop_on_success:
                    break
    finally:
        adapter.close()
    env_actions = np.stack(env_action_values, axis=0) if env_action_values else np.zeros((0, 8), dtype=np.float32)
    return {
        "qpos_l2_errors": np.asarray(qpos_l2_errors, dtype=np.float32),
        "qpos_linf_errors": np.asarray(qpos_linf_errors, dtype=np.float32),
        "joint_delta_limit": None,
        "steps": len(qpos_l2_errors),
        "success_grace_steps_taken": 0,
        "success": bool(success),
        "absolute_control_freq": int(absolute_control_freq),
        "absolute_control_frequency_mode": control_frequency_mode,
        "absolute_tracking_mode": "rollout_adapter",
        "absolute_gripper_substep_policy": gripper_substep_policy,
        "absolute_gripper_tracking_mode": adapter_gripper_tracking_mode,
        "absolute_gripper_target_delay_steps": 0,
        "absolute_joint_substep_policy": f"adapter_{execution_mode.value}",
        "absolute_joint_settle_l2_tolerance": None,
        "absolute_joint_kp": None if joint_kp is None else float(joint_kp),
        "absolute_disable_interpolator": bool(disable_interpolator),
        "rollout_adapter_execution_mode": execution_mode.value,
        "env_action_mean_abs": np.abs(env_actions).mean(axis=0) if env_actions.size else np.zeros(0, dtype=np.float32),
        "env_action_max_abs": np.abs(env_actions).max(axis=0) if env_actions.size else np.zeros(0, dtype=np.float32),
    }


def _gripper_command_for_target(
    obs: dict[str, Any],
    *,
    raw_action_command: float,
    target_gripper_positions: np.ndarray,
    substep_index: int,
    substeps: int,
    substep_policy: str,
    tracking_mode: str,
) -> float:
    if tracking_mode == "raw_action":
        return _raw_gripper_command_for_substep(
            raw_action_command,
            substep_index=substep_index,
            substeps=substeps,
            policy=substep_policy,
        )
    if tracking_mode == "observed_qpos":
        del substep_index, substeps, substep_policy
        return _gripper_qpos_tracking_command(
            current_gripper_positions=_extract_gripper_positions_from_obs(obs),
            target_gripper_positions=target_gripper_positions,
        )
    raise ValueError(f"Unknown gripper tracking mode: {tracking_mode!r}.")


def _joint_position_command_for_substep(
    *,
    previous_target_positions: np.ndarray,
    target_joint_positions: np.ndarray,
    substep_index: int,
    substeps: int,
    policy: str,
) -> np.ndarray:
    target = np.asarray(target_joint_positions, dtype=np.float32)
    if policy == "hold":
        return target
    if policy == "linear":
        previous = np.asarray(previous_target_positions, dtype=np.float32)
        alpha = float(int(substep_index) + 1) / float(max(1, int(substeps)))
        return (previous + alpha * (target - previous)).astype(np.float32)
    raise ValueError(f"Unknown joint substep policy: {policy!r}.")


def _set_joint_position_controller_gain(env: Any, *, kp: float) -> None:
    robot = getattr(env, "robots", None)
    if robot is None:
        robot = getattr(getattr(env, "env", env), "robots", None)
    if not robot:
        raise ValueError("LIBERO env does not expose robot handles for controller gain override.")
    controller = getattr(robot[0], "controller", None)
    if controller is None:
        raise ValueError("LIBERO env robot does not expose a controller for gain override.")
    joint_dim = int(getattr(controller, "control_dim", 7))
    controller.kp = np.full(joint_dim, float(kp), dtype=np.float64)
    controller.kd = 2.0 * np.sqrt(controller.kp)


def _disable_joint_position_controller_interpolator(env: Any) -> None:
    robot = getattr(env, "robots", None)
    if robot is None:
        robot = getattr(getattr(env, "env", env), "robots", None)
    if not robot:
        raise ValueError("LIBERO env does not expose robot handles for controller interpolator override.")
    controller = getattr(robot[0], "controller", None)
    if controller is None:
        raise ValueError("LIBERO env robot does not expose a controller for interpolator override.")
    controller.interpolator = None


def _env_reached_success(env: Any, *, done: bool) -> bool:
    """Mirror the replay metadata predicate while retaining done compatibility."""

    if bool(done):
        return True
    check_success = getattr(env, "check_success", None)
    if check_success is None:
        check_success = getattr(getattr(env, "env", env), "check_success", None)
    if check_success is None:
        return False
    return bool(check_success())


def _raw_gripper_command_for_substep(
    command: float,
    *,
    substep_index: int,
    substeps: int,
    policy: str,
) -> float:
    if policy == "repeat":
        return float(command)
    if policy == "first_only":
        return float(command) if int(substep_index) == 0 else 0.0
    if policy == "last_only":
        return float(command) if int(substep_index) == int(substeps) - 1 else 0.0
    raise ValueError(f"Unknown gripper substep policy: {policy!r}.")


def _gripper_qpos_tracking_command(
    *,
    current_gripper_positions: np.ndarray,
    target_gripper_positions: np.ndarray,
    tolerance: float = 0.001,
) -> float:
    current_opening = _gripper_opening(current_gripper_positions)
    target_opening = _gripper_opening(target_gripper_positions)
    if current_opening > target_opening + float(tolerance):
        return 1.0
    if current_opening < target_opening - float(tolerance):
        return -1.0
    return 0.0


def _extract_gripper_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_gripper_qpos" not in obs:
        raise KeyError("LIBERO observation does not expose `robot0_gripper_qpos`.")
    values = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LIBERO `robot0_gripper_qpos` is empty.")
    return values


def _gripper_opening(gripper_positions: np.ndarray) -> float:
    values = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
    if values.size >= 2:
        return float(values[0] - values[1])
    return float(values[0])


def _build_control_env(
    task_spec: Any,
    *,
    controller: str,
    camera_height: int,
    camera_width: int,
    control_freq: int,
    horizon: int = 5000,
) -> Any:
    ensure_local_libero_config()
    from libero.libero.envs.env_wrapper import ControlEnv  # type: ignore

    return ControlEnv(
        bddl_file_name=task_spec.bddl_file_path,
        controller=controller,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        horizon=int(horizon),
        ignore_done=False,
        control_freq=int(control_freq),
        camera_heights=camera_height,
        camera_widths=camera_width,
    )


def _reset_seeded_env(env: Any, *, init_state: Any, reset_seed: int) -> dict[str, Any]:
    random.seed(int(reset_seed))
    np.random.seed(int(reset_seed) % (2**32 - 1))
    env.seed(int(reset_seed))
    env.reset()
    return env.set_init_state(init_state)


def _resolved_reset_seed(row: dict[str, Any]) -> int:
    resolved_init = int(row["resolved_init_state_index"])
    attempts = row.get("attempts") or []
    for attempt in attempts:
        if int(attempt.get("init_state_index", -1)) == resolved_init and attempt.get("success_step") is not None:
            return int(attempt["reset_seed"])
    for attempt in attempts:
        if int(attempt.get("init_state_index", -1)) == resolved_init and attempt.get("reset_seed") is not None:
            return int(attempt["reset_seed"])
    raise ValueError(f"Could not find reset_seed for episode {row.get('dataset_episode_index')}.")


def _write_sidecar(
    path: Path,
    *,
    joint_positions_after_action: np.ndarray,
    gripper_positions_after_action: np.ndarray,
    raw_osc_actions: np.ndarray,
    absolute_joint_actions: np.ndarray,
    qpos_l2_errors: np.ndarray,
    qpos_linf_errors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            joint_positions_after_action=joint_positions_after_action.astype(np.float32),
            gripper_positions_after_action=gripper_positions_after_action.astype(np.float32),
            raw_osc_actions=raw_osc_actions.astype(np.float32),
            absolute_joint_actions=absolute_joint_actions.astype(np.float32),
            qpos_l2_errors=qpos_l2_errors.astype(np.float32),
            qpos_linf_errors=qpos_linf_errors.astype(np.float32),
        )
    tmp_path.replace(path)


def _build_joint_target_positions(
    *,
    raw: dict[str, Any],
    target_source: str,
    integrated_delta_scale: float,
) -> np.ndarray:
    if target_source == "measured_qpos":
        return np.asarray(raw["joint_positions"], dtype=np.float32)
    if target_source == "integrated_delta":
        return integrate_delta_joint_positions(
            initial_joint_positions=np.asarray(raw["initial_joint_positions"], dtype=np.float32),
            delta_actions=np.asarray(raw["raw_actions"], dtype=np.float32),
            scale=float(integrated_delta_scale),
            joint_dim=np.asarray(raw["joint_positions"], dtype=np.float32).shape[-1],
        )
    raise ValueError(f"Unknown absolute joint target source: {target_source!r}.")


def integrate_delta_joint_positions(
    *,
    initial_joint_positions: np.ndarray,
    delta_actions: np.ndarray,
    scale: float,
    joint_dim: int,
) -> np.ndarray:
    """Integrate normalized delta-joint commands into pseudo-absolute qpos targets."""

    initial = np.asarray(initial_joint_positions, dtype=np.float32).reshape(-1)
    deltas = np.asarray(delta_actions, dtype=np.float32)
    if deltas.ndim != 2:
        raise ValueError(f"Expected delta actions with shape [T, D], got {deltas.shape}.")
    if initial.shape[0] < int(joint_dim):
        raise ValueError(f"Initial joint state has dim {initial.shape[0]}, expected at least {joint_dim}.")
    if deltas.shape[1] < int(joint_dim):
        raise ValueError(f"Delta action dim {deltas.shape[1]} is smaller than joint dim {joint_dim}.")
    cumulative = np.cumsum(deltas[:, : int(joint_dim)], axis=0, dtype=np.float32)
    return initial[: int(joint_dim)][None, :] + cumulative * float(scale)


def fit_integrated_delta_scale(
    *,
    initial_joint_positions: np.ndarray,
    delta_actions: np.ndarray,
    measured_joint_positions: np.ndarray,
) -> float:
    numerator, denominator = _integrated_delta_fit_terms(
        initial_joint_positions=initial_joint_positions,
        delta_actions=delta_actions,
        measured_joint_positions=measured_joint_positions,
    )
    if denominator <= 1e-12:
        raise ValueError("Cannot fit integrated-delta scale because delta actions are all zero.")
    return float(numerator / denominator)


def _integrated_delta_fit_terms(
    *,
    initial_joint_positions: np.ndarray,
    delta_actions: np.ndarray,
    measured_joint_positions: np.ndarray,
) -> tuple[float, float]:
    measured = np.asarray(measured_joint_positions, dtype=np.float32)
    if measured.ndim != 2:
        raise ValueError(f"Expected measured joint positions with shape [T, J], got {measured.shape}.")
    joint_dim = measured.shape[-1]
    initial = np.asarray(initial_joint_positions, dtype=np.float32).reshape(-1)[:joint_dim]
    deltas = np.asarray(delta_actions, dtype=np.float32)
    if deltas.ndim != 2 or deltas.shape[0] != measured.shape[0] or deltas.shape[1] < joint_dim:
        raise ValueError(
            "Delta actions must have shape [T, D>=J] matching measured joint positions; "
            f"got deltas={deltas.shape}, measured={measured.shape}."
        )
    cumulative = np.cumsum(deltas[:, :joint_dim], axis=0, dtype=np.float32)
    centered_measured = measured - initial[None, :]
    return float(np.sum(cumulative * centered_measured)), float(np.sum(cumulative * cumulative))


def _parse_integrated_delta_scale(raw: str) -> float | None:
    if str(raw).strip().lower() == "auto":
        return None
    value = float(raw)
    if not math.isfinite(value) or math.isclose(value, 0.0):
        raise ValueError("--integrated-delta-scale must be a nonzero finite float or 'auto'.")
    return value


def _estimate_integrated_delta_scale(
    *,
    selected_rows: list[dict[str, Any]],
    dataset_root: Path,
    episode_records: list[Any],
    task_cache: dict[int, tuple[Any, Any]],
    camera_height: int,
    camera_width: int,
    warmup_override: int | None,
    raw_stop_on_success: bool,
    raw_cache: dict[int, dict[str, Any]],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in selected_rows:
        episode_index = int(row["dataset_episode_index"])
        task_id = int(row["upstream_task_id"])
        init_state_index = int(row["resolved_init_state_index"])
        warmup_steps = int(row.get("warmup_steps", 0) if warmup_override is None else warmup_override)
        raw_control_freq = int(row.get("env_control_freq", 20) or 20)
        benchmark = str(row.get("upstream_benchmark", "libero_10"))
        rows = _load_episode_rows(dataset_root, episode_index=episode_index)
        task_spec, init_states = _task_resources(task_cache, benchmark=benchmark, task_id=task_id)
        reset_seed = _resolved_reset_seed(row)
        raw = _rollout_raw_control_env(
            task_spec=task_spec,
            init_state=init_states[init_state_index],
            rows=rows,
            camera_height=camera_height,
            camera_width=camera_width,
            init_warmup_steps=warmup_steps,
            reset_seed=reset_seed,
            control_freq=raw_control_freq,
            stop_on_success=raw_stop_on_success,
        )
        raw_cache[episode_index] = raw
        if not bool(raw["success"]):
            continue
        ep_num, ep_den = _integrated_delta_fit_terms(
            initial_joint_positions=raw["initial_joint_positions"],
            delta_actions=raw["raw_actions"],
            measured_joint_positions=raw["joint_positions"],
        )
        numerator += ep_num
        denominator += ep_den
    if denominator <= 1e-12:
        raise ValueError("Cannot estimate integrated-delta scale; no successful raw replay had nonzero deltas.")
    scale = numerator / denominator
    if not math.isfinite(scale) or math.isclose(scale, 0.0):
        raise ValueError(f"Estimated invalid integrated-delta scale: {scale}.")
    return float(scale)


def _build_absolute_joint_actions(
    *,
    joint_positions_after_action: np.ndarray,
    gripper_positions_after_action: np.ndarray,
    raw_osc_actions: np.ndarray,
    rollout_data_config: Any,
) -> np.ndarray:
    """Build model-facing absolute-joint targets using the rollout config contract."""

    joint_positions = np.asarray(joint_positions_after_action, dtype=np.float32)
    gripper_positions = np.asarray(gripper_positions_after_action, dtype=np.float32)
    raw_actions = np.asarray(raw_osc_actions, dtype=np.float32)
    if joint_positions.ndim != 2:
        raise ValueError(f"Expected joint_positions_after_action with shape [T, J], got {joint_positions.shape}.")
    if gripper_positions.ndim != 2 or gripper_positions.shape[0] != joint_positions.shape[0]:
        raise ValueError(
            "Expected gripper_positions_after_action with shape [T, G] matching joint positions, "
            f"got {gripper_positions.shape} for joints {joint_positions.shape}."
        )
    if raw_actions.ndim != 2 or raw_actions.shape[0] != joint_positions.shape[0]:
        raise ValueError(
            f"Expected raw_osc_actions with shape [T, A] matching joints, got {raw_actions.shape}."
        )

    gripper_representation = GripperRepresentation(rollout_data_config.action_target.gripper_representation)
    if gripper_representation == GripperRepresentation.ACTION_COMMAND:
        gripper_index = int(rollout_data_config.action_target.gripper_action_index)
        gripper_tail = raw_actions[:, gripper_index : gripper_index + 1]
        if gripper_tail.shape[1] == 0:
            gripper_tail = raw_actions[:, gripper_index:].reshape(raw_actions.shape[0], 1)
    elif gripper_representation == GripperRepresentation.FIRST_CHANNEL:
        gripper_tail = gripper_positions[:, :1]
    elif gripper_representation == GripperRepresentation.ALL_CHANNELS:
        gripper_tail = gripper_positions
    else:
        raise ValueError(f"Unsupported absolute-joint gripper representation: {gripper_representation}.")
    return np.concatenate(
        [joint_positions.astype(np.float32, copy=False), gripper_tail.astype(np.float32, copy=False)],
        axis=1,
    )


def _write_summary(
    path: Path,
    *,
    records: list[dict[str, Any]],
    total_selected: int,
    replay_status_path: Path,
    dataset_root: Path,
    output_root: Path,
    started_at: float,
) -> None:
    status_counts = Counter(record.get("conversion_status") for record in records)
    substeps_counts = Counter(
        record.get("selected_substeps_per_target")
        for record in records
        if record.get("conversion_status") == "success"
    )
    profile_counts = Counter(
        record.get("selected_absolute_profile")
        for record in records
        if record.get("conversion_status") == "success"
    )
    failed = [
        {
            "dataset_episode_index": record.get("dataset_episode_index"),
            "conversion_status": record.get("conversion_status"),
            "failure_reason": record.get("failure_reason"),
        }
        for record in records
        if record.get("conversion_status") != "success"
    ]
    summary = {
        "dataset_root": str(dataset_root),
        "replay_status_path": str(replay_status_path),
        "output_root": str(output_root),
        "total_selected_metadata_success_rows": int(total_selected),
        "processed_rows": len(records),
        "remaining_rows": max(0, int(total_selected) - len(records)),
        "conversion_status_counts": dict(status_counts),
        "selected_substeps_per_target_counts": {str(k): v for k, v in sorted(substeps_counts.items())},
        "selected_absolute_profile_counts": {str(k): v for k, v in sorted(profile_counts.items())},
        "failed_records": failed,
        "wall_time_s": round(time.time() - started_at, 3),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_completed_manifest(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("conversion_status") == "success":
                completed[int(row["dataset_episode_index"])] = row
    return completed


def _load_latest_manifest_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                latest[int(row["dataset_episode_index"])] = row
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return latest


def _load_replay_status_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["dataset_episode_index"]))
    return rows


def _filter_episode_indices(rows: list[dict[str, Any]], raw_indices: str | None) -> list[dict[str, Any]]:
    if raw_indices is None:
        return rows
    wanted = set(_parse_int_csv(raw_indices))
    return [row for row in rows if int(row["dataset_episode_index"]) in wanted]


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}.")
    return values


def _build_absolute_replay_profiles(args: argparse.Namespace) -> list[AbsoluteReplayProfile]:
    if args.absolute_profile_sweep == "rollout_adapter":
        return [
            AbsoluteReplayProfile(
                name="rollout_adapter",
                tracking_mode="rollout_adapter",
                control_frequency_mode=args.absolute_control_frequency_mode,
                gripper_substep_policy=args.absolute_gripper_substep_policy,
                gripper_tracking_mode=args.absolute_gripper_tracking_mode,
                gripper_target_delay_steps=0,
                joint_substep_policy=args.absolute_joint_substep_policy,
                joint_settle_l2_tolerance=None,
                joint_kp=args.absolute_joint_kp,
                disable_interpolator=bool(args.absolute_disable_interpolator),
                rollout_adapter_execution_mode=args.rollout_adapter_execution_mode,
            )
        ]
    if args.absolute_profile_sweep == "single":
        return [
            AbsoluteReplayProfile(
                name="single",
                tracking_mode=args.absolute_tracking_mode,
                control_frequency_mode=args.absolute_control_frequency_mode,
                gripper_substep_policy=args.absolute_gripper_substep_policy,
                gripper_tracking_mode=args.absolute_gripper_tracking_mode,
                gripper_target_delay_steps=int(args.absolute_gripper_target_delay_steps),
                joint_substep_policy=args.absolute_joint_substep_policy,
                joint_settle_l2_tolerance=args.absolute_joint_settle_l2_tolerance,
                joint_kp=args.absolute_joint_kp,
                disable_interpolator=bool(args.absolute_disable_interpolator),
                rollout_adapter_execution_mode=args.rollout_adapter_execution_mode,
            )
        ]
    if args.absolute_profile_sweep not in {"robust", "legacy_robust"}:
        raise ValueError(f"Unknown absolute profile sweep: {args.absolute_profile_sweep!r}.")
    profiles = [
        AbsoluteReplayProfile(
            name="direct_goal_slow_observed_gripper",
            tracking_mode="direct_goal",
            control_frequency_mode="raw",
            gripper_substep_policy="first_only",
            gripper_tracking_mode="observed_qpos",
            gripper_target_delay_steps=0,
            joint_substep_policy="hold",
            joint_settle_l2_tolerance=None,
            joint_kp=None,
            disable_interpolator=False,
        ),
        AbsoluteReplayProfile(
            name="direct_goal_settle001_slow_observed_gripper",
            tracking_mode="direct_goal",
            control_frequency_mode="raw",
            gripper_substep_policy="first_only",
            gripper_tracking_mode="observed_qpos",
            gripper_target_delay_steps=0,
            joint_substep_policy="hold",
            joint_settle_l2_tolerance=0.01,
            joint_kp=None,
            disable_interpolator=False,
        ),
        AbsoluteReplayProfile(
            name="direct_goal_exact_time_kp500_observed_gripper",
            tracking_mode="direct_goal",
            control_frequency_mode="scaled_by_substeps",
            gripper_substep_policy="first_only",
            gripper_tracking_mode="observed_qpos",
            gripper_target_delay_steps=0,
            joint_substep_policy="hold",
            joint_settle_l2_tolerance=None,
            joint_kp=500.0,
            disable_interpolator=False,
        ),
        AbsoluteReplayProfile(
            name="legacy_delta_slow_raw_gripper",
            tracking_mode="normalized_delta",
            control_frequency_mode="raw",
            gripper_substep_policy="repeat",
            gripper_tracking_mode="raw_action",
            gripper_target_delay_steps=0,
            joint_substep_policy="hold",
            joint_settle_l2_tolerance=None,
            joint_kp=None,
            disable_interpolator=False,
        ),
    ]
    if args.include_nointerp_profile:
        profiles.insert(
            2,
            AbsoluteReplayProfile(
                name="direct_goal_exact_time_kp500_nointerp_observed_gripper",
                tracking_mode="direct_goal",
                control_frequency_mode="scaled_by_substeps",
                gripper_substep_policy="first_only",
                gripper_tracking_mode="observed_qpos",
                gripper_target_delay_steps=0,
                joint_substep_policy="hold",
                joint_settle_l2_tolerance=None,
                joint_kp=500.0,
                disable_interpolator=True,
            ),
        )
    return profiles


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_max(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))


def _print_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), file=sys.stderr, flush=True)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LIBERO-10 metadata-successful OSC trajectories into verified absolute joint-position sidecars."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-status-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--substeps-sweep",
        default="2,4",
        help=(
            "Adapter-verification substeps to try. Defaults match the current unified LIBERO "
            "absolute-joint controller profile."
        ),
    )
    parser.add_argument(
        "--rollout-config",
        default="configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_abs_joint.yaml",
        help="Experiment config whose data.action_target normalization defines rollout model actions.",
    )
    parser.add_argument(
        "--absolute-joint-target-source",
        choices=("measured_qpos", "integrated_delta"),
        default="measured_qpos",
        help=(
            "`measured_qpos` uses replayed simulator joint qpos as targets. "
            "`integrated_delta` defines pseudo-absolute targets by integrating source delta commands from reset qpos; "
            "the rollout adapter then recovers delta commands by differencing consecutive targets."
        ),
    )
    parser.add_argument(
        "--integrated-delta-scale",
        default="0.05",
        help=(
            "Positive scalar mapping source normalized delta-joint commands to pseudo-qpos increments, "
            "or `auto` to fit one fixed scalar over the selected episodes before conversion."
        ),
    )
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--init-warmup-steps", type=int, default=None)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--summarize-manifest-only",
        action="store_true",
        help="Rebuild summary.json from the deduped manifest without launching simulator replay.",
    )
    parser.add_argument("--raw-stop-on-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--absolute-stop-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--absolute-success-grace-steps",
        type=int,
        default=0,
        help=(
            "Validation-only env steps to hold the final absolute target before declaring absolute replay failed. "
            "Must remain 0 for rollout_adapter verification because online rollout does not get extra settle steps."
        ),
    )
    parser.add_argument(
        "--absolute-profile-sweep",
        choices=("rollout_adapter", "legacy_robust", "robust", "single"),
        default="rollout_adapter",
        help=(
            "`rollout_adapter` accepts only rows replayed through the same LiberoBenchmarkAdapter path as inference. "
            "`legacy_robust` / `robust` are old conversion probes and should not be used for training data."
        ),
    )
    parser.add_argument(
        "--absolute-tracking-mode",
        choices=("rollout_adapter", "direct_goal", "normalized_delta"),
        default="rollout_adapter",
        help=(
            "`rollout_adapter` uses LiberoBenchmarkAdapter.action_from_model_action() and step(), matching inference. "
            "`direct_goal` uses robosuite's absolute set_qpos hook on the JOINT_POSITION controller. "
            "`normalized_delta` uses only the public normalized relative joint-delta action."
        ),
    )
    parser.add_argument(
        "--rollout-adapter-execution-mode",
        choices=tuple(mode.value for mode in LiberoAbsoluteJointExecutionMode),
        default=LiberoAbsoluteJointExecutionMode.DIRECT_GOAL.value,
        help=(
            "Execution mode used when --absolute-tracking-mode=rollout_adapter. "
            "`direct_goal` makes LiberoBenchmarkAdapter.step() use the absolute set_qpos controller hook; "
            "`normalized_delta` tracks qpos through the public JOINT_POSITION delta mapping; "
            "`integrated_delta` differences consecutive pseudo-qpos targets to recover delta commands."
        ),
    )
    parser.add_argument(
        "--absolute-control-frequency-mode",
        choices=("raw", "scaled_by_substeps"),
        default="scaled_by_substeps",
        help=(
            "`raw` keeps the metadata env_control_freq while substeps slow the replay for robust validation. "
            "`scaled_by_substeps` preserves original simulated time but can under-track fast joint targets."
        ),
    )
    parser.add_argument(
        "--absolute-gripper-substep-policy",
        choices=("first_only", "repeat", "last_only"),
        default="first_only",
        help=(
            "How to apply the source gripper command when one source qpos target is tracked for multiple env steps. "
            "`first_only` preserves one gripper update per source action."
        ),
    )
    parser.add_argument(
        "--absolute-gripper-tracking-mode",
        choices=("observed_qpos", "raw_action"),
        default="observed_qpos",
        help=(
            "`observed_qpos` closes/opens toward the gripper qpos reached in the raw replay. "
            "`raw_action` reuses the original scalar action command."
        ),
    )
    parser.add_argument(
        "--absolute-gripper-target-delay-steps",
        type=int,
        default=0,
        help=(
            "For observed-qpos gripper tracking, replay target gripper qpos from this many source steps earlier. "
            "This can keep gripper timing closer to the slowed arm replay used for robust validation."
        ),
    )
    parser.add_argument(
        "--absolute-joint-substep-policy",
        choices=("hold", "linear"),
        default="hold",
        help=(
            "How to track one recorded absolute joint target across multiple replay substeps. "
            "`hold` commands the target every substep. `linear` ramps from the previous target to the current target, "
            "which can reduce contact impulses while preserving recorded qpos endpoints."
        ),
    )
    parser.add_argument(
        "--absolute-joint-settle-l2-tolerance",
        type=float,
        default=None,
        help=(
            "Optional adaptive replay threshold. If set, the converter advances to the next recorded target early "
            "once the current arm qpos is within this L2 tolerance, preserving timing better than a fixed hold."
        ),
    )
    parser.add_argument(
        "--absolute-joint-kp",
        type=float,
        default=500.0,
        help="Optional JOINT_POSITION controller kp override for high-gain exact-time replay tests.",
    )
    parser.add_argument(
        "--absolute-disable-interpolator",
        action="store_true",
        help="Disable the robosuite JOINT_POSITION interpolator for exact absolute-goal replay tests.",
    )
    parser.add_argument(
        "--include-nointerp-profile",
        action="store_true",
        help="Include the no-interpolator exact-time profile in the robust sweep. Useful for debugging, slower by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
