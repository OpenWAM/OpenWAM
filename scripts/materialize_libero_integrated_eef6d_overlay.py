from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_libero_integrated_eef_scale import (  # noqa: E402
    _axis_angle_to_rotation_matrix_np,
    _rotation_6d_to_matrix_np,
    _rotation_matrix_geodesic_distance,
    build_integrated_eef_targets,
    recover_osc_actions_from_integrated_eef_targets,
)

from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig  # noqa: E402
from open_wam.simulators import EpisodeSpec  # noqa: E402
from open_wam.utils.config_loader import load_experiment_config  # noqa: E402


DEFAULT_POSITION_SCALE = 0.010576533139391671
DEFAULT_ROTATION_SCALE = 0.1136411594890211


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    replay_status_path = Path(args.replay_status_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rows = _load_success_rows(replay_status_path)
    episode_indices = args.episode_indices
    selected_replay_records = None
    if args.episode_indices_from_replay_report is not None:
        if episode_indices is not None:
            raise ValueError("--episode-indices and --episode-indices-from-replay-report are mutually exclusive.")
        selected_replay_records = _load_success_records_from_adapter_report(
            Path(args.episode_indices_from_replay_report).expanduser().resolve()
        )
        episode_indices = ",".join(str(_adapter_report_source_episode_index(row)) for row in selected_replay_records)
    rows = _select_rows(rows, episode_indices=episode_indices, limit=args.limit)
    if not rows:
        raise ValueError("No metadata-success rows selected.")

    _prepare_output_root(output_root, overwrite=bool(args.overwrite))

    dataset_info = _load_json(dataset_root / "meta" / "info.json")
    chunk_size = int(dataset_info.get("chunks_size", 1000))
    selected_episode_indices = tuple(int(row["dataset_episode_index"]) for row in rows)
    episode_index_map = {
        source_episode_index: output_episode_index
        for output_episode_index, source_episode_index in enumerate(selected_episode_indices)
    }
    _copy_meta(dataset_root=dataset_root, output_root=output_root, episode_index_map=episode_index_map)
    _materialize_reindexed_episode_assets(
        dataset_root / "latents",
        output_root / "latents",
        episode_index_map=episode_index_map,
        chunk_size=chunk_size,
    )
    _materialize_reindexed_episode_assets(
        dataset_root / "videos",
        output_root / "videos",
        episode_index_map=episode_index_map,
        chunk_size=chunk_size,
    )

    row_metrics = []
    max_recovery_error = 0.0
    for row in rows:
        episode_index = int(row["dataset_episode_index"])
        metrics = _write_episode_overlay(
            dataset_root=dataset_root,
            output_root=output_root,
            dataset_info=dataset_info,
            episode_index=episode_index,
            output_episode_index=int(episode_index_map[episode_index]),
            position_scale=float(args.position_scale),
            rotation_scale=float(args.rotation_scale),
            overwrite=bool(args.overwrite),
        )
        max_recovery_error = max(max_recovery_error, float(metrics["recovered_action_max_abs_error"]))
        row_metrics.append(metrics)

    if max_recovery_error > float(args.max_recovery_error):
        raise ValueError(
            "Integrated EEF6D overlay is not perfectly recoverable enough: "
            f"max_error={max_recovery_error:.6g}, tolerance={float(args.max_recovery_error):.6g}."
        )

    _update_info_features(output_root / "meta" / "info.json")
    transform_summary = _write_transform_summary(
        output_root=output_root,
        dataset_root=dataset_root,
        replay_status_path=replay_status_path,
        rows=rows,
        row_metrics=row_metrics,
        position_scale=float(args.position_scale),
        rotation_scale=float(args.rotation_scale),
        max_recovery_error=max_recovery_error,
        adapter_replay_source_report=args.episode_indices_from_replay_report,
    )

    replay_summary = None
    if selected_replay_records is not None and args.attach_selected_replay_report:
        replay_summary = _write_attached_adapter_replay_report(
            output_root=output_root,
            records=selected_replay_records,
            episode_index_map=episode_index_map,
            source_report=Path(args.episode_indices_from_replay_report).expanduser().resolve(),
        )
    if args.replay_episode_indices is not None:
        replay_rows = _select_rows(rows, episode_indices=args.replay_episode_indices, limit=args.replay_limit)
        replay_summary = _run_adapter_replay_validation(
            replay_rows,
            dataset_root=dataset_root,
            dataset_info=dataset_info,
            output_root=output_root,
            episode_index_map=episode_index_map,
            rollout_config=Path(args.rollout_config).expanduser().resolve(),
            position_scale=float(args.position_scale),
            rotation_scale=float(args.rotation_scale),
            camera_height=int(args.camera_height),
            camera_width=int(args.camera_width),
            replay_env_backend=str(args.replay_env_backend),
            replay_use_camera_obs=bool(args.replay_use_camera_obs),
            replay_has_offscreen_renderer=bool(args.replay_has_offscreen_renderer),
            resume=bool(args.resume_replay),
        )
        if args.require_replay_success and replay_summary["success_count"] != replay_summary["episodes"]:
            raise ValueError(
                "Adapter replay validation did not pass for every selected episode: "
                f"{replay_summary['success_count']}/{replay_summary['episodes']} succeeded. "
                f"See {replay_summary['report_path']}."
            )

    print(
        json.dumps(
            {
                "event": "integrated_eef6d_overlay_done",
                "output_root": str(output_root),
                "episodes_written": len(rows),
                "rows_written": transform_summary["rows_written"],
                "max_recovery_error": max_recovery_error,
                "replay": replay_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_episode_overlay(
    *,
    dataset_root: Path,
    output_root: Path,
    dataset_info: dict[str, Any],
    episode_index: int,
    output_episode_index: int,
    position_scale: float,
    rotation_scale: float,
    overwrite: bool,
) -> dict[str, Any]:
    data_path_template = str(dataset_info["data_path"])
    chunk_size = int(dataset_info.get("chunks_size", 1000))
    source_path = dataset_root / data_path_template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    output_path = output_root / data_path_template.format(
        episode_chunk=output_episode_index // chunk_size,
        episode_index=output_episode_index,
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {output_path}")
    table = pq.read_table(source_path)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    targets = build_integrated_eef_targets(
        initial_state=states[0],
        actions=actions,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
    )
    recovered = recover_osc_actions_from_integrated_eef_targets(
        initial_state=states[0],
        targets=targets,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
    )
    recovery_error = np.abs(recovered - actions[:, :7])
    pose_metrics = _target_observation_metrics(targets=targets, states=states)
    if "integrated_eef6d_action" in table.column_names:
        table = table.drop(["integrated_eef6d_action"])
    table = table.append_column("integrated_eef6d_action", _float_list_array(targets))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.{os.getpid()}.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)
    return {
        "dataset_episode_index": int(episode_index),
        "rows": int(table.num_rows),
        "recovered_action_max_abs_error": float(np.max(recovery_error)),
        "recovered_action_mean_abs_error": float(np.mean(recovery_error)),
        **pose_metrics,
    }


def _target_observation_metrics(*, targets: np.ndarray, states: np.ndarray) -> dict[str, float]:
    target_rotations = _rotation_6d_to_matrix_np(targets[:, 3:9])
    observed_rotations = _axis_angle_to_rotation_matrix_np(states[:, 3:6])
    position_error = np.linalg.norm(np.asarray(targets[:, 0:3], dtype=np.float64) - states[:, 0:3], axis=1)
    rotation_error = _rotation_matrix_geodesic_distance(target_rotations, observed_rotations)
    return {
        "position_mean_l2": float(np.mean(position_error)),
        "position_p95_l2": float(np.quantile(position_error, 0.95)),
        "position_max_l2": float(np.max(position_error)),
        "rotation_mean_geodesic_rad": float(np.mean(rotation_error)),
        "rotation_p95_geodesic_rad": float(np.quantile(rotation_error, 0.95)),
        "rotation_max_geodesic_rad": float(np.max(rotation_error)),
    }


def _run_adapter_replay_validation(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    dataset_info: dict[str, Any],
    output_root: Path,
    episode_index_map: dict[int, int],
    rollout_config: Path,
    position_scale: float,
    rotation_scale: float,
    camera_height: int,
    camera_width: int,
    replay_env_backend: str,
    replay_use_camera_obs: bool,
    replay_has_offscreen_renderer: bool,
    resume: bool,
) -> dict[str, Any]:
    config = load_experiment_config(rollout_config)
    report_path = output_root / "meta" / "integrated_eef6d_adapter_replay_status.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if resume and report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source_episode_index = int(record.get("source_dataset_episode_index", record["dataset_episode_index"]))
            existing[source_episode_index] = record
    results = []
    with report_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            episode_index = int(row["dataset_episode_index"])
            output_episode_index = int(episode_index_map[episode_index])
            if episode_index in existing:
                results.append(existing[episode_index])
                continue
            targets = _load_overlay_targets(
                output_root,
                dataset_info=dataset_info,
                episode_index=output_episode_index,
            )
            env_config = LiberoEnvConfig(
                benchmark_name=str(row.get("upstream_benchmark", "libero_10")),
                controller="OSC_POSE",
                action_mode="integrated_eef6d_osc",
                env_backend=replay_env_backend,
                use_camera_obs=replay_use_camera_obs,
                has_offscreen_renderer=replay_has_offscreen_renderer,
                camera_height=camera_height,
                camera_width=camera_width,
                control_freq=int(row.get("env_control_freq", 20) or 20),
                ignore_done=False,
                init_state_index=int(row["resolved_init_state_index"]),
                integrated_eef_position_scale=position_scale,
                integrated_eef_rotation_scale=rotation_scale,
            )
            adapter = LiberoBenchmarkAdapter(config=env_config)
            success = False
            steps = 0
            error = None
            try:
                adapter.reset(
                    EpisodeSpec(
                        task_id=int(row["upstream_task_id"]),
                        episode_idx=int(row["resolved_init_state_index"]),
                        seed=_reset_seed(row),
                    )
                )
                adapter.set_integrated_eef6d_previous_target_from_state(
                    _load_source_initial_state(
                        dataset_root,
                        dataset_info=dataset_info,
                        episode_index=episode_index,
                    )
                )
                for target in targets:
                    env_action = adapter.action_from_model_action(target, data_config=config.data)
                    step = adapter.step(env_action)
                    steps += 1
                    if step.success or step.done:
                        success = bool(step.success or step.done)
                        break
            except Exception as exc:  # pragma: no cover - exercised only in real LIBERO validation.
                error = repr(exc)
            finally:
                adapter.close()
            record = {
                "dataset_episode_index": output_episode_index,
                "source_dataset_episode_index": episode_index,
                "upstream_task_id": int(row["upstream_task_id"]),
                "resolved_init_state_index": int(row["resolved_init_state_index"]),
                "success": bool(success),
                "steps": int(steps),
                "error": error,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            results.append(record)
    success_count = sum(1 for item in results if item.get("success"))
    summary = {
        "episodes": len(results),
        "success_count": int(success_count),
        "success_rate": None if not results else float(success_count / len(results)),
        "report_path": str(report_path),
    }
    (output_root / "meta" / "integrated_eef6d_adapter_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists; pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _load_overlay_targets(output_root: Path, *, dataset_info: dict[str, Any], episode_index: int) -> np.ndarray:
    data_path_template = str(dataset_info["data_path"])
    chunk_size = int(dataset_info.get("chunks_size", 1000))
    path = output_root / data_path_template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    table = pq.read_table(path, columns=["integrated_eef6d_action"])
    return np.asarray(table["integrated_eef6d_action"].to_pylist(), dtype=np.float32)


def _load_source_initial_state(dataset_root: Path, *, dataset_info: dict[str, Any], episode_index: int) -> np.ndarray:
    data_path_template = str(dataset_info["data_path"])
    chunk_size = int(dataset_info.get("chunks_size", 1000))
    path = dataset_root / data_path_template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    table = pq.read_table(path, columns=["observation.state"])
    return np.asarray(table["observation.state"].to_pylist()[0], dtype=np.float32)


def _reset_seed(row: dict[str, Any]) -> int | None:
    resolved_init = int(row["resolved_init_state_index"])
    attempts = row.get("attempts")
    if isinstance(attempts, list) and attempts:
        for attempt in attempts:
            if int(attempt.get("init_state_index", -1)) == resolved_init and attempt.get("success_step") is not None:
                return int(attempt["reset_seed"])
        for attempt in attempts:
            if int(attempt.get("init_state_index", -1)) == resolved_init and attempt.get("reset_seed") is not None:
                return int(attempt["reset_seed"])
        seed = attempts[0].get("reset_seed")
        return None if seed is None else int(seed)
    seed = row.get("reset_seed")
    return None if seed is None else int(seed)


def _write_transform_summary(
    *,
    output_root: Path,
    dataset_root: Path,
    replay_status_path: Path,
    rows: list[dict[str, Any]],
    row_metrics: list[dict[str, Any]],
    position_scale: float,
    rotation_scale: float,
    max_recovery_error: float,
    adapter_replay_source_report: str | None,
) -> dict[str, Any]:
    rows_written = sum(int(item["rows"]) for item in row_metrics)
    summary = {
        "source_dataset_root": str(dataset_root),
        "replay_status_path": str(replay_status_path),
        "episodes_written": len(rows),
        "rows_written": int(rows_written),
        "episode_indices": [int(row["dataset_episode_index"]) for row in rows],
        "action_column": "integrated_eef6d_action",
        "action_dim": 10,
        "position_scale": float(position_scale),
        "rotation_scale": float(rotation_scale),
        "max_recovery_error": float(max_recovery_error),
        "position_mean_l2": float(np.mean([item["position_mean_l2"] for item in row_metrics])),
        "rotation_mean_geodesic_rad": float(np.mean([item["rotation_mean_geodesic_rad"] for item in row_metrics])),
    }
    if adapter_replay_source_report is not None:
        summary["adapter_replay_source_report"] = str(Path(adapter_replay_source_report).expanduser().resolve())
    meta = output_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "integrated_eef6d_overlay.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (meta / "integrated_eef6d_transform_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for item in row_metrics:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    return summary


def _update_info_features(path: Path) -> None:
    info = _load_json(path)
    episodes_path = path.parent / "episodes.jsonl"
    if episodes_path.is_file():
        episode_indices = _episode_indices_from_metadata(episodes_path)
        _set_contiguous_episode_info(info, episode_indices)
    features = dict(info.get("features") or {})
    features["integrated_eef6d_action"] = {
        "dtype": "float32",
        "shape": [10],
        "names": {
            "motors": [
                "absolute_eef_x",
                "absolute_eef_y",
                "absolute_eef_z",
                "rotation6d_col0_x",
                "rotation6d_col0_y",
                "rotation6d_col0_z",
                "rotation6d_col1_x",
                "rotation6d_col1_y",
                "rotation6d_col1_z",
                "gripper_command",
            ]
        },
    }
    info["features"] = features
    path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")


def _episode_indices_from_metadata(path: Path) -> list[int]:
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            indices.append(int(json.loads(line)["episode_index"]))
    return sorted(indices)


def _set_contiguous_episode_info(info: dict[str, Any], episode_indices: list[int]) -> None:
    selected_count = len(episode_indices)
    expected = list(range(selected_count))
    if episode_indices != expected:
        raise ValueError(
            "Overlay materialization should reindex episodes contiguously, "
            f"got episode_indices={episode_indices}, expected={expected}."
        )
    info["selected_episode_count"] = int(selected_count)
    info["episode_indices"] = [int(index) for index in episode_indices]
    info["episode_index_policy"] = "contiguous_reindexed"
    info["total_episodes"] = int(selected_count)
    info.pop("total_episodes_semantics", None)


def _copy_meta(*, dataset_root: Path, output_root: Path, episode_index_map: dict[int, int]) -> None:
    source_meta = dataset_root / "meta"
    output_meta = output_root / "meta"
    if output_meta.exists():
        shutil.rmtree(output_meta)
    shutil.copytree(source_meta, output_meta)
    _filter_episode_metadata(output_meta / "episodes.jsonl", episode_index_map=episode_index_map)


def _filter_episode_metadata(path: Path, *, episode_index_map: dict[int, int]) -> None:
    wanted = set(int(index) for index in episode_index_map)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_episode_index = int(row["episode_index"])
            if source_episode_index in wanted:
                output_row = dict(row)
                output_row["source_dataset_episode_index"] = source_episode_index
                output_row["episode_index"] = int(episode_index_map[source_episode_index])
                rows.append(output_row)
    missing = sorted(wanted.difference(int(row["source_dataset_episode_index"]) for row in rows))
    if missing:
        raise ValueError(f"Cannot filter overlay metadata; missing episode rows: {missing[:10]}.")
    rows.sort(key=lambda row: int(row["episode_index"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _materialize_reindexed_episode_assets(
    source: Path,
    target: Path,
    *,
    episode_index_map: dict[int, int],
    chunk_size: int,
) -> None:
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    for source_episode_index, output_episode_index in episode_index_map.items():
        source_token = f"episode_{source_episode_index:06d}"
        output_token = f"episode_{output_episode_index:06d}"
        output_chunk = f"chunk-{output_episode_index // int(chunk_size):03d}"
        for source_path in source.rglob(f"{source_token}*"):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(source)
            parent_parts = [
                output_chunk if part.startswith("chunk-") and part[6:].isdigit() else part
                for part in relative.parts[:-1]
            ]
            output_name = relative.name.replace(source_token, output_token, 1)
            target_path = target.joinpath(*parent_parts, output_name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(source_path.resolve())


def _float_list_array(values: np.ndarray) -> pa.Array:
    return pa.array([row.astype(np.float32, copy=False).tolist() for row in values], type=pa.list_(pa.float32()))


def _load_success_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("replay_status") == "success"]
    rows.sort(key=lambda row: int(row["dataset_episode_index"]))
    return rows


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    episode_indices: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if episode_indices is not None:
        if episode_indices.strip().lower() == "all":
            selected = list(rows)
        else:
            wanted = {int(part.strip()) for part in episode_indices.split(",") if part.strip()}
            selected = [row for row in rows if int(row["dataset_episode_index"]) in wanted]
    else:
        selected = list(rows)
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def _adapter_report_source_episode_index(row: dict[str, Any]) -> int:
    return int(row.get("source_dataset_episode_index", row["dataset_episode_index"]))


def _load_success_records_from_adapter_report(path: Path) -> tuple[dict[str, Any], ...]:
    """Load current-simulator adapter-success rows from a JSONL report."""

    if not path.is_file():
        raise FileNotFoundError(f"Adapter replay report does not exist: {path}")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not bool(row.get("success")):
            continue
        episode_index = _adapter_report_source_episode_index(row)
        if episode_index in seen:
            continue
        normalized = dict(row)
        normalized["reused_report_dataset_episode_index"] = int(row["dataset_episode_index"])
        normalized["source_dataset_episode_index"] = episode_index
        records.append(normalized)
        seen.add(episode_index)
    if not records:
        raise ValueError(f"Adapter replay report contains no successful episodes: {path}")
    records.sort(key=_adapter_report_source_episode_index)
    return tuple(records)


def _write_attached_adapter_replay_report(
    *,
    output_root: Path,
    records: tuple[dict[str, Any], ...],
    episode_index_map: dict[int, int],
    source_report: Path,
) -> dict[str, Any]:
    meta = output_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    report_path = meta / "integrated_eef6d_adapter_replay_status.jsonl"
    with report_path.open("w", encoding="utf-8") as handle:
        for record in records:
            source_episode_index = _adapter_report_source_episode_index(record)
            if source_episode_index not in episode_index_map:
                raise ValueError(
                    "Cannot attach replay report row for source episode "
                    f"{source_episode_index}; selected overlay episode map only contains "
                    f"{sorted(episode_index_map)[:10]}."
                )
            copied = dict(record)
            copied["reused_report_dataset_episode_index"] = int(record["dataset_episode_index"])
            copied["dataset_episode_index"] = int(episode_index_map[source_episode_index])
            copied["source_dataset_episode_index"] = source_episode_index
            copied["reused_from_report"] = str(source_report)
            handle.write(json.dumps(copied, sort_keys=True) + "\n")
    summary = {
        "episodes": len(records),
        "success_count": len(records),
        "success_rate": 1.0,
        "report_path": str(report_path),
        "reused_from_report": str(source_report),
    }
    (meta / "integrated_eef6d_adapter_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize recoverable LIBERO integrated EEF6D pseudo-absolute targets."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-status-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--position-scale", type=float, default=DEFAULT_POSITION_SCALE)
    parser.add_argument("--rotation-scale", type=float, default=DEFAULT_ROTATION_SCALE)
    parser.add_argument("--episode-indices", default=None, help="Comma-separated dataset episode indices or 'all'.")
    parser.add_argument(
        "--episode-indices-from-replay-report",
        default=None,
        help=(
            "Select dataset episode indices from an integrated_eef6d_adapter_replay_status.jsonl report, "
            "keeping only rows with success=true."
        ),
    )
    parser.add_argument(
        "--attach-selected-replay-report",
        action="store_true",
        help=(
            "When --episode-indices-from-replay-report is used, copy the selected success rows into the new "
            "output metadata instead of rerunning simulator replay."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-recovery-error", type=float, default=1e-4)
    parser.add_argument(
        "--replay-episode-indices",
        default=None,
        help="Optional comma-separated dataset episode indices or 'all' to validate through the rollout adapter.",
    )
    parser.add_argument("--replay-limit", type=int, default=None)
    parser.add_argument("--resume-replay", action="store_true")
    parser.add_argument(
        "--require-replay-success",
        action="store_true",
        help="Fail if any selected episode does not replay successfully through the rollout adapter.",
    )
    parser.add_argument(
        "--rollout-config",
        default="configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_abs_eef6d.yaml",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--replay-env-backend", choices=("control", "offscreen"), default="control")
    parser.add_argument("--replay-use-camera-obs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay-has-offscreen-renderer", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
