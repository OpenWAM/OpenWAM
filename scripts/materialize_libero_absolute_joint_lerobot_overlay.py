from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    conversion_root = Path(args.conversion_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    summary = _load_json(conversion_root / "summary.json")
    if args.require_complete:
        _require_complete_conversion(summary)
    records = _load_success_manifest(conversion_root / "manifest.jsonl")
    if args.episode_indices is not None:
        wanted = set(_parse_int_csv(args.episode_indices))
        records = [record for record in records if int(record["dataset_episode_index"]) in wanted]
    if not records:
        raise ValueError(f"No successful conversion records found in {conversion_root / 'manifest.jsonl'}.")

    _prepare_output_root(output_root, overwrite=bool(args.overwrite))
    episode_index_map = {
        int(record["dataset_episode_index"]): new_episode_index
        for new_episode_index, record in enumerate(records)
    }
    _copy_meta(dataset_root=dataset_root, output_root=output_root, episode_index_map=episode_index_map)

    info = _load_json(output_root / "meta" / "info.json")
    data_path_template = str(info["data_path"])
    chunk_size = int(info.get("chunks_size", 1000))
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
    rows_written = 0
    action_dims: set[int] = set()
    for record in records:
        episode_index = int(record["dataset_episode_index"])
        output_episode_index = int(episode_index_map[episode_index])
        source_path = dataset_root / data_path_template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        output_path = output_root / data_path_template.format(
            episode_chunk=output_episode_index // chunk_size,
            episode_index=output_episode_index,
        )
        sidecar_path = Path(str(record["sidecar_path"])).expanduser()
        episode_rows, action_dim = _write_episode_overlay(
            source_path=source_path,
            output_path=output_path,
            sidecar_path=sidecar_path,
            overwrite=args.overwrite,
        )
        rows_written += episode_rows
        action_dims.add(action_dim)

    if len(action_dims) != 1:
        raise ValueError(f"Absolute-joint overlay has inconsistent action dims across sidecars: {sorted(action_dims)}.")
    action_dim = next(iter(action_dims))
    _write_overlay_info(
        output_root=output_root,
        source_dataset_root=dataset_root,
        conversion_root=conversion_root,
        success_records=records,
        rows_written=rows_written,
        action_dim=action_dim,
    )
    _update_info_features(output_root / "meta" / "info.json", action_dim=action_dim)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "conversion_root": str(conversion_root),
                "episodes_written": len(records),
                "rows_written": rows_written,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists; pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _write_episode_overlay(
    *,
    source_path: Path,
    output_path: Path,
    sidecar_path: Path,
    overwrite: bool,
) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output parquet already exists: {output_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source parquet does not exist: {source_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Absolute-joint sidecar does not exist: {sidecar_path}")
    table = pq.read_table(source_path)
    sidecar = np.load(sidecar_path)
    joint_qpos = np.asarray(sidecar["joint_positions_after_action"], dtype=np.float32)
    gripper_qpos = np.asarray(sidecar["gripper_positions_after_action"], dtype=np.float32)
    absolute_actions = np.asarray(sidecar["absolute_joint_actions"], dtype=np.float32)
    if absolute_actions.ndim != 2 or absolute_actions.shape[1] not in {8, 9}:
        raise ValueError(
            f"Expected absolute_joint_actions with shape [T, 8] or [T, 9], "
            f"got {absolute_actions.shape} from {sidecar_path}."
        )
    if table.num_rows != joint_qpos.shape[0]:
        raise ValueError(
            f"Sidecar length mismatch for {source_path.name}: parquet_rows={table.num_rows}, "
            f"joint_positions={joint_qpos.shape[0]}."
        )
    if table.num_rows != gripper_qpos.shape[0] or table.num_rows != absolute_actions.shape[0]:
        raise ValueError(
            f"Sidecar arrays must all match parquet length for {source_path.name}: "
            f"rows={table.num_rows}, gripper={gripper_qpos.shape[0]}, actions={absolute_actions.shape[0]}."
        )
    for name in ("robot0_joint_pos", "robot0_gripper_qpos", "absolute_joint_action"):
        if name in table.column_names:
            table = table.drop([name])
    table = table.append_column("robot0_joint_pos", _float_list_array(joint_qpos))
    table = table.append_column("robot0_gripper_qpos", _float_list_array(gripper_qpos))
    table = table.append_column("absolute_joint_action", _float_list_array(absolute_actions))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.{os.getpid()}.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(output_path)
    return int(table.num_rows), int(absolute_actions.shape[1])


def _float_list_array(values: np.ndarray) -> pa.Array:
    return pa.array([row.astype(np.float32, copy=False).tolist() for row in values], type=pa.list_(pa.float32()))


def _load_success_manifest(path: Path) -> list[dict[str, Any]]:
    records_by_episode: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("conversion_status") == "success":
                records_by_episode[int(record["dataset_episode_index"])] = record
    records = list(records_by_episode.values())
    records.sort(key=lambda record: int(record["dataset_episode_index"]))
    return records


def _copy_meta(*, dataset_root: Path, output_root: Path, episode_index_map: dict[int, int]) -> None:
    source_meta = dataset_root / "meta"
    output_meta = output_root / "meta"
    if output_meta.exists():
        shutil.rmtree(output_meta)
    shutil.copytree(source_meta, output_meta)
    _filter_episode_metadata(output_meta / "episodes.jsonl", episode_index_map=episode_index_map)


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


def _write_overlay_info(
    *,
    output_root: Path,
    source_dataset_root: Path,
    conversion_root: Path,
    success_records: list[dict[str, Any]],
    rows_written: int,
    action_dim: int,
) -> None:
    payload = {
        "source_dataset_root": str(source_dataset_root),
        "conversion_root": str(conversion_root),
        "episodes_written": len(success_records),
        "rows_written": int(rows_written),
        "absolute_joint_action_dim": int(action_dim),
        "episode_indices": [int(record["dataset_episode_index"]) for record in success_records],
    }
    path = output_root / "meta" / "absolute_joint_overlay.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"Cannot filter overlay metadata; missing episode rows: {preview}{suffix}.")
    rows.sort(key=lambda row: int(row["episode_index"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _update_info_features(path: Path, *, action_dim: int) -> None:
    info = _load_json(path)
    episodes_path = path.parent / "episodes.jsonl"
    if episodes_path.is_file():
        episode_indices = _episode_indices_from_metadata(episodes_path)
        _set_contiguous_episode_info(info, episode_indices)
    features = dict(info.get("features") or {})
    features["robot0_joint_pos"] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": [f"joint_{index}" for index in range(7)]},
    }
    features["robot0_gripper_qpos"] = {
        "dtype": "float32",
        "shape": [2],
        "names": {"motors": ["left_finger", "right_finger"]},
    }
    action_motor_names = [f"joint_{index}" for index in range(7)]
    if int(action_dim) == 8:
        action_motor_names.append("gripper_command")
    elif int(action_dim) == 9:
        action_motor_names.extend(["left_finger", "right_finger"])
    else:
        raise ValueError(f"Unsupported absolute_joint_action dim: {action_dim}.")
    features["absolute_joint_action"] = {
        "dtype": "float32",
        "shape": [int(action_dim)],
        "names": {"motors": action_motor_names},
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


def _require_complete_conversion(summary: dict[str, Any]) -> None:
    total = int(summary.get("total_selected_metadata_success_rows", -1))
    processed = int(summary.get("processed_rows", -1))
    status_counts = dict(summary.get("conversion_status_counts") or {})
    success = int(status_counts.get("success", 0))
    if total <= 0 or processed != total or success != total:
        raise ValueError(
            "Absolute-joint conversion is not complete and fully successful: "
            f"total={total}, processed={processed}, success={success}, status_counts={status_counts}."
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize verified LIBERO absolute-joint sidecars as a LeRobot-compatible overlay root."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--conversion-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
