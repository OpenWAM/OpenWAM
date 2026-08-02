#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SUBSETS = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_INIT_STATE_COUNT = 50
INIT_COVERAGE_SCHEMA_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LIBERO replay-label JSONL shards and write dataset metadata, including "
            "per-task/init-state successful-GT coverage."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--subsets", type=str, default=",".join(DEFAULT_SUBSETS))
    parser.add_argument(
        "--source-run",
        action="append",
        default=[],
        help=(
            "Replay source in the form <subset>=<run_id>. May be repeated; later sources override "
            "earlier rows for the same dataset_episode_index. Use this to overlay top-50 reruns on top-12."
        ),
    )
    parser.add_argument(
        "--source-run-expected-rows",
        action="append",
        default=[],
        help=(
            "Optional completeness assertion in the form <subset>=<run_id>=<row_count>. "
            "This is useful when a full top-12 pass is overlaid by partial top-50 failure reruns; "
            "episode-count completeness alone cannot prove the refinement run finished."
        ),
    )
    parser.add_argument(
        "--from-installed-meta",
        action="store_true",
        help="Read <dataset_root>/<subset>/meta/replay_status.jsonl instead of --source-run inputs.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--install-dataset-meta", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--default-init-state-count", type=int, default=DEFAULT_INIT_STATE_COUNT)
    args = parser.parse_args()

    subsets = parse_subset_selector(args.subsets)
    run_id = args.run_id or f"libero_replay_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = args.diagnostic_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if args.from_installed_meta and args.source_run:
        raise ValueError("--from-installed-meta and --source-run are mutually exclusive.")
    source_runs = parse_source_runs(args.source_run)
    expected_source_rows = parse_expected_source_rows(args.source_run_expected_rows)
    if not args.from_installed_meta and not source_runs:
        raise ValueError("Provide at least one --source-run or use --from-installed-meta.")

    overall: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "diagnostic_root": str(args.diagnostic_root),
        "subsets": {},
        "source_runs": source_runs,
        "expected_source_rows": format_expected_source_rows(expected_source_rows),
        "from_installed_meta": bool(args.from_installed_meta),
    }
    for subset in subsets:
        dataset_root = args.dataset_root / subset
        expected_episodes = expected_episode_count(dataset_root)
        if args.from_installed_meta:
            rows = read_jsonl(dataset_root / "meta" / "replay_status.jsonl")
        else:
            rows = merge_source_rows(
                args.diagnostic_root,
                subset=subset,
                source_runs=source_runs.get(subset, []),
                expected_source_rows=expected_source_rows,
            )
        complete = len(rows) == expected_episodes
        if not complete and not args.allow_incomplete:
            raise RuntimeError(
                f"Refusing to write incomplete metadata for {subset}: {len(rows)} / {expected_episodes} rows."
            )
        validate_unique_episode_rows(rows, subset=subset)
        rows = sorted(rows, key=lambda row: int(row["dataset_episode_index"]))
        status_summary = summarize_rows(rows)
        coverage_rows, coverage_summary = build_init_state_coverage(
            dataset_root=dataset_root,
            subset=subset,
            rows=rows,
            default_init_state_count=int(args.default_init_state_count),
        )
        summary = {
            **status_summary,
            "expected_episodes": expected_episodes,
            "complete": complete,
            "init_state_coverage": coverage_summary,
        }
        subset_root = run_root / subset
        subset_root.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(subset_root / "replay_status.jsonl", rows)
        write_json_atomic(subset_root / "replay_status.summary.json", summary)
        write_jsonl_atomic(subset_root / "replay_init_state_coverage.jsonl", coverage_rows)
        write_json_atomic(subset_root / "replay_init_state_coverage.summary.json", coverage_summary)
        if args.install_dataset_meta:
            install_subset_metadata(
                dataset_root=dataset_root,
                rows=rows,
                status_summary=summary,
                coverage_rows=coverage_rows,
                coverage_summary=coverage_summary,
                environment=overall,
            )
        overall["subsets"][subset] = summary

    write_json_atomic(run_root / "summary.json", overall)
    print(json.dumps(overall, indent=2))


def parse_source_runs(raw_specs: list[str]) -> dict[str, list[str]]:
    source_runs: dict[str, list[str]] = defaultdict(list)
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --source-run {spec!r}; expected <subset>=<run_id>.")
        subset, run_id = spec.split("=", 1)
        subset = subset.strip()
        run_id = run_id.strip()
        if not subset or not run_id:
            raise ValueError(f"Invalid --source-run {spec!r}; expected <subset>=<run_id>.")
        source_runs[subset].append(run_id)
    return dict(source_runs)


def parse_expected_source_rows(raw_specs: list[str]) -> dict[tuple[str, str], int]:
    expected: dict[tuple[str, str], int] = {}
    for spec in raw_specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --source-run-expected-rows {spec!r}; expected <subset>=<run_id>=<row_count>."
            )
        subset, run_id, row_count_text = (part.strip() for part in parts)
        if not subset or not run_id:
            raise ValueError(
                f"Invalid --source-run-expected-rows {spec!r}; expected <subset>=<run_id>=<row_count>."
            )
        expected[(subset, run_id)] = int(row_count_text)
    return expected


def format_expected_source_rows(expected_source_rows: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    formatted: dict[str, dict[str, int]] = defaultdict(dict)
    for (subset, run_id), row_count in sorted(expected_source_rows.items()):
        formatted[subset][run_id] = row_count
    return dict(formatted)


def merge_source_rows(
    diagnostic_root: Path,
    *,
    subset: str,
    source_runs: list[str],
    expected_source_rows: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    if not source_runs:
        raise ValueError(f"No --source-run entries were provided for {subset}.")
    rows_by_episode: dict[int, dict[str, Any]] = {}
    for source_order, run_id in enumerate(source_runs):
        source_root = diagnostic_root / run_id / subset
        source_rows = read_replay_rows_from_dir(source_root)
        if not source_rows:
            raise ValueError(f"No replay rows found for {subset} in {source_root}.")
        unique_source_episode_count = len({int(row["dataset_episode_index"]) for row in source_rows})
        expected_count = expected_source_rows.get((subset, run_id))
        if expected_count is not None and unique_source_episode_count != expected_count:
            raise RuntimeError(
                f"Source run {run_id} for {subset} has {unique_source_episode_count} unique rows; "
                f"expected {expected_count}."
            )
        for row in source_rows:
            merged_row = dict(row)
            merged_row["metadata_source_run_id"] = run_id
            merged_row["metadata_source_order"] = source_order
            rows_by_episode[int(merged_row["dataset_episode_index"])] = merged_row
    return [rows_by_episode[index] for index in sorted(rows_by_episode)]


def read_replay_rows_from_dir(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = ("replay_status.shard_*.jsonl", "replay_status.jsonl", "replay_status.merged.jsonl")
    for pattern in patterns:
        for path in sorted(source_root.glob(pattern)):
            rows.extend(read_jsonl(path))
    return rows


def build_init_state_coverage(
    *,
    dataset_root: Path,
    subset: str,
    rows: list[dict[str, Any]],
    default_init_state_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_info: dict[tuple[int, str], dict[str, Any]] = {}
    successful_by_task_local_init: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    failure_by_task_local_init: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    episode_by_task_local_init: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    successful_by_resolved_init: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    attempted_by_resolved_init: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for row in rows:
        if row.get("replay_status") not in {"success", "failure"}:
            continue
        task_id = int(row["upstream_task_id"])
        task_name = str(row["upstream_task_name"])
        task_key = (task_id, task_name)
        task_entry = task_info.setdefault(
            task_key,
            {
                "subset": subset,
                "dataset_root": str(dataset_root),
                "upstream_benchmark": str(row.get("upstream_benchmark") or subset),
                "upstream_task_id": task_id,
                "upstream_task_name": task_name,
                "task_texts": set(),
                "init_state_count": max(1, int(default_init_state_count)),
            },
        )
        task_entry["task_texts"].add(str(row.get("task_text", "")))
        task_entry["init_state_count"] = max(
            int(task_entry["init_state_count"]),
            max_observed_init_state_index(row) + 1,
        )
        episode_index = int(row["dataset_episode_index"])
        task_local_init_state_index = row.get("task_local_rank_init_state_index")
        if task_local_init_state_index is not None:
            task_local_init_state_index = int(task_local_init_state_index)
            episode_by_task_local_init[(task_id, task_name, task_local_init_state_index)].append(episode_index)
            if row.get("replay_status") == "success":
                successful_by_task_local_init[(task_id, task_name, task_local_init_state_index)].append(
                    episode_index
                )
            else:
                failure_by_task_local_init[(task_id, task_name, task_local_init_state_index)].append(episode_index)
        for init_state_index in row.get("attempted_init_state_indices") or []:
            attempted_by_resolved_init[(task_id, task_name, int(init_state_index))].append(episode_index)
        if row.get("replay_status") == "success" and row.get("resolved_init_state_index") is not None:
            init_state_index = int(row["resolved_init_state_index"])
            successful_by_resolved_init[(task_id, task_name, init_state_index)].append(episode_index)

    coverage_rows: list[dict[str, Any]] = []
    missing_by_task: dict[str, list[int]] = {}
    missing_resolved_by_task: dict[str, list[int]] = {}
    covered_count = 0
    missing_count = 0
    resolved_covered_count = 0
    resolved_missing_count = 0
    for (task_id, task_name), task_entry in sorted(task_info.items(), key=lambda item: (item[0][0], item[0][1])):
        missing_for_task: list[int] = []
        missing_resolved_for_task: list[int] = []
        init_state_count = int(task_entry["init_state_count"])
        for init_state_index in range(init_state_count):
            task_local_success_eps = sorted(
                set(successful_by_task_local_init.get((task_id, task_name, init_state_index), []))
            )
            task_local_failure_eps = sorted(
                set(failure_by_task_local_init.get((task_id, task_name, init_state_index), []))
            )
            task_local_eps = sorted(set(episode_by_task_local_init.get((task_id, task_name, init_state_index), [])))
            resolved_success_eps = sorted(
                set(successful_by_resolved_init.get((task_id, task_name, init_state_index), []))
            )
            attempted_eps = sorted(set(attempted_by_resolved_init.get((task_id, task_name, init_state_index), [])))
            has_successful_gt = bool(task_local_success_eps)
            has_successful_resolved_replay = bool(resolved_success_eps)
            if has_successful_gt:
                covered_count += 1
            else:
                missing_count += 1
                missing_for_task.append(init_state_index)
            if has_successful_resolved_replay:
                resolved_covered_count += 1
            else:
                resolved_missing_count += 1
                missing_resolved_for_task.append(init_state_index)
            coverage_rows.append(
                {
                    "subset": subset,
                    "dataset_root": str(dataset_root),
                    "init_coverage_schema_version": INIT_COVERAGE_SCHEMA_VERSION,
                    "upstream_benchmark": task_entry["upstream_benchmark"],
                    "upstream_task_id": task_id,
                    "upstream_task_name": task_name,
                    "task_texts": sorted(task_entry["task_texts"]),
                    "init_state_index": init_state_index,
                    "init_state_count": init_state_count,
                    "has_successful_gt": has_successful_gt,
                    "coverage_status": "has_successful_gt" if has_successful_gt else "no_successful_gt",
                    "successful_episode_indices": task_local_success_eps,
                    "successful_gt_count": len(task_local_success_eps),
                    "dataset_episode_indices_by_task_local_rank": task_local_eps,
                    "successful_episode_indices_by_task_local_rank": task_local_success_eps,
                    "failure_episode_indices_by_task_local_rank": task_local_failure_eps,
                    "has_successful_resolved_replay": has_successful_resolved_replay,
                    "successful_episode_indices_by_resolved_init": resolved_success_eps,
                    "successful_resolved_replay_count": len(resolved_success_eps),
                    "attempted_episode_indices": attempted_eps,
                    "attempted_episode_indices_by_resolved_init": attempted_eps,
                    "attempted_count": len(attempted_eps),
                }
            )
        task_key_text = f"{task_id}:{task_name}"
        missing_by_task[task_key_text] = missing_for_task
        missing_resolved_by_task[task_key_text] = missing_resolved_for_task

    summary = {
        "subset": subset,
        "dataset_root": str(dataset_root),
        "init_coverage_schema_version": INIT_COVERAGE_SCHEMA_VERSION,
        "task_count": len(task_info),
        "init_state_total": len(coverage_rows),
        "init_states_with_successful_gt": covered_count,
        "init_states_without_successful_gt": missing_count,
        "coverage_rate": (covered_count / len(coverage_rows)) if coverage_rows else 0.0,
        "missing_init_state_indices_by_task": missing_by_task,
        "resolved_init_states_with_successful_replay": resolved_covered_count,
        "resolved_init_states_without_successful_replay": resolved_missing_count,
        "resolved_replay_coverage_rate": (resolved_covered_count / len(coverage_rows)) if coverage_rows else 0.0,
        "missing_resolved_init_state_indices_by_task": missing_resolved_by_task,
    }
    return coverage_rows, summary


def max_observed_init_state_index(row: dict[str, Any]) -> int:
    observed: list[int] = []
    for key in (
        "resolved_init_state_index",
        "primary_init_state_index",
        "task_local_rank_init_state_index",
        "state_match_best_init_state_index",
    ):
        value = row.get(key)
        if value is not None:
            observed.append(int(value))
    observed.extend(int(index) for index in row.get("attempted_init_state_indices") or [])
    for candidate in row.get("state_match_candidates") or []:
        if candidate.get("init_state_index") is not None:
            observed.append(int(candidate["init_state_index"]))
    return max(observed, default=-1)


def install_subset_metadata(
    *,
    dataset_root: Path,
    rows: list[dict[str, Any]],
    status_summary: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    coverage_summary: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(meta_root / "replay_status.jsonl", rows)
    write_json_atomic(meta_root / "replay_status.summary.json", status_summary)
    write_json_atomic(meta_root / "replay_status.environment.json", environment)
    write_jsonl_atomic(meta_root / "replay_init_state_coverage.jsonl", coverage_rows)
    write_json_atomic(meta_root / "replay_init_state_coverage.summary.json", coverage_summary)


def expected_episode_count(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["total_episodes"])


def validate_unique_episode_rows(rows: list[dict[str, Any]], *, subset: str) -> None:
    seen: set[int] = set()
    duplicates: list[int] = []
    for row in rows:
        episode_index = int(row["dataset_episode_index"])
        if episode_index in seen:
            duplicates.append(episode_index)
        seen.add(episode_index)
        if row.get("subset") != subset:
            raise ValueError(f"Row subset mismatch: expected {subset!r}, got {row.get('subset')!r}.")
    if duplicates:
        raise ValueError(f"Duplicate episode rows for {subset}: {sorted(set(duplicates))[:20]}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("replay_status")) for row in rows)
    return {
        "total_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "success_rate": (status_counts.get("success", 0) / len(rows)) if rows else 0.0,
        "failure_episode_indices": [
            int(row["dataset_episode_index"]) for row in rows if row.get("replay_status") == "failure"
        ],
        "error_episode_indices": [
            int(row["dataset_episode_index"]) for row in rows if row.get("replay_status") == "error"
        ],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_subset_selector(raw: str) -> list[str]:
    subsets = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not subsets:
        raise ValueError("--subsets did not select any dataset subsets.")
    return subsets


if __name__ == "__main__":
    main()
