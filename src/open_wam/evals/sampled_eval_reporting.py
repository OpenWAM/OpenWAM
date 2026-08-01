"""Aggregate and render deterministic sampled-evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import glob
import json
import os
from pathlib import Path
import time
from typing import Any, TypedDict


__all__ = [
    "SampledEvalCaseReport",
    "build_sampled_eval_paired_rows",
    "build_sampled_eval_summary",
    "collect_sampled_eval_run",
    "find_case_summary_paths",
    "write_json_atomic",
    "write_sampled_eval_queue_note",
    "write_sampled_eval_results_csv",
    "write_sampled_eval_summary_markdown",
]


class SampledEvalCaseReport(TypedDict):
    """One rollout case joined with its process status and optional summary."""

    case: dict[str, Any]
    status: dict[str, Any] | None
    summary_path: str | None
    summary: dict[str, Any] | None


def collect_sampled_eval_run(path: Path) -> dict[str, Any]:
    """Collect one sampled-eval run and persist its aggregate artifacts."""

    root = path.resolve()
    log_root = root if (root / "manifest.json").is_file() else root / "_sampled_eval"
    manifest_path = log_root / "manifest.json"
    cases_path = log_root / "cases.json"
    status_dir = log_root / "status"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json under {log_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    reports: list[SampledEvalCaseReport] = []
    for case in cases:
        status_files = sorted(
            status_dir.glob(
                f"{int(case['index']):04d}_"
                f"{case['checkpoint_key']}_"
                f"sample{int(case['sample_index']):03d}.json"
            )
        )
        status = json.loads(status_files[-1].read_text(encoding="utf-8")) if status_files else None
        summary_paths = find_case_summary_paths(case)
        summary = json.loads(summary_paths[-1].read_text(encoding="utf-8")) if summary_paths else None
        reports.append(
            {
                "case": case,
                "status": status,
                "summary_path": str(summary_paths[-1]) if summary_paths else None,
                "summary": summary,
            }
        )

    summary_payload = build_sampled_eval_summary(manifest, reports)
    write_json_atomic(log_root / "summary.json", summary_payload)
    write_sampled_eval_results_csv(log_root / "results.csv", summary_payload)
    write_sampled_eval_summary_markdown(log_root / "summary.md", summary_payload)
    return summary_payload


def build_sampled_eval_summary(
    manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate process and rollout results without reading or writing files."""

    by_checkpoint: dict[str, dict[str, Any]] = {}
    by_checkpoint_task: dict[str, dict[str, dict[str, Any]]] = {}
    target_metadata = {
        str(spec["key"]): spec
        for spec in manifest.get("checkpoint_specs", [])
        if isinstance(spec, dict) and spec.get("key") is not None
    }
    target_keys = [
        str(spec["key"])
        for spec in manifest.get("checkpoint_specs", [])
        if isinstance(spec, dict) and spec.get("key") is not None
    ]
    for report in reports:
        case = report["case"]
        key = str(case["checkpoint_key"])
        task_id = str(case["task_id"])
        if key not in target_keys:
            target_keys.append(key)
        metadata = target_metadata.get(key, {})
        by_checkpoint.setdefault(
            key,
            {
                "label": metadata.get("label", case.get("checkpoint_label", key)),
                "method_key": metadata.get("method_key", case.get("method_key")),
                "method_label": metadata.get("method_label", case.get("method_label")),
                "total": 0,
                "finished": 0,
                "success": 0,
                "failed_cases": 0,
            },
        )
        by_checkpoint_task.setdefault(key, {}).setdefault(task_id, {"total": 0, "finished": 0, "success": 0})
        by_checkpoint[key]["total"] += 1
        by_checkpoint_task[key][task_id]["total"] += 1
        status = report.get("status") or {}
        summary = report.get("summary")
        if (
            status.get("state") in {"completed", "skipped_existing"}
            and status.get("returncode") == 0
            and summary is not None
        ):
            by_checkpoint[key]["finished"] += 1
            by_checkpoint_task[key][task_id]["finished"] += 1
            if bool(summary.get("success")):
                by_checkpoint[key]["success"] += 1
                by_checkpoint_task[key][task_id]["success"] += 1
        elif status.get("state") == "failed":
            by_checkpoint[key]["failed_cases"] += 1

    paired_rows = build_sampled_eval_paired_rows(reports)
    return {
        "run_id": manifest["run_id"],
        "output_root": manifest["output_root"],
        "log_root": manifest["log_root"],
        "dataset_root": manifest["dataset_root"],
        "benchmark": manifest.get("benchmark"),
        "sample_mode": manifest.get("sample_mode", "dataset_distribution"),
        "distribution_episode_strategy": manifest.get("distribution_episode_strategy"),
        "eval_profile": manifest.get("eval_profile"),
        "rollout_artifact_profile": manifest.get("rollout_artifact_profile"),
        "scheduler_profile": manifest.get("scheduler_profile"),
        "replay_status_path": manifest.get("replay_status_path"),
        "replay_status_policy": manifest.get("replay_status_policy"),
        "require_replay_status": manifest.get("require_replay_status"),
        "replay_status_filter": manifest.get("replay_status_filter"),
        "num_sampled_episodes": manifest["num_sampled_episodes"],
        "requested_num_episodes": manifest.get("requested_num_episodes", manifest["num_sampled_episodes"]),
        "task_allocations": manifest.get("task_allocations", {}),
        "task_axis_init_source": manifest.get("task_axis_init_source"),
        "full_init_counts_by_task_id": manifest.get("full_init_counts_by_task_id", {}),
        "sample_warnings": manifest.get("sample_warnings", []),
        "target_keys": target_keys,
        "checkpoint_specs": manifest.get("checkpoint_specs", []),
        "by_checkpoint": by_checkpoint,
        "by_checkpoint_task": by_checkpoint_task,
        "paired_rows": paired_rows,
        "cases": list(reports),
    }


def build_sampled_eval_paired_rows(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join target-specific reports into one row per sampled episode."""

    rows_by_sample: dict[int, dict[str, Any]] = {}
    for report in reports:
        case = report["case"]
        sample_index = int(case["sample_index"])
        row = rows_by_sample.setdefault(
            sample_index,
            {
                "sample_index": sample_index,
                "dataset_episode_index": case["dataset_episode_index"],
                "episode_id": case.get("episode_id", case["dataset_episode_index"]),
                "task_id": case["task_id"],
                "task_text": case["task_text"],
                "init_id": case.get("init_id", case["episode_idx"]),
                "episode_idx": case["episode_idx"],
                "resolved_init_state_index": case.get("resolved_init_state_index"),
                "init_id_source": case.get("init_id_source", "task_local_rank"),
                "replay_status": case.get("replay_status"),
            },
        )
        key = str(case["checkpoint_key"])
        summary = report.get("summary") or {}
        status = report.get("status") or {}
        row[f"{key}_success"] = summary.get("success")
        row[f"{key}_executed_actions"] = summary.get("executed_actions")
        row[f"{key}_fallback_actions"] = summary.get("fallback_actions")
        row[f"{key}_status"] = status.get("state")
        row[f"{key}_returncode"] = status.get("returncode")
        row[f"{key}_summary_path"] = report.get("summary_path")
    return [rows_by_sample[index] for index in sorted(rows_by_sample)]


def write_sampled_eval_results_csv(path: Path, summary: Mapping[str, Any]) -> None:
    """Write the stable one-row-per-sample CSV view."""

    rows = summary["paired_rows"]
    target_keys = summary["target_keys"]
    fieldnames = [
        "sample_index",
        "dataset_episode_index",
        "episode_id",
        "task_id",
        "init_id",
        "episode_idx",
        "resolved_init_state_index",
        "init_id_source",
        "replay_status",
        "task_text",
    ]
    for key in target_keys:
        fieldnames.extend(
            [
                f"{key}_status",
                f"{key}_returncode",
                f"{key}_success",
                f"{key}_executed_actions",
                f"{key}_fallback_actions",
                f"{key}_summary_path",
            ]
        )
    tmp_path = _temporary_write_path(path)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_sampled_eval_summary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    """Write the human-readable aggregate and per-sample report."""

    target_labels = {
        str(spec["key"]): str(spec.get("label") or spec["key"])
        for spec in summary.get("checkpoint_specs", [])
        if isinstance(spec, dict) and spec.get("key") is not None
    }
    lines = [
        "# LIBERO Sampled Eval",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Benchmark: `{summary.get('benchmark')}`",
        f"Sample mode: `{summary.get('sample_mode', 'dataset_distribution')}`",
        f"Distribution episode strategy: `{summary.get('distribution_episode_strategy') or 'n/a'}`",
        f"Eval profile: `{summary.get('eval_profile') or 'n/a'}`",
        f"Rollout artifact profile: `{summary.get('rollout_artifact_profile') or 'n/a'}`",
        f"Scheduler profile: `{summary.get('scheduler_profile') or 'n/a'}`",
        f"Replay-status policy: `{summary.get('replay_status_policy') or 'n/a'}`",
        f"Replay-status file: `{summary.get('replay_status_path') or 'n/a'}`",
        f"Dataset root: `{summary['dataset_root']}`",
        f"Output root: `{summary['output_root']}`",
        f"Log root: `{summary['log_root']}`",
        "",
    ]
    sample_warnings = summary.get("sample_warnings") or []
    if sample_warnings:
        lines.extend(["## Sample Warnings", ""])
        lines.extend([f"- {warning}" for warning in sample_warnings])
        lines.append("")
    lines.extend(
        [
            "## Aggregate",
            "",
            "| Target | Method | Success rate | Success | Finished | Total | Failed processes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in summary["target_keys"]:
        payload = summary["by_checkpoint"].get(key)
        if payload is None:
            continue
        lines.append(
            f"| `{key}` {_md_escape(str(payload.get('label') or target_labels.get(key, key)))} | "
            f"{_md_escape(str(payload.get('method_label') or ''))} | "
            f"{_format_success_rate(payload['success'], payload['finished'])} | "
            f"{payload['success']} | {payload['finished']} | {payload['total']} | {payload['failed_cases']} |"
        )
    lines.extend(["", "## Task Allocation", ""])
    lines.extend([f"- {task}: {count}" for task, count in sorted(summary["task_allocations"].items())])
    lines.extend(
        [
            "",
            "## Per-Sample",
            "",
        ]
    )
    headers = ["Sample", "Dataset episode", "Task", "Init id", "Replay"] + [
        target_labels.get(key, key) for key in summary["target_keys"]
    ]
    aligns = ["---:", "---:", "---:", "---:", "---"] + ["---" for _ in summary["target_keys"]]
    lines.append("| " + " | ".join(_md_escape(header) for header in headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in summary["paired_rows"]:
        cells = [
            str(row["sample_index"]),
            str(row["dataset_episode_index"]),
            str(row["task_id"]),
            str(row.get("init_id", row["episode_idx"])),
            str(row.get("replay_status") or "n/a"),
        ]
        for key in summary["target_keys"]:
            cells.append(
                _format_result_cell(
                    row.get(f"{key}_success"),
                    row.get(f"{key}_executed_actions"),
                    row.get(f"{key}_status"),
                )
            )
        lines.append("| " + " | ".join(_md_escape(cell) for cell in cells) + " |")
    _write_text_atomic(path, "\n".join(lines) + "\n")


def find_case_summary_paths(case: Mapping[str, Any]) -> list[Path]:
    """Resolve existing rollout summaries for one manifest case."""

    return sorted(path for raw_path in glob.glob(str(case["summary_glob"])) if (path := Path(raw_path)).is_file())


def write_sampled_eval_queue_note(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    case_count: int,
) -> None:
    """Write the generated queue and preflight report."""

    lines = [
        "# LIBERO Sampled Eval Queue",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        f"Benchmark: `{manifest['benchmark']}`",
        f"Sample mode: `{manifest.get('sample_mode', 'dataset_distribution')}`",
        f"Distribution episode strategy: `{manifest.get('distribution_episode_strategy') or 'n/a'}`",
        f"Eval profile: `{manifest.get('eval_profile') or 'n/a'}`",
        f"Rollout artifact profile: `{manifest.get('rollout_artifact_profile') or 'n/a'}`",
        f"Replay-status policy: `{manifest.get('replay_status_policy') or 'n/a'}`",
        f"Replay-status file: `{manifest.get('replay_status_path') or 'n/a'}`",
        f"Task id source: `{manifest.get('task_id_source', 'unknown')}`",
        f"Task-axis init source: `{manifest.get('task_axis_init_source', 'auto')}`",
        f"Dataset root: `{manifest['dataset_root']}`",
        f"Output root: `{manifest['output_root']}`",
        f"Cases: `{case_count}`",
        f"Scheduler profile: `{manifest['scheduler_profile']}`",
        "",
    ]
    sample_warnings = manifest.get("sample_warnings") or []
    if sample_warnings:
        lines.extend(["## Sample Warnings", ""])
        lines.extend([f"- {warning}" for warning in sample_warnings])
        lines.append("")
    lines.extend(["## Targets", ""])
    for spec in manifest["checkpoint_specs"]:
        transformer = spec.get("runtime_transformer_dir") or spec.get("preflight_problem") or "missing"
        lines.append(
            f"- `{spec['key']}` {spec['label']} ({spec['method_label']}): "
            f"`{spec['checkpoint']}`; transformer `{transformer}`"
        )
    lines.extend(
        [
            "",
            "## Sampled Task Allocation",
            "",
        ]
    )
    if manifest["task_allocations"]:
        lines.extend([f"- {task}: {count}" for task, count in sorted(manifest["task_allocations"].items())])
    else:
        lines.append("- unavailable; dataset preflight failed")
    lines.extend(["", "## Preflight", ""])
    if manifest["missing"]:
        lines.append("Blocked: at least one required path is missing or unusable.")
        lines.append("")
        for item in manifest["missing"]:
            lines.append(f"- `{item['kind']}`: `{item['path']}` ({item.get('reason', 'missing')})")
    else:
        lines.append("All required paths passed preflight.")
    if manifest.get("task_resolution_warnings"):
        lines.extend(["", "## Task Resolution Warnings", ""])
        for warning in manifest["task_resolution_warnings"]:
            lines.append(f"- {warning}")
    _write_text_atomic(path, "\n".join(lines) + "\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write indented JSON through an adjacent temporary file."""

    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_write_path(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _temporary_write_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{time.time_ns()}.tmp"
    return path.with_name(f".{path.name}{suffix}")


def _format_result_cell(success: Any, actions: Any, status: Any = None) -> str:
    if success is None:
        return str(status or "missing")
    return f"{'yes' if bool(success) else 'no'} / {actions}"


def _format_success_rate(success: int, finished: int) -> str:
    if finished <= 0:
        return "n/a"
    return f"{(100.0 * success / finished):.1f}%"


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")
