from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RolloutSuiteConfig, iter_episode_specs, validate_libero10_only
from .libero_rollout import LingBotVALiberoRunner, RolloutResult


@dataclass(frozen=True)
class SuiteRunResult:
    output_dir: Path
    results: tuple[RolloutResult | dict[str, Any], ...]
    summary_path: Path
    markdown_path: Path


def run_suite(config: RolloutSuiteConfig) -> SuiteRunResult:
    validate_libero10_only(config)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    episodes = tuple(iter_episode_specs(config))
    existing_results = _load_existing_results(results_path) if config.resume else {}
    summary_results: list[RolloutResult | dict[str, Any]] = []

    with results_path.open("a", encoding="utf-8") as jsonl:
        for checkpoint in config.checkpoints:
            missing_episodes = [
                episode
                for episode in episodes
                if _result_key(checkpoint.name, episode.to_json_dict()) not in existing_results
            ]
            if not missing_episodes:
                summary_results.extend(
                    existing_results[_result_key(checkpoint.name, episode.to_json_dict())] for episode in episodes
                )
                continue
            with LingBotVALiberoRunner(
                checkpoint,
                output_dir=output_dir,
                cuda_device=config.cuda_device,
                video_fps=config.video_fps,
                render_video=config.render_video,
            ) as runner:
                for episode in episodes:
                    key = _result_key(checkpoint.name, episode.to_json_dict())
                    if key in existing_results:
                        summary_results.append(existing_results[key])
                        continue
                    try:
                        result = runner.run_episode(
                            episode,
                            max_timestep=config.max_timestep,
                            max_chunks=config.max_chunks,
                        )
                    except Exception as exc:
                        if not config.continue_on_error:
                            raise
                        result = {
                            "checkpoint_name": checkpoint.name,
                            "benchmark": episode.benchmark,
                            "task_id": episode.task_id,
                            "episode_idx": episode.episode_idx,
                            "seed": episode.seed,
                            "sample_id": episode.sample_id,
                            "sample_kind": episode.sample_kind,
                            "sample_index": episode.sample_index,
                            "success": False,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    summary_results.append(result)
                    json_payload = _result_to_json_dict(result)
                    jsonl.write(json.dumps(json_payload, sort_keys=True) + "\n")
                    jsonl.flush()
                    existing_results[key] = json_payload
                    print(json.dumps(json_payload, sort_keys=True), flush=True)
                    _write_summary_files(config, summary_results, output_dir)

    summary_path, markdown_path = _write_summary_files(config, summary_results, output_dir)
    return SuiteRunResult(
        output_dir=output_dir,
        results=tuple(summary_results),
        summary_path=summary_path,
        markdown_path=markdown_path,
    )


def _load_existing_results(results_path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not results_path.is_file():
        return {}
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[_result_key(str(row["checkpoint_name"]), row)] = row
    return rows


def _result_key(checkpoint_name: str, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        checkpoint_name,
        row.get("benchmark"),
        int(row.get("task_id")),
        int(row.get("episode_idx")),
        row.get("seed"),
        row.get("sample_id"),
    )


def _write_summary_files(
    config: RolloutSuiteConfig,
    results: list[RolloutResult | dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    summary = _build_summary(config, results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")
    return summary_path, markdown_path


def _result_to_json_dict(result: RolloutResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, RolloutResult):
        return result.to_json_dict()
    return dict(result)


def _build_summary(config: RolloutSuiteConfig, results: list[RolloutResult | dict[str, Any]]) -> dict[str, Any]:
    rows = [_result_to_json_dict(result) for result in results]
    by_checkpoint: dict[str, dict[str, Any]] = {}
    by_benchmark_sample: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_task: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        name = str(row["checkpoint_name"])
        bucket = by_checkpoint.setdefault(
            name,
            {
                "checkpoint_name": name,
                "episodes": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "success_rate": 0.0,
                "env_timesteps": [],
                "chunk_counts": [],
            },
        )
        bucket["episodes"] += 1
        if row.get("error"):
            bucket["errors"] += 1
        if row.get("success"):
            bucket["successes"] += 1
        else:
            bucket["failures"] += 1
        if "env_timestep" in row:
            bucket["env_timesteps"].append(row["env_timestep"])
        if "chunk_count" in row:
            bucket["chunk_counts"].append(row["chunk_count"])

        group_key = (
            name,
            str(row.get("benchmark", "")),
            str(row.get("sample_kind") or "grid"),
        )
        group = by_benchmark_sample.setdefault(
            group_key,
            {
                "checkpoint_name": name,
                "benchmark": group_key[1],
                "sample_kind": group_key[2],
                "episodes": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "success_rate": 0.0,
                "env_timesteps": [],
                "chunk_counts": [],
            },
        )
        group["episodes"] += 1
        if row.get("error"):
            group["errors"] += 1
        if row.get("success"):
            group["successes"] += 1
        else:
            group["failures"] += 1
        if "env_timestep" in row:
            group["env_timesteps"].append(row["env_timestep"])
        if "chunk_count" in row:
            group["chunk_counts"].append(row["chunk_count"])

        task_key = (name, str(row.get("benchmark", "")), int(row.get("task_id", -1)))
        task_group = by_task.setdefault(
            task_key,
            {
                "checkpoint_name": name,
                "benchmark": task_key[1],
                "task_id": task_key[2],
                "episodes": 0,
                "successes": 0,
                "failures": 0,
                "errors": 0,
                "success_rate": 0.0,
                "env_timesteps": [],
                "chunk_counts": [],
            },
        )
        task_group["episodes"] += 1
        if row.get("error"):
            task_group["errors"] += 1
        if row.get("success"):
            task_group["successes"] += 1
        else:
            task_group["failures"] += 1
        if "env_timestep" in row:
            task_group["env_timesteps"].append(row["env_timestep"])
        if "chunk_count" in row:
            task_group["chunk_counts"].append(row["chunk_count"])

    for bucket in tuple(by_checkpoint.values()) + tuple(by_benchmark_sample.values()) + tuple(by_task.values()):
        episodes = int(bucket["episodes"])
        bucket["success_rate"] = float(bucket["successes"]) / episodes if episodes else 0.0
        bucket["mean_env_timestep"] = _mean(bucket.pop("env_timesteps"))
        bucket["mean_chunk_count"] = _mean(bucket.pop("chunk_counts"))

    return {
        "suite": config.to_json_dict(),
        "total_episodes": len(rows),
        "total_successes": sum(1 for row in rows if row.get("success")),
        "checkpoints": list(by_checkpoint.values()),
        "benchmark_samples": list(by_benchmark_sample.values()),
        "tasks": sorted(by_task.values(), key=lambda row: (row["checkpoint_name"], row["benchmark"], row["task_id"])),
        "results": rows,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LingBot-VA LIBERO Baseline Summary",
        "",
        f"Total episodes: `{summary['total_episodes']}`",
        f"Total successes: `{summary['total_successes']}`",
        "",
        "| Checkpoint | Successes | Episodes | Success rate | Mean env timestep | Mean chunks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in summary["checkpoints"]:
        lines.append(
            "| {name} | {successes} | {episodes} | {rate:.3f} | {timestep:.1f} | {chunks:.1f} |".format(
                name=checkpoint["checkpoint_name"],
                successes=checkpoint["successes"],
                episodes=checkpoint["episodes"],
                rate=checkpoint["success_rate"],
                timestep=checkpoint["mean_env_timestep"] or 0.0,
                chunks=checkpoint["mean_chunk_count"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## By Benchmark And Sample",
            "",
            "| Checkpoint | Benchmark | Sample | Successes | Episodes | Success rate | Mean env timestep | Mean chunks |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in summary["benchmark_samples"]:
        lines.append(
            "| {name} | {benchmark} | {sample} | {successes} | {episodes} | {rate:.3f} | {timestep:.1f} | {chunks:.1f} |".format(
                name=group["checkpoint_name"],
                benchmark=group["benchmark"],
                sample=group["sample_kind"],
                successes=group["successes"],
                episodes=group["episodes"],
                rate=group["success_rate"],
                timestep=group["mean_env_timestep"] or 0.0,
                chunks=group["mean_chunk_count"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Checkpoint | Benchmark | Task | Successes | Episodes | Success rate | Mean env timestep | Mean chunks |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in summary["tasks"]:
        lines.append(
            "| {name} | {benchmark} | {task_id} | {successes} | {episodes} | {rate:.3f} | {timestep:.1f} | {chunks:.1f} |".format(
                name=task["checkpoint_name"],
                benchmark=task["benchmark"],
                task_id=task["task_id"],
                successes=task["successes"],
                episodes=task["episodes"],
                rate=task["success_rate"],
                timestep=task["mean_env_timestep"] or 0.0,
                chunks=task["mean_chunk_count"] or 0.0,
            )
        )
    lines.extend(["", "## Episodes", ""])
    lines.append(
        "| Checkpoint | Benchmark | Sample | Task | Episode | Seed | Success | Env timestep | Chunks | Video |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for row in summary["results"]:
        video = row.get("video_path") or ""
        lines.append(
            "| {checkpoint} | {benchmark} | {sample} | {task} | {episode} | {seed} | {success} | {timestep} | {chunks} | {video} |".format(
                checkpoint=row["checkpoint_name"],
                benchmark=row.get("benchmark", ""),
                sample=row.get("sample_id") or row.get("sample_kind") or "",
                task=row["task_id"],
                episode=row["episode_idx"],
                seed=row.get("seed"),
                success="yes" if row.get("success") else "no",
                timestep=row.get("env_timestep", ""),
                chunks=row.get("chunk_count", ""),
                video=video,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / len(values)
