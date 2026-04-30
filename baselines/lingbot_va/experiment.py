from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RolloutSuiteConfig, iter_episode_specs
from .libero_rollout import LingBotVALiberoRunner, RolloutResult


@dataclass(frozen=True)
class SuiteRunResult:
    output_dir: Path
    results: tuple[RolloutResult | dict[str, Any], ...]
    summary_path: Path
    markdown_path: Path


def run_suite(config: RolloutSuiteConfig) -> SuiteRunResult:
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_results: list[RolloutResult | dict[str, Any]] = []

    with results_path.open("a", encoding="utf-8") as jsonl:
        for checkpoint in config.checkpoints:
            with LingBotVALiberoRunner(
                checkpoint,
                output_dir=output_dir,
                cuda_device=config.cuda_device,
                video_fps=config.video_fps,
                render_video=config.render_video,
            ) as runner:
                for episode in iter_episode_specs(config):
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
                            "success": False,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    summary_results.append(result)
                    json_payload = _result_to_json_dict(result)
                    jsonl.write(json.dumps(json_payload, sort_keys=True) + "\n")
                    jsonl.flush()
                    print(json.dumps(json_payload, sort_keys=True), flush=True)

    summary = _build_summary(config, summary_results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")
    return SuiteRunResult(
        output_dir=output_dir,
        results=tuple(summary_results),
        summary_path=summary_path,
        markdown_path=markdown_path,
    )


def _result_to_json_dict(result: RolloutResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, RolloutResult):
        return result.to_json_dict()
    return dict(result)


def _build_summary(config: RolloutSuiteConfig, results: list[RolloutResult | dict[str, Any]]) -> dict[str, Any]:
    rows = [_result_to_json_dict(result) for result in results]
    by_checkpoint: dict[str, dict[str, Any]] = {}
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

    for bucket in by_checkpoint.values():
        episodes = int(bucket["episodes"])
        bucket["success_rate"] = float(bucket["successes"]) / episodes if episodes else 0.0
        bucket["mean_env_timestep"] = _mean(bucket.pop("env_timesteps"))
        bucket["mean_chunk_count"] = _mean(bucket.pop("chunk_counts"))

    return {
        "suite": config.to_json_dict(),
        "total_episodes": len(rows),
        "total_successes": sum(1 for row in rows if row.get("success")),
        "checkpoints": list(by_checkpoint.values()),
        "results": rows,
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LingBot-VA LIBERO-10 Baseline Summary",
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
    lines.extend(["", "## Episodes", ""])
    lines.append("| Checkpoint | Task | Episode | Seed | Success | Env timestep | Chunks | Video |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for row in summary["results"]:
        video = row.get("video_path") or ""
        lines.append(
            "| {checkpoint} | {task} | {episode} | {seed} | {success} | {timestep} | {chunks} | {video} |".format(
                checkpoint=row["checkpoint_name"],
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
