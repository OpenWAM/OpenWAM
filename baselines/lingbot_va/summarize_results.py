from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import CheckpointSpec, RolloutSuiteConfig, parse_int_selection


def main() -> None:
    from .experiment import _build_markdown, _build_summary

    parser = argparse.ArgumentParser(description="Merge LingBot-VA baseline JSONL shards into one summary.")
    parser.add_argument("--results-jsonl", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--expect-benchmark")
    parser.add_argument("--expect-task-ids")
    parser.add_argument("--expect-episode-indices")
    parser.add_argument("--require-null-seed", action="store_true")
    parser.add_argument("--require-unique", action="store_true")
    parser.add_argument("--require-hf-revision")
    parser.add_argument("--require-model-root")
    args = parser.parse_args()

    rows = _load_rows(tuple(Path(path).expanduser() for path in args.results_jsonl))
    validate_rows(
        rows,
        expect_count=args.expect_count,
        expect_benchmark=args.expect_benchmark,
        expect_task_ids=parse_int_selection(args.expect_task_ids) if args.expect_task_ids else None,
        expect_episode_indices=(
            parse_int_selection(args.expect_episode_indices) if args.expect_episode_indices else None
        ),
        require_null_seed=args.require_null_seed,
        require_unique=args.require_unique,
        require_hf_revision=args.require_hf_revision,
        require_model_root=args.require_model_root,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(_suite_from_rows(rows, output_dir), rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"markdown: {markdown_path}")


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    expect_count: int | None = None,
    expect_benchmark: str | None = None,
    expect_task_ids: list[int] | None = None,
    expect_episode_indices: list[int] | None = None,
    require_null_seed: bool = False,
    require_unique: bool = False,
    require_hf_revision: str | None = None,
    require_model_root: str | None = None,
) -> None:
    if expect_count is not None and len(rows) != expect_count:
        raise ValueError(f"Expected {expect_count} rows, found {len(rows)}.")
    if expect_benchmark is not None:
        found = {str(row.get("benchmark")) for row in rows}
        if found != {expect_benchmark}:
            raise ValueError(f"Expected benchmark {expect_benchmark!r}, found {sorted(found)!r}.")
    if expect_task_ids is not None:
        found = {int(row["task_id"]) for row in rows}
        expected = set(expect_task_ids)
        if found != expected:
            raise ValueError(f"Expected task IDs {sorted(expected)!r}, found {sorted(found)!r}.")
    if expect_episode_indices is not None:
        found = {int(row["episode_idx"]) for row in rows}
        expected = set(expect_episode_indices)
        if found != expected:
            raise ValueError(f"Expected episode indices {sorted(expected)!r}, found {sorted(found)!r}.")
    if require_null_seed:
        bad = [(int(row["task_id"]), int(row["episode_idx"]), row.get("seed")) for row in rows if row.get("seed") is not None]
        if bad:
            raise ValueError(f"Expected all seeds to be null; first non-null rows: {bad[:5]!r}.")
    if require_unique:
        seen: set[tuple[str, int, int]] = set()
        duplicates: list[tuple[str, int, int]] = []
        for row in rows:
            key = (str(row.get("checkpoint_name")), int(row["task_id"]), int(row["episode_idx"]))
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        if duplicates:
            raise ValueError(f"Duplicate checkpoint/task/episode rows: {duplicates[:5]!r}.")
    if require_hf_revision is not None:
        found = {str((row.get("prepared_model") or {}).get("hf_revision")) for row in rows}
        if found != {require_hf_revision}:
            raise ValueError(f"Expected HF revision {require_hf_revision!r}, found {sorted(found)!r}.")
    if require_model_root is not None:
        expected = str(Path(require_model_root).expanduser().resolve())
        found = {
            str(Path(str((row.get("prepared_model") or {}).get("model_root", "."))).expanduser().resolve())
            for row in rows
        }
        if found != {expected}:
            raise ValueError(f"Expected model root {expected!r}, found {sorted(found)!r}.")


def _load_rows(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: (str(row.get("checkpoint_name")), int(row.get("task_id")), int(row.get("episode_idx"))))
    return rows


def _suite_from_rows(rows: list[dict[str, Any]], output_dir: Path) -> RolloutSuiteConfig:
    if not rows:
        raise ValueError("Cannot summarize an empty result set.")
    first = rows[0]
    prepared_model = first.get("prepared_model") or {}
    checkpoint = CheckpointSpec(
        name=str(first["checkpoint_name"]),
        model_root=Path(str(prepared_model.get("model_root", "."))),
        source_repo=Path(str(first["source_repo"])) if first.get("source_repo") else None,
        hf_repo_id=prepared_model.get("hf_repo_id"),
        hf_revision=prepared_model.get("hf_revision"),
    )
    return RolloutSuiteConfig(
        checkpoints=(checkpoint,),
        benchmark="libero_10",
        task_ids=tuple(sorted({int(row["task_id"]) for row in rows})),
        episode_indices=tuple(sorted({int(row["episode_idx"]) for row in rows})),
        seed=first.get("seed"),
        max_timestep=max(int(row.get("max_timestep", 0)) for row in rows),
        video_fps=60.0,
        output_dir=output_dir,
        render_video=any(bool(row.get("video_path")) for row in rows),
        continue_on_error=any(bool(row.get("error")) for row in rows),
        resume=True,
    )


if __name__ == "__main__":
    main()
