#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
DEFAULT_BASE_CHECKPOINT = (
    "/data/openwam_exp/runs/parallel_stream_libero_lingbot_exact_heng_compatible/checkpoints/checkpoint_step_400"
)
DEFAULT_POSTTRAINED_CHECKPOINT = (
    "/data/openwam_exp/runs/m1_exact_libero10_from_all_subsets_step400_wandb_online_from1200_modelonly/"
    "checkpoints/checkpoint_step_900"
)
DEFAULT_DATASET_ROOT = "/data/lingbot_data_exp/libero_heng/libero_10"
DEFAULT_LIBERO_REPO_ROOT = "/data/lingbot_data_exp/LIBERO"
DEFAULT_LOCAL_PATHS = "configs/local_paths.yaml"


@dataclass(frozen=True)
class DatasetEpisode:
    dataset_episode_index: int
    task_text: str
    task_index: int | None
    task_id: int
    task_name: str | None
    episode_idx: int
    length: int


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    config: str
    reference_assets_device_policy: str
    async_low_watermark: int
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchedulerSpec:
    key: str
    label: str
    flags: tuple[str, ...]
    use_method_low_watermark: bool = False


@dataclass(frozen=True)
class TargetRequest:
    method_key: str
    checkpoint_key: str
    label: str | None
    checkpoint: str


@dataclass(frozen=True)
class CheckpointResolution:
    raw: str | None
    checkpoint_file: str | None
    checkpoint_dir: str | None
    runtime_transformer_dir: str | None
    runtime_transformer_source: str | None
    problem: str | None = None


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    label: str
    checkpoint: str
    method_key: str = "m1"
    method_label: str = "M1 exact"
    config: str = DEFAULT_CONFIG
    checkpoint_raw: str | None = None
    checkpoint_file: str | None = None
    checkpoint_dir: str | None = None
    runtime_transformer_dir: str | None = None
    runtime_transformer_source: str | None = None
    reference_assets_device_policy: str = "runtime"
    extra_args: tuple[str, ...] = ()
    preflight_problem: str | None = None


@dataclass(frozen=True)
class EvalCase:
    index: int
    sample_index: int
    checkpoint_key: str
    checkpoint_label: str
    checkpoint: str
    checkpoint_raw: str | None
    checkpoint_file: str | None
    checkpoint_dir: str | None
    runtime_transformer_dir: str | None
    runtime_transformer_source: str | None
    method_key: str
    method_label: str
    config: str
    scheduler_key: str
    scheduler_label: str
    benchmark: str
    task_id: int
    task_text: str
    task_name: str | None
    dataset_episode_index: int
    episode_idx: int
    seed: int
    output_dir: str
    suffix: str
    summary_glob: str
    command_template: list[str]
    preflight_problem: str | None = None


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="m1",
        label="M1 exact",
        config=DEFAULT_CONFIG,
        reference_assets_device_policy="runtime",
        async_low_watermark=8,
    ),
    MethodSpec(
        key="m2",
        label="M2 joint denoise",
        config="configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml",
        reference_assets_device_policy="runtime",
        async_low_watermark=12,
    ),
    MethodSpec(
        key="m5",
        label="M5 action-only MoT",
        config="configs/evals/mot_libero_full_segment_non_joint_action_only_eval.yaml",
        reference_assets_device_policy="cpu_offload",
        async_low_watermark=16,
        extra_args=(
            "--runtime-devices",
            "{device}",
            "--runtime-prep-device",
            "{device}",
            "--runtime-output-device",
            "{device}",
        ),
    ),
)


SCHEDULERS: tuple[SchedulerSpec, ...] = (
    SchedulerSpec(
        key="blocking_control",
        label="blocking_control",
        flags=(
            "--planner-mode",
            "history_only",
            "--sequence-empty-plan-policy",
            "wait_for_replan",
            "--fallback-history-policy",
            "include_fallback_history",
            "--startup-open-loop-chunks",
            "0",
            "--replan-low-watermark-actions",
            "0",
        ),
    ),
    SchedulerSpec(
        key="freeze_until_clean_chunk",
        label="freeze_until_clean_chunk",
        flags=(
            "--planner-mode",
            "history_only",
            "--sequence-empty-plan-policy",
            "fallback",
            "--fallback-history-policy",
            "freeze_until_clean_chunk",
            "--replan-low-watermark-actions",
            "0",
        ),
    ),
    SchedulerSpec(
        key="async_history_first",
        label="async_history_first",
        flags=(
            "--planner-mode",
            "async_history_first",
            "--sequence-empty-plan-policy",
            "fallback",
            "--fallback-history-policy",
            "freeze_until_clean_chunk",
            "--startup-open-loop-chunks",
            "1",
        ),
        use_method_low_watermark=True,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample LIBERO dataset episodes according to the dataset task distribution, then run local "
            "realtime rollout comparisons for one or more method/checkpoint targets."
        )
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-label", type=str, default="sampled_eval")
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--task-id-source",
        choices=("auto", "libero", "metadata"),
        default="auto",
        help=(
            "`auto`/`libero` resolve task text through the requested upstream LIBERO benchmark. "
            "`metadata` explicitly assumes meta/tasks.jsonl task_index already matches LIBERO task_id."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of m1,m2,m5. Defaults to m1 unless --target is used, "
            "in which case methods are inferred from the targets."
        ),
    )
    parser.add_argument(
        "--scheduler-profile",
        choices=tuple(scheduler.key for scheduler in SCHEDULERS),
        default="blocking_control",
        help="Realtime scheduler profile to apply to every generated rollout case.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="METHOD:KEY[:LABEL]=CHECKPOINT",
        help=(
            "Explicit method/checkpoint target. May be repeated, e.g. "
            "`--target m2:base=/runs/m2_base --target m2:posttrained:latest=/runs/m2_ft`. "
            "When omitted, base/posttrained targets are built for --methods from method-specific args/env."
        ),
    )
    parser.add_argument("--base-checkpoint", type=str, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--posttrained-checkpoint", type=str, default=DEFAULT_POSTTRAINED_CHECKPOINT)
    parser.add_argument("--m1-base-checkpoint", type=str, default=None)
    parser.add_argument("--m1-posttrained-checkpoint", type=str, default=None)
    parser.add_argument("--m2-base-checkpoint", type=str, default=None)
    parser.add_argument("--m2-posttrained-checkpoint", type=str, default=None)
    parser.add_argument("--m5-base-checkpoint", type=str, default=None)
    parser.add_argument("--m5-posttrained-checkpoint", type=str, default=None)
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional config override for all targets. Omit to use each method's default eval config.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--local-paths", type=Path, default=Path(DEFAULT_LOCAL_PATHS))
    parser.add_argument("--libero-repo-root", type=Path, default=Path(DEFAULT_LIBERO_REPO_ROOT))
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--execute", action="store_true", help="Run generated cases locally.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-missing", action="store_true")
    parser.add_argument("--max-actions", type=int, default=3000)
    parser.add_argument("--env-horizon", type=int, default=5000)
    parser.add_argument("--target-action-hz", type=float, default=10.0)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed passed to run_libero_realtime_sandbox.py.")
    parser.add_argument("--deadline-miss-policy", type=str, default="hold_state")
    parser.add_argument(
        "--reference-assets-device-policy",
        choices=("cpu_offload", "runtime"),
        default=None,
        help="Optional reference-asset placement override. Omit to use the method default.",
    )
    parser.add_argument("--mujoco-gl", type=str, default="osmesa")
    parser.add_argument("--write-fallback-timeline-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collect", type=Path, default=None, help="Collect an existing run directory and exit.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.collect is not None:
        summary = collect_run(args.collect)
        print(json.dumps(summary, indent=2))
        raise SystemExit(0)

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    target_requests = parse_target_requests(args.target)
    method_selector = args.methods
    if method_selector is None:
        method_selector = ",".join(dict.fromkeys(target.method_key for target in target_requests)) or "m1"
    selected_methods = select_by_key(METHODS, method_selector, field_name="methods")
    scheduler_spec = select_by_key(SCHEDULERS, args.scheduler_profile, field_name="scheduler_profile")[0]
    checkpoint_specs = resolve_checkpoint_specs(
        args=args,
        selected_methods=selected_methods,
        target_requests=target_requests,
    )

    run_id = args.run_id or (
        f"{sanitize_label(args.run_label)}_{sanitize_label(args.benchmark)}_n{args.num_episodes}_"
        f"seed{args.sample_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root = (
        args.output_root
        or REPO_ROOT / "outputs" / run_id
    ).resolve()
    log_root = output_root / "_sampled_eval"
    logs_dir = log_root / "logs"
    status_dir = log_root / "status"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    dataset_preflight_problem = None
    sampled_episodes: list[DatasetEpisode] = []
    sample_allocations: dict[str, int] = {}
    task_resolution_warnings: list[str] = []
    if not args.dataset_root.is_dir():
        dataset_preflight_problem = f"dataset root does not exist: {args.dataset_root}"
    else:
        metadata = load_lerobot_metadata(args.dataset_root)
        task_id_map, task_name_map, task_resolution_warnings = resolve_task_ids(
            metadata["task_text_to_index"],
            benchmark=args.benchmark,
            mode=args.task_id_source,
            libero_repo_root=args.libero_repo_root,
        )
        dataset_episodes = build_dataset_episodes(
            metadata["episodes"],
            task_text_to_index=metadata["task_text_to_index"],
            task_text_to_task_id=task_id_map,
            task_text_to_task_name=task_name_map,
        )
        sampled_episodes, sample_allocations = sample_episodes_by_task_distribution(
            dataset_episodes,
            count=args.num_episodes,
            seed=args.sample_seed,
        )

    cases = build_cases(
        sampled_episodes,
        checkpoint_specs=checkpoint_specs,
        output_root=output_root,
        benchmark=args.benchmark,
        seed=args.seed,
        scheduler_spec=scheduler_spec,
        args=args,
    )

    cases_path = log_root / "cases.json"
    manifest_path = log_root / "manifest.json"
    sample_manifest_path = log_root / "sample_manifest.json"
    cases_path.write_text(json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8")
    sample_manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "num_episodes": args.num_episodes,
                "sample_seed": args.sample_seed,
                "task_allocations": sample_allocations,
                "episodes": [asdict(episode) for episode in sampled_episodes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    missing = preflight(
        args=args,
        checkpoint_specs=checkpoint_specs,
        dataset_problem=dataset_preflight_problem,
    )
    manifest = {
        "run_id": run_id,
        "run_label": args.run_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "dataset_root": str(args.dataset_root),
        "benchmark": args.benchmark,
        "num_sampled_episodes": args.num_episodes,
        "sample_seed": args.sample_seed,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "cases_path": str(cases_path),
        "sample_manifest_path": str(sample_manifest_path),
        "status_dir": str(status_dir),
        "devices": parse_devices(args.devices),
        "methods": [method.key for method in selected_methods],
        "scheduler_profile": scheduler_spec.key,
        "task_id_source": args.task_id_source,
        "checkpoint_specs": [asdict(spec) for spec in checkpoint_specs],
        "task_allocations": sample_allocations,
        "task_resolution_warnings": task_resolution_warnings,
        "missing": missing,
        "execute": bool(args.execute),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status_note(log_root / "status.md", manifest=manifest, cases=cases)

    if missing and args.execute and not args.ignore_missing:
        print(json.dumps({"status": "blocked_missing_preflight", **manifest}, indent=2))
        raise SystemExit(2)

    if not args.execute:
        print(json.dumps({"status": "generated", **manifest}, indent=2))
        return

    run_cases(cases, args=args, status_dir=status_dir, logs_dir=logs_dir)
    summary = collect_run(log_root)
    print(json.dumps(summary, indent=2))
    failed = [case for case in summary["cases"] if case["status"] and case["status"].get("state") == "failed"]
    raise SystemExit(1 if failed else 0)


def parse_target_requests(values: list[str]) -> list[TargetRequest]:
    requests: list[TargetRequest] = []
    seen: set[tuple[str, str]] = set()
    for raw_value in values:
        if "=" not in raw_value:
            raise ValueError(
                f"Invalid --target {raw_value!r}; expected METHOD:KEY[:LABEL]=CHECKPOINT."
            )
        raw_selector, checkpoint = raw_value.split("=", 1)
        pieces = [piece.strip() for piece in raw_selector.split(":", 2)]
        if len(pieces) < 2 or not pieces[0] or not pieces[1] or not checkpoint.strip():
            raise ValueError(
                f"Invalid --target {raw_value!r}; expected METHOD:KEY[:LABEL]=CHECKPOINT."
            )
        method_key = pieces[0].lower()
        checkpoint_key = sanitize_label(pieces[1])
        label = pieces[2].strip() if len(pieces) == 3 and pieces[2].strip() else None
        duplicate_key = (method_key, checkpoint_key)
        if duplicate_key in seen:
            raise ValueError(f"Duplicate --target for {method_key}:{checkpoint_key}.")
        seen.add(duplicate_key)
        requests.append(
            TargetRequest(
                method_key=method_key,
                checkpoint_key=checkpoint_key,
                label=label,
                checkpoint=checkpoint.strip(),
            )
        )
    return requests


def select_by_key(items: tuple[Any, ...], selector: str, *, field_name: str) -> list[Any]:
    by_key = {item.key: item for item in items}
    selected: list[Any] = []
    seen: set[str] = set()
    for raw_key in selector.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key not in by_key:
            valid = ", ".join(sorted(by_key))
            raise ValueError(f"Unknown {field_name} key {key!r}; expected one of: {valid}.")
        if key in seen:
            continue
        seen.add(key)
        selected.append(by_key[key])
    if not selected:
        raise ValueError(f"{field_name} selector did not select any entries.")
    return selected


def resolve_checkpoint_specs(
    *,
    args: argparse.Namespace,
    selected_methods: list[MethodSpec],
    target_requests: list[TargetRequest],
) -> list[CheckpointSpec]:
    method_by_key = {method.key: method for method in METHODS}
    selected_method_keys = {method.key for method in selected_methods}
    requests = target_requests or default_target_requests(args=args, selected_methods=selected_methods)
    specs: list[CheckpointSpec] = []
    seen_keys: set[str] = set()
    for request in requests:
        if request.method_key not in method_by_key:
            valid = ", ".join(sorted(method_by_key))
            raise ValueError(f"Unknown target method {request.method_key!r}; expected one of: {valid}.")
        if request.method_key not in selected_method_keys:
            raise ValueError(
                f"Target method {request.method_key!r} is not included in --methods "
                f"({', '.join(sorted(selected_method_keys))})."
            )
        method = method_by_key[request.method_key]
        key = sanitize_label(f"{method.key}_{request.checkpoint_key}")
        if key in seen_keys:
            raise ValueError(f"Duplicate resolved target key {key!r}.")
        seen_keys.add(key)
        resolution = resolve_checkpoint_input(request.checkpoint)
        config = args.cfg or method.config
        reference_policy = args.reference_assets_device_policy or method.reference_assets_device_policy
        label = request.label or f"{method.label} {request.checkpoint_key}"
        specs.append(
            CheckpointSpec(
                key=key,
                label=label,
                checkpoint=resolution.checkpoint_file or request.checkpoint,
                method_key=method.key,
                method_label=method.label,
                config=config,
                checkpoint_raw=resolution.raw,
                checkpoint_file=resolution.checkpoint_file,
                checkpoint_dir=resolution.checkpoint_dir,
                runtime_transformer_dir=resolution.runtime_transformer_dir,
                runtime_transformer_source=resolution.runtime_transformer_source,
                reference_assets_device_policy=reference_policy,
                extra_args=method.extra_args,
                preflight_problem=resolution.problem,
            )
        )
    return specs


def default_target_requests(*, args: argparse.Namespace, selected_methods: list[MethodSpec]) -> list[TargetRequest]:
    requests: list[TargetRequest] = []
    for method in selected_methods:
        for checkpoint_key in ("base", "posttrained"):
            checkpoint = checkpoint_arg_for_method_stage(method.key, checkpoint_key, args)
            requests.append(
                TargetRequest(
                    method_key=method.key,
                    checkpoint_key=checkpoint_key,
                    label=f"{method.label} {checkpoint_key}",
                    checkpoint=checkpoint or "",
                )
            )
    return requests


def checkpoint_arg_for_method_stage(method_key: str, checkpoint_key: str, args: argparse.Namespace) -> str | None:
    cli_value = getattr(args, f"{method_key}_{checkpoint_key}_checkpoint")
    if cli_value:
        return cli_value

    env_names = (
        f"OPEN_WAM_SAMPLED_{method_key.upper()}_{checkpoint_key.upper()}_CHECKPOINT",
        f"OPEN_WAM_{method_key.upper()}_{checkpoint_key.upper()}_CHECKPOINT",
    )
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    if method_key == "m1":
        return args.base_checkpoint if checkpoint_key == "base" else args.posttrained_checkpoint
    return None


def load_lerobot_metadata(dataset_root: Path) -> dict[str, Any]:
    meta_root = dataset_root / "meta"
    info_path = meta_root / "info.json"
    episodes_path = meta_root / "episodes.jsonl"
    tasks_path = meta_root / "tasks.jsonl"
    for path in (info_path, episodes_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required LeRobot metadata file: {path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_jsonl(episodes_path)
    task_records = read_jsonl(tasks_path)
    task_text_to_index = {str(record["task"]): int(record["task_index"]) for record in task_records}
    return {"info": info, "episodes": episodes, "task_text_to_index": task_text_to_index}


def build_dataset_episodes(
    episode_records: list[dict[str, Any]],
    *,
    task_text_to_index: dict[str, int],
    task_text_to_task_id: dict[str, int],
    task_text_to_task_name: dict[str, str | None],
) -> list[DatasetEpisode]:
    task_counts: dict[str, int] = defaultdict(int)
    episodes: list[DatasetEpisode] = []
    for record in sorted(episode_records, key=lambda item: int(item["episode_index"])):
        tasks = record.get("tasks") or ()
        if not tasks:
            raise ValueError(f"Episode {record.get('episode_index')} has no task text.")
        task_text = str(tasks[0])
        if task_text not in task_text_to_task_id:
            raise ValueError(f"Task text {task_text!r} has no resolved LIBERO task id.")
        task_local_rank = task_counts[task_text]
        task_counts[task_text] += 1
        episodes.append(
            DatasetEpisode(
                dataset_episode_index=int(record["episode_index"]),
                task_text=task_text,
                task_index=task_text_to_index.get(task_text),
                task_id=int(task_text_to_task_id[task_text]),
                task_name=task_text_to_task_name.get(task_text),
                episode_idx=task_local_rank,
                length=int(record.get("length", 0)),
            )
        )
    return episodes


def resolve_task_ids(
    task_text_to_index: dict[str, int],
    *,
    benchmark: str,
    mode: str,
    libero_repo_root: Path | None = None,
) -> tuple[dict[str, int], dict[str, str | None], list[str]]:
    if mode not in {"auto", "libero", "metadata"}:
        raise ValueError(f"Unsupported task id source: {mode}")
    if mode == "metadata":
        warning = (
            "Using LeRobot metadata task_index as LIBERO task_id. This is only valid after verifying "
            "the local metadata order matches the requested upstream LIBERO benchmark order."
        )
        return (
            {task_text: int(task_index) for task_text, task_index in task_text_to_index.items()},
            {task_text: None for task_text in task_text_to_index},
            [warning],
        )

    old_libero_repo_root = os.environ.get("LIBERO_REPO_ROOT")
    override_libero_repo_root = libero_repo_root is not None
    if override_libero_repo_root:
        os.environ["LIBERO_REPO_ROOT"] = str(libero_repo_root)

    try:
        try:
            from open_wam.integrations.libero_env import resolve_libero_task
        except Exception as exc:
            raise RuntimeError(
                "Could not import the LIBERO task resolver. Refusing to fall back to metadata task_index "
                "because local LeRobot task order may differ from upstream benchmark task ids. "
                "Use --task-id-source metadata only after verifying the orders match."
            ) from exc

        task_ids: dict[str, int] = {}
        task_names: dict[str, str | None] = {}
        warnings: list[str] = []
        for task_text in sorted(task_text_to_index):
            try:
                task_spec = resolve_libero_task(task_text, REPO_ROOT, benchmark_name=benchmark)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not resolve task text {task_text!r} inside LIBERO benchmark {benchmark!r}. "
                    "Refusing to fall back to metadata task_index because local LeRobot task order may "
                    "differ from upstream benchmark task ids. Use --task-id-source metadata only after "
                    "verifying the orders match."
                ) from exc
            if task_spec.benchmark_name != benchmark:
                warnings.append(
                    f"task {task_text!r} resolved to benchmark {task_spec.benchmark_name!r}, expected {benchmark!r}"
                )
            task_ids[task_text] = int(task_spec.task_id)
            task_names[task_text] = str(task_spec.task_name)
        return task_ids, task_names, warnings
    finally:
        if override_libero_repo_root:
            if old_libero_repo_root is None:
                os.environ.pop("LIBERO_REPO_ROOT", None)
            else:
                os.environ["LIBERO_REPO_ROOT"] = old_libero_repo_root


def sample_episodes_by_task_distribution(
    episodes: list[DatasetEpisode],
    *,
    count: int,
    seed: int,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    if count > len(episodes):
        raise ValueError(f"Cannot sample {count} episodes without replacement from only {len(episodes)} episodes.")

    by_task: dict[str, list[DatasetEpisode]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_text].append(episode)
    for task_episodes in by_task.values():
        task_episodes.sort(key=lambda item: item.dataset_episode_index)

    allocations = allocate_proportional_counts(
        {task_text: len(task_episodes) for task_text, task_episodes in by_task.items()},
        total=count,
    )
    rng = random.Random(seed)
    selected: list[DatasetEpisode] = []
    for task_text in sorted(by_task, key=lambda text: (by_task[text][0].task_id, text)):
        task_count = allocations[task_text]
        if task_count <= 0:
            continue
        selected.extend(sorted(rng.sample(by_task[task_text], task_count), key=lambda item: item.episode_idx))
    selected.sort(key=lambda item: (item.task_id, item.episode_idx, item.dataset_episode_index))
    return selected, allocations


def allocate_proportional_counts(group_sizes: dict[str, int], *, total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    if total > sum(group_sizes.values()):
        raise ValueError("total cannot exceed the sum of group sizes.")
    if any(size < 0 for size in group_sizes.values()):
        raise ValueError("group sizes must be non-negative.")

    usable = {key: size for key, size in group_sizes.items() if size > 0}
    if total and not usable:
        raise ValueError("cannot allocate positive total across empty groups.")
    population = sum(usable.values())
    ideals = {key: (total * size / population) for key, size in usable.items()}
    allocations = {key: min(int(math.floor(ideal)), usable[key]) for key, ideal in ideals.items()}
    remaining = total - sum(allocations.values())

    def priority(key: str) -> tuple[float, int, str]:
        return (ideals[key] - math.floor(ideals[key]), usable[key], key)

    while remaining > 0:
        candidates = [key for key in usable if allocations[key] < usable[key]]
        if not candidates:
            raise RuntimeError("proportional allocation exhausted all groups before reaching total.")
        for key in sorted(candidates, key=priority, reverse=True):
            if remaining <= 0:
                break
            allocations[key] += 1
            remaining -= 1

    return {key: allocations.get(key, 0) for key in group_sizes}


def build_cases(
    sampled_episodes: list[DatasetEpisode],
    *,
    checkpoint_specs: list[CheckpointSpec],
    output_root: Path,
    benchmark: str,
    seed: int,
    scheduler_spec: SchedulerSpec,
    args: argparse.Namespace,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    method_by_key = {method.key: method for method in METHODS}
    for sample_index, episode in enumerate(sampled_episodes):
        for checkpoint_spec in checkpoint_specs:
            method = method_by_key[checkpoint_spec.method_key]
            scheduler_flags = scheduler_flags_for(method, scheduler_spec)
            scheduler_suffix = scheduler_suffix_for(method, scheduler_spec)
            checkpoint_output_dir = output_root / checkpoint_spec.key
            suffix = sanitize_label(
                f"{checkpoint_spec.key}_{args.run_label}_{benchmark}_sample{sample_index:03d}_"
                f"dataset_ep{episode.dataset_episode_index:06d}_t{episode.task_id:02d}_e{episode.episode_idx}_"
                f"seed{seed}_{scheduler_suffix}"
            )
            command = [
                str(args.python),
                "scripts/run_libero_realtime_sandbox.py",
                "--cfg",
                checkpoint_spec.config,
                "--checkpoint",
                checkpoint_spec.checkpoint,
                "--benchmark",
                benchmark,
                "--task-id",
                str(episode.task_id),
                "--episode-idx",
                str(episode.episode_idx),
                "--max-actions",
                str(args.max_actions),
                "--env-horizon",
                str(args.env_horizon),
                "--target-action-hz",
                str(args.target_action_hz),
                "--video-fps",
                str(args.video_fps),
                "--runtime-device",
                "{device}",
                "--frontend-device",
                "{device}",
                "--decode-device",
                "{device}",
                "--reference-assets-device-policy",
                checkpoint_spec.reference_assets_device_policy,
                "--output-dir",
                str(checkpoint_output_dir),
                "--seed",
                str(seed),
                "--deadline-miss-policy",
                args.deadline_miss_policy,
                *checkpoint_spec.extra_args,
                *scheduler_flags,
                "--suffix",
                suffix,
            ]
            if args.write_fallback_timeline_video:
                command.append("--write-fallback-timeline-video")
            cases.append(
                EvalCase(
                    index=len(cases),
                    sample_index=sample_index,
                    checkpoint_key=checkpoint_spec.key,
                    checkpoint_label=checkpoint_spec.label,
                    checkpoint=checkpoint_spec.checkpoint,
                    checkpoint_raw=checkpoint_spec.checkpoint_raw,
                    checkpoint_file=checkpoint_spec.checkpoint_file,
                    checkpoint_dir=checkpoint_spec.checkpoint_dir,
                    runtime_transformer_dir=checkpoint_spec.runtime_transformer_dir,
                    runtime_transformer_source=checkpoint_spec.runtime_transformer_source,
                    method_key=checkpoint_spec.method_key,
                    method_label=checkpoint_spec.method_label,
                    config=checkpoint_spec.config,
                    scheduler_key=scheduler_spec.key,
                    scheduler_label=scheduler_spec.label,
                    benchmark=benchmark,
                    task_id=episode.task_id,
                    task_text=episode.task_text,
                    task_name=episode.task_name,
                    dataset_episode_index=episode.dataset_episode_index,
                    episode_idx=episode.episode_idx,
                    seed=seed,
                    output_dir=str(checkpoint_output_dir),
                    suffix=suffix,
                    summary_glob=str(
                        checkpoint_output_dir
                        / benchmark
                        / f"{episode.task_id}_*"
                        / f"{episode.episode_idx}_{suffix}.json"
                    ),
                    command_template=command,
                    preflight_problem=checkpoint_spec.preflight_problem,
                )
            )
    return cases


def preflight(
    *,
    args: argparse.Namespace,
    checkpoint_specs: list[CheckpointSpec],
    dataset_problem: str | None,
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    if dataset_problem is not None:
        missing.append({"kind": "dataset", "path": str(args.dataset_root), "reason": dataset_problem})
    if not args.python.is_file():
        missing.append({"kind": "python", "path": str(args.python), "reason": "python executable is missing"})
    local_paths = args.local_paths if args.local_paths.is_absolute() else REPO_ROOT / args.local_paths
    if not local_paths.is_file():
        missing.append({"kind": "local_paths", "path": str(local_paths), "reason": "local paths file is missing"})
    if not args.libero_repo_root.is_dir():
        missing.append(
            {
                "kind": "libero_repo_root",
                "path": str(args.libero_repo_root),
                "reason": "LIBERO repo root is missing",
            }
        )

    seen_checkpoints: set[str] = set()
    seen_configs: set[str] = set()
    for spec in checkpoint_specs:
        config_path = Path(spec.config)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        config_key = str(config_path.resolve())
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            if not config_path.is_file():
                missing.append({"kind": "config", "path": str(config_path), "reason": "config file is missing"})

        checkpoint_key = spec.checkpoint_file or spec.checkpoint_raw or spec.key
        if checkpoint_key in seen_checkpoints:
            continue
        seen_checkpoints.add(checkpoint_key)
        if spec.preflight_problem is not None:
            missing.append(
                {
                    "kind": "checkpoint",
                    "path": str(spec.checkpoint_raw or spec.checkpoint),
                    "reason": spec.preflight_problem,
                }
            )
    return missing


def scheduler_flags_for(method: MethodSpec, scheduler: SchedulerSpec) -> list[str]:
    flags = list(scheduler.flags)
    if scheduler.key == "freeze_until_clean_chunk":
        startup_chunks = "1" if method.key == "m5" else "0"
        flags.extend(["--startup-open-loop-chunks", startup_chunks])
    if scheduler.use_method_low_watermark:
        flags.extend(["--replan-low-watermark-actions", str(method.async_low_watermark)])
    return flags


def scheduler_suffix_for(method: MethodSpec, scheduler: SchedulerSpec) -> str:
    if scheduler.use_method_low_watermark:
        return f"async_k{method.async_low_watermark}_startup1"
    if method.key == "m5" and scheduler.key == "freeze_until_clean_chunk":
        return "freeze_until_clean_chunk_startup1"
    return scheduler.key


def resolve_checkpoint_input(raw_path: str | None) -> CheckpointResolution:
    if raw_path is None or not raw_path.strip():
        return CheckpointResolution(
            raw=raw_path,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem="checkpoint was not provided",
        )

    candidate = Path(raw_path).expanduser()
    if not candidate.exists():
        return CheckpointResolution(
            raw=raw_path,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem="checkpoint path does not exist",
        )

    checkpoint_file = find_checkpoint_file(candidate)
    if checkpoint_file is None:
        return CheckpointResolution(
            raw=raw_path,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem="could not resolve model_state.pt or full_training_state.pt",
        )

    checkpoint_dir = checkpoint_file.parent
    transformer_dir, transformer_source, transformer_problem = resolve_runtime_transformer_dir(checkpoint_dir)
    return CheckpointResolution(
        raw=raw_path,
        checkpoint_file=str(checkpoint_file.resolve()),
        checkpoint_dir=str(checkpoint_dir.resolve()),
        runtime_transformer_dir=str(transformer_dir.resolve()) if transformer_dir is not None else None,
        runtime_transformer_source=transformer_source,
        problem=transformer_problem,
    )


def find_checkpoint_file(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return candidate
    direct = state_file_in_dir(candidate)
    if direct is not None:
        return direct

    checkpoint_parent = candidate / "checkpoints"
    for root in (checkpoint_parent, candidate):
        if not root.is_dir():
            continue
        for checkpoint_dir in reversed(sorted_checkpoint_dirs(root)):
            checkpoint_file = state_file_in_dir(checkpoint_dir)
            if checkpoint_file is not None:
                return checkpoint_file
    return None


def state_file_in_dir(path: Path) -> Path | None:
    for filename in ("model_state.pt", "full_training_state.pt"):
        checkpoint_file = path / filename
        if checkpoint_file.is_file():
            return checkpoint_file.resolve()
    return None


def sorted_checkpoint_dirs(root: Path) -> list[Path]:
    checkpoint_dirs = [path for path in root.glob("checkpoint_step_*") if path.is_dir()]
    return sorted(checkpoint_dirs, key=checkpoint_step)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_runtime_transformer_dir(checkpoint_dir: Path) -> tuple[Path | None, str | None, str | None]:
    local_transformer = checkpoint_dir / "transformer"
    if is_usable_transformer_dir(local_transformer):
        return local_transformer, "checkpoint", None

    config_transformer = transformer_dir_from_resolved_config(checkpoint_dir / "resolved_config.yaml")
    if config_transformer is not None and is_usable_transformer_dir(config_transformer):
        return config_transformer, "resolved_config", None

    if local_transformer.is_dir():
        return (
            None,
            None,
            "checkpoint transformer directory exists but is empty or unusable, and resolved_config fallback is missing",
        )
    return None, None, "missing usable transformer export directory or resolved_config transformer_subdir fallback"


def transformer_dir_from_resolved_config(config_path: Path) -> Path | None:
    if not config_path.is_file():
        return None
    transformer_value = read_backbone_transformer_subdir(config_path)
    if not transformer_value:
        return None
    transformer_dir = Path(str(transformer_value)).expanduser()
    if not transformer_dir.is_absolute():
        transformer_dir = (config_path.parent / transformer_dir).resolve()
    return transformer_dir


def read_backbone_transformer_subdir(config_path: Path) -> str | None:
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return read_backbone_transformer_subdir_without_yaml(text)
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        return None
    backbone = raw.get("backbone", {})
    if not isinstance(backbone, dict):
        return None
    value = backbone.get("transformer_subdir")
    return str(value) if value else None


def read_backbone_transformer_subdir_without_yaml(text: str) -> str | None:
    in_backbone = False
    backbone_indent: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "backbone:":
            in_backbone = True
            backbone_indent = indent
            continue
        if in_backbone and backbone_indent is not None and indent <= backbone_indent:
            in_backbone = False
        if not in_backbone or not stripped.startswith("transformer_subdir:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip("'\"")
        return value or None
    return None


def is_usable_transformer_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def run_cases(cases: list[EvalCase], *, args: argparse.Namespace, status_dir: Path, logs_dir: Path) -> None:
    devices = parse_devices(args.devices)
    if not devices:
        raise ValueError("--devices selected no devices.")

    work_queue: queue.Queue[EvalCase] = queue.Queue()
    for case in cases:
        work_queue.put(case)

    def worker(device: str) -> None:
        while True:
            try:
                case = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                run_case(case, device=device, args=args, status_dir=status_dir, logs_dir=logs_dir)
            finally:
                work_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(worker, device): device for device in devices}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                device = futures.pop(future)
                exc = future.exception()
                if exc is not None:
                    if args.fail_fast:
                        raise exc
                    print(
                        json.dumps(
                            {
                                "event": "sampled_eval_worker_failed",
                                "device": device,
                                "error": repr(exc),
                            }
                        ),
                        flush=True,
                    )


def run_case(
    case: EvalCase,
    *,
    device: str,
    args: argparse.Namespace,
    status_dir: Path,
    logs_dir: Path,
) -> None:
    status_path = status_dir / f"{case.index:04d}_{case.checkpoint_key}_sample{case.sample_index:03d}.json"
    log_path = logs_dir / f"{case.index:04d}_{case.checkpoint_key}_sample{case.sample_index:03d}.log"
    existing = find_summary_paths(asdict(case))
    if args.resume and existing:
        write_case_status(
            status_path,
            state="skipped_existing",
            returncode=0,
            case=case,
            device=device,
            log_path=log_path,
        )
        return

    command = [part.replace("{device}", device) for part in case.command_template]
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    local_paths = args.local_paths if args.local_paths.is_absolute() else REPO_ROOT / args.local_paths
    env["OPEN_WAM_LOCAL_PATHS"] = str(local_paths)
    env["LIBERO_REPO_ROOT"] = str(args.libero_repo_root)
    env["MUJOCO_GL"] = args.mujoco_gl
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_MODE", "disabled")

    write_case_status(status_path, state="running", returncode=None, case=case, device=device, log_path=log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "sampled_eval_case_start", "case": asdict(case), "device": device}, indent=2))
        log.write("\n")
        log.flush()
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_case_status(
        status_path,
        state="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        case=case,
        device=device,
        log_path=log_path,
    )


def write_case_status(
    path: Path,
    *,
    state: str,
    returncode: int | None,
    case: EvalCase,
    device: str,
    log_path: Path,
) -> None:
    payload = {
        "state": state,
        "returncode": returncode,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "log_path": str(log_path),
        "case": asdict(case),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_run(path: Path) -> dict[str, Any]:
    root = path.resolve()
    log_root = root if (root / "manifest.json").is_file() else root / "_sampled_eval"
    manifest_path = log_root / "manifest.json"
    cases_path = log_root / "cases.json"
    status_dir = log_root / "status"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json under {log_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    reports: list[dict[str, Any]] = []
    for case in cases:
        status_files = sorted(
            status_dir.glob(
                f"{int(case['index']):04d}_"
                f"{case['checkpoint_key']}_"
                f"sample{int(case['sample_index']):03d}.json"
            )
        )
        status = json.loads(status_files[-1].read_text(encoding="utf-8")) if status_files else None
        summary_paths = find_summary_paths(case)
        summary = json.loads(summary_paths[-1].read_text(encoding="utf-8")) if summary_paths else None
        reports.append(
            {
                "case": case,
                "status": status,
                "summary_path": str(summary_paths[-1]) if summary_paths else None,
                "summary": summary,
            }
        )

    summary_payload = build_summary_payload(manifest, reports)
    (log_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_results_csv(log_root / "results.csv", summary_payload)
    write_summary_md(log_root / "summary.md", summary_payload)
    return summary_payload


def build_summary_payload(manifest: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
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

    paired_rows = build_paired_rows(reports)
    return {
        "run_id": manifest["run_id"],
        "output_root": manifest["output_root"],
        "log_root": manifest["log_root"],
        "dataset_root": manifest["dataset_root"],
        "benchmark": manifest.get("benchmark"),
        "num_sampled_episodes": manifest["num_sampled_episodes"],
        "task_allocations": manifest.get("task_allocations", {}),
        "target_keys": target_keys,
        "checkpoint_specs": manifest.get("checkpoint_specs", []),
        "by_checkpoint": by_checkpoint,
        "by_checkpoint_task": by_checkpoint_task,
        "paired_rows": paired_rows,
        "cases": reports,
    }


def build_paired_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_sample: dict[int, dict[str, Any]] = {}
    for report in reports:
        case = report["case"]
        sample_index = int(case["sample_index"])
        row = rows_by_sample.setdefault(
            sample_index,
            {
                "sample_index": sample_index,
                "dataset_episode_index": case["dataset_episode_index"],
                "task_id": case["task_id"],
                "task_text": case["task_text"],
                "episode_idx": case["episode_idx"],
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


def write_results_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["paired_rows"]
    target_keys = summary["target_keys"]
    fieldnames = [
        "sample_index",
        "dataset_episode_index",
        "task_id",
        "episode_idx",
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
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
        f"Dataset root: `{summary['dataset_root']}`",
        f"Output root: `{summary['output_root']}`",
        f"Log root: `{summary['log_root']}`",
        "",
        "## Aggregate",
        "",
        "| Target | Method | Success rate | Success | Finished | Total | Failed processes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in summary["target_keys"]:
        payload = summary["by_checkpoint"].get(key)
        if payload is None:
            continue
        lines.append(
            f"| `{key}` {md_escape(str(payload.get('label') or target_labels.get(key, key)))} | "
            f"{md_escape(str(payload.get('method_label') or ''))} | "
            f"{format_success_rate(payload['success'], payload['finished'])} | "
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
    headers = ["Sample", "Dataset episode", "Task", "Init state"] + [
        target_labels.get(key, key) for key in summary["target_keys"]
    ]
    aligns = ["---:", "---:", "---:", "---:"] + ["---" for _ in summary["target_keys"]]
    lines.append("| " + " | ".join(md_escape(header) for header in headers) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in summary["paired_rows"]:
        cells = [
            str(row["sample_index"]),
            str(row["dataset_episode_index"]),
            str(row["task_id"]),
            str(row["episode_idx"]),
        ]
        for key in summary["target_keys"]:
            cells.append(
                format_result_cell(
                    row.get(f"{key}_success"),
                    row.get(f"{key}_executed_actions"),
                    row.get(f"{key}_status"),
                )
            )
        lines.append(
            "| " + " | ".join(md_escape(cell) for cell in cells) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_result_cell(success: Any, actions: Any, status: Any = None) -> str:
    if success is None:
        return str(status or "missing")
    return f"{'yes' if bool(success) else 'no'} / {actions}"


def format_success_rate(success: int, finished: int) -> str:
    if finished <= 0:
        return "n/a"
    return f"{(100.0 * success / finished):.1f}%"


def md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def find_summary_paths(case: dict[str, Any]) -> list[Path]:
    import glob

    return sorted(path for raw_path in glob.glob(str(case["summary_glob"])) if (path := Path(raw_path)).is_file())


def write_status_note(path: Path, *, manifest: dict[str, Any], cases: list[EvalCase]) -> None:
    lines = [
        "# LIBERO Sampled Eval Queue",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        f"Benchmark: `{manifest['benchmark']}`",
        f"Task id source: `{manifest.get('task_id_source', 'unknown')}`",
        f"Dataset root: `{manifest['dataset_root']}`",
        f"Output root: `{manifest['output_root']}`",
        f"Cases: `{len(cases)}`",
        f"Scheduler profile: `{manifest['scheduler_profile']}`",
        "",
        "## Targets",
        "",
    ]
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


if __name__ == "__main__":
    main()
