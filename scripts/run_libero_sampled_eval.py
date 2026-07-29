#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
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
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data.replay_status import (  # noqa: E402
    REPLAY_STATUS_POLICIES,
    ReplayStatusFilterReport,
    filter_episode_indices_by_replay_status,
    load_replay_status_records,
    normalize_replay_status_policy,
)
from open_wam.configs.enums import ParallelStreamVariantProfile  # noqa: E402
from open_wam.configs import load_experiment_config  # noqa: E402

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
SAMPLE_MODE_ALIASES = {"uniform_task_distribution": "dataset_distribution"}
SAMPLE_MODE_CHOICES = ("dataset_distribution", "uniform_task_distribution", "task_episode_axis", "full")
GJD_CONFIG_MARKERS = ("generalist_joint_denoising",)


def normalize_sample_mode(mode: str) -> str:
    return SAMPLE_MODE_ALIASES.get(mode, mode)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(str(config_path))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _config_has_gjd_semantics(config: Any) -> bool:
    policy_variant = getattr(config, "policy_variant", None)
    if (
        _enum_value(getattr(policy_variant, "variant_profile", None))
        == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING.value
    ):
        return True
    if getattr(policy_variant, "mot_generalist_training_mode_probs", None) is not None:
        return True
    return False


def is_gjd_config_path(config_path: str | Path | None) -> bool:
    if config_path is None:
        return False
    try:
        config = load_experiment_config(_resolve_config_path(config_path))
    except Exception:
        normalized = str(config_path).lower()
        return any(marker in normalized for marker in GJD_CONFIG_MARKERS)
    return _config_has_gjd_semantics(config)


def reject_gjd_checkpoint_specs(
    checkpoint_specs: list["CheckpointSpec"],
    *,
    source: str = "run_libero_sampled_eval.py",
) -> None:
    offenders = [f"{spec.key} ({spec.config})" for spec in checkpoint_specs if is_gjd_config_path(spec.config)]
    if not offenders:
        return
    offender_text = ", ".join(offenders)
    raise ValueError(
        f"{source} does not implement the current GJD rollout contract for: {offender_text}. "
        "Use scripts/run_gjd_libero.sh rollout --method <m1|m5> --ablation "
        "<vanilla|pure_joint|mode_token>, or a GJD-aware batch wrapper that delegates to it. "
        "Generic sampled eval may silently change frontend, startup, inference-window, ablation, "
        "or config-override semantics for GJD."
    )


@dataclass(frozen=True)
class DatasetEpisode:
    dataset_episode_index: int
    task_text: str
    task_index: int | None
    task_id: int
    task_name: str | None
    episode_idx: int
    length: int
    replay_status: str | None = None
    episode_id: int | None = None
    init_id: int | None = None
    resolved_init_state_index: int | None = None
    init_id_source: str = "task_local_rank"

    def __post_init__(self) -> None:
        if self.episode_id is None:
            object.__setattr__(self, "episode_id", int(self.dataset_episode_index))
        if self.init_id is None:
            object.__setattr__(self, "init_id", int(self.episode_idx))


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
    episode_id: int
    init_id: int
    episode_idx: int
    replay_status: str | None
    seed: int
    output_dir: str
    suffix: str
    summary_glob: str
    command_template: list[str]
    preflight_problem: str | None = None
    resolved_init_state_index: int | None = None
    init_id_source: str = "task_local_rank"


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="m1",
        label="M1 exact",
        config=DEFAULT_CONFIG,
        reference_assets_device_policy="runtime",
        async_low_watermark=8,
        extra_args=("--merge-checkpoint-runtime-config",),
    ),
    MethodSpec(
        key="m2",
        label="M2 joint denoise",
        config="configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml",
        reference_assets_device_policy="runtime",
        async_low_watermark=12,
        extra_args=("--merge-checkpoint-runtime-config",),
    ),
    MethodSpec(
        key="m5",
        label="M5 action-only MoT",
        config="configs/evals/mot_libero_full_segment_non_joint_action_only_eval.yaml",
        reference_assets_device_policy="cpu_offload",
        async_low_watermark=16,
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
            "Sample LIBERO dataset episodes in an init-major task axis by default, then run local "
            "realtime rollout comparisons for one or more method/checkpoint targets."
        )
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-label", type=str, default="sampled_eval")
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of sampled dataset episodes. Ignored by --sample-mode full.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        choices=SAMPLE_MODE_CHOICES,
        default="task_episode_axis",
        help=(
            "`dataset_distribution` samples LeRobot episodes proportionally by dataset task distribution. "
            "`uniform_task_distribution` is a legacy alias for `dataset_distribution`. "
            "`task_episode_axis` selects explicit upstream LIBERO task ids and per-task episode indices, "
            "or, when episode indices are omitted, an init-major balanced task sweep from eligible episodes. "
            "`full` enumerates every upstream LIBERO init state for every selected task before replay-status "
            "policy validation."
        ),
    )
    parser.add_argument(
        "--replay-status-path",
        type=Path,
        default=None,
        help=(
            "Optional replay-status JSONL path. Relative paths are resolved under --dataset-root. "
            "Defaults to meta/replay_status.jsonl when that file exists."
        ),
    )
    parser.add_argument(
        "--replay-status-policy",
        choices=tuple(sorted(REPLAY_STATUS_POLICIES)),
        default="successful_only",
        help=(
            "Dataset episode filter applied before dataset-distribution sampling and before implicit "
            "task/init-axis sampling. Explicit task/init-axis selections are validated after selection. "
            "The default samples only successful demos when replay labels exist."
        ),
    )
    parser.add_argument(
        "--require-replay-status",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail if replay-status metadata is missing. The default keeps older unlabeled datasets runnable "
            "while still filtering automatically when replay_status.jsonl is present."
        ),
    )
    parser.add_argument(
        "--distribution-episode-strategy",
        choices=("first", "random", "evenly_spaced"),
        default="first",
        help=(
            "How --sample-mode dataset_distribution chooses task-local episode/init-state indices after "
            "proportional task allocation. `first` selects episode_idx 0..k-1 per task and is the default "
            "because it generalizes the #77 task-axis parity protocol. `random` preserves the old "
            "seed-based behavior. `evenly_spaced` covers the available task-local episode range."
        ),
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help=(
            "Task selector for --sample-mode task_episode_axis or full. Supports comma-separated ids and "
            "Python-style half-open ranges, e.g. `0`, `0,2,5`, or `0:10`. Defaults to all task ids present "
            "in dataset metadata for task_episode_axis and all benchmark tasks for full."
        ),
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help=(
            "Per-task episode selector for --sample-mode task_episode_axis. Supports comma-separated ids "
            "and Python-style half-open ranges. When omitted, task_episode_axis selects exactly "
            "--num-episodes in init-major order across the selected tasks. Not used by --sample-mode full."
        ),
    )
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
        "--task-axis-init-source",
        choices=("auto", "replay_status", "task_local"),
        default="auto",
        help=(
            "Controls which init-state index is passed to LIBERO rollouts after replay-status metadata is attached. "
            "`auto` preserves the historical behavior of using replay_status.resolved_init_state_index outside full "
            "grid evals. `task_local` keeps the explicit task-local episode/init index, which is required for "
            "apple-to-apple comparisons against upstream LIBERO runners."
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
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument(
        "--worker-start-stagger-seconds",
        type=float,
        default=0.0,
        help="Delay worker startup by N seconds per worker index to avoid simultaneous MuJoCo initialization.",
    )
    parser.add_argument("--execute", action="store_true", help="Run generated cases locally.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-missing", action="store_true")
    parser.add_argument(
        "--eval-profile",
        choices=("debug_short", "libero_10hz_full"),
        default="libero_10hz_full",
        help="Rollout-default profile passed to scripts/run_libero_realtime_sandbox.py.",
    )
    parser.add_argument("--max-actions", type=int, default=None, help="Optional override for --eval-profile.")
    parser.add_argument("--env-horizon", type=int, default=None, help="Optional override for --eval-profile.")
    parser.add_argument("--target-action-hz", type=float, default=None, help="Optional override for --eval-profile.")
    parser.add_argument("--video-fps", type=int, default=None, help="Optional override for --eval-profile.")
    parser.add_argument(
        "--rollout-artifact-profile",
        choices=("lean", "standard", "debug"),
        default="lean",
        help=(
            "Artifact profile passed to each realtime rollout. `lean` writes only per-rollout summary JSONs "
            "and sampled-eval logs/results; use `standard` or `debug` when you need MP4s and trace files."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed passed to run_libero_realtime_sandbox.py.")
    parser.add_argument("--deadline-miss-policy", type=str, default=None, help="Optional override for --eval-profile.")
    parser.add_argument(
        "--pretrained-model-root",
        type=str,
        default=None,
        help=(
            "Forward a VAE/text/tokenizer reference asset root to child realtime rollouts. "
            "Use this for transformer-only external checkpoints such as the released LingBot-VA model root."
        ),
    )
    parser.add_argument(
        "--reference-assets-device-policy",
        choices=("cpu_offload", "runtime"),
        default=None,
        help="Optional reference-asset placement override. Omit to use the method default.",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default=None,
        help="MuJoCo GL backend for child rollouts. Defaults to MUJOCO_GL from the shell, then egl.",
    )
    parser.add_argument(
        "--clear-ld-library-path",
        action="store_true",
        help=(
            "Remove LD_LIBRARY_PATH from rollout child processes. Disabled by default because LIBERO / MuJoCo "
            "parity runs may depend on the shell-provided graphics/runtime library path."
        ),
    )
    parser.add_argument("--write-fallback-timeline-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Forward the historical-config opt-in to child realtime rollouts. Without this, deprecated "
            "LIBERO M1/M5 configs fail before rollout."
        ),
    )
    parser.add_argument("--collect", type=Path, default=None, help="Collect an existing run directory and exit.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-device-sigaborts",
        type=int,
        default=2,
        help=(
            "Stop scheduling new cases on a worker device after this many consecutive SIGABRT exits. "
            "This keeps shared Slurm eval jobs moving when one EGL device is unstable."
        ),
    )
    parser.add_argument(
        "--case-claim-stale-seconds",
        type=float,
        default=6 * 60 * 60,
        help=(
            "Seconds after which an in-progress shared eval case claim may be stolen. "
            "This lets overlapping Slurm jobs recover from preemption without duplicate live rollouts."
        ),
    )
    args = parser.parse_args()
    args.sample_mode = normalize_sample_mode(args.sample_mode)

    if args.collect is not None:
        summary = collect_run(args.collect)
        print(json.dumps(summary, indent=2))
        raise SystemExit(0)

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    for positive_arg in ("max_actions", "env_horizon", "target_action_hz", "video_fps"):
        value = getattr(args, positive_arg)
        if value is not None and value <= 0:
            raise ValueError(f"--{positive_arg.replace('_', '-')} must be positive when provided.")

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
    reject_gjd_checkpoint_specs(checkpoint_specs)

    sample_count_label = "full" if args.sample_mode == "full" else f"n{args.num_episodes}"
    run_id = args.run_id or (
        f"{sanitize_label(args.run_label)}_{sanitize_label(args.benchmark)}_{sample_count_label}_"
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
    replay_status_report: ReplayStatusFilterReport | None = None
    replay_status_path: Path | None = None
    full_init_counts_by_task_id: dict[int, int] = {}
    if not args.dataset_root.is_dir():
        dataset_preflight_problem = f"dataset root does not exist: {args.dataset_root}"
    else:
        metadata = load_lerobot_metadata(args.dataset_root)
        replay_status_records, replay_status_path = load_replay_status_records(
            args.dataset_root,
            replay_status_path=args.replay_status_path,
            require=args.require_replay_status,
        )
        task_id_map, task_name_map, task_resolution_warnings = resolve_task_ids(
            metadata["task_text_to_index"],
            benchmark=args.benchmark,
            mode=args.task_id_source,
            libero_repo_root=args.libero_repo_root,
            local_paths=args.local_paths,
        )
        dataset_episodes = build_dataset_episodes(
            metadata["episodes"],
            task_text_to_index=metadata["task_text_to_index"],
            task_text_to_task_id=task_id_map,
            task_text_to_task_name=task_name_map,
        )
        dataset_episodes = attach_replay_status_to_dataset_episodes(
            dataset_episodes,
            replay_status_records,
            use_resolved_init_ids=use_replay_resolved_init_ids(args),
        )
        if args.sample_mode == "full":
            full_init_counts_by_task_id = resolve_libero_init_counts(
                benchmark=args.benchmark,
                task_ids=parse_int_selector(args.task_ids) if args.task_ids is not None else None,
                libero_repo_root=args.libero_repo_root,
                local_paths=args.local_paths,
            )
        if args.sample_mode == "dataset_distribution" or (
            args.sample_mode == "task_episode_axis" and args.episode_indices is None
        ):
            dataset_episodes, replay_status_report = filter_dataset_episodes_by_replay_status(
                dataset_episodes,
                policy=args.replay_status_policy,
                replay_status_records=replay_status_records,
                require_replay_status=args.require_replay_status,
                source_path=replay_status_path,
            )
        sampled_episodes, sample_allocations = select_sampled_episodes(
            dataset_episodes,
            mode=args.sample_mode,
            count=args.num_episodes,
            seed=args.sample_seed,
            task_ids=args.task_ids,
            episode_indices=args.episode_indices,
            distribution_episode_strategy=args.distribution_episode_strategy,
            full_init_counts_by_task_id=full_init_counts_by_task_id,
        )
        if args.sample_mode == "full" or (
            args.sample_mode == "task_episode_axis" and args.episode_indices is not None
        ):
            sampled_episodes, replay_status_report = filter_dataset_episodes_by_replay_status(
                sampled_episodes,
                policy=args.replay_status_policy,
                replay_status_records=replay_status_records,
                require_replay_status=args.require_replay_status,
                source_path=replay_status_path,
                task_axis_validation=True,
            )
    sample_warnings = build_sample_warnings(
        mode=args.sample_mode,
        requested_count=args.num_episodes,
        task_allocations=sample_allocations,
        distribution_episode_strategy=args.distribution_episode_strategy,
    )
    sample_warnings.extend(build_replay_status_warnings(replay_status_report))

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
    atomic_write_json(cases_path, [asdict(case) for case in cases])
    atomic_write_json(
        sample_manifest_path,
        {
            "dataset_root": str(args.dataset_root),
            "num_episodes": args.num_episodes,
            "sample_mode": args.sample_mode,
            "sample_seed": args.sample_seed,
            "distribution_episode_strategy": args.distribution_episode_strategy,
            "eval_profile": args.eval_profile,
            "rollout_artifact_profile": args.rollout_artifact_profile,
            "task_ids": args.task_ids,
            "episode_indices": args.episode_indices,
            "task_axis_init_source": args.task_axis_init_source,
            "full_init_counts_by_task_id": {
                str(task_id): count for task_id, count in sorted(full_init_counts_by_task_id.items())
            },
            "replay_status_path": str(replay_status_path) if replay_status_path is not None else None,
            "replay_status_policy": args.replay_status_policy,
            "require_replay_status": args.require_replay_status,
            "replay_status_filter": replay_status_report.to_dict() if replay_status_report is not None else None,
            "task_allocations": sample_allocations,
            "sample_warnings": sample_warnings,
            "episodes": [asdict(episode) for episode in sampled_episodes],
        },
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
        "num_sampled_episodes": len(sampled_episodes),
        "requested_num_episodes": args.num_episodes,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "distribution_episode_strategy": args.distribution_episode_strategy,
        "eval_profile": args.eval_profile,
        "rollout_artifact_profile": args.rollout_artifact_profile,
        "task_ids": args.task_ids,
        "episode_indices": args.episode_indices,
        "task_axis_init_source": args.task_axis_init_source,
        "full_init_counts_by_task_id": {
            str(task_id): count for task_id, count in sorted(full_init_counts_by_task_id.items())
        },
        "replay_status_path": str(replay_status_path) if replay_status_path is not None else None,
        "replay_status_policy": args.replay_status_policy,
        "require_replay_status": args.require_replay_status,
        "replay_status_filter": replay_status_report.to_dict() if replay_status_report is not None else None,
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
        "sample_warnings": sample_warnings,
        "missing": missing,
        "execute": bool(args.execute),
    }
    atomic_write_json(manifest_path, manifest)
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
        uses_transformer_only_input = resolution.checkpoint_file is None and resolution.runtime_transformer_dir is not None
        extra_args = extra_args_for_transformer_only_input(method.extra_args) if uses_transformer_only_input else method.extra_args
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
                extra_args=extra_args,
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


def _replay_resolved_init_state_index(record: Any) -> int | None:
    raw = getattr(record, "raw", None)
    if isinstance(raw, Mapping):
        value = raw.get("resolved_init_state_index")
        if value is not None:
            return int(value)
    value = getattr(record, "resolved_init_state_index", None)
    return None if value is None else int(value)


def attach_replay_status_to_dataset_episodes(
    episodes: list[DatasetEpisode],
    replay_status_records: dict[int, Any],
    *,
    use_resolved_init_ids: bool = False,
) -> list[DatasetEpisode]:
    if not replay_status_records:
        return list(episodes)
    attached: list[DatasetEpisode] = []
    for episode in episodes:
        record = replay_status_records.get(episode.dataset_episode_index)
        if record is None:
            attached.append(replace(episode, replay_status=None))
            continue
        resolved_init_state_index = _replay_resolved_init_state_index(record)
        updates: dict[str, Any] = {
            "replay_status": record.replay_status,
            "resolved_init_state_index": resolved_init_state_index,
        }
        if use_resolved_init_ids and resolved_init_state_index is not None:
            updates["init_id"] = resolved_init_state_index
            updates["init_id_source"] = "replay_status.resolved_init_state_index"
        attached.append(replace(episode, **updates))
    return attached


def use_replay_resolved_init_ids(args: argparse.Namespace) -> bool:
    source = getattr(args, "task_axis_init_source", "auto")
    if source == "task_local":
        return False
    if source == "replay_status":
        return True
    if source != "auto":
        raise ValueError(f"Unsupported task_axis_init_source={source!r}.")
    return args.sample_mode != "full"


def filter_dataset_episodes_by_replay_status(
    episodes: list[DatasetEpisode],
    *,
    policy: str,
    replay_status_records: dict[int, Any],
    require_replay_status: bool,
    source_path: Path | None,
    task_axis_validation: bool = False,
) -> tuple[list[DatasetEpisode], ReplayStatusFilterReport]:
    normalized_policy = normalize_replay_status_policy(policy)
    selected_indices = [episode.dataset_episode_index for episode in episodes]
    kept_indices, report = filter_episode_indices_by_replay_status(
        selected_indices,
        replay_status_records=replay_status_records,
        policy=normalized_policy,
        require_labeled=bool(replay_status_records) or bool(require_replay_status),
        source_path=source_path,
    )
    kept_index_set = set(kept_indices)
    kept_episodes = [episode for episode in episodes if episode.dataset_episode_index in kept_index_set]
    if task_axis_validation and len(kept_episodes) != len(episodes):
        failed = [
            f"task_id={episode.task_id},init_id={episode.init_id},dataset_episode_index={episode.dataset_episode_index}"
            for episode in episodes
            if episode.dataset_episode_index not in kept_index_set
        ]
        preview = ", ".join(failed[:10])
        suffix = "" if len(failed) <= 10 else f", ... ({len(failed)} filtered total)"
        raise ValueError(
            f"Requested task/init-axis episodes do not satisfy replay_status_policy={normalized_policy!r}: "
            f"{preview}{suffix}"
        )
    return kept_episodes, report


def resolve_task_ids(
    task_text_to_index: dict[str, int],
    *,
    benchmark: str,
    mode: str,
    libero_repo_root: Path | None = None,
    local_paths: Path | None = None,
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
    old_local_paths = os.environ.get("OPEN_WAM_LOCAL_PATHS")
    override_libero_repo_root = libero_repo_root is not None
    override_local_paths = local_paths is not None
    if override_libero_repo_root:
        os.environ["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    if override_local_paths:
        resolved_local_paths = local_paths if local_paths.is_absolute() else REPO_ROOT / local_paths
        os.environ["OPEN_WAM_LOCAL_PATHS"] = str(resolved_local_paths)

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
        if override_local_paths:
            if old_local_paths is None:
                os.environ.pop("OPEN_WAM_LOCAL_PATHS", None)
            else:
                os.environ["OPEN_WAM_LOCAL_PATHS"] = old_local_paths


def resolve_libero_init_counts(
    *,
    benchmark: str,
    task_ids: list[int] | None,
    libero_repo_root: Path | None = None,
    local_paths: Path | None = None,
) -> dict[int, int]:
    old_libero_repo_root = os.environ.get("LIBERO_REPO_ROOT")
    old_local_paths = os.environ.get("OPEN_WAM_LOCAL_PATHS")
    override_libero_repo_root = libero_repo_root is not None
    override_local_paths = local_paths is not None
    if override_libero_repo_root:
        os.environ["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    if override_local_paths:
        resolved_local_paths = local_paths if local_paths.is_absolute() else REPO_ROOT / local_paths
        os.environ["OPEN_WAM_LOCAL_PATHS"] = str(resolved_local_paths)

    try:
        try:
            from open_wam.integrations.libero_env import (
                LiberoTaskSpec,
                ensure_local_libero_config,
                load_libero_task_init_states,
            )
        except Exception as exc:
            raise RuntimeError("Could not import the LIBERO config bootstrap helper.") from exc

        config_path = ensure_local_libero_config(REPO_ROOT)
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            libero_config = yaml.safe_load(handle)
        from libero.libero import benchmark as libero_benchmark  # type: ignore

        benchmark_classes = libero_benchmark.get_benchmark_dict()
        try:
            benchmark_instance = benchmark_classes[benchmark]()
        except KeyError as exc:
            available = ", ".join(sorted(benchmark_classes))
            raise ValueError(f"Unknown LIBERO benchmark {benchmark!r}; available benchmarks: {available}") from exc

        get_num_tasks = getattr(benchmark_instance, "get_num_tasks", None)
        if callable(get_num_tasks):
            task_count = int(get_num_tasks())
        else:
            n_tasks = getattr(benchmark_instance, "n_tasks", None)
            task_count = int(n_tasks) if n_tasks is not None else len(benchmark_instance.tasks)
        selected_task_ids = task_ids if task_ids is not None else list(range(task_count))
        invalid_task_ids = [task_id for task_id in selected_task_ids if task_id < 0 or task_id >= task_count]
        if invalid_task_ids:
            raise ValueError(
                f"Requested task ids exceed benchmark {benchmark!r} task count {task_count}: {invalid_task_ids}"
            )

        init_counts: dict[int, int] = {}
        for task_id in selected_task_ids:
            task = benchmark_instance.get_task(int(task_id))
            task_spec = LiberoTaskSpec(
                benchmark_name=benchmark,
                task_id=int(task_id),
                task_name=task.name,
                task_language=task.language,
                problem_folder=task.problem_folder,
                bddl_file_path=benchmark_instance.get_task_bddl_file_path(int(task_id)),
                init_states_path=str(
                    Path(libero_config["init_states"]) / task.problem_folder / f"{task.name}.pruned_init"
                ),
            )
            init_counts[int(task_id)] = int(len(load_libero_task_init_states(task_spec, REPO_ROOT)))
        return init_counts
    finally:
        if override_libero_repo_root:
            if old_libero_repo_root is None:
                os.environ.pop("LIBERO_REPO_ROOT", None)
            else:
                os.environ["LIBERO_REPO_ROOT"] = old_libero_repo_root
        if override_local_paths:
            if old_local_paths is None:
                os.environ.pop("OPEN_WAM_LOCAL_PATHS", None)
            else:
                os.environ["OPEN_WAM_LOCAL_PATHS"] = old_local_paths


def select_sampled_episodes(
    episodes: list[DatasetEpisode],
    *,
    mode: str,
    count: int,
    seed: int,
    task_ids: str | None = None,
    episode_indices: str | None = None,
    distribution_episode_strategy: str = "first",
    full_init_counts_by_task_id: dict[int, int] | None = None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    if mode == "dataset_distribution":
        if task_ids is not None or episode_indices is not None:
            raise ValueError("--task-ids/--episode-indices require --sample-mode task_episode_axis or full.")
        return sample_episodes_by_task_distribution(
            episodes,
            count=count,
            seed=seed,
            episode_strategy=distribution_episode_strategy,
        )
    if mode == "task_episode_axis":
        return select_task_episode_axis(
            episodes,
            count=count,
            task_ids=task_ids,
            episode_indices=episode_indices,
        )
    if mode == "full":
        if full_init_counts_by_task_id is None:
            raise ValueError("--sample-mode full requires benchmark init-state counts.")
        return select_full_task_init_axis(
            episodes,
            init_counts_by_task_id=full_init_counts_by_task_id,
            task_ids=task_ids,
            episode_indices=episode_indices,
        )
    raise ValueError(f"Unsupported sample mode: {mode!r}")


def select_task_episode_axis(
    episodes: list[DatasetEpisode],
    *,
    count: int,
    task_ids: str | None,
    episode_indices: str | None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    selected_task_ids = (
        parse_int_selector(task_ids) if task_ids is not None else sorted({int(item.task_id) for item in episodes})
    )
    if not selected_task_ids:
        raise ValueError("--task-ids did not select any task ids.")

    if episode_indices is None:
        by_task: dict[int, list[DatasetEpisode]] = defaultdict(list)
        for episode in episodes:
            by_task[int(episode.task_id)].append(episode)
        missing_task_ids = [task_id for task_id in selected_task_ids if task_id not in by_task]
        if missing_task_ids:
            preview = ", ".join(str(task_id) for task_id in missing_task_ids[:10])
            suffix = "" if len(missing_task_ids) <= 10 else f", ... ({len(missing_task_ids)} missing total)"
            raise ValueError(f"Requested LIBERO task ids are not present in dataset metadata: {preview}{suffix}")
        for task_episodes in by_task.values():
            task_episodes.sort(key=lambda item: (item.episode_idx, item.dataset_episode_index))

        available = sum(len(by_task[task_id]) for task_id in selected_task_ids)
        if count > available:
            raise ValueError(
                f"Cannot select {count} task/init-axis episodes from only {available} eligible episodes "
                "for the selected tasks."
            )

        selected: list[DatasetEpisode] = []
        max_task_episodes = max(len(by_task[task_id]) for task_id in selected_task_ids)
        for task_local_rank in range(max_task_episodes):
            for task_id in selected_task_ids:
                task_episodes = by_task[task_id]
                if task_local_rank >= len(task_episodes):
                    continue
                selected.append(task_episodes[task_local_rank])
                if len(selected) >= count:
                    allocations: dict[str, int] = defaultdict(int)
                    for episode in selected:
                        allocations[episode.task_text] += 1
                    return selected, dict(allocations)

        raise RuntimeError("task/init-axis selection exhausted eligible episodes before reaching requested count.")

    selected_episode_indices = parse_int_selector(episode_indices)
    if not selected_episode_indices:
        raise ValueError("--episode-indices did not select any episode indices.")

    by_key = {(int(episode.task_id), int(episode.episode_idx)): episode for episode in episodes}
    selected: list[DatasetEpisode] = []
    missing: list[str] = []
    for episode_idx in selected_episode_indices:
        for task_id in selected_task_ids:
            episode = by_key.get((int(task_id), int(episode_idx)))
            if episode is None:
                missing.append(f"task_id={task_id},episode_idx={episode_idx}")
                continue
            selected.append(episode)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(f"Requested LIBERO task/episode pairs are not present in dataset metadata: {preview}{suffix}")

    allocations: dict[str, int] = defaultdict(int)
    for episode in selected:
        allocations[episode.task_text] += 1
    selected.sort(key=lambda item: (item.episode_idx, item.task_id, item.dataset_episode_index))
    return selected, dict(allocations)


def select_full_task_init_axis(
    episodes: list[DatasetEpisode],
    *,
    init_counts_by_task_id: dict[int, int],
    task_ids: str | None,
    episode_indices: str | None,
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    if episode_indices is not None:
        raise ValueError("--episode-indices is not used with --sample-mode full; full enumerates every init id.")
    if not init_counts_by_task_id:
        raise ValueError("--sample-mode full requires at least one benchmark task/init count.")
    selected_task_ids = parse_int_selector(task_ids) if task_ids is not None else sorted(init_counts_by_task_id)
    if not selected_task_ids:
        raise ValueError("--task-ids did not select any task ids.")

    for task_id in selected_task_ids:
        if task_id not in init_counts_by_task_id:
            available = ", ".join(str(item) for item in sorted(init_counts_by_task_id))
            raise ValueError(f"Task id {task_id} is not available for --sample-mode full; available task ids: {available}")
        init_count = int(init_counts_by_task_id[task_id])
        if init_count <= 0:
            raise ValueError(f"Task id {task_id} has no LIBERO init states.")

    by_key = {(int(episode.task_id), int(episode.init_id)): episode for episode in episodes}
    selected: list[DatasetEpisode] = []
    missing: list[str] = []
    max_init_count = max(int(init_counts_by_task_id[task_id]) for task_id in selected_task_ids)
    for init_id in range(max_init_count):
        for task_id in selected_task_ids:
            init_count = int(init_counts_by_task_id[task_id])
            if init_id >= init_count:
                continue
            episode = by_key.get((int(task_id), int(init_id)))
            if episode is None:
                missing.append(f"task_id={task_id},init_id={init_id}")
                continue
            selected.append(episode)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(
            "Full LIBERO task/init grid is not present in dataset metadata: "
            f"{preview}{suffix}. Use task_episode_axis for a partial grid or refresh the dataset metadata."
        )

    allocations: dict[str, int] = defaultdict(int)
    for episode in selected:
        allocations[episode.task_text] += 1
    selected.sort(key=lambda item: (item.init_id, item.task_id, item.dataset_episode_index))
    return selected, dict(allocations)


def parse_int_selector(value: str) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for raw_piece in value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        if ":" in piece:
            parts = piece.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid integer range selector {piece!r}.")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step == 0:
                raise ValueError(f"Invalid integer range selector {piece!r}: step cannot be zero.")
            values = range(start, stop, step)
        else:
            values = (int(piece),)
        for item in values:
            if item < 0:
                raise ValueError(f"Negative indices are not supported: {item}")
            if item in seen:
                continue
            seen.add(item)
            selected.append(item)
    return selected


def sample_episodes_by_task_distribution(
    episodes: list[DatasetEpisode],
    *,
    count: int,
    seed: int,
    episode_strategy: str = "first",
) -> tuple[list[DatasetEpisode], dict[str, int]]:
    if count > len(episodes):
        raise ValueError(f"Cannot sample {count} episodes without replacement from only {len(episodes)} episodes.")
    if episode_strategy not in {"first", "random", "evenly_spaced"}:
        raise ValueError(f"Unsupported dataset distribution episode strategy: {episode_strategy!r}")

    by_task: dict[str, list[DatasetEpisode]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_text].append(episode)
    for task_episodes in by_task.values():
        task_episodes.sort(key=lambda item: item.dataset_episode_index)

    task_order = sorted(by_task, key=lambda text: (by_task[text][0].task_id, text))
    allocations = allocate_proportional_counts(
        {task_text: len(task_episodes) for task_text, task_episodes in by_task.items()},
        total=count,
        tie_break_order=task_order,
    )
    rng = random.Random(seed)
    selected: list[DatasetEpisode] = []
    for task_text in task_order:
        task_count = allocations[task_text]
        if task_count <= 0:
            continue
        selected.extend(
            select_distribution_task_episodes(
                by_task[task_text],
                count=task_count,
                strategy=episode_strategy,
                rng=rng,
            )
        )
    selected.sort(key=lambda item: (item.task_id, item.episode_idx, item.dataset_episode_index))
    return selected, allocations


def select_distribution_task_episodes(
    episodes: list[DatasetEpisode],
    *,
    count: int,
    strategy: str,
    rng: random.Random,
) -> list[DatasetEpisode]:
    if count > len(episodes):
        raise ValueError(f"Cannot select {count} task episodes from only {len(episodes)} candidates.")
    if strategy == "first":
        return list(episodes[:count])
    if strategy == "random":
        return sorted(rng.sample(episodes, count), key=lambda item: item.episode_idx)
    if strategy == "evenly_spaced":
        return [episodes[index] for index in evenly_spaced_indices(len(episodes), count)]
    raise ValueError(f"Unsupported dataset distribution episode strategy: {strategy!r}")


def evenly_spaced_indices(population: int, count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count > population:
        raise ValueError("count cannot exceed population.")
    if count == 0:
        return []
    if count == 1:
        return [0]
    return sorted({round(index * (population - 1) / (count - 1)) for index in range(count)})


def allocate_proportional_counts(
    group_sizes: dict[str, int],
    *,
    total: int,
    tie_break_order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
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
    ordered_keys = tuple(tie_break_order) if tie_break_order is not None else tuple(group_sizes)
    tie_rank = {key: index for index, key in enumerate(ordered_keys)}

    def priority(key: str) -> tuple[float, int, int]:
        return (ideals[key] - math.floor(ideals[key]), usable[key], -tie_rank.get(key, len(tie_rank)))

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


def build_sample_warnings(
    *,
    mode: str,
    requested_count: int,
    task_allocations: dict[str, int],
    distribution_episode_strategy: str,
) -> list[str]:
    if mode == "full":
        if not task_allocations:
            return []
        return [
            "`--sample-mode full` ignores --num-episodes and enumerates every benchmark task/init pair "
            "before replay-status policy validation."
        ]
    if mode != "dataset_distribution" or not task_allocations:
        return []
    warnings: list[str] = []
    task_count = len(task_allocations)
    covered_task_count = sum(1 for count in task_allocations.values() if count > 0)
    if requested_count < task_count:
        warnings.append(
            f"Requested {requested_count} sampled episodes across {task_count} tasks; only "
            f"{covered_task_count} tasks are covered. Increase --num-episodes to at least {task_count} "
            "for a cross-task smoke run."
        )
    if distribution_episode_strategy == "random":
        warnings.append(
            "`--distribution-episode-strategy random` samples arbitrary task-local init states and is "
            "not directly comparable to #77-style episode_idx prefix parity runs."
        )
    return warnings


def build_replay_status_warnings(report: ReplayStatusFilterReport | None) -> list[str]:
    if report is None:
        return []
    warnings: list[str] = []
    if report.missing_status_file and report.policy != "include_all":
        warnings.append(
            f"Replay-status policy {report.policy!r} was requested, but no replay-status file was found; "
            "sampling fell back to all dataset episodes. Pass --require-replay-status to make this fatal."
        )
    if (
        not report.missing_status_file
        and report.policy != "include_all"
        and report.total_episodes
        and report.labeled_episodes == 0
    ):
        warnings.append(
            f"Replay-status policy {report.policy!r} was requested, but the replay-status file contains "
            "no labels for the selected dataset episodes; sampling fell back to all selected episodes. "
            "Pass --require-replay-status to make an empty or incomplete status file fatal."
        )
    if report.filtered_episodes:
        warnings.append(
            f"Replay-status policy {report.policy!r} filtered {report.filtered_episodes} of "
            f"{report.total_episodes} candidate dataset episodes."
        )
    return warnings


def append_optional_arg(
    command: list[str],
    flag: str,
    value: object | None,
    *,
    default: object | None = None,
) -> None:
    if value is None:
        return
    if default is not None and value == default:
        return
    command.extend([flag, str(value)])


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
            uses_transformer_dir = (
                checkpoint_spec.checkpoint_file is None
                and checkpoint_spec.runtime_transformer_dir is not None
            )
            suffix = sanitize_label(
                f"{checkpoint_spec.key}_{args.run_label}_{benchmark}_sample{sample_index:03d}_"
                f"dataset_ep{episode.dataset_episode_index:06d}_t{episode.task_id:02d}_init{episode.init_id}_"
                f"seed{seed}_{scheduler_suffix}"
            )
            command = [
                str(args.python),
                "scripts/run_libero_realtime_sandbox.py",
                "--cfg",
                checkpoint_spec.config,
                "--task-id",
                str(episode.task_id),
                "--episode-idx",
                str(episode.init_id),
                "--eval-profile",
                args.eval_profile,
                "--realtime-scheduler-profile",
                scheduler_spec.key,
                "--runtime-device",
                "{device}",
                "--artifact-profile",
                args.rollout_artifact_profile,
                "--output-dir",
                str(checkpoint_output_dir),
                "--suffix",
                suffix,
            ]
            if uses_transformer_dir:
                command.extend(["--transformer-dir", str(checkpoint_spec.runtime_transformer_dir)])
            else:
                command.extend(["--checkpoint", checkpoint_spec.checkpoint])
            append_optional_arg(command, "--benchmark", benchmark, default="libero_10")
            append_optional_arg(command, "--max-actions", args.max_actions)
            append_optional_arg(command, "--env-horizon", args.env_horizon)
            append_optional_arg(command, "--target-action-hz", args.target_action_hz)
            append_optional_arg(command, "--video-fps", args.video_fps)
            append_optional_arg(command, "--seed", seed, default=0)
            append_optional_arg(command, "--deadline-miss-policy", args.deadline_miss_policy)
            append_optional_arg(command, "--pretrained-model-root", getattr(args, "pretrained_model_root", None))
            append_optional_arg(
                command,
                "--reference-assets-device-policy",
                checkpoint_spec.reference_assets_device_policy,
                default="runtime",
            )
            command.extend(extra_args_for_case(checkpoint_spec, uses_transformer_dir=uses_transformer_dir))
            command.extend(scheduler_flags)
            if args.write_fallback_timeline_video:
                command.append("--write-fallback-timeline-video")
            if getattr(args, "allow_deprecated_libero_config", False):
                command.append("--allow-deprecated-libero-config")
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
                    episode_id=int(episode.episode_id),
                    init_id=int(episode.init_id),
                    episode_idx=episode.episode_idx,
                    replay_status=episode.replay_status,
                    seed=seed,
                    output_dir=str(checkpoint_output_dir),
                    suffix=suffix,
                    summary_glob=str(
                        checkpoint_output_dir
                        / benchmark
                        / f"{episode.task_id}_*"
                        / f"{episode.init_id}_{suffix}.json"
                    ),
                    command_template=command,
                    preflight_problem=checkpoint_spec.preflight_problem,
                    resolved_init_state_index=episode.resolved_init_state_index,
                    init_id_source=episode.init_id_source,
                )
            )
    return cases


def extra_args_for_case(checkpoint_spec: CheckpointSpec, *, uses_transformer_dir: bool) -> list[str]:
    """Return checkpoint-only runtime flags for one sampled-eval command."""

    extra_args = list(checkpoint_spec.extra_args)
    if uses_transformer_dir:
        extra_args = list(extra_args_for_transformer_only_input(tuple(extra_args)))
    return extra_args


def extra_args_for_transformer_only_input(extra_args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(arg for arg in extra_args if arg != "--merge-checkpoint-runtime-config")


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
    flags: list[str] = []
    if scheduler.key == "freeze_until_clean_chunk":
        startup_chunks = "1" if method.key == "m5" else "0"
        if startup_chunks != "0":
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
        transformer_dir, transformer_source = resolve_transformer_only_input(candidate)
        if transformer_dir is not None:
            return CheckpointResolution(
                raw=raw_path,
                checkpoint_file=None,
                checkpoint_dir=str(candidate.resolve()),
                runtime_transformer_dir=str(transformer_dir.resolve()),
                runtime_transformer_source=transformer_source,
                problem=None,
            )
        return CheckpointResolution(
            raw=raw_path,
            checkpoint_file=None,
            checkpoint_dir=None,
            runtime_transformer_dir=None,
            runtime_transformer_source=None,
            problem=(
                "could not resolve model_state.pt, full_training_state.pt, or transformer export "
                "(config.json plus diffusion_pytorch_model*.safetensors)"
            ),
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


def resolve_transformer_only_input(path: Path) -> tuple[Path | None, str | None]:
    candidate = path.expanduser().resolve()
    if is_transformer_only_input_dir(candidate):
        return candidate, "input_transformer_dir"
    nested = candidate / "transformer"
    if is_transformer_only_input_dir(nested):
        return nested.resolve(), "input_transformer_subdir"
    return None, None


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


def is_transformer_only_input_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and _has_transformer_weights(path)
    )


def _has_transformer_weights(path: Path) -> bool:
    return (
        (path / "diffusion_pytorch_model.safetensors").is_file()
        or (path / "diffusion_pytorch_model.safetensors.index.json").is_file()
        or any(path.glob("diffusion_pytorch_model-*.safetensors"))
    )


def run_cases(cases: list[EvalCase], *, args: argparse.Namespace, status_dir: Path, logs_dir: Path) -> None:
    devices = parse_devices(args.devices)
    if not devices:
        raise ValueError("--devices selected no devices.")

    work_queue: queue.Queue[EvalCase] = queue.Queue()
    for case in cases:
        work_queue.put(case)

    def worker(device: str, worker_index: int) -> None:
        stagger_seconds = float(getattr(args, "worker_start_stagger_seconds", 0.0) or 0.0)
        if worker_index and stagger_seconds > 0:
            time.sleep(worker_index * stagger_seconds)
        consecutive_sigaborts = 0
        while True:
            try:
                case = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                returncode = run_case(case, device=device, args=args, status_dir=status_dir, logs_dir=logs_dir)
                if returncode == -6:
                    consecutive_sigaborts += 1
                    if consecutive_sigaborts >= args.max_device_sigaborts:
                        print(
                            json.dumps(
                                {
                                    "event": "sampled_eval_device_retired",
                                    "device": device,
                                    "reason": "consecutive_sigaborts",
                                    "count": consecutive_sigaborts,
                                }
                            ),
                            flush=True,
                        )
                        return
                elif returncode is not None:
                    consecutive_sigaborts = 0
            finally:
                work_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(worker, device, index): device for index, device in enumerate(devices)}
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
) -> int | None:
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
        return 0
    claim_path = acquire_case_claim(case, status_dir=status_dir, stale_seconds=args.case_claim_stale_seconds)
    if claim_path is None:
        return None
    try:
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
            return 0

        command = [part.replace("{device}", device) for part in case.command_template]
        env = build_child_env(args, device=device)

        write_case_status(status_path, state="running", returncode=None, case=case, device=device, log_path=log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        completed: subprocess.CompletedProcess[Any] | None = None
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "event": "sampled_eval_case_start",
                        "case": asdict(case),
                        "device": device,
                        "child_env": child_env_report(env, clear_ld_library_path=bool(args.clear_ld_library_path)),
                    },
                    indent=2,
                )
            )
            log.write("\n")
            log.flush()
            completed = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert completed is not None
        write_case_status(
            status_path,
            state="completed" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            case=case,
            device=device,
            log_path=log_path,
        )
        return int(completed.returncode)
    finally:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass


def acquire_case_claim(case: EvalCase, *, status_dir: Path, stale_seconds: float) -> Path | None:
    claim_dir = status_dir / "claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    path = claim_dir / f"{case.index:04d}_{case.checkpoint_key}_sample{case.sample_index:03d}.lock"
    payload = {
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_index": case.index,
        "checkpoint_key": case.checkpoint_key,
        "sample_index": case.sample_index,
        "task_id": case.task_id,
        "init_id": case.init_id,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError:
        if stale_seconds > 0 and is_stale_claim(path, stale_seconds=stale_seconds):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            try:
                fd = os.open(path, flags, 0o644)
            except FileExistsError:
                return None
        else:
            return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def is_stale_claim(path: Path, *, stale_seconds: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime > stale_seconds
    except FileNotFoundError:
        return False


def build_child_env(args: argparse.Namespace, *, device: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if getattr(args, "clear_ld_library_path", False):
        env.pop("LD_LIBRARY_PATH", None)
    local_paths = args.local_paths if args.local_paths.is_absolute() else REPO_ROOT / args.local_paths
    env["OPEN_WAM_LOCAL_PATHS"] = str(local_paths)
    env["LIBERO_REPO_ROOT"] = str(args.libero_repo_root)
    mujoco_gl = args.mujoco_gl or env.get("MUJOCO_GL") or "egl"
    env["MUJOCO_GL"] = mujoco_gl
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env["PYOPENGL_PLATFORM"] = "egl" if mujoco_gl == "egl" else mujoco_gl
    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    slurm_allocated_devices = env.get("SLURM_STEP_GPUS") or env.get("SLURM_JOB_GPUS")
    egl_allocated_devices = cuda_visible_devices or slurm_allocated_devices
    egl_device_id = egl_device_id_for_runtime_device(
        device,
        allocated_devices=egl_allocated_devices,
    )
    if mujoco_gl == "egl" and egl_device_id is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
        env["EGL_DEVICE_ID"] = str(egl_device_id)
    env.setdefault("WANDB_MODE", "disabled")
    return env


def egl_device_id_for_runtime_device(device: str | None, *, allocated_devices: str | None = None) -> int | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if normalized == "cuda":
        local_index = 0
    else:
        match = re.fullmatch(r"cuda:(\d+)", normalized)
        if not match:
            return None
        local_index = int(match.group(1))
    visible_devices = parse_allocated_gpu_ids(allocated_devices)
    if visible_devices and local_index < len(visible_devices):
        return visible_devices[local_index]
    return local_index


def parse_allocated_gpu_ids(value: str | None) -> list[int]:
    if not value:
        return []
    ids: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if start <= end else -1
            ids.extend(range(start, end + step, step))
            continue
        if token.isdigit():
            ids.append(int(token))
            continue
        trailing_number = re.search(r"(\d+)$", token)
        if trailing_number:
            ids.append(int(trailing_number.group(1)))
    return ids


def child_env_report(env: dict[str, str], *, clear_ld_library_path: bool) -> dict[str, str | bool | None]:
    return {
        "clear_ld_library_path": clear_ld_library_path,
        "ld_library_path": env.get("LD_LIBRARY_PATH"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": env.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": env.get("SLURM_STEP_GPUS"),
        "mujoco_gl": env.get("MUJOCO_GL"),
        "mujoco_egl_device_id": env.get("MUJOCO_EGL_DEVICE_ID"),
        "egl_device_id": env.get("EGL_DEVICE_ID"),
        "pyopengl_platform": env.get("PYOPENGL_PLATFORM"),
        "open_wam_local_paths": env.get("OPEN_WAM_LOCAL_PATHS"),
        "libero_repo_root": env.get("LIBERO_REPO_ROOT"),
        "wandb_mode": env.get("WANDB_MODE"),
    }


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
    atomic_write_json(path, payload)


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
    atomic_write_json(log_root / "summary.json", summary_payload)
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


def write_results_csv(path: Path, summary: dict[str, Any]) -> None:
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
    tmp_path = temporary_write_path(path)
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    os.replace(tmp_path, path)


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
    headers = ["Sample", "Dataset episode", "Task", "Init id", "Replay"] + [
        target_labels.get(key, key) for key in summary["target_keys"]
    ]
    aligns = ["---:", "---:", "---:", "---:", "---"] + ["---" for _ in summary["target_keys"]]
    lines.append("| " + " | ".join(md_escape(header) for header in headers) + " |")
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
                format_result_cell(
                    row.get(f"{key}_success"),
                    row.get(f"{key}_executed_actions"),
                    row.get(f"{key}_status"),
                )
            )
        lines.append(
            "| " + " | ".join(md_escape(cell) for cell in cells) + " |"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


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
        f"Cases: `{len(cases)}`",
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
    atomic_write_text(path, "\n".join(lines) + "\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temporary_write_path(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def temporary_write_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{time.time_ns()}.tmp"
    return path.with_name(f".{path.name}{suffix}")


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
