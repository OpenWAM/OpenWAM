#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data.replay_status import (  # noqa: E402
    REPLAY_STATUS_POLICIES,
    ReplayStatusFilterReport,
    load_replay_status_records,
)
from open_wam.configs.enums import ParallelStreamVariantProfile  # noqa: E402
from open_wam.configs import load_experiment_config  # noqa: E402
import open_wam.evals.sampled_eval_reporting as sampled_eval_reporting  # noqa: E402
import open_wam.evals.sampled_eval_planning as sampled_eval_planning  # noqa: E402
import open_wam.evals.sampled_eval_sampling as sampled_eval_sampling  # noqa: E402
import open_wam.runtime.checkpoint_artifacts as checkpoint_artifacts  # noqa: E402


# Preserve the script helpers exercised by callers and tests while package code
# owns the reusable report contract and implementation.
atomic_write_json = sampled_eval_reporting.write_json_atomic
build_paired_rows = sampled_eval_reporting.build_sampled_eval_paired_rows
build_summary_payload = sampled_eval_reporting.build_sampled_eval_summary
collect_run = sampled_eval_reporting.collect_sampled_eval_run
find_summary_paths = sampled_eval_reporting.find_case_summary_paths
write_results_csv = sampled_eval_reporting.write_sampled_eval_results_csv
write_summary_md = sampled_eval_reporting.write_sampled_eval_summary_markdown
allocate_proportional_counts = sampled_eval_sampling.allocate_proportional_counts
attach_replay_status_to_dataset_episodes = (
    sampled_eval_sampling.attach_replay_status_to_dataset_episodes
)
build_dataset_episodes = sampled_eval_sampling.build_dataset_episodes
build_replay_status_warnings = sampled_eval_sampling.build_replay_status_warnings
build_sample_warnings = sampled_eval_sampling.build_sample_warnings
DatasetEpisode = sampled_eval_sampling.DatasetEpisode
evenly_spaced_indices = sampled_eval_sampling.evenly_spaced_indices
filter_dataset_episodes_by_replay_status = (
    sampled_eval_sampling.filter_dataset_episodes_by_replay_status
)
normalize_sample_mode = sampled_eval_sampling.normalize_sample_mode
parse_int_selector = sampled_eval_sampling.parse_int_selector
sample_episodes_by_task_distribution = (
    sampled_eval_sampling.sample_episodes_by_task_distribution
)
select_distribution_task_episodes = sampled_eval_sampling.select_distribution_task_episodes
select_full_task_init_axis = sampled_eval_sampling.select_full_task_init_axis
select_sampled_episodes = sampled_eval_sampling.select_sampled_episodes
select_task_episode_axis = sampled_eval_sampling.select_task_episode_axis
MethodSpec = sampled_eval_planning.SampledEvalMethodSpec
SchedulerSpec = sampled_eval_planning.SampledEvalSchedulerSpec
TargetRequest = sampled_eval_planning.SampledEvalTargetRequest
CheckpointSpec = sampled_eval_planning.SampledEvalCheckpointSpec
EvalCase = sampled_eval_planning.SampledEvalCase
METHODS = sampled_eval_planning.SAMPLED_EVAL_METHODS
SCHEDULERS = sampled_eval_planning.SAMPLED_EVAL_SCHEDULERS
parse_target_requests = sampled_eval_planning.parse_sampled_eval_target_requests
select_by_key = sampled_eval_planning.select_sampled_eval_specs_by_key
append_optional_arg = sampled_eval_planning._append_optional_arg
extra_args_for_case = sampled_eval_planning._sampled_eval_case_extra_args
extra_args_for_transformer_only_input = (
    sampled_eval_planning._extra_args_for_transformer_only_input
)
scheduler_flags_for = sampled_eval_planning.sampled_eval_scheduler_flags
scheduler_suffix_for = sampled_eval_planning.sampled_eval_scheduler_suffix
sanitize_label = sampled_eval_planning.sanitize_sampled_eval_label
CheckpointResolution = checkpoint_artifacts.CheckpointArtifactResolution
checkpoint_step = checkpoint_artifacts.checkpoint_step
find_checkpoint_file = checkpoint_artifacts.find_checkpoint_state_file
is_transformer_only_input_dir = checkpoint_artifacts.is_transformer_only_input_dir
is_usable_transformer_dir = checkpoint_artifacts.is_usable_transformer_dir
read_backbone_transformer_subdir = checkpoint_artifacts.read_backbone_transformer_subdir
read_backbone_transformer_subdir_without_yaml = (
    checkpoint_artifacts.read_backbone_transformer_subdir_without_yaml
)
resolve_checkpoint_input = checkpoint_artifacts.resolve_checkpoint_artifacts
resolve_runtime_transformer_dir = checkpoint_artifacts.resolve_runtime_transformer_dir
resolve_transformer_only_input = checkpoint_artifacts.resolve_transformer_only_input
sorted_checkpoint_dirs = checkpoint_artifacts.sorted_checkpoint_dirs
state_file_in_dir = checkpoint_artifacts.state_file_in_dir
transformer_dir_from_resolved_config = checkpoint_artifacts.transformer_dir_from_resolved_config
_has_transformer_weights = checkpoint_artifacts.has_transformer_weights

DEFAULT_CONFIG = sampled_eval_planning.SAMPLED_EVAL_DEFAULT_CONFIG
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
SAMPLE_MODE_CHOICES = sampled_eval_sampling.SAMPLE_MODE_CHOICES
GJD_CONFIG_MARKERS = ("generalist_joint_denoising",)


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
        choices=tuple(strategy.value for strategy in sampled_eval_sampling.DistributionEpisodeStrategy),
        default=sampled_eval_sampling.DistributionEpisodeStrategy.FIRST.value,
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
        choices=tuple(source.value for source in sampled_eval_sampling.TaskAxisInitSource),
        default=sampled_eval_sampling.TaskAxisInitSource.AUTO.value,
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
    sampled_eval_reporting.write_sampled_eval_queue_note(
        log_root / "status.md",
        manifest=manifest,
        case_count=len(cases),
    )

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


def resolve_checkpoint_specs(
    *,
    args: argparse.Namespace,
    selected_methods: list[MethodSpec],
    target_requests: list[TargetRequest],
) -> list[CheckpointSpec]:
    requests = target_requests or default_target_requests(args=args, selected_methods=selected_methods)
    return sampled_eval_planning.resolve_sampled_eval_checkpoint_specs(
        selected_methods=selected_methods,
        target_requests=requests,
        config_override=args.cfg,
        reference_assets_device_policy_override=args.reference_assets_device_policy,
    )


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


def use_replay_resolved_init_ids(args: argparse.Namespace) -> bool:
    """Adapt CLI arguments to the package-owned init-source contract."""

    return sampled_eval_sampling.uses_replay_resolved_init_ids(
        sample_mode=args.sample_mode,
        task_axis_init_source=getattr(args, "task_axis_init_source", "auto"),
    )


@contextmanager
def _libero_path_overrides(
    *,
    libero_repo_root: Path | None,
    local_paths: Path | None,
) -> Iterator[None]:
    """Temporarily apply command-level LIBERO discovery overrides."""

    overrides: dict[str, str] = {}
    if libero_repo_root is not None:
        overrides["LIBERO_REPO_ROOT"] = str(libero_repo_root)
    if local_paths is not None:
        resolved_local_paths = (
            local_paths if local_paths.is_absolute() else REPO_ROOT / local_paths
        )
        overrides["OPEN_WAM_LOCAL_PATHS"] = str(resolved_local_paths)
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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

    with _libero_path_overrides(
        libero_repo_root=libero_repo_root,
        local_paths=local_paths,
    ):
        try:
            from open_wam.integrations.libero_tasks import resolve_libero_task
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


def resolve_libero_init_counts(
    *,
    benchmark: str,
    task_ids: list[int] | None,
    libero_repo_root: Path | None = None,
    local_paths: Path | None = None,
) -> dict[int, int]:
    with _libero_path_overrides(
        libero_repo_root=libero_repo_root,
        local_paths=local_paths,
    ):
        try:
            from open_wam.integrations.libero_tasks import (
                load_libero_benchmark_init_state_counts,
            )
        except Exception as exc:
            raise RuntimeError("Could not import the LIBERO config bootstrap helper.") from exc
        return load_libero_benchmark_init_state_counts(
            benchmark,
            task_ids=task_ids,
            project_root=REPO_ROOT,
        )


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
    return sampled_eval_planning.build_sampled_eval_cases(
        sampled_episodes,
        checkpoint_specs=checkpoint_specs,
        output_root=output_root,
        benchmark=benchmark,
        seed=seed,
        scheduler_spec=scheduler_spec,
        options=sampled_eval_planning.SampledEvalCaseOptions(
            python=args.python,
            run_label=args.run_label,
            eval_profile=args.eval_profile,
            rollout_artifact_profile=args.rollout_artifact_profile,
            max_actions=args.max_actions,
            env_horizon=args.env_horizon,
            target_action_hz=args.target_action_hz,
            video_fps=args.video_fps,
            deadline_miss_policy=args.deadline_miss_policy,
            pretrained_model_root=getattr(args, "pretrained_model_root", None),
            write_fallback_timeline_video=args.write_fallback_timeline_video,
            allow_deprecated_libero_config=getattr(
                args,
                "allow_deprecated_libero_config",
                False,
            ),
        ),
    )


def preflight(
    *,
    args: argparse.Namespace,
    checkpoint_specs: list[CheckpointSpec],
    dataset_problem: str | None,
) -> list[dict[str, str]]:
    return sampled_eval_planning.preflight_sampled_eval_cases(
        options=sampled_eval_planning.SampledEvalPreflightOptions(
            repo_root=REPO_ROOT,
            dataset_root=args.dataset_root,
            python=args.python,
            local_paths=args.local_paths,
            libero_repo_root=args.libero_repo_root,
        ),
        checkpoint_specs=checkpoint_specs,
        dataset_problem=dataset_problem,
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


def parse_devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
