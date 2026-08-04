"""Simulator-free target and command planning for sampled evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from open_wam.configs.enums import (
    DeadlineMissPolicy,
    RealtimeSchedulerProfile,
    ReferenceAssetsDevicePolicy,
    RolloutArtifactProfile,
)
from open_wam.evals.sampled_eval_sampling import DatasetEpisode
from open_wam.runtime.checkpoint_artifacts import resolve_checkpoint_artifacts

SAMPLED_EVAL_DEFAULT_CONFIG = (
    "configs/experiments/parallel_stream_libero_lingbot_exact.yaml"
)


@dataclass(frozen=True)
class SampledEvalMethodSpec:
    """One policy profile supported by the generic realtime rollout command.

    The `method_*` field names are retained in sampled-eval artifacts for
    schema compatibility. Values identify architecture/runtime profiles, not
    implementation families.
    """

    key: str
    label: str
    config: str
    reference_assets_device_policy: ReferenceAssetsDevicePolicy
    async_low_watermark: int
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_assets_device_policy",
            ReferenceAssetsDevicePolicy(self.reference_assets_device_policy),
        )


@dataclass(frozen=True)
class SampledEvalSchedulerSpec:
    """Named realtime scheduler profile and its documented effective flags."""

    key: RealtimeSchedulerProfile
    label: str
    flags: tuple[str, ...]
    use_method_low_watermark: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", RealtimeSchedulerProfile(self.key))


@dataclass(frozen=True)
class SampledEvalTargetRequest:
    """Unresolved METHOD:KEY[:LABEL]=CHECKPOINT target supplied by a caller."""

    method_key: str
    checkpoint_key: str
    label: str | None
    checkpoint: str


@dataclass(frozen=True)
class SampledEvalCheckpointSpec:
    """Resolved checkpoint artifacts and runtime policy for one target."""

    key: str
    label: str
    checkpoint: str
    method_key: str = "m1"
    method_label: str = "parallel-stream exact"
    config: str = SAMPLED_EVAL_DEFAULT_CONFIG
    checkpoint_raw: str | None = None
    checkpoint_file: str | None = None
    checkpoint_dir: str | None = None
    runtime_transformer_dir: str | None = None
    runtime_transformer_source: str | None = None
    reference_assets_device_policy: ReferenceAssetsDevicePolicy = (
        ReferenceAssetsDevicePolicy.RUNTIME
    )
    extra_args: tuple[str, ...] = ()
    preflight_problem: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_assets_device_policy",
            ReferenceAssetsDevicePolicy(self.reference_assets_device_policy),
        )


@dataclass(frozen=True)
class SampledEvalCase:
    """One fully resolved rollout command and its dataset coordinates."""

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
    scheduler_key: RealtimeSchedulerProfile
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduler_key", RealtimeSchedulerProfile(self.scheduler_key))


@dataclass(frozen=True)
class SampledEvalCaseOptions:
    """Command-level options shared by every case in one evaluation matrix."""

    python: str | Path
    run_label: str
    eval_profile: str
    rollout_artifact_profile: RolloutArtifactProfile
    max_actions: int | None = None
    env_horizon: int | None = None
    target_action_hz: float | None = None
    video_fps: float | None = None
    deadline_miss_policy: DeadlineMissPolicy | None = None
    pretrained_model_root: str | Path | None = None
    write_fallback_timeline_video: bool = False
    allow_deprecated_libero_config: bool = False
    rollout_script: str = "scripts/run_libero_realtime_sandbox.py"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_artifact_profile",
            RolloutArtifactProfile(self.rollout_artifact_profile),
        )
        if self.deadline_miss_policy is not None:
            object.__setattr__(
                self,
                "deadline_miss_policy",
                DeadlineMissPolicy(self.deadline_miss_policy),
            )


@dataclass(frozen=True)
class SampledEvalPreflightOptions:
    """Filesystem inputs needed to validate an evaluation matrix."""

    repo_root: Path
    dataset_root: Path
    python: Path
    local_paths: Path
    libero_repo_root: Path


SAMPLED_EVAL_METHODS: tuple[SampledEvalMethodSpec, ...] = (
    SampledEvalMethodSpec(
        key="m1",
        label="parallel-stream exact",
        config=SAMPLED_EVAL_DEFAULT_CONFIG,
        reference_assets_device_policy=ReferenceAssetsDevicePolicy.RUNTIME,
        async_low_watermark=8,
        extra_args=("--merge-checkpoint-runtime-config",),
    ),
    SampledEvalMethodSpec(
        key="m2",
        label="parallel-stream joint denoise",
        config=(
            "configs/experiments/"
            "parallel_stream_libero_joint_denoise.yaml"
        ),
        reference_assets_device_policy=ReferenceAssetsDevicePolicy.RUNTIME,
        async_low_watermark=12,
        extra_args=("--merge-checkpoint-runtime-config",),
    ),
    SampledEvalMethodSpec(
        key="m5",
        label="dual-expert action-only",
        config="configs/evals/dual_expert_libero_full_segment_non_joint_action_only_eval.yaml",
        reference_assets_device_policy=ReferenceAssetsDevicePolicy.CPU_OFFLOAD,
        async_low_watermark=16,
    ),
)


SAMPLED_EVAL_SCHEDULERS: tuple[SampledEvalSchedulerSpec, ...] = (
    SampledEvalSchedulerSpec(
        key=RealtimeSchedulerProfile.BLOCKING_CONTROL,
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
    SampledEvalSchedulerSpec(
        key=RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK,
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
    SampledEvalSchedulerSpec(
        key=RealtimeSchedulerProfile.ASYNC_HISTORY_FIRST,
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


class _KeyedSpec(Protocol):
    key: str


_KeyedSpecT = TypeVar("_KeyedSpecT", bound=_KeyedSpec)


def sanitize_sampled_eval_label(value: str) -> str:
    """Return a stable filesystem-safe label, falling back to ``run``."""

    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def parse_sampled_eval_target_requests(values: Sequence[str]) -> list[SampledEvalTargetRequest]:
    """Parse and validate PROFILE:KEY[:LABEL]=CHECKPOINT declarations."""

    requests: list[SampledEvalTargetRequest] = []
    seen: set[tuple[str, str]] = set()
    for raw_value in values:
        if "=" not in raw_value:
            raise ValueError(
                f"Invalid --target {raw_value!r}; expected PROFILE:KEY[:LABEL]=CHECKPOINT."
            )
        raw_selector, checkpoint = raw_value.split("=", 1)
        pieces = [piece.strip() for piece in raw_selector.split(":", 2)]
        if len(pieces) < 2 or not pieces[0] or not pieces[1] or not checkpoint.strip():
            raise ValueError(
                f"Invalid --target {raw_value!r}; expected PROFILE:KEY[:LABEL]=CHECKPOINT."
            )
        method_key = pieces[0].lower()
        checkpoint_key = sanitize_sampled_eval_label(pieces[1])
        label = pieces[2].strip() if len(pieces) == 3 and pieces[2].strip() else None
        duplicate_key = (method_key, checkpoint_key)
        if duplicate_key in seen:
            raise ValueError(f"Duplicate --target for {method_key}:{checkpoint_key}.")
        seen.add(duplicate_key)
        requests.append(
            SampledEvalTargetRequest(
                method_key=method_key,
                checkpoint_key=checkpoint_key,
                label=label,
                checkpoint=checkpoint.strip(),
            )
        )
    return requests


def select_sampled_eval_specs_by_key(
    items: Sequence[_KeyedSpecT],
    selector: str,
    *,
    field_name: str,
) -> list[_KeyedSpecT]:
    """Select registry entries in caller order while removing duplicates."""

    by_key = {item.key: item for item in items}
    selected: list[_KeyedSpecT] = []
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


def resolve_sampled_eval_checkpoint_specs(
    *,
    selected_methods: Sequence[SampledEvalMethodSpec],
    target_requests: Sequence[SampledEvalTargetRequest],
    config_override: str | None = None,
    reference_assets_device_policy_override: ReferenceAssetsDevicePolicy | str | None = None,
    methods: Sequence[SampledEvalMethodSpec] = SAMPLED_EVAL_METHODS,
) -> list[SampledEvalCheckpointSpec]:
    """Resolve target artifacts and effective policy-profile settings."""

    method_by_key = {method.key: method for method in methods}
    selected_method_keys = {method.key for method in selected_methods}
    specs: list[SampledEvalCheckpointSpec] = []
    seen_keys: set[str] = set()
    for request in target_requests:
        if request.method_key not in method_by_key:
            valid = ", ".join(sorted(method_by_key))
            raise ValueError(
                f"Unknown target method {request.method_key!r}; expected one of: {valid}."
            )
        if request.method_key not in selected_method_keys:
            raise ValueError(
                f"Target method {request.method_key!r} is not included in --methods "
                f"({', '.join(sorted(selected_method_keys))})."
            )
        method = method_by_key[request.method_key]
        key = sanitize_sampled_eval_label(f"{method.key}_{request.checkpoint_key}")
        if key in seen_keys:
            raise ValueError(f"Duplicate resolved target key {key!r}.")
        seen_keys.add(key)
        resolution = resolve_checkpoint_artifacts(request.checkpoint)
        config = config_override or method.config
        reference_policy = (
            reference_assets_device_policy_override
            or method.reference_assets_device_policy
        )
        label = request.label or f"{method.label} {request.checkpoint_key}"
        uses_transformer_only_input = (
            resolution.checkpoint_file is None
            and resolution.runtime_transformer_dir is not None
        )
        extra_args = (
            _extra_args_for_transformer_only_input(method.extra_args)
            if uses_transformer_only_input
            else method.extra_args
        )
        specs.append(
            SampledEvalCheckpointSpec(
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


def build_sampled_eval_cases(
    sampled_episodes: Sequence[DatasetEpisode],
    *,
    checkpoint_specs: Sequence[SampledEvalCheckpointSpec],
    output_root: Path,
    benchmark: str,
    seed: int,
    scheduler_spec: SampledEvalSchedulerSpec,
    options: SampledEvalCaseOptions,
    methods: Sequence[SampledEvalMethodSpec] = SAMPLED_EVAL_METHODS,
) -> list[SampledEvalCase]:
    """Construct deterministic rollout cases without importing a simulator."""

    cases: list[SampledEvalCase] = []
    method_by_key = {method.key: method for method in methods}
    for sample_index, episode in enumerate(sampled_episodes):
        for checkpoint_spec in checkpoint_specs:
            method = method_by_key[checkpoint_spec.method_key]
            scheduler_flags = sampled_eval_scheduler_flags(method, scheduler_spec)
            scheduler_suffix = sampled_eval_scheduler_suffix(method, scheduler_spec)
            checkpoint_output_dir = output_root / checkpoint_spec.key
            uses_transformer_dir = (
                checkpoint_spec.checkpoint_file is None
                and checkpoint_spec.runtime_transformer_dir is not None
            )
            suffix = sanitize_sampled_eval_label(
                f"{checkpoint_spec.key}_{options.run_label}_{benchmark}_sample{sample_index:03d}_"
                f"dataset_ep{episode.dataset_episode_index:06d}_t{episode.task_id:02d}_"
                f"init{episode.init_id}_seed{seed}_{scheduler_suffix}"
            )
            command = [
                str(options.python),
                options.rollout_script,
                "--cfg",
                checkpoint_spec.config,
                "--task-id",
                str(episode.task_id),
                "--episode-idx",
                str(episode.init_id),
                "--eval-profile",
                options.eval_profile,
                "--realtime-scheduler-profile",
                str(scheduler_spec.key),
                "--runtime-device",
                "{device}",
                "--artifact-profile",
                str(options.rollout_artifact_profile),
                "--output-dir",
                str(checkpoint_output_dir),
                "--suffix",
                suffix,
            ]
            if uses_transformer_dir:
                command.extend(["--transformer-dir", str(checkpoint_spec.runtime_transformer_dir)])
            else:
                command.extend(["--checkpoint", checkpoint_spec.checkpoint])
            _append_optional_arg(command, "--benchmark", benchmark, default="libero_10")
            _append_optional_arg(command, "--max-actions", options.max_actions)
            _append_optional_arg(command, "--env-horizon", options.env_horizon)
            _append_optional_arg(
                command,
                "--target-action-hz",
                options.target_action_hz,
            )
            _append_optional_arg(command, "--video-fps", options.video_fps)
            _append_optional_arg(command, "--seed", seed, default=0)
            _append_optional_arg(
                command,
                "--deadline-miss-policy",
                options.deadline_miss_policy,
            )
            _append_optional_arg(
                command,
                "--pretrained-model-root",
                options.pretrained_model_root,
            )
            _append_optional_arg(
                command,
                "--reference-assets-device-policy",
                checkpoint_spec.reference_assets_device_policy,
                default=ReferenceAssetsDevicePolicy.RUNTIME,
            )
            command.extend(
                _sampled_eval_case_extra_args(
                    checkpoint_spec,
                    uses_transformer_dir=uses_transformer_dir,
                )
            )
            command.extend(scheduler_flags)
            if options.write_fallback_timeline_video:
                command.append("--write-fallback-timeline-video")
            if options.allow_deprecated_libero_config:
                command.append("--allow-deprecated-libero-config")
            cases.append(
                SampledEvalCase(
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


def preflight_sampled_eval_cases(
    *,
    options: SampledEvalPreflightOptions,
    checkpoint_specs: Sequence[SampledEvalCheckpointSpec],
    dataset_problem: str | None,
) -> list[dict[str, str]]:
    """Report missing local inputs without importing benchmark dependencies."""

    missing: list[dict[str, str]] = []
    if dataset_problem is not None:
        missing.append(
            {
                "kind": "dataset",
                "path": str(options.dataset_root),
                "reason": dataset_problem,
            }
        )
    if not options.python.is_file():
        missing.append(
            {
                "kind": "python",
                "path": str(options.python),
                "reason": "python executable is missing",
            }
        )
    local_paths = (
        options.local_paths
        if options.local_paths.is_absolute()
        else options.repo_root / options.local_paths
    )
    if not local_paths.is_file():
        missing.append(
            {
                "kind": "local_paths",
                "path": str(local_paths),
                "reason": "local paths file is missing",
            }
        )
    if not options.libero_repo_root.is_dir():
        missing.append(
            {
                "kind": "libero_repo_root",
                "path": str(options.libero_repo_root),
                "reason": "LIBERO repo root is missing",
            }
        )

    seen_checkpoints: set[str] = set()
    seen_configs: set[str] = set()
    for spec in checkpoint_specs:
        config_path = Path(spec.config)
        if not config_path.is_absolute():
            config_path = options.repo_root / config_path
        config_key = str(config_path.resolve())
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            if not config_path.is_file():
                missing.append(
                    {
                        "kind": "config",
                        "path": str(config_path),
                        "reason": "config file is missing",
                    }
                )

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


def _append_optional_arg(
    command: list[str],
    flag: str,
    value: object | None,
    *,
    default: object | None = None,
) -> None:
    """Append a value flag unless it is absent or matches its implicit default."""

    if value is None:
        return
    if default is not None and value == default:
        return
    command.extend([flag, str(value)])


def _sampled_eval_case_extra_args(
    checkpoint_spec: SampledEvalCheckpointSpec,
    *,
    uses_transformer_dir: bool,
) -> list[str]:
    """Return checkpoint-only runtime flags for one sampled-eval command."""

    extra_args = list(checkpoint_spec.extra_args)
    if uses_transformer_dir:
        extra_args = list(_extra_args_for_transformer_only_input(tuple(extra_args)))
    return extra_args


def _extra_args_for_transformer_only_input(
    extra_args: tuple[str, ...],
) -> tuple[str, ...]:
    """Drop runtime-config merging when no training checkpoint is loaded."""

    return tuple(arg for arg in extra_args if arg != "--merge-checkpoint-runtime-config")


def sampled_eval_scheduler_flags(
    method: SampledEvalMethodSpec,
    scheduler: SampledEvalSchedulerSpec,
) -> list[str]:
    """Return policy-profile overrides on top of a named scheduler profile."""

    flags: list[str] = []
    if scheduler.key is RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK:
        startup_chunks = "1" if method.key == "m5" else "0"
        if startup_chunks != "0":
            flags.extend(["--startup-open-loop-chunks", startup_chunks])
    if scheduler.use_method_low_watermark:
        flags.extend(["--replan-low-watermark-actions", str(method.async_low_watermark)])
    return flags


def sampled_eval_scheduler_suffix(
    method: SampledEvalMethodSpec,
    scheduler: SampledEvalSchedulerSpec,
) -> str:
    """Return the stable scheduler fragment used in case artifact names."""

    if scheduler.use_method_low_watermark:
        return f"async_k{method.async_low_watermark}_startup1"
    if (
        method.key == "m5"
        and scheduler.key is RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK
    ):
        return "freeze_until_clean_chunk_startup1"
    return str(scheduler.key)


__all__ = [
    "SAMPLED_EVAL_DEFAULT_CONFIG",
    "SAMPLED_EVAL_METHODS",
    "SAMPLED_EVAL_SCHEDULERS",
    "SampledEvalCase",
    "SampledEvalCaseOptions",
    "SampledEvalCheckpointSpec",
    "SampledEvalMethodSpec",
    "SampledEvalPreflightOptions",
    "SampledEvalSchedulerSpec",
    "SampledEvalTargetRequest",
    "build_sampled_eval_cases",
    "parse_sampled_eval_target_requests",
    "preflight_sampled_eval_cases",
    "resolve_sampled_eval_checkpoint_specs",
    "sampled_eval_scheduler_flags",
    "sampled_eval_scheduler_suffix",
    "sanitize_sampled_eval_label",
    "select_sampled_eval_specs_by_key",
]
