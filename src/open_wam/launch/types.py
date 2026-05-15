from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from open_wam.configs import DatasetPreflightKind


SubmitBackend = Literal["slurm", "local"]


@dataclass(frozen=True)
class ConfigOverride:
    """One typed config override that can be rendered as a train CLI `--set`."""

    key: str
    value: Any


@dataclass(frozen=True)
class ResourceSpec:
    """Scheduler resources for one launch job."""

    name: str
    gpus: int
    cpus_per_task: int
    mem: str
    time: str


@dataclass(frozen=True)
class ClusterProfile:
    """Cluster-local launch defaults. Semantic matrix entries should not depend on this."""

    name: str
    env_script: str
    default_runs_root: Path
    default_log_root: Path
    submit_backend: SubmitBackend
    default_account: str | None
    default_partition: str | None


@dataclass(frozen=True)
class MethodProfile:
    """Method-specific launch contract and scoped default overrides."""

    name: str
    allowed_policy_types: tuple[type, ...]
    default_overrides: tuple[ConfigOverride, ...]
    launcher: str
    method_key: str


@dataclass(frozen=True)
class DatasetProfile:
    """Dataset root and latent-camera contract used for preflight checks."""

    name: str
    root: Path
    latent_cameras: tuple[str, ...]
    preflight: DatasetPreflightKind = DatasetPreflightKind.LOCAL_LATENT


@dataclass(frozen=True)
class CheckpointSpec:
    """Warm-start checkpoint/export paths for one launch spec."""

    name: str
    transformer_subdir: Path


@dataclass(frozen=True)
class EvalHook:
    """Typed eval hook metadata recorded in training manifests."""

    name: str
    checkpoint_step: int
    benchmark: str
    dataset_profile: str
    num_episodes: int
    sample_mode: str = "task_episode_axis"
    distribution_episode_strategy: str = "evenly_spaced"


@dataclass(frozen=True)
class LaunchSpec:
    """One semantic training case before cluster-specific rendering."""

    name: str
    label: str
    config_name: str
    save_root_name: str
    dataset_profile: str
    method_profile: str
    checkpoint: str
    resources: str
    overrides: tuple[ConfigOverride, ...] = ()
    eval_hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchJob:
    """One fully materialized training job."""

    spec: LaunchSpec
    index: int
    save_root: Path
    sbatch_path: Path
    command: list[str]
    env: dict[str, str]
    slurm: dict[str, str | int]
    method_key: str
    dataset: DatasetProfile
    checkpoint: CheckpointSpec
    resource: ResourceSpec
    eval_hooks: tuple[EvalHook, ...] = ()


@dataclass(frozen=True)
class LaunchMatrix:
    """Checked-in declarative launch matrix and its referenced profiles."""

    source_path: Path
    source_recipe: str
    default_wandb_project: str
    default_run_id_prefix: str
    job_name_prefix: str
    clusters: dict[str, ClusterProfile]
    resources: dict[str, ResourceSpec]
    method_profiles: dict[str, MethodProfile]
    dataset_profiles: dict[str, DatasetProfile]
    checkpoints: dict[str, CheckpointSpec]
    eval_hooks: dict[str, EvalHook]
    specs: tuple[LaunchSpec, ...]
    case_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetPreflight:
    """Dataset availability and latent-window summary for one launch spec."""

    key: str
    data_root: str
    repo_count: int
    latent_cameras: tuple[str, ...]
    latent_counts: dict[str, Any]
    missing: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchValidation:
    """Validation state for one materialized job."""

    key: str
    config_name: str
    method_profile: str
    policy_variant_type: str | None
    ok: bool
    errors: tuple[str, ...] = ()
