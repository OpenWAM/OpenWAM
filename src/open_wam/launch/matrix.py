from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import DatasetPreflightKind

from .types import (
    CheckpointSpec,
    ClusterProfile,
    ConfigOverride,
    DatasetProfile,
    EvalHook,
    LaunchMatrix,
    LaunchSpec,
    MethodProfile,
    ResourceSpec,
)


def load_launch_matrix(path: Path) -> LaunchMatrix:
    """Load a declarative launch matrix YAML into typed profiles."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    path_defaults = _resolve_path_defaults(raw.get("path_defaults", {}))
    source_recipe = str(raw.get("source_recipe", ""))
    default_wandb_project = str(raw.get("default_wandb_project", "openwam-launch"))
    default_run_id_prefix = str(raw.get("default_run_id_prefix", path.stem))
    job_name_prefix = str(raw.get("job_name_prefix", "ow"))

    clusters = {
        name: ClusterProfile(
            name=name,
            env_script=str(value["env_script"]),
            default_runs_root=Path(_resolve_string(value["default_runs_root"], path_defaults)),
            default_log_root=Path(_resolve_string(value["default_log_root"], path_defaults)),
            submit_backend=value.get("submit_backend", "slurm"),
            default_account=value.get("default_account"),
            default_partition=value.get("default_partition"),
        )
        for name, value in _mapping(raw.get("clusters")).items()
    }
    resources = {
        name: ResourceSpec(
            name=name,
            gpus=int(value["gpus"]),
            cpus_per_task=int(value["cpus_per_task"]),
            mem=str(value["mem"]),
            time=str(value["time"]),
        )
        for name, value in _mapping(raw.get("resources")).items()
    }
    method_profiles = {
        name: MethodProfile(
            name=name,
            allowed_policy_types=tuple(_import_type(item) for item in value.get("allowed_policy_types", ())),
            default_overrides=_parse_overrides(value.get("default_overrides", ()), path_defaults),
            launcher=str(value["launcher"]),
            method_key=str(value.get("method_key", name.split("_", 1)[0])),
        )
        for name, value in _mapping(raw.get("method_profiles")).items()
    }
    dataset_profiles = {
        name: DatasetProfile(
            name=name,
            root=Path(_resolve_string(value["root"], path_defaults)),
            latent_cameras=tuple(str(item) for item in value.get("latent_cameras", ())),
            preflight=DatasetPreflightKind(value.get("preflight", DatasetPreflightKind.LOCAL_LATENT)),
        )
        for name, value in _mapping(raw.get("dataset_profiles")).items()
    }
    checkpoints = {
        name: CheckpointSpec(
            name=name,
            transformer_subdir=Path(_resolve_string(value["transformer_subdir"], path_defaults)),
        )
        for name, value in _mapping(raw.get("checkpoints")).items()
    }
    eval_hooks = {
        name: EvalHook(
            name=name,
            checkpoint_step=int(value["checkpoint_step"]),
            benchmark=str(value["benchmark"]),
            dataset_profile=str(value["dataset_profile"]),
            num_episodes=int(value["num_episodes"]),
            sample_mode=str(value.get("sample_mode", "task_episode_axis")),
            distribution_episode_strategy=str(value.get("distribution_episode_strategy", "evenly_spaced")),
        )
        for name, value in _mapping(raw.get("eval_hooks")).items()
    }
    specs = tuple(
        LaunchSpec(
            name=str(value["name"]),
            label=str(value.get("label", value["name"])),
            config_name=str(value["config_name"]),
            save_root_name=str(value.get("save_root_name", value["name"])),
            dataset_profile=str(value["dataset_profile"]),
            method_profile=str(value["method_profile"]),
            checkpoint=str(value["checkpoint"]),
            resources=str(value["resources"]),
            overrides=_parse_overrides(value.get("overrides", ()), path_defaults),
            eval_hooks=tuple(str(item) for item in value.get("eval_hooks", ())),
        )
        for value in raw.get("matrix", ())
    )
    case_groups = {
        name: tuple(str(item) for item in items)
        for name, items in _mapping(raw.get("case_groups", {})).items()
    }
    if "all" not in case_groups:
        case_groups["all"] = tuple(spec.name for spec in specs)

    matrix = LaunchMatrix(
        source_path=path,
        source_recipe=source_recipe,
        default_wandb_project=default_wandb_project,
        default_run_id_prefix=default_run_id_prefix,
        job_name_prefix=job_name_prefix,
        clusters=clusters,
        resources=resources,
        method_profiles=method_profiles,
        dataset_profiles=dataset_profiles,
        checkpoints=checkpoints,
        eval_hooks=eval_hooks,
        specs=specs,
        case_groups=case_groups,
    )
    _validate_references(matrix)
    return matrix


def select_launch_specs(matrix: LaunchMatrix, raw: str) -> tuple[LaunchSpec, ...]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    by_name = {spec.name: spec for spec in matrix.specs}
    expanded: list[str] = []
    for item in requested:
        if item in matrix.case_groups:
            expanded.extend(matrix.case_groups[item])
        else:
            expanded.append(item)
    missing = [key for key in expanded if key not in by_name]
    if missing:
        valid = [*matrix.case_groups, *by_name]
        raise ValueError(f"Unknown launch case(s): {', '.join(missing)}. Valid keys/groups: {', '.join(valid)}")
    selected: list[LaunchSpec] = []
    seen: set[str] = set()
    for name in expanded:
        if name in seen:
            continue
        selected.append(by_name[name])
        seen.add(name)
    return tuple(selected)


def _parse_overrides(raw: Any, path_defaults: dict[str, str]) -> tuple[ConfigOverride, ...]:
    return tuple(
        ConfigOverride(key=str(item["key"]), value=_resolve_value(item.get("value"), path_defaults))
        for item in raw or ()
    )


def _resolve_path_defaults(raw: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        env_name = value.get("env") if isinstance(value, dict) else None
        default = value.get("default") if isinstance(value, dict) else value
        resolved[key] = os.environ.get(str(env_name), str(default)) if env_name else str(default)
    return resolved


def _resolve_value(value: Any, path_defaults: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, path_defaults)
    if isinstance(value, list):
        return [_resolve_value(item, path_defaults) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, path_defaults) for key, item in value.items()}
    return value


def _resolve_string(value: Any, path_defaults: dict[str, str]) -> str:
    text = str(value)
    for key, replacement in path_defaults.items():
        text = text.replace("{{" + key + "}}", replacement)
    return os.path.expandvars(text)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping, got {type(value).__name__}.")
    return value


def _import_type(path: str) -> type:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Expected fully-qualified type path, got {path!r}.")
    module = import_module(module_name)
    value = getattr(module, attr)
    if not isinstance(value, type):
        raise TypeError(f"{path!r} did not resolve to a type.")
    return value


def _validate_references(matrix: LaunchMatrix) -> None:
    errors: list[str] = []
    for spec in matrix.specs:
        if spec.method_profile not in matrix.method_profiles:
            errors.append(f"{spec.name}: unknown method_profile {spec.method_profile!r}")
        if spec.dataset_profile not in matrix.dataset_profiles:
            errors.append(f"{spec.name}: unknown dataset_profile {spec.dataset_profile!r}")
        if spec.checkpoint not in matrix.checkpoints:
            errors.append(f"{spec.name}: unknown checkpoint {spec.checkpoint!r}")
        if spec.resources not in matrix.resources:
            errors.append(f"{spec.name}: unknown resources {spec.resources!r}")
        for hook in spec.eval_hooks:
            if hook not in matrix.eval_hooks:
                errors.append(f"{spec.name}: unknown eval_hook {hook!r}")
    for hook in matrix.eval_hooks.values():
        if hook.dataset_profile not in matrix.dataset_profiles:
            errors.append(f"{hook.name}: unknown eval dataset_profile {hook.dataset_profile!r}")
    for group, names in matrix.case_groups.items():
        known = {spec.name for spec in matrix.specs}
        for name in names:
            if name not in known:
                errors.append(f"group {group}: unknown case {name!r}")
    if errors:
        raise ValueError("Invalid launch matrix:\n" + "\n".join(f"- {error}" for error in errors))
