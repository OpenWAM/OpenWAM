"""Dependency-light request and result contracts for generic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_wam.configs import (
    DataSplit,
    EvalMode,
    EvalPredictionSource,
    read_yaml_with_local_paths,
    resolve_config_path_alias,
)


@dataclass(frozen=True)
class EvaluationRequest:
    """Resolved evaluation request after applying YAML defaults and CLI overrides."""

    experiment_config_path: Path
    mode: EvalMode
    split: DataSplit
    max_batches: int
    max_trajectories: int | None
    max_steps_per_trajectory: int | None
    batch_size: int | None
    checkpoint_path: Path | None
    device: str
    seed: int


@dataclass(frozen=True)
class EvaluationSummary:
    """Minimal structured result for CLI output and tests."""

    experiment_name: str
    mode: EvalMode
    split: DataSplit
    num_batches: int
    num_trajectories: int
    device: str
    video_num_inference_steps: int
    action_num_inference_steps: int
    joint_num_inference_steps: int | None
    guidance_scale: float
    action_guidance_scale: float
    action_prediction_source: EvalPredictionSource
    action_prediction_shape: tuple[int, ...]
    target_action_shape: tuple[int, ...]
    video_prediction_source: EvalPredictionSource
    video_prediction_shape: tuple[int, ...]
    target_video_shape: tuple[int, ...]
    mean_action_mse: float | None
    mean_trajectory_action_mse: float | None
    mean_video_latent_mse: float | None
    mean_trajectory_video_latent_mse: float | None
    checkpoint_path: str | None


def _read_yaml(path: Path) -> dict[str, Any]:
    return read_yaml_with_local_paths(path)


def _resolve_relative_path(base_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    direct_candidates = (
        base_path.parent / candidate,
        Path.cwd() / candidate,
    )
    for direct_candidate in direct_candidates:
        direct_candidate = direct_candidate.resolve()
        if direct_candidate.exists():
            return direct_candidate
    aliased_candidates = tuple(
        resolve_config_path_alias(path).resolve() for path in direct_candidates
    )
    for aliased_candidate in aliased_candidates:
        if aliased_candidate.exists():
            return aliased_candidate
    raise FileNotFoundError(
        f"Could not resolve relative path '{value}' from base '{base_path}'. "
        f"Checked: {', '.join(str(path) for path in (*direct_candidates, *aliased_candidates))}."
    )


def _coerce_optional_positive_int(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name} {value!r} in {config_path}; expected a positive integer or null."
        ) from exc
    if coerced <= 0:
        raise ValueError(
            f"Invalid {field_name} {coerced!r} in {config_path}; expected a positive integer > 0."
        )
    return coerced


def resolve_evaluation_request(
    config_path: str | Path,
    *,
    mode_override: EvalMode | str | None = None,
    split_override: DataSplit | str | None = None,
    max_batches_override: int | None = None,
    max_trajectories_override: int | None = None,
    max_steps_per_trajectory_override: int | None = None,
    batch_size_override: int | None = None,
    checkpoint_override: str | None = None,
    device_override: str | None = None,
    seed_override: int | None = None,
) -> EvaluationRequest:
    """Resolve either an experiment YAML or an eval-wrapper YAML.

    Eval wrappers are lightweight YAMLs under `configs/evals/` with an
    `experiment_config` field plus optional eval defaults such as split, device,
    checkpoint path, and batch count.
    """

    config_path = resolve_config_path_alias(config_path).resolve()
    raw = _read_yaml(config_path)
    experiment_config_path = (
        _resolve_relative_path(config_path, raw.get("experiment_config"))
        if "experiment_config" in raw
        else config_path
    )
    if experiment_config_path is None:
        raise ValueError(f"Eval config {config_path} is missing `experiment_config`.")
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else _coerce_optional_positive_int(raw.get("batch_size"), field_name="batch_size", config_path=config_path)
    )
    max_trajectories = (
        max_trajectories_override
        if max_trajectories_override is not None
        else _coerce_optional_positive_int(
            raw.get("max_trajectories"),
            field_name="max_trajectories",
            config_path=config_path,
        )
    )
    max_steps_per_trajectory = (
        max_steps_per_trajectory_override
        if max_steps_per_trajectory_override is not None
        else _coerce_optional_positive_int(
            raw.get("max_steps_per_trajectory"),
            field_name="max_steps_per_trajectory",
            config_path=config_path,
        )
    )

    return EvaluationRequest(
        experiment_config_path=experiment_config_path,
        mode=EvalMode(mode_override or raw.get("mode", "batch")),
        split=DataSplit(split_override or raw.get("split", "val")),
        max_batches=max_batches_override if max_batches_override is not None else int(raw.get("max_batches", 1)),
        max_trajectories=max_trajectories,
        max_steps_per_trajectory=max_steps_per_trajectory,
        batch_size=batch_size,
        checkpoint_path=_resolve_relative_path(config_path, checkpoint_override or raw.get("checkpoint_path")),
        device=device_override or raw.get("device", "auto"),
        seed=seed_override if seed_override is not None else int(raw.get("seed", 0)),
    )


__all__ = ["EvaluationRequest", "EvaluationSummary", "resolve_evaluation_request"]
