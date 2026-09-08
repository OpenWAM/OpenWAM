from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import ExperimentConfig, load_experiment_config
from open_wam.runtime.checkpoint_artifacts import (
    CHECKPOINT_FILENAMES,
    CheckpointOperation,
    CheckpointSearchLayout,
    find_checkpoint_state_file,
    is_usable_transformer_dir,
)

_PRESERVED_BASE_DATA_FIELDS = frozenset(
    {
        "dataset_name",
        "dataset_type",
        "repo_id",
        "local_root",
        "empty_text_embedding_path",
        "latent_root",
        "latent_subdir",
        "split",
        "cache_dir",
        "episode_cache_size",
        "train_fraction",
        "split_seed",
        "max_train_episodes",
        "max_val_episodes",
        "train_batch_size",
        "val_batch_size",
        "num_workers",
    }
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Result of loading model weights into a runtime pipeline."""

    checkpoint_path: Path
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...] = ()


class CheckpointCompatibilityPolicy(StrEnum):
    """How runtime loading handles model/checkpoint key mismatches."""

    STRICT = "strict"
    ALLOW_CHECKPOINT_SUPERSET = "allow_checkpoint_superset"
    ALLOW_PARTIAL = "allow_partial"


class CheckpointCompatibilityError(RuntimeError):
    """Raised when strict runtime loading finds incompatible state keys."""

    def __init__(self, report: CheckpointLoadReport) -> None:
        self.report = report
        details: list[str] = []
        if report.missing_keys:
            details.append(
                f"missing {len(report.missing_keys)} keys: "
                f"{_preview_keys(report.missing_keys)}"
            )
        if report.unexpected_keys:
            details.append(
                f"unexpected {len(report.unexpected_keys)} keys: "
                f"{_preview_keys(report.unexpected_keys)}"
            )
        if report.shape_mismatches:
            details.append(
                f"incompatible shapes for {len(report.shape_mismatches)} keys: "
                f"{_preview_keys(report.shape_mismatches)}"
            )
        super().__init__(
            f"Checkpoint {report.checkpoint_path} is incompatible with the runtime "
            f"pipeline ({'; '.join(details)}). Use the matching resolved config, or "
            "select `allow_partial` only for an intentional migration diagnostic."
        )


@runtime_checkable
class _CheckpointLoadLifecycle(Protocol):
    def on_checkpoint_loaded(
        self,
        *,
        loaded_state_keys: frozenset[str],
        missing_state_keys: frozenset[str],
    ) -> None: ...


def notify_checkpoint_loaded(
    model: nn.Module,
    *,
    loaded_state_keys: frozenset[str],
    missing_state_keys: frozenset[str],
) -> None:
    """Notify models that maintain state derived from checkpoint coverage."""

    if isinstance(model, _CheckpointLoadLifecycle):
        model.on_checkpoint_loaded(
            loaded_state_keys=loaded_state_keys,
            missing_state_keys=missing_state_keys,
        )


def resolve_checkpoint_file(
    path: str | Path,
    *,
    operation: CheckpointOperation | str = CheckpointOperation.EVALUATE,
) -> Path:
    """Resolve a checkpoint file, step directory, or run directory.

    Model-only weights are preferred when both model-only and full training
    state files exist. Inference should not deserialize optimizer state merely
    because a run uses resumable checkpoints.
    """

    checkpoint_file = find_checkpoint_state_file(
        path,
        layout=CheckpointSearchLayout.RUN_OR_STEP,
        operation=operation,
    )
    if checkpoint_file is not None:
        return checkpoint_file
    expected = (
        "full_training_state.pt"
        if CheckpointOperation(operation) is CheckpointOperation.RESUME_TRAINING
        else " or ".join(CHECKPOINT_FILENAMES)
    )
    raise FileNotFoundError(f"Could not resolve {expected} from {path}.")


def resolve_checkpoint_step_dir_from_transformer_dir(transformer_dir: str | Path) -> Path:
    """Recover a checkpoint step directory from its exported transformer."""

    candidate = Path(transformer_dir).expanduser().resolve()
    if candidate.name != "transformer":
        raise ValueError(
            "Expected the runtime-backbone artifact to point at a "
            f"`.../checkpoint_step_*/transformer` directory, got {candidate}."
        )
    checkpoint_step_dir = candidate.parent
    if not checkpoint_step_dir.name.startswith("checkpoint_step_"):
        raise ValueError(
            "Expected transformer parent to be named `checkpoint_step_*`, "
            f"got {checkpoint_step_dir}."
        )
    return checkpoint_step_dir


def normalize_checkpoint_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract pipeline tensors from supported OpenWAM checkpoint layouts."""

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "Checkpoint must be a raw state dict or contain "
            "`state_dict`/`model_state_dict`."
        )
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        normalized_key = key.removeprefix("pipeline.")
        if normalized_key in normalized:
            raise ValueError(
                "Checkpoint contains ambiguous keys after removing the "
                f"optional `pipeline.` prefix: {normalized_key!r}."
            )
        normalized[normalized_key] = value
    return normalized


def load_pipeline_checkpoint(
    pipeline: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    compatibility: CheckpointCompatibilityPolicy | str = (
        CheckpointCompatibilityPolicy.STRICT
    ),
) -> CheckpointLoadReport:
    """Load one model checkpoint and initialize checkpoint-backed lazy modules."""

    resolved_path = resolve_checkpoint_file(checkpoint_path)
    checkpoint = load_tensor_artifact(resolved_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Expected checkpoint mapping at {resolved_path}, "
            f"got {type(checkpoint).__name__}."
        )
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    runtime_state = pipeline.state_dict()
    missing_keys = tuple(key for key in runtime_state if key not in state_dict)
    unexpected_keys = tuple(key for key in state_dict if key not in runtime_state)
    shape_mismatches = _checkpoint_shape_mismatches(
        runtime_state=runtime_state,
        checkpoint_state=state_dict,
    )
    report = CheckpointLoadReport(
        checkpoint_path=resolved_path,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
    )
    resolved_compatibility = CheckpointCompatibilityPolicy(compatibility)
    if resolved_compatibility is CheckpointCompatibilityPolicy.STRICT:
        incompatible_keys = missing_keys or unexpected_keys
    elif (
        resolved_compatibility
        is CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET
    ):
        incompatible_keys = missing_keys
    else:
        incompatible_keys = ()
    if shape_mismatches or incompatible_keys:
        raise CheckpointCompatibilityError(report)
    loadable_state = (
        {key: value for key, value in state_dict.items() if key in runtime_state}
        if resolved_compatibility
        is CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET
        else state_dict
    )
    pipeline.load_state_dict(
        loadable_state,
        strict=resolved_compatibility
        is not CheckpointCompatibilityPolicy.ALLOW_PARTIAL,
    )
    notify_checkpoint_loaded(
        pipeline,
        loaded_state_keys=frozenset(loadable_state),
        missing_state_keys=frozenset(missing_keys),
    )
    return report


def _preview_keys(keys: tuple[str, ...], *, limit: int = 8) -> str:
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview = f"{preview}, ..."
    return preview


def _checkpoint_shape_mismatches(
    *,
    runtime_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for key, runtime_value in runtime_state.items():
        checkpoint_value = checkpoint_state.get(key)
        if checkpoint_value is None:
            continue
        try:
            runtime_shape = tuple(runtime_value.shape)
            checkpoint_shape = tuple(checkpoint_value.shape)
        except RuntimeError:
            # Lazy modules validate their shape when PyTorch materializes them.
            continue
        if runtime_shape != checkpoint_shape:
            mismatches.append(
                f"{key} (checkpoint={checkpoint_shape}, runtime={runtime_shape})"
            )
    return tuple(mismatches)


def find_checkpoint_resolved_config(path: str | Path | None) -> Path | None:
    """Read sibling metadata even when model weights have not arrived yet."""
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        # Run-root lookup still follows the selected checkpoint's metadata,
        # never a stale root YAML. This inspects filenames, not tensor storage.
        selected_state = find_checkpoint_state_file(candidate)
        if selected_state is not None and selected_state.parent != candidate.resolve():
            selected_config = selected_state.parent / "resolved_config.yaml"
            return selected_config.resolve() if selected_config.is_file() else None
    directory = candidate if candidate.is_dir() else candidate.parent
    sibling_config = directory / "resolved_config.yaml"
    if sibling_config.is_file():
        return sibling_config.resolve()
    checkpoint_file = resolve_checkpoint_file(path)
    resolved_config_path = checkpoint_file.parent / "resolved_config.yaml"
    if resolved_config_path.is_file():
        return resolved_config_path.resolve()
    return None


def merge_checkpoint_runtime_config(
    base_config: ExperimentConfig,
    checkpoint_config: ExperimentConfig,
) -> ExperimentConfig:
    merged_data = replace(
        base_config.data,
        **{
            field.name: (
                getattr(base_config.data, field.name)
                if field.name in _PRESERVED_BASE_DATA_FIELDS
                else getattr(checkpoint_config.data, field.name)
            )
            for field in fields(type(base_config.data))
        },
    )
    return replace(
        base_config,
        data=merged_data,
        backbone=checkpoint_config.backbone,
        policy_variant=checkpoint_config.policy_variant,
        action_decoder=checkpoint_config.action_decoder,
        inference=checkpoint_config.inference,
    )


def _path_or_none(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(str(path)).expanduser()


def _apply_portable_checkpoint_backbone_paths(
    config: ExperimentConfig,
    *,
    base_config: ExperimentConfig,
    checkpoint_dir: Path,
) -> ExperimentConfig:
    backbone_updates: dict[str, str] = {}

    base_pretrained = _path_or_none(base_config.backbone.pretrained_model_name_or_path)
    checkpoint_pretrained = _path_or_none(config.backbone.pretrained_model_name_or_path)
    if (
        base_pretrained is not None
        and base_pretrained.exists()
        and (checkpoint_pretrained is None or not checkpoint_pretrained.exists())
    ):
        backbone_updates["pretrained_model_name_or_path"] = str(base_pretrained.resolve())

    checkpoint_transformer = checkpoint_dir / "transformer"
    if is_usable_transformer_dir(checkpoint_transformer):
        backbone_updates["runtime_backbone_artifact_path"] = str(
            checkpoint_transformer.resolve()
        )

    if not backbone_updates:
        return config
    return replace(config, backbone=replace(config.backbone, **backbone_updates))


def merge_runtime_config_from_checkpoint(
    base_config: ExperimentConfig,
    checkpoint_path: str | Path | None,
) -> tuple[ExperimentConfig, Path | None]:
    resolved_config_path = find_checkpoint_resolved_config(checkpoint_path)
    if resolved_config_path is None:
        return base_config, None
    checkpoint_config = load_experiment_config(
        resolved_config_path,
        checkpoint_runtime_compat=True,
    )
    merged_config = merge_checkpoint_runtime_config(base_config, checkpoint_config)
    merged_config = _apply_portable_checkpoint_backbone_paths(
        merged_config,
        base_config=base_config,
        checkpoint_dir=resolved_config_path.parent,
    )
    return merged_config, resolved_config_path
