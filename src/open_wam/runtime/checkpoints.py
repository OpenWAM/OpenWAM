from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import ExperimentConfig, load_experiment_config
from open_wam.runtime.checkpoint_artifacts import (
    CHECKPOINT_FILENAMES,
    CheckpointSearchLayout,
    find_checkpoint_state_file,
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


def resolve_checkpoint_file(path: str | Path) -> Path:
    """Resolve a checkpoint file, step directory, or run directory.

    Model-only weights are preferred when both model-only and full training
    state files exist. Inference should not deserialize optimizer state merely
    because a run uses resumable checkpoints.
    """

    checkpoint_file = find_checkpoint_state_file(
        path,
        layout=CheckpointSearchLayout.STEP_OR_CHILD_STEPS,
    )
    if checkpoint_file is not None:
        return checkpoint_file
    expected = " or ".join(CHECKPOINT_FILENAMES)
    raise FileNotFoundError(f"Could not resolve {expected} from {path}.")


def resolve_checkpoint_step_dir_from_transformer_dir(transformer_dir: str | Path) -> Path:
    """Recover a checkpoint step directory from its exported transformer."""

    candidate = Path(transformer_dir).expanduser().resolve()
    if candidate.name != "transformer":
        raise ValueError(
            "Expected `backbone.transformer_subdir` to point at a "
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
    """Extract pipeline tensors from supported Open-WAM checkpoint layouts."""

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
    if shape_mismatches or (
        resolved_compatibility is CheckpointCompatibilityPolicy.STRICT
        and (missing_keys or unexpected_keys)
    ):
        raise CheckpointCompatibilityError(report)
    pipeline.load_state_dict(
        state_dict,
        strict=resolved_compatibility is CheckpointCompatibilityPolicy.STRICT,
    )
    _mark_loaded_lazy_components_initialized(
        pipeline,
        state_dict,
        missing_keys=missing_keys,
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


def _mark_loaded_lazy_components_initialized(
    pipeline: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    missing_keys: tuple[str, ...] = (),
) -> None:
    policy_variant = getattr(pipeline, "policy_variant", None)
    if policy_variant is None:
        return
    prefix = "policy_variant.action_expert."
    has_action_expert_weights = any(key.startswith(prefix) for key in state_dict)
    missing_action_expert_weights = any(key.startswith(prefix) for key in missing_keys)
    if (
        hasattr(policy_variant, "_action_expert_initialized")
        and has_action_expert_weights
        and not missing_action_expert_weights
    ):
        policy_variant._action_expert_initialized = True


def find_checkpoint_resolved_config(path: str | Path | None) -> Path | None:
    if path is None:
        return None
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
    if checkpoint_transformer.is_dir() and any(checkpoint_transformer.iterdir()):
        backbone_updates["transformer_subdir"] = str(checkpoint_transformer.resolve())

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
