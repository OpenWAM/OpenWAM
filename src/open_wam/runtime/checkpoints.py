from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from open_wam.configs import ExperimentConfig, load_experiment_config


CHECKPOINT_FILENAMES = ("model_state.pt", "full_training_state.pt")
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


def resolve_checkpoint_file(path: str | Path) -> Path:
    """Resolve a checkpoint file, step directory, or run directory.

    Model-only weights are preferred when both model-only and full training
    state files exist. Inference should not deserialize optimizer state merely
    because a run uses resumable checkpoints.
    """

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    for filename in CHECKPOINT_FILENAMES:
        direct_file = candidate / filename
        if direct_file.is_file():
            return direct_file
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    for checkpoint_dir in reversed(checkpoint_dirs):
        for filename in CHECKPOINT_FILENAMES:
            checkpoint_file = checkpoint_dir / filename
            if checkpoint_file.is_file():
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
        normalized[normalized_key] = value
    return normalized


def load_pipeline_checkpoint(
    pipeline: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadReport:
    """Load one model checkpoint and initialize checkpoint-backed lazy modules."""

    resolved_path = resolve_checkpoint_file(checkpoint_path)
    try:
        checkpoint = torch.load(
            resolved_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(resolved_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Expected checkpoint mapping at {resolved_path}, "
            f"got {type(checkpoint).__name__}."
        )
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    incompatible = pipeline.load_state_dict(state_dict, strict=False)
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    _mark_loaded_lazy_components_initialized(
        pipeline,
        state_dict,
        missing_keys=missing_keys,
    )
    return CheckpointLoadReport(
        checkpoint_path=resolved_path,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


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
