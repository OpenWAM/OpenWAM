from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from open_wam.configs import ExperimentConfig
from open_wam.utils.config_loader import load_experiment_config


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


def resolve_checkpoint_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    for filename in ("model_state.pt", "full_training_state.pt"):
        direct_file = candidate / filename
        if direct_file.is_file():
            return direct_file
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    for checkpoint_dir in reversed(checkpoint_dirs):
        for filename in ("model_state.pt", "full_training_state.pt"):
            checkpoint_file = checkpoint_dir / filename
            if checkpoint_file.is_file():
                return checkpoint_file
    raise FileNotFoundError(f"Could not resolve model_state.pt or full_training_state.pt from {path}.")


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
    checkpoint_config = load_experiment_config(resolved_config_path, checkpoint_runtime_compat=True)
    merged_config = merge_checkpoint_runtime_config(base_config, checkpoint_config)
    merged_config = _apply_portable_checkpoint_backbone_paths(
        merged_config,
        base_config=base_config,
        checkpoint_dir=resolved_config_path.parent,
    )
    return merged_config, resolved_config_path
