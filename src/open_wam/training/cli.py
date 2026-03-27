from __future__ import annotations

import argparse
from dataclasses import dataclass, is_dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from open_wam.configs import ExperimentConfig, WandBMode
from open_wam.utils import load_experiment_config


EXPERIMENT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "experiments"


@dataclass(frozen=True)
class TrainCliOverrides:
    """Resolved CLI-level overrides for one training launch."""

    config: str | None = None
    config_name: str | None = None
    save_root: str | None = None
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    run_name: str | None = None
    dataset_root: str | None = None
    latent_root: str | None = None
    transformer_subdir: str | None = None
    devices: int | None = None
    enable_wandb: bool = False
    disable_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None
    overrides: tuple[str, ...] = ()


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--cfg", "--config", dest="config", type=str)
    config_group.add_argument("--config-name", dest="config_name", type=str)
    parser.add_argument(
        "--save-root",
        type=str,
        help="Full run output directory. This mirrors LingBot's `save_root` semantics.",
    )
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--latent-root", type=str)
    parser.add_argument("--transformer-subdir", type=str)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-mode", type=str)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Repeatable `section.field=value` override.",
    )
    return parser


def parse_train_cli(argv: list[str] | None = None) -> TrainCliOverrides:
    parser = build_train_arg_parser()
    args, extras = parser.parse_known_args(argv)
    return TrainCliOverrides(
        config=args.config,
        config_name=args.config_name,
        save_root=args.save_root,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        transformer_subdir=args.transformer_subdir,
        devices=args.devices,
        enable_wandb=args.enable_wandb,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        overrides=tuple(_normalize_override_tokens([*args.set_overrides, *extras])),
    )


def resolve_experiment_config_path(overrides: TrainCliOverrides) -> Path:
    if overrides.config is not None:
        return Path(overrides.config).expanduser()
    if overrides.config_name is None:
        raise ValueError("Either `config` or `config_name` must be provided.")
    raw_name = overrides.config_name
    candidate = Path(raw_name).expanduser()
    if candidate.is_absolute() or candidate.suffix in {".yaml", ".yml"} or len(candidate.parts) > 1:
        if candidate.suffix:
            return candidate
        return candidate.with_suffix(".yaml")
    return EXPERIMENT_CONFIG_ROOT / f"{raw_name}.yaml"


def load_training_cli_config(
    overrides: TrainCliOverrides,
    *,
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    config = load_experiment_config(resolve_experiment_config_path(overrides))
    return apply_train_cli_overrides(config, overrides=overrides, env=env)


def apply_train_cli_overrides(
    config: ExperimentConfig,
    *,
    overrides: TrainCliOverrides,
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    if overrides.enable_wandb and overrides.disable_wandb:
        raise ValueError("Choose either `--enable-wandb` or `--disable-wandb`, not both.")

    update_map: dict[str, Any] = {}
    if overrides.save_root is not None:
        save_root = Path(overrides.save_root).expanduser()
        if overrides.run_name is not None and overrides.run_name != save_root.name:
            raise ValueError(
                "`--save-root` is a full run directory. If `--run-name` is also set, "
                "it must match the basename of `--save-root`."
            )
        update_map["trainer.default_root_dir"] = str(save_root.parent)
        update_map["trainer.run_name"] = save_root.name
        if overrides.checkpoint_dir is None:
            update_map["trainer.checkpoint_dir"] = str(save_root / "checkpoints")
    elif overrides.run_name is not None:
        update_map["trainer.run_name"] = overrides.run_name

    if overrides.checkpoint_dir is not None:
        update_map["trainer.checkpoint_dir"] = overrides.checkpoint_dir
    if overrides.resume_from is not None:
        update_map["trainer.resume_from"] = overrides.resume_from
    if overrides.dataset_root is not None:
        update_map["data.local_root"] = overrides.dataset_root
    if overrides.latent_root is not None:
        update_map["data.latent_root"] = overrides.latent_root
    if overrides.transformer_subdir is not None:
        update_map["backbone.transformer_subdir"] = overrides.transformer_subdir
    if overrides.devices is not None:
        update_map["trainer.devices"] = overrides.devices
    if overrides.enable_wandb:
        update_map["trainer.enable_wandb"] = True
    if overrides.disable_wandb:
        update_map["trainer.enable_wandb"] = False
    if overrides.wandb_project is not None:
        update_map["trainer.wandb_project"] = overrides.wandb_project
    if overrides.wandb_entity is not None:
        update_map["trainer.wandb_entity"] = overrides.wandb_entity
    if overrides.wandb_mode is not None:
        update_map["trainer.wandb_mode"] = overrides.wandb_mode

    update_map.update(parse_override_assignments(overrides.overrides))
    config = apply_config_overrides(config, update_map)
    return apply_wandb_env_defaults(config, env=env or os.environ)


def parse_override_assignments(tokens: tuple[str, ...] | list[str]) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for token in tokens:
        key, raw_value = _split_override_token(token)
        assignments[key] = yaml.safe_load(raw_value)
    return assignments


def apply_config_overrides(config: ExperimentConfig, overrides: Mapping[str, Any]) -> ExperimentConfig:
    updated = config
    for key, value in overrides.items():
        updated = _replace_dataclass_path(updated, key.split("."), value)
    return updated


def apply_wandb_env_defaults(
    config: ExperimentConfig,
    *,
    env: Mapping[str, str],
) -> ExperimentConfig:
    if not config.trainer.enable_wandb:
        return config
    updates: dict[str, Any] = {}
    if config.trainer.wandb_project is None and env.get("WANDB_PROJECT"):
        updates["wandb_project"] = env["WANDB_PROJECT"]
    entity = env.get("WANDB_ENTITY") or env.get("WANDB_TEAM_NAME")
    if config.trainer.wandb_entity is None and entity:
        updates["wandb_entity"] = entity
    if config.trainer.wandb_mode == WandBMode.DISABLED and env.get("WANDB_MODE"):
        updates["wandb_mode"] = env["WANDB_MODE"]
    if not updates:
        return config
    return replace(config, trainer=replace(config.trainer, **updates))


def _normalize_override_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        stripped = token.lstrip("-")
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError(
                "Additional CLI overrides must use `section.field=value` syntax. "
                f"Got {token!r}."
            )
        normalized.append(stripped)
    return normalized


def _split_override_token(token: str) -> tuple[str, str]:
    key, raw_value = token.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise ValueError(f"Override key is empty in token {token!r}.")
    return key, raw_value


def _replace_dataclass_path(node: object, path: list[str], value: Any):
    if not is_dataclass(node):
        raise ValueError(f"Cannot override nested path on non-dataclass node {type(node).__name__}.")
    field_name = path[0].replace("-", "_")
    if not hasattr(node, field_name):
        raise ValueError(f"{type(node).__name__} has no field {field_name!r}.")
    current_value = getattr(node, field_name)
    if len(path) == 1:
        coerced = _coerce_override_value(current_value, value)
        return replace(node, **{field_name: coerced})
    nested_value = _replace_dataclass_path(current_value, path[1:], value)
    return replace(node, **{field_name: nested_value})


def _coerce_override_value(current_value: Any, value: Any) -> Any:
    if isinstance(current_value, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current_value, bool) and isinstance(value, int):
        return bool(value)
    if isinstance(current_value, float) and isinstance(value, int):
        return float(value)
    return value
