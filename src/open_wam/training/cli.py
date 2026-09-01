from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from open_wam.cli.train_arguments import build_train_arg_parser as build_cli_arg_parser
from open_wam.configs import (
    ExperimentConfig,
    load_experiment_config,
    resolve_config_path_alias,
)
from open_wam.configs.config_paths import EXPERIMENT_CONFIG_ROOT
from open_wam.extensions import load_extension_modules
from open_wam.runtime.checkpoint_artifacts import CheckpointOperation
from open_wam.runtime.checkpoints import resolve_checkpoint_file
from open_wam.utils.config_overrides import (
    apply_config_overrides,
    parse_override_assignments,
)


@dataclass(frozen=True)
class TrainCliOverrides:
    """Resolved CLI-level overrides for one training launch."""

    config: str | None = None
    config_name: str | None = None
    save_root: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_root: str | None = None
    initialize_weights_from: str | None = None
    resume_from: str | None = None
    run_name: str | None = None
    dataset_root: str | None = None
    latent_root: str | None = None
    runtime_backbone_artifact_path: str | None = None
    devices: int | None = None
    expected_world_size: int | None = None
    num_steps: int | None = None
    enable_wandb: bool = False
    disable_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None
    extensions: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()


def build_train_arg_parser() -> ArgumentParser:
    """Return the package-owned public parser used by every train entrypoint."""

    return build_cli_arg_parser()


def parse_train_cli(argv: list[str] | None = None) -> TrainCliOverrides:
    parser = build_train_arg_parser()
    args, extras = parser.parse_known_args(argv)
    if (
        args.devices is not None
        and args.expected_world_size is not None
        and args.devices != args.expected_world_size
    ):
        parser.error(
            "--devices and --expected-world-size must match when both are provided."
        )
    return TrainCliOverrides(
        config=args.config,
        config_name=args.config_name,
        save_root=args.save_root,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_root=args.checkpoint_root,
        initialize_weights_from=args.initialize_weights_from,
        resume_from=args.resume_from,
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        runtime_backbone_artifact_path=args.runtime_backbone_artifact_path,
        devices=args.devices,
        expected_world_size=args.expected_world_size,
        num_steps=args.num_steps,
        enable_wandb=args.enable_wandb,
        disable_wandb=args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        extensions=tuple(args.extension),
        overrides=tuple(_normalize_override_tokens([*args.set_overrides, *extras])),
    )


def resolve_experiment_config_path(overrides: TrainCliOverrides) -> Path:
    if overrides.config is not None:
        return resolve_config_path_alias(overrides.config)
    if overrides.config_name is None:
        raise ValueError("Either `config` or `config_name` must be provided.")
    raw_name = overrides.config_name
    candidate = Path(raw_name).expanduser()
    if candidate.is_absolute() or candidate.suffix in {".yaml", ".yml"} or len(candidate.parts) > 1:
        if candidate.suffix:
            return resolve_config_path_alias(candidate)
        return resolve_config_path_alias(candidate.with_suffix(".yaml"))
    return resolve_config_path_alias(EXPERIMENT_CONFIG_ROOT / f"{raw_name}.yaml")


def load_training_cli_config(
    overrides: TrainCliOverrides,
    *,
    env: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    load_extension_modules(overrides.extensions)
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
    if overrides.checkpoint_root is not None:
        raise ValueError(
            "`--checkpoint-root` is no longer supported because its training "
            "operation was ambiguous. Use `--initialize-weights-from` to start "
            "fresh from model weights or `--resume-from` to continue full state."
        )
    initialization_source = overrides.initialize_weights_from
    if initialization_source is not None:
        update_map["trainer.initialize_weights_from"] = str(
            resolve_checkpoint_file(
                initialization_source,
                operation=CheckpointOperation.INITIALIZE_WEIGHTS,
            )
        )
    if overrides.resume_from is not None:
        update_map["trainer.resume_from"] = str(
            resolve_checkpoint_file(
                overrides.resume_from,
                operation=CheckpointOperation.RESUME_TRAINING,
            )
        )
    if overrides.dataset_root is not None:
        update_map["data.local_root"] = overrides.dataset_root
    if overrides.latent_root is not None:
        update_map["data.latent_root"] = overrides.latent_root
    if overrides.runtime_backbone_artifact_path is not None:
        update_map["backbone.runtime_backbone_artifact_path"] = (
            overrides.runtime_backbone_artifact_path
        )
    if overrides.devices is not None:
        update_map["trainer.devices"] = overrides.devices
    if overrides.num_steps is not None:
        update_map["training.num_steps"] = overrides.num_steps
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
    return apply_wandb_env_defaults(
        config,
        env=env or os.environ,
        use_env_project=overrides.wandb_project is None,
        use_env_entity=overrides.wandb_entity is None,
        use_env_mode=overrides.wandb_mode is None,
    )


def apply_wandb_env_defaults(
    config: ExperimentConfig,
    *,
    env: Mapping[str, str],
    use_env_project: bool = True,
    use_env_entity: bool = True,
    use_env_mode: bool = True,
) -> ExperimentConfig:
    if not config.trainer.enable_wandb:
        return config
    updates: dict[str, Any] = {}
    if use_env_project and env.get("WANDB_PROJECT"):
        updates["wandb_project"] = env["WANDB_PROJECT"]
    entity = env.get("WANDB_ENTITY") or env.get("WANDB_TEAM_NAME")
    if use_env_entity and entity:
        updates["wandb_entity"] = entity
    if use_env_mode and env.get("WANDB_MODE"):
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
