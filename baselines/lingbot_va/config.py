from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


_ENV_WITH_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_env_vars(value: str) -> str:
    """Expand $VAR and ${VAR:-fallback} without making YAML machine-specific."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if fallback is not None:
            return fallback
        return match.group(0)

    return os.path.expandvars(_ENV_WITH_DEFAULT.sub(replace, value))


def resolve_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(expand_env_vars(str(value))).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def parse_int_selection(selection: str | int | Sequence[int]) -> list[int]:
    """Parse comma/range syntax such as '0', '0,2,4', '0:10', or '0-9'."""

    if isinstance(selection, int):
        return [selection]
    if not isinstance(selection, str):
        return [int(value) for value in selection]

    values: list[int] = []
    for raw_part in selection.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_str, end_str = part.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            values.extend(range(start, end))
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    pretrained_root: Path
    transformer_dir: Path | None = None
    source_repo: Path | None = None
    enable_offload: bool | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretrained_root": str(self.pretrained_root),
            "transformer_dir": str(self.transformer_dir) if self.transformer_dir else None,
            "source_repo": str(self.source_repo) if self.source_repo else None,
            "enable_offload": self.enable_offload,
        }


@dataclass(frozen=True)
class EpisodeSpec:
    benchmark: str
    task_id: int
    episode_idx: int
    seed: int | None = None

    def suffix(self) -> str:
        seed_part = "none" if self.seed is None else str(self.seed)
        return f"{self.benchmark}_task{self.task_id}_ep{self.episode_idx}_seed{seed_part}"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "episode_idx": self.episode_idx,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RolloutSuiteConfig:
    checkpoints: tuple[CheckpointSpec, ...]
    benchmark: str = "libero_10"
    task_ids: tuple[int, ...] = tuple(range(10))
    episode_indices: tuple[int, ...] = (0,)
    seed: int | None = 0
    max_timestep: int = 1000
    max_chunks: int | None = None
    video_fps: float = 15.0
    output_dir: Path = Path("outputs/lingbot_va_baseline")
    cuda_device: int = 0
    render_video: bool = True
    continue_on_error: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "checkpoints": [checkpoint.to_json_dict() for checkpoint in self.checkpoints],
            "benchmark": self.benchmark,
            "task_ids": list(self.task_ids),
            "episode_indices": list(self.episode_indices),
            "seed": self.seed,
            "max_timestep": self.max_timestep,
            "max_chunks": self.max_chunks,
            "video_fps": self.video_fps,
            "output_dir": str(self.output_dir),
            "cuda_device": self.cuda_device,
            "render_video": self.render_video,
            "continue_on_error": self.continue_on_error,
        }


def iter_episode_specs(config: RolloutSuiteConfig) -> Iterable[EpisodeSpec]:
    for task_id in config.task_ids:
        for episode_idx in config.episode_indices:
            yield EpisodeSpec(
                benchmark=config.benchmark,
                task_id=int(task_id),
                episode_idx=int(episode_idx),
                seed=config.seed,
            )


def load_suite_config(path: str | Path) -> RolloutSuiteConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, Mapping):
        raise TypeError(f"LingBot baseline suite must be a mapping: {config_path}")
    return suite_config_from_mapping(raw_config, base_dir=config_path.parent)


def suite_config_from_mapping(raw_config: Mapping[str, Any], *, base_dir: Path | None = None) -> RolloutSuiteConfig:
    source_repo = resolve_path(raw_config.get("source_repo"), base_dir=base_dir)
    pretrained_root = resolve_path(raw_config.get("pretrained_root"), base_dir=base_dir)

    checkpoint_items = raw_config.get("checkpoints", [])
    if not checkpoint_items:
        checkpoint_items = [
            {
                "name": raw_config.get("checkpoint_name", "lingbot_va"),
                "transformer_dir": raw_config.get("transformer_dir"),
            }
        ]
    checkpoints = tuple(
        _checkpoint_from_mapping(
            item,
            default_source_repo=source_repo,
            default_pretrained_root=pretrained_root,
            base_dir=base_dir,
        )
        for item in checkpoint_items
    )
    if not checkpoints:
        raise ValueError("At least one checkpoint is required for a LingBot-VA baseline suite.")

    episodes = raw_config.get("episodes", {})
    runtime = raw_config.get("runtime", {})
    if not isinstance(episodes, Mapping):
        raise TypeError("suite 'episodes' section must be a mapping.")
    if not isinstance(runtime, Mapping):
        raise TypeError("suite 'runtime' section must be a mapping.")

    output_dir = resolve_path(raw_config.get("output_dir", "outputs/lingbot_va_baseline"), base_dir=Path.cwd())
    assert output_dir is not None
    return RolloutSuiteConfig(
        checkpoints=checkpoints,
        benchmark=str(episodes.get("benchmark", raw_config.get("benchmark", "libero_10"))),
        task_ids=tuple(parse_int_selection(episodes.get("task_ids", raw_config.get("task_ids", "0:10")))),
        episode_indices=tuple(
            parse_int_selection(episodes.get("episode_indices", raw_config.get("episode_indices", "0")))
        ),
        seed=_optional_int(episodes.get("seed", raw_config.get("seed", 0))),
        max_timestep=int(runtime.get("max_timestep", raw_config.get("max_timestep", 1000))),
        max_chunks=_optional_int(runtime.get("max_chunks", raw_config.get("max_chunks"))),
        video_fps=float(runtime.get("video_fps", raw_config.get("video_fps", 15.0))),
        output_dir=output_dir,
        cuda_device=int(runtime.get("cuda_device", raw_config.get("cuda_device", 0))),
        render_video=_bool_value(runtime.get("render_video", raw_config.get("render_video", True))),
        continue_on_error=_bool_value(
            runtime.get("continue_on_error", raw_config.get("continue_on_error", False))
        ),
    )


def _checkpoint_from_mapping(
    item: Mapping[str, Any],
    *,
    default_source_repo: Path | None,
    default_pretrained_root: Path | None,
    base_dir: Path | None,
) -> CheckpointSpec:
    if not isinstance(item, Mapping):
        raise TypeError(f"Checkpoint entries must be mappings, got {type(item)!r}.")
    name = str(item.get("name") or item.get("checkpoint_name") or "lingbot_va")
    source_repo = resolve_path(item.get("source_repo"), base_dir=base_dir) or default_source_repo
    pretrained_root = resolve_path(item.get("pretrained_root"), base_dir=base_dir) or default_pretrained_root
    if pretrained_root is None:
        raise ValueError(f"Checkpoint '{name}' must define pretrained_root or use suite-level pretrained_root.")
    return CheckpointSpec(
        name=name,
        pretrained_root=pretrained_root,
        transformer_dir=resolve_path(item.get("transformer_dir"), base_dir=base_dir),
        source_repo=source_repo,
        enable_offload=_optional_bool(item.get("enable_offload")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _bool_value(value: Any) -> bool:
    parsed = _optional_bool(value)
    if parsed is None:
        raise ValueError("Boolean value cannot be empty here.")
    return parsed
