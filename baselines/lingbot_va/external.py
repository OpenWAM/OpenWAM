from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_SOURCE_REPO_CANDIDATES = (
    REPO_ROOT / "previous_works" / "lingbot-va",
    Path("/path/to/private-resource"),
)


@dataclass(frozen=True)
class ExternalModules:
    source_repo: Path
    VA_CONFIGS: Any
    VA_Server: type
    benchmark: Any
    OffScreenRenderEnv: type
    LiberoTaskSpec: type
    ensure_local_libero_config: Callable[..., Any]
    load_libero_task_init_states: Callable[..., Any]
    seed_everywhere: Callable[..., Any]


def resolve_source_repo(source_repo: str | Path | None = None) -> Path:
    candidates = (Path(source_repo).expanduser(),) if source_repo is not None else DEFAULT_SOURCE_REPO_CANDIDATES
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "wan_va" / "wan_va_server.py").is_file():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"LingBot-VA source repo not found. Searched: {searched}")


def prepare_python_paths(source_repo: str | Path | None = None) -> Path:
    resolved_source = resolve_source_repo(source_repo)
    for path in (SRC_ROOT, resolved_source, resolved_source / "wan_va"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return resolved_source


def load_external_modules(source_repo: str | Path | None = None) -> ExternalModules:
    resolved_source = prepare_python_paths(source_repo)
    _bootstrap_libero_config_without_prompt()

    from open_wam.third_party.lingbot import _ensure_flash_attn_shims
    from open_wam.integrations import (
        LiberoTaskSpec,
        ensure_local_libero_config,
        load_libero_task_init_states,
    )
    from open_wam.utils import seed_everywhere

    _ensure_flash_attn_shims()
    ensure_local_libero_config(REPO_ROOT)

    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")
    configs = importlib.import_module("wan_va.configs")
    server = importlib.import_module("wan_va.wan_va_server")

    return ExternalModules(
        source_repo=resolved_source,
        VA_CONFIGS=configs.VA_CONFIGS,
        VA_Server=server.VA_Server,
        benchmark=benchmark,
        OffScreenRenderEnv=envs.OffScreenRenderEnv,
        LiberoTaskSpec=LiberoTaskSpec,
        ensure_local_libero_config=ensure_local_libero_config,
        load_libero_task_init_states=load_libero_task_init_states,
        seed_everywhere=seed_everywhere,
    )


def _bootstrap_libero_config_without_prompt() -> None:
    """Seed LIBERO_CONFIG_PATH before any upstream LIBERO import can prompt."""

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", REPO_ROOT / ".cache" / "libero_config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if config_path.exists():
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        return

    repo_root = Path(os.environ.get("LIBERO_REPO_ROOT", REPO_ROOT.parent / "LIBERO")).expanduser()
    package_root = repo_root / "libero" / "libero"
    if not (package_root / "__init__.py").is_file():
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        return

    config = {
        "benchmark_root": str(package_root.resolve()),
        "bddl_files": str((package_root / "bddl_files").resolve()),
        "init_states": str((package_root / "init_files").resolve()),
        "datasets": str((repo_root / "libero" / "datasets").resolve()),
        "assets": str((package_root / "assets").resolve()),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
