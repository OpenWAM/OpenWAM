from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_reference_model_path(config: LingbotCompatibleVideoBackboneConfig) -> Path:
    if config.reference_model_path is None:
        raise ValueError(
            "No external reference model path was provided. "
            "Set `backbone.reference_model_path` only if you want to override the vendored LingBot reference model."
        )
    raw_path = Path(config.reference_model_path)
    if raw_path.is_absolute():
        resolved = raw_path
    else:
        resolved = (_repo_root() / raw_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            "Unable to find the LingBot reference model source file at "
            f"{resolved}. Set `backbone.reference_model_path` to a valid model.py path."
        )
    return resolved


def _shim_root() -> Path:
    return _repo_root() / "src" / "open_wam" / "_shims"


def _install_shim_module(module_name: str, relative_path: str) -> None:
    module_path = (_shim_root() / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import flash-attn shim from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _ensure_flash_attn_shims() -> None:
    for module_name, relative_path in (
        ("flash_attn_interface", "flash_attn_interface.py"),
        ("flash_attn", "flash_attn.py"),
    ):
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except ImportError:
            _install_shim_module(module_name, relative_path)


@lru_cache(maxsize=1)
def load_internal_wan_transformer_class() -> type:
    _ensure_flash_attn_shims()
    from open_wam.third_party.lingbot import WanTransformer3DModel

    return WanTransformer3DModel


@lru_cache(maxsize=1)
def load_reference_wan_transformer_class(reference_model_path: str) -> type:
    module_path = Path(reference_model_path).resolve()
    _ensure_flash_attn_shims()
    spec = importlib.util.spec_from_file_location("open_wam._lingbot_reference_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import LingBot reference model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WanTransformer3DModel


def load_wan_transformer_class(config: LingbotCompatibleVideoBackboneConfig) -> type:
    if config.reference_model_path is None:
        return load_internal_wan_transformer_class()
    return load_reference_wan_transformer_class(str(resolve_reference_model_path(config)))


def resolve_pretrained_component_dir(
    pretrained_model_name_or_path: str | None,
    subdir: str,
) -> Path | None:
    if pretrained_model_name_or_path is None:
        return None
    root = Path(pretrained_model_name_or_path).expanduser()
    candidate = root / subdir
    if candidate.exists():
        return candidate
    if (root / "config.json").exists():
        return root
    return candidate
