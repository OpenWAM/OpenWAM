from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from open_wam._shims.loader import ensure_flash_attn_shims
from open_wam.configs.backbone import LingbotCompatibleVideoBackboneConfig
from open_wam.contracts.paths import resolve_repo_path


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
        resolved = resolve_repo_path(raw_path)
    if not resolved.exists():
        raise FileNotFoundError(
            "Unable to find the LingBot reference model source file at "
            f"{resolved}. Set `backbone.reference_model_path` to a valid model.py path."
        )
    return resolved

@lru_cache(maxsize=1)
def load_internal_wan_transformer_class() -> type:
    ensure_flash_attn_shims()
    from open_wam.third_party.lingbot import WanTransformer3DModel

    return WanTransformer3DModel


@lru_cache(maxsize=1)
def load_reference_wan_transformer_class(reference_model_path: str) -> type:
    module_path = Path(reference_model_path).resolve()
    ensure_flash_attn_shims()
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
