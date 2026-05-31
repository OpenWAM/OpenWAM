from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from open_wam.configs import ExportedRuntimeActionInitMode
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig

from .reference_core_weights import BackboneLoadReport
from .reference_loader import resolve_pretrained_component_dir

_EXPORT_WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"
_OPEN_WAM_EXPORT_PREFIXES = (
    "time_conditioner.",
    "action_time_conditioner.",
    "text_proj.",
    "action_text_proj.",
)
_REFERENCE_PREFIXES = (
    "condition_embedder.",
    "condition_embedder_action.",
)
_OPTIONAL_RUNTIME_TARGET_PREFIXES = (
    "proprio_context_encoder.",
    "proprio_hidden_context_encoder.",
    "generalist_mode_context_encoder.",
)
_ACTION_RUNTIME_TARGET_PREFIXES = (
    "action_time_conditioner.",
    "action_text_proj.",
    "runtime_stream_adapters.action_register_adapter.",
)
_ACTION_RUNTIME_TARGET_KEYS = frozenset(
    {
        "action_embedder.weight",
        "action_embedder.bias",
        "action_proj_out.weight",
        "action_proj_out.bias",
    }
)


def is_action_runtime_target_key(key: str) -> bool:
    """Return true for exported-runtime tensors owned by the action path."""

    return key in _ACTION_RUNTIME_TARGET_KEYS or key.startswith(_ACTION_RUNTIME_TARGET_PREFIXES)


def is_allowed_runtime_missing_key(key: str, *, allow_random_action: bool) -> bool:
    """Classify intentionally missing/skipped runtime-backbone load keys."""

    return key.startswith(_OPTIONAL_RUNTIME_TARGET_PREFIXES) or (
        allow_random_action and is_action_runtime_target_key(key)
    )


def resolve_runtime_backbone_dir(backbone_config: SharedVideoTransformerConfig) -> Path | None:
    return resolve_pretrained_component_dir(
        backbone_config.pretrained_model_name_or_path,
        backbone_config.transformer_subdir,
    )


def is_open_wam_exported_runtime_backbone_dir(path: Path | None) -> bool:
    if path is None:
        return False
    weights_path = path / _EXPORT_WEIGHTS_FILENAME
    if not weights_path.exists():
        return False
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
    has_open_wam_prefix = any(key.startswith(_OPEN_WAM_EXPORT_PREFIXES) for key in keys)
    has_reference_prefix = any(key.startswith(_REFERENCE_PREFIXES) for key in keys)
    return has_open_wam_prefix and not has_reference_prefix


def load_exported_runtime_backbone_into_replica_core(
    replica_core: torch.nn.Module,
    *,
    backbone_config: SharedVideoTransformerConfig,
) -> BackboneLoadReport:
    runtime_backbone_dir = resolve_runtime_backbone_dir(backbone_config)
    if runtime_backbone_dir is None:
        raise ValueError("Runtime backbone export loading requires a resolved transformer directory.")
    weights_path = runtime_backbone_dir / _EXPORT_WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(f"Unable to find exported runtime backbone weights at {weights_path}.")

    exported_state = load_file(str(weights_path), device="cpu")
    target_state = replica_core.state_dict()
    loaded_keys: list[str] = []
    missing_reference_keys: list[str] = []
    random_action_init = backbone_config.exported_runtime_action_init_mode == ExportedRuntimeActionInitMode.RANDOM
    for target_key, target_value in target_state.items():
        if random_action_init and is_action_runtime_target_key(target_key):
            missing_reference_keys.append(target_key)
            continue
        source_value = exported_state.get(target_key)
        if source_value is None:
            missing_reference_keys.append(target_key)
            continue
        source_value = source_value.detach()
        if tuple(source_value.shape) != tuple(target_value.shape):
            missing_reference_keys.append(target_key)
            continue
        if torch.is_floating_point(source_value):
            source_value = source_value.to(dtype=target_value.dtype)
        target_state[target_key] = source_value.clone()
        loaded_keys.append(target_key)
    replica_core.load_state_dict(target_state, strict=False)
    return BackboneLoadReport(
        loaded_keys=tuple(loaded_keys),
        missing_reference_keys=tuple(missing_reference_keys),
    )
