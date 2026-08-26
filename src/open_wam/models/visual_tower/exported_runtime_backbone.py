from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from open_wam.configs import ExportedRuntimeActionInitMode
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.runtime_backbone_components import (
    is_complete_runtime_backbone_selection,
)
from open_wam.runtime.runtime_backbone_manifest import (
    load_runtime_backbone_manifest,
)

from .component_ownership import owned_state_dict_keys, resolve_visual_components
from .reference_core_weights import BackboneLoadReport
from .reference_loader import resolve_runtime_backbone_dir

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
    "runtime_stream_adapters.state_register_adapter.",
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

    return key in _ACTION_RUNTIME_TARGET_KEYS or key.startswith(
        _ACTION_RUNTIME_TARGET_PREFIXES
    )


def _is_optional_runtime_component_key(key: str) -> bool:
    return key.startswith(_OPTIONAL_RUNTIME_TARGET_PREFIXES)


def is_allowed_runtime_missing_key(key: str, *, allow_random_action: bool) -> bool:
    """Classify intentionally missing/skipped runtime-backbone load keys."""

    return _is_optional_runtime_component_key(key) or (
        allow_random_action and is_action_runtime_target_key(key)
    )


def is_open_wam_exported_runtime_backbone_dir(path: Path | None) -> bool:
    if path is None:
        return False
    weights_path = path / _EXPORT_WEIGHTS_FILENAME
    if not weights_path.exists():
        return False
    if load_runtime_backbone_manifest(path) is not None:
        return True
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
        raise ValueError(
            "Runtime backbone export loading requires a resolved transformer directory."
        )
    weights_path = runtime_backbone_dir / _EXPORT_WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Unable to find exported runtime backbone weights at {weights_path}."
        )

    exported_state = load_file(str(weights_path), device="cpu")
    manifest = load_runtime_backbone_manifest(runtime_backbone_dir)
    target_state = replica_core.state_dict()
    complete_artifact = manifest is None or is_complete_runtime_backbone_selection(
        manifest.components
    )
    random_action_init = (
        backbone_config.exported_runtime_action_init_mode
        == ExportedRuntimeActionInitMode.RANDOM
    )
    if manifest is not None:
        declared_keys = set(manifest.state_keys)
        exported_keys = set(exported_state)
        if declared_keys != exported_keys:
            raise ValueError(
                "Runtime-backbone weights do not match their manifest: "
                f"missing={sorted(declared_keys - exported_keys)[:20]}, "
                f"undeclared={sorted(exported_keys - declared_keys)[:20]}."
            )
        unexpected_keys = sorted(
            key
            for key in exported_keys - set(target_state)
            if not (complete_artifact and _is_optional_runtime_component_key(key))
        )
        if unexpected_keys:
            raise ValueError(
                "Runtime-backbone manifest declares tensors absent from the active model: "
                f"{unexpected_keys[:20]}."
            )
        if complete_artifact:
            required_target_keys = {
                key
                for key in target_state
                if not is_allowed_runtime_missing_key(
                    key,
                    allow_random_action=random_action_init,
                )
            }
            missing_required_keys = sorted(required_target_keys - declared_keys)
            if missing_required_keys:
                raise ValueError(
                    "Complete runtime-backbone manifest omits tensors required by "
                    f"the active model: {missing_required_keys[:20]}."
                )
        if not complete_artifact:
            topology_resolver = getattr(replica_core, "component_topology", None)
            if not callable(topology_resolver):
                raise TypeError(
                    "A scoped runtime-backbone artifact requires the active visual "
                    "core to expose component_topology()."
                )
            selected_components = resolve_visual_components(
                topology_resolver(),
                manifest.components,
            )
            component_keys = set(
                owned_state_dict_keys(
                    replica_core,
                    selected_components,
                    available_keys=target_state,
                )
            )
            if declared_keys != component_keys:
                raise ValueError(
                    "Runtime-backbone manifest tensor inventory does not match its "
                    "declared semantic components: "
                    f"missing={sorted(component_keys - declared_keys)[:20]}, "
                    f"outside_components={sorted(declared_keys - component_keys)[:20]}."
                )
    loaded_keys: list[str] = []
    missing_reference_keys: list[str] = []
    for target_key, target_value in target_state.items():
        if random_action_init and is_action_runtime_target_key(target_key):
            missing_reference_keys.append(target_key)
            continue
        source_value = exported_state.get(target_key)
        if source_value is None:
            if complete_artifact:
                missing_reference_keys.append(target_key)
            continue
        source_value = source_value.detach()
        if tuple(source_value.shape) != tuple(target_value.shape):
            if manifest is not None:
                raise ValueError(
                    "Runtime-backbone manifest tensor shape does not match the active model: "
                    f"key={target_key!r}, artifact={tuple(source_value.shape)}, "
                    f"model={tuple(target_value.shape)}."
                )
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
