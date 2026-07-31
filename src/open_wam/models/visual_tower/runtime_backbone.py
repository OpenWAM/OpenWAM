"""Runtime-backbone loading, access validation, and compatibility operations."""

from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import (
    BackboneImplementation,
    ExportedRuntimeActionInitMode,
)
from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    normalize_backbone_implementation,
)

from .exported_runtime_backbone import (
    is_allowed_runtime_missing_key,
    is_open_wam_exported_runtime_backbone_dir,
    load_exported_runtime_backbone_into_replica_core,
    resolve_runtime_backbone_dir,
)
from .reference_core_weights import (
    BackboneLoadReport,
    load_reference_weights_into_replica_core,
)
from .reference_transformer import preferred_reference_dtype


def initialize_runtime_backbone(
    *,
    current_report: BackboneLoadReport | None,
    core: nn.Module,
    config: SharedVideoTransformerConfig,
    action_dim: int | None,
) -> BackboneLoadReport | None:
    """Load the configured reference or exported weights once."""

    if current_report is not None:
        return current_report
    if config.pretrained_model_name_or_path is None:
        return None

    runtime_backbone_dir = resolve_runtime_backbone_dir(config)
    is_exported_runtime_dir = is_open_wam_exported_runtime_backbone_dir(
        runtime_backbone_dir
    )
    print(
        "[runtime_backbone_load] "
        f"resolved_dir={runtime_backbone_dir} "
        f"is_exported_runtime_dir={is_exported_runtime_dir}",
        flush=True,
    )
    if is_exported_runtime_dir:
        report = load_exported_runtime_backbone_into_replica_core(
            core,
            backbone_config=config,
        )
        print(
            "[runtime_backbone_load] "
            f"mode=exported_runtime loaded_keys={len(report.loaded_keys)} "
            f"missing_keys={len(report.missing_reference_keys)}",
            flush=True,
        )
        log_runtime_backbone_missing_keys(report, config=config)
        return report

    report = load_reference_weights_into_replica_core(
        core,
        backbone_config=config,
        action_dim=action_dim,
    )
    print(
        "[runtime_backbone_load] "
        f"mode=reference loaded_keys={len(report.loaded_keys)} "
        f"missing_keys={len(report.missing_reference_keys)}",
        flush=True,
    )
    log_runtime_backbone_missing_keys(report, config=config)
    return report


def log_runtime_backbone_missing_keys(
    report: BackboneLoadReport | None,
    *,
    config: SharedVideoTransformerConfig,
) -> None:
    """Report allowed and unexpected gaps in a runtime-backbone export."""

    if report is None or not report.missing_reference_keys:
        return
    allow_random_action = (
        config.exported_runtime_action_init_mode
        == ExportedRuntimeActionInitMode.RANDOM
    )
    allowed = tuple(
        key
        for key in report.missing_reference_keys
        if is_allowed_runtime_missing_key(
            key,
            allow_random_action=allow_random_action,
        )
    )
    unexpected = tuple(
        key
        for key in report.missing_reference_keys
        if not is_allowed_runtime_missing_key(
            key,
            allow_random_action=allow_random_action,
        )
    )
    if allowed:
        print(
            "[runtime_backbone_load] "
            f"allowed_missing_keys={list(allowed)}",
            flush=True,
        )
    if unexpected:
        preview = list(unexpected[:20])
        print(
            "[runtime_backbone_load] "
            f"unexpected_missing_keys_count={len(unexpected)} "
            f"unexpected_missing_keys_preview={preview}",
            flush=True,
        )


def validate_runtime_backbone_request(
    *,
    config: SharedVideoTransformerConfig,
    configured_action_dim: int | None,
    requested_action_dim: int,
) -> None:
    """Validate access to a tower-owned shared runtime backbone."""

    if (
        normalize_backbone_implementation(config.implementation)
        != BackboneImplementation.SHARED_TRANSFORMER
    ):
        raise ValueError(
            "Runtime backbone access requires "
            "`backbone.implementation = shared_transformer`."
        )
    if configured_action_dim is None:
        raise ValueError(
            "VisualTower runtime backbone access requires a configured action_dim."
        )
    if int(requested_action_dim) != int(configured_action_dim):
        raise ValueError(
            "Shared video-transformer backbone was constructed for a different "
            f"action_dim, requested={requested_action_dim}, "
            f"tower_action_dim={configured_action_dim}."
        )


def ensure_runtime_module_device(
    runtime_backbone: nn.Module,
    *,
    device,
) -> nn.Module:
    """Move a runtime backbone only when its device or floating dtype differs."""

    device = torch.device(device)
    target_dtype = preferred_reference_dtype(device)
    needs_move = False
    for parameter in runtime_backbone.parameters():
        if parameter.device != device:
            needs_move = True
            break
        if parameter.is_floating_point() and parameter.dtype != target_dtype:
            needs_move = True
            break
    if not needs_move:
        for buffer in runtime_backbone.buffers():
            if buffer.device != device:
                needs_move = True
                break
            if buffer.is_floating_point() and buffer.dtype != target_dtype:
                needs_move = True
                break
    if needs_move:
        runtime_backbone.to(device=device, dtype=target_dtype)
    return runtime_backbone


def reset_runtime_module_cache(
    runtime_backbone: nn.Module,
    *,
    cache_name: str,
) -> None:
    """Clear current and legacy named runtime caches when implemented."""

    try:
        runtime_backbone.clear_runtime_prediction_cache(cache_name)
    except KeyError:
        pass
    except AttributeError:
        try:
            runtime_backbone.clear_pred_cache(cache_name)
        except KeyError:
            pass
    try:
        runtime_backbone.clear_runtime_cache_state(cache_name)
    except KeyError:
        pass
    except AttributeError:
        try:
            runtime_backbone.clear_cache(cache_name)
        except KeyError:
            pass
