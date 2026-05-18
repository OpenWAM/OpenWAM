from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionMappingLossMaskMode,
    ActionMappingMode,
    ActionMappingSamplerMaskMode,
    ActionNormalizationMode,
)


@dataclass(frozen=True)
class ActionMappingResult:
    """Mapped action sequence plus loss mask and audit metadata."""

    actions: torch.Tensor
    action_mask: torch.Tensor
    metadata: dict[str, Any]
    sampler_mask: torch.Tensor | None = None


def action_mapping_is_active(config: ActionMappingConfig) -> bool:
    """Return whether one mapping config changes the action representation."""

    return config.mode != ActionMappingMode.NONE


def resolve_action_source_dim(config: ActionMappingConfig, fallback_dim: int) -> int:
    """Return the source dim a loader should extract before action mapping."""

    if not action_mapping_is_active(config):
        return int(fallback_dim)
    if config.source_dim is None:
        raise ValueError("Active action mapping requires `source_dim`.")
    return int(config.source_dim)


def resolve_action_target_dim(config: ActionMappingConfig, fallback_dim: int) -> int:
    """Return the final model-facing action dim after action mapping."""

    if not action_mapping_is_active(config):
        return int(fallback_dim)
    if config.target_dim is None:
        raise ValueError("Active action mapping requires `target_dim`.")
    return int(config.target_dim)


def apply_action_mapping(
    source_actions: torch.Tensor,
    source_mask: torch.Tensor,
    config: ActionMappingConfig,
    *,
    target_dim: int,
) -> ActionMappingResult:
    """Map one `[H, D_source]` action sequence into the configured target dim."""

    if source_actions.shape != source_mask.shape:
        raise ValueError(
            "Action mapping requires source action and mask shapes to match, "
            f"got actions={tuple(source_actions.shape)}, mask={tuple(source_mask.shape)}."
        )
    if source_actions.ndim != 2:
        raise ValueError(f"Expected source action sequence [H, D], got {tuple(source_actions.shape)}.")

    if not action_mapping_is_active(config):
        if source_actions.shape[-1] != target_dim:
            raise ValueError(
                "Unmapped action sequence dim must match target dim, "
                f"got source={source_actions.shape[-1]}, target={target_dim}."
            )
        return ActionMappingResult(
            actions=source_actions.to(dtype=torch.float32),
            action_mask=source_mask.to(dtype=torch.float32),
            sampler_mask=None,
            metadata={"action_mapping_mode": str(config.mode)},
        )

    source_dim = resolve_action_source_dim(config, fallback_dim=source_actions.shape[-1])
    configured_target_dim = resolve_action_target_dim(config, fallback_dim=target_dim)
    if source_actions.shape[-1] != source_dim:
        raise ValueError(
            f"Action mapping expected source dim {source_dim}, got {source_actions.shape[-1]}."
        )
    if configured_target_dim != target_dim:
        raise ValueError(
            f"Action mapping target_dim={configured_target_dim} must match action_schema.action_dim={target_dim}."
        )
    _validate_normalization_stats_for_mapping(config, source_dim=source_dim, target_dim=target_dim)

    normalized_source = _normalize_source_actions(source_actions.to(dtype=torch.float32), config)
    actions = torch.full(
        (source_actions.shape[0], target_dim),
        fill_value=float(config.inactive_value),
        dtype=torch.float32,
        device=source_actions.device,
    )
    action_mask = torch.zeros_like(actions)

    for source_index, target_index in enumerate(config.source_to_target_indices):
        actions[:, target_index] = normalized_source[:, source_index]
        action_mask[:, target_index] = source_mask[:, source_index].to(dtype=torch.float32)

    if config.loss_mask_mode == ActionMappingLossMaskMode.ACTIVE_TARGET_INDICES and config.active_target_indices:
        active_mask = torch.zeros(target_dim, dtype=torch.float32, device=source_actions.device)
        active_mask[list(config.active_target_indices)] = 1.0
        action_mask = action_mask * active_mask.unsqueeze(0)

    actions = _normalize_target_actions(actions, config)
    actions = _apply_inactive_fill(
        actions,
        action_mask,
        inactive_value=float(config.inactive_value),
    )
    sampler_mask = build_action_sampler_mask(
        config,
        action_horizon=source_actions.shape[0],
        target_dim=target_dim,
        device=source_actions.device,
        dtype=torch.float32,
    )
    metadata = {
        "action_mapping_mode": str(config.mode),
        "action_mapping_source_dim": source_dim,
        "action_mapping_target_dim": target_dim,
        "action_mapping_source_to_target_indices": list(config.source_to_target_indices),
        "action_mapping_active_target_indices": list(_active_target_indices(config)),
        "action_mapping_sampler_mask_mode": str(config.sampler_mask_mode),
        "action_mapping_sampler_active_target_indices": list(_active_target_indices(config))
        if sampler_mask is not None
        else [],
        "action_mapping_inactive_value": float(config.inactive_value),
        "action_mapping_normalization_mode": str(config.normalization.mode),
    }
    return ActionMappingResult(actions=actions, action_mask=action_mask, sampler_mask=sampler_mask, metadata=metadata)


def build_action_sampler_mask(
    config: ActionMappingConfig,
    *,
    action_horizon: int,
    target_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Return an inference-time sampler mask for mapped inactive channels.

    The loss mask remains data-validity oriented. This mask is intentionally
    separate: samplers can pin inactive model-facing channels without changing
    supervised loss accounting for active channels or padded timesteps.
    """

    if not action_mapping_is_active(config) or config.sampler_mask_mode == ActionMappingSamplerMaskMode.NONE:
        return None
    if config.sampler_mask_mode != ActionMappingSamplerMaskMode.PIN_INACTIVE_CHANNELS:
        raise ValueError(f"Unsupported action sampler mask mode {config.sampler_mask_mode!r}.")
    configured_target_dim = resolve_action_target_dim(config, fallback_dim=target_dim)
    if configured_target_dim != target_dim:
        raise ValueError(
            f"Action sampler mask target_dim={configured_target_dim} must match action dim {target_dim}."
        )
    mask = torch.zeros(int(action_horizon), int(target_dim), device=device, dtype=dtype)
    active_indices = list(_active_target_indices(config))
    if active_indices:
        mask[:, active_indices] = 1.0
    return mask


def inverse_action_mapping(
    mapped_actions: torch.Tensor,
    config: ActionMappingConfig,
) -> torch.Tensor:
    """Recover source action channels from a mapped target sequence or batch."""

    if not action_mapping_is_active(config):
        return mapped_actions
    source_dim = resolve_action_source_dim(config, fallback_dim=mapped_actions.shape[-1])
    target_dim = resolve_action_target_dim(config, fallback_dim=mapped_actions.shape[-1])
    if mapped_actions.shape[-1] != target_dim:
        raise ValueError(
            f"Mapped action tensor last dim must be {target_dim}, got {mapped_actions.shape[-1]}."
        )
    _validate_normalization_stats_for_mapping(config, source_dim=source_dim, target_dim=target_dim)
    source = mapped_actions.new_empty(*mapped_actions.shape[:-1], source_dim)
    denormalized = _denormalize_target_actions(mapped_actions.to(dtype=torch.float32), config)
    for source_index, target_index in enumerate(config.source_to_target_indices):
        source[..., source_index] = denormalized[..., target_index]
    return _denormalize_source_actions(source, config).to(dtype=mapped_actions.dtype)


def validate_action_mapping_preflight(
    config: ActionMappingConfig,
    *,
    action_schema_dim: int,
) -> dict[str, Any]:
    """Validate shape invariants without needing dataset rows."""

    if not action_mapping_is_active(config):
        return {"action_mapping_mode": str(config.mode), "action_dim": int(action_schema_dim)}
    target_dim = resolve_action_target_dim(config, fallback_dim=action_schema_dim)
    source_dim = resolve_action_source_dim(config, fallback_dim=action_schema_dim)
    if target_dim != action_schema_dim:
        raise ValueError(
            f"Action mapping target_dim={target_dim} must equal action_schema.action_dim={action_schema_dim}."
        )
    _validate_normalization_stats_for_mapping(config, source_dim=source_dim, target_dim=target_dim)
    probe = torch.arange(source_dim, dtype=torch.float32).reshape(1, source_dim)
    probe_mask = torch.ones_like(probe)
    mapped = apply_action_mapping(probe, probe_mask, config, target_dim=action_schema_dim)
    recovered = inverse_action_mapping(mapped.actions, config)
    if not torch.allclose(recovered, probe):
        raise ValueError("Action mapping inverse did not recover the active source channels.")
    inactive_indices = sorted(set(range(action_schema_dim)) - set(_active_target_indices(config)))
    if inactive_indices:
        inactive_values = mapped.actions[:, inactive_indices]
        expected_inactive = torch.full_like(inactive_values, fill_value=float(config.inactive_value))
        if not torch.allclose(inactive_values, expected_inactive):
            raise ValueError("Inactive mapped action channels must stay at `inactive_value` after preflight mapping.")
        inactive_mask = mapped.action_mask[:, inactive_indices]
        if not torch.allclose(inactive_mask, torch.zeros_like(inactive_mask)):
            raise ValueError("Inactive mapped action channels must be masked out.")
    return {
        "action_mapping_mode": str(config.mode),
        "source_dim": source_dim,
        "target_dim": target_dim,
        "active_channels": list(_active_target_indices(config)),
        "inactive_channel_count": len(inactive_indices),
    }


def _validate_normalization_stats_for_mapping(
    config: ActionMappingConfig,
    *,
    source_dim: int,
    target_dim: int,
) -> None:
    stat_dim = _normalization_stats_dim(config)
    if stat_dim is None:
        return
    if int(source_dim) == int(target_dim):
        raise ValueError(
            "Normalized action mappings with source_dim == target_dim are ambiguous because the same "
            "stats length could mean source-space or target-space normalization. Use an unmapped action "
            "target normalization or disable action_mapping normalization until the normalization space is explicit."
        )
    valid_dims = {int(source_dim), int(target_dim)}
    if int(stat_dim) not in valid_dims:
        raise ValueError(
            "Action mapping normalization stats length must match either source_dim or target_dim, "
            f"got stats_dim={stat_dim}, source_dim={source_dim}, target_dim={target_dim}."
        )


def _normalization_stats_dim(config: ActionMappingConfig) -> int | None:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.NONE:
        return None
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        return len(normalization.q01)
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        return len(normalization.lower)
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        return len(normalization.mean)
    raise ValueError(f"Unsupported action normalization mode {normalization.mode!r}.")


def _active_target_indices(config: ActionMappingConfig) -> tuple[int, ...]:
    if config.active_target_indices:
        return tuple(int(value) for value in config.active_target_indices)
    return tuple(int(value) for value in config.source_to_target_indices)


def _apply_inactive_fill(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    inactive_value: float,
) -> torch.Tensor:
    valid_mask = (action_mask > 0).to(actions.dtype)
    fill = actions.new_full((), float(inactive_value))
    return actions * valid_mask + fill * (1.0 - valid_mask)


def _normalize_source_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    normalized = _normalize_actions(actions, config)
    return _clip_if_requested(normalized, config)


def _denormalize_source_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    return _denormalize_actions(actions, config)


def _normalize_target_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    normalized = _normalize_actions(actions, config)
    return _clip_if_requested(normalized, config)


def _denormalize_target_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    return _denormalize_actions(actions, config)


def _normalize_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        q01 = _quantile_tensor(normalization.q01, device=actions.device, dtype=actions.dtype)
        q99 = _quantile_tensor(normalization.q99, device=actions.device, dtype=actions.dtype)
        if q01.numel() != actions.shape[-1]:
            return actions
        return _normalize_by_quantiles(actions, q01=q01, q99=q99)
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _limit_tensors(config, device=actions.device, dtype=actions.dtype)
        if lower.numel() != actions.shape[-1]:
            return actions
        return _normalize_by_limits(actions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_tensors(config, device=actions.device, dtype=actions.dtype)
        if mean.numel() != actions.shape[-1]:
            return actions
        return (actions - mean) / std.clamp_min(1e-6)
    raise ValueError(f"Unsupported action normalization mode {normalization.mode!r}.")


def _denormalize_actions(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    normalization = config.normalization
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        q01 = _quantile_tensor(normalization.q01, device=actions.device, dtype=actions.dtype)
        q99 = _quantile_tensor(normalization.q99, device=actions.device, dtype=actions.dtype)
        if q01.numel() != actions.shape[-1]:
            return actions
        return _denormalize_by_quantiles(actions, q01=q01, q99=q99)
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _limit_tensors(config, device=actions.device, dtype=actions.dtype)
        if lower.numel() != actions.shape[-1]:
            return actions
        return _denormalize_by_limits(actions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_tensors(config, device=actions.device, dtype=actions.dtype)
        if mean.numel() != actions.shape[-1]:
            return actions
        return actions * std.clamp_min(1e-6) + mean
    raise ValueError(f"Unsupported action normalization mode {normalization.mode!r}.")


def _limit_tensors(
    config: ActionMappingConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.tensor(config.normalization.lower, dtype=dtype, device=device)
    upper = torch.tensor(config.normalization.upper, dtype=dtype, device=device)
    return lower, upper


def _gaussian_tensors(
    config: ActionMappingConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(config.normalization.mean, dtype=dtype, device=device)
    std = torch.tensor(config.normalization.std, dtype=dtype, device=device)
    return mean, std


def _normalize_by_limits(actions: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return (actions - center) / scale


def _denormalize_by_limits(actions: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return actions * scale + center


def _quantile_tensor(values: tuple[float, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _normalize_by_quantiles(actions: torch.Tensor, *, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    center = (q99 + q01) * 0.5
    scale = (q99 - q01).clamp_min(1e-6) * 0.5
    return (actions - center) / scale


def _denormalize_by_quantiles(actions: torch.Tensor, *, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    center = (q99 + q01) * 0.5
    scale = (q99 - q01).clamp_min(1e-6) * 0.5
    return actions * scale + center


def _clip_if_requested(actions: torch.Tensor, config: ActionMappingConfig) -> torch.Tensor:
    clip_min = config.normalization.clip_min
    clip_max = config.normalization.clip_max
    if clip_min is None and clip_max is None:
        return actions
    min_value = -torch.inf if clip_min is None else float(clip_min)
    max_value = torch.inf if clip_max is None else float(clip_max)
    return actions.clamp(min=min_value, max=max_value)
