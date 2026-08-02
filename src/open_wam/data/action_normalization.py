"""Invertible normalization for public action and joint targets."""

from __future__ import annotations

import torch

from open_wam.configs import ActionNormalizationConfig, ActionNormalizationMode


def normalize_joint_positions(
    joint_positions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Normalize joint-position channels using a configured numeric contract."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return joint_positions
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, joint_positions)
        normalized = normalize_joint_positions_by_limits(joint_positions, lower=lower, upper=upper)
    elif normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, joint_positions)
        normalized = normalize_joint_positions_by_limits(joint_positions, lower=lower, upper=upper)
    else:
        raise ValueError(f"Unsupported joint-position normalization mode: {normalization.mode}")

    if normalization.clip_min is not None or normalization.clip_max is not None:
        min_value = -torch.inf if normalization.clip_min is None else float(normalization.clip_min)
        max_value = torch.inf if normalization.clip_max is None else float(normalization.clip_max)
        normalized = normalized.clamp(min=min_value, max=max_value)
    return normalized


def denormalize_joint_positions(
    normalized_joint_positions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Invert joint-position normalization for rollout adapters."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return normalized_joint_positions
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, normalized_joint_positions)
        return denormalize_joint_positions_by_limits(normalized_joint_positions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, normalized_joint_positions)
        return denormalize_joint_positions_by_limits(normalized_joint_positions, lower=lower, upper=upper)
    raise ValueError(f"Unsupported joint-position normalization mode: {normalization.mode}")


def normalize_action_targets(
    actions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Normalize one final action-target tensor with an invertible contract."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_stats(normalization, actions)
        normalized = (actions - mean) / std.clamp_min(1e-6)
    elif normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, actions)
        normalized = normalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    elif normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, actions)
        normalized = normalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    else:
        raise ValueError(f"Unsupported action-target normalization mode: {normalization.mode}")

    if normalization.clip_min is not None or normalization.clip_max is not None:
        min_value = -torch.inf if normalization.clip_min is None else float(normalization.clip_min)
        max_value = torch.inf if normalization.clip_max is None else float(normalization.clip_max)
        normalized = normalized.clamp(min=min_value, max=max_value)
    return normalized


def denormalize_action_targets(
    actions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Invert `normalize_action_targets` for rollout adapters."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_stats(normalization, actions)
        return actions * std.clamp_min(1e-6) + mean
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, actions)
        return denormalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, actions)
        return denormalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    raise ValueError(f"Unsupported action-target normalization mode: {normalization.mode}")


def normalize_joint_positions_by_limits(
    joint_positions: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map absolute joint positions from configured limits to roughly `[-1, 1]`."""

    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return (joint_positions - center) / scale


def denormalize_joint_positions_by_limits(
    normalized_joint_positions: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map normalized joint-position channels back to physical joint units."""

    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return normalized_joint_positions * scale + center


def _gaussian_stats(
    normalization: ActionNormalizationConfig,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(normalization.mean, dtype=reference.dtype, device=reference.device)
    std = torch.as_tensor(normalization.std, dtype=reference.dtype, device=reference.device)
    if mean.numel() != reference.shape[-1] or std.numel() != reference.shape[-1]:
        raise ValueError(
            "Gaussian action-target normalization stats must match the last action dimension, "
            f"got mean={mean.numel()}, std={std.numel()}, action_dim={reference.shape[-1]}."
        )
    return mean, std


def _normalization_bounds(
    normalization: ActionNormalizationConfig,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.tensor(normalization.lower, dtype=tensor.dtype, device=tensor.device)
    upper = torch.tensor(normalization.upper, dtype=tensor.dtype, device=tensor.device)
    if lower.numel() != tensor.shape[-1] or upper.numel() != tensor.shape[-1]:
        raise ValueError(
            "Joint-limit normalization bounds must match joint dimension, "
            f"got lower={lower.numel()}, upper={upper.numel()}, dim={tensor.shape[-1]}."
        )
    return lower, upper


def _quantile_bounds(
    normalization: ActionNormalizationConfig,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q01 = torch.tensor(normalization.q01, dtype=tensor.dtype, device=tensor.device)
    q99 = torch.tensor(normalization.q99, dtype=tensor.dtype, device=tensor.device)
    if q01.numel() != tensor.shape[-1] or q99.numel() != tensor.shape[-1]:
        raise ValueError(
            "Quantile normalization bounds must match joint dimension, "
            f"got q01={q01.numel()}, q99={q99.numel()}, dim={tensor.shape[-1]}."
        )
    return q01, q99


__all__ = [
    "denormalize_action_targets",
    "denormalize_joint_positions",
    "denormalize_joint_positions_by_limits",
    "normalize_action_targets",
    "normalize_joint_positions",
    "normalize_joint_positions_by_limits",
]
