from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .types import FdmAblationMode, FdmWindowSelection


def latent_mse_per_frame(predicted: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Compute video-latent MSE for each frame in `[B, C, F, H, W]` tensors."""

    if predicted.shape != target.shape:
        raise ValueError(f"Latent tensors must have identical shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}.")
    if predicted.ndim != 5:
        raise ValueError(f"Expected latent tensors shaped [B, C, F, H, W], got {tuple(predicted.shape)}.")
    error = (predicted.float() - target.float()).square().mean(dim=(0, 1, 3, 4))
    return [float(value) for value in error.detach().cpu()]


def action_mse_per_frame(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    action_per_frame: int,
) -> list[float]:
    """Compute action MSE per latent frame from `[B, F * A, D]` tensors."""

    if predicted.shape != target.shape:
        raise ValueError(f"Action tensors must have identical shapes, got {tuple(predicted.shape)} and {tuple(target.shape)}.")
    if predicted.ndim != 3:
        raise ValueError(f"Expected action tensors shaped [B, T, D], got {tuple(predicted.shape)}.")
    action_per_frame = int(action_per_frame)
    if action_per_frame <= 0:
        raise ValueError(f"action_per_frame must be positive, got {action_per_frame}.")
    if predicted.shape[1] % action_per_frame != 0:
        raise ValueError(
            "Action horizon must be divisible by action_per_frame, "
            f"got horizon={predicted.shape[1]}, action_per_frame={action_per_frame}."
        )
    frame_count = int(predicted.shape[1] // action_per_frame)
    pred = predicted.float().reshape(predicted.shape[0], frame_count, action_per_frame, predicted.shape[2])
    tgt = target.float().reshape(target.shape[0], frame_count, action_per_frame, target.shape[2])
    error = (pred - tgt).square().mean(dim=(0, 2, 3))
    return [float(value) for value in error.detach().cpu()]


def rgb_mse_per_frame(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Compute RGB MSE for `[F, H, W, C]` arrays in uint8 or `[0, 1]` float."""

    predicted_f = _as_unit_float(predicted)
    target_f = _as_unit_float(target)
    if predicted_f.shape != target_f.shape:
        raise ValueError(f"RGB arrays must have identical shapes, got {predicted_f.shape} and {target_f.shape}.")
    if predicted_f.ndim != 4:
        raise ValueError(f"Expected RGB arrays shaped [F, H, W, C], got {predicted_f.shape}.")
    return [float(value) for value in np.mean(np.square(predicted_f - target_f), axis=(1, 2, 3))]


def psnr_from_mse(mse: float, *, max_value: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(max_value) - 10.0 * math.log10(mse))


def simple_ssim_per_frame(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Small dependency-free global SSIM estimate for debugging trends."""

    predicted_f = _as_unit_float(predicted)
    target_f = _as_unit_float(target)
    if predicted_f.shape != target_f.shape:
        raise ValueError(f"RGB arrays must have identical shapes, got {predicted_f.shape} and {target_f.shape}.")
    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []
    for pred_frame, target_frame in zip(predicted_f, target_f, strict=True):
        pred = pred_frame.reshape(-1, pred_frame.shape[-1])
        tgt = target_frame.reshape(-1, target_frame.shape[-1])
        mu_x = pred.mean(axis=0)
        mu_y = tgt.mean(axis=0)
        var_x = pred.var(axis=0)
        var_y = tgt.var(axis=0)
        cov_xy = ((pred - mu_x) * (tgt - mu_y)).mean(axis=0)
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        values.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return values


def build_metric_rows(
    *,
    selection: FdmWindowSelection,
    mode: FdmAblationMode,
    latent_mse: list[float] | None = None,
    rgb_mse: list[float] | None = None,
    rgb_ssim: list[float] | None = None,
    action_mse: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lengths = [
        len(values)
        for values in (latent_mse, rgb_mse, rgb_ssim, action_mse)
        if values is not None
    ]
    if not lengths:
        raise ValueError("At least one metric sequence is required.")
    if len(set(lengths)) != 1:
        raise ValueError(f"Metric sequences must have equal lengths, got {lengths}.")
    for horizon_index in range(lengths[0]):
        latent_value = None if latent_mse is None else latent_mse[horizon_index]
        rgb_value = None if rgb_mse is None else rgb_mse[horizon_index]
        action_value = None if action_mse is None else action_mse[horizon_index]
        row = {
            **asdict(selection),
            "mode": mode.value,
            "horizon_index": horizon_index,
            "future_frame": selection.target_start_frame + horizon_index,
            "latent_mse": None if latent_value is None else float(latent_value),
            "rgb_mse": rgb_value,
            "rgb_psnr": None if rgb_value is None else psnr_from_mse(float(rgb_value)),
            "rgb_ssim": None if rgb_ssim is None else float(rgb_ssim[horizon_index]),
            "action_mse": None if action_value is None else float(action_value),
        }
        rows.append(row)
    return rows


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["horizon_index"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (mode, horizon_index), group_rows in sorted(groups.items()):
        latent_values = np.asarray(
            [float(row["latent_mse"]) for row in group_rows if row.get("latent_mse") is not None],
            dtype=np.float64,
        )
        rgb_values = np.asarray(
            [float(row["rgb_mse"]) for row in group_rows if row.get("rgb_mse") is not None],
            dtype=np.float64,
        )
        action_values = np.asarray(
            [float(row["action_mse"]) for row in group_rows if row.get("action_mse") is not None],
            dtype=np.float64,
        )
        summary = {
            "mode": mode,
            "horizon_index": horizon_index,
            "count": len(group_rows),
            "latent_mse_mean": None if latent_values.size == 0 else float(latent_values.mean()),
            "latent_mse_std": None if latent_values.size == 0 else float(latent_values.std(ddof=0)),
            "rgb_mse_mean": None if rgb_values.size == 0 else float(rgb_values.mean()),
            "rgb_mse_std": None if rgb_values.size == 0 else float(rgb_values.std(ddof=0)),
        }
        if action_values.size > 0:
            summary["action_mse_mean"] = float(action_values.mean())
            summary["action_mse_std"] = float(action_values.std(ddof=0))
        summaries.append(summary)
    return summaries


def _as_unit_float(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype == np.uint8:
        return value.astype(np.float32) / 255.0
    value = value.astype(np.float32)
    if value.size and float(np.nanmax(value)) > 1.0001:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)
