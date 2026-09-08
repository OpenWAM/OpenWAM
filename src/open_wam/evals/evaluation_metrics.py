"""Tensor selection, alignment, and metrics for generic evaluation."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch

from open_wam.configs import EvalPredictionSource


def _masked_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> float:
    squared_error = (predicted.float() - target.float()).pow(2)
    if action_mask is not None:
        squared_error = squared_error * action_mask.float()
        denom = action_mask.float().sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(squared_error.numel()), device=squared_error.device)
    return float((squared_error.sum() / denom).item())


def _video_latent_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    squared_error = (predicted.float() - target.float()).pow(2)
    return float(squared_error.mean().item())


def _select_eval_action_prediction(
    *,
    target_actions: torch.Tensor,
    decoder_action_pred: torch.Tensor,
    policy_aux: dict[str, Any],
) -> tuple[EvalPredictionSource, torch.Tensor]:
    if decoder_action_pred.shape == target_actions.shape:
        return EvalPredictionSource.DECODER_ACTION_PRED, decoder_action_pred
    raw_chunk_action_pred = policy_aux.get("raw_chunk_action_pred")
    if (
        isinstance(raw_chunk_action_pred, torch.Tensor)
        and raw_chunk_action_pred.ndim == target_actions.ndim
        and raw_chunk_action_pred.shape[0] == target_actions.shape[0]
        and raw_chunk_action_pred.shape[-1] == target_actions.shape[-1]
        and target_actions.shape[1] >= raw_chunk_action_pred.shape[1]
    ):
        return EvalPredictionSource.RAW_CHUNK_ACTION_PRED, raw_chunk_action_pred
    return EvalPredictionSource.DECODER_ACTION_PRED_UNMATCHED, decoder_action_pred


def _align_eval_action_tensors(
    *,
    source: EvalPredictionSource,
    prediction: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> tuple[EvalPredictionSource, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if prediction.shape == target_actions.shape:
        return source, prediction, target_actions, action_mask
    if (
        source == EvalPredictionSource.RAW_CHUNK_ACTION_PRED
        and prediction.ndim == target_actions.ndim
        and prediction.shape[0] == target_actions.shape[0]
        and prediction.shape[-1] == target_actions.shape[-1]
        and target_actions.shape[1] >= prediction.shape[1]
    ):
        target_start = int(target_actions.shape[1] - prediction.shape[1])
        aligned_target = target_actions[:, target_start:]
        aligned_mask = None if action_mask is None else action_mask[:, target_start:]
        return EvalPredictionSource.RAW_CHUNK_ACTION_PRED_TAIL_ALIGNED, prediction, aligned_target, aligned_mask
    return source, prediction, target_actions, action_mask


def _align_eval_action_tensors_by_generation_frame(
    *,
    prediction: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    generation_frame_start: int | None,
    action_tokens_per_frame: int,
) -> tuple[EvalPredictionSource, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Score the generated chunk at its reported frame, never a guessed tail.

    The target slice starts at frame_start * tokens_per_frame. A window can
    contain an incomplete trailing chunk, so its tail is not that same slice.
    Unsupported geometry returns None rather than inventing a valid target.
    """
    if (
        isinstance(generation_frame_start, bool)
        or not isinstance(generation_frame_start, Integral)
        or isinstance(action_tokens_per_frame, bool)
        or not isinstance(action_tokens_per_frame, Integral)
        or action_tokens_per_frame <= 0
        or generation_frame_start < 0
        or prediction.ndim != 3
        or target_actions.ndim != 3
        or prediction.shape[0] != target_actions.shape[0]
        or prediction.shape[2] != target_actions.shape[2]
    ):
        return None
    start = int(generation_frame_start) * int(action_tokens_per_frame)
    horizon = int(prediction.shape[1])
    if horizon <= 0 or start + horizon > int(target_actions.shape[1]):
        return None
    # The metric denominator counts mask elements; accepting a broadcast-only
    # channel mask would multiply the MSE by the number of action channels.
    if action_mask is not None and action_mask.shape != target_actions.shape:
        return None
    return (
        EvalPredictionSource.GENERATED_CHUNK_FRAME_ALIGNED,
        prediction,
        target_actions[:, start : start + horizon],
        None if action_mask is None else action_mask[:, start : start + horizon],
    )


def _select_rollout_previous_action(
    *,
    decoder_action_pred: torch.Tensor,
    policy_aux: dict[str, Any],
) -> torch.Tensor:
    """Return the model-facing action tensor to feed into the next rollout step."""

    chunk_action_pred = policy_aux.get("chunk_action_pred")
    if isinstance(chunk_action_pred, torch.Tensor) and chunk_action_pred.ndim == 3:
        return chunk_action_pred
    return decoder_action_pred


def _select_eval_video_prediction(
    *,
    target_video_latents: torch.Tensor,
    decoder_aux: dict[str, Any],
    policy_aux: dict[str, Any],
) -> tuple[EvalPredictionSource, torch.Tensor | None, torch.Tensor]:
    for source_name in ("predicted_latents", "predicted_video_latents"):
        candidate = decoder_aux.get(source_name)
        if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
            return (
                EvalPredictionSource.DECODER_PREDICTED_LATENTS
                if source_name == "predicted_latents"
                else EvalPredictionSource.DECODER_PREDICTED_VIDEO_LATENTS,
                candidate,
                target_video_latents,
            )
    for source_name in ("predicted_latents", "predicted_video_latents"):
        candidate = policy_aux.get(source_name)
        if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
            return (
                EvalPredictionSource.POLICY_PREDICTED_LATENTS
                if source_name == "predicted_latents"
                else EvalPredictionSource.POLICY_PREDICTED_VIDEO_LATENTS,
                candidate,
                target_video_latents,
            )
    return EvalPredictionSource.UNAVAILABLE, None, target_video_latents


__all__: list[str] = []
