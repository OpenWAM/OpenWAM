"""Tensor selection, alignment, and metrics for generic evaluation."""

from __future__ import annotations

import torch

from open_wam.configs import EvalPredictionSource
from open_wam.contracts.action_space import ActionSpaceAdapter
from open_wam.models.policy_variants import PolicyGeneratedVideo


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
    action_adapter: ActionSpaceAdapter | None,
) -> tuple[EvalPredictionSource, torch.Tensor]:
    if decoder_action_pred.shape == target_actions.shape:
        return EvalPredictionSource.DECODER_ACTION_PRED, decoder_action_pred
    raw_chunk_action_pred = (
        action_adapter.to_source(decoder_action_pred)
        if action_adapter is not None else None
    )
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


def _select_eval_video_prediction(
    *,
    target_video_latents: torch.Tensor,
    generated_video: PolicyGeneratedVideo | None,
) -> tuple[EvalPredictionSource, torch.Tensor | None, torch.Tensor]:
    if generated_video is not None:
        candidate = generated_video.latents
        start = generated_video.frame_start
        if start is not None and start > 0 and start + candidate.shape[2] == target_video_latents.shape[2]:
            # Preserve full-window scoring: the observed prefix has zero error.
            candidate = torch.cat([target_video_latents[:, :, :start], candidate], dim=2)
        if candidate.shape == target_video_latents.shape:
            return EvalPredictionSource.POLICY_PREDICTED_VIDEO_LATENTS, candidate, target_video_latents
    return EvalPredictionSource.UNAVAILABLE, None, target_video_latents


__all__: list[str] = []
