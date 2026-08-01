"""Tensor selection, alignment, and metrics for generic evaluation."""

from __future__ import annotations

from typing import Any

import torch

from open_wam.configs import EvalPredictionSource
from open_wam.models.policy_variants.contracts import DecoderSequenceContext


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
    sequence_context: DecoderSequenceContext | None = None,
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
        aligned_target = _align_local_future_video_prediction(
            candidate,
            target_video_latents=target_video_latents,
            sequence_context=sequence_context,
        )
        if aligned_target is not None:
            return EvalPredictionSource.DECODER_PREDICTED_LOCAL_FUTURE_LATENTS, candidate, aligned_target
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
        aligned_target = _align_local_future_video_prediction(
            candidate,
            target_video_latents=target_video_latents,
            sequence_context=sequence_context,
        )
        if aligned_target is not None:
            return EvalPredictionSource.POLICY_PREDICTED_LOCAL_FUTURE_LATENTS, candidate, aligned_target
    return EvalPredictionSource.UNAVAILABLE, None, target_video_latents


def _align_local_future_video_prediction(
    candidate: Any,
    *,
    target_video_latents: torch.Tensor,
    sequence_context: DecoderSequenceContext | None,
) -> torch.Tensor | None:
    if not isinstance(candidate, torch.Tensor):
        return None
    if candidate.ndim != target_video_latents.ndim or candidate.ndim != 5:
        return None
    if candidate.shape[0:2] != target_video_latents.shape[0:2] or candidate.shape[3:] != target_video_latents.shape[3:]:
        return None
    if sequence_context is None or sequence_context.video_condition_window is None:
        return None
    window = sequence_context.video_condition_window
    metadata = window.metadata
    if metadata.get("source_family") != "generated_future_video_tokens":
        return None
    observed_frames = int(metadata.get("observed_prefix_frames", window.observed_frame_count))
    observed_start = int(metadata.get("observed_prefix_start_index", 0))
    target_start = observed_start + observed_frames
    target_end = target_start + int(candidate.shape[2])
    if target_end > int(target_video_latents.shape[2]):
        return None
    return target_video_latents[:, :, target_start:target_end]


__all__: list[str] = []
