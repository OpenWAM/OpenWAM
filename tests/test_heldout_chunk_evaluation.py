"""Regression coverage for heldout chunk evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import DataSplit, EvalMode, EvalPredictionSource, ExperimentConfig, LatentTemporalLayout
from open_wam.data import LatentWAMBatch
from open_wam.evals import evaluate
from open_wam.evals.evaluation_contracts import EvaluationRequest
from open_wam.evals.evaluation_metrics import (
    _align_eval_action_tensors_by_generation_frame,
    _masked_action_mse,
)


def _batch(frames=35, slots=None, *, rgb_frames=None):
    slots = frames * 4 if slots is None else slots
    actions = torch.arange(slots * 2, dtype=torch.float32).reshape(1, slots, 2)
    return LatentWAMBatch(
        video_latents=torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1),
        actions=actions,
        action_mask=torch.ones_like(actions),
        state=torch.tensor([[[2.0, 3.0]]]),
        canonical_video=(
            None if rgb_frames is None else torch.arange(rgb_frames, dtype=torch.float32).reshape(1, 1, rgb_frames, 1, 1)
        ),
    )


@pytest.mark.parametrize("frames,chunk,observed", [(35, 4, 28), (32, 4, 28), (8, 4, 4), (11, 3, 6), (2, 1, 1)])
def test_holdout_preserves_established_complete_chunk_geometry(frames, chunk, observed):
    batch = _batch(frames)
    original = batch.video_latents.clone()
    latents, rgb, rate = evaluate._prepare_heldout_latent_eval_inputs(
        batch, frame_chunk_size=chunk, latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4
    )
    assert rate == 4
    assert rgb is None
    assert latents.is_contiguous()
    assert torch.equal(latents, original[:, :, :observed])
    assert torch.equal(batch.video_latents, original)


@pytest.mark.parametrize("frames,slots,chunk", [(4, 16, 4), (7, 28, 4), (35, 139, 4), (35, 0, 4), (0, 0, 4), (35, 140, 0)])
def test_short_or_nonuniform_windows_keep_original_fallback(frames, slots, chunk):
    batch = _batch(frames, slots, rgb_frames=frames)
    latents, rgb, rate = evaluate._prepare_heldout_latent_eval_inputs(
        batch, frame_chunk_size=chunk, latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4
    )
    assert latents is batch.video_latents
    assert rgb is batch.canonical_video
    assert rate == 0


@pytest.mark.parametrize("raw_frames,expected", [(35, 28), (137, 109), (140, 109), (70, None), (0, None), (None, None)])
def test_future_rgb_is_trimmed_only_for_known_mappings_otherwise_omitted(raw_frames, expected):
    batch = _batch(rgb_frames=raw_frames)
    original_rgb = None if batch.canonical_video is None else batch.canonical_video.clone()
    _, rgb, rate = evaluate._prepare_heldout_latent_eval_inputs(
        batch, frame_chunk_size=4, latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4
    )
    assert rate == 4
    if expected is None:
        assert rgb is None
    else:
        assert torch.equal(rgb, original_rgb[:, :, :expected])
    if original_rgb is not None:
        assert torch.equal(batch.canonical_video, original_rgb)


def test_frame_alignment_scores_actual_generated_slots_not_window_tail():
    batch = _batch()
    prediction = batch.actions[:, 112:128].clone()
    batch.action_mask[:, 114:116] = 0
    aligned = _align_eval_action_tensors_by_generation_frame(
        prediction=prediction, target_actions=batch.actions, action_mask=batch.action_mask,
        generation_frame_start=28, action_tokens_per_frame=4,
    )
    assert aligned is not None
    source, output, target, mask = aligned
    assert source is EvalPredictionSource.GENERATED_CHUNK_FRAME_ALIGNED
    assert output is prediction
    assert torch.equal(target, batch.actions[:, 112:128])
    assert torch.equal(mask, batch.action_mask[:, 112:128])
    assert _masked_action_mse(output, target, mask) == 0
    assert _masked_action_mse(output, batch.actions[:, -16:], None) > 0


@pytest.mark.parametrize("frame,rate", [(0, 4), (31, 4), (2, 1)])
def test_frame_alignment_accepts_valid_start_and_exact_end(frame, rate):
    batch = _batch()
    aligned = _align_eval_action_tensors_by_generation_frame(
        prediction=batch.actions[:, :16], target_actions=batch.actions, action_mask=None,
        generation_frame_start=frame, action_tokens_per_frame=rate,
    )
    assert aligned is not None
    assert torch.equal(aligned[2], batch.actions[:, frame * rate:frame * rate + 16])
    assert aligned[3] is None


@pytest.mark.parametrize("frame,rate", [(-1, 4), (32, 4), (28, 0), (28, -1), (True, 4), (28, True), (None, 4), (28.5, 4), (28, 4.5)])
def test_invalid_generation_geometry_is_not_scored(frame, rate):
    batch = _batch()
    assert _align_eval_action_tensors_by_generation_frame(
        prediction=batch.actions[:, :16], target_actions=batch.actions, action_mask=batch.action_mask,
        generation_frame_start=frame, action_tokens_per_frame=rate,
    ) is None


@pytest.mark.parametrize("prediction_shape,target_shape,mask_shape", [
    ((16,), (1, 140, 2), None), ((1, 16, 2), (140, 2), None),
    ((2, 16, 2), (1, 140, 2), None), ((1, 16, 3), (1, 140, 2), None),
    ((1, 0, 2), (1, 140, 2), None), ((1, 16, 2), (1, 140, 2), (1, 10, 2)),
    ((1, 16, 2), (1, 140, 2), (1, 140, 3)),
    ((1, 16, 2), (1, 140, 2), (1, 140, 1)),
])
def test_invalid_tensor_geometry_returns_none_before_indexing(prediction_shape, target_shape, mask_shape):
    assert _align_eval_action_tensors_by_generation_frame(
        prediction=torch.zeros(prediction_shape), target_actions=torch.zeros(target_shape),
        action_mask=None if mask_shape is None else torch.ones(mask_shape),
        generation_frame_start=28, action_tokens_per_frame=4,
    ) is None


def _run_mock_batch_eval(monkeypatch, batch, *, prediction, generation_start, aux_start=None):
    config = ExperimentConfig()
    config = replace(config, inference=replace(config.inference, frame_chunk_size=4))
    calls = []

    class Pipeline(torch.nn.Module):
        def forward_infer_step_from_latents(self, latents, context, **kwargs):
            calls.append((latents.clone(), context, kwargs))
            return SimpleNamespace(
                policy_output=SimpleNamespace(
                    generation_frame_start=generation_start,
                    aux={} if aux_start is None else {"generation_frame_start": aux_start},
                ),
                decoder_output=SimpleNamespace(action_pred=prediction, aux={}),
                visual_outputs=SimpleNamespace(frontend=SimpleNamespace(video_latents=latents)),
            )

    monkeypatch.setattr(evaluate, "_resolve_evaluation_runtime", lambda _: (config, None))
    monkeypatch.setattr(evaluate, "build_variant_pipeline_from_config", lambda _: Pipeline())
    monkeypatch.setattr(evaluate, "_build_eval_dataloader", lambda *args, **kwargs: [batch])
    request = EvaluationRequest(
        experiment_config_path=Path("unused.yaml"), mode=EvalMode.BATCH, split=DataSplit.VAL,
        max_batches=1, max_trajectories=None, max_steps_per_trajectory=None,
        batch_size=1, checkpoint_path=None, device="cpu", seed=7,
    )
    return evaluate.run_evaluation(request), calls


@pytest.mark.parametrize("raw_frames,observed_rgb_frames", [(35, 28), (137, 109), (70, None), (None, None)])
def test_batch_eval_holds_out_future_inputs_and_uses_typed_generation_geometry(monkeypatch, raw_frames, observed_rgb_frames):
    batch = _batch(rgb_frames=raw_frames)
    prediction = batch.actions[:, 112:128].clone()
    summary, calls = _run_mock_batch_eval(
        monkeypatch, batch, prediction=prediction, generation_start=28, aux_start=31
    )
    assert summary.mean_action_mse == 0
    assert summary.action_prediction_source is EvalPredictionSource.GENERATED_CHUNK_FRAME_ALIGNED
    assert summary.action_prediction_shape == summary.target_action_shape == (1, 16, 2)
    assert len(calls) == 1
    observed_latents, context, kwargs = calls[0]
    assert torch.equal(observed_latents, batch.video_latents[:, :, :28])
    assert torch.equal(context.state, batch.state)  # Preserve the timestamp-less anchor.
    assert "actions" not in context.extra
    if observed_rgb_frames is None:
        assert kwargs["canonical_video"] is None
    else:
        assert torch.equal(kwargs["canonical_video"], batch.canonical_video[:, :, :observed_rgb_frames])


def test_batch_eval_can_use_legacy_aux_generation_geometry(monkeypatch):
    batch = _batch()
    summary, _ = _run_mock_batch_eval(
        monkeypatch, batch, prediction=batch.actions[:, 112:128], generation_start=None, aux_start=28
    )
    assert summary.mean_action_mse == 0
    assert summary.action_prediction_source is EvalPredictionSource.GENERATED_CHUNK_FRAME_ALIGNED


@pytest.mark.parametrize("generation_start", [None, 0, 27, 32, 35])
def test_heldout_eval_never_scores_observed_or_out_of_range_predictions(monkeypatch, generation_start):
    batch = _batch()
    summary, _ = _run_mock_batch_eval(
        monkeypatch, batch, prediction=batch.actions[:, :16], generation_start=generation_start
    )
    assert summary.mean_action_mse is None
    assert summary.action_prediction_source is EvalPredictionSource.UNAVAILABLE


def test_short_window_batch_eval_preserves_legacy_full_target_fallback(monkeypatch):
    batch = _batch(frames=4, rgb_frames=4)
    summary, calls = _run_mock_batch_eval(
        monkeypatch, batch, prediction=batch.actions.clone(), generation_start=None
    )
    assert summary.mean_action_mse == 0
    assert summary.action_prediction_source is EvalPredictionSource.DECODER_ACTION_PRED
    assert torch.equal(calls[0][0], batch.video_latents)
    assert torch.equal(calls[0][2]["canonical_video"], batch.canonical_video)
