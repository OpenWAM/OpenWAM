"""Exact parity with the formerly decoder-local video reducers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.action_decoders import dual_expert_decoder, video_only_decoder
from open_wam.models.common.flow_supervision import masked_video_flow_match_loss, masked_video_latent_mse
from open_wam.models.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
    DualExpertActionTrainArtifacts,
    DualExpertTrainArtifacts,
    DualExpertVideoTrainArtifacts,
    VideoFlowTrainArtifacts,
)
from open_wam.models.policy_variants.contracts import DecoderArtifactEnvelope, PolicyTrainBatch, PolicyTrainOutput


# Frozen pre-extraction formulas: these intentionally do not call shared reducers.
def _reference_flow_loss(*, flow_pred, targets, timesteps, scheduler, future_loss_mask):
    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, None, :, None, None]
    per_token_loss = per_token_loss * future_loss_mask.float()
    denom = future_loss_mask.float().sum().clamp_min(1.0) * float(
        flow_pred.shape[1] * flow_pred.shape[3] * flow_pred.shape[4]
    )
    return per_token_loss.sum() / denom


def _reference_latent_mse(*, predicted_latents, target_latents, future_loss_mask):
    per_token = torch.nn.functional.mse_loss(predicted_latents.float(), target_latents.float(), reduction="none")
    per_token = per_token * future_loss_mask.float()
    denom = future_loss_mask.float().sum().clamp_min(1.0) * float(
        predicted_latents.shape[1] * predicted_latents.shape[3] * predicted_latents.shape[4]
    )
    return per_token.sum() / denom


def _inputs(dtype, mask_kind):
    prediction = torch.linspace(-1, 2, 144).reshape(2, 3, 4, 2, 3).to(dtype).requires_grad_()
    target = torch.linspace(2, -2, 144).reshape_as(prediction).to(dtype).requires_grad_()
    mask_values = {
        "empty": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "full": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "ragged": [[0, 1, 1, 1], [0, 0, 1, 0]],
        "fractional": [[0, 0.125, 0, 0], [0, 0, 0.25, 0]],
    }
    return dict(
        flow_pred=prediction,
        targets=target,
        timesteps=torch.tensor([[1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0]]),
        scheduler=SimpleNamespace(training_weight=lambda timesteps: timesteps * 0.125 + 0.25),
        predicted_latents=prediction,
        target_latents=target,
        future_loss_mask=torch.tensor(mask_values[mask_kind], dtype=dtype).reshape(2, 1, 4, 1, 1),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("mask_kind", ["empty", "full", "ragged", "fractional"])
def test_masked_video_losses_and_gradients_match_decoder_formulas(dtype, mask_kind) -> None:
    inputs = _inputs(dtype, mask_kind)
    flow_args = {key: inputs[key] for key in ("flow_pred", "targets", "timesteps", "scheduler", "future_loss_mask")}
    mse_args = {key: inputs[key] for key in ("predicted_latents", "target_latents", "future_loss_mask")}
    leaves = (inputs["flow_pred"], inputs["targets"])

    for actual_fn, reference_fn, args, target_detached in (
        (masked_video_flow_match_loss, _reference_flow_loss, flow_args, True),
        (masked_video_latent_mse, _reference_latent_mse, mse_args, False),
    ):
        actual, expected = actual_fn(**args), reference_fn(**args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.float32
        actual_grad = torch.autograd.grad(actual, leaves, allow_unused=True)
        expected_grad = torch.autograd.grad(expected, leaves, allow_unused=True)
        for got, want in zip(actual_grad, expected_grad, strict=True):
            if want is None:
                assert got is None
            else:
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert (actual_grad[1] is None) == target_detached
        if mask_kind == "empty":
            assert actual.item() == 0
            assert torch.count_nonzero(actual_grad[0]) == 0


@pytest.mark.parametrize("decoder_module", [dual_expert_decoder, video_only_decoder])
@pytest.mark.parametrize("enabled_objectives", [("action", "latent"), ("action",), ("latent",)])
def test_decoder_outputs_metrics_and_backward_are_unchanged(monkeypatch, decoder_module, enabled_objectives) -> None:
    assert decoder_module.masked_video_flow_match_loss is masked_video_flow_match_loss
    assert decoder_module.masked_video_latent_mse is masked_video_latent_mse
    inputs = _inputs(torch.float32, "ragged")
    training = TrainingConfig(
        enabled_objectives=enabled_objectives, action_loss_weight=0.3, latent_loss_weight=1.7
    )
    actions = torch.linspace(-1, 1, 36).reshape(2, 6, 3).requires_grad_()
    batch = PolicyTrainBatch(actions=torch.zeros_like(actions))
    if decoder_module is dual_expert_decoder:
        decoder_type = decoder_module.DualExpertActionDecoder
        contract = DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT
        payload = DualExpertTrainArtifacts(
            action=DualExpertActionTrainArtifacts(
                flow_pred=actions,
                targets=torch.ones_like(actions),
                timesteps=torch.ones(2, 6),
                scheduler=inputs["scheduler"],
                denoised_actions=actions,
                action_mask=torch.ones_like(actions),
            ),
            video=DualExpertVideoTrainArtifacts(**inputs),
            condition_mode="test",
            program="test",
            history_frames=1,
        )
    else:
        decoder_type = decoder_module.VideoOnlyActionDecoder
        contract = VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT
        payload = VideoFlowTrainArtifacts(**inputs)
    decoder = decoder_type(
        hidden_size=3, action_dim=3, action_horizon=6,
        training_config=training, inference_config=InferenceConfig(),
    )
    output = PolicyTrainOutput(
        policy_features=torch.empty(0), metrics={},
        decoder_artifacts=DecoderArtifactEnvelope(contract=contract, payload=payload),
    )
    actual = decoder.forward_train(output, batch)
    leaves = (actions, inputs["flow_pred"], inputs["targets"])
    actual_grad = torch.autograd.grad(actual.loss, leaves, allow_unused=True)
    monkeypatch.setattr(decoder_module, "masked_video_flow_match_loss", _reference_flow_loss)
    monkeypatch.setattr(decoder_module, "masked_video_latent_mse", _reference_latent_mse)
    expected = decoder.forward_train(output, batch)
    expected_grad = torch.autograd.grad(expected.loss, leaves, allow_unused=True)

    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.action_pred, expected.action_pred, rtol=0, atol=0)
    assert actual.metrics.keys() == expected.metrics.keys()
    assert actual.aux.keys() == expected.aux.keys()
    for name in actual.metrics:
        torch.testing.assert_close(actual.metrics[name], expected.metrics[name], rtol=0, atol=0)
    for name in actual.aux:
        torch.testing.assert_close(actual.aux[name], expected.aux[name], rtol=0, atol=0)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        if want is None:
            assert got is None
        else:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
