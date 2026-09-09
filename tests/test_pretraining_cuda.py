"""Small real CUDA kernels: video batching, gradients, and isolation."""

import os
from dataclasses import replace

import pytest
import torch

from open_wam.pipelines import build_variant_pipeline_from_config
from tests.test_causal_video_prediction import _tiny_causal_video_pipeline


@pytest.mark.gpu
def test_packed_video_cuda_matches_independent_sequences_and_gradients():
    if not torch.cuda.is_available() or os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1":
        pytest.skip("Requires an explicitly selected CUDA device.")
    torch.compiler.reset()
    config, _, _ = _tiny_causal_video_pipeline()
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            hidden_size=64,
            num_heads=2,
            attention_head_dim=32,
            ffn_dim=128,
        ),
        policy_variant=replace(config.policy_variant, hidden_size=64),
        action_decoder=replace(config.action_decoder, hidden_size=64),
    )
    torch.manual_seed(123)
    tower = build_variant_pipeline_from_config(config).visual_tower.cuda()
    video = torch.randn(2, 48, 6, 2, 4, device="cuda", requires_grad=True)
    text = torch.randn(2, 3, 8, device="cuda", requires_grad=True)
    times = torch.rand(2, 6, device="cuda") * 1000
    lengths = (3, 6)
    packed = tower.predict_video_flow(
        noisy_latents=video,
        timesteps=times,
        text_context=text,
        sequence_lengths=lengths,
    )
    independent = [
        tower.predict_video_flow(
            noisy_latents=video[i : i + 1, :, :n],
            timesteps=times[i : i + 1, :n],
            text_context=text[i : i + 1],
        )
        for i, n in enumerate(lengths)
    ]
    for i, n in enumerate(lengths):
        torch.testing.assert_close(
            packed[i : i + 1, :, :n], independent[i], rtol=2e-4, atol=3e-6
        )
    assert not packed[0, :, 3:].any()
    parameters = tuple(p for p in tower.parameters() if p.requires_grad)
    packed_loss = (
        sum(packed[i : i + 1, :, :n].square().sum() for i, n in enumerate(lengths))
        / 1000
    )
    single_loss = sum(value.square().sum() for value in independent) / 1000
    packed_gradients = torch.autograd.grad(packed_loss, parameters, allow_unused=True)
    single_gradients = torch.autograd.grad(single_loss, parameters, allow_unused=True)
    for actual, expected in zip(packed_gradients, single_gradients, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=5e-6)
    isolated = tower.predict_video_flow(
        noisy_latents=video,
        timesteps=times,
        text_context=text,
        sequence_lengths=lengths,
    )
    video_grad, text_grad = torch.autograd.grad(
        isolated[0].square().sum(), (video, text)
    )
    assert video_grad[0].abs().sum() > 0 and text_grad[0].abs().sum() > 0
    assert not video_grad[1].any() and not text_grad[1].any()
