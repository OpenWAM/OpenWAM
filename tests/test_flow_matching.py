from __future__ import annotations

import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.common import (
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
    build_block_coupled_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
    FlowMatchScheduler,
)


def test_action_flow_match_artifacts_stay_on_input_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = torch.randn(2, 6, 7, device=device)
    action_mask = torch.ones_like(actions)

    artifacts = build_action_flow_match_train_artifacts(
        actions,
        action_mask,
        training_config=TrainingConfig(),
    )

    assert artifacts.timesteps.device.type == device.type
    assert artifacts.noisy_actions.device.type == device.type
    assert artifacts.targets.device.type == device.type


def test_action_flow_match_scheduler_step_preserves_sample_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = build_action_flow_match_inference_scheduler(
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(action_num_inference_steps=4),
    )
    sample = torch.randn(2, 6, 7, device=device)
    model_output = torch.randn_like(sample)

    next_sample = scheduler.step(model_output, scheduler.timesteps[0], sample)

    assert next_sample.device.type == device.type


def test_flow_match_scheduler_add_noise_supports_per_sample_timesteps() -> None:
    scheduler = FlowMatchScheduler(num_train_timesteps=10, shift=1.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(10, training=True)
    original = torch.zeros(2, 1, 3, 1, 1)
    noise = torch.ones_like(original)
    timesteps = torch.tensor(
        [
            [float(scheduler.timesteps[0]), float(scheduler.timesteps[1]), float(scheduler.timesteps[2])],
            [float(scheduler.timesteps[3]), float(scheduler.timesteps[4]), float(scheduler.timesteps[5])],
        ]
    )

    noisy = scheduler.add_noise(original, noise, timesteps, t_dim=2)

    expected = scheduler.sigmas[[0, 1, 2, 3, 4, 5]].view(2, 1, 3, 1, 1)
    assert torch.allclose(noisy, expected.to(noisy.dtype))


def test_video_flow_match_train_artifacts_sample_timesteps_per_sample() -> None:
    torch.manual_seed(0)
    video_latents = torch.randn(2, 4, 3, 2, 2)

    artifacts = build_video_flow_match_train_artifacts(
        video_latents,
        training_config=TrainingConfig(),
    )

    assert artifacts.timesteps.shape == (2, 3)
    assert not torch.equal(artifacts.timesteps[0], artifacts.timesteps[1])


def test_video_flow_match_train_artifacts_preserve_clean_prefix() -> None:
    torch.manual_seed(0)
    video_latents = torch.randn(1, 4, 4, 2, 2)
    condition_latents = torch.randn_like(video_latents)

    torch.manual_seed(17)
    baseline = build_video_flow_match_train_artifacts(
        video_latents,
        condition_latents=condition_latents,
        noisy_condition_prob=1.0,
        training_config=TrainingConfig(video_num_train_timesteps=8),
    )
    torch.manual_seed(17)
    artifacts = build_video_flow_match_train_artifacts(
        video_latents,
        condition_latents=condition_latents,
        noisy_condition_prob=1.0,
        clean_prefix_frames=1,
        training_config=TrainingConfig(video_num_train_timesteps=8),
    )

    torch.testing.assert_close(
        artifacts.noisy_latents[:, :, :1], video_latents[:, :, :1]
    )
    torch.testing.assert_close(
        artifacts.condition_latents[:, :, :1], condition_latents[:, :, :1]
    )
    assert not torch.any(artifacts.targets[:, :, :1])
    assert not torch.any(artifacts.timesteps[:, :1])
    assert not torch.any(artifacts.condition_timesteps[:, :1])
    assert torch.all(artifacts.timesteps[:, 1:] > 0)
    torch.testing.assert_close(
        artifacts.noisy_latents[:, :, 1:], baseline.noisy_latents[:, :, 1:]
    )
    torch.testing.assert_close(artifacts.targets[:, :, 1:], baseline.targets[:, :, 1:])
    torch.testing.assert_close(artifacts.timesteps[:, 1:], baseline.timesteps[:, 1:])
    torch.testing.assert_close(
        artifacts.condition_latents[:, :, 1:], baseline.condition_latents[:, :, 1:]
    )
    torch.testing.assert_close(
        artifacts.condition_timesteps[:, 1:],
        baseline.condition_timesteps[:, 1:],
    )


def test_clean_video_prefix_preserves_condition_autograd() -> None:
    video_latents = torch.randn(1, 4, 4, 2, 2, requires_grad=True)
    condition_latents = torch.randn_like(video_latents, requires_grad=True)

    artifacts = build_video_flow_match_train_artifacts(
        video_latents,
        condition_latents=condition_latents,
        clean_prefix_frames=1,
        training_config=TrainingConfig(video_num_train_timesteps=8),
    )
    (artifacts.noisy_latents.sum() + artifacts.condition_latents.sum()).backward()

    assert video_latents.grad is not None
    assert condition_latents.grad is not None


def test_block_coupled_action_flow_match_uses_per_sample_video_blocks() -> None:
    actions = torch.randn(2, 6, 7)
    future_video_timesteps = torch.tensor(
        [
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ]
    )

    artifacts = build_block_coupled_action_flow_match_train_artifacts(
        actions,
        action_mask=None,
        training_config=TrainingConfig(),
        future_video_timesteps=future_video_timesteps,
        num_frame_per_block=2,
        num_action_per_block=3,
    )

    assert torch.equal(artifacts.block_timesteps[0], torch.tensor([100.0, 200.0]))
    assert torch.equal(artifacts.block_timesteps[1], torch.tensor([300.0, 400.0]))
    assert torch.equal(artifacts.timesteps[0], torch.tensor([100.0, 100.0, 100.0, 200.0, 200.0, 200.0]))
    assert torch.equal(artifacts.timesteps[1], torch.tensor([300.0, 300.0, 300.0, 400.0, 400.0, 400.0]))
