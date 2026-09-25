from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.common import (
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
    build_block_coupled_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
    FlowMatchScheduler,
)
from open_wam.models.common.flow_supervision import build_video_frame_loss_mask
from open_wam.models.common import flow_noise_plan, flow_schedule


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["vector", "matrix", "scalar", "empty"])
def test_shared_sigma_lookup_preserves_grid_ties_shape_and_dtype(dtype, layout) -> None:
    scheduler = SimpleNamespace(
        num_train_timesteps=1000,
        sigmas=torch.tensor([1.0, 0.5, 0.5, 0.0], dtype=torch.float64),
        timesteps=torch.tensor([1000.0, 600.0, 400.0, 0.0], dtype=torch.float64),
    )
    values = torch.tensor([0.5, 0.75, 0.25, -0.25, 1.25, 0.0], dtype=dtype)
    expected = scheduler.timesteps[torch.tensor([1, 0, 1, 3, 0, 3])]
    if layout == "matrix":
        values, expected = values.reshape(2, 3).T, expected.reshape(2, 3).T
    elif layout == "scalar":
        values, expected = values[0], expected[0]
    elif layout == "empty":
        values, expected = values[:0], expected[:0]

    actual = flow_schedule.timesteps_matching_sigmas(scheduler, values)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device == values.device


def test_coupled_sampling_reuses_lookup_without_changing_rng_consumption() -> None:
    video = FlowMatchScheduler(num_train_timesteps=20, shift=5.0)
    action = FlowMatchScheduler(num_train_timesteps=20, shift=1.0)
    video.set_timesteps(20)
    action.set_timesteps(13)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        indices = flow_schedule.sample_timestep_id(
            batch_size=9, num_train_timesteps=20, device=torch.device("cpu")
        )
        sigmas = video.sigmas[indices]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(71)
        result = flow_noise_plan.sample_coupled_timestep_values(
            video_scheduler=video,
            action_scheduler=action,
            num_frames=9,
            device=torch.device("cpu"),
        )
        assert torch.equal(torch.get_rng_state(), expected_rng)

    assert torch.equal(result.sigma_values, sigmas)
    for scheduler, actual in ((video, result.video_timesteps), (action, result.action_timesteps)):
        nearest = (scheduler.sigmas[:, None] - sigmas[None]).abs().argmin(dim=0)
        assert torch.equal(actual, scheduler.timesteps[nearest])


def test_video_frame_loss_mask_uses_target_local_ranges_after_prefix() -> None:
    latents = torch.ones(2, 4, 6, 1, 1)

    mask = build_video_frame_loss_mask(
        latents,
        sample_metadata=(
            {"latent_loss_frame_start": 1, "latent_loss_frame_end": 3},
            {"latent_loss_frame_start": 0, "latent_loss_frame_end": 4},
        ),
        prefix_frame_count=1,
        target_frame_count=5,
    )

    assert torch.equal(
        mask[:, 0, :, 0, 0],
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ]
        ),
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
