"""Condition augmentation synchronization and unchanged local noise semantics."""

from __future__ import annotations

import pytest
import torch

from open_wam.configs import TrainingConfig
from open_wam.models.common.flow_schedule import FlowMatchScheduler, sample_timestep_id
from open_wam.models.common.flow_training import build_video_flow_match_train_artifacts


_TENSOR_FIELDS = (
    "timesteps",
    "noisy_latents",
    "targets",
    "condition_latents",
    "condition_timesteps",
)


def _inputs():
    video = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 2, 2) / 32
    return video, video + 5, TrainingConfig(video_num_train_timesteps=16)


@pytest.mark.parametrize("rank_zero_decision", [0.25, 0.75])
def test_video_condition_noise_default_broadcast_controls_augmentation(
    monkeypatch, rank_zero_decision
):
    video, condition, config = _inputs()
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast(decision, src):
        calls.append((decision.shape, decision.device, src))
        decision.fill_(rank_zero_decision)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    artifacts = build_video_flow_match_train_artifacts(
        video,
        training_config=config,
        noisy_condition_prob=0.5,
        condition_latents=condition,
    )

    assert calls == [(torch.Size([1]), video.device, 0)]
    assert bool(artifacts.condition_timesteps.any()) == (rank_zero_decision < 0.5)
    assert torch.equal(artifacts.condition_latents, condition) == (
        rank_zero_decision >= 0.5
    )


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_video_condition_noise_opt_out_never_broadcasts(monkeypatch, probability):
    video, condition, config = _inputs()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    kwargs = dict(
        training_config=config,
        noisy_condition_prob=probability,
        condition_latents=condition,
        clean_prefix_frames=1,
    )
    torch.manual_seed(17)
    expected = build_video_flow_match_train_artifacts(video, **kwargs)
    expected_rng = torch.get_rng_state()

    def unexpected_broadcast(*args, **kwargs):
        pytest.fail("A local condition decision must not issue a collective.")

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", unexpected_broadcast)
    torch.manual_seed(17)
    actual = build_video_flow_match_train_artifacts(
        video, synchronize_noisy_condition_decision=False, **kwargs
    )
    for field in _TENSOR_FIELDS:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(actual.noisy_latents[:, :, :1], video[:, :, :1])
    assert torch.equal(actual.condition_latents[:, :, :1], condition[:, :, :1])
    assert not actual.targets[:, :, :1].any()
    assert not actual.timesteps[:, :1].any()
    assert not actual.condition_timesteps[:, :1].any()
    if probability == 0:
        assert not actual.condition_timesteps.any()
        assert torch.equal(actual.condition_latents, condition)
    elif probability == 1:
        assert actual.condition_timesteps[:, 1:].gt(0).all()
        assert not torch.equal(actual.condition_latents[:, :, 1:], condition[:, :, 1:])


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("prefix", [0, 1])
@pytest.mark.parametrize("seed", [17, 29])
def test_video_condition_noise_preserves_original_rng_order_and_schedule(
    monkeypatch, probability, prefix, seed
):
    video, condition, config = _inputs()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    scheduler = FlowMatchScheduler(
        shift=config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=config.video_num_train_timesteps,
    )
    scheduler.set_timesteps(config.video_num_train_timesteps, training=True)

    # This is the pre-batching draw order: frame timesteps, video noise,
    # decision (only p > 0), then conditional timesteps/noise only if selected.
    torch.manual_seed(seed)
    ids = sample_timestep_id(
        batch_size=1,
        sample_shape=(4,),
        num_train_timesteps=config.video_num_train_timesteps,
        device=video.device,
    )
    timesteps = scheduler.timesteps[ids]
    noise = torch.randn_like(video)
    noisy = scheduler.add_noise(video, noise, timesteps, t_dim=2)
    targets = scheduler.training_target(video, noise, timesteps)
    condition_times = torch.zeros_like(timesteps)
    expected_condition = condition.clone()
    if probability > 0 and torch.rand(1).item() < probability:
        condition_ids = sample_timestep_id(
            batch_size=1,
            sample_shape=(4,),
            min_timestep_bd=0.5,
            max_timestep_bd=1.0,
            num_train_timesteps=config.video_num_train_timesteps,
            device=video.device,
        )
        condition_times = scheduler.timesteps[condition_ids]
        condition_noise = torch.randn_like(video)
        expected_condition = scheduler.add_noise(
            condition, condition_noise, condition_times, t_dim=2
        )
    expected_rng = torch.get_rng_state()
    if prefix:
        noisy[:, :, :prefix] = video[:, :, :prefix]
        targets[:, :, :prefix] = 0
        timesteps[:, :prefix] = 0
        expected_condition[:, :, :prefix] = condition[:, :, :prefix]
        condition_times[:, :prefix] = 0
    expected = (timesteps, noisy, targets, expected_condition, condition_times)

    for synchronize in (True, False):
        torch.manual_seed(seed)
        actual = build_video_flow_match_train_artifacts(
            video,
            training_config=config,
            noisy_condition_prob=probability,
            condition_latents=condition,
            clean_prefix_frames=prefix,
            synchronize_noisy_condition_decision=synchronize,
        )
        for field, value in zip(_TENSOR_FIELDS, expected, strict=True):
            torch.testing.assert_close(getattr(actual, field), value, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
