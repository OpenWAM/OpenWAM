from __future__ import annotations

import pytest
import torch

from open_wam.models.common.flow_matching import (
    FlowMatchScheduler,
    sample_timestep_id,
)
from open_wam.models.common.flow_noise_plan import (
    sample_coupled_timestep_values as sample_shared_coupled_timestep_values,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.training_noise import (
    build_parallel_flow_noise_artifacts,
    sample_coupled_parallel_timestep_values,
    sample_index_matched_timestep_values,
    sample_shared_video_schedule_timestep_values,
    share_video_scheduler_grid_with_action_scheduler,
)
from open_wam.models.visual_tower.exact_runtime import build_reference_mesh_id


def _scheduler(*, shift: float, steps: int) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=steps,
    )
    scheduler.set_timesteps(steps, training=True)
    return scheduler


def test_reference_runtime_training_noise_names_alias_canonical_contract() -> None:
    assert reference_runtime.sample_timestep_id is sample_timestep_id
    assert reference_runtime._add_noise is build_parallel_flow_noise_artifacts
    assert (
        reference_runtime._sample_coupled_timestep_values
        is sample_coupled_parallel_timestep_values
    )
    assert (
        reference_runtime._sample_index_matched_timestep_values
        is sample_index_matched_timestep_values
    )
    assert (
        reference_runtime._sample_shared_video_schedule_timestep_values
        is sample_shared_video_schedule_timestep_values
    )
    assert (
        reference_runtime._share_video_scheduler_grid_with_action_scheduler
        is share_video_scheduler_grid_with_action_scheduler
    )


def test_parallel_flow_noise_explicit_sigma_values_and_gradients() -> None:
    scheduler = _scheduler(shift=2.0, steps=8)
    latent = (
        torch.arange(16, dtype=torch.float64).reshape(1, 2, 2, 2, 2) / 10
    ).requires_grad_()
    condition = (latent.detach() + 3).requires_grad_()
    timesteps = scheduler.timesteps[[1, 5]].to(dtype=torch.float64)
    resolved_timesteps = timesteps.to(dtype=scheduler.timesteps.dtype)
    sigmas = torch.tensor([0.25, 0.75], dtype=torch.float64)

    torch.manual_seed(19)
    expected_noise = torch.zeros_like(latent).normal_()
    torch.manual_seed(19)
    artifacts = build_parallel_flow_noise_artifacts(
        latent,
        train_scheduler=scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(1, 1, 1),
        condition_latent=condition,
        frame_shift=4,
        timestep_values=timesteps,
        sigma_values=sigmas,
    )

    sigma_volume = sigmas.view(1, 1, 2, 1, 1)
    expected_noisy = (1 - sigma_volume) * latent + sigma_volume * expected_noise
    expected_targets = scheduler.training_target(
        latent,
        expected_noise,
        resolved_timesteps,
    )
    expected_grid = build_reference_mesh_id(
        2,
        2,
        2,
        t=0,
        f_w=1,
        f_shift=4,
        action=False,
        device=latent.device,
    )[None]

    torch.testing.assert_close(
        artifacts["noisy_latents"],
        expected_noisy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["targets"],
        expected_targets,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["latent"],
        condition,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["timesteps"],
        resolved_timesteps[None],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["cond_timesteps"],
        torch.zeros_like(resolved_timesteps)[None],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["grid_id"],
        expected_grid,
        rtol=0.0,
        atol=0.0,
    )

    actual_loss = (
        artifacts["noisy_latents"].square().sum()
        + artifacts["targets"].square().sum()
        + artifacts["latent"].square().sum()
    )
    expected_loss = (
        expected_noisy.square().sum()
        + expected_targets.square().sum()
        + condition.square().sum()
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        (latent, condition),
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (latent, condition),
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_parallel_flow_noise_preserves_implicit_rng_order_and_action_mask() -> None:
    scheduler = _scheduler(shift=1.0, steps=16)
    latent = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2, 1)
    mask = torch.tensor(
        [
            [
                [[[1.0], [0.0]], [[1.0], [0.0]]],
                [[[0.0], [1.0]], [[0.0], [1.0]]],
                [[[1.0], [1.0]], [[0.0], [0.0]]],
            ]
        ]
    )

    torch.manual_seed(23)
    expected_noise = torch.zeros_like(latent).normal_()
    expected_ids = sample_timestep_id(
        batch_size=2,
        num_train_timesteps=16,
        device=latent.device,
    )
    expected_timesteps = scheduler.timesteps[expected_ids]
    expected_noisy = scheduler.add_noise(
        latent,
        expected_noise,
        expected_timesteps,
        t_dim=2,
    )
    expected_targets = scheduler.training_target(
        latent,
        expected_noise,
        expected_timesteps,
    )

    torch.manual_seed(23)
    artifacts = build_parallel_flow_noise_artifacts(
        latent,
        train_scheduler=scheduler,
        action_mask=mask,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(2, 4, 4),
        frame_shift=3,
    )

    torch.testing.assert_close(
        artifacts["timesteps"],
        expected_timesteps[None],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["noisy_latents"],
        expected_noisy * mask,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["targets"],
        expected_targets * mask,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["latent"],
        latent * mask,
        rtol=0.0,
        atol=0.0,
    )
    expected_grid = build_reference_mesh_id(
        2,
        2,
        1,
        t=1,
        f_w=1,
        f_shift=3,
        action=True,
        device=latent.device,
    )[None]
    torch.testing.assert_close(
        artifacts["grid_id"],
        expected_grid,
        rtol=0.0,
        atol=0.0,
    )


def test_parallel_flow_noise_noisy_condition_preserves_rng_order() -> None:
    scheduler = _scheduler(shift=1.0, steps=16)
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    condition = latent + 5

    torch.manual_seed(29)
    expected_noise = torch.zeros_like(latent).normal_()
    expected_ids = sample_timestep_id(
        batch_size=2,
        num_train_timesteps=16,
        device=latent.device,
    )
    expected_timesteps = scheduler.timesteps[expected_ids]
    torch.rand(1, device=latent.device)
    expected_condition_ids = sample_timestep_id(
        batch_size=2,
        min_timestep_bd=0.5,
        max_timestep_bd=1.0,
        num_train_timesteps=16,
        device=latent.device,
    )
    expected_condition_noise = torch.zeros_like(latent).normal_()
    expected_condition_timesteps = scheduler.timesteps[expected_condition_ids]
    expected_condition = scheduler.add_noise(
        condition,
        expected_condition_noise,
        expected_condition_timesteps,
        t_dim=2,
    )

    torch.manual_seed(29)
    artifacts = build_parallel_flow_noise_artifacts(
        latent,
        train_scheduler=scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=1.0,
        patch_size=(1, 1, 1),
        condition_latent=condition,
    )

    torch.testing.assert_close(
        artifacts["noisy_latents"],
        scheduler.add_noise(
            latent,
            expected_noise,
            expected_timesteps,
            t_dim=2,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["cond_timesteps"],
        expected_condition_timesteps[None],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        artifacts["latent"],
        expected_condition,
        rtol=0.0,
        atol=0.0,
    )


def test_parallel_timestep_adapters_preserve_shared_scheduler_contracts() -> None:
    video_scheduler = _scheduler(shift=5.0, steps=16)
    action_scheduler = _scheduler(shift=1.0, steps=8)

    torch.manual_seed(31)
    expected_coupled = sample_shared_coupled_timestep_values(
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        num_frames=4,
        device=torch.device("cpu"),
    )
    torch.manual_seed(31)
    video_values, action_values, sigma_values = (
        sample_coupled_parallel_timestep_values(
            latent_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
            num_frames=4,
            device=torch.device("cpu"),
        )
    )
    torch.testing.assert_close(video_values, expected_coupled.video_timesteps)
    torch.testing.assert_close(action_values, expected_coupled.action_timesteps)
    torch.testing.assert_close(sigma_values, expected_coupled.sigma_values)

    torch.manual_seed(37)
    expected_ids = sample_timestep_id(
        batch_size=4,
        num_train_timesteps=16,
        device=torch.device("cpu"),
    )
    torch.manual_seed(37)
    shared_video, shared_action, shared_sigma = (
        sample_shared_video_schedule_timestep_values(
            latent_scheduler=video_scheduler,
            num_frames=4,
            device=torch.device("cpu"),
        )
    )
    torch.testing.assert_close(shared_video, video_scheduler.timesteps[expected_ids])
    torch.testing.assert_close(shared_action, shared_video)
    torch.testing.assert_close(shared_sigma, video_scheduler.sigmas[expected_ids])


def test_index_matched_timesteps_and_shared_grid_mutation() -> None:
    video_scheduler = _scheduler(shift=5.0, steps=8)
    action_scheduler = _scheduler(shift=1.0, steps=8)

    torch.manual_seed(41)
    expected_ids = sample_timestep_id(
        batch_size=3,
        num_train_timesteps=8,
        device=torch.device("cpu"),
    )
    torch.manual_seed(41)
    video_values, action_values = sample_index_matched_timestep_values(
        latent_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        num_frames=3,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(video_values, video_scheduler.timesteps[expected_ids])
    torch.testing.assert_close(action_values, action_scheduler.timesteps[expected_ids])

    share_video_scheduler_grid_with_action_scheduler(
        latent_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(action_scheduler.timesteps, video_scheduler.timesteps)
    torch.testing.assert_close(action_scheduler.sigmas, video_scheduler.sigmas)
    torch.testing.assert_close(
        action_scheduler.linear_timesteps_weights,
        video_scheduler.linear_timesteps_weights,
    )


def test_parallel_training_noise_rejects_invalid_shapes_and_grid_lengths() -> None:
    scheduler = _scheduler(shift=1.0, steps=8)
    latent = torch.zeros(1, 2, 2, 2, 2)
    kwargs = {
        "train_scheduler": scheduler,
        "action_mask": None,
        "action_mode": False,
        "noisy_cond_prob": 0.0,
        "patch_size": (1, 1, 1),
    }

    with pytest.raises(ValueError, match="one scalar per frame"):
        build_parallel_flow_noise_artifacts(
            latent,
            timestep_values=torch.zeros(3),
            **kwargs,
        )
    with pytest.raises(ValueError, match="one scalar per frame"):
        build_parallel_flow_noise_artifacts(
            latent,
            sigma_values=torch.zeros(3),
            **kwargs,
        )
    with pytest.raises(ValueError, match="Condition latent shape must match"):
        build_parallel_flow_noise_artifacts(
            latent,
            condition_latent=torch.zeros(1, 2, 1, 2, 2),
            **kwargs,
        )
    with pytest.raises(ValueError, match="equal video/action"):
        sample_index_matched_timestep_values(
            latent_scheduler=scheduler,
            action_scheduler=_scheduler(shift=1.0, steps=4),
            num_frames=2,
            device=torch.device("cpu"),
        )
