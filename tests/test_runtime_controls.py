from __future__ import annotations

import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.common import (
    build_joint_video_timestep_grid,
    build_joint_runtime_schedulers,
    build_unconditional_conditioning,
    combine_joint_cfg_predictions,
    preserve_joint_observed_video_prefix,
    resolve_runtime_cache_policy,
    resolve_runtime_guidance,
    resolve_runtime_warmup_reference,
)
from open_wam.models.video_backbone.contracts import ConditioningState


def test_runtime_guidance_resolves_from_negative_text_context() -> None:
    conditioning = ConditioningState(
        supported=True,
        text_context=torch.randn(2, 4, 8),
        negative_text_context=torch.zeros(2, 4, 8),
    )
    guidance = resolve_runtime_guidance(
        conditioning,
        inference_config=InferenceConfig(guidance_scale=3.0, action_guidance_scale=1.5),
    )

    assert guidance.enabled is True
    assert guidance.cfg_mode == "joint_cfg"
    assert guidance.video_guidance_scale == 3.0
    assert guidance.action_guidance_scale == 1.5
    assert guidance.video_mode == "guided"
    assert guidance.action_mode == "conditioned"


def test_build_unconditional_conditioning_swaps_in_negative_text_context() -> None:
    conditioning = ConditioningState(
        supported=True,
        text_context=torch.randn(2, 4, 8),
        negative_text_context=torch.zeros(2, 4, 8),
    )

    unconditional = build_unconditional_conditioning(conditioning)

    assert unconditional is not None
    assert torch.equal(unconditional.text_context, conditioning.negative_text_context)
    assert torch.equal(unconditional.negative_text_context, conditioning.negative_text_context)


def test_combine_joint_cfg_predictions_defaults_to_video_only_guidance() -> None:
    guidance = resolve_runtime_guidance(
        ConditioningState(
            supported=True,
            text_context=torch.ones(1, 1, 1),
            negative_text_context=torch.zeros(1, 1, 1),
        ),
        inference_config=InferenceConfig(guidance_scale=2.0, action_guidance_scale=3.0),
    )
    video_pred, action_pred = combine_joint_cfg_predictions(
        conditioned_video_prediction=torch.tensor([2.0]),
        unconditioned_video_prediction=torch.tensor([1.0]),
        conditioned_action_prediction=torch.tensor([3.0]),
        unconditioned_action_prediction=torch.tensor([1.0]),
        guidance=guidance,
    )

    assert torch.equal(video_pred, torch.tensor([3.0]))
    assert torch.equal(action_pred, torch.tensor([3.0]))


def test_combine_joint_cfg_predictions_can_still_apply_joint_guidance() -> None:
    guidance = resolve_runtime_guidance(
        ConditioningState(
            supported=True,
            text_context=torch.ones(1, 1, 1),
            negative_text_context=torch.zeros(1, 1, 1),
        ),
        inference_config=InferenceConfig(
            guidance_scale=2.0,
            action_guidance_scale=3.0,
            video_cfg_mode="guided",
            action_cfg_mode="guided",
        ),
    )
    video_pred, action_pred = combine_joint_cfg_predictions(
        conditioned_video_prediction=torch.tensor([2.0]),
        unconditioned_video_prediction=torch.tensor([1.0]),
        conditioned_action_prediction=torch.tensor([3.0]),
        unconditioned_action_prediction=torch.tensor([1.0]),
        guidance=guidance,
    )

    assert torch.equal(video_pred, torch.tensor([3.0]))
    assert torch.equal(action_pred, torch.tensor([7.0]))


def test_build_joint_video_timestep_grid_keeps_observed_prefix_at_zero() -> None:
    grid = build_joint_video_timestep_grid(
        batch_size=2,
        num_video_frames=4,
        timestep_value=5.0,
        device=torch.device("cpu"),
        observed_prefix_frames=1,
    )

    assert torch.equal(grid[:, 0], torch.zeros(2))
    assert torch.equal(grid[:, 1:], torch.full((2, 3), 5.0))


def test_preserve_joint_observed_video_prefix_restores_observed_frames() -> None:
    rollout = torch.full((1, 2, 4, 3, 3), fill_value=-1.0)
    observed = torch.arange(1 * 2 * 4 * 3 * 3, dtype=torch.float32).view(1, 2, 4, 3, 3)

    preserved = preserve_joint_observed_video_prefix(
        rollout_video_latents=rollout,
        observed_video_latents=observed,
        observed_prefix_frames=2,
    )

    assert torch.equal(preserved[:, :, :2], observed[:, :, :2])
    assert torch.equal(preserved[:, :, 2:], rollout[:, :, 2:])


def test_build_joint_runtime_schedulers_matches_requested_joint_steps() -> None:
    schedulers = build_joint_runtime_schedulers(
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(joint_num_inference_steps=4, joint_sampler="unipc"),
        device=torch.device("cpu"),
    )

    assert schedulers.use_unipc is True
    assert schedulers.num_steps == 4
    assert len(schedulers.video_scheduler.timesteps) == 4
    assert len(schedulers.action_scheduler.timesteps) == 4


def test_resolve_runtime_cache_policy_defaults_to_warmup_then_frozen() -> None:
    policy = resolve_runtime_cache_policy(
        inference_config=InferenceConfig(),
    )

    assert policy.warmup_before_denoise is True
    assert policy.warmup_source == "reference_video"
    assert policy.update_mode == "none"
    assert policy.update_cross_attention_on_warmup is True
    assert policy.update_cross_attention_during_denoise is False
    assert policy.initial_warmup_anchor == "start"
    assert policy.initial_warmup_frames == 1
    assert policy.rollout_warmup_anchor == "end"
    assert policy.rollout_warmup_frames is None


def test_resolve_runtime_warmup_reference_matches_dreamzero_schedule() -> None:
    policy = resolve_runtime_cache_policy(inference_config=InferenceConfig())

    first_chunk = resolve_runtime_warmup_reference(
        policy=policy,
        current_start_frame=0,
        num_video_frames=4,
        num_frame_per_block=1,
    )
    later_chunk = resolve_runtime_warmup_reference(
        policy=policy,
        current_start_frame=2,
        num_video_frames=4,
        num_frame_per_block=1,
    )

    assert first_chunk is not None
    assert first_chunk.frame_start == 0
    assert first_chunk.frame_count == 1
    assert later_chunk is not None
    assert later_chunk.frame_start == 3
    assert later_chunk.frame_count == 1


def test_resolve_runtime_cache_policy_preserves_legacy_reference_video_full_window() -> None:
    policy = resolve_runtime_cache_policy(
        inference_config=InferenceConfig(
            joint_cache_warmup_source="reference_video",
            joint_cache_initial_warmup_anchor="full",
            joint_cache_rollout_warmup_anchor="full",
        ),
    )

    warmup = resolve_runtime_warmup_reference(
        policy=policy,
        current_start_frame=0,
        num_video_frames=4,
        num_frame_per_block=2,
    )

    assert warmup is not None
    assert warmup.frame_start == 0
    assert warmup.frame_count == 4
