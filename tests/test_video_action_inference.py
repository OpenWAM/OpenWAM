"""The shared executor owns stage, conditioning, RNG, and guidance semantics."""

from dataclasses import replace

import pytest
import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    JointTimestepCoupling,
    TrainingConfig,
    VideoActionProgram,
)
from open_wam.configs.enums import CFGMode, PolicyOutputModality
from open_wam.models.common.dynamics_objectives import resolve_dynamics_rollout_plan
from open_wam.models.common.video_action_inference import (
    VideoActionFlowInput,
    denoise_video_action,
)
from open_wam.models.policy_variants.dual_expert.attention_packed import (
    build_dual_expert_packed_coupling_attention_profile,
)

V, A = PolicyOutputModality.VIDEO, PolicyOutputModality.ACTION


@pytest.mark.parametrize("coupling", [CurrentBlockCoupling.JOINT, CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO])
@pytest.mark.parametrize("clock", list(JointTimestepCoupling))
def test_joint_clock_uses_the_declared_scheduler(coupling, clock):
    from open_wam.models.common.flow_inference import (
        build_action_flow_match_inference_scheduler, build_video_flow_match_inference_scheduler,
    )
    from open_wam.models.common.flow_schedule import timesteps_matching_sigmas

    training = TrainingConfig(video_sigma_shift=5.0, action_sigma_shift=1.0)
    inference = InferenceConfig(video_num_inference_steps=4, action_num_inference_steps=4)
    video = build_video_flow_match_inference_scheduler(training_config=training, inference_config=inference)
    action = build_action_flow_match_inference_scheduler(training_config=training, inference_config=inference)
    seen = []

    def record(step, cache):
        seen.append(step.action_timesteps[0, 2].clone())
        return torch.zeros_like(step.noisy_video), step.action_timesteps[:, :, None].expand_as(step.noisy_action) / 1000

    _, output = run(inputs(coupling), coupling, predict=record, training=training,
                    inference=inference, timestep_coupling=clock,
                    initial_action_noise=torch.zeros(1, 4, 2))
    if clock is JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
        expected = video.timesteps
        sigmas = video.sigmas
    elif clock is JointTimestepCoupling.MATCH_SIGMA:
        lookup = build_action_flow_match_inference_scheduler(
            training_config=training, inference_config=inference,
            num_inference_steps_override=training.action_num_train_timesteps,
        )
        expected = timesteps_matching_sigmas(lookup, video.sigmas[:4])
        sigmas = video.sigmas
    else:
        expected = action.timesteps
        sigmas = action.sigmas
    torch.testing.assert_close(torch.stack(seen), expected, rtol=0, atol=0)
    # A time-dependent velocity exposes integration with the wrong sigma grid.
    expected_value = 0.0
    for i in range(4):
        expected_value += ((sigmas[i + 1] if i < 3 else 0) - sigmas[i]) * expected[i] / 1000
    torch.testing.assert_close(output[:, 2:], torch.full_like(output[:, 2:], expected_value))


def inputs(coupling):
    video = torch.zeros(1, 1, 3, 1, 1)
    video[:, :, 0] = 3
    return VideoActionFlowInput(
        noisy_video=video,
        clean_video=video.clone(),
        video_timesteps=torch.zeros(1, 3),
        noisy_action=torch.zeros(1, 6, 2),
        clean_action=torch.zeros(1, 6, 2),
        action_timesteps=torch.zeros(1, 6),
        attention=build_dual_expert_packed_coupling_attention_profile(
            num_video_frames=3,
            video_tokens_per_frame=1,
            num_action_frames=3,
            action_tokens_per_frame=2,
            chunk_size_frames=2,
            chunk_origin_frame=1,
            device=torch.device("cpu"),
            current_block_coupling=coupling,
        ),
        text_context=torch.ones(1, 2, 4),
        frame_start=0,
    )


def predict(step, cache):
    text = step.text_context.mean()
    return (
        step.noisy_video * 0.1 + text,
        step.noisy_action * 0.1 + step.clean_video.mean() + text,
    )


def run(prepared, coupling, **kwargs):
    return denoise_video_action(
        prepared,
        predict=kwargs.pop("predict", predict),
        training=kwargs.pop("training", TrainingConfig()),
        inference=kwargs.pop(
            "inference",
            InferenceConfig(video_num_inference_steps=2, action_num_inference_steps=3),
        ),
        coupling=coupling,
        timestep_coupling=kwargs.pop(
            "timestep_coupling", JointTimestepCoupling.INDEPENDENT
        ),
        dynamics=kwargs.pop(
            "dynamics",
            resolve_dynamics_rollout_plan(
                program=VideoActionProgram.JOINT, request=None
            ),
        ),
        video_start=1,
        action_start=2,
        **kwargs,
    )


def test_independent_producer_consumer_matches_one_vta_execution_exactly():
    coupling = CurrentBlockCoupling.VIDEO_THEN_ACTION
    prepared = inputs(coupling)
    torch.manual_seed(913)
    native_video, native_action = run(prepared, coupling)
    torch.manual_seed(913)
    produced, _ = run(prepared, coupling, requested=frozenset({V}))
    _, consumed = run(
        replace(prepared, noisy_video=produced, clean_video=produced),
        coupling,
        requested=frozenset({A}),
        supplied=frozenset({V}),
    )
    torch.testing.assert_close(produced, native_video, rtol=0, atol=0)
    torch.testing.assert_close(consumed, native_action, rtol=0, atol=0)


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
def test_every_program_keeps_observed_prefix_and_invalid_actions_fixed(coupling):
    prepared = inputs(coupling)
    video, action = run(
        prepared,
        coupling,
        inference=InferenceConfig(
            video_num_inference_steps=2, action_num_inference_steps=2
        ),
    )
    assert torch.equal(video[:, :, :1], prepared.noisy_video[:, :, :1])
    assert torch.equal(action[:, :2], prepared.noisy_action[:, :2])
    assert torch.isfinite(video).all() and torch.isfinite(action).all()


def test_decoupled_stage_does_not_observe_completed_video():
    coupling = CurrentBlockCoupling.DECOUPLED_SAME_STEP
    prepared = inputs(coupling)
    conditions = []

    def record(step, cache):
        conditions.append(step.clean_video.clone())
        return predict(step, cache)

    run(prepared, coupling, predict=record)
    assert len(conditions) == 5
    assert all(torch.equal(value, prepared.clean_video) for value in conditions)


def test_action_first_only_promotes_completed_action_for_the_video_stage():
    coupling = CurrentBlockCoupling.ACTION_THEN_VIDEO
    prepared = inputs(coupling)
    conditions = []

    def record(step, cache):
        conditions.append((step.clean_video.clone(), step.clean_action.clone()))
        return predict(step, cache)

    _, completed_action = run(prepared, coupling, predict=record)
    assert len(conditions) == 5
    assert all(torch.equal(video, prepared.clean_video) for video, _ in conditions)
    assert all(
        torch.equal(action, prepared.clean_action) for _, action in conditions[:3]
    )
    assert all(torch.equal(action, completed_action) for _, action in conditions[3:])


def test_guidance_and_cache_enable_are_independent_controls():
    coupling = CurrentBlockCoupling.JOINT
    prepared = inputs(coupling)
    config = InferenceConfig(
        video_num_inference_steps=2,
        action_num_inference_steps=2,
        guidance_scale=5.0,
        action_cfg_mode=CFGMode.CONDITIONED,
    )
    results = []
    for enabled in (False, True):
        torch.manual_seed(33)
        results.append(
            run(
                prepared,
                coupling,
                inference=replace(config, use_cache=enabled),
                negative_text_context=torch.zeros_like(prepared.text_context),
            )
        )
    for cached, recomputed in zip(*results, strict=True):
        torch.testing.assert_close(cached, recomputed, rtol=0, atol=0)


@pytest.mark.parametrize("clock", tuple(JointTimestepCoupling))
def test_pruning_preserves_the_program_clock(clock):
    coupling = CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO
    prepared = inputs(coupling)
    traces = []
    actions = []
    for requested in (None, frozenset({A})):
        trace = []

        def record(step, cache):
            trace.append(step.action_timesteps.clone())
            return (
                torch.zeros_like(step.noisy_video),
                0.2 * step.noisy_action + step.action_timesteps[:, :, None] / 1000,
            )

        _, action = run(
            prepared,
            coupling,
            requested=requested,
            predict=record,
            timestep_coupling=clock,
            training=TrainingConfig(video_sigma_shift=5.0, action_sigma_shift=1.0),
            inference=InferenceConfig(
                video_num_inference_steps=4, action_num_inference_steps=4
            ),
            initial_video_noise=torch.ones_like(prepared.noisy_video[:, :, 1:]),
            initial_action_noise=torch.ones_like(prepared.noisy_action[:, 2:]),
        )
        traces.append(torch.stack(trace))
        actions.append(action)
    torch.testing.assert_close(*traces, rtol=0, atol=0)
    torch.testing.assert_close(*actions, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", (False, True))
def test_feature_storage_is_optional_and_separate_from_dependency_selection(enabled):
    coupling = CurrentBlockCoupling.VIDEO_THEN_ACTION
    prepared = inputs(coupling)
    caches = []

    def record(step, cache):
        assert step.required_tokens is not None
        assert step.required_tokens.any()
        assert (cache is not None) == enabled
        caches.append(cache)
        return predict(step, cache)

    run(
        prepared,
        coupling,
        predict=record,
        inference=InferenceConfig(
            video_num_inference_steps=2,
            action_num_inference_steps=2,
            use_cache=enabled,
            guidance_scale=5.0,
        ),
        negative_text_context=torch.zeros_like(prepared.text_context),
    )
    if enabled:
        assert caches[0] is caches[2]
        assert caches[1] is caches[3]
        assert caches[0] is not caches[1]
        assert caches[0] is not caches[4]


@pytest.mark.parametrize(
    "program",
    (VideoActionProgram.FORWARD_DYNAMICS, VideoActionProgram.INVERSE_DYNAMICS),
)
def test_conditional_objectives_only_update_the_supervised_modality(program):
    coupling = CurrentBlockCoupling.JOINT
    prepared = inputs(coupling)
    dynamics = resolve_dynamics_rollout_plan(program=program, request=None)
    video, action = run(prepared, coupling, dynamics=dynamics)
    assert (
        torch.equal(video, prepared.noisy_video)
        if not dynamics.semantics.video_loss_active
        else torch.equal(action, prepared.noisy_action)
    )
