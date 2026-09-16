"""Execution order is derived once, independently of architecture/cache layout."""

import pytest
import torch

from open_wam.configs import CurrentBlockCoupling, DynamicsObjective, VideoActionProgram
from open_wam.models.common.denoising import (
    denoise,
    independently_generated_modalities,
    resolve_denoising_stages,
)
from open_wam.models.common.dynamics_objectives import (
    DynamicsRolloutRequest,
    resolve_dynamics_rollout_plan,
)
from open_wam.models.common.flow_schedule import FlowMatchScheduler
from open_wam.models.policy_variants.contracts import (
    PolicyOutputModality,
)

V = PolicyOutputModality.VIDEO
A = PolicyOutputModality.ACTION


@pytest.mark.parametrize(
    ("coupling", "expected"),
    [
        (CurrentBlockCoupling.VIDEO_THEN_ACTION, {V}),
        (CurrentBlockCoupling.ACTION_THEN_VIDEO, {A}),
        (CurrentBlockCoupling.DECOUPLED_SAME_STEP, {V, A}),
        (CurrentBlockCoupling.JOINT, set()),
        (CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION, set()),
        (CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO, set()),
    ],
)
def test_published_output_subsets_follow_available_whole_stages(coupling, expected):
    assert independently_generated_modalities(coupling) == expected


def test_denoising_driver_preserves_evaluation_update_and_rng_order():
    scheduler = FlowMatchScheduler(num_train_timesteps=10)
    scheduler.set_timesteps(3)
    sample = torch.ones(1, 4)
    torch.manual_seed(781)
    expected = sample
    for timestep in scheduler.timesteps:
        prediction = expected * timestep + torch.randn_like(expected)
        expected = scheduler.step(prediction, timestep, expected)
    expected_rng = torch.random.get_rng_state()
    events = []

    def predict(sample, timestep):
        events.append("predict")
        return sample * timestep + torch.randn_like(sample)

    def update(prediction, timestep, sample):
        events.append("update")
        return scheduler.step(prediction, timestep, sample)

    torch.manual_seed(781)
    actual = denoise(sample, scheduler.timesteps, predict=predict, update=update)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert events == ["predict", "update"] * 3


@pytest.mark.parametrize(
    ("coupling", "expected"),
    (
        (CurrentBlockCoupling.VIDEO_THEN_ACTION, ((V,), (A,))),
        (CurrentBlockCoupling.ACTION_THEN_VIDEO, ((A,), (V,))),
        (CurrentBlockCoupling.DECOUPLED_SAME_STEP, ((V,), (A,))),
        (CurrentBlockCoupling.JOINT, ((V, A),)),
        (CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION, ((V, A),)),
        (CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO, ((V, A),)),
    ),
)
def test_policy_stage_order_preserves_native_rng_order(coupling, expected):
    assert (
        tuple(stage.updates for stage in resolve_denoising_stages(coupling=coupling))
        == expected
    )


def test_ordered_conditioning_is_explicit_and_decoupled_never_promotes_it():
    assert resolve_denoising_stages(coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION)[
        1
    ].clean_conditions == (V,)
    assert resolve_denoising_stages(coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO)[
        1
    ].clean_conditions == (A,)
    assert all(
        not stage.clean_conditions
        for stage in resolve_denoising_stages(
            coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP
        )
    )


def test_supplied_video_uses_the_same_action_stage_as_native_vta():
    native = resolve_denoising_stages(coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION)
    external = resolve_denoising_stages(
        coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        requested=frozenset({A}),
        supplied=frozenset({V}),
    )
    assert external == native[1:]


def test_requested_output_does_not_discard_its_live_dependencies():
    joint = resolve_denoising_stages(
        coupling=CurrentBlockCoupling.JOINT, requested=frozenset({A})
    )
    assert joint[0].updates == (V, A)
    vna = resolve_denoising_stages(
        coupling=CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION, requested=frozenset({A})
    )
    assert vna[0].updates == (V, A)


@pytest.mark.parametrize(
    ("objective", "program", "expected"),
    (
        (
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            VideoActionProgram.FORWARD_DYNAMICS,
            ((V,),),
        ),
        (
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            VideoActionProgram.INVERSE_DYNAMICS,
            ((A,),),
        ),
    ),
)
def test_strict_and_generalist_objectives_resolve_identical_stages(
    objective, program, expected
):
    explicit = resolve_dynamics_rollout_plan(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        request=DynamicsRolloutRequest(objective=objective),
    )
    fixed = resolve_dynamics_rollout_plan(program=program, request=None)
    for plan in (explicit, fixed):
        assert (
            tuple(
                stage.updates
                for stage in resolve_denoising_stages(
                    coupling=CurrentBlockCoupling.JOINT, dynamics=plan
                )
            )
            == expected
        )
        assert plan.semantics.drop_text_conditioning
        assert plan.semantics.history_frame_count == 1


def test_joint_objective_does_not_acquire_conditional_semantics():
    plan = resolve_dynamics_rollout_plan(
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        request=DynamicsRolloutRequest(objective=DynamicsObjective.JOINT),
    )
    assert tuple(
        stage.updates
        for stage in resolve_denoising_stages(
            coupling=CurrentBlockCoupling.JOINT, dynamics=plan
        )
    ) == ((V, A),)
    assert not plan.semantics.drop_text_conditioning
    assert not plan.semantics.is_conditional


def test_supplied_clean_input_cannot_silently_replace_a_live_dependency():
    with pytest.raises(ValueError, match="live denoising dependencies"):
        resolve_denoising_stages(
            coupling=CurrentBlockCoupling.JOINT,
            requested=frozenset({A}),
            supplied=frozenset({V}),
        )
