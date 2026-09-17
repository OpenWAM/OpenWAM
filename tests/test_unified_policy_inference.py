"""Behavioral gates for the one executor, independent of retired runner APIs."""

import copy
import os
from dataclasses import replace

import pytest
import torch

from open_wam.configs import (
    CFGMode,
    DynamicsObjective,
    InferenceConfig,
    VideoActionProgram,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyInferenceOutputRequest,
    PolicyObservedHistory,
    PolicyTemporalSpan,
)
from open_wam.pipelines import VariantRolloutRunner
from tests.characterization.capture_denoising_execution import _parallel_pipeline
from tests.test_variable_batch_pipeline import PROGRAMS, _tiny_pipeline


CASES = (
    [(program, False, None) for program in PROGRAMS]
    + [
        (VideoActionProgram.GENERALIST_JOINT_DENOISING, token, objective)
        for token in (False, True)
        for objective in DynamicsObjective
    ]
    + [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            False,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            False,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ]
)


@pytest.fixture(params=("cpu", "cuda"), autouse=True)
def execution_device(request):
    if request.param == "cuda" and (
        os.environ.get("OPEN_WAM_RUN_GPU_SANITY") != "1"
        or not torch.cuda.is_available()
    ):
        pytest.skip("Set OPEN_WAM_RUN_GPU_SANITY=1 for the CUDA execution gate.")
    with torch.device(request.param):
        yield


def pipeline_for(architecture, program, token=False):
    torch.manual_seed(1901)
    pipeline, _ = (
        _tiny_pipeline(program, generalist_mode_text_token=token, attention_head_dim=32)
        if architecture == "dual_expert"
        else _parallel_pipeline(program, token=token, head_dim=32)
    )
    pipeline.eval()
    pipeline.policy_variant.inference_config = InferenceConfig(
        frame_chunk_size=2,
        attention_window_size=4,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    pipeline.default_temporal_geometry = replace(
        pipeline.default_temporal_geometry, attention_window_size=4
    )
    return pipeline


def request(objective):
    if objective is None:
        return None
    return DynamicsRolloutRequest(
        objective=objective,
        clean_action=torch.full((1, 4, 4), 0.25)
        if objective is DynamicsObjective.ACTION_CONDITIONED_VIDEO
        else None,
        clean_video=torch.full((1, 48, 1, 4, 4), 0.5)
        if objective is DynamicsObjective.VIDEO_CONDITIONED_ACTION
        else None,
        frame_chunk_size=1 if objective.is_conditional else 2,
    )


def tensors(output):
    return (
        output.decoder_output.action_pred,
        output.policy_output.aux["predicted_latents"],
    )


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("program,token,objective", CASES)
@torch.no_grad()
def test_public_runner_lifecycle(architecture, program, token, objective, monkeypatch):
    """One runner handles prediction, retry, partial/fallback commits and reset."""
    pipeline = pipeline_for(architecture, program, token)
    runner = VariantRolloutRunner(pipeline)
    text = torch.ones(1, 3, 16)
    inputs = dict(
        video_latents=torch.ones(1, 48, 1, 4, 4),
        context=PolicyInferContext(state=torch.ones(1, 1, 4), dynamics=request(objective)),
    )
    initial = runner.reset(text_context=text)
    torch.manual_seed(811)
    step = runner.infer_step(session=initial, **inputs)
    assert initial.policy_state is None
    assert step.infer_output.policy_output.generation_frame_start == 1
    first_predictions = tuple(t.clone() for t in tensors(step.infer_output))

    for iteration in range(6):
        output = step.infer_output.policy_output
        start = output.generation_frame_start
        count = step.session.policy_state.cursor.current_start_frame - start
        # Execute a partial plan first, then commit measured fallback controls.
        executed = 1 if iteration == 0 else count
        video = torch.full((1, 48, executed, 4, 4), float(iteration + 1))
        actions = torch.full((1, 2 * executed, 4), float(iteration + 2))
        validity = torch.ones(1, 2 * executed, 1)
        if iteration == 0:
            validity[:, :1] = 0
        prepared = pipeline.prepare_visual_outputs_from_latents(video, text_context=text)
        commit = runner.reconcile_observed_history(session=step.session, history=PolicyObservedHistory(video_latents=prepared.frontend.video_latents, observation_frame_count=executed * 2, action_history=actions, action_mask=validity, proprio_history=torch.ones(1, executed, 4), execution_commit=PolicyExecutionCommit(PolicyTemporalSpan(start, count), executed)))
        assert commit.applied
        state = commit.session.policy_state
        assert state.cursor.current_start_frame == start + executed
        history = state.variant_state
        torch.testing.assert_close(history.past_clean_latents[:, :, -executed:], video, rtol=0, atol=0)
        torch.testing.assert_close(history.past_clean_actions[:, -2 * executed:], actions * validity, rtol=0, atol=0)
        torch.testing.assert_close(history.past_clean_action_mask[:, -2 * executed:].float(), validity, rtol=0, atol=0)
        assert history.pending_predicted_video_frames == history.chunk_advance_frames == 0

        untouched = copy.deepcopy(commit.session)
        if iteration == 0:
            original = pipeline.resolve_infer_decoder_output

            def fail_after_decode(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected decode failure")

            with monkeypatch.context() as patch:
                patch.setattr(pipeline, "resolve_infer_decoder_output", fail_after_decode)
                with pytest.raises(RuntimeError, match="injected decode failure"):
                    runner.infer_step(session=commit.session, **inputs)
        torch.manual_seed(812 + iteration)
        step = runner.infer_step(session=commit.session, **inputs)
        torch.manual_seed(812 + iteration)
        reference = runner.infer_step(session=untouched, **inputs)
        assert step.infer_output.policy_output.generation_frame_start == start + executed
        assert state.cursor == untouched.policy_state.cursor
        for actual, expected in zip(tensors(step.infer_output), tensors(reference.infer_output), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    state = step.session.policy_state
    assert state.variant_state.past_clean_latents.shape[2] < state.cursor.current_start_frame
    reset = runner.reset(text_context=text)
    assert reset.policy_state is None
    torch.manual_seed(811)
    restarted = runner.infer_step(session=reset, **inputs)
    for actual, expected in zip(tensors(restarted.infer_output), first_predictions, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("program,token,objective", CASES)
@torch.no_grad()
def test_cache_setting_preserves_recurrent_denoising(
    architecture, program, token, objective
):
    pipeline = pipeline_for(architecture, program, token)
    parameter_dtypes = [parameter.dtype for parameter in pipeline.parameters()]
    reference = []
    for enabled in (False, True):
        pipeline.policy_variant.inference_config = replace(
            pipeline.policy_variant.inference_config, use_cache=enabled
        )
        state = None
        outputs = []
        for chunk in range(3):
            torch.manual_seed(314 + chunk)
            output = pipeline.forward_infer_step_from_latents(
                torch.randn(1, 48, 1, 4, 4),
                PolicyInferContext(
                    state=torch.randn(1, 1, 4), dynamics=request(objective)
                ),
                infer_state=state,
                text_context=torch.randn(1, 3, 16),
            )
            outputs.append(tuple(t.detach().clone() for t in tensors(output)))
            state = output.policy_output.next_state
            if objective is None or not objective.is_conditional:
                assert state.temporal_geometry.attention_window_size == 4
        if objective is None or not objective.is_conditional:
            # Three two-frame predictions plus t0 exceed the six-frame cache.
            assert state.variant_state.past_clean_latents.shape[2] < 7
        reference.append((outputs, copy.deepcopy(state.cursor)))
    assert reference[0][1] == reference[1][1]
    assert [parameter.dtype for parameter in pipeline.parameters()] == parameter_dtypes
    for uncached, cached in zip(reference[0][0], reference[1][0], strict=True):
        for expected, actual in zip(uncached, cached, strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("program", PROGRAMS)
@torch.no_grad()
def test_observation_commit_changes_the_next_plan(architecture, program):
    pipeline = pipeline_for(architecture, program)
    text = torch.randn(1, 3, 16)
    first = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=text,
    )
    predictions = []
    for value in (0.0, 30.0):
        state = copy.deepcopy(first.policy_output.next_state)
        observed = torch.full((1, 48, 2, 4, 4), value)
        update = pipeline.reconcile_observed_history(
            PolicyObservedHistory(
                video_latents=observed,
                observation_frame_count=8,
                proprio_history=torch.ones(1, 2, 4),
                action_history=torch.ones(1, 4, 4),
                execution_commit=PolicyExecutionCommit(PolicyTemporalSpan(1, 2), 2),
            ),
            state,
        )
        assert update.applied
        torch.manual_seed(912)
        output = pipeline.forward_infer_step_from_latents(
            observed,
            PolicyInferContext(state=torch.ones(1, 1, 4)),
            infer_state=update.next_state,
            text_context=text,
        )
        predictions.append(output.decoder_output.action_pred)
    assert not torch.equal(*predictions)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize(
    "program,objective",
    [
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
    ],
)
@torch.no_grad()
def test_strict_and_gjd_conditional_predictions_match(architecture, program, objective):
    standalone = pipeline_for(architecture, program)
    gjd = pipeline_for(architecture, VideoActionProgram.GENERALIST_JOINT_DENOISING)
    gjd.load_state_dict(standalone.state_dict(), strict=True)
    predictions = []
    for pipeline in (standalone, gjd):
        torch.manual_seed(912)
        output = pipeline.forward_infer_step_from_latents(
            torch.randn(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4), dynamics=request(objective)),
            text_context=torch.randn(1, 3, 16),
        )
        predictions.append(tensors(output))
    for expected, actual in zip(*predictions, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize(
    "objective",
    (
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
    ),
)
@torch.no_grad()
def test_conditional_generation_does_not_inherit_planning_text_or_old_history(
    architecture,
    objective,
):
    pipeline = pipeline_for(architecture, VideoActionProgram.GENERALIST_JOINT_DENOISING)
    state = None
    for _ in range(3):
        output = pipeline.forward_infer_step_from_latents(
            torch.randn(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4)),
            infer_state=state,
            text_context=torch.randn(1, 3, 16),
        )
        state = output.policy_output.next_state
    outputs = []
    for changed in (False, True):
        candidate = copy.deepcopy(state)
        if changed:
            history = candidate.variant_state
            history.past_clean_latents[:, :, :-1] += 100
            history.past_clean_actions[:, :-2] += 100
            history.past_hidden_proprio_states[:, :-1] += 100
        torch.manual_seed(510)
        output = pipeline.forward_infer_step_from_latents(
            torch.ones(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4), dynamics=request(objective)),
            infer_state=candidate,
            text_context=torch.full((1, 3, 16), 99.0 if changed else 0.0),
        )
        outputs.append(tensors(output))
    for expected, actual in zip(*outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@torch.no_grad()
def test_invalid_observation_commit_is_atomic(architecture):
    pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    first = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.zeros(1, 3, 16),
    )
    state = first.policy_output.next_state
    cursor = copy.deepcopy(state.cursor)
    before = copy.deepcopy(state.variant_state)
    with pytest.raises(ValueError, match="actions must contain exactly"):
        pipeline.reconcile_observed_history(
            PolicyObservedHistory(
                video_latents=torch.zeros(1, 48, 1, 4, 4),
                observation_frame_count=4,
                action_history=torch.zeros(1, 1, 4),
                execution_commit=PolicyExecutionCommit(PolicyTemporalSpan(1, 2), 1),
            ),
            state,
        )
    assert state.cursor == cursor
    for name in (
        "past_clean_latents",
        "past_clean_actions",
        "past_hidden_proprio_states",
    ):
        torch.testing.assert_close(
            getattr(state.variant_state, name), getattr(before, name), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "program",
    [VideoActionProgram.ACTION_THEN_VIDEO, VideoActionProgram.DECOUPLED_SAME_STEP],
)
@torch.no_grad()
def test_parallel_action_only_prediction_does_not_commit_unpredicted_video(program):
    pipeline = pipeline_for("parallel_stream", program)
    first = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.zeros(1, 3, 16),
    )
    state = first.policy_output.next_state
    past = state.variant_state.past_clean_latents.clone()
    output = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4), output_request=PolicyInferenceOutputRequest.action_only()),
        infer_state=state,
        text_context=torch.zeros(1, 3, 16),
    ).policy_output
    history = output.next_state.variant_state
    assert output.aux["predicted_latents"].shape[2] == 0
    assert history.pending_predicted_video_frames == 0
    torch.testing.assert_close(history.past_clean_latents, past, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@torch.no_grad()
def test_guidance_uses_the_supplied_negative_context(architecture):
    pipeline = pipeline_for(architecture, VideoActionProgram.JOINT)
    pipeline.policy_variant.inference_config = replace(
        pipeline.policy_variant.inference_config,
        guidance_scale=5.0,
        action_cfg_mode=CFGMode.GUIDED,
        action_guidance_scale=3.0,
    )
    outputs = []
    for negative in (0.0, 20.0):
        torch.manual_seed(532)
        output = pipeline.forward_infer_step_from_latents(
            torch.ones(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4)),
            text_context=torch.ones(1, 3, 16),
            negative_text_context=torch.full((1, 3, 16), negative),
        )
        outputs.append(tensors(output))
    for first, second in zip(*outputs, strict=True):
        assert not torch.equal(first, second)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@torch.no_grad()
def test_idm_can_commit_measured_actions_without_returning_them(architecture):
    pipeline = pipeline_for(architecture, VideoActionProgram.INVERSE_DYNAMICS)
    dynamics = replace(
        request(DynamicsObjective.VIDEO_CONDITIONED_ACTION),
        history_action=torch.full((1, 4, 4), 50.0),
    )
    output = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4), dynamics=dynamics),
        text_context=torch.ones(1, 3, 16),
    )
    predicted = output.decoder_output.action_pred
    committed = output.policy_output.next_state.variant_state.past_clean_actions
    assert not torch.equal(predicted, torch.full_like(predicted, 50.0))
    torch.testing.assert_close(
        committed[:, -predicted.shape[1] :],
        torch.full_like(predicted, 50.0),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_parallel_inactive_action_channels_stay_zero(monkeypatch):
    pipeline = pipeline_for("parallel_stream", VideoActionProgram.JOINT)
    monkeypatch.setattr(
        pipeline.policy_variant,
        "_reference_action_channel_mask",
        lambda **kwargs: torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1, 1),
    )
    output = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.ones(1, 3, 16),
    )
    assert torch.count_nonzero(output.decoder_output.action_pred[:, :, 1::2]) == 0
    history = output.policy_output.next_state.variant_state.past_clean_actions
    assert torch.count_nonzero(history[:, :, 1::2]) == 0


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("program", PROGRAMS)
@torch.no_grad()
def test_previous_action_is_not_an_implicit_history_commit(architecture, program):
    pipeline = pipeline_for(architecture, program)
    results = []
    for previous in (None, torch.ones(1, 4)):
        torch.manual_seed(321)
        output = pipeline.forward_infer_step_from_latents(
            torch.ones(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4), previous_action=previous),
            text_context=torch.ones(1, 3, 16),
        )
        results.append(tensors(output))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("shape", [(1, 1, 4), (1, 5, 4), (2, 4, 4), (1, 4, 3)])
@torch.no_grad()
def test_idm_rejects_unaligned_action_history(architecture, shape):
    pipeline = pipeline_for(architecture, VideoActionProgram.INVERSE_DYNAMICS)
    dynamics = DynamicsRolloutRequest(
        objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        clean_video=torch.ones(1, 48, 2, 4, 4),
        history_action=torch.ones(shape),
        frame_chunk_size=2,
    )
    with pytest.raises(ValueError, match="history_action"):
        pipeline.forward_infer_step_from_latents(
            torch.ones(1, 48, 1, 4, 4),
            PolicyInferContext(state=torch.ones(1, 1, 4), dynamics=dynamics),
            text_context=torch.ones(1, 3, 16),
        )


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("explicit_span", (False, True))
@torch.no_grad()
def test_action_only_requires_video_commit_before_continuing(architecture, explicit_span):
    pipeline = pipeline_for(architecture, VideoActionProgram.ACTION_THEN_VIDEO)
    first = pipeline.forward_infer_step_from_latents(
        torch.ones(1, 48, 1, 4, 4),
        PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.ones(1, 3, 16),
    )

    def action_only(state):
        return pipeline.forward_infer_step_from_latents(
            torch.ones(1, 48, 1, 4, 4),
            PolicyInferContext(
                state=torch.ones(1, 1, 4),
                output_request=PolicyInferenceOutputRequest.action_only(),
            ),
            infer_state=state,
            text_context=torch.ones(1, 3, 16),
        ).policy_output.next_state

    state = action_only(first.policy_output.next_state)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="video history"):
        action_only(state)
    assert state.cursor == before.cursor
    for key, value in vars(before.variant_state).items():
        actual = getattr(state.variant_state, key)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        else:
            assert actual == value
    update = pipeline.reconcile_observed_history(
        PolicyObservedHistory(
            video_latents=torch.zeros(1, 48, 2, 4, 4),
            proprio_history=torch.ones(1, 2, 4),
            observation_frame_count=8,
            execution_commit=(
                PolicyExecutionCommit(PolicyTemporalSpan(3, 2), 2)
                if explicit_span
                else None
            ),
        ),
        state,
    )
    assert action_only(update.next_state).cursor.current_start_frame == 7
