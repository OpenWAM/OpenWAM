"""Session publication and observation commits share one temporal authority."""

import copy
from dataclasses import replace

import pytest
import torch

from open_wam.configs import DynamicsObjective, VideoActionProgram
from open_wam.models.common.dynamics_contracts import DynamicsRolloutRequest
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyObservedHistory,
    PolicyInferState,
    PolicyTemporalGeometry,
    PolicyTemporalSpan,
    PolicyVideoGenerationRequest,
)
from open_wam.pipelines import VariantRolloutRunner
from tests.test_unified_policy_inference import pipeline_for


@pytest.mark.parametrize(
    "architecture", ("dual_expert", "parallel_stream", "causal_video")
)
@pytest.mark.parametrize("failure", ("preparation", "prediction", "decoder"))
@torch.no_grad()
def test_failed_first_call_does_not_bind_session_geometry(
    architecture, failure, monkeypatch
):
    if architecture == "causal_video":
        from tests.test_causal_video_prediction import (
            _tiny_chunked_conditioned_video_pipeline,
        )

        _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    else:
        pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    pipeline.eval()
    pipeline.policy_variant.inference_config = replace(
        pipeline.policy_variant.inference_config,
        frame_chunk_size=4,
    )
    previous = PolicyInferState()
    inputs = dict(
        video_latents=torch.ones(1, 48, 1, 4, 4),
        text_context=torch.ones(1, 3, 8 if architecture == "causal_video" else 16),
    )
    context = PolicyInferContext(
        state=torch.ones(1, 1, 4),
        temporal_geometry=PolicyTemporalGeometry(4, 30),
        task_text=("move the object",),
        video_generation=PolicyVideoGenerationRequest(frame_count=4)
        if architecture == "causal_video"
        else None,
    )
    owner, method = {
        "preparation": (pipeline.policy_variant, "prepare_infer_state"),
        "prediction": (pipeline.policy_variant, "forward_infer_step"),
        "decoder": (pipeline, "resolve_infer_decoder_output"),
    }[failure]
    original = getattr(owner, method)

    def fail_after_computation(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(owner, method, fail_after_computation)
        with pytest.raises(RuntimeError, match="injected failure"):
            pipeline.forward_infer_step_from_latents(
                **inputs, context=context, infer_state=previous
            )
    assert previous == PolicyInferState()
    retry = replace(context, temporal_geometry=PolicyTemporalGeometry(2, 30))
    outputs = []
    for state in (previous, PolicyInferState()):
        torch.manual_seed(717)
        output = pipeline.forward_infer_step_from_latents(
            **inputs, context=retry, infer_state=state
        )
        assert (
            output.policy_output.next_state.temporal_geometry == retry.temporal_geometry
        )
        outputs.append(output)
    assert previous == PolicyInferState()
    if outputs[0].decoder_output.action_pred is not None:
        torch.testing.assert_close(
            outputs[0].decoder_output.action_pred,
            outputs[1].decoder_output.action_pred,
            rtol=0,
            atol=0,
        )
    videos = [
        output.policy_output.generated_video.latents
        if output.policy_output.generated_video is not None
        else output.policy_output.decoder_artifacts.payload.predicted_latents
        for output in outputs
    ]
    torch.testing.assert_close(*videos, rtol=0, atol=0)


def assert_state_equal(actual, expected):
    assert actual.cursor == expected.cursor
    assert actual.step_index == expected.step_index
    assert actual.revision == expected.revision
    assert actual.observed_frame_end == expected.observed_frame_end
    assert actual.temporal_geometry == expected.temporal_geometry
    assert actual.decoder_state == expected.decoder_state
    for key, value in vars(expected.variant_state).items():
        result = getattr(actual.variant_state, key)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(result, value, rtol=0, atol=0)
        else:
            assert result == value


@pytest.mark.parametrize("architecture", ("dual_expert", "parallel_stream"))
@pytest.mark.parametrize("failure", ("validation", "prediction", "decoder"))
@torch.no_grad()
def test_failed_prediction_does_not_publish_session_state(
    architecture, failure, monkeypatch
):
    pipeline = pipeline_for(architecture, VideoActionProgram.GENERALIST_JOINT_DENOISING)
    common = dict(
        video_latents=torch.ones(1, 48, 1, 4, 4),
        text_context=torch.ones(1, 3, 16),
    )
    context = PolicyInferContext(state=torch.ones(1, 1, 4))
    state = pipeline.forward_infer_step_from_latents(
        **common,
        context=context,
    ).policy_output.next_state
    state = pipeline.reconcile_observed_history(
        PolicyObservedHistory(
            video_latents=torch.ones(1, 48, 2, 4, 4),
            proprio_history=torch.ones(1, 2, 4),
            observation_frame_count=8,
            execution_commit=PolicyExecutionCommit(PolicyTemporalSpan(1, 2), 2),
        ),
        state,
    ).next_state
    before = copy.deepcopy(state)
    bad = replace(context, state=torch.full((1, 1, 4), 9.0))
    with monkeypatch.context() as patch:
        if failure == "validation":
            bad = replace(
                bad,
                dynamics=DynamicsRolloutRequest(
                    objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                    clean_video=torch.ones(1, 48, 2, 4, 4),
                    history_action=torch.ones(1, 1, 4),
                    frame_chunk_size=2,
                ),
            )
        else:
            owner = pipeline.policy_variant if failure == "prediction" else pipeline
            method = (
                "forward_infer_step"
                if failure == "prediction"
                else "resolve_infer_decoder_output"
            )
            original = getattr(owner, method)

            def fail_after_computation(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected failure")

            patch.setattr(owner, method, fail_after_computation)
        with pytest.raises((ValueError, RuntimeError)):
            pipeline.forward_infer_step_from_latents(
                **common, context=bad, infer_state=state
            )
    assert_state_equal(state, before)
    outputs = []
    for previous in (state, before):
        torch.manual_seed(717)
        outputs.append(
            pipeline.forward_infer_step_from_latents(
                **common,
                context=context,
                infer_state=previous,
            ).decoder_output.action_pred
        )
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("start", [0, 11])
@pytest.mark.parametrize("frames", [1, 3])
def test_observed_seed_accepts_an_actionless_anchor(architecture, start, frames):
    pipeline = pipeline_for(architecture, VideoActionProgram.VIDEO_THEN_ACTION)
    runner = VariantRolloutRunner(pipeline)
    visual = pipeline.prepare_visual_outputs_from_latents(
        torch.ones(1, 48, frames, 4, 4),
        text_context=torch.ones(1, 3, 16),
    )
    initial = runner.reset()
    update = runner.reconcile_observed_history(
        session=initial,
        history=PolicyObservedHistory(
            video_latents=visual.frontend.video_latents,
            observation_frame_count=2 * (frames - 1),
            action_history=torch.ones(1, 2 * (frames - 1), 4),
            proprio_history=torch.ones(1, frames, 4),
            start_frame=start,
        ),
    )
    assert update.applied
    assert initial.policy_state is None
    state = update.session.policy_state
    assert state.cursor.current_start_frame == start + frames
    assert state.variant_state.past_clean_actions.shape[1] == frames * 2
    assert not state.variant_state.past_clean_action_mask[:, :2].any()
    assert state.variant_state.past_clean_action_mask[:, 2:].all()
    step = runner.infer_step(
        session=update.session,
        video_latents=visual.frontend.video_latents,
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
    )
    assert step.infer_output.policy_output.generation_frame_start == start + frames
