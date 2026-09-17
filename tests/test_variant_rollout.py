from __future__ import annotations
from open_wam.models.common.rollout import RolloutCursor

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from open_wam.models.action_decoders import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderRolloutPlan,
)
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferenceCapabilities,
    PolicyInferenceOutputRequest,
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyOutputModality,
    PolicyTemporalGeometry,
)
from open_wam.models.policy_variants.base import PolicyVariant
from open_wam.pipelines import VariantPipeline, VariantRolloutRunner


class _PreparedPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def forward_infer_step_from_visual_outputs(
        self,
        visual_outputs,
        *,
        context,
        infer_state,
    ):
        self.calls.append(
            {
                "visual_outputs": visual_outputs,
                "context": context,
                "infer_state": infer_state,
            }
        )
        return SimpleNamespace(
            visual_outputs=visual_outputs,
            decoder_output=ActionDecoderInferOutput(action_pred=torch.empty(1, 0, 0)),
            policy_output=SimpleNamespace(
                next_state=PolicyInferState(cursor=RolloutCursor(block_index=7)),
                generated_span=None,
            ),
        )


class _ObservedHistoryTarget:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def reconcile_observed_history(self, history, policy_state):
        self.calls.append((history, policy_state))
        return PolicyObservedHistoryOutput(
            next_state=PolicyInferState(cursor=RolloutCursor(block_index=9)),
            debug={"committed": True},
            applied=True,
        )


class _RolloutPlanDecoder(ActionDecoder):
    def forward_train(self, policy_output, batch):
        raise NotImplementedError

    def forward_infer(self, policy_output, previous_state=None):
        raise NotImplementedError


def _rollout_plan_runner() -> tuple[VariantRolloutRunner, _RolloutPlanDecoder]:
    decoder = _RolloutPlanDecoder()
    pipeline = SimpleNamespace(action_decoder=decoder)
    return VariantRolloutRunner(pipeline), decoder  # type: ignore[arg-type]


def test_prepared_rollout_step_forwards_state_and_updates_conditioning() -> None:
    pipeline = _PreparedPipeline()
    runner = VariantRolloutRunner(pipeline)  # type: ignore[arg-type]
    previous_state = PolicyInferState(cursor=RolloutCursor(block_index=6))
    previous_negative = torch.full((1, 1, 2), -1.0)
    session = runner.reset(
        task_text=("pick up the mug",),
        text_context=torch.zeros(1, 1, 2),
        negative_text_context=previous_negative,
    )
    session = replace(session, policy_state=previous_state)
    next_text = torch.ones(1, 1, 2)
    visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            conditioning=SimpleNamespace(
                text_context=next_text,
                negative_text_context=None,
            )
        )
    )
    state = torch.arange(4, dtype=torch.float32).view(1, 4)

    output_request = PolicyInferenceOutputRequest.video_only()
    result = runner.infer_prepared_step(
        session=session,
        context=PolicyInferContext(state=state, output_request=output_request),
        visual_outputs=visual_outputs,  # type: ignore[arg-type]
    )

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["visual_outputs"] is visual_outputs
    assert call["infer_state"] is previous_state
    resolved_context = call["context"]
    assert isinstance(resolved_context, PolicyInferContext)
    assert resolved_context.state is state
    assert resolved_context.output_request is output_request
    assert resolved_context.task_text == ("pick up the mug",)
    assert result.session.policy_state.step_index == 7
    assert result.session.task_text == ("pick up the mug",)
    assert result.session.text_context is next_text
    assert result.session.negative_text_context is previous_negative


def test_policy_variant_rejects_unimplemented_selective_outputs() -> None:
    policy = SimpleNamespace(
        inference_capabilities=PolicyInferenceCapabilities(
            native_modalities=frozenset({PolicyOutputModality.ACTION})
        )
    )

    PolicyVariant.validate_inference_output_request(
        policy,  # type: ignore[arg-type]
        PolicyInferenceOutputRequest.action_only(),
    )
    with pytest.raises(ValueError, match="does not support the requested"):
        PolicyVariant.validate_inference_output_request(
            policy,  # type: ignore[arg-type]
            PolicyInferenceOutputRequest.video_only(),
        )


def test_native_policy_inference_does_not_resolve_capabilities_without_request() -> (
    None
):
    class NativePolicy:
        @property
        def inference_capabilities(self):
            raise AssertionError("native inference resolved optional capabilities")

    PolicyVariant.validate_inference_output_request(
        NativePolicy(),  # type: ignore[arg-type]
        None,
    )


def test_pipeline_resolves_default_inference_temporal_geometry() -> None:
    geometry = PolicyTemporalGeometry(
        frame_chunk_size=4,
        attention_window_size=17,
    )
    context = VariantPipeline._resolve_temporal_geometry(
        PolicyInferContext(),
        default=geometry,
    )

    assert context.temporal_geometry == geometry


def test_pipeline_rejects_temporal_geometry_changes_within_session() -> None:
    initial = PolicyTemporalGeometry(frame_chunk_size=4, attention_window_size=30)
    state = PolicyInferState(temporal_geometry=initial)

    state.with_temporal_geometry(
        initial,
        label="test state",
    )
    with pytest.raises(ValueError, match="cannot change within an inference session"):
        state.with_temporal_geometry(
            PolicyTemporalGeometry(
                frame_chunk_size=2,
                attention_window_size=30,
            ),
            label="test state",
        )


@pytest.mark.parametrize(
    "action_mask", [None, torch.tensor([[[0.0]] * 4 + [[1.0]] * 4])]
)
def test_observed_history_reconciliation_preserves_task_conditioning(
    action_mask,
) -> None:
    pipeline = _ObservedHistoryTarget()
    runner = VariantRolloutRunner(pipeline)  # type: ignore[arg-type]
    previous_state = PolicyInferState(cursor=RolloutCursor(block_index=8))
    previous_text = torch.zeros(1, 1, 2)
    next_negative = torch.full((1, 1, 2), -2.0)
    session = runner.reset(
        task_text=("pick up the mug",),
        text_context=previous_text,
        negative_text_context=torch.full((1, 1, 2), -1.0),
    )
    session = replace(session, policy_state=previous_state)
    video_latents = torch.randn(1, 4, 2, 3, 3)
    visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            video_latents=video_latents,
            conditioning=SimpleNamespace(
                text_context=None,
                negative_text_context=next_negative,
            ),
        )
    )
    actions = torch.randn(1, 8, 7)
    proprio = torch.randn(2, 9)

    result = runner.reconcile_observed_history(
        session=session,
        history=PolicyObservedHistory(
            video_latents=visual_outputs.frontend.video_latents,
            observation_frame_count=8,
            action_history=actions,
            action_mask=action_mask,
            proprio_history=proprio,
        ),
    )

    assert len(pipeline.calls) == 1
    history, policy_state = pipeline.calls[0]
    assert policy_state is previous_state
    assert history.video_latents is video_latents
    assert history.observation_frame_count == 8
    assert history.action_history is actions
    assert history.action_mask is action_mask
    assert history.proprio_history is proprio
    assert result.session.policy_state is not previous_state
    assert result.session.policy_state.step_index == 9
    assert result.session.task_text == session.task_text
    assert result.session.text_context is previous_text
    assert result.session.negative_text_context is session.negative_text_context
    assert result.debug == {"committed": True}
    assert result.applied is True


def test_pipeline_delegates_observed_history_to_policy_owner() -> None:
    policy = _ObservedHistoryTarget()
    pipeline = SimpleNamespace(policy_variant=policy)
    policy_state = PolicyInferState(cursor=RolloutCursor(block_index=4))
    history = PolicyObservedHistory(
        video_latents=torch.randn(1, 4, 2, 3, 3),
        observation_frame_count=8,
    )

    result = VariantPipeline.reconcile_observed_history(
        pipeline,  # type: ignore[arg-type]
        history,
        policy_state,
    )

    assert len(policy.calls) == 1
    delegated_history, delegated_state = policy.calls[0]
    assert delegated_history is history
    assert delegated_state is policy_state
    assert result.next_state is not None
    assert result.next_state.step_index == 9
    assert result.debug == {"committed": True}


def test_action_decoder_rollout_plan_uses_full_action_chunk_by_default() -> None:
    runner, _ = _rollout_plan_runner()
    output = ActionDecoderInferOutput(
        action_pred=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
    )

    plan = runner.build_action_rollout_plan(output)

    assert isinstance(plan, ActionDecoderRolloutPlan)
    assert plan.actions.dtype == torch.float32
    assert plan.actions.device.type == "cpu"
    torch.testing.assert_close(
        plan.actions,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert plan.to_metadata() == {
        "action_plan_source": "decoder_action_chunk",
        "decoder_rollout_chunk_steps": None,
        "decoder_rollout_commit_start_index": None,
        "decoder_rollout_commit_end_index": None,
        "decoder_rollout_committed_actions": 2,
    }


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 4)])
def test_executable_plan_requires_one_environment(shape):
    runner, _ = _rollout_plan_runner()
    with pytest.raises(ValueError, match="one environment"):
        runner.build_action_rollout_plan(
            ActionDecoderInferOutput(action_pred=torch.zeros(shape))
        )


@pytest.mark.parametrize("space", ["raw", "auto"])
def test_decoder_plan_cannot_bypass_the_action_space_adapter(space):
    with pytest.raises(ValueError, match="model-space"):
        ActionDecoderRolloutPlan(
            actions=torch.zeros(2, 4), source="test", action_space=space
        )


def test_runner_publishes_decoder_plan_and_state_together(monkeypatch) -> None:
    from open_wam.configs import VideoActionProgram
    from tests.test_unified_policy_inference import pipeline_for

    pipeline = pipeline_for("dual_expert", VideoActionProgram.VIDEO_THEN_ACTION)
    runner = VariantRolloutRunner(pipeline)
    calls = []

    def commit(state, plan):
        calls.append((state, plan))
        return {"planned_controls": plan.actions.shape[0]}

    monkeypatch.setattr(pipeline.action_decoder, "commit_rollout_plan", commit)
    initial = runner.reset(text_context=torch.ones(1, 3, 16))
    result = runner.infer_step(
        session=initial,
        video_latents=torch.ones(1, 48, 1, 4, 4),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
    )
    assert initial.policy_state is None
    assert len(calls) == 1
    assert calls[0][1] is result.action_plan
    assert result.session.policy_state.decoder_state == {"planned_controls": 4}
    assert result.infer_output.policy_output.next_state is result.session.policy_state


def test_decoder_diagnostics_cannot_override_executable_actions() -> None:
    runner, _ = _rollout_plan_runner()
    output = ActionDecoderInferOutput(
        action_pred=torch.tensor([[[1.0], [2.0]]]),
        aux={
            "current_action": torch.tensor([99.0]),
            "current_action_index": -1,
            "rollout_chunk_steps": 1,
        },
    )
    plan = runner.build_action_rollout_plan(output)
    torch.testing.assert_close(plan.actions, torch.tensor([[1.0], [2.0]]))
