from __future__ import annotations

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
            policy_output=SimpleNamespace(
                next_state=PolicyInferState(step_index=7),
            ),
        )


class _ObservedHistoryTarget:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def reconcile_observed_history(self, history, policy_state):
        self.calls.append((history, policy_state))
        return PolicyObservedHistoryOutput(
            next_state=PolicyInferState(step_index=9),
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
    previous_state = PolicyInferState(step_index=6)
    previous_negative = torch.full((1, 1, 2), -1.0)
    session = runner.reset(
        task_text=("pick up the mug",),
        text_context=torch.zeros(1, 1, 2),
        negative_text_context=previous_negative,
    )
    session.policy_state = previous_state
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
    assert resolved_context.extra["task_text"] == ("pick up the mug",)
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


def test_native_policy_inference_does_not_resolve_capabilities_without_request() -> None:
    class NativePolicy:
        @property
        def inference_capabilities(self):
            raise AssertionError("native inference resolved optional capabilities")

    PolicyVariant.validate_inference_output_request(
        NativePolicy(),  # type: ignore[arg-type]
        None,
    )


def test_observed_history_reconciliation_updates_policy_and_conditioning() -> None:
    pipeline = _ObservedHistoryTarget()
    runner = VariantRolloutRunner(pipeline)  # type: ignore[arg-type]
    previous_state = PolicyInferState(step_index=8)
    previous_text = torch.zeros(1, 1, 2)
    next_negative = torch.full((1, 1, 2), -2.0)
    session = runner.reset(
        task_text=("pick up the mug",),
        text_context=previous_text,
        negative_text_context=torch.full((1, 1, 2), -1.0),
    )
    session.policy_state = previous_state
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
        visual_outputs=visual_outputs,  # type: ignore[arg-type]
        observation_frame_count=8,
        action_history=actions,
        proprio_history=proprio,
        inference_window_size=30,
        rollout_frame_chunk_size=2,
    )

    assert len(pipeline.calls) == 1
    history, policy_state = pipeline.calls[0]
    assert policy_state is previous_state
    assert history.video_latents is video_latents
    assert history.observation_frame_count == 8
    assert history.action_history is actions
    assert history.proprio_history is proprio
    assert history.inference_window_size == 30
    assert history.rollout_frame_chunk_size == 2
    assert result.session.policy_state is not previous_state
    assert result.session.policy_state.step_index == 9
    assert result.session.task_text == session.task_text
    assert result.session.text_context is previous_text
    assert result.session.negative_text_context is next_negative
    assert result.debug == {"committed": True}
    assert result.applied is True


def test_pipeline_delegates_observed_history_to_policy_owner() -> None:
    policy = _ObservedHistoryTarget()
    pipeline = SimpleNamespace(policy_variant=policy)
    policy_state = PolicyInferState(step_index=4)
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


def test_action_decoder_rollout_plan_commits_configured_cached_chunk() -> None:
    runner, _ = _rollout_plan_runner()
    decoder_state = SimpleNamespace(step_within_chunk=1)
    session = runner.reset()
    session.policy_state = PolicyInferState(decoder_state=decoder_state)
    output = ActionDecoderInferOutput(
        action_pred=torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        next_state=SimpleNamespace(
            step_within_chunk=1,
            aux={"rollout_chunk_steps": 6},
        ),
        aux={
            "current_action": torch.tensor([0.0]),
            "current_action_index": torch.tensor(0.0),
            "rollout_chunk_steps": torch.tensor(6.0),
        },
    )

    plan = runner.build_action_rollout_plan(output)
    runner.commit_action_rollout_plan(session=session, plan=plan)

    torch.testing.assert_close(plan.actions[:, 0], torch.arange(6, dtype=torch.float32))
    assert plan.to_metadata()["decoder_rollout_commit_end_index"] == 6
    assert decoder_state.step_within_chunk == 6


def test_action_decoder_rollout_plan_uses_remaining_cached_actions() -> None:
    runner, _ = _rollout_plan_runner()
    output = ActionDecoderInferOutput(
        action_pred=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        next_state=SimpleNamespace(
            step_within_chunk=2,
            aux={"rollout_chunk_steps": 2},
        ),
        aux={"current_action": torch.tensor([[3.0, 4.0]])},
    )

    plan = runner.build_action_rollout_plan(output)

    assert plan.source == "decoder_current_action_rollout_chunk"
    assert plan.commit_start_index == 1
    assert plan.commit_end_index == 2
    torch.testing.assert_close(plan.actions, torch.tensor([[3.0, 4.0]]))


def test_action_decoder_rollout_plan_falls_back_to_current_action_after_horizon() -> None:
    runner, _ = _rollout_plan_runner()
    output = ActionDecoderInferOutput(
        action_pred=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        aux={
            "current_action": torch.tensor([9.0, 10.0]),
            "current_action_index": 8,
            "rollout_chunk_steps": 2,
        },
    )

    plan = runner.build_action_rollout_plan(output)

    assert plan.source == "decoder_current_action"
    assert plan.commit_start_index == 8
    assert plan.commit_end_index == 9
    torch.testing.assert_close(plan.actions, torch.tensor([[9.0, 10.0]]))


def test_action_decoder_rollout_plan_rejects_negative_current_action_index() -> None:
    runner, _ = _rollout_plan_runner()
    output = ActionDecoderInferOutput(
        action_pred=torch.tensor([[[1.0, 2.0]]]),
        aux={
            "current_action": torch.tensor([1.0, 2.0]),
            "current_action_index": -1,
        },
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        runner.build_action_rollout_plan(output)


def test_variant_rollout_runner_delegates_custom_decoder_plan_and_commit() -> None:
    class CustomPlanDecoder(_RolloutPlanDecoder):
        def build_rollout_plan(self, output):
            del output
            return ActionDecoderRolloutPlan(
                actions=torch.tensor([[42.0]], dtype=torch.float32),
                source="custom_decoder",
                commit_start_index=4,
                commit_end_index=5,
            )

        def commit_rollout_plan(self, state, plan):
            state.committed_source = plan.source

    decoder = CustomPlanDecoder()
    runner = VariantRolloutRunner(SimpleNamespace(action_decoder=decoder))  # type: ignore[arg-type]
    decoder_state = SimpleNamespace(committed_source=None)
    session = runner.reset()
    session.policy_state = PolicyInferState(decoder_state=decoder_state)

    plan = runner.build_action_rollout_plan(
        ActionDecoderInferOutput(action_pred=torch.zeros(1, 1, 1)),
    )
    runner.commit_action_rollout_plan(session=session, plan=plan)

    assert plan.source == "custom_decoder"
    assert plan.actions.item() == 42.0
    assert decoder_state.committed_source == "custom_decoder"
