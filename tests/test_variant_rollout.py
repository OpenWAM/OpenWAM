from __future__ import annotations

from types import SimpleNamespace

import torch

from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferState,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
)
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
        )


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

    result = runner.infer_prepared_step(
        session=session,
        context=PolicyInferContext(state=state),
        visual_outputs=visual_outputs,  # type: ignore[arg-type]
    )

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["visual_outputs"] is visual_outputs
    assert call["infer_state"] is previous_state
    resolved_context = call["context"]
    assert isinstance(resolved_context, PolicyInferContext)
    assert resolved_context.state is state
    assert resolved_context.extra["task_text"] == ("pick up the mug",)
    assert result.session.policy_state.step_index == 7
    assert result.session.task_text == ("pick up the mug",)
    assert result.session.text_context is next_text
    assert result.session.negative_text_context is previous_negative


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
