from __future__ import annotations

from types import SimpleNamespace

import torch

from open_wam.models.policy_variants import PolicyInferContext, PolicyInferState
from open_wam.pipelines import VariantRolloutRunner


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
