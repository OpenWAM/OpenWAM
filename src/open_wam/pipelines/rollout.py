from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from open_wam.models.policy_variants import PolicyInferContext, PolicyInferState

from .variant_pipeline import VariantPipeline, VariantPipelineInferOutput


@dataclass
class VariantRolloutSession:
    """Shared rollout session for stateless and cache-aware variants."""

    policy_state: PolicyInferState | None = None
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None


@dataclass
class VariantRolloutStepOutput:
    """One rollout step plus the next reusable session."""

    session: VariantRolloutSession
    infer_output: VariantPipelineInferOutput


class VariantRolloutRunner:
    """Shared reset/step interface for rollout-capable pipelines."""

    def __init__(self, pipeline: VariantPipeline) -> None:
        self.pipeline = pipeline

    def reset(
        self,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ) -> VariantRolloutSession:
        return VariantRolloutSession(
            policy_state=None,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )

    def infer_step(
        self,
        *,
        session: VariantRolloutSession,
        context: PolicyInferContext,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
    ) -> VariantRolloutStepOutput:
        resolved_context = PolicyInferContext(
            state=context.state,
            previous_action=context.previous_action,
            extra={
                **context.extra,
                "task_text": context.extra.get("task_text", session.task_text),
            },
        )
        if video_latents is not None:
            infer_output = self.pipeline.forward_infer_step_from_latents(
                video_latents,
                resolved_context,
                infer_state=session.policy_state,
                canonical_video=canonical_video,
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
            )
        else:
            if views is None:
                raise ValueError("VariantRolloutRunner.infer_step requires either `views` or `video_latents`.")
            infer_output = self.pipeline.forward_infer_step(
                views,
                resolved_context,
                infer_state=session.policy_state,
            )
        next_session = VariantRolloutSession(
            policy_state=infer_output.policy_output.next_state,
            task_text=resolved_context.extra.get("task_text", session.task_text),
            text_context=(
                infer_output.visual_outputs.frontend.conditioning.text_context
                if infer_output.visual_outputs.frontend.conditioning.text_context is not None
                else session.text_context
            ),
            negative_text_context=(
                infer_output.visual_outputs.frontend.conditioning.negative_text_context
                if infer_output.visual_outputs.frontend.conditioning.negative_text_context is not None
                else session.negative_text_context
            ),
        )
        return VariantRolloutStepOutput(session=next_session, infer_output=infer_output)
