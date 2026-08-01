from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from open_wam.models.action_decoders import (
    ActionDecoderInferOutput,
    ActionDecoderRolloutPlan,
)
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferState,
    PolicyObservedHistory,
)
from open_wam.models.visual_tower import VisualStageOutputs

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


@dataclass
class VariantRolloutHistoryOutput:
    """Updated session and diagnostics after committing observed history."""

    session: VariantRolloutSession
    debug: dict[str, object]


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

    def build_action_rollout_plan(
        self,
        output: ActionDecoderInferOutput,
    ) -> ActionDecoderRolloutPlan:
        """Delegate model-space rollout slicing to the configured decoder."""

        return self.pipeline.action_decoder.build_rollout_plan(output)

    def commit_action_rollout_plan(
        self,
        *,
        session: VariantRolloutSession,
        plan: ActionDecoderRolloutPlan,
    ) -> None:
        """Commit every action released by `plan` to decoder-owned state."""

        policy_state = session.policy_state
        decoder_state = None if policy_state is None else policy_state.decoder_state
        self.pipeline.action_decoder.commit_rollout_plan(decoder_state, plan)

    def infer_step(
        self,
        *,
        session: VariantRolloutSession,
        context: PolicyInferContext,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
    ) -> VariantRolloutStepOutput:
        resolved_context = self._resolve_context(session, context)
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
        return self._build_step_output(
            session=session,
            resolved_context=resolved_context,
            infer_output=infer_output,
        )

    def infer_prepared_step(
        self,
        *,
        session: VariantRolloutSession,
        context: PolicyInferContext,
        visual_outputs: VisualStageOutputs,
    ) -> VariantRolloutStepOutput:
        """Advance a session from visual stages prepared by a runtime integration."""

        resolved_context = self._resolve_context(session, context)
        infer_output = self.pipeline.forward_infer_step_from_visual_outputs(
            visual_outputs,
            context=resolved_context,
            infer_state=session.policy_state,
        )
        return self._build_step_output(
            session=session,
            resolved_context=resolved_context,
            infer_output=infer_output,
        )

    def reconcile_observed_history(
        self,
        *,
        session: VariantRolloutSession,
        visual_outputs: VisualStageOutputs,
        observation_frame_count: int,
        action_history: torch.Tensor | None = None,
        proprio_history: torch.Tensor | None = None,
        inference_window_size: int | None = None,
        rollout_frame_chunk_size: int | None = None,
    ) -> VariantRolloutHistoryOutput:
        """Replace speculative policy history with newly observed execution."""

        update = self.pipeline.reconcile_observed_history(
            PolicyObservedHistory(
                video_latents=visual_outputs.frontend.video_latents,
                observation_frame_count=int(observation_frame_count),
                action_history=action_history,
                proprio_history=proprio_history,
                inference_window_size=inference_window_size,
                rollout_frame_chunk_size=rollout_frame_chunk_size,
            ),
            session.policy_state,
        )
        next_session = VariantRolloutSession(
            policy_state=update.next_state,
            task_text=session.task_text,
            text_context=(
                visual_outputs.frontend.conditioning.text_context
                if visual_outputs.frontend.conditioning.text_context is not None
                else session.text_context
            ),
            negative_text_context=(
                visual_outputs.frontend.conditioning.negative_text_context
                if visual_outputs.frontend.conditioning.negative_text_context
                is not None
                else session.negative_text_context
            ),
        )
        return VariantRolloutHistoryOutput(
            session=next_session,
            debug=dict(update.debug),
        )

    @staticmethod
    def _resolve_context(
        session: VariantRolloutSession,
        context: PolicyInferContext,
    ) -> PolicyInferContext:
        return PolicyInferContext(
            state=context.state,
            previous_action=context.previous_action,
            extra={
                **context.extra,
                "task_text": context.extra.get("task_text", session.task_text),
            },
        )

    @staticmethod
    def _build_step_output(
        *,
        session: VariantRolloutSession,
        resolved_context: PolicyInferContext,
        infer_output: VariantPipelineInferOutput,
    ) -> VariantRolloutStepOutput:
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
