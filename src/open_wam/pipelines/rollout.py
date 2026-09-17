from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class VariantRolloutSession:
    """Shared rollout session for stateless and cache-aware variants."""

    policy_state: PolicyInferState | None = None
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None


@dataclass(frozen=True)
class VariantRolloutStepOutput:
    """One rollout step plus the next reusable session."""

    session: VariantRolloutSession
    infer_output: VariantPipelineInferOutput
    action_plan: ActionDecoderRolloutPlan | None = None


@dataclass(frozen=True)
class VariantRolloutHistoryOutput:
    """Updated session and diagnostics after committing observed history."""

    session: VariantRolloutSession
    debug: dict[str, object]
    applied: bool = False


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
                raise ValueError(
                    "VariantRolloutRunner.infer_step requires either `views` or `video_latents`."
                )
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
        history: PolicyObservedHistory,
    ) -> VariantRolloutHistoryOutput:
        """Publish an executed interval without changing task conditioning.

        Text belongs to the inference request/session, not the observation
        commit. All history representations cross the same typed boundary.
        """
        update = self.pipeline.reconcile_observed_history(history, session.policy_state)
        return VariantRolloutHistoryOutput(
            session=replace(session, policy_state=update.next_state),
            debug=dict(update.debug),
            applied=bool(update.applied),
        )

    @staticmethod
    def _resolve_context(
        session: VariantRolloutSession,
        context: PolicyInferContext,
    ) -> PolicyInferContext:
        return replace(
            context,
            task_text=context.task_text
            if context.task_text is not None
            else session.task_text,
        )

    def _build_step_output(
        self,
        *,
        session: VariantRolloutSession,
        resolved_context: PolicyInferContext,
        infer_output: VariantPipelineInferOutput,
    ) -> VariantRolloutStepOutput:
        next_session = VariantRolloutSession(
            policy_state=infer_output.policy_output.next_state,
            task_text=resolved_context.task_text,
            text_context=(
                infer_output.visual_outputs.frontend.conditioning.text_context
                if infer_output.visual_outputs.frontend.conditioning.text_context
                is not None
                else session.text_context
            ),
            negative_text_context=(
                infer_output.visual_outputs.frontend.conditioning.negative_text_context
                if infer_output.visual_outputs.frontend.conditioning.negative_text_context
                is not None
                else session.negative_text_context
            ),
        )
        actions = infer_output.decoder_output.action_pred
        plan = None
        if actions.shape[1] and actions.shape[-1]:
            plan = replace(
                self.build_action_rollout_plan(infer_output.decoder_output),
                frame_span=infer_output.policy_output.generated_span,
            )
            state = next_session.policy_state
            state = replace(
                state,
                decoder_state=self.pipeline.action_decoder.commit_rollout_plan(
                    state.decoder_state,
                    plan,
                ),
            )
            next_session = replace(next_session, policy_state=state)
            infer_output = replace(
                infer_output,
                policy_output=replace(
                    infer_output.policy_output,
                    next_state=state,
                ),
            )
        return VariantRolloutStepOutput(
            session=next_session,
            infer_output=infer_output,
            action_plan=plan,
        )
