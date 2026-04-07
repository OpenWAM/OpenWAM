from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from open_wam.data import CanonicalVideoBatch, ConfiguredCanonicalVideoPreprocessor, RobotWinCanonicalVideoPreprocessor
from open_wam.models.action_decoders import ActionDecoder, ActionDecoderInferOutput, ActionDecoderTrainOutput
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVariant,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower


@dataclass
class VariantPipelineTrainOutput:
    """Combined outputs for one train-time variant pipeline step."""

    visual_outputs: VisualStageOutputs
    policy_output: PolicyTrainOutput
    decoder_output: ActionDecoderTrainOutput


@dataclass
class VariantPipelineInferOutput:
    """Combined outputs for one inference-time variant pipeline step."""

    visual_outputs: VisualStageOutputs
    policy_output: PolicyInferOutput
    decoder_output: ActionDecoderInferOutput


class VariantPipeline(nn.Module):
    """Stage-aware pipeline for visual tower plus policy variant plus decoder."""

    def __init__(
        self,
        visual_tower: VisualTower,
        policy_variant: PolicyVariant,
        action_decoder: ActionDecoder,
        preprocessor: ConfiguredCanonicalVideoPreprocessor | None = None,
    ) -> None:
        super().__init__()
        self.visual_tower = visual_tower
        self.policy_variant = policy_variant
        self.action_decoder = action_decoder
        self.preprocessor = preprocessor or RobotWinCanonicalVideoPreprocessor()

    def canonicalize(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        return self.preprocessor(views)

    def prepare_visual_outputs(
        self,
        views: Mapping[str, torch.Tensor],
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        preserve_stream_cache: bool = False,
    ) -> VisualStageOutputs:
        canonical_batch = self.canonicalize(views)
        frontend_output = self.visual_tower.run_frontend(
            canonical_batch.video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        return self._complete_visual_outputs(frontend_output)

    def prepare_visual_outputs_from_latents(
        self,
        video_latents: torch.Tensor,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
    ) -> VisualStageOutputs:
        frontend_output = self.visual_tower.run_frontend_from_latents(
            video_latents,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            canonical_video=canonical_video,
        )
        return self._complete_visual_outputs(frontend_output)

    def _complete_visual_outputs(self, frontend_output) -> VisualStageOutputs:
        requested_stages = set(self.policy_variant.required_visual_stages())
        requested_readout = self.policy_variant.requested_visual_readout()
        core_output = None
        decode_output = None
        if "core" in requested_stages:
            core_output = self.visual_tower.run_default_core(frontend_output, readout_request=requested_readout)
        if "decode" in requested_stages:
            if core_output is None:
                core_output = self.visual_tower.run_default_core(frontend_output, readout_request=requested_readout)
            decode_output = self.visual_tower.run_decode(frontend_output, core_output)
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)

    def resolve_train_decoder_output(
        self,
        policy_output: PolicyTrainOutput,
        train_batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        # Variants may optionally own the full decoder contract themselves.
        # Keep a temporary `aux["decoder_output"]` fallback for compatibility
        # with older branches while the explicit `owned_decoder_output` field
        # becomes the canonical path.
        direct_decoder_output = policy_output.owned_decoder_output
        if direct_decoder_output is None:
            direct_decoder_output = policy_output.aux.get("decoder_output")
        if isinstance(direct_decoder_output, ActionDecoderTrainOutput):
            return direct_decoder_output
        return self.action_decoder.forward_train(policy_output, train_batch)

    def resolve_infer_decoder_output(
        self,
        policy_output: PolicyInferOutput,
        *,
        previous_decoder_state: object | None = None,
    ) -> ActionDecoderInferOutput:
        # The infer-side rule mirrors the train-side rule above.
        direct_decoder_output = policy_output.owned_decoder_output
        if direct_decoder_output is None:
            direct_decoder_output = policy_output.aux.get("decoder_output")
        if isinstance(direct_decoder_output, ActionDecoderInferOutput):
            return direct_decoder_output
        return self.action_decoder.forward_infer(policy_output, previous_state=previous_decoder_state)

    def forward(
        self,
        *,
        batch: PolicyTrainBatch,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ) -> VariantPipelineTrainOutput:
        """Standard train-time forward used by distributed wrappers.

        FSDP/DDP only intercept the module's public ``forward``. Keep this as
        the single training entrypoint so distributed strategies can safely
        wrap the pipeline while preserving the existing view-based and
        latent-based execution paths.
        """

        if views is not None:
            if video_latents is not None:
                raise ValueError("Pass either `views` or `video_latents` to VariantPipeline.forward, not both.")
            return self.forward_train(views, batch)
        if video_latents is not None:
            return self.forward_train_from_latents(
                video_latents,
                batch,
                canonical_video=canonical_video,
                text_context=text_context,
                negative_text_context=negative_text_context,
            )
        raise ValueError("VariantPipeline.forward requires either `views` or `video_latents`.")

    def forward_train(
        self,
        views: Mapping[str, torch.Tensor],
        batch: PolicyTrainBatch,
    ) -> VariantPipelineTrainOutput:
        visual_outputs = self.prepare_visual_outputs(
            views,
            task_text=batch.extra.get("task_text"),
        )
        return self._forward_train_with_visual_outputs(visual_outputs, batch=batch)

    def forward_train_from_latents(
        self,
        video_latents: torch.Tensor,
        batch: PolicyTrainBatch,
        *,
        canonical_video: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ) -> VariantPipelineTrainOutput:
        visual_outputs = self.prepare_visual_outputs_from_latents(
            video_latents,
            task_text=batch.extra.get("task_text"),
            text_context=text_context,
            negative_text_context=negative_text_context,
            canonical_video=canonical_video,
        )
        return self._forward_train_with_visual_outputs(visual_outputs, batch=batch)

    def _forward_train_with_visual_outputs(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        batch: PolicyTrainBatch,
    ) -> VariantPipelineTrainOutput:
        prepared_inputs = self.policy_variant.prepare_train_inputs(visual_outputs, batch)
        policy_output = self.policy_variant.forward_train(
            visual_tower=self.visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )
        decoder_output = self.resolve_train_decoder_output(policy_output, prepared_inputs.batch)
        return VariantPipelineTrainOutput(
            visual_outputs=visual_outputs,
            policy_output=policy_output,
            decoder_output=decoder_output,
        )

    def forward_infer_step(
        self,
        views: Mapping[str, torch.Tensor],
        context: PolicyInferContext,
        infer_state: PolicyInferState | None = None,
    ) -> VariantPipelineInferOutput:
        visual_outputs = self.prepare_visual_outputs(
            views,
            task_text=context.extra.get("task_text"),
        )
        return self._forward_infer_with_visual_outputs(
            visual_outputs,
            context=context,
            infer_state=infer_state,
        )

    def _forward_infer_with_visual_outputs(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        context: PolicyInferContext,
        infer_state: PolicyInferState | None = None,
    ) -> VariantPipelineInferOutput:
        # Views and latents should share the exact same policy/decode
        # orchestration once `VisualStageOutputs` already exist. Keep this
        # helper as the single infer-side execution body so rollout behavior
        # does not drift between RGB-driven and latent-driven evaluation paths.
        resolved_state = self.policy_variant.prepare_infer_state(
            visual_tower=self.visual_tower,
            visual_outputs=visual_outputs,
            context=context,
            previous_state=infer_state,
        )
        policy_output = self.policy_variant.forward_infer_step(
            visual_tower=self.visual_tower,
            visual_outputs=visual_outputs,
            context=context,
            infer_state=resolved_state,
        )
        decoder_output = self.resolve_infer_decoder_output(
            policy_output,
            previous_decoder_state=resolved_state.decoder_state,
        )
        policy_output.next_state.decoder_state = decoder_output.next_state
        return VariantPipelineInferOutput(
            visual_outputs=visual_outputs,
            policy_output=policy_output,
            decoder_output=decoder_output,
        )

    def forward_infer_step_from_latents(
        self,
        video_latents: torch.Tensor,
        context: PolicyInferContext,
        infer_state: PolicyInferState | None = None,
        *,
        canonical_video: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
    ) -> VariantPipelineInferOutput:
        """Run one inference step from already-computed video latents.

        Joint video+action trajectory evaluation needs an open-loop mode where
        the next step consumes the previous step's predicted video latents
        rather than re-reading ground-truth RGB windows. This mirrors the
        regular inference path, but skips raw-view canonicalization.
        """

        visual_outputs = self.prepare_visual_outputs_from_latents(
            video_latents,
            task_text=context.extra.get("task_text"),
            text_context=text_context,
            negative_text_context=negative_text_context,
            canonical_video=canonical_video,
        )
        return self._forward_infer_with_visual_outputs(
            visual_outputs,
            context=context,
            infer_state=infer_state,
        )
