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
        preserve_stream_cache: bool = False,
    ) -> VisualStageOutputs:
        canonical_batch = self.canonicalize(views)
        frontend_output = self.visual_tower.run_frontend(
            canonical_batch.video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        return self._complete_visual_outputs(frontend_output)

    def prepare_visual_outputs_from_latents(
        self,
        video_latents: torch.Tensor,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        canonical_video: torch.Tensor | None = None,
    ) -> VisualStageOutputs:
        frontend_output = self.visual_tower.run_frontend_from_latents(
            video_latents,
            task_text=task_text,
            text_context=text_context,
            canonical_video=canonical_video,
        )
        return self._complete_visual_outputs(frontend_output)

    def _complete_visual_outputs(self, frontend_output) -> VisualStageOutputs:
        requested_stages = set(self.policy_variant.required_visual_stages())
        core_output = None
        decode_output = None
        if "core" in requested_stages:
            core_output = self.visual_tower.run_default_core(frontend_output)
        if "decode" in requested_stages:
            if core_output is None:
                core_output = self.visual_tower.run_default_core(frontend_output)
            decode_output = self.visual_tower.run_decode(frontend_output, core_output)
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)

    def forward_train(
        self,
        views: Mapping[str, torch.Tensor],
        batch: PolicyTrainBatch,
    ) -> VariantPipelineTrainOutput:
        visual_outputs = self.prepare_visual_outputs(
            views,
            task_text=batch.extra.get("task_text"),
        )
        prepared_inputs = self.policy_variant.prepare_train_inputs(visual_outputs, batch)
        policy_output = self.policy_variant.forward_train(
            visual_tower=self.visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )
        decoder_output = self.action_decoder.forward_train(policy_output, prepared_inputs.batch)
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
        resolved_state = self.policy_variant.prepare_infer_state(
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
        decoder_output = self.action_decoder.forward_infer(policy_output)
        return VariantPipelineInferOutput(
            visual_outputs=visual_outputs,
            policy_output=policy_output,
            decoder_output=decoder_output,
        )
