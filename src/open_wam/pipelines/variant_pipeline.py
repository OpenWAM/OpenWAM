from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from open_wam.data import (
    CanonicalVideoBatch,
    ConfiguredCanonicalVideoPreprocessor,
)
from open_wam.models.action_decoders import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
)
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyModuleTopology,
    PolicyObservedHistory,
    PolicyObservedHistoryOutput,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVariant,
    PolicyVisualStage,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower


@dataclass
class VariantPipelineTrainOutput:
    """Combined outputs for one train-time variant pipeline step."""

    visual_outputs: VisualStageOutputs | None
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
        preprocessor: ConfiguredCanonicalVideoPreprocessor,
        action_sampler_mask: torch.Tensor | None = None,
        action_sampler_inactive_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_tower = visual_tower
        self.policy_variant = policy_variant
        self.action_decoder = action_decoder
        self._validate_decoder_artifact_contract()
        self.preprocessor = preprocessor
        self.action_sampler_inactive_value = float(action_sampler_inactive_value)
        if action_sampler_mask is not None:
            self.register_buffer(
                "_action_sampler_mask",
                action_sampler_mask.detach().to(dtype=torch.float32),
                persistent=False,
            )
        else:
            self._action_sampler_mask = None
        self.action_decoder.configure_action_sampler_mask(
            self._action_sampler_mask,
            inactive_value=self.action_sampler_inactive_value,
        )

    def _validate_decoder_artifact_contract(self) -> None:
        produced = self.policy_variant.decoder_artifact_contract
        consumed = self.action_decoder.decoder_artifact_contract
        if produced == consumed:
            return
        raise ValueError(
            "Policy and action decoder artifact contracts do not match: "
            f"policy emits {produced!r}, decoder accepts {consumed!r}. "
            "Select a decoder compatible with the configured policy architecture."
        )

    def module_topology(self) -> PolicyModuleTopology:
        """Return the active variant's module ownership contract."""

        return self.policy_variant.module_topology(self.visual_tower)

    def on_checkpoint_loaded(
        self,
        *,
        loaded_state_keys: frozenset[str],
        missing_state_keys: frozenset[str],
    ) -> None:
        """Forward policy-local checkpoint lifecycle state without key leakage."""

        prefix = "policy_variant."

        def _policy_keys(keys: frozenset[str]) -> frozenset[str]:
            return frozenset(
                key.removeprefix(prefix) for key in keys if key.startswith(prefix)
            )

        self.policy_variant.on_checkpoint_loaded(
            loaded_state_keys=_policy_keys(loaded_state_keys),
            missing_state_keys=_policy_keys(missing_state_keys),
        )

    def reconcile_observed_history(
        self,
        history: PolicyObservedHistory,
        infer_state: PolicyInferState | None,
    ) -> PolicyObservedHistoryOutput:
        """Delegate observed-history semantics to the active policy variant."""

        return self.policy_variant.reconcile_observed_history(
            history,
            infer_state,
        )

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
        try:
            requested_stages = {
                PolicyVisualStage(stage)
                for stage in self.policy_variant.required_visual_stages()
            }
        except ValueError as exc:
            supported = ", ".join(stage.value for stage in PolicyVisualStage)
            raise ValueError(
                "Policy requested an unsupported visual stage; "
                f"supported stages: {supported}."
            ) from exc
        core_output = None
        if PolicyVisualStage.CORE in requested_stages:
            core_output = self.visual_tower.run_default_core(frontend_output)
        return VisualStageOutputs(frontend=frontend_output, core=core_output)

    def resolve_train_decoder_output(
        self,
        policy_output: PolicyTrainOutput,
        train_batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        return self.action_decoder.forward_train(policy_output, train_batch)

    def resolve_infer_decoder_output(
        self,
        policy_output: PolicyInferOutput,
        *,
        previous_decoder_state: object | None = None,
    ) -> ActionDecoderInferOutput:
        return self._apply_action_sampler_mask_to_infer_output(
            self.action_decoder.forward_infer(
                policy_output, previous_state=previous_decoder_state
            )
        )

    def _apply_action_sampler_mask_to_infer_output(
        self,
        output: ActionDecoderInferOutput,
    ) -> ActionDecoderInferOutput:
        if self._action_sampler_mask is None:
            return output
        masked_action_pred = self._apply_action_sampler_mask(output.action_pred)
        aux = dict(output.aux)
        current_action = aux.get("current_action")
        if isinstance(current_action, torch.Tensor):
            aux["current_action"] = self._apply_action_sampler_mask(
                current_action,
                start_index=self._current_action_index_from_aux(aux),
            )
        return ActionDecoderInferOutput(
            action_pred=masked_action_pred,
            next_state=output.next_state,
            aux=aux,
        )

    def _apply_action_sampler_mask(
        self, actions: torch.Tensor, *, start_index: int = 0
    ) -> torch.Tensor:
        sampler_mask = self._action_sampler_mask
        if sampler_mask is None:
            return actions
        if actions.ndim not in {2, 3}:
            raise ValueError(
                f"Action sampler mask supports [B, D] or [B, H, D], got {tuple(actions.shape)}."
            )
        if actions.shape[-1] != sampler_mask.shape[-1]:
            raise ValueError(
                f"Action sampler mask dim {sampler_mask.shape[-1]} does not match action dim {actions.shape[-1]}."
            )
        horizon = actions.shape[-2] if actions.ndim == 3 else 1
        end_index = int(start_index) + int(horizon)
        if end_index > sampler_mask.shape[0]:
            raise ValueError(
                "Action sampler mask horizon is shorter than the requested action slice, "
                f"got mask_horizon={sampler_mask.shape[0]}, start_index={start_index}, horizon={horizon}."
            )
        mask = sampler_mask[int(start_index) : end_index].to(
            device=actions.device, dtype=actions.dtype
        )
        if actions.ndim == 3:
            mask = mask.unsqueeze(0)
        inactive = actions.new_full((), self.action_sampler_inactive_value)
        return actions * mask + inactive * (1.0 - mask)

    @staticmethod
    def _current_action_index_from_aux(aux: dict[str, object]) -> int:
        value = aux.get("current_action_index", 0)
        if isinstance(value, torch.Tensor):
            return int(value.detach().float().cpu().item())
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

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
                raise ValueError(
                    "Pass either `views` or `video_latents` to VariantPipeline.forward, not both."
                )
            return self.forward_train(views, batch)
        if video_latents is not None:
            return self.forward_train_from_latents(
                video_latents,
                batch,
                canonical_video=canonical_video,
                text_context=text_context,
                negative_text_context=negative_text_context,
            )
        raise ValueError(
            "VariantPipeline.forward requires either `views` or `video_latents`."
        )

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
        prepared_inputs = self.policy_variant.prepare_train_inputs(
            visual_outputs, batch
        )
        policy_output = self.policy_variant.forward_train(
            visual_tower=self.visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )
        decoder_output = self.resolve_train_decoder_output(
            policy_output, prepared_inputs.batch
        )
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
        return self.forward_infer_step_from_visual_outputs(
            visual_outputs,
            context=context,
            infer_state=infer_state,
        )

    def forward_infer_step_from_visual_outputs(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        context: PolicyInferContext,
        infer_state: PolicyInferState | None = None,
    ) -> VariantPipelineInferOutput:
        """Run policy and decoder inference from prepared visual stages.

        Runtime integrations that manage frontend/cache execution separately
        can use this entry point without reaching into pipeline internals.
        RGB- and latent-driven inference delegate here after preparing the same
        :class:`VisualStageOutputs` contract.
        """

        self.policy_variant.validate_inference_output_request(context.output_request)
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
        return self.forward_infer_step_from_visual_outputs(
            visual_outputs,
            context=context,
            infer_state=infer_state,
        )
