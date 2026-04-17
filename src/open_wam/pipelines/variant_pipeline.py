from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import nn

from open_wam.data import CanonicalVideoBatch, ConfiguredCanonicalVideoPreprocessor, RobotWinCanonicalVideoPreprocessor
from open_wam.models.action_decoders import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    DecoderRolloutState,
    DirectActionDecoderTrainInputs,
)
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
        preprocessor: ConfiguredCanonicalVideoPreprocessor | None = None,
        action_sampler_mask: torch.Tensor | None = None,
        action_sampler_inactive_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_tower = visual_tower
        self.policy_variant = policy_variant
        self.action_decoder = action_decoder
        self.preprocessor = preprocessor or RobotWinCanonicalVideoPreprocessor()
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

    def _supports_direct_train_inputs(self) -> bool:
        return bool(getattr(self.action_decoder, "supports_direct_train_inputs", lambda: False)())

    def _resolve_current_video_frame_index(self) -> int:
        current_index = getattr(self.policy_variant.config, "current_video_frame_index", 0)
        return int(current_index)

    def _build_direct_train_inputs(
        self,
        *,
        batch: PolicyTrainBatch,
        views: Mapping[str, torch.Tensor] | None,
        video_latents: torch.Tensor | None,
        canonical_video: torch.Tensor | None,
        text_context: torch.Tensor | None,
    ) -> DirectActionDecoderTrainInputs:
        input_space = str(getattr(self.action_decoder, "input_space", "video_latent"))
        current_action_index = 0
        current_frame_index = self._resolve_current_video_frame_index()
        if input_space == "video_latent":
            if video_latents is None:
                raise ValueError(
                    "Direct current-frame regression with `video_latent` input requires latent batches. "
                    "Use the latent batch adapter or choose `rgb_video` input."
                )
            current_frame = video_latents[:, :, current_frame_index]
        elif input_space == "rgb_video":
            resolved_canonical = canonical_video
            if resolved_canonical is None:
                if views is None:
                    raise ValueError(
                        "Direct current-frame regression with `rgb_video` input requires either raw views or "
                        "`canonical_video`."
                    )
                resolved_canonical = self.canonicalize(views).video
            current_frame = resolved_canonical[:, :, current_frame_index]
        else:
            raise ValueError(f"Unsupported direct-train method-4 input space {input_space!r}.")
        return DirectActionDecoderTrainInputs(
            current_frame=current_frame,
            input_space=input_space,
            current_action_index=current_action_index,
            state=batch.state,
            text_context=text_context,
            metadata={
                "current_frame_index": current_frame_index,
                "task_text": batch.extra.get("task_text"),
            },
        )

    def _forward_train_direct(
        self,
        *,
        batch: PolicyTrainBatch,
        views: Mapping[str, torch.Tensor] | None,
        video_latents: torch.Tensor | None,
        canonical_video: torch.Tensor | None,
        text_context: torch.Tensor | None,
    ) -> VariantPipelineTrainOutput:
        direct_inputs = self._build_direct_train_inputs(
            batch=batch,
            views=views,
            video_latents=video_latents,
            canonical_video=canonical_video,
            text_context=text_context,
        )
        batch_size = int(batch.actions.shape[0])
        device = direct_inputs.current_frame.device
        decoder_param = next(self.action_decoder.parameters(), None)
        policy_feature_dim = max(int(getattr(self.action_decoder, "hidden_size", 0)), 1)
        policy_feature_dtype = decoder_param.dtype if decoder_param is not None else torch.float32
        policy_output = PolicyTrainOutput(
            policy_features=torch.zeros(
                batch_size,
                0,
                policy_feature_dim,
                device=device,
                dtype=policy_feature_dtype,
            ),
            metrics={},
            aux={
                "variant": getattr(self.policy_variant.config, "name", "unknown"),
                "direct_train_mode": "current_frame_regression",
            },
        )
        decoder_output = self.action_decoder.forward_train_direct(direct_inputs, batch)
        return VariantPipelineTrainOutput(
            visual_outputs=None,
            policy_output=policy_output,
            decoder_output=decoder_output,
        )

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
            return self._apply_action_sampler_mask_to_infer_output(direct_decoder_output)
        return self._apply_action_sampler_mask_to_infer_output(
            self.action_decoder.forward_infer(policy_output, previous_state=previous_decoder_state)
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
        next_state = output.next_state
        if isinstance(next_state, DecoderRolloutState) and next_state.action_chunk is not None:
            next_state = replace(
                next_state,
                action_chunk=self._apply_action_sampler_mask(next_state.action_chunk),
            )
        return ActionDecoderInferOutput(
            action_pred=masked_action_pred,
            next_state=next_state,
            aux=aux,
        )

    def _apply_action_sampler_mask(self, actions: torch.Tensor, *, start_index: int = 0) -> torch.Tensor:
        sampler_mask = self._action_sampler_mask
        if sampler_mask is None:
            return actions
        if actions.ndim not in {2, 3}:
            raise ValueError(f"Action sampler mask supports [B, D] or [B, H, D], got {tuple(actions.shape)}.")
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
        mask = sampler_mask[int(start_index) : end_index].to(device=actions.device, dtype=actions.dtype)
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
                raise ValueError("Pass either `views` or `video_latents` to VariantPipeline.forward, not both.")
            if self._supports_direct_train_inputs():
                return self._forward_train_direct(
                    batch=batch,
                    views=views,
                    video_latents=None,
                    canonical_video=canonical_video,
                    text_context=text_context,
                )
            return self.forward_train(views, batch)
        if video_latents is not None:
            if self._supports_direct_train_inputs():
                return self._forward_train_direct(
                    batch=batch,
                    views=None,
                    video_latents=video_latents,
                    canonical_video=canonical_video,
                    text_context=text_context,
                )
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
        if self._supports_direct_train_inputs():
            return self._forward_train_direct(
                batch=batch,
                views=views,
                video_latents=None,
                canonical_video=None,
                text_context=None,
            )
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
        if self._supports_direct_train_inputs():
            return self._forward_train_direct(
                batch=batch,
                views=None,
                video_latents=video_latents,
                canonical_video=canonical_video,
                text_context=text_context,
            )
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
        context = self._prepare_infer_context_for_decoder(context)
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

    def _prepare_infer_context_for_decoder(self, context: PolicyInferContext) -> PolicyInferContext:
        uses_video_condition_window = getattr(self.action_decoder, "uses_video_condition_window", None)
        if not callable(uses_video_condition_window) or not bool(uses_video_condition_window()):
            return context
        extra = dict(context.extra)
        extra.setdefault("video_condition_source", "generated_future")
        return replace(context, extra=extra)

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
