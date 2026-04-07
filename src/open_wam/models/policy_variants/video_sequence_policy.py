from __future__ import annotations

import torch

from open_wam.configs import (
    InferenceConfig,
    TrainingConfig,
    TrainingComponentSelector,
    VideoSequencePolicyConfig,
    VisualReadoutSourceFamily,
    VisualStateSource,
)
from open_wam.models.visual_tower import VisualReadoutRequest, VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import (
    SharedVisualReadout,
    advance_default_runtime_infer_state,
    prepare_default_runtime_infer_state,
)
from .common.layouts import align_sequence_length, pool_frame_tokens, tokens_to_frame_major
from .contracts import (
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class VideoSequencePolicyVariant(PolicyVariant):
    """Sequence-preserving policy over denoised-video or shared-core states."""

    def __init__(
        self,
        config: VideoSequencePolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_horizon: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.visual_readout = SharedVisualReadout(config.visual_readout, hidden_size=config.hidden_size)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        if self.config.visual_readout is not None:
            if self.config.visual_readout.source_family in {
                VisualReadoutSourceFamily.FINAL_CORE_TOKENS,
                VisualReadoutSourceFamily.CORE_LAYER_TOKENS,
                VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS,
            }:
                return ("frontend", "core")
            return ("frontend",)
        if self.config.visual_state_source == VisualStateSource.CORE_TOKENS:
            return ("frontend", "core")
        return ("frontend",)

    def requested_visual_readout(self) -> VisualReadoutRequest | None:
        return self.visual_readout.requested_capture()

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        del visual_outputs
        return PolicyPreparedInputs(batch=batch)

    def _build_goal_features(self, visual_outputs: VisualStageOutputs):
        if not self.config.use_goal_context:
            return None
        return visual_outputs.frontend.conditioning.text_context

    @staticmethod
    def _observed_prefix_frames() -> int:
        return 1

    def _backbone_trainable(self) -> bool:
        return TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE in self.training_config.trainable_components

    def _split_window_latents(self, video_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_prefix_frames = self._observed_prefix_frames()
        if video_latents.shape[2] <= observed_prefix_frames:
            raise ValueError(
                "Video-sequence policy requires at least one future latent frame after the observed prefix, "
                f"got video_latents.shape={tuple(video_latents.shape)}."
            )
        observed_prefix = video_latents[:, :, :observed_prefix_frames]
        future_latents = video_latents[:, :, observed_prefix_frames:]
        return observed_prefix, future_latents

    def _clean_future_visual_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
    ) -> tuple[torch.Tensor, object, torch.Tensor]:
        _, future_latents = self._split_window_latents(visual_outputs.frontend.video_latents)
        future_tokens, future_token_grid = visual_tower.frontend.tokenize_video_latents(future_latents)
        frame_tokens = tokens_to_frame_major(future_tokens, future_token_grid)
        return frame_tokens, future_token_grid, future_latents

    def _resolve_visual_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        infer_state: PolicyInferState | None = None,
    ) -> tuple[torch.Tensor, object, torch.Tensor, str, dict[str, object]]:
        if self.config.visual_readout is not None:
            source_family = self.config.visual_readout.source_family
            if source_family in {
                VisualReadoutSourceFamily.FINAL_CORE_TOKENS,
                VisualReadoutSourceFamily.CORE_LAYER_TOKENS,
                VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS,
            }:
                if visual_outputs.core is None:
                    raise ValueError("Configured visual readout requires shared core outputs.")
                resolved_readout = self.visual_readout.resolve_from_core(visual_outputs.core)
                frame_tokens = tokens_to_frame_major(resolved_readout.tokens, visual_outputs.frontend.token_grid)
                predicted_latents = visual_tower.project_video_tokens_to_latents(
                    hidden_states=resolved_readout.tokens,
                    token_grid=visual_outputs.frontend.token_grid,
                )
                return (
                    frame_tokens,
                    visual_outputs.frontend.token_grid,
                    predicted_latents,
                    resolved_readout.source_stage,
                    resolved_readout.metadata,
                )
            if source_family == VisualReadoutSourceFamily.GENERATED_FUTURE_TOKENS:
                denoised_latents = self._run_video_denoise(
                    visual_tower,
                    visual_outputs,
                    frame_start=0 if infer_state is None else int(infer_state.cursor.current_start_frame),
                )
                denoised_tokens, denoised_token_grid = visual_tower.frontend.tokenize_video_latents(denoised_latents)
                frame_tokens = tokens_to_frame_major(denoised_tokens, denoised_token_grid)
                return (
                    frame_tokens,
                    denoised_token_grid,
                    denoised_latents,
                    "generated_future",
                    {"source_family": source_family},
                )
            if source_family == VisualReadoutSourceFamily.DIFFUSION_FEATURE_TOKENS:
                frame_tokens, token_grid, predicted_latents = visual_tower.extract_diffusion_feature_readout(
                    frontend_output=visual_outputs.frontend,
                    readout_config=self.config.visual_readout,
                )
                return (
                    frame_tokens,
                    token_grid,
                    predicted_latents,
                    "diffusion_feature",
                    {"source_family": source_family},
                )
            raise ValueError(f"Unsupported visual readout source family {source_family!r}.")
        if self.config.visual_state_source == VisualStateSource.CORE_TOKENS:
            if visual_outputs.core is None:
                raise ValueError("Video-sequence policy requires shared core outputs when `visual_state_source=core_tokens`.")
            frame_tokens = tokens_to_frame_major(visual_outputs.core.tokens, visual_outputs.frontend.token_grid)
            predicted_latents = visual_tower.project_video_tokens_to_latents(
                hidden_states=visual_outputs.core.tokens,
                token_grid=visual_outputs.frontend.token_grid,
            )
            return frame_tokens, visual_outputs.frontend.token_grid, predicted_latents, "core", {
                "source_family": VisualReadoutSourceFamily.FINAL_CORE_TOKENS,
            }

        if infer_state is None and not self._backbone_trainable():
            frame_tokens, token_grid, predicted_latents = self._clean_future_visual_state(visual_tower, visual_outputs)
            return frame_tokens, token_grid, predicted_latents, "clean_future", {
                "source_family": VisualStateSource.DENOISED_VIDEO_TOKENS,
            }
        if visual_outputs.frontend.conditioning.text_context is None and not self._backbone_trainable():
            frame_tokens, token_grid, predicted_latents = self._clean_future_visual_state(visual_tower, visual_outputs)
            return frame_tokens, token_grid, predicted_latents, "clean_future", {
                "source_family": VisualStateSource.DENOISED_VIDEO_TOKENS,
                "fallback_reason": "missing_text_context",
            }

        denoised_latents = self._run_video_denoise(
            visual_tower,
            visual_outputs,
            frame_start=0 if infer_state is None else int(infer_state.cursor.current_start_frame),
        )
        denoised_tokens, denoised_token_grid = visual_tower.frontend.tokenize_video_latents(denoised_latents)
        frame_tokens = tokens_to_frame_major(denoised_tokens, denoised_token_grid)
        return frame_tokens, denoised_token_grid, denoised_latents, "denoised_future", {
            "source_family": VisualStateSource.DENOISED_VIDEO_TOKENS,
        }

    def _run_video_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        frame_start: int,
    ) -> torch.Tensor:
        if frame_start == 0 and self._backbone_trainable():
            raise NotImplementedError(
                "Method-3 denoise training with a trainable visual backbone is not implemented yet. "
                "Use the frozen-backbone clean-future path for training, or extend the reference runtime "
                "with a differentiable future-latent denoise path."
            )
        observed_prefix, target_future_latents = self._split_window_latents(visual_outputs.frontend.video_latents)
        text_emb = visual_outputs.frontend.conditioning.text_context
        if text_emb is None:
            batch_size = visual_outputs.frontend.video_latents.shape[0]
            text_emb = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=visual_outputs.frontend.video_latents.device,
                dtype=visual_outputs.frontend.video_latents.dtype,
            )
        return visual_tower.generate_conditioned_future_latents(
            observed_prefix=observed_prefix,
            future_template=target_future_latents,
            text_context=text_emb,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            frame_start=frame_start,
            num_inference_steps=self.inference_config.video_num_inference_steps,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
            sigma_shift=self.training_config.video_sigma_shift,
            guidance_scale=self.inference_config.guidance_scale,
            denoise_ratio=float(self.config.visual_denoise_ratio),
            cache_name="video_sequence_policy_denoise_state",
        )

    def _build_policy_features(
        self,
        frame_tokens: torch.Tensor,
    ) -> torch.Tensor:
        frame_features = pool_frame_tokens(frame_tokens, mode="mean")
        return align_sequence_length(frame_features, self.action_horizon)

    def _build_decoder_sequence_context(
        self,
        *,
        frame_tokens: torch.Tensor,
        token_grid,
        visual_outputs: VisualStageOutputs,
        state,
        source_stage: str,
        readout_metadata: dict[str, object],
    ) -> DecoderSequenceContext:
        return DecoderSequenceContext(
            sequence_tokens=frame_tokens,
            sequence_layout={
                "family": "video_sequence_policy",
                "kind": "frame_token_grid",
                "attach_site": str(self.config.attach_site),
                "temporal_projection": str(self.config.temporal_projection),
                "visual_state_source": str(self.config.visual_state_source),
                "visual_denoise_ratio": float(self.config.visual_denoise_ratio),
                **readout_metadata,
            },
            token_grid=token_grid,
            frame_count=int(frame_tokens.shape[1]),
            source_stage=source_stage,
            state_sequence=(state if self.config.use_state_context else None),
            goal_features=self._build_goal_features(visual_outputs),
            aux_features={
                "negative_goal_features": visual_outputs.frontend.conditioning.negative_text_context,
            },
        )

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        frame_tokens, token_grid, predicted_latents, source_stage, readout_metadata = self._resolve_visual_state(
            visual_tower,
            visual_outputs,
        )
        policy_features = self._build_policy_features(frame_tokens)
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={
                "policy_feature_norm": policy_features.norm(dim=-1).mean().detach(),
                "visual_denoise_ratio": torch.tensor(float(self.config.visual_denoise_ratio), device=policy_features.device),
            },
            decoder_sequence_context=self._build_decoder_sequence_context(
                frame_tokens=frame_tokens,
                token_grid=token_grid,
                visual_outputs=visual_outputs,
                state=prepared_inputs.batch.state,
                source_stage=source_stage,
                readout_metadata=readout_metadata,
            ),
            aux={
                "variant": self.config.name,
                "method_family": "video_sequence_policy",
                "predicted_latents": predicted_latents,
                "visual_readout": readout_metadata,
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del visual_outputs, context
        return prepare_default_runtime_infer_state(
            visual_tower,
            previous_state=previous_state,
            stage="video_sequence_policy",
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        frame_tokens, token_grid, predicted_latents, source_stage, readout_metadata = self._resolve_visual_state(
            visual_tower,
            visual_outputs,
            infer_state=infer_state,
        )
        policy_features = self._build_policy_features(frame_tokens)
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=advance_default_runtime_infer_state(
                visual_tower,
                infer_state=infer_state,
                stage="video_sequence_policy",
            ),
            decoder_sequence_context=self._build_decoder_sequence_context(
                frame_tokens=frame_tokens,
                token_grid=token_grid,
                visual_outputs=visual_outputs,
                state=context.state,
                source_stage=source_stage,
                readout_metadata=readout_metadata,
            ),
            aux={
                "variant": self.config.name,
                "method_family": "video_sequence_policy",
                "predicted_latents": predicted_latents,
                "visual_readout": readout_metadata,
            },
        )
