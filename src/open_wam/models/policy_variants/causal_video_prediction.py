from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs import InferenceConfig, TextConditioningMode, TrainingConfig
from open_wam.configs.policy_contracts import CausalVideoPredictionPolicyConfig
from open_wam.contracts import VideoFrameMapping
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
    sample_timestep_id,
)
from open_wam.models.common.flow_supervision import (
    denoised_video_latents_from_flow,
)
from open_wam.models.video_backbone.contracts import TokenGridMetadata
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .contracts import (
    DecoderArtifactEnvelope,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPipelineRequirements,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVisualStage,
    RolloutCursor,
)
from .video_flow_artifacts import (
    VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
    VideoFlowInferArtifacts,
    VideoFlowTrainArtifacts,
)


@dataclass(frozen=True)
class _PrefixSuffixLayout:
    observed_frames: int
    future_frames: int
    total_frames: int


class CausalVideoPredictionPolicyVariant(PolicyVariant):
    """Standalone causal prefix/suffix video prediction over the shared visual backbone."""

    def __init__(
        self,
        config: CausalVideoPredictionPolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config

    @property
    def decoder_artifact_contract(self) -> str:
        return VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT

    def pipeline_requirements(
        self,
        *,
        default_action_dim: int,
        default_action_horizon: int,
        default_state_dim: int,
    ) -> PolicyPipelineRequirements:
        conditioning = self.config.conditioning_requirements
        return PolicyPipelineRequirements(
            action_dim=default_action_dim,
            action_horizon=default_action_horizon,
            state_dim=default_state_dim,
            text_conditioning_mode=conditioning.text_conditioning_mode,
        )

    @staticmethod
    def _validate_text_tensor(
        value: torch.Tensor | None,
        *,
        batch_size: int,
        description: str,
        require_nonzero: bool,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(
                f"Causal video prediction requires {description} embeddings."
            )
        if (
            value.ndim != 3
            or int(value.shape[0]) != batch_size
            or int(value.shape[1]) <= 0
            or int(value.shape[2]) <= 0
        ):
            raise ValueError(
                f"Causal video prediction expects {description} embeddings with "
                f"shape [B, tokens, dim], got {tuple(value.shape)}."
            )
        flattened = value.detach().reshape(batch_size, -1)
        if not bool(torch.isfinite(flattened).all()):
            raise ValueError(
                f"Causal video {description} embeddings must contain finite values."
            )
        if require_nonzero and bool((flattened.abs().sum(dim=1) == 0).any()):
            raise ValueError(
                "Causal video prediction received an all-zero text embedding "
                f"for {description}; a real encoded embedding is required."
            )
        return value

    def _validate_text_conditioning(
        self,
        *,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None = None,
        task_text: object,
        batch_size: int,
        require_negative_text: bool = False,
    ) -> None:
        """Validate the effective text context selected by the shared frontend."""

        mode = self.config.text_conditioning_mode
        if mode == TextConditioningMode.DISABLED and require_negative_text:
            raise ValueError(
                "Classifier-free guidance is unavailable when causal video text "
                "conditioning is disabled."
            )
        if (
            mode == TextConditioningMode.DISABLED
            and text_context is None
            and negative_text_context is None
        ):
            return
        if mode == TextConditioningMode.TASK_PROMPT and (
            not isinstance(task_text, tuple)
            or len(task_text) != batch_size
            or any(
                not isinstance(value, str) or not value.strip() for value in task_text
            )
        ):
            raise ValueError(
                "Task-prompt causal video prediction requires one non-empty task "
                "instruction per sample."
            )

        description = (
            "task-prompt text"
            if mode == TextConditioningMode.TASK_PROMPT
            else "blank-text"
        )
        resolved_text_context = self._validate_text_tensor(
            text_context,
            batch_size=batch_size,
            description=description,
            require_nonzero=mode == TextConditioningMode.TASK_PROMPT,
        )
        negative_required = (
            mode == TextConditioningMode.DISABLED or require_negative_text
        )
        if negative_text_context is None:
            if negative_required:
                raise ValueError(
                    f"Causal video prediction requires negative text embeddings "
                    f"for {mode.value!r} conditioning."
                )
            return
        if tuple(negative_text_context.shape) != tuple(resolved_text_context.shape):
            raise ValueError(
                "Causal video prediction requires effective and negative text "
                "embeddings with identical shapes, got "
                f"positive={tuple(resolved_text_context.shape)}, "
                f"negative={tuple(negative_text_context.shape)}."
            )
        self._validate_text_tensor(
            negative_text_context,
            batch_size=batch_size,
            description="negative text",
            require_nonzero=negative_required,
        )

    def required_visual_stages(self) -> tuple[PolicyVisualStage, ...]:
        return (PolicyVisualStage.FRONTEND,)

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        del visual_outputs
        return PolicyPreparedInputs(batch=batch)

    @staticmethod
    def _metadata_tuple(
        metadata: object,
        *,
        expected_batch_size: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(metadata, tuple):
            raise TypeError(
                "Causal video prediction expects batched metadata as a tuple of mappings."
            )
        if len(metadata) != expected_batch_size:
            raise ValueError(
                "Causal video prediction metadata cardinality must match the latent "
                f"batch size: metadata={len(metadata)}, batch={expected_batch_size}."
            )
        for index, sample_metadata in enumerate(metadata):
            if not isinstance(sample_metadata, Mapping):
                raise TypeError(
                    "Causal video prediction metadata entries must be mappings; "
                    f"entry {index} is {type(sample_metadata).__name__}."
                )
        return metadata

    def _resolve_layouts(
        self,
        *,
        metadata: tuple[Mapping[str, Any], ...],
        available_frames: int,
        frame_mapping: dict[str, Any] | None = None,
    ) -> list[_PrefixSuffixLayout]:
        layouts: list[_PrefixSuffixLayout] = []
        for sample_metadata in metadata:
            observed_frames = int(sample_metadata["observed_prefix_frames"])
            future_frames = int(sample_metadata["future_suffix_frames"])
            total_frames = int(
                sample_metadata.get(
                    "valid_video_frames", observed_frames + future_frames
                )
            )
            if observed_frames <= 0 or future_frames <= 0:
                raise ValueError(
                    "Causal video prediction requires positive observed/future frames, "
                    f"got observed_frames={observed_frames}, future_frames={future_frames}."
                )
            if total_frames != observed_frames + future_frames:
                raise ValueError(
                    "Causal video prediction expects `valid_video_frames == observed_prefix_frames + future_suffix_frames`, "
                    f"got valid_video_frames={total_frames}, observed_frames={observed_frames}, future_frames={future_frames}."
                )
            if self._uses_wan_temporal_mapping(frame_mapping):
                observed_frames, future_frames, total_frames = (
                    self._map_raw_layout_to_wan_latents(
                        raw_observed_frames=observed_frames,
                        raw_total_frames=total_frames,
                        available_frames=available_frames,
                    )
                )
            if total_frames > available_frames:
                raise ValueError(
                    "Causal video prediction metadata exceeds the available latent window, "
                    f"got total_frames={total_frames}, available_frames={available_frames}."
                )
            layouts.append(
                _PrefixSuffixLayout(
                    observed_frames=observed_frames,
                    future_frames=future_frames,
                    total_frames=total_frames,
                )
            )
        return layouts

    @staticmethod
    def _uses_wan_temporal_mapping(frame_mapping: dict[str, Any] | None) -> bool:
        if not isinstance(frame_mapping, dict):
            return False
        return frame_mapping.get("kind") == "wan_temporal_downsample"

    @staticmethod
    def _map_raw_layout_to_wan_latents(
        *,
        raw_observed_frames: int,
        raw_total_frames: int,
        available_frames: int,
    ) -> tuple[int, int, int]:
        mapping = VideoFrameMapping.wan_causal_prefix_suffix(
            raw_observed_frames=raw_observed_frames,
            raw_future_frames=int(raw_total_frames) - int(raw_observed_frames),
            available_frames=available_frames,
        )
        return mapping.observed_frames, mapping.future_frames, mapping.total_frames

    @staticmethod
    def _build_valid_token_attention_mask(
        layouts: list[_PrefixSuffixLayout],
        *,
        token_grid: TokenGridMetadata,
        device: torch.device,
    ) -> torch.Tensor | None:
        if all(layout.total_frames == token_grid.num_frames for layout in layouts):
            return None
        patch_t, _, _ = token_grid.patch_size
        if patch_t <= 0:
            raise ValueError(f"Invalid video token temporal patch size: {patch_t}.")
        unaligned = [
            layout.total_frames
            for layout in layouts
            if layout.total_frames % patch_t != 0
        ]
        if unaligned:
            raise ValueError(
                "Causal video prediction cannot mask padded frames at sub-token granularity; "
                f"valid frame counts must be divisible by patch_t={patch_t}, got {unaligned}."
            )
        sequence_length = int(token_grid.sequence_length)
        tokens_per_frame = int(token_grid.tokens_per_frame)
        if sequence_length <= 0 or tokens_per_frame <= 0:
            raise ValueError(
                "Causal video prediction received invalid token grid metadata, "
                f"sequence_length={sequence_length}, tokens_per_frame={tokens_per_frame}."
            )
        token_indices = torch.arange(sequence_length, device=device)
        temporal_patch_indices = token_indices // tokens_per_frame
        valid_patch_counts = torch.tensor(
            [layout.total_frames // patch_t for layout in layouts],
            device=device,
            dtype=temporal_patch_indices.dtype,
        )
        valid_key_tokens = temporal_patch_indices.unsqueeze(
            0
        ) < valid_patch_counts.unsqueeze(1)
        return valid_key_tokens[:, None, :].expand(-1, sequence_length, -1).contiguous()

    def _build_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        metadata: tuple[dict[str, Any], ...],
        text_context: torch.Tensor | None,
    ) -> dict[str, Any]:
        video_latents = visual_outputs.frontend.video_latents
        batch_size, _, num_frames, _, _ = video_latents.shape
        layouts = self._resolve_layouts(
            metadata=metadata,
            available_frames=num_frames,
            frame_mapping=visual_outputs.frontend.conditioning.metadata.get(
                "video_frame_mapping"
            ),
        )
        scheduler = FlowMatchScheduler(
            shift=self.training_config.video_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
        )
        scheduler.set_timesteps(
            self.training_config.video_num_train_timesteps, training=True
        )

        timestep_ids = sample_timestep_id(
            batch_size=batch_size,
            sample_shape=(num_frames,),
            num_train_timesteps=self.training_config.video_num_train_timesteps,
            device=video_latents.device,
        )
        timesteps = scheduler.timesteps.to(device=video_latents.device)[timestep_ids]
        noise = torch.randn_like(video_latents)
        noisy_latents = scheduler.add_noise(video_latents, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(video_latents, noise, timesteps)
        future_loss_mask = torch.zeros(
            batch_size,
            1,
            num_frames,
            1,
            1,
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        for batch_index, layout in enumerate(layouts):
            noisy_latents[batch_index, :, : layout.observed_frames] = video_latents[
                batch_index, :, : layout.observed_frames
            ]
            timesteps[batch_index, : layout.observed_frames] = 0.0
            if layout.total_frames < num_frames:
                noisy_latents[batch_index, :, layout.total_frames :] = 0.0
                targets[batch_index, :, layout.total_frames :] = 0.0
                timesteps[batch_index, layout.total_frames :] = 0.0
            future_loss_mask[
                batch_index, :, layout.observed_frames : layout.total_frames
            ] = 1.0

        attention_mask = self._build_valid_token_attention_mask(
            layouts,
            token_grid=visual_outputs.frontend.token_grid,
            device=video_latents.device,
        )
        flow_pred = visual_tower.predict_video_flow(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_context=text_context,
            frame_start=0,
            attention_mask=attention_mask,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_latents,
            flow_pred=flow_pred,
            timesteps=timesteps,
            scheduler=scheduler,
        )
        return {
            "flow_pred": flow_pred,
            "flow_targets": targets,
            "predicted_latents": predicted_latents,
            "target_latents": video_latents,
            "timesteps": timesteps,
            "scheduler": scheduler,
            "future_loss_mask": future_loss_mask,
            "layouts": layouts,
        }

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        conditioning = visual_outputs.frontend.conditioning
        batch_size = int(visual_outputs.frontend.video_latents.shape[0])
        text_context_to_validate = conditioning.text_context
        if (
            self.config.text_conditioning_mode == TextConditioningMode.TASK_PROMPT
            and prepared_inputs.batch.source_text_context is not None
        ):
            text_context_to_validate = prepared_inputs.batch.source_text_context
        self._validate_text_conditioning(
            text_context=text_context_to_validate,
            negative_text_context=conditioning.negative_text_context,
            task_text=prepared_inputs.batch.extra.get("task_text"),
            batch_size=batch_size,
            require_negative_text=(
                float(self.training_config.text_condition_dropout_prob) > 0.0
            ),
        )
        rollout = self._build_train_rollout(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            metadata=self._metadata_tuple(
                prepared_inputs.batch.extra.get("metadata"),
                expected_batch_size=batch_size,
            ),
            text_context=conditioning.text_context,
        )
        policy_features = visual_outputs.frontend.video_latents.new_zeros(
            batch_size, 0, self.config.hidden_size
        )
        future_frame_counts = torch.tensor(
            [layout.future_frames for layout in rollout["layouts"]],
            device=visual_outputs.frontend.video_latents.device,
            dtype=torch.float32,
        )
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={
                "future_frame_count": future_frame_counts.mean().detach(),
            },
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
                payload=VideoFlowTrainArtifacts(
                    flow_pred=rollout["flow_pred"],
                    targets=rollout["flow_targets"],
                    timesteps=rollout["timesteps"],
                    scheduler=rollout["scheduler"],
                    predicted_latents=rollout["predicted_latents"],
                    target_latents=rollout["target_latents"],
                    future_loss_mask=rollout["future_loss_mask"],
                ),
            ),
            aux={
                "variant": self.config.name,
                "architecture": "causal_video_prediction",
                "layouts": rollout["layouts"],
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del visual_tower, visual_outputs, context
        if previous_state is not None:
            return previous_state
        cursor = RolloutCursor(
            current_start_frame=0,
            block_index=0,
            chunk_size=self.inference_config.frame_chunk_size,
        )
        return PolicyInferState(step_index=0, cursor=cursor)

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        video_latents = visual_outputs.frontend.video_latents
        batch_size = int(video_latents.shape[0])
        if batch_size != 1:
            raise ValueError(
                "Causal video prediction inference currently supports batch size 1; "
                f"got {batch_size}."
            )
        self._validate_text_conditioning(
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            task_text=context.extra.get("task_text"),
            batch_size=batch_size,
            require_negative_text=self.inference_config.guidance_scale > 1.0,
        )
        metadata = self._metadata_tuple(
            context.extra.get("metadata"),
            expected_batch_size=batch_size,
        )
        layouts = self._resolve_layouts(
            metadata=metadata,
            available_frames=int(visual_outputs.frontend.video_latents.shape[2]),
            frame_mapping=visual_outputs.frontend.conditioning.metadata.get(
                "video_frame_mapping"
            ),
        )
        layout = layouts[0]
        observed_prefix = video_latents[:, :, : layout.observed_frames]
        future_template = torch.zeros_like(
            video_latents[:, :, layout.observed_frames : layout.total_frames]
        )
        predicted_future = visual_tower.generate_conditioned_future_latents(
            observed_prefix=observed_prefix,
            future_template=future_template,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            frame_start=0,
            num_inference_steps=self.inference_config.video_num_inference_steps,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
            sigma_shift=self.training_config.video_sigma_shift,
            guidance_scale=self.inference_config.guidance_scale,
        )
        predicted_latents = torch.cat([observed_prefix, predicted_future], dim=2)
        policy_features = video_latents.new_zeros(
            video_latents.shape[0], 0, self.config.hidden_size
        )
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=infer_state.cursor,
            ),
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
                payload=VideoFlowInferArtifacts(
                    predicted_latents=predicted_latents,
                ),
            ),
            aux={
                "variant": self.config.name,
                "architecture": "causal_video_prediction",
            },
        )
