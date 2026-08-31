from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs import (
    CausalVideoProgram,
    InferenceConfig,
    TextConditioningMode,
    TrainingConfig,
)
from open_wam.configs.policy_contracts import CausalVideoPredictionPolicyConfig
from open_wam.contracts import (
    SampleConstructionMetadata,
    VideoFrameMapping,
    VideoLatentSpaceIdentity,
)
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
    sample_timestep_id,
)
from open_wam.models.common.flow_supervision import (
    denoised_video_latents_from_flow,
)
from open_wam.models.common.flow_training import (
    build_video_flow_match_train_artifacts,
)
from open_wam.models.video_backbone.contracts import TokenGridMetadata
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .contracts import (
    DecoderArtifactEnvelope,
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferenceCapabilities,
    PolicyInferOutput,
    PolicyInferState,
    PolicyOutputModality,
    PolicyPipelineRequirements,
    PolicyPreparedInputs,
    PolicyRecurrentHistoryPolicy,
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


@dataclass(frozen=True)
class _ChunkedConditionedLayout:
    chunk_size: int
    window_size: int
    frame_shift: int
    chunk_origin_frame: int
    singleton_chunk_frame: int | None


class CausalVideoPredictionPolicyVariant(PolicyVariant):
    """Text-conditioned video prediction over explicit sequence programs."""

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

    @property
    def inference_capabilities(self) -> PolicyInferenceCapabilities:
        return PolicyInferenceCapabilities(
            native_modalities=frozenset({PolicyOutputModality.VIDEO}),
            recurrent_history_policy=(
                PolicyRecurrentHistoryPolicy.NEXT_OBSERVATION
            ),
        )

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

    @staticmethod
    def _resolve_chunked_conditioned_layout(
        *,
        metadata: tuple[Mapping[str, Any], ...],
        target_frames: int,
    ) -> _ChunkedConditionedLayout:
        layouts: list[_ChunkedConditionedLayout] = []
        for sample in metadata:
            typed = SampleConstructionMetadata.from_mapping(sample)
            if typed is None:  # pragma: no cover - mapping checked by caller
                raise TypeError("Chunked conditioned-video metadata is missing.")
            chunk_size = typed.sampled_chunk_size_for(target_frames)
            window_size = typed.sampled_window_size
            if chunk_size is None or window_size is None:
                raise ValueError(
                    "Chunked conditioned-video training requires sampled chunk and "
                    "window geometry in every sample's metadata."
                )
            layouts.append(
                _ChunkedConditionedLayout(
                    chunk_size=int(chunk_size),
                    window_size=int(window_size),
                    frame_shift=0 if typed.frame_shift is None else int(typed.frame_shift),
                    chunk_origin_frame=typed.chunk_origin_frame_for(
                        observed_num_frames=target_frames
                    ),
                    singleton_chunk_frame=typed.singleton_chunk_frame_for(
                        observed_num_frames=target_frames
                    ),
                )
            )
        first = layouts[0]
        if any(layout != first for layout in layouts[1:]):
            raise ValueError(
                "Chunked conditioned-video batches must share sampled chunk, "
                "window, and frame geometry. Use rank-local batch size 1 when "
                "geometry is randomized."
            )
        return first

    def _build_chunked_conditioned_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
        metadata: tuple[Mapping[str, Any], ...],
        text_context: torch.Tensor | None,
    ) -> dict[str, Any]:
        target_latents = visual_outputs.frontend.video_latents
        batch_size, _, target_frames, _, _ = target_latents.shape
        condition_latents = batch.extra.get("condition_latents")
        if not isinstance(condition_latents, torch.Tensor):
            raise ValueError(
                "Chunked conditioned-video training requires precomputed "
                "`condition_latents` with source-frame offset -1."
            )
        if (
            condition_latents.ndim != 5
            or tuple(condition_latents.shape[:2]) != tuple(target_latents.shape[:2])
            or tuple(condition_latents.shape[-2:]) != tuple(target_latents.shape[-2:])
            or int(condition_latents.shape[2]) < 1
        ):
            raise ValueError(
                "Chunked conditioned-video condition latents must match target "
                "batch, channel, and spatial dimensions and contain an external "
                "frame, "
                f"got condition={tuple(condition_latents.shape)}, "
                f"target={tuple(target_latents.shape)}."
            )
        layout = self._resolve_chunked_conditioned_layout(
            metadata=metadata,
            target_frames=target_frames,
        )
        prefix_latents = condition_latents[:, :, :1].to(
            device=target_latents.device,
            dtype=target_latents.dtype,
        )
        model_latents = torch.cat([prefix_latents, target_latents], dim=2)
        artifacts = build_video_flow_match_train_artifacts(
            model_latents,
            training_config=self.training_config,
            condition_latents=model_latents,
            noisy_condition_prob=float(self.config.noisy_video_condition_prob or 0.0),
            clean_prefix_frames=1,
        )
        loss_mask = torch.ones(
            batch_size,
            1,
            target_frames + 1,
            1,
            1,
            device=target_latents.device,
            dtype=target_latents.dtype,
        )
        loss_mask[:, :, :1] = 0
        flow_pred = visual_tower.predict_chunked_conditioned_video_flow(
            noisy_latents=artifacts.noisy_latents,
            condition_latents=artifacts.condition_latents,
            timesteps=artifacts.timesteps,
            condition_timesteps=artifacts.condition_timesteps,
            text_context=text_context,
            chunk_size=layout.chunk_size,
            window_size=layout.window_size,
            frame_start=layout.frame_shift - 1,
            chunk_origin_frame=layout.chunk_origin_frame,
            prefix_condition_frames=1,
            singleton_chunk_frame=layout.singleton_chunk_frame,
            use_activation_checkpointing=(
                self.config.use_activation_checkpointing
            ),
            stage="train",
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=artifacts.noisy_latents,
            flow_pred=flow_pred,
            timesteps=artifacts.timesteps,
            scheduler=artifacts.scheduler,
        )
        return {
            "flow_pred": flow_pred,
            "flow_targets": artifacts.targets,
            "predicted_latents": predicted_latents,
            "target_latents": model_latents,
            "timesteps": artifacts.timesteps,
            "scheduler": artifacts.scheduler,
            "future_loss_mask": loss_mask,
            "layouts": (layout,),
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
        metadata = self._metadata_tuple(
            prepared_inputs.batch.extra.get("metadata"),
            expected_batch_size=batch_size,
        )
        if self.config.program == CausalVideoProgram.PREFIX_SUFFIX:
            rollout = self._build_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                metadata=metadata,
                text_context=conditioning.text_context,
            )
            supervised_frame_count = torch.tensor(
                [layout.future_frames for layout in rollout["layouts"]],
                device=visual_outputs.frontend.video_latents.device,
                dtype=torch.float32,
            ).mean()
        else:
            rollout = self._build_chunked_conditioned_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                batch=prepared_inputs.batch,
                metadata=metadata,
                text_context=conditioning.text_context,
            )
            supervised_frame_count = torch.tensor(
                float(visual_outputs.frontend.video_latents.shape[2]),
                device=visual_outputs.frontend.video_latents.device,
            )
        policy_features = visual_outputs.frontend.video_latents.new_zeros(
            batch_size, 0, self.config.hidden_size
        )
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={
                "future_frame_count": supervised_frame_count.detach(),
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
                "program": self.config.program.value,
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

    def _build_video_infer_output(
        self,
        *,
        video_latents: torch.Tensor,
        observed_prefix: torch.Tensor,
        predicted_future: torch.Tensor,
        next_state: PolicyInferState,
        generated_frame_start: int | None = None,
        latent_space_identity: VideoLatentSpaceIdentity | None = None,
        aux_extra: Mapping[str, Any] | None = None,
    ) -> PolicyInferOutput:
        """Publish one future-video result through the shared policy contract."""

        predicted_latents = torch.cat([observed_prefix, predicted_future], dim=2)
        return PolicyInferOutput(
            policy_features=video_latents.new_zeros(
                video_latents.shape[0], 0, self.config.hidden_size
            ),
            next_state=next_state,
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
                payload=VideoFlowInferArtifacts(
                    predicted_latents=predicted_latents,
                ),
            ),
            generated_video=PolicyGeneratedVideo(
                latents=predicted_future.detach(),
                frame_start=generated_frame_start,
                latent_space_identity=latent_space_identity,
            ),
            generation_frame_start=generated_frame_start,
            aux={
                "variant": self.config.name,
                "architecture": "causal_video_prediction",
                "program": self.config.program.value,
                **dict(aux_extra or {}),
            },
        )

    def _forward_chunked_conditioned_infer_step(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        metadata: tuple[Mapping[str, Any], ...],
    ) -> PolicyInferOutput:
        video_latents = visual_outputs.frontend.video_latents
        sample_metadata = metadata[0]
        observed_frames = int(sample_metadata.get("observed_prefix_frames", 1))
        future_frames = int(
            sample_metadata.get(
                "future_suffix_frames",
                int(video_latents.shape[2]) - observed_frames,
            )
        )
        if observed_frames != 1 or future_frames <= 0:
            raise ValueError(
                "Chunked conditioned-video inference requires one external prefix "
                f"and at least one target frame, got observed={observed_frames}, "
                f"future={future_frames}."
            )
        if observed_frames + future_frames > int(video_latents.shape[2]):
            raise ValueError(
                "Chunked conditioned-video inference metadata exceeds its latent "
                f"input: observed={observed_frames}, future={future_frames}, "
                f"available={video_latents.shape[2]}."
            )
        observed_prefix = video_latents[:, :, :1]
        future_template = video_latents[:, :, 1 : 1 + future_frames]
        predicted_future = visual_tower.generate_chunked_conditioned_video_latents(
            observed_prefix=observed_prefix,
            future_template=future_template,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=(
                visual_outputs.frontend.conditioning.negative_text_context
            ),
            frame_start=int(sample_metadata.get("frame_shift", 0)),
            chunk_size=int(self.inference_config.frame_chunk_size),
            window_size=int(self.training_config.window_size),
            chunk_origin_frame=int(sample_metadata.get("chunk_origin_frame", 0)),
            num_inference_steps=int(
                self.inference_config.video_num_inference_steps
            ),
            num_train_timesteps=int(self.training_config.video_num_train_timesteps),
            sigma_shift=float(self.training_config.video_sigma_shift),
            guidance_scale=float(self.inference_config.guidance_scale),
            sample_seed=(
                None
                if context.extra.get("sample_seed") is None
                else int(context.extra["sample_seed"])
            ),
        )
        return self._build_video_infer_output(
            video_latents=video_latents,
            observed_prefix=observed_prefix,
            predicted_future=predicted_future,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=infer_state.cursor,
            ),
            latent_space_identity=visual_outputs.frontend.latent_space_identity,
        )

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
        if context.video_generation is not None:
            return self._forward_requested_video_generation(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                context=context,
                infer_state=infer_state,
            )
        metadata = self._metadata_tuple(
            context.extra.get("metadata"),
            expected_batch_size=batch_size,
        )
        if self.config.program == CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO:
            return self._forward_chunked_conditioned_infer_step(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                context=context,
                infer_state=infer_state,
                metadata=metadata,
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
        return self._build_video_infer_output(
            video_latents=video_latents,
            observed_prefix=observed_prefix,
            predicted_future=predicted_future,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=infer_state.cursor,
            ),
            latent_space_identity=visual_outputs.frontend.latent_space_identity,
        )

    def _forward_requested_video_generation(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        """Generate a future-only chunk from observed history for composition."""

        request = context.video_generation
        if request is None:
            raise RuntimeError("Requested video generation requires typed geometry.")
        video_latents = visual_outputs.frontend.video_latents
        frame_count = int(request.frame_count)
        future_template = video_latents.new_zeros(
            video_latents.shape[0],
            video_latents.shape[1],
            frame_count,
            video_latents.shape[3],
            video_latents.shape[4],
        )
        if self.config.program is CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO:
            observed_prefix = video_latents[:, :, -1:]
            condition_frame_start = int(infer_state.cursor.current_start_frame) + int(
                video_latents.shape[2]
            ) - 1
            generated_frame_start = condition_frame_start + int(
                observed_prefix.shape[2]
            )
            predicted_future = visual_tower.generate_chunked_conditioned_video_latents(
                observed_prefix=observed_prefix,
                future_template=future_template,
                text_context=visual_outputs.frontend.conditioning.text_context,
                negative_text_context=(
                    visual_outputs.frontend.conditioning.negative_text_context
                ),
                # The visual-tower contract takes the first target frame. It
                # prepends the one-frame condition internally when assigning
                # rotary positions, matching training's `frame_shift`.
                frame_start=generated_frame_start,
                chunk_size=int(self.inference_config.frame_chunk_size),
                window_size=int(self.training_config.window_size),
                chunk_origin_frame=(
                    (condition_frame_start + 1)
                    % int(self.inference_config.frame_chunk_size)
                ),
                num_inference_steps=int(
                    self.inference_config.video_num_inference_steps
                ),
                num_train_timesteps=int(
                    self.training_config.video_num_train_timesteps
                ),
                sigma_shift=float(self.training_config.video_sigma_shift),
                guidance_scale=float(self.inference_config.guidance_scale),
                sample_seed=(
                    None
                    if context.extra.get("sample_seed") is None
                    else int(context.extra["sample_seed"])
                ),
            )
        else:
            observed_prefix = video_latents
            condition_frame_start = int(infer_state.cursor.current_start_frame)
            generated_frame_start = condition_frame_start + int(
                observed_prefix.shape[2]
            )
            predicted_future = visual_tower.generate_conditioned_future_latents(
                observed_prefix=observed_prefix,
                future_template=future_template,
                text_context=visual_outputs.frontend.conditioning.text_context,
                negative_text_context=(
                    visual_outputs.frontend.conditioning.negative_text_context
                ),
                frame_start=condition_frame_start,
                num_inference_steps=self.inference_config.video_num_inference_steps,
                num_train_timesteps=self.training_config.video_num_train_timesteps,
                sigma_shift=self.training_config.video_sigma_shift,
                guidance_scale=self.inference_config.guidance_scale,
            )
        return self._build_video_infer_output(
            video_latents=video_latents,
            observed_prefix=observed_prefix,
            predicted_future=predicted_future,
            next_state=PolicyInferState(
                step_index=infer_state.step_index + 1,
                cursor=RolloutCursor(
                    # The next real observation chunk starts where this
                    # speculative chunk starts. Its length determines the next
                    # generated frame on the following call.
                    current_start_frame=generated_frame_start,
                    block_index=infer_state.cursor.block_index + 1,
                    chunk_size=frame_count,
                ),
            ),
            generated_frame_start=generated_frame_start,
            latent_space_identity=visual_outputs.frontend.latent_space_identity,
            aux_extra={
                "composition_video_generation": True,
                "generated_video_frame_start": generated_frame_start,
                "generated_video_frame_count": frame_count,
            },
        )
