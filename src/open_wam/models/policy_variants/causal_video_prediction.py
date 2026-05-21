from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs import CausalVideoPredictionPolicyConfig, InferenceConfig, TrainingConfig
from open_wam.models.common.flow_matching import (
    FlowMatchScheduler,
    denoised_video_latents_from_flow,
    sample_timestep_id,
)
from open_wam.models.video_backbone.contracts import TokenGridMetadata
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
from open_wam.utils.video_timeline import VideoFrameMapping

from .base import PolicyVariant
from .common.rollout import advance_rollout_cursor
from .contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
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

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        del visual_outputs
        return PolicyPreparedInputs(batch=batch)

    @staticmethod
    def _metadata_tuple(batch: PolicyTrainBatch) -> tuple[dict[str, Any], ...]:
        metadata = batch.extra.get("metadata", ())
        if not isinstance(metadata, tuple):
            raise ValueError("Causal video prediction expects batched metadata as a tuple of mappings.")
        return metadata

    def _resolve_layouts(
        self,
        *,
        metadata: tuple[dict[str, Any], ...],
        available_frames: int,
        frame_mapping: dict[str, Any] | None = None,
    ) -> list[_PrefixSuffixLayout]:
        layouts: list[_PrefixSuffixLayout] = []
        for sample_metadata in metadata:
            observed_frames = int(sample_metadata["observed_prefix_frames"])
            future_frames = int(sample_metadata["future_suffix_frames"])
            total_frames = int(sample_metadata.get("valid_video_frames", observed_frames + future_frames))
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
                observed_frames, future_frames, total_frames = self._map_raw_layout_to_wan_latents(
                    raw_observed_frames=observed_frames,
                    raw_total_frames=total_frames,
                    available_frames=available_frames,
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
        unaligned = [layout.total_frames for layout in layouts if layout.total_frames % patch_t != 0]
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
        valid_key_tokens = temporal_patch_indices.unsqueeze(0) < valid_patch_counts.unsqueeze(1)
        return valid_key_tokens[:, None, :].expand(-1, sequence_length, -1).contiguous()

    def _build_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        metadata: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        video_latents = visual_outputs.frontend.video_latents
        batch_size, _, num_frames, _, _ = video_latents.shape
        layouts = self._resolve_layouts(
            metadata=metadata,
            available_frames=num_frames,
            frame_mapping=visual_outputs.frontend.conditioning.metadata.get("video_frame_mapping"),
        )
        scheduler = FlowMatchScheduler(
            shift=self.training_config.video_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
        )
        scheduler.set_timesteps(self.training_config.video_num_train_timesteps, training=True)

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
            noisy_latents[batch_index, :, : layout.observed_frames] = video_latents[batch_index, :, : layout.observed_frames]
            timesteps[batch_index, : layout.observed_frames] = 0.0
            if layout.total_frames < num_frames:
                noisy_latents[batch_index, :, layout.total_frames :] = 0.0
                targets[batch_index, :, layout.total_frames :] = 0.0
                timesteps[batch_index, layout.total_frames :] = 0.0
            future_loss_mask[batch_index, :, layout.observed_frames : layout.total_frames] = 1.0

        attention_mask = self._build_valid_token_attention_mask(
            layouts,
            token_grid=visual_outputs.frontend.token_grid,
            device=video_latents.device,
        )
        flow_pred = visual_tower.predict_video_flow(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_context=visual_outputs.frontend.conditioning.text_context,
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
        rollout = self._build_train_rollout(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            metadata=self._metadata_tuple(prepared_inputs.batch),
        )
        batch_size = visual_outputs.frontend.video_latents.shape[0]
        policy_features = visual_outputs.frontend.video_latents.new_zeros(batch_size, 0, self.config.hidden_size)
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
            aux={
                "variant": self.config.name,
                "method_family": "causal_video_prediction",
                **rollout,
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
        metadata = context.extra.get("metadata")
        if not isinstance(metadata, tuple) or not metadata:
            raise ValueError(
                "Causal video prediction inference expects metadata with `observed_prefix_frames` "
                "and `future_suffix_frames`."
            )
        layouts = self._resolve_layouts(
            metadata=metadata,
            available_frames=int(visual_outputs.frontend.video_latents.shape[2]),
            frame_mapping=visual_outputs.frontend.conditioning.metadata.get("video_frame_mapping"),
        )
        if len(layouts) != 1:
            raise ValueError("Causal video prediction inference currently supports batch size 1.")
        layout = layouts[0]
        video_latents = visual_outputs.frontend.video_latents
        observed_prefix = video_latents[:, :, : layout.observed_frames]
        future_template = torch.zeros_like(video_latents[:, :, layout.observed_frames : layout.total_frames])
        predicted_future = visual_tower.generate_conditioned_future_latents(
            observed_prefix=observed_prefix,
            future_template=future_template,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            frame_start=int(infer_state.cursor.current_start_frame),
            num_inference_steps=self.inference_config.video_num_inference_steps,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
            sigma_shift=self.training_config.video_sigma_shift,
            guidance_scale=self.inference_config.guidance_scale,
        )
        predicted_latents = torch.cat([observed_prefix, predicted_future], dim=2)
        policy_features = video_latents.new_zeros(video_latents.shape[0], 0, self.config.hidden_size)
        next_cursor = advance_rollout_cursor(infer_state.cursor)
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=PolicyInferState(step_index=infer_state.step_index + 1, cursor=next_cursor),
            aux={
                "variant": self.config.name,
                "method_family": "causal_video_prediction",
                "predicted_latents": predicted_latents,
            },
        )
