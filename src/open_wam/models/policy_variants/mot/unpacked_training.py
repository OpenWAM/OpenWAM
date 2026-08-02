"""Training programs for MoT policies without a packed block stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from open_wam.configs import TrainingConfig
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.models.common.flow_training import (
    build_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.common.flow_supervision import (
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import (
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .attention import build_chunk_causal_video_mask, build_mot_attention_mask
from .cache_execution import (
    forward_action_with_video_cache,
    prefill_video_kv_cache,
)
from .conditioning import MoTConditioning
from .contracts import (
    MoTActionTrainArtifacts,
    MoTTrainArtifacts,
    MoTVideoTrainArtifacts,
)
from .dual_stream_execution import forward_joint_video_action_denoise
from .modules import MoTActionExpert
from .runtime_routing import (
    is_mot_same_step_coupling,
    resolve_mot_current_block_coupling,
)
from .sequence_layout import MoTTrainingLayout, build_action_grid_ids_for_sequence


@dataclass(frozen=True)
class MoTUnpackedTrainingProgram:
    """Execute M5 training while video and action blocks retain separate owners."""

    config: MoTPolicyConfig
    training_config: TrainingConfig
    conditioning: MoTConditioning
    training_layout: MoTTrainingLayout
    action_expert: MoTActionExpert
    initialize_action_expert: Callable[[VisualTower], None]
    should_detach_video_cache: Callable[[VisualTower], bool]

    def _maybe_initialize_action_expert(self, visual_tower: VisualTower) -> None:
        self.initialize_action_expert(visual_tower)

    def _should_detach_train_video_cache(self, visual_tower: VisualTower) -> bool:
        return self.should_detach_video_cache(visual_tower)

    def _build_video_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
        history_frames: int,
        condition_latents: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> MoTVideoTrainArtifacts:
        video_latents = visual_outputs.frontend.video_latents
        clean_condition_latents, _ = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_condition_latents,
        )
        noisy_latents = video_artifacts.noisy_latents.clone()
        timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_condition_latents if clean_condition_latents is not None else video_latents
        noisy_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        timesteps[:, :history_frames] = 0.0
        future_loss_mask = self.training_layout.build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=batch,
            default_history_frames=history_frames,
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
            scheduler=video_artifacts.scheduler,
        )
        return MoTVideoTrainArtifacts(
            flow_pred=flow_pred,
            targets=video_artifacts.targets,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
            predicted_latents=predicted_latents,
            target_latents=video_latents,
            future_loss_mask=future_loss_mask,
        )

    def run_prefill_action_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        history_frames = self.training_layout.resolve_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        # This unused draw is part of the legacy prefill training RNG contract:
        # action noise/timesteps are sampled after the video artifacts.
        _ = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        effective_action_mask = self.training_layout.build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self.training_layout.resolve_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        video_tokens_per_frame = int(prepared_inputs.variant_inputs["video_tokens_per_frame"])
        sampled_chunk_size = self.training_layout.resolve_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(
                1,
                min(int(self.training_config.chunk_size), int(video_latents.shape[2])),
            )
        sampled_window_size = self.training_layout.resolve_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self.training_layout.resolve_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self.training_layout.resolve_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        chunk_causal_video_mask = build_chunk_causal_video_mask(
            video_seq_len=video_tokens_per_frame * int(video_latents.shape[2]),
            video_tokens_per_frame=video_tokens_per_frame,
            action_chunk_size_frames=sampled_chunk_size,
            device=video_latents.device,
            attention_window_size=sampled_window_size,
            chunk_origin_frame=chunk_origin_frame,
        )

        # Method-5 video-prefill action denoise is aligned to method-1 full-seg
        # semantics: the action expert conditions on the full clean video
        # sample, but the video K/V prefill itself stays chunk-causal so future
        # chunks do not leak through the shared video backbone.
        action_condition_latents = (
            clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        )
        video_text_context = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=self.conditioning.uses_proprio_context(),
        )
        video_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
            resolved_text_context=video_text_context,
            proprio_state=proprio_state,
            query_frames_per_copy=int(action_condition_latents.shape[2]),
            tokens_per_frame=video_tokens_per_frame,
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
        ) if video_text_context is not None else None
        video_cache = prefill_video_kv_cache(
            visual_tower=visual_tower,
            observed_prefix=action_condition_latents,
            text_context=video_text_context,
            frame_start=0,
            attention_mask=chunk_causal_video_mask,
            cross_attention_mask=video_cross_attention_mask,
            detach_cache=self._should_detach_train_video_cache(visual_tower),
        )
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        train_artifacts = self.training_layout.apply_history_action_condition(
            train_artifacts=train_artifacts,
            actions=prepared_inputs.batch.actions,
            observed_num_frames=int(video_latents.shape[2]),
            history_frames=history_frames,
        )
        resolved_text = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
            )
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
        )
        action_hidden_states = forward_action_with_video_cache(
            action_expert=self.action_expert,
            action_pre=action_pre,
            video_cache=video_cache,
            attention_mask=build_mot_attention_mask(
                video_seq_len=video_cache.video_seq_len,
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=int(video_latents.shape[2]),
                clean_action_frames=history_frames,
                attention_window_size=sampled_window_size,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        video_rollout: MoTVideoTrainArtifacts | None = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = self._build_video_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                batch=prepared_inputs.batch,
                history_frames=history_frames,
                condition_latents=condition_latents,
                attention_mask=chunk_causal_video_mask,
            )
        batch_size = action_condition_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=action_condition_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": action_condition_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": action_condition_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "video_cache_seq_len": video_cache.video_seq_len,
                "runtime_mode": str(self.config.runtime_mode),
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=video_rollout,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                    video_cache_seq_len=video_cache.video_seq_len,
                ),
            },
        )

    def run_joint_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        if not is_mot_same_step_coupling(current_block_coupling):
            raise NotImplementedError(
                "M5 joint_denoise train supports same-step couplings only; "
                f"got current_block_coupling={current_block_coupling.value!r}. "
                "Use runtime_mode='non_joint_two_stream' for staged video_then_action."
            )
        history_frames = self.training_layout.resolve_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self.conditioning.train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        noisy_video_latents = video_artifacts.noisy_latents.clone()
        video_timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        noisy_video_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        video_timesteps[:, :history_frames] = 0.0
        future_loss_mask = self.training_layout.build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=prepared_inputs.batch,
            default_history_frames=history_frames,
        )
        effective_action_mask = self.training_layout.build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self.training_layout.resolve_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_chunk_size = self.training_layout.resolve_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_window_size = self.training_layout.resolve_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(1, int(self.training_config.chunk_size))
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self.training_layout.resolve_frame_shift(batch=prepared_inputs.batch)
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        resolved_text = self.conditioning.resolve_text_context(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(video_latents.shape[0]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 joint action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
            resolved_text_context=resolved_text,
            proprio_state=proprio_state,
            query_frames_per_copy=int(video_latents.shape[2]),
            tokens_per_frame=int(prepared_inputs.variant_inputs["video_tokens_per_frame"]),
            chunk_size_frames=sampled_chunk_size,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
        )
        video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
            visual_tower=visual_tower,
            noisy_video_latents=noisy_video_latents,
            video_timesteps=video_timesteps,
            action_expert=self.action_expert,
            action_pre=action_pre,
            text_context=resolved_text,
            attention_mask=build_mot_attention_mask(
                video_seq_len=prepared_inputs.variant_inputs["video_tokens_per_frame"] * video_latents.shape[2],
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
                video_can_attend_action=self.config.video_can_attend_action,
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=history_frames,
                attention_window_size=sampled_window_size,
                current_block_coupling=current_block_coupling,
            ),
            use_activation_checkpointing=self.config.use_activation_checkpointing,
            video_cross_attention_mask=video_cross_attention_mask,
            video_hidden_context=self.conditioning.video_hidden_context_for_tokens(
                visual_tower,
                hidden_proprio_state,
                video_latents=video_latents,
                chunk_size_frames=sampled_chunk_size,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_video_latents,
            flow_pred=video_flow_pred,
            timesteps=video_timesteps,
            scheduler=video_artifacts.scheduler,
        )
        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": video_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": video_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "runtime_mode": str(self.config.runtime_mode),
                "current_block_coupling": current_block_coupling.value,
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=MoTVideoTrainArtifacts(
                        flow_pred=video_flow_pred,
                        targets=video_artifacts.targets,
                        timesteps=video_timesteps,
                        scheduler=video_artifacts.scheduler,
                        predicted_latents=predicted_latents,
                        target_latents=video_latents,
                        future_loss_mask=future_loss_mask,
                    ),
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                ),
            },
        )
