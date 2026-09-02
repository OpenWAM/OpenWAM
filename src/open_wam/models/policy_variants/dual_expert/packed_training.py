"""Packed-coupling training program for DualExpert policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from open_wam.configs import (
    HistoryStreamVisibility,
    JointTimestepCoupling,
    TrainingConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.contracts import (
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
)
from open_wam.models.common.dynamics_objectives import (
    apply_dynamics_training_plan,
    resolve_dynamics_training_plan,
)
from open_wam.models.common.flow_noise_plan import frame_sigmas_for_timesteps
from open_wam.models.common.flow_schedule import (
    sample_timestep_id,
)
from open_wam.models.common.flow_supervision import (
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
)
from open_wam.models.common.flow_training import (
    build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_train_artifacts,
)
from open_wam.models.common.proprio_conditioning import (
    HiddenProprioContext,
    project_hidden_proprio_context_to_frames,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import DecoderArtifactEnvelope, PolicyPreparedInputs, PolicyTrainOutput
from .attention_packed import build_dual_expert_packed_coupling_attention_profile
from .conditioning import DualExpertConditioning
from .coupling_semantics import (
    resolve_dual_expert_current_block_coupling,
    resolve_dual_expert_joint_timestep_coupling,
)
from .decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertActionTrainArtifacts,
    DualExpertTrainArtifacts,
    DualExpertVideoTrainArtifacts,
)
from .dual_stream_execution import forward_dual_expert_packed_coupling_denoise
from .modules import DualExpertActionExpert
from .packed_block import DualExpertPackedBlockStack
from .sequence_layout import (
    DualExpertTrainingLayout,
    build_action_grid_ids_for_sequence,
)


@dataclass(frozen=True)
class DualExpertPackedTrainingProgram:
    """Execute packed dual-expert training without owning model parameters."""

    config: DualExpertPolicyConfig
    training_config: TrainingConfig
    conditioning: DualExpertConditioning
    training_layout: DualExpertTrainingLayout
    action_expert: DualExpertActionExpert
    packed_block_stack: DualExpertPackedBlockStack | None
    initialize_action_expert: Callable[[VisualTower], None]

    def _maybe_initialize_action_expert(self, visual_tower: VisualTower) -> None:
        self.initialize_action_expert(visual_tower)

    def _resolve_history_stream_visibility(self) -> HistoryStreamVisibility:
        return HistoryStreamVisibility(self.config.history_stream_visibility)

    def run(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        # parallel-stream-style four-branch packed training for dual-expert's two-expert
        # architecture. Query/key layout is [V_noisy, V_clean, A_noisy,
        # A_clean]; the coupling mask determines current-chunk visibility for
        # all six modes while both experts remain separate transformer stacks.
        self._maybe_initialize_action_expert(visual_tower)
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_context = prepared_inputs.variant_inputs.get(
            "hidden_proprio_context"
        )
        if hidden_proprio_context is not None and not isinstance(
            hidden_proprio_context,
            HiddenProprioContext,
        ):
            raise TypeError(
                "Dual Expert hidden proprio input must use HiddenProprioContext, "
                f"got {type(hidden_proprio_context).__name__}."
            )
        video_tokens_per_frame = int(
            prepared_inputs.variant_inputs["video_tokens_per_frame"]
        )
        target_video_latents = video_latents
        target_num_video_frames = int(target_video_latents.shape[2])
        num_video_frames = target_num_video_frames
        sample_metadata = prepared_inputs.variant_inputs.get("sample_metadata")
        dynamics_sample_plan = prepared_inputs.variant_inputs.get(
            "dynamics_sample_plan"
        )
        dynamics_sequence = (
            None if dynamics_sample_plan is None else dynamics_sample_plan.sequence
        )
        # Dataset adapters may stamp sampled geometry into per-sample metadata.
        # Full-segment samples leave it unset, so draw geometry per step through
        # the shared sample-construction contract.
        metadata_has_geometry = (
            sample_metadata is not None
            and sample_metadata.sampled_chunk_size is not None
        )
        if metadata_has_geometry:
            history_frames = self.training_layout.resolve_history_frames(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
            sampled_chunk_size = self.training_layout.resolve_sampled_chunk_size(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
            sampled_window_size = self.training_layout.resolve_sampled_window_size(
                batch=prepared_inputs.batch,
            )
        else:
            sampled_chunk_size, sampled_window_size, sampled_history_frames = (
                self.training_layout.sample_full_segment_geometry(
                    observed_num_frames=target_num_video_frames,
                    device=video_latents.device,
                )
            )
            history_frames = (
                sampled_history_frames
                if dynamics_sequence is None
                else dynamics_sequence.history_frames
            )
        (
            video_latents,
            hidden_proprio_context,
            prefix_condition_frames,
            legacy_video_condition_source,
        ) = self.conditioning.prepare_train_video_sequence(
            video_latents=target_video_latents,
            condition_latents=condition_latents,
            hidden_proprio_context=hidden_proprio_context,
            batch=prepared_inputs.batch,
            dynamics_sample_plan=dynamics_sample_plan,
        )
        num_video_frames = int(video_latents.shape[2])
        current_block_coupling = resolve_dual_expert_current_block_coupling(self.config)
        effective_action_mask = self.training_layout.build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )
        clean_action_condition_mask = prepared_inputs.batch.action_mask
        action_tokens_per_frame = self.training_layout.resolve_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )
        frame_shift = self.training_layout.resolve_frame_shift(
            batch=prepared_inputs.batch
        )
        chunk_origin_frame = (
            dynamics_sequence.chunk_origin_frame
            if dynamics_sequence is not None
            else self.training_layout.resolve_chunk_origin_frame(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
        )
        singleton_chunk_frame = (
            dynamics_sequence.singleton_chunk_frame
            if dynamics_sequence is not None
            else self.training_layout.resolve_singleton_chunk_frame(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
        )

        if action_tokens_per_frame is None:
            raise ValueError(
                "Dual Expert packed training requires "
                "`action_tokens_per_frame` resolvable from the batch, got None."
            )
        if sampled_chunk_size is None:
            raise ValueError(
                "Dual Expert packed training requires "
                "`sampled_chunk_size` resolvable from the batch metadata or full-segment fallback, got None."
            )
        history_stream_visibility = self._resolve_history_stream_visibility()
        conditional_history_policy = None

        dynamics_plan = resolve_dynamics_training_plan(
            program=self.config.program,
            sample_metadata=sample_metadata,
            device=video_latents.device,
            sample_plan=dynamics_sample_plan,
        )
        dynamics_objective = None if dynamics_plan is None else dynamics_plan.objective
        routed_dynamics_objective = (
            None if dynamics_plan is None else dynamics_plan.routed_objective
        )
        dynamics_source = None if dynamics_plan is None else dynamics_plan.source
        if dynamics_objective is not None and int(video_latents.shape[0]) != 1:
            raise ValueError(
                "Dual Expert routed dynamics training requires rank-local train_batch_size=1 because "
                "one objective is applied per segment forward and per-sample routed objectives are only "
                f"unambiguous for batch size 1; got batch_size={int(video_latents.shape[0])}."
            )
        joint_timestep_coupling = resolve_dual_expert_joint_timestep_coupling(
            self.config
        )
        shared_timestep_ids = None
        if joint_timestep_coupling in {
            JointTimestepCoupling.MATCH_INDEX,
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
        }:
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_INDEX and int(
                self.training_config.video_num_train_timesteps
            ) != int(self.training_config.action_num_train_timesteps):
                raise ValueError(
                    "dual-expert index-matched joint denoising requires equal video/action train timestep counts, "
                    f"got video={self.training_config.video_num_train_timesteps}, "
                    f"action={self.training_config.action_num_train_timesteps}."
                )
            shared_timestep_ids = sample_timestep_id(
                batch_size=int(video_latents.shape[0]),
                sample_shape=(num_video_frames,),
                num_train_timesteps=int(self.training_config.video_num_train_timesteps),
                device=video_latents.device,
            )
        if prefix_condition_frames > 0:
            clean_video_condition_latents = video_latents
            video_condition_source = legacy_video_condition_source
        else:
            clean_video_condition_latents, video_condition_source = (
                self.conditioning.train_clean_video_condition_latents(
                    video_latents=video_latents,
                    condition_latents=condition_latents,
                    history_frames=history_frames,
                    dynamics_sample_plan=dynamics_sample_plan,
                )
            )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
            timestep_ids=shared_timestep_ids,
            noisy_condition_prob=0.0
            if dynamics_plan is not None
            and dynamics_plan.semantics.force_clean_video_condition
            else float(self.config.noisy_video_condition_prob),
            clean_prefix_frames=prefix_condition_frames,
        )
        coupled_action_sigma_values = (
            frame_sigmas_for_timesteps(
                video_artifacts.scheduler,
                video_artifacts.timesteps[:, prefix_condition_frames:],
            )
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA
            else None
        )
        action_scheduler_override = (
            video_artifacts.scheduler
            if joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE
            else None
        )
        future_loss_mask = self.training_layout.build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=prepared_inputs.batch,
            default_history_frames=history_frames,
            prefix_condition_frames=prefix_condition_frames,
            target_num_video_frames=target_num_video_frames,
        )
        action_artifacts = build_frame_aligned_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
            num_frames=target_num_video_frames,
            action_per_frame=int(action_tokens_per_frame),
            frame_sigma_values=coupled_action_sigma_values,
            frame_timestep_ids=(
                shared_timestep_ids[:, prefix_condition_frames:]
                if shared_timestep_ids is not None and prefix_condition_frames > 0
                else shared_timestep_ids
            ),
            scheduler_override=action_scheduler_override,
        )
        noisy_actions = action_artifacts.noisy_actions
        clean_actions = action_artifacts.condition_actions.to(
            device=noisy_actions.device, dtype=noisy_actions.dtype
        )
        if clean_actions.shape != noisy_actions.shape:
            raise ValueError(
                "Packed action training requires noisy/clean actions to share shape, "
                f"got noisy={tuple(noisy_actions.shape)}, clean={tuple(clean_actions.shape)}."
            )
        action_seq_len = int(noisy_actions.shape[1])
        num_action_frames = action_seq_len // int(action_tokens_per_frame)

        # Per-token timesteps broadcast from the per-frame sample (matches
        # parallel-stream's `_time_embed` repeat-interleave of per-frame timesteps).
        noisy_slot_timesteps = action_artifacts.slot_timesteps

        # Apply the one mode selected by the data route (or pure-joint GJD).
        # Resolution lives at the segment top so the same
        # mode flows through every layer / block of this forward; it must
        # NOT be re-sampled at block granularity (would break attention
        # profile cache + cause same-step layers to disagree).
        if dynamics_plan is not None:
            training_tensors = apply_dynamics_training_plan(
                dynamics_plan,
                clean_video=video_latents,
                noisy_video=video_artifacts.noisy_latents,
                video_targets=video_artifacts.targets,
                video_timesteps=video_artifacts.timesteps,
                video_loss_mask=future_loss_mask,
                clean_action=clean_actions,
                noisy_action=noisy_actions,
                action_targets=action_artifacts.targets,
                action_timesteps=noisy_slot_timesteps,
                action_loss_mask=effective_action_mask,
                clean_action_mask=clean_action_condition_mask,
            )
            video_artifacts = replace(
                video_artifacts,
                noisy_latents=training_tensors.noisy_video,
                targets=training_tensors.video_targets,
                timesteps=training_tensors.video_timesteps,
            )
            action_artifacts = replace(
                action_artifacts,
                targets=training_tensors.action_targets,
            )
            noisy_actions = training_tensors.noisy_action
            noisy_slot_timesteps = training_tensors.action_timesteps
            future_loss_mask = training_tensors.video_loss_mask
            effective_action_mask = training_tensors.action_loss_mask
            if dynamics_plan.semantics.is_conditional:
                # FDM/IDM keep sampled future chunk geometry while exposing
                # only the immediately preceding boundary frame as clean history.
                sampled_window_size = dynamics_plan.semantics.attention_window_size(
                    fallback_window_size=sampled_window_size,
                )
                history_stream_visibility = (
                    dynamics_plan.semantics.resolve_history_stream_visibility(
                        fallback=history_stream_visibility,
                    )
                )
                if dynamics_plan.sequence is None:  # pragma: no cover - plan invariant
                    raise RuntimeError(
                        "Conditional dynamics training plan is missing its sequence contract."
                    )
                conditional_history_policy = dynamics_plan.sequence.history_policy

        packed_action_tokens = torch.cat([noisy_actions, clean_actions], dim=1)
        projected_hidden_proprio_state = (
            None
            if hidden_proprio_context is None
            else project_hidden_proprio_context_to_frames(
                hidden_proprio_context,
                num_frames=num_video_frames,
                chunk_size=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
                prefix_frames=prefix_condition_frames,
            )
        )
        action_hidden_proprio_state = (
            None
            if projected_hidden_proprio_state is None
            else projected_hidden_proprio_state[:, prefix_condition_frames:, :]
        )
        packed_action_hidden_context = (
            self.conditioning.action_hidden_context_for_tokens(
                visual_tower,
                action_hidden_proprio_state,
                action_tokens=noisy_actions,
                action_tokens_per_frame=int(action_tokens_per_frame),
                copies=2,
                chunk_size_frames=sampled_chunk_size,
            )
        )
        clean_slot_timesteps = torch.zeros_like(noisy_slot_timesteps)
        packed_action_timesteps = torch.cat(
            [noisy_slot_timesteps, clean_slot_timesteps], dim=1
        )

        text_dropped = False
        if dynamics_plan is not None:
            text_dropped = dynamics_plan.semantics.drop_text_conditioning
        resolved_text = text_context
        if resolved_text is None:
            resolved_text = video_latents.new_zeros(
                video_latents.shape[0],
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
            )
        elif text_dropped:
            resolved_text = torch.zeros_like(resolved_text)
        resolved_text = self.conditioning.resolve_text_context(
            visual_tower,
            resolved_text,
            proprio_state,
            batch_size=int(video_latents.shape[0]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError(
                "dual-expert packed text context unexpectedly resolved to None."
            )
        generalist_mode_text_token_count = 0
        if bool(self.config.generalist_mode_text_token):
            if dynamics_objective is None:
                raise ValueError(
                    "DualExpert `generalist_mode_text_token=true` requires an active sampled or forced GJD mode."
                )
            resolved_text, generalist_mode_text_token_count = (
                self.conditioning.append_generalist_mode_text_token(
                    visual_tower,
                    resolved_text,
                    dynamics_objective,
                )
            )
        packed_video_cross_attention_mask = (
            self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=num_video_frames,
                tokens_per_frame=video_tokens_per_frame,
                chunk_size_frames=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
                singleton_chunk_frame=singleton_chunk_frame,
                repeat_copies=2,
                global_suffix_token_count=generalist_mode_text_token_count,
            )
        )

        single_action_grid = build_action_grid_ids_for_sequence(
            batch_size=noisy_actions.shape[0],
            seq_len=action_seq_len,
            action_tokens_per_frame=action_tokens_per_frame,
            device=noisy_actions.device,
            frame_shift=frame_shift,
        )  # [B, 4, T_a*ppF_a]
        packed_action_grid = torch.cat([single_action_grid, single_action_grid], dim=-1)
        packed_action_cross_attention_mask = (
            self.conditioning.build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=num_action_frames,
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
                singleton_chunk_frame=singleton_chunk_frame,
                repeat_copies=2,
                global_suffix_token_count=generalist_mode_text_token_count,
            )
        )

        packed_action_pre = self.action_expert.pre_dit(
            action_tokens=packed_action_tokens,
            timestep=packed_action_timesteps,
            context=resolved_text,
            cross_attention_mask=packed_action_cross_attention_mask,
            action_grid_ids=packed_action_grid,
            hidden_context=packed_action_hidden_context,
        )
        packed_attention_profile = build_dual_expert_packed_coupling_attention_profile(
            num_video_frames=num_video_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            num_action_frames=num_action_frames,
            action_tokens_per_frame=int(action_tokens_per_frame),
            chunk_size_frames=sampled_chunk_size,
            device=noisy_actions.device,
            attention_window_size=sampled_window_size,
            current_block_coupling=current_block_coupling,
            chunk_origin_frame=chunk_origin_frame,
            singleton_chunk_frame=singleton_chunk_frame,
            action_context_mask=clean_action_condition_mask,
            history_stream_visibility=history_stream_visibility.value,
            prefix_condition_frames=prefix_condition_frames,
            conditional_history_policy=conditional_history_policy,
        )
        packed_video_hidden_context = (
            None
            if prefix_condition_frames > 0
            else self.conditioning.video_hidden_context_for_tokens(
                visual_tower,
                projected_hidden_proprio_state,
                video_latents=video_latents,
                copies=2,
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_flow_pred, packed_action_hidden = (
            forward_dual_expert_packed_coupling_denoise(
                visual_tower=visual_tower,
                noisy_video_latents=video_artifacts.noisy_latents,
                clean_video_latents=video_artifacts.condition_latents,
                noisy_video_timesteps=video_artifacts.timesteps,
                clean_video_timesteps=video_artifacts.condition_timesteps,
                action_expert=self.action_expert,
                packed_action_pre=packed_action_pre,
                attention_profile=packed_attention_profile,
                text_context=resolved_text,
                frame_start=frame_shift - prefix_condition_frames,
                use_activation_checkpointing=bool(
                    self.config.use_activation_checkpointing
                ),
                packed_block_stack=self.packed_block_stack,
                video_cross_attention_mask=packed_video_cross_attention_mask,
                video_hidden_context=packed_video_hidden_context,
            )
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=video_artifacts.noisy_latents,
            flow_pred=video_flow_pred,
            timesteps=video_artifacts.timesteps,
            scheduler=video_artifacts.scheduler,
        )
        packed_action_flow = self.action_expert.post_dit(
            packed_action_hidden, packed_action_pre
        )
        # Loss from the A_noisy half only (first action_seq_len tokens).
        action_flow_pred = packed_action_flow[:, :action_seq_len]
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=noisy_actions,
            flow_pred=action_flow_pred,
            timesteps=noisy_slot_timesteps,
            scheduler=action_artifacts.scheduler,
        )

        # ---- Assemble training artifacts ----
        video_rollout: DualExpertVideoTrainArtifacts | None = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = DualExpertVideoTrainArtifacts(
                flow_pred=video_flow_pred,
                targets=video_artifacts.targets,
                timesteps=video_artifacts.timesteps,
                scheduler=video_artifacts.scheduler,
                predicted_latents=predicted_latents,
                target_latents=video_latents,
                future_loss_mask=future_loss_mask,
            )

        decoder_payload = DualExpertTrainArtifacts(
            action=DualExpertActionTrainArtifacts(
                flow_pred=action_flow_pred,
                targets=action_artifacts.targets,
                timesteps=noisy_slot_timesteps,
                scheduler=action_artifacts.scheduler,
                denoised_actions=denoised_actions,
                # Use the post-generalist mask. The flow builder retains the
                # pre-rewrite mask, so this is the decoder-authoritative mask.
                action_mask=effective_action_mask,
            ),
            video=video_rollout,
            condition_mode=str(self.config.condition_mode),
            program=self.config.program.value,
            history_frames=int(history_frames),
        )

        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(
                batch_size, 0, self.action_expert.hidden_size
            ),
            metrics={
                "dual_expert_history_frames": video_latents.new_tensor(
                    float(history_frames)
                ),
                "dual_expert_video_prefix_frames": video_latents.new_tensor(
                    float(history_frames)
                ),
            },
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
                dynamics_objective=dynamics_objective,
            ),
            aux={
                "variant": self.config.name,
                "architecture": "dual_expert",
                "condition_mode": str(self.config.condition_mode),
                "program": self.config.program.value,
                "current_block_coupling": current_block_coupling.value,
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "chunk_origin_frame": chunk_origin_frame,
                "singleton_chunk_frame": singleton_chunk_frame,
                "conditional_history_policy": conditional_history_policy,
                DYNAMICS_ROUTING_SOURCE_METADATA_KEY: dynamics_source,
                "video_condition_source": video_condition_source,
                "dual_expert_generalist_training_mode_override": (
                    routed_dynamics_objective.value
                    if routed_dynamics_objective is not None
                    else None
                ),
                "dual_expert_generalist_text_dropped": bool(text_dropped),
                "dual_expert_generalist_training_mode": (
                    dynamics_objective.value if dynamics_objective is not None else None
                ),
                "dual_expert_generalist_mode_text_token": (
                    dynamics_objective.value
                    if generalist_mode_text_token_count > 0
                    and dynamics_objective is not None
                    else None
                ),
                "dual_expert_generalist_mode_text_token_count": int(
                    generalist_mode_text_token_count
                ),
            },
        )
