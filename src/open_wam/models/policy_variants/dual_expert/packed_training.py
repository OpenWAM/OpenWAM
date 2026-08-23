"""Packed-coupling training program for DualExpert policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from open_wam.configs import (
    GeneralistDenoisingMode,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    TrainingConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.models.common.attention_contracts import (
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
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
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
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
from .generalist_modes import (
    apply_generalist_training_mode as _apply_dual_expert_generalist_training_mode,
)
from .generalist_modes import (
    generalist_forces_clean_video_condition as _dual_expert_generalist_forces_clean_video_condition,
)
from .generalist_modes import (
    resolve_generalist_training_mode,
)
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
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        video_tokens_per_frame = int(prepared_inputs.variant_inputs["video_tokens_per_frame"])
        target_video_latents = video_latents
        target_num_video_frames = int(target_video_latents.shape[2])
        num_video_frames = target_num_video_frames
        # Dataset adapters may stamp sampled geometry into per-sample metadata.
        # Full-segment samples leave it unset, so draw geometry per step using
        # the same contract as parallel-stream training.
        metadata_for_geometry = prepared_inputs.batch.extra.get("metadata")
        metadata_has_geometry = (
            isinstance(metadata_for_geometry, tuple)
            and len(metadata_for_geometry) > 0
            and metadata_for_geometry[0].get("sampled_chunk_size") is not None
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
            sampled_chunk_size, sampled_window_size, history_frames = (
                self.training_layout.sample_full_segment_geometry(
                    observed_num_frames=target_num_video_frames,
                    device=video_latents.device,
                )
            )
        video_latents, hidden_proprio_state, prefix_condition_frames, legacy_video_condition_source = (
            self.conditioning.prepend_legacy_prefix_video_latents(
                video_latents=target_video_latents,
                condition_latents=condition_latents,
                hidden_proprio_state=hidden_proprio_state,
                batch=prepared_inputs.batch,
            )
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
        frame_shift = self.training_layout.resolve_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self.training_layout.resolve_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )
        singleton_chunk_frame = self.training_layout.resolve_singleton_chunk_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
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

        (
            sampled_generalist_mode,
            forced_generalist_mode,
            metadata_drop_text,
            generalist_source,
        ) = resolve_generalist_training_mode(
            self.config,
            prepared_inputs.batch,
            device=video_latents.device,
        )
        if sampled_generalist_mode is not None and int(video_latents.shape[0]) != 1:
            raise ValueError(
                "dual-expert generalist joint denoising currently requires rank-local train_batch_size=1 because "
                "one GJD mode is sampled/applied per segment forward and per-sample forced modes are only "
                f"unambiguous for batch size 1; got batch_size={int(video_latents.shape[0])}."
            )

        joint_timestep_coupling = resolve_dual_expert_joint_timestep_coupling(
            self.config,
            current_block_coupling,
        )
        shared_timestep_ids = None
        if joint_timestep_coupling in {
            JointTimestepCoupling.MATCH_INDEX,
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
        }:
            if int(self.training_config.video_num_train_timesteps) != int(self.training_config.action_num_train_timesteps):
                if joint_timestep_coupling == JointTimestepCoupling.MATCH_INDEX:
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
            clean_video_condition_latents, video_condition_source = self.conditioning.train_clean_video_condition_latents(
                video_latents=video_latents,
                condition_latents=condition_latents,
                history_frames=history_frames,
            )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
            timestep_ids=shared_timestep_ids,
            noisy_condition_prob=0.0
            if _dual_expert_generalist_forces_clean_video_condition(sampled_generalist_mode)
            else float(self.config.noisy_video_condition_prob),
        )
        if prefix_condition_frames > 0:
            prefix_latents = video_latents[:, :, :prefix_condition_frames]
            video_artifacts.noisy_latents[:, :, :prefix_condition_frames] = prefix_latents
            video_artifacts.condition_latents[:, :, :prefix_condition_frames] = prefix_latents
            video_artifacts.targets[:, :, :prefix_condition_frames] = 0
            video_artifacts.timesteps[:, :prefix_condition_frames] = 0.0
            video_artifacts.condition_timesteps[:, :prefix_condition_frames] = 0.0
        coupled_action_sigma_values = (
            frame_sigmas_for_timesteps(video_artifacts.scheduler, video_artifacts.timesteps[:, prefix_condition_frames:])
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
        )
        if prefix_condition_frames > 0:
            future_loss_mask.zero_()
            explicit_video_loss_range = self.training_layout.resolve_loss_frame_range(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
                start_key="latent_loss_frame_start",
                end_key="latent_loss_frame_end",
            )
            if explicit_video_loss_range is None:
                future_loss_mask[:, :, prefix_condition_frames:] = 1.0
            else:
                loss_frame_start, loss_frame_end = explicit_video_loss_range
                shifted_start = int(prefix_condition_frames) + int(loss_frame_start)
                shifted_end = int(prefix_condition_frames) + int(loss_frame_end)
                future_loss_mask[:, :, shifted_start:shifted_end] = 1.0
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

        # ---- A1 generalist mode sampling (strict parallel-stream PR #95 parity) ----
        # When ``generalist_denoising_mode_probs`` is set, sample one
        # regime per segment. Sampling lives at the segment top so the same
        # mode flows through every layer / block of this forward; it must
        # NOT be re-sampled at block granularity (would break attention
        # profile cache + cause same-step layers to disagree).
        if sampled_generalist_mode is not None:
            generalist_semantics = resolve_generalist_joint_conditioning_semantics(
                sampled_generalist_mode,
                joint_mode=GeneralistDenoisingMode.JOINT,
                action_conditioned_video_mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
                video_conditioned_action_mode=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
                drop_text_conditioning=metadata_drop_text,
            )
            (
                video_artifacts,
                noisy_actions,
                clean_actions,
                noisy_slot_timesteps,
                future_loss_mask,
                effective_action_mask,
            ) = _apply_dual_expert_generalist_training_mode(
                sampled_mode=sampled_generalist_mode,
                video_artifacts=video_artifacts,
                noisy_actions=noisy_actions,
                clean_actions=clean_actions,
                noisy_slot_timesteps=noisy_slot_timesteps,
                future_loss_mask=future_loss_mask,
                effective_action_mask=effective_action_mask,
                clean_action_condition_mask=clean_action_condition_mask,
            )
            if generalist_semantics.is_conditional:
                # FDM/IDM keep the sampled GJD chunk geometry, but restrict
                # clean history to the immediately previous video chunk.
                sampled_window_size = generalist_semantics.attention_window_size(
                    fallback_window_size=sampled_window_size,
                )
                history_stream_visibility = HistoryStreamVisibility.VIDEO_ONLY
                conditional_history_policy = (
                    self.training_layout.resolve_conditional_history_policy(batch=prepared_inputs.batch)
                    or CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
                )

        packed_action_tokens = torch.cat([noisy_actions, clean_actions], dim=1)
        action_hidden_proprio_state = self.conditioning.legacy_prefix_action_hidden_proprio_state(
            hidden_proprio_state,
            prefix_condition_frames=prefix_condition_frames,
            target_num_frames=target_num_video_frames,
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
        )
        packed_action_hidden_context = self.conditioning.action_hidden_context_for_tokens(
            visual_tower,
            action_hidden_proprio_state,
            action_tokens=noisy_actions,
            action_tokens_per_frame=int(action_tokens_per_frame),
            copies=2,
            chunk_size_frames=sampled_chunk_size,
        )
        clean_slot_timesteps = torch.zeros_like(noisy_slot_timesteps)
        packed_action_timesteps = torch.cat(
            [noisy_slot_timesteps, clean_slot_timesteps], dim=1
        )

        text_dropped = False
        if sampled_generalist_mode is not None:
            text_dropped = resolve_generalist_joint_conditioning_semantics(
                sampled_generalist_mode,
                joint_mode=GeneralistDenoisingMode.JOINT,
                action_conditioned_video_mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
                video_conditioned_action_mode=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
                drop_text_conditioning=metadata_drop_text,
            ).drop_text_conditioning
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
            raise RuntimeError("dual-expert packed text context unexpectedly resolved to None.")
        generalist_mode_text_token_count = 0
        if bool(getattr(self.config, "generalist_mode_text_token", False)):
            if sampled_generalist_mode is None:
                raise ValueError(
                    "DualExpert `generalist_mode_text_token=true` requires an active sampled or forced GJD mode."
                )
            resolved_text, generalist_mode_text_token_count = self.conditioning.append_generalist_mode_text_token(
                visual_tower,
                resolved_text,
                sampled_generalist_mode,
            )
        packed_video_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
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

        single_action_grid = build_action_grid_ids_for_sequence(
            batch_size=noisy_actions.shape[0],
            seq_len=action_seq_len,
            action_tokens_per_frame=action_tokens_per_frame,
            device=noisy_actions.device,
            frame_shift=frame_shift,
        )  # [B, 4, T_a*ppF_a]
        packed_action_grid = torch.cat([single_action_grid, single_action_grid], dim=-1)
        packed_action_cross_attention_mask = self.conditioning.build_proprio_cross_attention_mask(
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
                hidden_proprio_state,
                video_latents=video_latents,
                copies=2,
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_flow_pred, packed_action_hidden = forward_dual_expert_packed_coupling_denoise(
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
            use_activation_checkpointing=bool(self.config.use_activation_checkpointing),
            packed_block_stack=self.packed_block_stack,
            video_cross_attention_mask=packed_video_cross_attention_mask,
            video_hidden_context=packed_video_hidden_context,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=video_artifacts.noisy_latents,
            flow_pred=video_flow_pred,
            timesteps=video_artifacts.timesteps,
            scheduler=video_artifacts.scheduler,
        )
        packed_action_flow = self.action_expert.post_dit(packed_action_hidden, packed_action_pre)
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
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "dual_expert_history_frames": video_latents.new_tensor(float(history_frames)),
                "dual_expert_video_prefix_frames": video_latents.new_tensor(float(history_frames)),
            },
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
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
                "generalist_training_paradigm": self.config.generalist_training_paradigm.value,
                "generalist_training_source": generalist_source,
                "video_condition_source": video_condition_source,
                "dual_expert_generalist_training_mode_override": (
                    forced_generalist_mode.value if forced_generalist_mode is not None else None
                ),
                "dual_expert_generalist_text_dropped": bool(text_dropped),
                "dual_expert_generalist_training_mode": (
                    sampled_generalist_mode.value if sampled_generalist_mode is not None else None
                ),
                "dual_expert_generalist_mode_text_token": (
                    sampled_generalist_mode.value
                    if generalist_mode_text_token_count > 0 and sampled_generalist_mode is not None
                    else None
                ),
                "dual_expert_generalist_mode_text_token_count": int(generalist_mode_text_token_count),
            },
        )
