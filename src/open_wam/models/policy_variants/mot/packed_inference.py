"""Packed-coupling recurrent inference program for MoT policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    MoTPolicyConfig,
    ParallelHistoryStreamVisibility,
    TrainingConfig,
)
from open_wam.models.common.attention_profiles import (
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
)
from open_wam.models.common.flow_matching import (
    build_action_flow_match_inference_scheduler,
    build_video_flow_match_inference_scheduler,
    timesteps_matching_sigmas,
)
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
)
from open_wam.models.common.rollout_startup import (
    build_strict_action_context_mask,
    resolve_strict_startup_plan,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import PolicyInferContext, PolicyInferOutput, PolicyInferState
from .attention import build_mot_packed_coupling_attention_profile
from .conditioning import MoTConditioning
from .contracts import MoTInferArtifacts, MoTRuntimeState
from .generalist_modes import (
    generalist_rollout_enabled as _mot_generalist_rollout_enabled,
    is_generalist_conditional_rollout as _is_mot_generalist_conditional_rollout,
    resolve_generalist_rollout_mode as _resolve_mot_generalist_rollout_mode,
)
from .inference_layout import MoTPackedHistory, MoTPackedInferenceLayout
from .modules import MoTActionExpert
from .packed_block import MoTPackedBlockStack
from .runtime import (
    expand_mot_scalar_timestep,
    forward_mot_packed_coupling_denoise,
    mot_scheduler_next_sigma,
    step_mot_flow_with_sigmas,
)
from .runtime_routing import (
    resolve_mot_action_only_rollout,
    resolve_mot_current_block_coupling,
    resolve_mot_inference_window_size,
    resolve_mot_joint_timestep_coupling,
    resolve_mot_rollout_cache_window_frames,
    resolve_mot_rollout_frame_chunk_size,
    should_couple_mot_action_to_video_sigmas,
)
from .sequence_layout import build_action_grid_ids_for_sequence


@dataclass(frozen=True)
class MoTPackedInferenceProgram:
    """Execute native packed M5 rollout without owning model parameters."""

    config: MoTPolicyConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    conditioning: MoTConditioning
    action_expert: MoTActionExpert
    packed_block_stack: MoTPackedBlockStack | None
    action_dim: int
    action_horizon: int

    def _resolve_history_stream_visibility(self) -> ParallelHistoryStreamVisibility:
        return ParallelHistoryStreamVisibility(self.config.history_stream_visibility)

    def run(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: MoTRuntimeState,
    ) -> PolicyInferOutput:
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        action_only_rollout = resolve_mot_action_only_rollout(
            context,
            current_block_coupling=current_block_coupling,
        )
        if (
            action_only_rollout
            and current_block_coupling != CurrentBlockCoupling.ACTION_THEN_VIDEO
        ):
            raise ValueError(
                "M5 packed action-only rollout is only used for action_then_video; "
                "decoupled_same_step action-only rollout uses the legacy split-cache route."
            )
        generalist_rollout_mode = (
            _resolve_mot_generalist_rollout_mode(context)
            if _mot_generalist_rollout_enabled(self.config)
            else MoTGeneralistTrainingMode.JOINT
        )
        if (
            _is_mot_generalist_conditional_rollout(generalist_rollout_mode)
            and current_block_coupling != CurrentBlockCoupling.JOINT
        ):
            raise ValueError(
                "M5 GJD conditional FDM/IDM rollout requires packed joint coupling, "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        generalist_rollout_semantics = resolve_generalist_joint_conditioning_semantics(
            generalist_rollout_mode,
            joint_mode=MoTGeneralistTrainingMode.JOINT,
            action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
            video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
        )
        inference_window_size = resolve_mot_inference_window_size(
            context,
            default_window_size=int(self.training_config.window_size),
        )
        if generalist_rollout_semantics.is_conditional:
            inference_window_size = generalist_rollout_semantics.attention_window_size(
                fallback_window_size=inference_window_size,
            )
        device = next(visual_tower.core.parameters()).device
        action_device = next(self.action_expert.parameters()).device
        if action_device != device:
            raise ValueError(
                "M5 packed coupling inference currently requires visual tower and action expert on the same device, "
                f"got visual_device={device}, action_device={action_device}."
            )
        dtype = next(self.action_expert.parameters()).dtype
        batch_size = int(visual_outputs.frontend.video_latents.shape[0])
        frame_chunk_size, action_horizon, action_tokens_per_frame = resolve_mot_rollout_frame_chunk_size(
            context,
            default_frame_chunk_size=int(self.inference_config.frame_chunk_size),
            base_action_horizon=int(self.action_horizon),
        )
        video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        video_tokens_per_frame = int(visual_outputs.frontend.token_grid.tokens_per_frame)
        current_start_frame = int(infer_state.cursor.current_start_frame)
        startup_plan = resolve_strict_startup_plan(
            step_index=int(infer_state.step_index),
            current_start_frame=current_start_frame,
            frame_chunk_size=frame_chunk_size,
            action_tokens_per_frame=action_tokens_per_frame,
            action_horizon=action_horizon,
        )
        first_step_bootstrap = startup_plan.is_startup
        current_video_prefix_frames = startup_plan.video_prefix_frames
        generation_frame_start = startup_plan.generation_frame_start
        current_action_prefix_tokens = startup_plan.action_prefix_tokens
        current_action_sequence_tokens = startup_plan.current_action_sequence_tokens
        packed_inference_layout = MoTPackedInferenceLayout(
            batch_size=batch_size,
            action_dim=self.action_dim,
            configured_action_horizon=self.action_horizon,
            frame_chunk_size=frame_chunk_size,
            action_tokens_per_frame=action_tokens_per_frame,
            current_video_prefix_frames=current_video_prefix_frames,
            current_video_sequence_frames=current_video_prefix_frames + frame_chunk_size,
            current_action_prefix_tokens=current_action_prefix_tokens,
            current_action_sequence_tokens=current_action_sequence_tokens,
            video_channels=int(video_latents.shape[1]),
            video_height=latent_height,
            video_width=latent_width,
            device=device,
            dtype=dtype,
        )
        packed_history = MoTPackedHistory.from_runtime_state(
            runtime_state,
            layout=packed_inference_layout,
            current_video_latents=video_latents,
        )
        past_clean_latents = packed_history.video_latents
        past_clean_actions = packed_history.action_latents

        if first_step_bootstrap:
            observed_prefix = video_latents[:, :, -1:].contiguous()
            current_generated_video = torch.randn(
                batch_size,
                video_latents.shape[1],
                frame_chunk_size,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
            current_noisy_video = torch.cat([observed_prefix.to(dtype=dtype), current_generated_video], dim=2)
            current_clean_video = torch.zeros_like(current_noisy_video)
            current_clean_video[:, :, :1] = observed_prefix.to(dtype=dtype)
        else:
            current_video_observation = video_latents
            current_video_frames = int(current_video_observation.shape[2])
            if current_video_frames >= frame_chunk_size:
                current_video_condition = current_video_observation[:, :, -frame_chunk_size:].contiguous()
            else:
                pad_frames = frame_chunk_size - current_video_frames
                current_video_condition = torch.cat(
                    [
                        current_video_observation,
                        current_video_observation[:, :, -1:].expand(-1, -1, pad_frames, -1, -1),
                    ],
                    dim=2,
                ).contiguous()
            current_noisy_video = torch.randn_like(current_video_condition, device=device, dtype=dtype)
            current_clean_video = torch.zeros_like(current_noisy_video)
        current_video_sequence_frames = int(current_noisy_video.shape[2])
        current_action_sample = torch.randn(batch_size, action_horizon, self.action_dim, device=device, dtype=dtype)
        diagnostic_zero_video_noise = bool(context.extra.get("mot_diagnostic_zero_current_video_noise", False))
        diagnostic_zero_action_noise = bool(context.extra.get("mot_diagnostic_zero_current_action_noise", False))
        if diagnostic_zero_video_noise:
            current_noisy_video = current_clean_video.clone()
        if diagnostic_zero_action_noise:
            current_action_sample = torch.zeros_like(current_action_sample)
        conditional_inputs = packed_inference_layout.resolve_conditional_rollout_inputs(
            context.extra,
            generalist_rollout_mode=generalist_rollout_mode,
            current_clean_video=current_clean_video,
        )
        forced_action_latents = conditional_inputs.forced_action_latents
        commit_action_latents = conditional_inputs.commit_action_latents
        video_condition_latents = conditional_inputs.video_condition_latents
        if generalist_rollout_mode == MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO:
            if forced_action_latents is None:  # pragma: no cover - guarded by required=True
                raise RuntimeError("M5 FDM rollout missing forced action latents.")
            current_action_sample = forced_action_latents.contiguous()
            if commit_action_latents is None:
                commit_action_latents = forced_action_latents
        elif generalist_rollout_mode == MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION:
            if video_condition_latents is None:  # pragma: no cover - guarded by required=True
                raise RuntimeError("M5 IDM rollout missing video condition latents.")
            current_noisy_video = video_condition_latents.contiguous()
            current_clean_video = video_condition_latents.contiguous()
            current_video_sequence_frames = int(current_noisy_video.shape[2])
        if commit_action_latents is not None and int(commit_action_latents.shape[1]) != self.action_horizon:
            raise ValueError(
                "M5 GJD commit action latents must contain the generated action horizon only after coercion, "
                f"got shape={tuple(commit_action_latents.shape)}, action_horizon={self.action_horizon}."
            )

        history_window_frames = resolve_mot_rollout_cache_window_frames(
            window_size=inference_window_size,
            frame_chunk_size=frame_chunk_size,
        )
        hidden_proprio_state = runtime_state.hidden_proprio_state
        history = packed_history.select_window(
            layout=packed_inference_layout,
            history_window_frames=history_window_frames,
            hidden_proprio_state=hidden_proprio_state,
            require_hidden_proprio_history=self.conditioning.uses_per_chunk_proprio_context(),
        )
        history_actions = history.action_latents
        history_action_tokens = history.action_tokens
        shared_history_frames = history.frames
        max_history_frames = history.max_frames
        video_hidden_proprio_sequence = history.hidden_proprio_sequence
        noisy_video_sequence = history.prepend_video(current_noisy_video)
        clean_video_sequence = history.prepend_video(current_clean_video)
        history_video_timesteps = torch.zeros(batch_size, shared_history_frames, device=device, dtype=torch.float32)
        zero_current_action_condition = current_action_sample.new_zeros(
            batch_size,
            current_action_sequence_tokens,
            self.action_dim,
        )

        text_context = runtime_state.text_context
        if text_context is None:
            text_context = visual_outputs.frontend.conditioning.text_context
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        else:
            text_context = text_context.to(device=device, dtype=dtype)
        if (
            bool(getattr(self.config, "generalist_mode_text_token", False))
            and int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) <= 0
        ):
            text_context, token_count = self.conditioning.append_generalist_mode_text_token(
                visual_tower,
                text_context,
                generalist_rollout_mode,
            )
            runtime_state.text_context = text_context
            runtime_state.generalist_mode_text_token_count = int(token_count)

        video_scheduler = build_video_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        action_scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
            raise ValueError(
                "M5 packed coupling inference expects matched video/action denoise step counts, "
                f"got video_steps={len(video_scheduler.timesteps)}, action_steps={len(action_scheduler.timesteps)}."
            )
        couple_action_video_sigmas = should_couple_mot_action_to_video_sigmas(
            self.config,
            current_block_coupling,
        )
        joint_timestep_coupling = resolve_mot_joint_timestep_coupling(
            self.config,
            current_block_coupling,
        )
        action_timestep_lookup_scheduler = None
        if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
            action_timestep_lookup_scheduler = build_action_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
                num_inference_steps_override=self.training_config.action_num_train_timesteps,
            )
        sequence_frame_start = current_start_frame - shared_history_frames
        packed_chunk_origin_frame = startup_plan.chunk_origin_frame(shared_history_frames)
        packed_action_context_mask = build_strict_action_context_mask(
            batch_size=batch_size,
            history_action_tokens=history_action_tokens,
            current_action_sequence_tokens=current_action_sequence_tokens,
            invalid_current_prefix_tokens=current_action_prefix_tokens,
            device=device,
            dtype=torch.float32,
        )
        attention_profile = build_mot_packed_coupling_attention_profile(
            num_video_frames=shared_history_frames + current_video_sequence_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            num_action_frames=shared_history_frames + current_video_prefix_frames + frame_chunk_size,
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=frame_chunk_size,
            device=device,
            attention_window_size=inference_window_size,
            current_block_coupling=current_block_coupling,
            chunk_origin_frame=packed_chunk_origin_frame,
            action_context_mask=packed_action_context_mask,
            build_dense_masks=True,
            build_flex_masks=False,
            history_stream_visibility=(
                ParallelHistoryStreamVisibility.VIDEO_ONLY.value
                if generalist_rollout_semantics.is_conditional
                else self._resolve_history_stream_visibility().value
            ),
            conditional_history_policy=(
                CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
                if generalist_rollout_semantics.is_conditional
                else None
            ),
        )
        action_grid_ids = build_action_grid_ids_for_sequence(
            batch_size=batch_size,
            seq_len=current_action_sequence_tokens,
            action_tokens_per_frame=action_tokens_per_frame,
            device=device,
            frame_shift=current_start_frame,
        )
        if shared_history_frames > 0:
            history_action_grid_ids = build_action_grid_ids_for_sequence(
                batch_size=batch_size,
                seq_len=history_action_tokens,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
                frame_shift=int(sequence_frame_start),
            )
            action_sequence_grid_ids = torch.cat([history_action_grid_ids, action_grid_ids], dim=2)
        else:
            action_sequence_grid_ids = action_grid_ids
        packed_action_grid_ids = torch.cat([action_sequence_grid_ids, action_sequence_grid_ids], dim=2)

        predicted_video_sequence = noisy_video_sequence
        action_sample = current_action_sample
        apply_video_hidden_proprio = not self.conditioning.uses_legacy_prefix_contract()
        zero_current_video_timestep = torch.zeros(
            batch_size,
            current_video_sequence_frames,
            device=device,
            dtype=torch.float32,
        )
        zero_current_action_timestep = torch.zeros(
            batch_size,
            current_action_sequence_tokens,
            device=device,
            dtype=torch.float32,
        )
        forced_clean_action_condition = (
            None
            if forced_action_latents is None
            else packed_inference_layout.compose_current_action_sequence(
                forced_action_latents
            )
        )

        def _build_packed_action_pre(
            *,
            action_tokens: torch.Tensor,
            action_timestep: torch.Tensor,
            current_clean_action_for_step: torch.Tensor,
        ):
            if history_actions is None:
                noisy_action_sequence = action_tokens
                noisy_action_timesteps = action_timestep
                clean_action_sequence = current_clean_action_for_step
            else:
                noisy_action_sequence = torch.cat([history_actions, action_tokens], dim=1)
                noisy_action_timesteps = torch.cat(
                    [
                        torch.zeros(batch_size, history_action_tokens, device=device, dtype=torch.float32),
                        action_timestep,
                    ],
                    dim=1,
                )
                clean_action_sequence = torch.cat([history_actions, current_clean_action_for_step], dim=1)
            packed_action_tokens = torch.cat([noisy_action_sequence, clean_action_sequence], dim=1)
            packed_action_hidden_context = self.conditioning.action_hidden_context_for_tokens(
                visual_tower,
                video_hidden_proprio_sequence,
                action_tokens=noisy_action_sequence,
                action_tokens_per_frame=int(action_tokens_per_frame),
                copies=2,
            )
            packed_action_timesteps = torch.cat(
                [
                    noisy_action_timesteps,
                    torch.zeros(
                        batch_size,
                        history_action_tokens + current_action_sequence_tokens,
                        device=device,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
            return self.action_expert.pre_dit(
                action_tokens=packed_action_tokens,
                timestep=packed_action_timesteps,
                context=text_context,
                action_grid_ids=packed_action_grid_ids,
                hidden_context=packed_action_hidden_context,
            )

        def _run_packed_step(
            *,
            video_timestep: torch.Tensor,
            action_timestep: torch.Tensor,
            current_clean_video_for_step: torch.Tensor,
            current_clean_action_for_step: torch.Tensor,
        ):
            dense_video_timestep = torch.cat([history_video_timesteps, video_timestep], dim=1)
            packed_video_hidden_context = (
                self.conditioning.video_hidden_context_for_tokens(
                    visual_tower,
                    video_hidden_proprio_sequence,
                    video_latents=predicted_video_sequence,
                    copies=2,
                )
                if apply_video_hidden_proprio
                else None
            )
            packed_action_pre = _build_packed_action_pre(
                action_tokens=packed_inference_layout.compose_current_action_sequence(
                    action_sample
                ),
                action_timestep=action_timestep,
                current_clean_action_for_step=current_clean_action_for_step,
            )
            return forward_mot_packed_coupling_denoise(
                visual_tower=visual_tower,
                noisy_video_latents=predicted_video_sequence,
                clean_video_latents=history.prepend_video(
                    current_clean_video_for_step
                ),
                noisy_video_timesteps=dense_video_timestep,
                clean_video_timesteps=torch.zeros_like(dense_video_timestep),
                action_expert=self.action_expert,
                packed_action_pre=packed_action_pre,
                attention_profile=attention_profile,
                text_context=text_context,
                frame_start=int(sequence_frame_start),
                use_activation_checkpointing=False,
                packed_block_stack=self.packed_block_stack,
                prefer_flex_attention=False,
                video_hidden_context=packed_video_hidden_context,
            ) + (packed_action_pre,)

        def _video_timestep(value: torch.Tensor) -> torch.Tensor:
            timestep = expand_mot_scalar_timestep(
                value,
                shape=(batch_size, current_video_sequence_frames),
                device=device,
            )
            if current_video_prefix_frames > 0:
                timestep[:, :current_video_prefix_frames] = 0.0
                predicted_video_sequence[
                    :,
                    :,
                    shared_history_frames : shared_history_frames + current_video_prefix_frames,
                ] = current_clean_video[:, :, :current_video_prefix_frames]
            return timestep

        def _action_timestep(value: torch.Tensor) -> torch.Tensor:
            timestep = expand_mot_scalar_timestep(
                value,
                shape=(batch_size, current_action_sequence_tokens),
                device=device,
            )
            if current_action_prefix_tokens > 0:
                timestep[:, :current_action_prefix_tokens] = 0.0
            return timestep

        def _update_video(
            video_flow_pred: torch.Tensor,
            video_timestep: torch.Tensor,
            *,
            sigma: torch.Tensor | None = None,
            sigma_next: torch.Tensor | None = None,
        ) -> None:
            nonlocal predicted_video_sequence
            generated_start = shared_history_frames + current_video_prefix_frames
            current_video_flow = video_flow_pred[:, :, generated_start : generated_start + frame_chunk_size].contiguous()
            current_predicted_video = predicted_video_sequence[
                :,
                :,
                generated_start : generated_start + frame_chunk_size,
            ].contiguous()
            if sigma is None or sigma_next is None:
                current_predicted_video = video_scheduler.step(current_video_flow, video_timestep, current_predicted_video)
            else:
                current_predicted_video = step_mot_flow_with_sigmas(
                    current_predicted_video,
                    current_video_flow,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )
            predicted_video_sequence = torch.cat(
                [predicted_video_sequence[:, :, :generated_start], current_predicted_video],
                dim=2,
            )

        def _update_action(
            packed_action_hidden: torch.Tensor,
            packed_action_pre,
            action_timestep: torch.Tensor,
            *,
            sigma: torch.Tensor | None = None,
            sigma_next: torch.Tensor | None = None,
        ) -> None:
            nonlocal action_sample
            packed_action_flow = self.action_expert.post_dit(packed_action_hidden, packed_action_pre)
            flow_start = history_action_tokens + current_action_prefix_tokens
            action_flow_pred = packed_action_flow[:, flow_start : flow_start + action_horizon].contiguous()
            generated_action_timestep = action_timestep[:, current_action_prefix_tokens:].contiguous()
            scheduler_timestep = generated_action_timestep.reshape(-1)[0]
            if sigma is None or sigma_next is None:
                action_sample = action_scheduler.step(action_flow_pred, scheduler_timestep, action_sample)
            else:
                action_sample = step_mot_flow_with_sigmas(
                    action_sample,
                    action_flow_pred,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )

        def _coupled_action_timestep_for_video_step(
            *,
            step_index: int,
            video_timestep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            action_timestep = action_scheduler.timesteps[int(step_index)]
            shared_sigma = None
            shared_sigma_next = None
            if couple_action_video_sigmas:
                shared_sigma = video_scheduler.sigmas[int(step_index)].to(device=device, dtype=torch.float32)
                shared_sigma_next = mot_scheduler_next_sigma(video_scheduler, int(step_index)).to(
                    device=device,
                    dtype=torch.float32,
                )
                if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                    if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                        raise RuntimeError(
                            "M5 match-sigma same-step inference requires an action timestep lookup scheduler."
                        )
                    action_timestep = timesteps_matching_sigmas(
                        action_timestep_lookup_scheduler,
                        shared_sigma.reshape(1),
                    )[0].to(device=device, dtype=torch.float32)
                elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
                    action_timestep = video_timestep.to(device=device, dtype=torch.float32)
            return action_timestep, shared_sigma, shared_sigma_next

        if generalist_rollout_mode == MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO:
            if forced_clean_action_condition is None:  # pragma: no cover - guarded earlier
                raise RuntimeError("M5 FDM rollout missing clean action condition.")
            for video_timestep in video_scheduler.timesteps:
                current_video_timestep = _video_timestep(video_timestep)
                video_flow_pred, _, _ = _run_packed_step(
                    video_timestep=current_video_timestep,
                    action_timestep=zero_current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=forced_clean_action_condition,
                )
                _update_video(video_flow_pred, video_timestep)
        elif generalist_rollout_mode == MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION:
            current_clean_video = predicted_video_sequence[:, :, shared_history_frames:].contiguous()
            for step_index, video_timestep in enumerate(video_scheduler.timesteps):
                action_timestep, shared_sigma, shared_sigma_next = _coupled_action_timestep_for_video_step(
                    step_index=step_index,
                    video_timestep=video_timestep,
                )
                current_action_timestep = _action_timestep(action_timestep)
                _, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=zero_current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_action(
                    packed_action_hidden,
                    packed_action_pre,
                    current_action_timestep,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )
        elif current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
            for video_timestep in video_scheduler.timesteps:
                current_video_timestep = _video_timestep(video_timestep)
                video_flow_pred, _, _ = _run_packed_step(
                    video_timestep=current_video_timestep,
                    action_timestep=zero_current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_video(video_flow_pred, video_timestep)
            current_clean_video = predicted_video_sequence[:, :, shared_history_frames:].contiguous()
            for action_timestep in action_scheduler.timesteps:
                current_action_timestep = _action_timestep(action_timestep)
                _, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=zero_current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_action(packed_action_hidden, packed_action_pre, current_action_timestep)
        elif current_block_coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO:
            for action_timestep in action_scheduler.timesteps:
                current_action_timestep = _action_timestep(action_timestep)
                _, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=zero_current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_action(packed_action_hidden, packed_action_pre, current_action_timestep)
            if not action_only_rollout:
                current_clean_action = packed_inference_layout.compose_current_action_sequence(
                    action_sample
                )
                for video_timestep in video_scheduler.timesteps:
                    current_video_timestep = _video_timestep(video_timestep)
                    video_flow_pred, _, _ = _run_packed_step(
                        video_timestep=current_video_timestep,
                        action_timestep=zero_current_action_timestep,
                        current_clean_video_for_step=current_clean_video,
                        current_clean_action_for_step=current_clean_action,
                    )
                    _update_video(video_flow_pred, video_timestep)
        else:
            for step_index, video_timestep in enumerate(video_scheduler.timesteps):
                action_timestep, shared_sigma, shared_sigma_next = _coupled_action_timestep_for_video_step(
                    step_index=step_index,
                    video_timestep=video_timestep,
                )
                current_video_timestep = _video_timestep(video_timestep)
                current_action_timestep = _action_timestep(action_timestep)
                video_flow_pred, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_video(
                    video_flow_pred,
                    video_timestep,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )
                _update_action(
                    packed_action_hidden,
                    packed_action_pre,
                    current_action_timestep,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )

        clean_video_prefix_frames = shared_history_frames + current_video_prefix_frames
        if action_only_rollout:
            predicted_chunk_latents = predicted_video_sequence.new_empty(
                batch_size,
                predicted_video_sequence.shape[1],
                0,
                latent_height,
                latent_width,
            )
            next_clean_context = clean_video_sequence[:, :, :clean_video_prefix_frames].contiguous()
            pending_predicted_video_frames = 0
        else:
            predicted_chunk_latents = predicted_video_sequence[:, :, -frame_chunk_size:].contiguous()
            next_clean_context = torch.cat(
                [clean_video_sequence[:, :, :clean_video_prefix_frames], predicted_chunk_latents],
                dim=2,
            )
            pending_predicted_video_frames = frame_chunk_size
        runtime_state.past_clean_latents = next_clean_context[:, :, -history_window_frames:].detach()
        if video_hidden_proprio_sequence is not None:
            next_hidden_context = video_hidden_proprio_sequence[
                :,
                : clean_video_prefix_frames + pending_predicted_video_frames,
            ].contiguous()
            runtime_state.past_hidden_proprio_states = next_hidden_context[:, -history_window_frames:].detach()
        else:
            runtime_state.past_hidden_proprio_states = None
        action_history_commit = action_sample if commit_action_latents is None else commit_action_latents
        if history_actions is None:
            next_clean_actions = action_history_commit
        else:
            next_clean_actions = torch.cat([history_actions, action_history_commit], dim=1)
        max_action_history_tokens = history_window_frames * action_tokens_per_frame
        runtime_state.past_clean_actions = next_clean_actions[:, -max_action_history_tokens:].detach()
        runtime_state.next_condition_frame_start = int(generation_frame_start + frame_chunk_size)
        runtime_state.pending_predicted_video_frames = int(pending_predicted_video_frames)
        next_state = infer_state
        next_state.step_index += 1
        next_state.cursor.current_start_frame = int(generation_frame_start + frame_chunk_size)
        next_state.variant_state = runtime_state
        return PolicyInferOutput(
            policy_features=action_sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "action_conditioning_mode": str(context.extra.get("action_conditioning_mode", "vanilla_joint_rollout")),
                "mot_generalist_rollout_mode": generalist_rollout_mode.value,
                "generation_frame_start": int(generation_frame_start),
                "mot_action_only_rollout": bool(action_only_rollout),
                "predicted_latents": predicted_chunk_latents.detach(),
                "predicted_video_latents": predicted_chunk_latents.detach(),
                "mot_first_step_bootstrap": first_step_bootstrap,
                "mot_action_cond_tokens": 0,
                "mot_invalid_startup_action_tokens": int(current_action_prefix_tokens),
                "mot_action_context_invalid_tokens": int(
                    attention_profile.metadata.get("invalid_action_context_tokens", 0)
                ),
                "mot_generalist_mode_text_token": (
                    generalist_rollout_mode.value
                    if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(
                    getattr(runtime_state, "generalist_mode_text_token_count", 0)
                ),
                "forced_action_denoise": generalist_rollout_mode
                == MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
                "forced_clean_action_conditioning": forced_action_latents is not None,
                "forced_video_conditioning": generalist_rollout_mode
                == MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
                "mot_diagnostic_zero_current_video_noise": diagnostic_zero_video_noise,
                "mot_diagnostic_zero_current_action_noise": diagnostic_zero_action_noise,
                "mot_attention_focus": None,
                "commit_action_override": commit_action_latents is not None,
                "returned_action_source": "predicted"
                if generalist_rollout_mode != MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO
                else "forced_action",
                "cache_action_source": "commit_action_override"
                if commit_action_latents is not None
                else "predicted",
                "mot_history_anchor_frames": int(shared_history_frames),
                "mot_packed_history_debug": {
                    "past_clean_latent_frames": 0 if past_clean_latents is None else int(past_clean_latents.shape[2]),
                    "past_clean_action_frames": 0 if past_clean_actions is None else int(past_clean_actions.shape[1] // action_tokens_per_frame),
                    "shared_history_frames": int(shared_history_frames),
                    "current_observed_latent_frames": int(video_latents.shape[2]),
                    "current_clean_condition_frames": int(current_clean_video.shape[2]),
                    "packed_video_frames": int(shared_history_frames + current_video_sequence_frames),
                    "packed_action_frames": int(shared_history_frames + current_video_prefix_frames + frame_chunk_size),
                    "rollout_frame_chunk_size": int(frame_chunk_size),
                    "rollout_action_horizon": int(action_horizon),
                    "current_action_flow_start": int(history_action_tokens + current_action_prefix_tokens),
                    "current_action_flow_end": int(history_action_tokens + current_action_prefix_tokens + action_horizon),
                    "history_window_frames": int(history_window_frames),
                    "inference_window_size": int(inference_window_size),
                    "max_history_frames": int(max_history_frames),
                    "next_past_clean_latent_frames": int(runtime_state.past_clean_latents.shape[2]),
                    "next_past_clean_action_frames": int(runtime_state.past_clean_actions.shape[1] // action_tokens_per_frame),
                    "pending_predicted_video_frames": int(runtime_state.pending_predicted_video_frames),
                    "sequence_frame_start": int(sequence_frame_start),
                    "current_frame_start": int(current_start_frame),
                    "current_video_prefix_frames": int(current_video_prefix_frames),
                    "current_action_prefix_tokens": int(current_action_prefix_tokens),
                    "mode_uses_packed_cache": True,
                    "joint_timestep_coupling": joint_timestep_coupling.value,
                    "coupled_action_video_sigmas": bool(couple_action_video_sigmas),
                },
                "mot_infer_artifacts": MoTInferArtifacts(
                    action_pred=action_sample,
                    predicted_latents=predicted_chunk_latents.detach(),
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                ),
            },
        )
