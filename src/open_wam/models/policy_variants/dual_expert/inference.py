"""Dual Expert preparation for the shared video/action denoising executor."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    DynamicsObjective,
    InferenceConfig,
    JointTimestepCoupling,
    TrainingConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.models.common.video_action_inference import (
    VideoActionFlowInput,
    denoise_video_action,
)
from open_wam.models.common.denoising_cache import (
    DenoisingCache,
    select_attention_profile,
)
from open_wam.models.common.packed_token_layout import PackedTokenStream
from open_wam.models.common.dynamics_objectives import (
    is_conditional_dynamics_objective,
    resolve_dynamics_rollout_plan,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import (
    DecoderArtifactEnvelope,
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyOutputModality,
)
from ..output_semantics import video_action_program_output_modalities
from .attention_packed import build_dual_expert_packed_coupling_attention_profile
from .conditioning import DualExpertConditioning
from open_wam.models.common.video_action_state import VideoActionRolloutState
from .coupling_semantics import (
    resolve_dual_expert_current_block_coupling,
    resolve_dual_expert_joint_timestep_coupling,
)
from open_wam.models.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertInferArtifacts,
)
from .dual_stream_execution import forward_dual_expert_packed_coupling_denoise
from open_wam.models.common.video_action_layout import (
    prepare_video_action_sequence,
)
from .modules import DualExpertActionExpert
from .packed_block import DualExpertPackedBlockStack
from .rollout_geometry import (
    resolve_dual_expert_inference_output_request,
    resolve_dual_expert_sequence_actions_per_frame,
)
from .sequence_layout import build_action_grid_ids_for_sequence


@dataclass(frozen=True)
class DualExpertInferenceProgram:
    """Prepare model-space inputs without selecting a method-specific runner."""

    config: DualExpertPolicyConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    conditioning: DualExpertConditioning
    action_expert: DualExpertActionExpert
    packed_block_stack: DualExpertPackedBlockStack | None
    action_dim: int
    action_horizon: int

    def run(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: VideoActionRolloutState,
    ) -> PolicyInferOutput:
        current_block_coupling = resolve_dual_expert_current_block_coupling(self.config)
        native_modalities = video_action_program_output_modalities(self.config.program)
        output_request = resolve_dual_expert_inference_output_request(
            context,
            current_block_coupling=current_block_coupling,
            native_modalities=native_modalities,
        )
        action_only_rollout = (
            output_request.modalities != native_modalities
            and output_request.modalities == frozenset({PolicyOutputModality.ACTION})
        )
        supplied_video = context.video_conditioned_action
        dynamics_rollout_plan = resolve_dynamics_rollout_plan(
            program=self.config.program,
            request=context.dynamics,
        )
        generalist_rollout_mode = dynamics_rollout_plan.objective
        if (
            is_conditional_dynamics_objective(generalist_rollout_mode)
            and current_block_coupling != CurrentBlockCoupling.JOINT
        ):
            raise ValueError(
                "dual-expert conditional dynamics rollout requires packed joint "
                "coupling, "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        inference_window_size = int(
            context.require_temporal_geometry().attention_window_size
        )
        device = next(visual_tower.core.parameters()).device
        action_device = next(self.action_expert.parameters()).device
        if action_device != device:
            raise ValueError(
                "dual-expert packed coupling inference currently requires visual tower and action expert on the same device, "
                f"got visual_device={device}, action_device={action_device}."
            )
        dtype = next(self.action_expert.parameters()).dtype
        batch_size = int(visual_outputs.frontend.video_latents.shape[0])
        frame_chunk_size = int(context.require_temporal_geometry().frame_chunk_size)
        action_tokens_per_frame = resolve_dual_expert_sequence_actions_per_frame(
            action_horizon=int(self.action_horizon),
            frame_chunk_size=int(self.inference_config.frame_chunk_size),
        )
        action_horizon = frame_chunk_size * action_tokens_per_frame
        rollout_geometry = dynamics_rollout_plan.resolve_geometry(
            fallback_frame_chunk_size=frame_chunk_size,
            fallback_attention_window_size=inference_window_size,
            fallback_history_stream_visibility=self.config.history_stream_visibility,
        )
        required_frame_chunk_size = rollout_geometry.frame_chunk_size
        inference_window_size = rollout_geometry.attention_window_size
        frame_chunk_size = required_frame_chunk_size
        action_horizon = frame_chunk_size * action_tokens_per_frame
        video_latents = visual_outputs.frontend.video_latents.to(
            device=device, dtype=dtype
        )
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        video_tokens_per_frame = int(
            visual_outputs.frontend.token_grid.tokens_per_frame
        )
        sequence = prepare_video_action_sequence(
            observation=video_latents,
            state=replace(infer_state, variant_state=runtime_state),
            chunk_frames=frame_chunk_size,
            actions_per_frame=action_tokens_per_frame,
            action_dim=self.action_dim,
            window_size=inference_window_size,
            dynamics=dynamics_rollout_plan,
            supplied_video=None
            if supplied_video is None
            else supplied_video.generated_video,
        )
        current_start_frame = infer_state.cursor.current_start_frame
        current_video_prefix_frames = sequence.startup_frames
        first_step_bootstrap = bool(current_video_prefix_frames)
        current_action_prefix_tokens = (
            current_video_prefix_frames * action_tokens_per_frame
        )
        current_video_sequence_frames = current_video_prefix_frames + frame_chunk_size
        current_action_sequence_tokens = (
            current_video_sequence_frames * action_tokens_per_frame
        )
        generation_frame_start = sequence.generation_start
        shared_history_frames = sequence.history_frames - current_video_prefix_frames
        history_action_tokens = shared_history_frames * action_tokens_per_frame
        max_history_frames = sequence.max_history_frames
        history_window_frames = sequence.retention_frames
        past_clean_latents = runtime_state.past_clean_latents
        past_clean_actions = runtime_state.past_clean_actions
        noisy_video_sequence = clean_video_sequence = sequence.video
        current_clean_video = sequence.video[:, :, shared_history_frames:]
        video_hidden_proprio_sequence = sequence.proprio
        packed_action_context_mask = sequence.action_mask.float()
        action_sequence = action_condition = sequence.action
        forced_action_latents = sequence.forced_action
        commit_action_latents = sequence.commit_action
        diagnostic_zero_video_noise = context.initial_video_noise is not None
        diagnostic_zero_action_noise = context.initial_action_noise is not None

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
            self.config.generalist_mode_text_token
            and runtime_state.generalist_mode_text_token_count <= 0
        ):
            text_context, token_count = (
                self.conditioning.append_generalist_mode_text_token(
                    visual_tower,
                    text_context,
                    generalist_rollout_mode,
                )
            )
            runtime_state = replace(
                runtime_state,
                text_context=text_context,
                generalist_mode_text_token_count=int(token_count),
            )

        sequence_frame_start = sequence.frame_start
        packed_chunk_origin_frame = sequence.history_frames
        attention_profile = build_dual_expert_packed_coupling_attention_profile(
            num_video_frames=shared_history_frames + current_video_sequence_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            num_action_frames=shared_history_frames + current_video_sequence_frames,
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=frame_chunk_size,
            device=device,
            attention_window_size=inference_window_size,
            current_block_coupling=current_block_coupling,
            chunk_origin_frame=packed_chunk_origin_frame,
            action_context_mask=packed_action_context_mask,
            build_dense_masks=True,
            build_flex_masks=False,
            history_stream_visibility=rollout_geometry.history_stream_visibility,
            conditional_history_policy=rollout_geometry.conditional_history_policy,
        )
        action_sequence_grid_ids = build_action_grid_ids_for_sequence(
            batch_size=batch_size,
            seq_len=history_action_tokens + current_action_sequence_tokens,
            action_tokens_per_frame=action_tokens_per_frame,
            device=device,
            frame_shift=int(sequence_frame_start),
        )
        packed_action_grid_ids = torch.cat(
            [action_sequence_grid_ids, action_sequence_grid_ids],
            dim=2,
        )

        def predict(step: VideoActionFlowInput, cache: DenoisingCache | None):
            layout = step.attention.token_layout
            action_needed = bool(
                (
                    step.required_tokens
                    & (layout.stream_id == PackedTokenStream.ACTION)
                ).any()
            )
            action_pre = None
            profile = step.attention
            invariant = layout.noise_id == 1
            required = step.required_tokens
            if action_needed:
                action_tokens = torch.cat([step.noisy_action, step.clean_action], dim=1)
                packed_action_hidden_context = (
                    self.conditioning.action_hidden_context_for_tokens(
                        visual_tower,
                        step.proprio_frames,
                        action_tokens=step.noisy_action,
                        action_tokens_per_frame=action_tokens_per_frame,
                        copies=2,
                    )
                )
                action_pre = self.action_expert.pre_dit(
                    action_tokens=action_tokens,
                    timestep=torch.cat(
                        [
                            step.action_timesteps,
                            torch.zeros_like(step.action_timesteps),
                        ],
                        dim=1,
                    ),
                    context=step.text_context,
                    action_grid_ids=packed_action_grid_ids,
                    hidden_context=packed_action_hidden_context,
                )
            else:
                video_ids = (
                    (layout.stream_id == PackedTokenStream.VIDEO).nonzero().flatten()
                )
                profile = select_attention_profile(profile, video_ids, video_ids)
                invariant, required = invariant[video_ids], required[video_ids]
            if cache is not None and not cache.bound:
                action_count = (
                    0 if action_pre is None else int(action_pre.tokens.shape[1])
                )
                cache.bind(
                    profile=profile,
                    invariant_tokens=invariant,
                    required_tokens=required,
                    stream_lengths=(invariant.numel() - action_count, action_count),
                    num_layers=len(self.packed_block_stack.packed_blocks),
                )
            video_hidden_context = (
                self.conditioning.video_hidden_context_for_tokens(
                    visual_tower,
                    step.proprio_frames,
                    video_latents=step.noisy_video,
                    copies=2,
                )
                if not self.conditioning.uses_legacy_prefix_contract()
                else None
            )
            video_flow, action_hidden = forward_dual_expert_packed_coupling_denoise(
                visual_tower=visual_tower,
                noisy_video_latents=step.noisy_video,
                clean_video_latents=step.clean_video,
                noisy_video_timesteps=step.video_timesteps,
                clean_video_timesteps=torch.zeros_like(step.video_timesteps),
                packed_action_pre=action_pre,
                attention_profile=profile,
                text_context=step.text_context,
                frame_start=step.frame_start,
                use_activation_checkpointing=False,
                packed_block_stack=self.packed_block_stack,
                denoising_cache=cache,
                prefer_flex_attention=False,
                video_hidden_context=video_hidden_context,
            )
            if action_pre is None:
                return video_flow, torch.zeros_like(step.noisy_action)
            action_flow = self.action_expert.post_dit(action_hidden, action_pre)
            return video_flow, action_flow[:, : step.noisy_action.shape[1]]

        negative_text_context = (
            visual_outputs.frontend.conditioning.negative_text_context
        )
        if (
            negative_text_context is not None
            and not dynamics_rollout_plan.semantics.drop_text_conditioning
        ):
            negative_text_context = self.conditioning.resolve_text_context(
                visual_tower,
                negative_text_context,
                runtime_state.proprio_state,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                materialize_if_missing=False,
            )
            if self.config.generalist_mode_text_token:
                negative_text_context, _ = (
                    self.conditioning.append_generalist_mode_text_token(
                        visual_tower,
                        negative_text_context,
                        generalist_rollout_mode,
                    )
                )
        prepared = VideoActionFlowInput(
            noisy_video=noisy_video_sequence,
            clean_video=clean_video_sequence,
            video_timesteps=torch.zeros(
                batch_size, noisy_video_sequence.shape[2], device=device
            ),
            noisy_action=action_sequence,
            clean_action=action_condition,
            action_timesteps=torch.zeros(
                batch_size, action_sequence.shape[1], device=device
            ),
            attention=attention_profile,
            text_context=text_context,
            frame_start=sequence_frame_start,
            proprio_frames=video_hidden_proprio_sequence,
        )
        predicted_video_sequence, full_actions = denoise_video_action(
            prepared,
            predict=predict,
            training=self.training_config,
            inference=self.inference_config,
            coupling=current_block_coupling,
            timestep_coupling=resolve_dual_expert_joint_timestep_coupling(self.config),
            dynamics=dynamics_rollout_plan,
            video_start=shared_history_frames + current_video_prefix_frames,
            action_start=history_action_tokens + current_action_prefix_tokens,
            negative_text_context=negative_text_context,
            requested=output_request.modalities,
            supplied=frozenset({PolicyOutputModality.VIDEO})
            if supplied_video is not None
            else frozenset(),
            initial_video_noise=context.initial_video_noise,
            initial_action_noise=context.initial_action_noise,
        )
        action_sample = full_actions[
            :, history_action_tokens + current_action_prefix_tokens :
        ]
        if action_only_rollout and supplied_video is None:
            predicted_chunk_latents = predicted_video_sequence.new_empty(
                batch_size,
                predicted_video_sequence.shape[1],
                0,
                latent_height,
                latent_width,
            )
            pending_predicted_video_frames = 0
        else:
            predicted_chunk_latents = predicted_video_sequence[
                :, :, -frame_chunk_size:
            ].contiguous()
            pending_predicted_video_frames = frame_chunk_size
        next_state = sequence.publish(
            replace(infer_state, variant_state=runtime_state),
            video=predicted_video_sequence,
            action=full_actions,
            retain_future_video=bool(pending_predicted_video_frames),
            retain_future_action=PolicyOutputModality.ACTION
            in output_request.modalities,
        )
        runtime_state = next_state.variant_state
        returned_action = (
            action_sample
            if PolicyOutputModality.ACTION in output_request.modalities
            or forced_action_latents is not None
            else action_sample[:, :0]
        )
        decoder_payload = DualExpertInferArtifacts(
            action_pred=returned_action,
            predicted_latents=predicted_chunk_latents.detach(),
            condition_mode=str(self.config.condition_mode),
            program=self.config.program.value,
        )
        return PolicyInferOutput(
            policy_features=action_sample.new_zeros(
                batch_size, 0, self.action_expert.hidden_size
            ),
            next_state=next_state,
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
            ),
            generated_video=(
                PolicyGeneratedVideo(
                    latents=predicted_chunk_latents.detach(),
                    frame_start=int(generation_frame_start),
                    latent_space_identity=(
                        visual_outputs.frontend.latent_space_identity
                    ),
                )
                if (
                    dynamics_rollout_plan.semantics.video_loss_active
                    and int(predicted_chunk_latents.shape[2]) > 0
                )
                else None
            ),
            generation_frame_start=int(generation_frame_start),
            aux={
                "variant": self.config.name,
                "architecture": "dual_expert",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "action_conditioning_mode": generalist_rollout_mode.value,
                "generation_frame_start": int(generation_frame_start),
                "action_only_rollout": bool(action_only_rollout),
                "dual_expert_video_only_rollout": PolicyOutputModality.ACTION
                not in output_request.modalities,
                "action_pred_executable": PolicyOutputModality.ACTION
                in output_request.modalities,
                "predicted_latents": predicted_chunk_latents.detach(),
                "predicted_video_latents": predicted_chunk_latents.detach(),
                "dual_expert_first_step_bootstrap": first_step_bootstrap,
                "dual_expert_action_cond_tokens": 0,
                "dual_expert_invalid_startup_action_tokens": int(
                    current_action_prefix_tokens
                ),
                "dual_expert_action_context_invalid_tokens": int(
                    attention_profile.metadata.get("invalid_action_context_tokens", 0)
                ),
                "dual_expert_generalist_mode_text_token": (
                    generalist_rollout_mode.value
                    if int(runtime_state.generalist_mode_text_token_count) > 0
                    else None
                ),
                "dual_expert_generalist_mode_text_token_count": int(
                    runtime_state.generalist_mode_text_token_count
                ),
                "forced_action_denoise": generalist_rollout_mode
                == DynamicsObjective.ACTION_CONDITIONED_VIDEO,
                "forced_clean_action_conditioning": forced_action_latents is not None,
                "forced_video_conditioning": generalist_rollout_mode
                == DynamicsObjective.VIDEO_CONDITIONED_ACTION,
                "dual_expert_diagnostic_zero_current_video_noise": diagnostic_zero_video_noise,
                "dual_expert_diagnostic_zero_current_action_noise": diagnostic_zero_action_noise,
                "dual_expert_attention_focus": None,
                "commit_action_override": commit_action_latents is not None,
                "returned_action_source": "predicted"
                if generalist_rollout_mode != DynamicsObjective.ACTION_CONDITIONED_VIDEO
                else "forced_action",
                "cache_action_source": "commit_action_override"
                if commit_action_latents is not None
                else "predicted",
                "dual_expert_history_anchor_frames": int(shared_history_frames),
                "dual_expert_packed_history_debug": {
                    "coupled_action_video_sigmas": resolve_dual_expert_joint_timestep_coupling(
                        self.config
                    )
                    is not JointTimestepCoupling.INDEPENDENT,
                    "past_clean_latent_frames": 0
                    if past_clean_latents is None
                    else int(past_clean_latents.shape[2]),
                    "past_clean_action_frames": 0
                    if past_clean_actions is None
                    else int(past_clean_actions.shape[1] // action_tokens_per_frame),
                    "shared_history_frames": int(shared_history_frames),
                    "current_observed_latent_frames": int(video_latents.shape[2]),
                    "current_clean_condition_frames": int(current_clean_video.shape[2]),
                    "packed_video_frames": int(
                        shared_history_frames + current_video_sequence_frames
                    ),
                    "packed_action_frames": int(
                        shared_history_frames
                        + current_video_prefix_frames
                        + frame_chunk_size
                    ),
                    "rollout_frame_chunk_size": int(frame_chunk_size),
                    "rollout_action_horizon": int(action_horizon),
                    "current_action_flow_start": int(
                        history_action_tokens + current_action_prefix_tokens
                    ),
                    "current_action_flow_end": int(
                        history_action_tokens
                        + current_action_prefix_tokens
                        + action_horizon
                    ),
                    "history_window_frames": int(history_window_frames),
                    "inference_window_size": int(inference_window_size),
                    "max_history_frames": int(max_history_frames),
                    "next_past_clean_latent_frames": int(
                        runtime_state.past_clean_latents.shape[2]
                    ),
                    "next_past_clean_action_frames": int(
                        runtime_state.past_clean_actions.shape[1]
                        // action_tokens_per_frame
                    ),
                    "pending_predicted_video_frames": int(
                        runtime_state.pending_predicted_video_frames
                    ),
                    "sequence_frame_start": int(sequence_frame_start),
                    "current_frame_start": int(current_start_frame),
                    "current_video_prefix_frames": int(current_video_prefix_frames),
                    "current_action_prefix_tokens": int(current_action_prefix_tokens),
                    "use_cache": bool(self.inference_config.use_cache),
                    "joint_timestep_coupling": self.config.joint_timestep_coupling.value,
                },
            },
        )
