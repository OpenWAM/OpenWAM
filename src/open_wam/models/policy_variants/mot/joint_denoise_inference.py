"""Simultaneous joint-denoise inference program for MoT policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import (
    InferenceConfig,
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    TrainingConfig,
)
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.models.common.flow_inference import (
    build_action_flow_match_inference_scheduler,
    build_video_flow_match_inference_scheduler,
)
from open_wam.models.common.flow_schedule import (
    expand_scalar_timestep as expand_mot_scalar_timestep,
    explicit_sigma_euler_step as step_mot_flow_with_sigmas,
    timesteps_matching_sigmas,
    zero_terminal_next_sigma as mot_scheduler_next_sigma,
)
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import PolicyInferContext, PolicyInferOutput, PolicyInferState
from .attention_unpacked import build_mot_attention_mask
from .conditioning import MoTConditioning
from .contracts import MoTInferArtifacts, MoTRuntimeState
from .dual_stream_execution import forward_joint_video_action_denoise
from .generalist_modes import (
    generalist_rollout_enabled as _mot_generalist_rollout_enabled,
    is_generalist_conditional_rollout as _is_mot_generalist_conditional_rollout,
    resolve_generalist_rollout_mode as _resolve_mot_generalist_rollout_mode,
)
from .modules import MoTActionExpert
from .runtime_routing import (
    is_mot_same_step_coupling,
    resolve_mot_current_block_coupling,
    resolve_mot_joint_timestep_coupling,
    resolve_mot_rollout_frame_chunk_size,
    should_couple_mot_action_to_video_sigmas,
)


@dataclass(frozen=True)
class MoTJointDenoiseInferenceProgram:
    """Execute the historical simultaneous video/action denoise route."""

    config: MoTPolicyConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    conditioning: MoTConditioning
    action_expert: MoTActionExpert
    action_dim: int
    action_horizon: int

    def run(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: MoTRuntimeState,
    ) -> PolicyInferOutput:
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        generalist_rollout_mode = (
            _resolve_mot_generalist_rollout_mode(context)
            if _mot_generalist_rollout_enabled(self.config)
            else MoTGeneralistTrainingMode.JOINT
        )
        if _is_mot_generalist_conditional_rollout(generalist_rollout_mode):
            raise ValueError(
                "M5 GJD conditional FDM/IDM rollout is implemented for native packed joint coupling, "
                f"not legacy runtime_mode={self.config.runtime_mode!r}."
            )
        if not is_mot_same_step_coupling(current_block_coupling):
            raise NotImplementedError(
                "M5 joint_denoise inference supports same-step couplings only; "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        device = next(visual_tower.core.parameters()).device
        action_device = next(self.action_expert.parameters()).device
        if action_device != device:
            raise ValueError(
                "MoT joint_denoise inference currently requires visual tower and action expert on the same device, "
                f"got visual_device={device}, action_device={action_device}, "
                f"runtime_mode={self.config.runtime_mode!r}."
            )
        dtype = next(self.action_expert.parameters()).dtype
        batch_size = visual_outputs.frontend.video_latents.shape[0]
        observed_prefix_frames = int(self.config.video_prefix_frames)
        video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
        if video_latents.shape[2] <= observed_prefix_frames:
            raise ValueError(
                "MoT two-stream inference requires at least one future frame after the observed prefix, "
                f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}, "
                f"runtime_mode={self.config.runtime_mode!r}."
            )
        if self.inference_config.video_num_inference_steps != self.inference_config.action_num_inference_steps:
            raise ValueError(
                "MoT two-stream inference currently requires matching video/action inference step counts, "
                f"got video_num_inference_steps={self.inference_config.video_num_inference_steps}, "
                f"action_num_inference_steps={self.inference_config.action_num_inference_steps}, "
                f"runtime_mode={self.config.runtime_mode!r}."
            )
        frame_chunk_size, action_horizon, action_tokens_per_frame = resolve_mot_rollout_frame_chunk_size(
            context,
            default_frame_chunk_size=int(self.inference_config.frame_chunk_size),
            base_action_horizon=int(self.action_horizon),
        )
        observed_prefix = video_latents[:, :, :observed_prefix_frames]
        future_template = video_latents[:, :, observed_prefix_frames:]
        if future_template.shape[2] > frame_chunk_size:
            future_template = future_template[:, :, :frame_chunk_size].contiguous()
        elif future_template.shape[2] < frame_chunk_size:
            missing_frames = int(frame_chunk_size - future_template.shape[2])
            future_template = torch.cat(
                [
                    future_template,
                    future_template[:, :, -1:].expand(-1, -1, missing_frames, -1, -1),
                ],
                dim=2,
            ).contiguous()
        noisy_video_latents = torch.cat(
            [
                observed_prefix,
                torch.randn_like(future_template, device=device, dtype=dtype),
            ],
            dim=2,
        )
        rollout_video_frames = int(noisy_video_latents.shape[2])
        action_scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        video_scheduler = build_video_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
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
        sample = torch.randn(
            batch_size,
            action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        text_context = runtime_state.text_context
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
        hidden_proprio_state = runtime_state.hidden_proprio_state
        hidden_proprio_sequence = None
        if hidden_proprio_state is not None:
            hidden_proprio_sequence = hidden_proprio_state.to(device=device, dtype=dtype)[:, None, :].expand(
                -1,
                rollout_video_frames,
                -1,
            )
        attention_mask = build_mot_attention_mask(
            video_seq_len=visual_outputs.frontend.token_grid.tokens_per_frame * rollout_video_frames,
            action_seq_len=action_horizon,
            device=device,
            condition_mode=self.config.condition_mode,
            video_tokens_per_frame=visual_outputs.frontend.token_grid.tokens_per_frame,
            video_can_attend_action=self.config.video_can_attend_action,
            action_tokens_per_frame=action_tokens_per_frame,
            action_chunk_size_frames=frame_chunk_size,
            clean_video_frames=observed_prefix_frames,
            current_block_coupling=current_block_coupling,
        )
        for step_index, video_timestep in enumerate(video_scheduler.timesteps):
            action_timestep = action_scheduler.timesteps[step_index]
            shared_sigma = None
            shared_sigma_next = None
            if couple_action_video_sigmas:
                shared_sigma = video_scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
                shared_sigma_next = mot_scheduler_next_sigma(video_scheduler, step_index).to(
                    device=device,
                    dtype=torch.float32,
                )
                if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                    if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                        raise RuntimeError("M5 match-sigma joint denoise requires an action timestep lookup scheduler.")
                    action_timestep = timesteps_matching_sigmas(
                        action_timestep_lookup_scheduler,
                        shared_sigma.reshape(1),
                    )[0].to(device=device, dtype=torch.float32)
                elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
                    action_timestep = video_timestep.to(device=device, dtype=torch.float32)
            dense_video_timestep = expand_mot_scalar_timestep(
                video_timestep,
                shape=(batch_size, rollout_video_frames),
                device=device,
            )
            dense_video_timestep[:, :observed_prefix_frames] = 0.0
            dense_action_timestep = expand_mot_scalar_timestep(
                action_timestep,
                shape=(batch_size, action_horizon),
                device=device,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=sample,
                timestep=dense_action_timestep,
                context=text_context,
                hidden_context=self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_sequence,
                    action_tokens=sample,
                    action_tokens_per_frame=action_tokens_per_frame,
                ),
            )
            video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
                visual_tower=visual_tower,
                noisy_video_latents=noisy_video_latents,
                video_timesteps=dense_video_timestep,
                action_expert=self.action_expert,
                action_pre=action_pre,
                text_context=text_context,
                attention_mask=attention_mask,
                frame_start=int(infer_state.cursor.current_start_frame),
                video_hidden_context=self.conditioning.video_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_sequence,
                    video_latents=noisy_video_latents,
                ),
            )
            flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
            if shared_sigma is None or shared_sigma_next is None:
                noisy_video_latents = video_scheduler.step(video_flow_pred, video_timestep, noisy_video_latents)
            else:
                noisy_video_latents = step_mot_flow_with_sigmas(
                    noisy_video_latents,
                    video_flow_pred,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )
            noisy_video_latents[:, :, :observed_prefix_frames] = observed_prefix
            if shared_sigma is None or shared_sigma_next is None:
                sample = action_scheduler.step(flow_pred, action_timestep, sample)
            else:
                sample = step_mot_flow_with_sigmas(
                    sample,
                    flow_pred,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )
        predicted_latents = noisy_video_latents[:, :, observed_prefix_frames:].detach()
        next_state = infer_state
        next_state.step_index += 1
        next_state.variant_state = runtime_state
        return PolicyInferOutput(
            policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "rollout_action_horizon": int(action_horizon),
                "predicted_latents": predicted_latents,
                "predicted_video_latents": predicted_latents,
                "mot_infer_artifacts": MoTInferArtifacts(
                    action_pred=sample,
                    predicted_latents=predicted_latents,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                ),
                "joint_timestep_coupling": joint_timestep_coupling.value,
                "coupled_action_video_sigmas": bool(couple_action_video_sigmas),
                "mot_generalist_mode_text_token": (
                    generalist_rollout_mode.value
                    if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(
                    getattr(runtime_state, "generalist_mode_text_token_count", 0)
                ),
            },
        )
