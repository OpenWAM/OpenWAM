"""Shared-transformer packing for the common video/action inference program."""

from __future__ import annotations

from dataclasses import replace

import torch
from einops import rearrange

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import PolicyOutputModality
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.dynamics_contracts import DynamicsRolloutRequest
from open_wam.models.common.denoising_cache import DenoisingCache
from open_wam.models.common.proprio_conditioning import ProprioContextGranularity
from open_wam.models.common.chunked_attention import (
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.common.dynamics_objectives import resolve_dynamics_rollout_plan
from open_wam.models.common.video_action_inference import (
    VideoActionFlowInput,
    denoise_video_action,
)
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.models.common.video_action_layout import prepare_video_action_sequence
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.visual_tower.exact_runtime import build_reference_mesh_id
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    build_chunked_dual_stream_exact_train_program,
)
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore

from ..contracts import (
    PolicyInferState,
    PolicyInferenceOutputRequest,
    PolicyInferContext,
)

from .conditional_rollout import uses_dynamics_mode_text_token
from .inference_artifacts import ParallelInferArtifacts
from .inference_conditioning import append_generalist_mode_text_context
from .proprio_conditioning import inject_deprecated_proprio_text_context
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling,
    resolve_parallel_current_block_coupling,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract,
)


@torch.no_grad()
def run_parallel_inference(
    *,
    transformer: SharedVideoTransformerCore,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_state: PolicyInferState,
    dynamics: DynamicsRolloutRequest | None,
    output_request: PolicyInferenceOutputRequest,
    context: PolicyInferContext,
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> ParallelInferArtifacts:
    """Pack tensors and project flow; stages, guidance and clocks are shared."""
    parameter = next(transformer.parameters())
    device = parameter.device
    plan = resolve_dynamics_rollout_plan(
        program=policy_config.program, request=dynamics
    )
    plan.require_generation_inputs()
    geometry = plan.resolve_geometry(
        fallback_frame_chunk_size=inference_config.frame_chunk_size,
        fallback_attention_window_size=inference_config.attention_window_size,
        fallback_history_stream_visibility=policy_config.history_stream_visibility,
    )
    chunk = geometry.frame_chunk_size
    density = policy_config.action_per_frame
    history = replace(
        infer_state.variant_state or VideoActionRolloutState(),
        proprio_state=proprio_state,
        hidden_proprio_state=hidden_proprio_state,
    )
    infer_state = replace(infer_state, variant_state=history)
    if condition_latents is None:
        condition_latents = history.past_clean_latents
    sequence = prepare_video_action_sequence(
        observation=condition_latents.to(parameter),
        state=infer_state,
        chunk_frames=chunk,
        actions_per_frame=density,
        action_dim=action_dim,
        window_size=geometry.attention_window_size,
        dynamics=plan,
        supplied_video=(
            None
            if context.video_conditioned_action is None
            else context.video_conditioned_action.generated_video
        ),
    )
    video = clean_video = sequence.video
    action = sequence.action
    action_mask = sequence.action_mask
    channel_mask = (
        action.new_ones(1, 1, action_dim)
        if action_channel_mask is None
        else action_channel_mask.reshape(1, 1, action_dim).to(action)
    )
    action = clean_action = action * channel_mask
    batch, channels, frames, height, width = video.shape
    history_frames = sequence.history_frames
    generation_start = sequence.generation_start
    frame_start = sequence.frame_start
    chunk_origin = history_frames
    text_emb = text_emb if text_emb is not None else history.text_context
    if text_emb is None:
        text_emb = video.new_zeros(
            batch, backbone_config.max_text_tokens, backbone_config.text_dim
        )
    text_emb = text_emb.to(parameter)
    if negative_text_emb is not None:
        negative_text_emb = negative_text_emb.to(parameter)
    raw_text = text_emb
    if plan.semantics.drop_text_conditioning:
        text_emb = torch.zeros_like(text_emb)
    if uses_dynamics_mode_text_token(policy_config):
        text_emb, negative_text_emb = append_generalist_mode_text_context(
            transformer,
            policy_config=policy_config,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            mode=plan.objective,
        )
    text_emb, negative_text_emb = inject_deprecated_proprio_text_context(
        transformer,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=proprio_state,
    )
    proprio_frames = sequence.proprio
    patch_t, patch_h, patch_w = transformer.patch_size
    if patch_t != 1:
        raise ValueError(
            "Frame-aligned video/action inference requires temporal patch size one."
        )
    video_grid = build_reference_mesh_id(
        frames,
        height // patch_h,
        width // patch_w,
        t=0,
        f_w=1,
        f_shift=frame_start,
        action=False,
        device=device,
    )[None].expand(batch, -1, -1)
    action_grid = build_reference_mesh_id(
        frames,
        density,
        1,
        t=1,
        f_w=1,
        f_shift=frame_start,
        action=True,
        device=device,
    )[None].expand(batch, -1, -1)
    coupling = resolve_parallel_current_block_coupling(policy_config)
    token_count = (
        2 * batch * frames * ((height // patch_h) * (width // patch_w) + density)
    )
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=tuple(video.shape),
        action_shape=(batch, action_dim, frames, density, 1),
        padded_length=(-token_count) % 128,
        chunk_size=chunk,
        window_size=geometry.attention_window_size,
        patch_size=transformer.patch_size,
        text_token_count=text_emb.shape[1],
        device=device,
        current_block_coupling=coupling,
        chunk_origin_frame=chunk_origin,
        action_context_mask=action_mask,
        history_stream_visibility=geometry.history_stream_visibility,
        conditional_history_policy=geometry.conditional_history_policy,
        build_dense_masks=True,
        build_flex_masks=False,
    )

    def predict(
        step: VideoActionFlowInput,
        cache: DenoisingCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def pack(value: torch.Tensor) -> torch.Tensor:
            return rearrange(value * channel_mask, "b (f n) c -> b c f n 1", n=density)

        payload = {
            "latent_dict": {
                "noisy_latents": step.noisy_video,
                "latent": step.clean_video,
                "text_emb": step.text_context,
                "grid_id": video_grid,
                "timesteps": step.video_timesteps,
                "cond_timesteps": torch.zeros_like(step.video_timesteps),
            },
            "action_dict": {
                "noisy_latents": pack(step.noisy_action),
                "latent": pack(step.clean_action),
                "text_emb": step.text_context,
                "grid_id": action_grid,
                "timesteps": step.action_timesteps[:, ::density],
                "cond_timesteps": torch.zeros_like(step.action_timesteps[:, ::density]),
                "actions_mask": pack(action_mask.expand_as(action)),
            },
            "chunk_size": chunk,
            "window_size": geometry.attention_window_size,
            "chunk_origin_frame": chunk_origin,
            "attention_profile_name": attention_profile_name_for_current_block_coupling(
                coupling
            ),
            "history_stream_visibility": geometry.history_stream_visibility,
            "conditional_history_policy": geometry.conditional_history_policy,
            "per_chunk_proprio_state": proprio_frames,
            "per_chunk_proprio_state_granularity": ProprioContextGranularity.FRAME,
            "per_chunk_proprio_apply_to_video": not uses_legacy_prefix_per_chunk_proprio_contract(
                policy_config
            ),
        }
        output = transformer.execute_runtime_step(
            RuntimeStepInput(
                program=build_chunked_dual_stream_exact_train_program(
                    attention_profile_name=attention_profile_name_for_current_block_coupling(
                        coupling
                    ),
                ),
                payload=payload,
                denoising_cache=cache,
                required_tokens=step.required_tokens,
                attention_profile=step.attention,
            )
        )
        video_flow = output.projected_outputs["video_prediction"]
        action_flow = output.projected_outputs["action_prediction"] * channel_mask
        return unpatchify_video_sequence(
            transformer.patch_size, video_flow, frames, height, width, batch_size=batch
        ), action_flow

    predicted_video, predicted_action = denoise_video_action(
        VideoActionFlowInput(
            noisy_video=video,
            clean_video=clean_video,
            noisy_action=action,
            clean_action=clean_action,
            video_timesteps=video.new_zeros(batch, frames),
            action_timesteps=action.new_zeros(batch, frames * density),
            attention=profile,
            text_context=text_emb,
            frame_start=frame_start,
            proprio_frames=proprio_frames,
        ),
        predict=predict,
        training=training_config,
        inference=inference_config,
        coupling=coupling,
        timestep_coupling=resolve_parallel_joint_timestep_coupling(policy_config),
        dynamics=plan,
        video_start=history_frames,
        action_start=history_frames * density,
        negative_text_context=negative_text_emb,
        requested=output_request.modalities,
        supplied=frozenset({PolicyOutputModality.VIDEO})
        if context.video_conditioned_action is not None
        else frozenset(),
        initial_video_noise=context.initial_video_noise,
        initial_action_noise=context.initial_action_noise,
    )
    predicted_action = predicted_action * channel_mask
    returns_video = (
        PolicyOutputModality.VIDEO in output_request.modalities
        or plan.clean_video is not None
        or context.video_conditioned_action is not None
    )
    if sequence.commit_action is not None:
        sequence = replace(
            sequence, commit_action=sequence.commit_action * channel_mask
        )
    next_state = sequence.publish(
        replace(infer_state, variant_state=replace(history, text_context=raw_text)),
        video=predicted_video,
        action=predicted_action,
        retain_future_video=returns_video,
        retain_future_action=PolicyOutputModality.ACTION in output_request.modalities,
    )
    return ParallelInferArtifacts(
        action_pred=predicted_action[:, -chunk * density :],
        predicted_latents=predicted_video[:, :, -chunk:]
        if returns_video
        else predicted_video[:, :, :0],
        next_state=next_state,
        generation_frame_start=generation_start,
        debug={
            "generation_frame_start": generation_start,
            "current_block_coupling": coupling.value,
            "joint_timestep_coupling": resolve_parallel_joint_timestep_coupling(
                policy_config
            ).value,
            "action_conditioning_mode": plan.objective.value,
            "rollout_frame_chunk_size": chunk,
            "rollout_window_size": geometry.attention_window_size,
            "history_stream_visibility": geometry.history_stream_visibility.value,
            "use_cache": inference_config.use_cache,
        },
    )
