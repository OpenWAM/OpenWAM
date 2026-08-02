"""Action-only recurrent rollouts anchored by observed video latents."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
)
from open_wam.models.visual_tower.exact_runtime import (
    build_reference_mesh_id as get_mesh_id,
    clear_exact_prediction_cache as _clear_exact_prediction_cache,
)

from .exact_cache import resolve_exact_cache_context as _resolve_exact_cache_context
from .forward_execution import (
    run_parallel_action_conditioned_forward as _run_parallel_action_conditioned_forward,
    run_parallel_first_frame_conditioned_forward as _run_parallel_fastwam_first_frame_forward_manual,
)
from .inference_artifacts import LingbotParallelInferArtifacts
from .latent_conditioning import (
    build_repeated_first_frame_condition as _build_clean_video_condition_from_anchor,
)
from .proprio_conditioning import (
    inject_deprecated_proprio_text_context as _inject_proprio_text_context,
)


def run_parallel_current_frame_action_chunk_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool = True,
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> LingbotParallelInferArtifacts:
    if condition_latents is None:
        raise ValueError("Current-frame action-chunk inference requires current condition latents every chunk.")
    if float(inference_config.guidance_scale) != 1.0:
        raise ValueError(
            "current_frame_action_chunk does not support video CFG; "
            f"set inference.guidance_scale=1.0, got {inference_config.guidance_scale}."
        )
    device = condition_latents.device
    batch_size = int(condition_latents.shape[0])
    latent_height = int(condition_latents.shape[-2])
    latent_width = int(condition_latents.shape[-1])
    cache_context, text_emb, negative_text_emb = _resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=inference_config,
        infer_cache=infer_cache,
        batch_size=batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
    )
    _clear_exact_prediction_cache(transformer, cache_name=cache_context.cache_name)
    text_emb, negative_text_emb = _inject_proprio_text_context(
        transformer,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=proprio_state,
    )
    model_dtype = cache_context.model_dtype
    frame_chunk_size = int(inference_config.frame_chunk_size)
    generation_frame_start = int(infer_cache.get("frame_start", 0))
    condition_video_latents = _build_clean_video_condition_from_anchor(
        condition_latents.to(device=device, dtype=model_dtype),
        target_frames=frame_chunk_size,
    )
    action_condition_latents = torch.zeros(
        batch_size,
        action_dim,
        frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )
    actions = torch.randn_like(action_condition_latents)
    action_denoise_mask = None
    if action_channel_mask is not None:
        action_denoise_mask = action_channel_mask.to(device=device, dtype=model_dtype)
        actions = actions * action_denoise_mask

    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(inference_config.action_num_inference_steps)
    latent_grid_id = get_mesh_id(
        frame_chunk_size // backbone_config.patch_size_t,
        latent_height // backbone_config.patch_size_h,
        latent_width // backbone_config.patch_size_w,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
        device=device,
    )[None].repeat(batch_size, 1, 1)
    action_grid_id = get_mesh_id(
        frame_chunk_size,
        policy_config.action_per_frame,
        1,
        t=1,
        f_w=1,
        f_shift=0,
        action=True,
        device=device,
    )[None].repeat(batch_size, 1, 1)
    zero_video_timesteps = torch.zeros(
        batch_size,
        frame_chunk_size,
        device=device,
        dtype=torch.float32,
    )
    for timestep in action_scheduler.timesteps.to(device=device, dtype=torch.float32):
        action_timestep_values = timestep.expand(batch_size, frame_chunk_size)
        action_mask_latents = (
            action_denoise_mask.expand_as(actions)
            if action_denoise_mask is not None
            else torch.ones_like(actions)
        )
        input_dict = {
            "latent_dict": {
                "noisy_latents": condition_video_latents,
                "latent": condition_video_latents,
                "text_emb": text_emb,
                "grid_id": latent_grid_id,
                "timesteps": zero_video_timesteps,
                "cond_timesteps": zero_video_timesteps,
            },
            "action_dict": {
                "noisy_latents": actions,
                "latent": action_condition_latents,
                "text_emb": text_emb,
                "grid_id": action_grid_id,
                "timesteps": action_timestep_values,
                "cond_timesteps": torch.zeros_like(action_timestep_values),
                "actions_mask": action_mask_latents,
            },
            "chunk_size": frame_chunk_size,
            "window_size": frame_chunk_size,
            "attention_profile_name": "none",
            "preserve_video_pretrain_history": False,
            "current_frame_action_chunk": True,
        }
        if hidden_proprio_state is not None:
            input_dict["per_chunk_proprio_state"] = hidden_proprio_state[:, None, :].to(
                device=device,
                dtype=model_dtype,
            )
            input_dict["per_chunk_proprio_state_granularity"] = "chunk"
        _, action_noise_pred = _run_parallel_action_conditioned_forward(
            transformer,
            input_dict=input_dict,
            video_guidance_scale=1.0,
            action_guidance_scale=float(inference_config.action_guidance_scale),
            negative_text_emb=negative_text_emb,
            update_cache=0,
            cache_name=cache_context.cache_name,
        )
        action_noise_pred = rearrange(
            action_noise_pred,
            "b (f n) c -> b c f n 1",
            f=frame_chunk_size,
            n=policy_config.action_per_frame,
        )
        actions = action_scheduler.step(action_noise_pred, timestep, actions)
        if action_denoise_mask is not None:
            actions = actions * action_denoise_mask

    output_dtype = condition_latents.dtype
    next_frame_start = generation_frame_start + frame_chunk_size if advance_frame_start else generation_frame_start
    action_pred = rearrange(actions, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    empty_latents = condition_latents.new_empty(
        batch_size,
        backbone_config.latent_channels,
        0,
        latent_height,
        latent_width,
    )
    next_cache = {
        "runtime_mode": "current_frame_action_chunk",
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": False,
        "frame_start": int(next_frame_start),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
    }
    debug = {
        "runtime_mode": "current_frame_action_chunk",
        "uses_current_frame_condition": True,
        "uses_exact_history_cache": False,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": bool(advance_frame_start),
        "action_timesteps": action_scheduler.timesteps.tolist(),
        "action_guidance_scale": float(inference_config.action_guidance_scale),
        "condition_latents_shape": list(condition_latents.shape),
        "action_shape": list(action_pred.shape),
    }
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=empty_latents,
        next_cache=next_cache,
        debug=debug,
    )


def run_parallel_fastwam_first_frame_inference_rollout(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    action_dim: int,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    advance_frame_start: bool = True,
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> LingbotParallelInferArtifacts:
    if condition_latents is None:
        raise ValueError("FastWAM first-frame inference requires current condition latents every chunk.")
    if float(inference_config.guidance_scale) != 1.0:
        raise ValueError(
            "fastwam_first_frame inference does not denoise video; "
            f"set inference.guidance_scale=1.0, got {inference_config.guidance_scale}."
        )
    if float(inference_config.action_guidance_scale) != 1.0:
        raise ValueError(
            "fastwam_first_frame currently runs action CFG disabled; "
            f"set inference.action_guidance_scale=1.0, got {inference_config.action_guidance_scale}."
        )
    del negative_text_emb

    device = condition_latents.device
    batch_size = int(condition_latents.shape[0])
    latent_height = int(condition_latents.shape[-2])
    latent_width = int(condition_latents.shape[-1])
    cache_context, text_emb, _ = _resolve_exact_cache_context(
        transformer=transformer,
        backbone_config=backbone_config,
        inference_config=inference_config,
        infer_cache=infer_cache,
        batch_size=batch_size,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
        text_emb=text_emb,
        negative_text_emb=None,
    )
    _clear_exact_prediction_cache(transformer, cache_name=cache_context.cache_name)
    text_emb, _ = _inject_proprio_text_context(
        transformer,
        text_emb=text_emb,
        negative_text_emb=None,
        proprio_state=proprio_state,
    )

    model_dtype = cache_context.model_dtype
    action_frames = int(inference_config.frame_chunk_size)
    action_per_frame = int(policy_config.action_per_frame)
    generation_frame_start = int(infer_cache.get("frame_start", 0))
    first_frame_latents = condition_latents[:, :, :1].to(device=device, dtype=model_dtype)
    actions = torch.randn(
        batch_size,
        action_dim,
        action_frames,
        action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )
    action_denoise_mask = None
    if action_channel_mask is not None:
        action_denoise_mask = action_channel_mask.to(device=device, dtype=model_dtype)
        actions = actions * action_denoise_mask

    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(inference_config.action_num_inference_steps)
    latent_grid_id = get_mesh_id(
        1,
        latent_height // backbone_config.patch_size_h,
        latent_width // backbone_config.patch_size_w,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
        device=device,
    )[None].repeat(batch_size, 1, 1)
    action_grid_id = get_mesh_id(
        action_frames,
        action_per_frame,
        1,
        t=1,
        f_w=1,
        f_shift=0,
        action=True,
        device=device,
    )[None].repeat(batch_size, 1, 1)
    zero_video_timesteps = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

    for timestep in action_scheduler.timesteps.to(device=device, dtype=torch.float32):
        action_timestep_values = timestep.expand(batch_size, action_frames)
        action_mask_latents = (
            action_denoise_mask.expand_as(actions)
            if action_denoise_mask is not None
            else torch.ones_like(actions)
        )
        input_dict = {
            "latent_dict": {
                "noisy_latents": first_frame_latents,
                "latent": torch.zeros_like(first_frame_latents),
                "text_emb": text_emb,
                "grid_id": latent_grid_id,
                "timesteps": zero_video_timesteps,
                "cond_timesteps": zero_video_timesteps,
            },
            "action_dict": {
                "noisy_latents": actions,
                "latent": torch.zeros_like(actions),
                "text_emb": text_emb,
                "grid_id": action_grid_id,
                "timesteps": action_timestep_values,
                "cond_timesteps": torch.zeros_like(action_timestep_values),
                "actions_mask": action_mask_latents,
            },
            "chunk_size": action_frames,
            "window_size": action_frames,
            "attention_profile_name": "fastwam_first_frame",
            "fastwam_first_frame": True,
        }
        if hidden_proprio_state is not None:
            input_dict["per_chunk_proprio_state"] = hidden_proprio_state[:, None, :].to(
                device=device,
                dtype=model_dtype,
            )
            input_dict["per_chunk_proprio_state_granularity"] = "chunk"
        _, action_noise_pred = _run_parallel_fastwam_first_frame_forward_manual(transformer, input_dict)
        action_noise_pred = rearrange(
            action_noise_pred,
            "b (f n) c -> b c f n 1",
            f=action_frames,
            n=action_per_frame,
        )
        actions = action_scheduler.step(action_noise_pred, timestep, actions)
        if action_denoise_mask is not None:
            actions = actions * action_denoise_mask

    output_dtype = condition_latents.dtype
    next_frame_start = generation_frame_start + action_frames if advance_frame_start else generation_frame_start
    action_pred = rearrange(actions, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    empty_latents = condition_latents.new_empty(
        batch_size,
        backbone_config.latent_channels,
        0,
        latent_height,
        latent_width,
    )
    next_cache = {
        "runtime_mode": "fastwam_first_frame",
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": False,
        "frame_start": int(next_frame_start),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": False,
    }
    debug = {
        "runtime_mode": "fastwam_first_frame",
        "uses_first_frame_condition": True,
        "uses_exact_history_cache": False,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": bool(advance_frame_start),
        "action_timesteps": action_scheduler.timesteps.tolist(),
        "condition_latents_shape": list(condition_latents.shape),
        "first_frame_latents_shape": list(first_frame_latents.shape),
        "action_shape": list(action_pred.shape),
    }
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=empty_latents,
        next_cache=next_cache,
        debug=debug,
    )
