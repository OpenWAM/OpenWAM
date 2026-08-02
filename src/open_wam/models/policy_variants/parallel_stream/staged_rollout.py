"""Staged ordered and decoupled parallel-stream inference rollout."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    ParallelExactCacheWriteMode,
)
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common import (
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
)
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
)
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.visual_tower.exact_runtime import (
    prepare_exact_single_stream_input,
    run_exact_single_stream_forward,
)

from .cache_execution import write_exact_cache_chunk
from .cache_lifecycle import commit_initial_observed_video_context
from .conditional_rollout import uses_generalist_mode_text_token
from .exact_cache import (
    build_exact_cache_spec,
    count_single_stream_action_tokens,
    ensure_exact_cache_initialized,
    resolve_exact_cache_context,
    restore_slot_pool_layer_metadata,
    set_slot_pool_layer_metadata,
    validate_existing_exact_cache_attention_window,
)
from .inference_artifacts import ParallelInferArtifacts
from .inference_conditioning import append_generalist_mode_text_context
from .proprio_conditioning import (
    build_single_stream_hidden_proprio_context,
    inject_deprecated_proprio_text_context,
)
from .runtime_semantics import (
    prefix_visibility_mode_for_policy,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    uses_legacy_prefix_per_chunk_proprio_contract,
)


def run_parallel_staged_inference_rollout(
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
    advance_frame_start: bool = False,
    skip_video_prediction: bool = False,
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> ParallelInferArtifacts:
    if condition_latents is not None:
        device = condition_latents.device
        batch_size = condition_latents.shape[0]
        latent_height = condition_latents.shape[-2]
        latent_width = condition_latents.shape[-1]
    else:
        if "batch_size" not in infer_cache or "latent_height" not in infer_cache or "latent_width" not in infer_cache:
            raise ValueError(
                "Exact LingBot inference without condition latents requires cached batch/latent shape metadata."
            )
        device = next(transformer.parameters()).device
        batch_size = int(infer_cache["batch_size"])
        latent_height = int(infer_cache["latent_height"])
        latent_width = int(infer_cache["latent_width"])
    cache_context, text_emb, negative_text_emb = resolve_exact_cache_context(
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
    generalist_mode = None
    if uses_generalist_mode_text_token(policy_config):
        generalist_mode = JointDenoiseTrainingMode.JOINT
        text_emb, negative_text_emb = append_generalist_mode_text_context(
            transformer,
            policy_config=policy_config,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            mode=generalist_mode,
        )
    text_emb, negative_text_emb = inject_deprecated_proprio_text_context(
        transformer,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=proprio_state,
    )
    model_dtype = cache_context.model_dtype
    cache_name = cache_context.cache_name
    cache_backend_name = cache_context.cache_backend_name
    current_frame_start = int(infer_cache.get("frame_start", 0))
    current_block_coupling = resolve_parallel_current_block_coupling(policy_config)
    joint_packed_couplings = {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }
    if current_block_coupling in joint_packed_couplings:
        raise ValueError(
            "Joint-like M1 coupling must use `run_parallel_action_conditioned_inference_rollout`; "
            "the staged exact rollout only supports ordered or decoupled same-step coupling."
        )
    if skip_video_prediction and current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
        raise ValueError("`skip_video_prediction` is incompatible with `video_then_action` because action depends on video.")
    cache_spec = build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=prefix_visibility_mode_for_policy(policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError("Exact LingBot inference requires condition latents on the first chunk when cache is empty.")
        cache_context = ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
            attn_window=int(policy_config.attn_window),
        )
    elif inference_config.use_cache and cache_context.cache_initialized:
        validate_existing_exact_cache_attention_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=int(policy_config.attn_window),
        )
    generation_frame_start = current_frame_start
    initial_observed_context_committed = False
    if cache_context.cache_initialized:
        generation_frame_start, initial_observed_context_committed = commit_initial_observed_video_context(
            transformer=transformer,
            cache_spec=cache_spec,
            cache_name=cache_name,
            backbone_config=backbone_config,
            policy_config=policy_config,
            inference_config=inference_config,
            condition_latents=condition_latents,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            use_cfg=cache_context.use_cfg and inference_config.use_cache,
            action_channel_mask=action_channel_mask,
            action_dim=action_dim,
            model_dtype=model_dtype,
            current_frame_start=current_frame_start,
            step_index=int(infer_cache.get("step_index", 0)),
            current_block_coupling=current_block_coupling,
            window_size=int(policy_config.attn_window),
            hidden_proprio_state=hidden_proprio_state,
        )
    latent_cond = None
    if (
        not initial_observed_context_committed
        and infer_cache.get("step_index", 0) == 0
        and condition_latents is not None
        and current_frame_start == 0
    ):
        latent_cond = condition_latents[:, :, 0:1].to(dtype=model_dtype)

    latents = torch.randn(
        batch_size,
        backbone_config.latent_channels,
        inference_config.frame_chunk_size,
        latent_height,
        latent_width,
        device=device,
        dtype=model_dtype,
    )
    # One generated chunk always has aligned video/action frame count:
    # - `latents`: `[B, C_latent, F_chunk, H_latent, W_latent]`
    # - `actions`: `[B, D_action, F_chunk, action_per_frame, 1]`
    # Both streams share `F_chunk = inference_config.frame_chunk_size`.
    actions = torch.randn(
        batch_size,
        action_dim,
        inference_config.frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )

    video_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    video_scheduler.set_timesteps(inference_config.video_num_inference_steps)
    action_scheduler.set_timesteps(inference_config.action_num_inference_steps)
    video_timesteps = F.pad(video_scheduler.timesteps.to(device=device), (0, 1), mode="constant", value=0)
    if inference_config.video_exec_step != -1:
        video_timesteps = video_timesteps[: inference_config.video_exec_step]
    action_timesteps = F.pad(action_scheduler.timesteps.to(device=device), (0, 1), mode="constant", value=0)

    action_cond = None
    if generation_frame_start == 0:
        action_cond = torch.zeros(
            batch_size,
            actions.shape[1],
            1,
            policy_config.action_per_frame,
            1,
            device=device,
            dtype=model_dtype,
        )
    action_hidden_context = build_single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=actions,
        action_mode=True,
    )
    video_hidden_context = (
        None
        if uses_legacy_prefix_per_chunk_proprio_contract(policy_config)
        else build_single_stream_hidden_proprio_context(
            transformer,
            proprio_state=hidden_proprio_state,
            stream_latents=latents,
            action_mode=False,
        )
    )

    def denoise_video_chunk(*, commit_to_cache: bool) -> None:
        nonlocal latents
        for index, timestep in enumerate(video_timesteps):
            last_step = index == len(video_timesteps) - 1
            video_input = prepare_exact_single_stream_input(
                latents=latents,
                timestep=timestep,
                text_emb=text_emb,
                frame_st_id=generation_frame_start,
                backbone_config=backbone_config,
                action_mode=False,
                cond=latent_cond,
            )
            if video_hidden_context is not None:
                video_input["hidden_context"] = video_hidden_context
            video_noise_pred = run_exact_single_stream_forward(
                transformer,
                input_dict=video_input,
                update_cache=1 if (last_step and commit_to_cache and inference_config.use_cache) else 0,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=inference_config.guidance_scale,
                negative_text_emb=negative_text_emb,
                force_cfg_batch=cache_context.use_cfg and inference_config.use_cache,
            )
            if not last_step or inference_config.video_exec_step != -1:
                video_noise_pred = unpatchify_video_sequence(
                    transformer.patch_size,
                    video_noise_pred,
                    inference_config.frame_chunk_size,
                    latent_height,
                    latent_width,
                    batch_size=batch_size,
                )
                latents = video_scheduler.step(video_noise_pred, timestep, latents)
            if latent_cond is not None:
                latents[:, :, 0:1] = latent_cond

    def denoise_action_chunk(*, commit_to_cache: bool) -> None:
        nonlocal actions
        # Actions are denoised in their native `[B, D_action, F_chunk, A, 1]`
        # volume and converted back to `[B, F_chunk * A, D_action]` once the
        # chunk is complete.
        for index, timestep in enumerate(action_timesteps):
            last_step = index == len(action_timesteps) - 1
            action_input = prepare_exact_single_stream_input(
                latents=actions,
                timestep=timestep,
                text_emb=text_emb,
                frame_st_id=generation_frame_start,
                backbone_config=backbone_config,
                action_mode=True,
                cond=action_cond,
                action_channel_mask=action_channel_mask,
            )
            if action_hidden_context is not None:
                action_input["hidden_context"] = action_hidden_context
            action_noise_pred = run_exact_single_stream_forward(
                transformer,
                input_dict=action_input,
                update_cache=1 if (last_step and commit_to_cache and inference_config.use_cache) else 0,
                cache_name=cache_name,
                action_mode=True,
                guidance_scale=inference_config.action_guidance_scale,
                negative_text_emb=negative_text_emb,
                force_cfg_batch=cache_context.use_cfg and inference_config.use_cache,
            )
            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=inference_config.frame_chunk_size,
                )
                actions = action_scheduler.step(action_noise_pred, timestep, actions)
            if action_cond is not None:
                actions[:, :, 0:1] = action_cond

    if current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
        cache_commit_strategy = "video_then_action_staged"
        denoise_video_chunk(commit_to_cache=True)
        denoise_action_chunk(commit_to_cache=True)
    elif current_block_coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO:
        if skip_video_prediction:
            cache_commit_strategy = "action_then_video_action_only_no_predicted_cache"
            denoise_action_chunk(commit_to_cache=False)
            latents = latents[:, :, :0].contiguous()
        else:
            cache_commit_strategy = "action_then_video_staged"
            denoise_action_chunk(commit_to_cache=True)
            metadata_previous = set_slot_pool_layer_metadata(
                transformer,
                cache_name=cache_name,
                updates={
                    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS: count_single_stream_action_tokens(actions),
                },
            )
            try:
                denoise_video_chunk(commit_to_cache=True)
            finally:
                restore_slot_pool_layer_metadata(metadata_previous)
    elif current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        if skip_video_prediction:
            cache_commit_strategy = "decoupled_same_step_action_only_no_predicted_cache"
            denoise_action_chunk(commit_to_cache=False)
            latents = latents[:, :, :0].contiguous()
        else:
            cache_commit_strategy = "decoupled_same_step_deferred"
            denoise_video_chunk(commit_to_cache=False)
            denoise_action_chunk(commit_to_cache=False)
        if inference_config.use_cache and not skip_video_prediction:
            write_exact_cache_chunk(
                transformer=transformer,
                cache_spec=cache_spec,
                cache_name=cache_name,
                frame_start=generation_frame_start,
                backbone_config=backbone_config,
                video_latents=latents,
                action_latents=actions,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                use_cfg=cache_context.use_cfg,
                action_channel_mask=action_channel_mask,
                update_cache=1,
                chunk_size=inference_config.frame_chunk_size,
                window_size=policy_config.attn_window,
                current_block_coupling=current_block_coupling,
                preserve_video_pretrain_history=bool(
                    getattr(policy_config, "preserve_video_pretrain_history", False)
                ),
                history_stream_visibility=resolve_parallel_history_stream_visibility(policy_config),
                video_hidden_context=video_hidden_context,
                action_hidden_context=action_hidden_context,
            )
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported M1 current-block coupling: {current_block_coupling!r}")

    next_cache = {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": int(
            generation_frame_start + inference_config.frame_chunk_size if advance_frame_start else generation_frame_start
        ),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
    }
    debug = {
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "use_cfg": cache_context.use_cfg,
        "generation_frame_start": generation_frame_start,
        "initial_observed_context_committed": bool(initial_observed_context_committed),
        "advance_frame_start": advance_frame_start,
        "video_timesteps": video_timesteps.tolist(),
        "action_timesteps": action_timesteps.tolist(),
        "current_block_coupling": current_block_coupling.value,
        "cache_commit_strategy": cache_commit_strategy,
        "video_guidance_scale": float(inference_config.guidance_scale),
        "action_guidance_scale": float(inference_config.action_guidance_scale),
        "cache_write_mode": str(cache_spec.write_mode),
        "skip_video_prediction": bool(skip_video_prediction),
        "generalist_mode_text_token": None if generalist_mode is None else generalist_mode.value,
        "generalist_mode_text_token_count": int(generalist_mode is not None),
    }
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(actions, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return ParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )
