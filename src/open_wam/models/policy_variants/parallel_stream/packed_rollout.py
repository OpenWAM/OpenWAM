"""Packed joint and generalist-dynamics recurrent inference rollout."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelExactCacheWriteMode,
    ProprioContextMode,
)
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
)
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.visual_tower.exact_runtime import build_reference_mesh_id

from .cache_diagnostics import summarize_slot_pool_cache_state
from .cache_execution import write_exact_cache_chunk
from .cache_lifecycle import commit_initial_observed_video_context
from .conditional_rollout import (
    generalist_conditioning_chunk_size,
    generalist_conditioning_history_stream_visibility,
    generalist_conditioning_prefix_visibility_mode,
    generalist_conditioning_window_size,
    is_conditional_joint_denoise_mode,
    resolve_action_conditioning_mode,
    slice_conditioning_chunk,
    uses_generalist_mode_text_token,
)
from .exact_cache import (
    build_exact_cache_spec,
    ensure_exact_cache_initialized,
    resolve_exact_cache_context,
    validate_existing_exact_cache_attention_window,
)
from .forward_execution import run_parallel_action_conditioned_forward
from .inference_artifacts import ParallelInferArtifacts
from .inference_conditioning import append_generalist_mode_text_context
from .proprio_conditioning import (
    build_single_stream_hidden_proprio_context,
    inject_deprecated_proprio_text_context,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling,
    resolve_parallel_current_block_coupling,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract,
)


def _run_parallel_packed_inference_rollout_impl(
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
    forced_action_latents: torch.Tensor | None = None,
    commit_action_latents: torch.Tensor | None = None,
    forced_action_noise: torch.Tensor | None = None,
    action_conditioning_mode: str = "vanilla_joint_rollout",
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> ParallelInferArtifacts:
    current_block_coupling = resolve_parallel_current_block_coupling(policy_config)
    joint_packed_couplings = {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }
    if current_block_coupling not in joint_packed_couplings:
        raise ValueError(
            "`run_parallel_action_conditioned_inference_rollout` only implements packed noisy same-step coupling; "
            f"got {current_block_coupling.value!r}."
        )
    if policy_config.current_block_coupling is None and not policy_config.video_condition_on_action:
        raise ValueError(
            "`lingbot_exact_action_conditioned` requires `video_condition_on_action = true`."
        )
    if condition_latents is not None:
        device = condition_latents.device
        batch_size = condition_latents.shape[0]
        latent_height = condition_latents.shape[-2]
        latent_width = condition_latents.shape[-1]
    else:
        if "batch_size" not in infer_cache or "latent_height" not in infer_cache or "latent_width" not in infer_cache:
            raise ValueError(
                "Joint exact inference without current condition latents requires cached batch/latent shape metadata."
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
    rollout_mode = resolve_action_conditioning_mode(action_conditioning_mode)
    rollout_window_size = generalist_conditioning_window_size(
        rollout_mode,
        fallback_window_size=int(policy_config.attn_window),
    )
    rollout_frame_chunk_size = generalist_conditioning_chunk_size(
        rollout_mode,
        fallback_chunk_size=int(inference_config.frame_chunk_size),
    )
    rollout_history_stream_visibility = generalist_conditioning_history_stream_visibility(
        rollout_mode,
        policy_config,
    )
    condition_latents = slice_conditioning_chunk(
        condition_latents,
        target_frames=rollout_frame_chunk_size,
        source="condition latents",
    )
    generalist_mode = None
    if uses_generalist_mode_text_token(policy_config):
        generalist_mode = rollout_mode
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
    current_frame_start = int(infer_cache.get("frame_start", 0))
    generation_frame_start = current_frame_start
    cache_name = cache_context.cache_name
    cache_backend_name = cache_context.cache_backend_name
    cache_spec = build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=generalist_conditioning_prefix_visibility_mode(rollout_mode, policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError(
                "Joint exact inference requires condition latents on the first chunk when cache is empty."
            )
        cache_context = ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
            attn_window=rollout_window_size,
            frame_chunk_size=rollout_frame_chunk_size,
        )
    elif inference_config.use_cache and cache_context.cache_initialized:
        validate_existing_exact_cache_attention_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=rollout_window_size,
        )
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
            window_size=rollout_window_size,
            frame_chunk_size=rollout_frame_chunk_size,
            history_stream_visibility=rollout_history_stream_visibility,
            hidden_proprio_state=hidden_proprio_state,
        )
    latents = torch.randn(
        batch_size,
        backbone_config.latent_channels,
        rollout_frame_chunk_size,
        latent_height,
        latent_width,
        device=device,
        dtype=model_dtype,
    )
    actions = torch.randn(
        batch_size,
        action_dim,
        rollout_frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )
    action_denoise_mask = None
    if action_channel_mask is not None:
        action_denoise_mask = action_channel_mask.to(device=device, dtype=model_dtype)
        actions = actions * action_denoise_mask
    if forced_action_latents is not None:
        forced_action_latents = slice_conditioning_chunk(
            forced_action_latents,
            target_frames=rollout_frame_chunk_size,
            source="forced action latents",
        )
        forced_action_latents = forced_action_latents.to(device=device, dtype=model_dtype)
        if tuple(forced_action_latents.shape) != tuple(actions.shape):
            raise ValueError(
                "Forced joint-denoise action latents must match the generated action chunk shape, "
                f"got forced={tuple(forced_action_latents.shape)} and expected={tuple(actions.shape)}."
            )
        if action_denoise_mask is not None:
            forced_action_latents = forced_action_latents * action_denoise_mask
        if forced_action_noise is None:
            forced_action_noise = torch.randn_like(forced_action_latents)
        else:
            forced_action_noise = slice_conditioning_chunk(
                forced_action_noise,
                target_frames=rollout_frame_chunk_size,
                source="forced action noise",
            )
            forced_action_noise = forced_action_noise.to(device=device, dtype=model_dtype)
            if tuple(forced_action_noise.shape) != tuple(actions.shape):
                raise ValueError(
                    "Forced joint-denoise action noise must match the generated action chunk shape, "
                    f"got noise={tuple(forced_action_noise.shape)} and expected={tuple(actions.shape)}."
                )
        if action_denoise_mask is not None:
            forced_action_noise = forced_action_noise * action_denoise_mask
    if commit_action_latents is not None:
        commit_action_latents = slice_conditioning_chunk(
            commit_action_latents,
            target_frames=rollout_frame_chunk_size,
            source="committed action latents",
        )
        commit_action_latents = commit_action_latents.to(device=device, dtype=model_dtype)
        if tuple(commit_action_latents.shape) != tuple(actions.shape):
            raise ValueError(
                "Committed joint-denoise action latents must match the generated action chunk shape, "
                f"got commit={tuple(commit_action_latents.shape)} and expected={tuple(actions.shape)}."
            )
        if action_denoise_mask is not None:
            commit_action_latents = commit_action_latents * action_denoise_mask
    forced_video_latents = None
    if rollout_mode == JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION:
        if condition_latents is None:
            raise ValueError("video_conditioned_action rollout requires current video condition latents.")
        forced_video_latents = condition_latents.to(device=device, dtype=model_dtype)
        if tuple(forced_video_latents.shape) != tuple(latents.shape):
            raise ValueError(
                "Video-conditioned action rollout requires condition latents matching the generated chunk shape, "
                f"got condition={tuple(forced_video_latents.shape)} and expected={tuple(latents.shape)}."
            )
    initial_observed_video_anchor = None
    if (
        not initial_observed_context_committed
        and infer_cache.get("step_index", 0) == 0
        and condition_latents is not None
        and generation_frame_start == 0
        and rollout_mode != JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
    ):
        initial_observed_video_anchor = condition_latents[:, :, 0:1].to(device=device, dtype=model_dtype)
    # Keep the packed four-branch sequence contract for compatibility with the
    # trained backbone, but do not provide any explicit clean conditioning
    # signal at inference time. History should come only from the runtime
    # cache; the clean branches are zero placeholders.
    condition_video_latents = forced_video_latents if forced_video_latents is not None else torch.zeros_like(latents)
    condition_action_latents = torch.zeros(
        batch_size,
        action_dim,
        rollout_frame_chunk_size,
        policy_config.action_per_frame,
        1,
        device=device,
        dtype=model_dtype,
    )
    forced_clean_action_conditioning = (
        forced_action_latents is not None
        and rollout_mode == JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    if forced_clean_action_conditioning:
        condition_action_latents = forced_action_latents
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
    if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
        raise ValueError(
            "Joint LingBot denoising expects matched video/action inference step counts; "
            "set `video_num_inference_steps == action_num_inference_steps` for this mode."
        )

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    action_timestep_lookup_scheduler: FlowMatchScheduler | None = None
    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
        action_timestep_lookup_scheduler = FlowMatchScheduler(
            shift=training_config.action_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=training_config.action_num_train_timesteps,
        )
        action_timestep_lookup_scheduler.set_timesteps(training_config.action_num_train_timesteps)
        action_timestep_lookup_scheduler.sigmas = action_timestep_lookup_scheduler.sigmas.to(device=device)
        action_timestep_lookup_scheduler.timesteps = action_timestep_lookup_scheduler.timesteps.to(device=device)
    attention_profile_name = None
    if str(policy_config.video_action_attention_scope) == "block_local":
        if resolve_stage_attention_mode(backbone_config, stage="train", exact_runtime=True) == "flex":
            attention_profile_name = attention_profile_name_for_current_block_coupling(current_block_coupling)

    video_timestep_values_list = list(video_scheduler.timesteps.to(device=device, dtype=torch.float32))
    action_timestep_values_list = list(action_scheduler.timesteps.to(device=device, dtype=torch.float32))
    video_sigma_values_list = list(video_scheduler.sigmas.to(device=device, dtype=torch.float32))
    for index, (video_timestep, action_timestep) in enumerate(
        zip(video_timestep_values_list, action_timestep_values_list)
    ):
        video_timestep_values = video_timestep.expand(batch_size, rollout_frame_chunk_size)
        if forced_video_latents is not None:
            latents = forced_video_latents.clone()
            video_timestep_values = torch.zeros_like(video_timestep_values)
        if initial_observed_video_anchor is not None:
            latents[:, :, 0:1] = initial_observed_video_anchor
            video_timestep_values = video_timestep_values.clone()
            video_timestep_values[:, 0] = 0.0
        if joint_timestep_coupling in {
            JointTimestepCoupling.MATCH_SIGMA,
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
        }:
            shared_sigma = video_sigma_values_list[index]
            shared_sigma_next = video_scheduler.next_sigma(index).to(device=device, dtype=torch.float32)
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                    raise RuntimeError("Coupled joint denoise requires an action timestep lookup scheduler.")
                action_timestep = action_timestep_lookup_scheduler.timestep_matching_sigma(shared_sigma).to(
                    device=device,
                    dtype=torch.float32,
                )
            else:
                action_timestep = video_timestep.to(device=device, dtype=torch.float32)
            action_timestep_values = action_timestep.expand(batch_size, rollout_frame_chunk_size)
        else:
            shared_sigma = None
            shared_sigma_next = None
            action_timestep_values = action_timestep.expand(batch_size, rollout_frame_chunk_size)
        if forced_action_latents is not None:
            if forced_clean_action_conditioning:
                actions = forced_action_latents
                action_timestep_values = torch.zeros_like(action_timestep_values)
            elif joint_timestep_coupling in {
                JointTimestepCoupling.MATCH_SIGMA,
                JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
            }:
                sigma = shared_sigma.to(device=device, dtype=model_dtype).view(1, 1, 1, 1, 1)
                actions = (1 - sigma) * forced_action_latents + sigma * forced_action_noise
            else:
                actions = action_scheduler.add_noise(
                    forced_action_latents,
                    forced_action_noise,
                    action_timestep,
                    t_dim=2,
                )
            if action_denoise_mask is not None:
                actions = actions * action_denoise_mask
        action_mask_latents = (
            action_denoise_mask.expand_as(actions)
            if action_denoise_mask is not None
            else torch.ones_like(actions)
        )
        latent_grid_id = build_reference_mesh_id(
            rollout_frame_chunk_size // backbone_config.patch_size_t,
            latent_height // backbone_config.patch_size_h,
            latent_width // backbone_config.patch_size_w,
            t=0,
            f_w=1,
            f_shift=generation_frame_start,
            action=False,
            device=device,
        )[None].repeat(batch_size, 1, 1)
        action_grid_id = build_reference_mesh_id(
            rollout_frame_chunk_size,
            policy_config.action_per_frame,
            1,
            t=1,
            f_w=1,
            f_shift=generation_frame_start,
            action=True,
            device=device,
        )[None].repeat(batch_size, 1, 1)
        input_dict = {
            "latent_dict": {
                "noisy_latents": latents,
                "latent": condition_video_latents,
                "text_emb": text_emb,
                "grid_id": latent_grid_id,
                "timesteps": video_timestep_values,
                "cond_timesteps": torch.zeros_like(video_timestep_values),
            },
            "action_dict": {
                "noisy_latents": actions,
                "latent": condition_action_latents,
                "text_emb": text_emb,
                "grid_id": action_grid_id,
                "timesteps": action_timestep_values,
                "cond_timesteps": torch.zeros_like(action_timestep_values),
                "actions_mask": action_mask_latents,
            },
            "chunk_size": rollout_frame_chunk_size,
            "window_size": rollout_window_size,
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            "history_stream_visibility": rollout_history_stream_visibility,
        }
        if hidden_proprio_state is not None:
            input_dict["per_chunk_proprio_state"] = hidden_proprio_state[:, None, :].to(
                device=device,
                dtype=model_dtype,
            )
            if uses_legacy_prefix_per_chunk_proprio_contract(policy_config):
                input_dict["per_chunk_proprio_apply_to_video"] = False
        video_noise_pred, action_noise_pred = run_parallel_action_conditioned_forward(
            transformer,
            input_dict=input_dict,
            video_guidance_scale=float(inference_config.guidance_scale),
            action_guidance_scale=float(inference_config.action_guidance_scale),
            negative_text_emb=negative_text_emb,
            update_cache=0,
            cache_name=cache_name,
        )
        video_noise_pred = unpatchify_video_sequence(
            transformer.patch_size,
            video_noise_pred,
            rollout_frame_chunk_size,
            latent_height,
            latent_width,
            batch_size=batch_size,
        )
        if joint_timestep_coupling in {
            JointTimestepCoupling.MATCH_SIGMA,
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
        }:
            latents = video_scheduler.step_with_sigmas(
                video_noise_pred,
                sigma=shared_sigma,
                sigma_next=shared_sigma_next,
                sample=latents,
            )
        else:
            latents = video_scheduler.step(video_noise_pred, video_timestep, latents)
        if forced_video_latents is not None:
            latents = forced_video_latents.clone()
        if initial_observed_video_anchor is not None:
            latents[:, :, 0:1] = initial_observed_video_anchor
        action_noise_pred = rearrange(
            action_noise_pred,
            "b (f n) c -> b c f n 1",
            f=rollout_frame_chunk_size,
        )
        if forced_action_latents is None:
            if joint_timestep_coupling in {
                JointTimestepCoupling.MATCH_SIGMA,
                JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
            }:
                actions = action_scheduler.step_with_sigmas(
                    action_noise_pred,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                    sample=actions,
                )
            else:
                actions = action_scheduler.step(action_noise_pred, action_timestep, actions)
            if action_denoise_mask is not None:
                actions = actions * action_denoise_mask

    returned_action_latents = forced_action_latents if forced_action_latents is not None else actions
    cache_action_latents = (
        commit_action_latents
        if commit_action_latents is not None
        else (forced_action_latents if forced_action_latents is not None else actions)
    )
    video_hidden_context = build_single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=latents,
        action_mode=False,
    )
    action_hidden_context = build_single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=cache_action_latents,
        action_mode=True,
    )

    if inference_config.use_cache:
        write_exact_cache_chunk(
            transformer=transformer,
            cache_spec=cache_spec,
            cache_name=cache_name,
            frame_start=generation_frame_start,
            backbone_config=backbone_config,
            video_latents=latents,
            action_latents=cache_action_latents,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            use_cfg=cache_context.use_cfg,
            action_channel_mask=action_channel_mask,
            update_cache=1,
            chunk_size=rollout_frame_chunk_size,
            window_size=rollout_window_size,
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            history_stream_visibility=rollout_history_stream_visibility,
            video_hidden_context=video_hidden_context,
            action_hidden_context=action_hidden_context,
            allow_cache_prefix_during_update_write=is_conditional_joint_denoise_mode(rollout_mode),
        )

    next_cache = {
        "runtime_mode": "lingbot_exact_action_conditioned",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": int(
            generation_frame_start + rollout_frame_chunk_size if advance_frame_start else generation_frame_start
        ),
        "latent_height": latent_height,
        "latent_width": latent_width,
        "batch_size": batch_size,
        "step_index": int(infer_cache.get("step_index", 0) + 1),
        "use_cfg": cache_context.use_cfg,
        "frame_chunk_size": int(rollout_frame_chunk_size),
    }
    debug = {
        "runtime_mode": "lingbot_exact_action_conditioned",
        "cache_name": cache_name,
        "cache_backend_name": cache_backend_name,
        "generation_frame_start": generation_frame_start,
        "advance_frame_start": advance_frame_start,
        "video_condition_on_action": bool(policy_config.video_condition_on_action),
        "video_action_condition_source": str(policy_config.video_action_condition_source),
        "video_action_attention_scope": str(policy_config.video_action_attention_scope),
        "current_block_coupling": current_block_coupling.value,
        "joint_timestep_coupling": joint_timestep_coupling.value,
        "couple_action_to_video_timesteps": bool(
            joint_timestep_coupling
            in {JointTimestepCoupling.MATCH_SIGMA, JointTimestepCoupling.SHARED_VIDEO_SCHEDULE}
        ),
        "joint_denoise": True,
        "uses_explicit_clean_condition": False,
        "use_cache": bool(inference_config.use_cache),
        "cache_commit_mode": str(cache_spec.write_mode),
        "use_cfg": cache_context.use_cfg,
        "initial_observed_context_committed": bool(initial_observed_context_committed),
        "video_num_inference_steps": int(inference_config.video_num_inference_steps),
        "action_num_inference_steps": int(inference_config.action_num_inference_steps),
        "action_conditioning_mode": action_conditioning_mode,
        "generalist_mode_text_token": None if generalist_mode is None else generalist_mode.value,
        "generalist_mode_text_token_count": int(generalist_mode is not None),
        "initial_observed_video_anchor": initial_observed_video_anchor is not None,
        "forced_action_denoise": forced_action_latents is not None,
        "forced_clean_action_conditioning": bool(forced_clean_action_conditioning),
        "forced_video_conditioning": forced_video_latents is not None,
        "commit_action_override": commit_action_latents is not None,
        "returned_action_source": "forced" if forced_action_latents is not None else "predicted",
        "cache_action_source": (
            "commit_override"
            if commit_action_latents is not None
            else ("forced" if forced_action_latents is not None else "predicted")
        ),
        "rollout_window_size": int(rollout_window_size),
        "rollout_frame_chunk_size": int(rollout_frame_chunk_size),
        "history_stream_visibility": rollout_history_stream_visibility.value,
        "generalist_conditional_history_chunks": 1 if is_conditional_joint_denoise_mode(rollout_mode) else 0,
    }
    cache_summary = summarize_slot_pool_cache_state(transformer, cache_name)
    if cache_summary is not None:
        debug.update(cache_summary)
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(returned_action_latents, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return ParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def run_parallel_packed_inference_rollout(
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
    action_conditioning_mode: JointDenoiseTrainingMode | str = "vanilla_joint_rollout",
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> ParallelInferArtifacts:
    return _run_parallel_packed_inference_rollout_impl(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=action_dim,
        condition_latents=condition_latents,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=action_channel_mask,
        infer_cache=infer_cache,
        advance_frame_start=advance_frame_start,
        action_conditioning_mode=action_conditioning_mode,
        proprio_state=proprio_state,
        hidden_proprio_state=hidden_proprio_state,
    )


def run_parallel_packed_action_override_rollout(
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
    advance_frame_start: bool,
    forced_action_latents: torch.Tensor | None = None,
    commit_action_latents: torch.Tensor | None = None,
    forced_action_noise: torch.Tensor | None = None,
    action_conditioning_mode: str = "forced_action_joint_fdm",
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> ParallelInferArtifacts:
    """Run packed joint denoising with caller-owned action overrides.

    For action-conditioned-video modes, `forced_action_latents` is exposed as a
    clean current action condition. `commit_action_latents` only changes the
    clean action tokens committed into history after the chunk is generated.
    """

    resolved_proprio_state = proprio_state
    resolved_hidden_proprio_state = hidden_proprio_state
    if ProprioContextMode(policy_config.proprio_context_mode) == ProprioContextMode.PER_CHUNK_ADDITIVE:
        if resolved_hidden_proprio_state is None:
            resolved_hidden_proprio_state = proprio_state
        if isinstance(resolved_hidden_proprio_state, torch.Tensor) and resolved_hidden_proprio_state.ndim == 3:
            resolved_hidden_proprio_state = resolved_hidden_proprio_state[:, -1, :]
        resolved_proprio_state = None

    return _run_parallel_packed_inference_rollout_impl(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=action_dim,
        condition_latents=condition_latents,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=action_channel_mask,
        infer_cache=infer_cache,
        advance_frame_start=advance_frame_start,
        forced_action_latents=forced_action_latents,
        commit_action_latents=commit_action_latents,
        forced_action_noise=forced_action_noise,
        action_conditioning_mode=action_conditioning_mode,
        proprio_state=resolved_proprio_state,
        hidden_proprio_state=resolved_hidden_proprio_state,
    )


# Compatibility for callers that adopted the research-era function name.
run_parallel_action_conditioned_action_override_inference_rollout = (
    run_parallel_packed_action_override_rollout
)
