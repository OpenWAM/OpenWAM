from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelExactCacheWriteMode,
    ParallelHistoryStreamVisibility,
    ProprioContextMode,
)
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common import (
    AttentionProfileSpec,
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    build_chunked_temporal_exact_attention_profile,
    cache_backend_uses_slot_pool,
    materialize_cache_backend_entries,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler, sample_timestep_id
from open_wam.models.common.rollout_startup import resolve_strict_startup_plan
from open_wam.models.common.video_geometry import (
    unpatchify_video_sequence as data_seq_to_patch,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig, resolve_stage_attention_mode
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    build_chunked_dual_stream_exact_train_program,
)
from open_wam.models.visual_tower.exact_runtime import (
    build_reference_mesh_id as get_mesh_id,
    clear_exact_prediction_cache as _clear_exact_prediction_cache,
    initialize_exact_runtime_cache as initialize_reference_cache,
    prepare_exact_single_stream_forward_input as prepare_reference_forward_input,
    prepare_exact_single_stream_input as prepare_reference_single_stream_input,
    repeat_exact_single_stream_input_for_cfg as repeat_input_for_cfg,
    resolve_runtime_module_dtype as reference_runtime_dtype,
    run_exact_single_stream_forward as run_reference_single_stream_forward,
)
from open_wam.models.visual_tower.sequence_adapters import prepare_exact_dual_stream_train_sequence

from .cache_execution import (
    build_joint_clean_cache_attention_mask as _build_joint_clean_cache_attention_mask,
    build_joint_clean_cache_attention_profile as _build_joint_clean_cache_attention_profile,
    summarize_slot_pool_cache_state as _summarize_slot_pool_cache_state,
    write_exact_cache_chunk as _write_exact_cache_chunk,
    write_joint_clean_tokens_to_exact_cache as _write_joint_clean_tokens_to_exact_cache,
)
from .conditional_rollout import (
    generalist_conditioning_chunk_size as _chunk_size_for_generalist_conditioning,
    generalist_conditioning_history_stream_visibility as _history_stream_visibility_for_generalist_conditioning,
    generalist_conditioning_prefix_visibility_mode as _prefix_visibility_mode_for_generalist_conditioning,
    generalist_conditioning_window_size as _window_size_for_generalist_conditioning,
    is_conditional_joint_denoise_mode as _is_conditional_joint_denoise_mode,
    resolve_action_conditioning_mode as _generalist_mode_for_action_conditioning,
    select_conditional_warmup_history_suffix as _select_conditional_warmup_history_suffix,
    slice_conditioning_chunk as _slice_conditioning_chunk,
    uses_generalist_mode_text_token as _uses_generalist_mode_text_token,
)
from .exact_cache import (
    ExactCacheContext,
    ExactCacheInterfaceSpec,
    build_clean_video_action_cache_stream_ids as _stream_ids_for_clean_video_action_tokens,
    build_dual_stream_cache_stream_ids as _stream_ids_for_exact_dual_stream_split,
    build_exact_cache_spec as _build_exact_cache_spec,
    count_single_stream_action_tokens as _single_stream_action_token_count,
    ensure_exact_cache_initialized as _ensure_exact_cache_initialized,
    ensure_exact_text_embeddings as ensure_reference_text_embeddings,
    existing_exact_cache_attention_window as _existing_exact_cache_attn_window,
    restore_slot_pool_layer_metadata as _restore_slot_pool_layer_metadata,
    resolve_exact_cache_context as _resolve_exact_cache_context,
    set_slot_pool_layer_metadata as _set_slot_pool_layer_metadata,
    validate_existing_exact_cache_attention_window as _validate_existing_exact_cache_attn_window,
)
from .generalist_training import (
    apply_generalist_joint_denoise_training_mode as _apply_generalist_joint_denoise_training_mode,
    apply_generalist_legacy_prefix_joint_training_mode as _apply_generalist_legacy_prefix_joint_training_mode,
    sample_generalist_joint_denoise_training_mode as _sample_joint_denoise_training_mode,
)
from .latent_conditioning import (
    build_repeated_first_frame_condition as _build_clean_video_condition_from_anchor,
    resolve_full_window_condition_latents as _resolve_full_condition_latents,
    select_first_frame_condition_latents as _select_first_frame_condition_latents,
)
from .proprio_conditioning import (
    apply_parallel_chunk_proprio_context as _apply_parallel_chunk_proprio_context,
    build_single_stream_hidden_proprio_context as _single_stream_hidden_proprio_context,
    inject_deprecated_proprio_text_context as _inject_proprio_text_context,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
    prefix_visibility_mode_for_policy as _prefix_visibility_mode_for_policy,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract as _uses_legacy_prefix_per_chunk_proprio_contract,
)
from .training_artifacts import (
    LingbotParallelTrainArtifacts,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
)
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
    sample_coupled_parallel_timestep_values as _sample_coupled_timestep_values,
    sample_index_matched_timestep_values as _sample_index_matched_timestep_values,
    sample_shared_video_schedule_timestep_values as _sample_shared_video_schedule_timestep_values,
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
)


@dataclass
class LingbotParallelInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor
    next_cache: dict[str, Any]
    debug: dict[str, Any]


def _inject_generalist_mode_text_context(
    transformer: torch.nn.Module,
    *,
    policy_config: ParallelStreamPolicyConfig,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    mode: JointDenoiseTrainingMode | str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not bool(getattr(policy_config, "generalist_mode_text_token", False)):
        return text_emb, negative_text_emb
    append = getattr(transformer, "append_generalist_mode_context_token", None)
    if not callable(append):
        raise ValueError(
            "Generalist mode text-token ablation requires the runtime transformer "
            "to support mode-token appending."
        )
    mode_value = JointDenoiseTrainingMode(mode).value
    base_text_tokens = int(text_emb.shape[1])
    text_emb = append(text_emb, mode_value)
    token_count = int(text_emb.shape[1] - base_text_tokens)
    if token_count != 1:
        raise ValueError(
            "Generalist mode text-token ablation expects the runtime transformer "
            f"to append exactly one token, got {token_count}."
        )
    if negative_text_emb is not None:
        base_negative_tokens = int(negative_text_emb.shape[1])
        negative_text_emb = append(negative_text_emb, mode_value)
        negative_token_count = int(negative_text_emb.shape[1] - base_negative_tokens)
        if negative_token_count != token_count:
            raise ValueError(
                "Generalist mode text-token ablation expects conditioned and CFG-negative "
                "branches to append the same number of tokens, "
                f"got conditioned={token_count} and negative={negative_token_count}."
            )
    return text_emb, negative_text_emb


def _repeat_joint_input_for_cfg(
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    latent_dict = dict(input_dict["latent_dict"])  # type: ignore[index]
    action_dict = dict(input_dict["action_dict"])  # type: ignore[index]
    repeated_latent_dict = {
        **latent_dict,
        "noisy_latents": latent_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": latent_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": latent_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": latent_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": latent_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([latent_dict["text_emb"], negative_text_emb], dim=0),
    }
    repeated_action_dict = {
        **action_dict,
        "noisy_latents": action_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": action_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": action_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": action_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": action_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([action_dict["text_emb"], negative_text_emb], dim=0),
    }
    if "actions_mask" in action_dict:
        repeated_action_dict["actions_mask"] = action_dict["actions_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in latent_dict:
        repeated_latent_dict["loss_mask"] = latent_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in action_dict:
        repeated_action_dict["loss_mask"] = action_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    repeated_input = {
        **input_dict,
        "latent_dict": repeated_latent_dict,
        "action_dict": repeated_action_dict,
    }
    proprio_state = input_dict.get("per_chunk_proprio_state")
    if isinstance(proprio_state, torch.Tensor):
        repeated_input["per_chunk_proprio_state"] = proprio_state.repeat(2, 1, 1)
    return repeated_input


def run_parallel_exact_cache_warmup(
    *,
    transformer: torch.nn.Module,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    observed_video_latents: torch.Tensor,
    observed_action_latents: torch.Tensor,
    text_emb: torch.Tensor | None,
    negative_text_emb: torch.Tensor | None,
    action_channel_mask: torch.Tensor | None,
    infer_cache: dict[str, Any],
    cache_write_mode: ParallelExactCacheWriteMode | str = ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
    frame_start_override: int | None = None,
    action_conditioning_mode: JointDenoiseTrainingMode | str = "vanilla_joint_rollout",
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = observed_video_latents.device
    batch_size, _, observed_frames, latent_height, latent_width = observed_video_latents.shape
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
    rollout_mode = _generalist_mode_for_action_conditioning(action_conditioning_mode)
    rollout_window_size = _window_size_for_generalist_conditioning(
        rollout_mode,
        fallback_window_size=int(policy_config.attn_window),
    )
    rollout_frame_chunk_size = _chunk_size_for_generalist_conditioning(
        rollout_mode,
        fallback_chunk_size=int(inference_config.frame_chunk_size),
    )
    rollout_history_stream_visibility = _history_stream_visibility_for_generalist_conditioning(
        rollout_mode,
        policy_config,
    )
    generalist_mode = None
    if _uses_generalist_mode_text_token(policy_config):
        generalist_mode = rollout_mode
        text_emb, negative_text_emb = _inject_generalist_mode_text_context(
            transformer,
            policy_config=policy_config,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            mode=generalist_mode,
        )
    text_emb, negative_text_emb = _inject_proprio_text_context(
        transformer,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        proprio_state=proprio_state,
    )
    cache_spec = _build_exact_cache_spec(
        write_mode=cache_write_mode,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_generalist_conditioning(rollout_mode, policy_config),
    )
    current_frame_start = (
        int(infer_cache.get("frame_start", 0))
        if frame_start_override is None
        else int(frame_start_override)
    )
    cached_batch_size = int(infer_cache.get("batch_size", batch_size))
    cached_latent_height = int(infer_cache.get("latent_height", latent_height))
    cached_latent_width = int(infer_cache.get("latent_width", latent_width))

    if inference_config.use_cache and cache_context.cache_initialized:
        _validate_existing_exact_cache_attn_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=rollout_window_size,
        )
    if inference_config.use_cache and (
        not cache_context.cache_initialized
        or cached_batch_size != batch_size
        or cached_latent_height != latent_height
        or cached_latent_width != latent_width
    ):
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
            attn_window=rollout_window_size,
            frame_chunk_size=rollout_frame_chunk_size,
        )
        if frame_start_override is None:
            current_frame_start = 0

    (
        warmup_video_latents,
        warmup_action_latents,
        warmup_frame_start,
        warmup_dropped_frames,
    ) = _select_conditional_warmup_history_suffix(
        video_latents=observed_video_latents,
        action_latents=observed_action_latents,
        frame_start=current_frame_start,
        frame_chunk_size=rollout_frame_chunk_size,
        mode=rollout_mode,
    )
    video_hidden_context = _single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=warmup_video_latents,
        action_mode=False,
    )
    action_hidden_context = _single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=warmup_action_latents,
        action_mode=True,
    )

    if inference_config.use_cache:
        _clear_exact_prediction_cache(transformer, cache_name=cache_context.cache_name)

    # Warmup pushes already-observed history into the transformer cache without
    # denoising it. Both streams therefore use timestep `0.0`, and the
    # resulting KV cache represents the observed prefix before generation
    # starts at `frame_start_after`.
    _write_exact_cache_chunk(
        transformer=transformer,
        cache_spec=cache_spec,
        cache_name=cache_context.cache_name,
        frame_start=warmup_frame_start,
        backbone_config=backbone_config,
        video_latents=warmup_video_latents.to(dtype=cache_context.model_dtype),
        action_latents=warmup_action_latents.to(device=device, dtype=cache_context.model_dtype),
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        use_cfg=cache_context.use_cfg and inference_config.use_cache,
        action_channel_mask=action_channel_mask,
        update_cache=2 if inference_config.use_cache else 0,
        chunk_size=rollout_frame_chunk_size,
        window_size=rollout_window_size,
        current_block_coupling=resolve_parallel_current_block_coupling(policy_config),
        preserve_video_pretrain_history=bool(
            getattr(policy_config, "preserve_video_pretrain_history", False)
        ),
        history_stream_visibility=rollout_history_stream_visibility,
        video_hidden_context=video_hidden_context,
        action_hidden_context=action_hidden_context,
        allow_cache_prefix_during_update_write=_is_conditional_joint_denoise_mode(rollout_mode),
    )
    debug = {
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "use_cfg": cache_context.use_cfg,
        "batch_size": batch_size,
        "observed_frames": observed_frames,
        "warmup_retained_frames": int(max(warmup_video_latents.shape[2], warmup_action_latents.shape[2])),
        "warmup_dropped_frames": int(warmup_dropped_frames),
        "warmup_frame_start": int(warmup_frame_start),
        "frame_start_before": int(infer_cache.get("frame_start", 0)),
        "frame_start_override": None if frame_start_override is None else int(frame_start_override),
        "frame_start_after": current_frame_start + observed_frames,
        "cache_write_mode": str(cache_spec.write_mode),
        "action_conditioning_mode": str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
        "generalist_mode_text_token": None if generalist_mode is None else generalist_mode.value,
        "generalist_mode_text_token_count": int(generalist_mode is not None),
        "rollout_window_size": int(rollout_window_size),
        "rollout_frame_chunk_size": int(rollout_frame_chunk_size),
        "history_stream_visibility": rollout_history_stream_visibility.value,
        "generalist_conditional_history_chunks": 1 if _is_conditional_joint_denoise_mode(rollout_mode) else 0,
    }
    return {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": cache_context.cache_initialized and inference_config.use_cache,
        "frame_start": current_frame_start + observed_frames,
        "latent_height": cache_context.latent_height,
        "latent_width": cache_context.latent_width,
        "batch_size": cache_context.batch_size,
        "step_index": int(infer_cache.get("step_index", 0)),
        "use_cfg": cache_context.use_cfg,
        "frame_chunk_size": int(rollout_frame_chunk_size),
        "debug_last_warmup": debug,
    }


def _maybe_commit_initial_observed_video_context(
    *,
    transformer: torch.nn.Module,
    cache_spec: ExactCacheInterfaceSpec,
    cache_name: str,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    inference_config: InferenceConfig,
    condition_latents: torch.Tensor | None,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
    action_dim: int,
    model_dtype: torch.dtype,
    current_frame_start: int,
    step_index: int,
    current_block_coupling: CurrentBlockCoupling,
    window_size: int,
    frame_chunk_size: int | None = None,
    history_stream_visibility: ParallelHistoryStreamVisibility | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> tuple[int, bool]:
    """Commit frame 0 as pure prefix context before generating frame 1.

    The rollout-parity contract is: observed frame 0 is conditioning only, and
    the first denoised chunk starts at frame 1. This helper writes that observed
    video frame into the exact cache without materializing dummy action tokens.
    """

    resolved_frame_chunk_size = max(
        1,
        int(inference_config.frame_chunk_size if frame_chunk_size is None else frame_chunk_size),
    )
    resolved_history_stream_visibility = (
        resolve_parallel_history_stream_visibility(policy_config)
        if history_stream_visibility is None
        else history_stream_visibility
    )
    startup_plan = resolve_strict_startup_plan(
        step_index=step_index,
        current_start_frame=current_frame_start,
        frame_chunk_size=resolved_frame_chunk_size,
        action_tokens_per_frame=policy_config.action_per_frame,
        action_horizon=resolved_frame_chunk_size * policy_config.action_per_frame,
    )
    if not inference_config.use_cache or not startup_plan.is_startup or condition_latents is None:
        return int(current_frame_start), False

    observed_video = condition_latents[:, :, :1].to(dtype=model_dtype)
    observed_actions = observed_video.new_empty(
        observed_video.shape[0],
        int(action_dim),
        0,
        int(policy_config.action_per_frame),
        1,
    )
    prefix_cache_spec = cache_spec
    if cache_spec.write_mode != ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED:
        prefix_cache_spec = _build_exact_cache_spec(
            write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
            batch_size=int(observed_video.shape[0]),
            use_cfg=bool(use_cfg),
            prefix_visibility_mode=cache_spec.prefix_visibility_mode,
        )
    prefix_coupling = (
        current_block_coupling
        if current_block_coupling
        in {
            CurrentBlockCoupling.VIDEO_THEN_ACTION,
            CurrentBlockCoupling.ACTION_THEN_VIDEO,
            CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        }
        else CurrentBlockCoupling.VIDEO_THEN_ACTION
    )
    video_hidden_context = (
        None
        if _uses_legacy_prefix_per_chunk_proprio_contract(policy_config)
        else _single_stream_hidden_proprio_context(
            transformer,
            proprio_state=hidden_proprio_state,
            stream_latents=observed_video,
            action_mode=False,
        )
    )
    _write_exact_cache_chunk(
        transformer=transformer,
        cache_spec=prefix_cache_spec,
        cache_name=cache_name,
        frame_start=0,
        backbone_config=backbone_config,
        video_latents=observed_video,
        action_latents=observed_actions,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        use_cfg=bool(use_cfg),
        action_channel_mask=action_channel_mask,
        update_cache=2,
        chunk_size=resolved_frame_chunk_size,
        window_size=window_size,
        current_block_coupling=prefix_coupling,
        preserve_video_pretrain_history=bool(
            getattr(policy_config, "preserve_video_pretrain_history", False)
        ),
        history_stream_visibility=resolved_history_stream_visibility,
        video_hidden_context=video_hidden_context,
    )
    return startup_plan.generation_frame_start, True


def run_parallel_exact_inference_rollout(
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
) -> LingbotParallelInferArtifacts:
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
    generalist_mode = None
    if _uses_generalist_mode_text_token(policy_config):
        generalist_mode = JointDenoiseTrainingMode.JOINT
        text_emb, negative_text_emb = _inject_generalist_mode_text_context(
            transformer,
            policy_config=policy_config,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            mode=generalist_mode,
        )
    text_emb, negative_text_emb = _inject_proprio_text_context(
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
    cache_spec = _build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_policy(policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError("Exact LingBot inference requires condition latents on the first chunk when cache is empty.")
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
            attn_window=int(policy_config.attn_window),
        )
    elif inference_config.use_cache and cache_context.cache_initialized:
        _validate_existing_exact_cache_attn_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=int(policy_config.attn_window),
        )
    generation_frame_start = current_frame_start
    initial_observed_context_committed = False
    if cache_context.cache_initialized:
        generation_frame_start, initial_observed_context_committed = _maybe_commit_initial_observed_video_context(
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
    action_hidden_context = _single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=actions,
        action_mode=True,
    )
    video_hidden_context = (
        None
        if _uses_legacy_prefix_per_chunk_proprio_contract(policy_config)
        else _single_stream_hidden_proprio_context(
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
            video_input = prepare_reference_single_stream_input(
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
            video_noise_pred = run_reference_single_stream_forward(
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
                video_noise_pred = data_seq_to_patch(
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
            action_input = prepare_reference_single_stream_input(
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
            action_noise_pred = run_reference_single_stream_forward(
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
            metadata_previous = _set_slot_pool_layer_metadata(
                transformer,
                cache_name=cache_name,
                updates={
                    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS: _single_stream_action_token_count(actions),
                },
            )
            try:
                denoise_video_chunk(commit_to_cache=True)
            finally:
                _restore_slot_pool_layer_metadata(metadata_previous)
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
            _write_exact_cache_chunk(
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
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def _run_parallel_exact_joint_forward_manual(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = prepare_exact_dual_stream_train_sequence(
        input_dict,
        config=transformer.config,
        patch_size=transformer.patch_size,
        model_dtype=reference_runtime_dtype(transformer),
        input_embed=lambda tensor, input_type: transformer._input_embed(tensor, input_type=input_type),
        exact_text_hidden_states=lambda text_emb: transformer._exact_text_hidden_states(
            text_emb,
            dtype=reference_runtime_dtype(transformer),
        ),
        time_embed=lambda timesteps, height, width, dtype, action_mode: transformer._time_embed(
            timesteps,
            height,
            width,
            dtype=dtype,
            action_mode=action_mode,
        ),
        rope=transformer.rope,
    )
    batch_size = prepared.batch_size
    hidden_states = prepared.hidden_states
    text_hidden_states = prepared.text_hidden_states
    rotary_emb = prepared.rotary_emb
    temb = prepared.temb
    timestep_proj = prepared.timestep_proj
    split_list = prepared.split_list
    exact_attention_profile = prepared.attention_profile
    hidden_states = _apply_parallel_chunk_proprio_context(
        transformer,
        hidden_states=hidden_states,
        split_list=split_list,
        input_dict=input_dict,
    )
    cache_stream_ids = _stream_ids_for_exact_dual_stream_split(
        split_list,
        device=hidden_states.device,
    )
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    cache_backend_name = cache_state.backend_name if cache_state is not None else None
    cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
    if cache_backend_uses_slot_pool(cache_backend_name):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        assert isinstance(latent_dict, dict)
        assert isinstance(action_dict, dict)
        attention_profile_name = input_dict.get("attention_profile_name")
        rebuilt_dense_profile = build_chunked_temporal_exact_attention_profile(
            latent_shape=tuple(int(dim) for dim in latent_dict["noisy_latents"].shape),
            action_shape=tuple(int(dim) for dim in action_dict["noisy_latents"].shape),
            padded_length=int(hidden_states.shape[1] - sum(int(length) for length in split_list[:4])),
            chunk_size=int(input_dict["chunk_size"]),
            window_size=int(input_dict["window_size"]),
            patch_size=transformer.patch_size,
            text_token_count=int(latent_dict["text_emb"].shape[1]),
            base_text_token_count=(
                None
                if input_dict.get("base_text_token_count") is None
                else int(input_dict["base_text_token_count"])
            ),
            proprio_context_token_count=int(input_dict.get("proprio_context_token_count", 0) or 0),
            chunk_origin_frame=int(input_dict.get("chunk_origin_frame", 0) or 0),
            prefix_condition_frames=int(input_dict.get("prefix_condition_frames", 0) or 0),
            singleton_chunk_frame=(
                None
                if input_dict.get("singleton_chunk_frame") is None
                else int(input_dict["singleton_chunk_frame"])
            ),
            action_context_mask=(
                action_dict.get("actions_mask")
                if torch.is_tensor(action_dict.get("actions_mask"))
                else None
            ),
            device=hidden_states.device,
            build_dense_masks=True,
            build_flex_masks=False,
            current_block_coupling=(
                str(attention_profile_name)
                if attention_profile_name not in (None, "none")
                else None
            ),
            preserve_video_pretrain_history=bool(
                input_dict.get("preserve_video_pretrain_history", False)
            ),
            history_stream_visibility=input_dict.get("history_stream_visibility"),
            conditional_history_policy=input_dict.get("conditional_history_policy"),
        )
        exact_attention_profile = PreparedAttentionProfile(
            spec=rebuilt_dense_profile.spec,
            self_attention_mask=rebuilt_dense_profile.self_attention_mask,
            cross_attention_mask=rebuilt_dense_profile.cross_attention_mask,
            self_attention_block_mask=None,
            cross_attention_block_mask=None,
            metadata=dict(rebuilt_dense_profile.metadata),
        )

    for layer_index, block in enumerate(transformer.blocks):
        hidden_states, _, _ = block(
            hidden_states,
            encoder_hidden_states=text_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            attention_profile=exact_attention_profile,
            self_attention_cache_backend_name=cache_backend_name,
            self_attention_cache_backend_state=(
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            ),
            self_attention_cache_update_mode=update_cache,
            self_attention_cache_stream_ids=cache_stream_ids,
        )

    temb_scale_shift_table = transformer.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
    if cache_state is not None and cache_backend_uses_slot_pool(cache_backend_name):
        materialized_entries = materialize_cache_backend_entries(cache_backend_payload)
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=materialized_entries,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )
    latent_hidden_states, _, action_hidden_states, _, _ = torch.split(
        hidden_states,
        tuple(int(length) for length in split_list),
        dim=1,
    )
    effective_batch_size = int(input_dict["latent_dict"]["noisy_latents"].shape[0])  # type: ignore[index]
    latent_hidden_states = transformer.proj_out(latent_hidden_states)
    if latent_hidden_states.shape[0] == 1:
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif latent_hidden_states.shape[0] == effective_batch_size:
        latent_hidden_states = latent_hidden_states.contiguous()
    else:
        raise ValueError(
            "Unexpected exact joint latent output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {latent_hidden_states.shape[0]}."
        )
    action_hidden_states = transformer.action_proj_out(action_hidden_states)
    if action_hidden_states.shape[0] == 1:
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif action_hidden_states.shape[0] != effective_batch_size:
        raise ValueError(
            "Unexpected exact joint action output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {action_hidden_states.shape[0]}."
        )
    return latent_hidden_states, action_hidden_states


def _build_fastwam_first_frame_attention_profile(
    *,
    batch_size: int,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    padded_length: int,
    text_token_count: int,
    device: torch.device,
) -> PreparedAttentionProfile:
    video_seq_ids = torch.arange(batch_size, device=device)[:, None].expand(-1, video_seq_len).flatten()
    action_seq_ids = torch.arange(batch_size, device=device)[:, None].expand(-1, action_seq_len).flatten()
    seq_ids = torch.cat([video_seq_ids, action_seq_ids])

    video_local_ids = torch.arange(video_seq_len, device=device)[None].expand(batch_size, -1).flatten()
    action_local_ids = torch.arange(action_seq_len, device=device)[None].expand(batch_size, -1).flatten()
    local_ids = torch.cat([video_local_ids, action_local_ids])

    stream_ids = torch.cat(
        [
            torch.zeros_like(video_seq_ids),
            torch.ones_like(action_seq_ids),
        ]
    )
    if padded_length > 0:
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        local_ids = F.pad(local_ids, (0, padded_length), value=-1)
        stream_ids = F.pad(stream_ids, (0, padded_length), value=-1)

    q_seq = seq_ids[:, None]
    kv_seq = seq_ids[None, :]
    q_local = local_ids[:, None]
    kv_local = local_ids[None, :]
    q_stream = stream_ids[:, None]
    kv_stream = stream_ids[None, :]
    same_seq = (q_seq == kv_seq) & (q_seq >= 0) & (kv_seq >= 0)

    first_frame_tokens = max(1, int(video_tokens_per_frame))
    video_to_video = (q_stream == 0) & (kv_stream == 0)
    first_frame_query_to_future_video = (q_local < first_frame_tokens) & (kv_local >= first_frame_tokens)
    video_to_video = video_to_video & ~first_frame_query_to_future_video
    action_to_action = (q_stream == 1) & (kv_stream == 1)
    action_to_first_frame_video = (q_stream == 1) & (kv_stream == 0) & (kv_local < first_frame_tokens)
    self_attention_mask = same_seq & (video_to_video | action_to_action | action_to_first_frame_video)

    text_seq_ids = torch.arange(batch_size, device=device)[:, None].expand(-1, text_token_count).flatten()
    cross_attention_mask = (
        (seq_ids[:, None] == text_seq_ids[None, :])
        & (seq_ids[:, None] >= 0)
        & (text_seq_ids[None, :] >= 0)
    )
    return PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="fastwam_first_frame",
            family="fastwam",
            backend="sdpa_dense",
        ),
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        metadata={
            "batch_size": int(batch_size),
            "video_seq_len": int(video_seq_len),
            "action_seq_len": int(action_seq_len),
            "video_tokens_per_frame": int(video_tokens_per_frame),
            "padded_length": int(padded_length),
            "text_token_count": int(text_token_count),
        },
    )


def _run_parallel_fastwam_first_frame_forward_manual(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the dedicated FastWAM first-frame two-stream transformer path.

    This intentionally bypasses the standard exact-runtime dispatch because the
    FastWAM mask is a compact two-stream topology: first-frame video tokens,
    future video tokens, and action tokens. Keep this path in sync with
    SharedTransformerBlock.forward if that block signature or return contract
    changes.
    """
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)

    model_dtype = reference_runtime_dtype(transformer)
    latent_noisy = latent_dict["noisy_latents"].to(model_dtype)
    action_noisy = action_dict["noisy_latents"].to(model_dtype)
    text_emb = latent_dict["text_emb"].to(model_dtype)
    batch_size = int(latent_noisy.shape[0])

    video_hidden_states = transformer._input_embed(latent_noisy, input_type="latent").flatten(0, 1).contiguous()[None].clone()
    action_hidden_states = transformer._input_embed(action_noisy, input_type="action").flatten(0, 1).contiguous()[None].clone()
    text_hidden_states = transformer._exact_text_hidden_states(text_emb, dtype=model_dtype).flatten(0, 1).contiguous()[None].clone()
    hidden_states = torch.cat([video_hidden_states, action_hidden_states], dim=1)
    video_seq_len = int(video_hidden_states.shape[1])
    action_seq_len = int(action_hidden_states.shape[1])
    hidden_states = _apply_parallel_chunk_proprio_context(
        transformer,
        hidden_states=hidden_states,
        split_list=(video_seq_len, 0, action_seq_len, 0),
        input_dict=input_dict,
    )

    latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    full_grid_id = torch.cat([latent_grid_id, action_grid_id], dim=2)
    rotary_emb = transformer.rope(full_grid_id)[:, :, None]

    latent_time_steps = latent_dict["timesteps"].flatten(0, 1).contiguous()[None].clone()
    action_time_steps = action_dict["timesteps"].flatten(0, 1).contiguous()[None].clone()
    latent_temb, latent_timestep_proj = transformer._time_embed(
        latent_time_steps,
        int(latent_noisy.shape[-2]),
        int(latent_noisy.shape[-1]),
        dtype=hidden_states.dtype,
        action_mode=False,
    )
    action_temb, action_timestep_proj = transformer._time_embed(
        action_time_steps,
        int(action_noisy.shape[-2]),
        int(action_noisy.shape[-1]),
        dtype=hidden_states.dtype,
        action_mode=True,
    )
    temb = torch.cat([latent_temb, action_temb], dim=1)
    timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

    total_length = int(hidden_states.shape[1])
    padded_length = (128 - total_length % 128) % 128
    if padded_length > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

    patch_t, patch_h, patch_w = transformer.patch_size
    video_tokens_per_frame = (int(latent_noisy.shape[-2]) // patch_h) * (int(latent_noisy.shape[-1]) // patch_w)
    attention_profile = _build_fastwam_first_frame_attention_profile(
        batch_size=batch_size,
        video_seq_len=video_seq_len // batch_size,
        action_seq_len=action_seq_len // batch_size,
        video_tokens_per_frame=video_tokens_per_frame,
        padded_length=padded_length,
        text_token_count=int(text_emb.shape[1]),
        device=hidden_states.device,
    )

    for block in transformer.blocks:
        hidden_states, _, _ = block(
            hidden_states,
            encoder_hidden_states=text_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            attention_profile=attention_profile,
        )

    temb_scale_shift_table = transformer.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
    video_hidden_states, action_hidden_states, _ = torch.split(
        hidden_states,
        (video_seq_len, action_seq_len, padded_length),
        dim=1,
    )
    video_pred = transformer.proj_out(video_hidden_states)
    video_pred = rearrange(video_pred, "1 (b l) c -> b l c", b=batch_size)
    action_pred = transformer.action_proj_out(action_hidden_states)
    action_pred = rearrange(action_pred, "1 (b l) c -> b l c", b=batch_size)
    return video_pred, action_pred


def run_parallel_fastwam_first_frame_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_parallel_fastwam_first_frame_forward_manual(transformer, input_dict)


def _run_parallel_action_conditioned_forward(
    transformer: torch.nn.Module,
    *,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    video_guidance_scale: float,
    action_guidance_scale: float,
    negative_text_emb: torch.Tensor | None,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    def _split_cfg_prediction(
        prediction: torch.Tensor,
        *,
        logical_batch_size: int,
        expected_tokens: int,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.ndim != 3:
            raise ValueError(f"Expected {name} prediction rank 3, got shape {tuple(prediction.shape)}.")
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == expected_tokens:
            return prediction[:logical_batch_size], prediction[logical_batch_size:]
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == logical_batch_size * 2 * expected_tokens:
            packed = rearrange(
                prediction,
                "(g b_row) (h b_seq l) c -> g b_row h b_seq l c",
                g=2,
                h=2,
                b_row=logical_batch_size,
                b_seq=logical_batch_size,
                l=expected_tokens,
            )
            batch_index = torch.arange(logical_batch_size, device=prediction.device)
            cond = packed[0, batch_index, 0, batch_index]
            uncond = packed[1, batch_index, 1, batch_index]
            return cond.contiguous(), uncond.contiguous()
        if prediction.shape[0] == logical_batch_size and prediction.shape[1] == expected_tokens * 2:
            return prediction[:, :expected_tokens], prediction[:, expected_tokens:]
        if prediction.shape[0] == 1 and prediction.shape[1] == logical_batch_size * expected_tokens * 2:
            unpacked = rearrange(
                prediction,
                "1 (g b l) c -> (g b) l c",
                g=2,
                b=logical_batch_size,
                l=expected_tokens,
            )
            return unpacked[:logical_batch_size], unpacked[logical_batch_size:]
        raise ValueError(
            f"Unable to split CFG {name} prediction with shape {tuple(prediction.shape)}; "
            f"expected logical_batch_size={logical_batch_size}, expected_tokens={expected_tokens}."
        )

    batch_size = input_dict["latent_dict"]["noisy_latents"].shape[0]  # type: ignore[index]
    latent_noisy = input_dict["latent_dict"]["noisy_latents"]  # type: ignore[index]
    action_noisy = input_dict["action_dict"]["noisy_latents"]  # type: ignore[index]
    expected_video_tokens = (
        int(latent_noisy.shape[2]) // transformer.patch_size[0]
    ) * (
        int(latent_noisy.shape[3]) // transformer.patch_size[1]
    ) * (
        int(latent_noisy.shape[4]) // transformer.patch_size[2]
    )
    expected_action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
    use_cfg = negative_text_emb is not None and (video_guidance_scale > 1.0 or action_guidance_scale > 1.0)
    effective_input = input_dict
    if use_cfg:
        effective_input = _repeat_joint_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)
    with torch.inference_mode():
        video_pred, action_pred = _run_parallel_exact_joint_forward_manual(
            transformer,
            effective_input,
            update_cache=update_cache,
            cache_name=cache_name,
        )
    if not use_cfg:
        return video_pred, action_pred
    cond_video_pred, uncond_video_pred = _split_cfg_prediction(
        video_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_video_tokens,
        name="video",
    )
    cond_action_pred, uncond_action_pred = _split_cfg_prediction(
        action_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_action_tokens,
        name="action",
    )
    combined_video_pred = uncond_video_pred + video_guidance_scale * (cond_video_pred - uncond_video_pred)
    combined_action_pred = uncond_action_pred + action_guidance_scale * (cond_action_pred - uncond_action_pred)
    return combined_video_pred, combined_action_pred


def _run_parallel_action_conditioned_inference_rollout_impl(
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
) -> LingbotParallelInferArtifacts:
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
    rollout_mode = _generalist_mode_for_action_conditioning(action_conditioning_mode)
    rollout_window_size = _window_size_for_generalist_conditioning(
        rollout_mode,
        fallback_window_size=int(policy_config.attn_window),
    )
    rollout_frame_chunk_size = _chunk_size_for_generalist_conditioning(
        rollout_mode,
        fallback_chunk_size=int(inference_config.frame_chunk_size),
    )
    rollout_history_stream_visibility = _history_stream_visibility_for_generalist_conditioning(
        rollout_mode,
        policy_config,
    )
    condition_latents = _slice_conditioning_chunk(
        condition_latents,
        target_frames=rollout_frame_chunk_size,
        source="condition latents",
    )
    generalist_mode = None
    if _uses_generalist_mode_text_token(policy_config):
        generalist_mode = rollout_mode
        text_emb, negative_text_emb = _inject_generalist_mode_text_context(
            transformer,
            policy_config=policy_config,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            mode=generalist_mode,
        )
    text_emb, negative_text_emb = _inject_proprio_text_context(
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
    cache_spec = _build_exact_cache_spec(
        write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        batch_size=batch_size,
        use_cfg=cache_context.use_cfg,
        prefix_visibility_mode=_prefix_visibility_mode_for_generalist_conditioning(rollout_mode, policy_config),
    )
    if inference_config.use_cache and not cache_context.cache_initialized:
        if condition_latents is None:
            raise ValueError(
                "Joint exact inference requires condition latents on the first chunk when cache is empty."
            )
        cache_context = _ensure_exact_cache_initialized(
            transformer=transformer,
            policy_config=policy_config,
            inference_config=inference_config,
            cache_context=cache_context,
            cache_spec=cache_spec,
            attn_window=rollout_window_size,
            frame_chunk_size=rollout_frame_chunk_size,
        )
    elif inference_config.use_cache and cache_context.cache_initialized:
        _validate_existing_exact_cache_attn_window(
            transformer,
            cache_name=cache_context.cache_name,
            requested_attn_window=rollout_window_size,
        )
    initial_observed_context_committed = False
    if cache_context.cache_initialized:
        generation_frame_start, initial_observed_context_committed = _maybe_commit_initial_observed_video_context(
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
        forced_action_latents = _slice_conditioning_chunk(
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
            forced_action_noise = _slice_conditioning_chunk(
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
        commit_action_latents = _slice_conditioning_chunk(
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
            attention_profile_name = _attention_profile_name_for_current_block_coupling(current_block_coupling)

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
        latent_grid_id = get_mesh_id(
            rollout_frame_chunk_size // backbone_config.patch_size_t,
            latent_height // backbone_config.patch_size_h,
            latent_width // backbone_config.patch_size_w,
            t=0,
            f_w=1,
            f_shift=generation_frame_start,
            action=False,
            device=device,
        )[None].repeat(batch_size, 1, 1)
        action_grid_id = get_mesh_id(
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
            if _uses_legacy_prefix_per_chunk_proprio_contract(policy_config):
                input_dict["per_chunk_proprio_apply_to_video"] = False
        video_noise_pred, action_noise_pred = _run_parallel_action_conditioned_forward(
            transformer,
            input_dict=input_dict,
            video_guidance_scale=float(inference_config.guidance_scale),
            action_guidance_scale=float(inference_config.action_guidance_scale),
            negative_text_emb=negative_text_emb,
            update_cache=0,
            cache_name=cache_name,
        )
        video_noise_pred = data_seq_to_patch(
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
    video_hidden_context = _single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=latents,
        action_mode=False,
    )
    action_hidden_context = _single_stream_hidden_proprio_context(
        transformer,
        proprio_state=hidden_proprio_state,
        stream_latents=cache_action_latents,
        action_mode=True,
    )

    if inference_config.use_cache:
        _write_exact_cache_chunk(
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
            allow_cache_prefix_during_update_write=_is_conditional_joint_denoise_mode(rollout_mode),
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
        "generalist_conditional_history_chunks": 1 if _is_conditional_joint_denoise_mode(rollout_mode) else 0,
    }
    cache_summary = _summarize_slot_pool_cache_state(transformer, cache_name)
    if cache_summary is not None:
        debug.update(cache_summary)
    output_dtype = condition_latents.dtype if condition_latents is not None else model_dtype
    action_pred = rearrange(returned_action_latents, "b c f n 1 -> b (f n) c").to(dtype=output_dtype)
    return LingbotParallelInferArtifacts(
        action_pred=action_pred,
        predicted_latents=latents.to(dtype=output_dtype),
        next_cache=next_cache,
        debug=debug,
    )


def run_parallel_action_conditioned_inference_rollout(
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
) -> LingbotParallelInferArtifacts:
    return _run_parallel_action_conditioned_inference_rollout_impl(
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


def run_parallel_action_conditioned_action_override_inference_rollout(
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
) -> LingbotParallelInferArtifacts:
    """Run joint-denoise inference with ablation-owned action overrides.

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

    return _run_parallel_action_conditioned_inference_rollout_impl(
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


def run_parallel_exact_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_dict.get("per_chunk_proprio_state") is not None:
        return _run_parallel_exact_joint_forward_manual(transformer, input_dict)
    if hasattr(transformer, "execute_runtime_step"):
        step_output = transformer.execute_runtime_step(
            RuntimeStepInput(
                program=build_chunked_dual_stream_exact_train_program(
                    attention_profile_name=input_dict.get("attention_profile_name"),  # type: ignore[arg-type]
                    cache_backend_name="slot_pool_exact",
                ),
                payload=input_dict,
            )
        )
        try:
            return (
                step_output.projected_outputs["video_prediction"],
                step_output.projected_outputs["action_prediction"],
            )
        except KeyError as exc:
            raise ValueError("Exact dual-stream runtime step did not return both video/action predictions.") from exc

    forward_train = getattr(transformer, "forward_train", None)
    if callable(forward_train):
        return forward_train(input_dict)
    return _run_parallel_exact_joint_forward_manual(transformer, input_dict)


def run_parallel_action_conditioned_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the LingBot exact train-time packed layout and shared runtime
    # execution path; the joint-denoise variant changes inference rollout
    # semantics, not the backbone's train-time sequence contract.
    return run_parallel_exact_train(transformer, input_dict)
