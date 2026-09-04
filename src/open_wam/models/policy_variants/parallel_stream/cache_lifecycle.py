"""Manage parallel-stream exact-cache startup and observed-history commits."""

from __future__ import annotations

from typing import Any

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    DynamicsObjective,
    HistoryStreamVisibility,
    ParallelExactCacheWriteMode,
)
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.dynamics_objectives import (
    resolve_dynamics_objective_semantics,
    resolve_dynamics_rollout_geometry,
    resolve_dynamics_rollout_objective,
)
from open_wam.models.common.rollout_startup import resolve_strict_startup_plan
from open_wam.models.visual_tower.exact_runtime import (
    clear_exact_prediction_cache as _clear_exact_prediction_cache,
)

from .cache_execution import (
    write_exact_cache_chunk as _write_exact_cache_chunk,
)
from .conditional_rollout import (
    dynamics_rollout_prefix_visibility_mode,
    select_dynamics_warmup_history_suffix,
    uses_dynamics_mode_text_token,
)
from .exact_cache import (
    ExactCacheInterfaceSpec,
)
from .exact_cache import (
    build_exact_cache_spec as _build_exact_cache_spec,
)
from .exact_cache import (
    ensure_exact_cache_initialized as _ensure_exact_cache_initialized,
)
from .exact_cache import (
    resolve_exact_cache_context as _resolve_exact_cache_context,
)
from .exact_cache import (
    validate_existing_exact_cache_attention_window as _validate_existing_exact_cache_attn_window,
)
from .inference_conditioning import append_generalist_mode_text_context
from .proprio_conditioning import (
    build_single_stream_hidden_proprio_context as _single_stream_hidden_proprio_context,
)
from .proprio_conditioning import (
    build_single_stream_hidden_proprio_history_context as _single_stream_hidden_proprio_history_context,
)
from .proprio_conditioning import (
    inject_deprecated_proprio_text_context as _inject_proprio_text_context,
)
from .runtime_semantics import (
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
)
from .runtime_semantics import (
    uses_legacy_prefix_per_chunk_proprio_contract as _uses_legacy_prefix_per_chunk_proprio_contract,
)


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
    cache_write_mode: ParallelExactCacheWriteMode
    | str = ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED,
    frame_start_override: int | None = None,
    dynamics_objective: DynamicsObjective | str | None = None,
    proprio_state: torch.Tensor | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
    hidden_proprio_history: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = observed_video_latents.device
    batch_size, _, observed_frames, latent_height, latent_width = (
        observed_video_latents.shape
    )
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
    rollout_mode = resolve_dynamics_rollout_objective(
        program=policy_config.program,
        requested_objective=dynamics_objective,
    )
    rollout_semantics = resolve_dynamics_objective_semantics(rollout_mode)
    rollout_geometry = resolve_dynamics_rollout_geometry(
        rollout_semantics,
        fallback_frame_chunk_size=int(inference_config.frame_chunk_size),
        fallback_attention_window_size=int(inference_config.attention_window_size),
        fallback_history_stream_visibility=policy_config.history_stream_visibility,
    )
    rollout_window_size = rollout_geometry.attention_window_size
    rollout_frame_chunk_size = rollout_geometry.frame_chunk_size
    rollout_history_stream_visibility = rollout_geometry.history_stream_visibility
    if rollout_semantics.drop_text_conditioning:
        text_emb = torch.zeros_like(text_emb)
        if negative_text_emb is not None:
            negative_text_emb = torch.zeros_like(negative_text_emb)
    generalist_mode = None
    if uses_dynamics_mode_text_token(policy_config):
        generalist_mode = rollout_mode
        text_emb, negative_text_emb = append_generalist_mode_text_context(
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
        prefix_visibility_mode=dynamics_rollout_prefix_visibility_mode(
            rollout_geometry, policy_config
        ),
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
    ) = select_dynamics_warmup_history_suffix(
        video_latents=observed_video_latents,
        action_latents=observed_action_latents,
        frame_start=current_frame_start,
        geometry=rollout_geometry,
    )
    warmup_hidden_proprio_history = _select_warmup_hidden_proprio_history(
        hidden_proprio_history,
        observed_frames=int(observed_frames),
        retained_frames=int(warmup_video_latents.shape[2]),
    )
    if warmup_hidden_proprio_history is None:
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
    else:
        video_hidden_context = _single_stream_hidden_proprio_history_context(
            transformer,
            proprio_history=warmup_hidden_proprio_history,
            stream_latents=warmup_video_latents,
            action_mode=False,
        )
        action_hidden_context = _single_stream_hidden_proprio_history_context(
            transformer,
            proprio_history=warmup_hidden_proprio_history,
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
        action_latents=warmup_action_latents.to(
            device=device, dtype=cache_context.model_dtype
        ),
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        use_cfg=cache_context.use_cfg and inference_config.use_cache,
        action_channel_mask=action_channel_mask,
        update_cache=2 if inference_config.use_cache else 0,
        chunk_size=rollout_frame_chunk_size,
        window_size=rollout_window_size,
        current_block_coupling=resolve_parallel_current_block_coupling(policy_config),
        history_stream_visibility=rollout_history_stream_visibility,
        video_hidden_context=video_hidden_context,
        action_hidden_context=action_hidden_context,
        allow_cache_prefix_during_update_write=(
            rollout_geometry.conditional_history_policy is not None
        ),
    )
    debug = {
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "use_cfg": cache_context.use_cfg,
        "batch_size": batch_size,
        "observed_frames": observed_frames,
        "warmup_retained_frames": int(
            max(warmup_video_latents.shape[2], warmup_action_latents.shape[2])
        ),
        "warmup_dropped_frames": int(warmup_dropped_frames),
        "warmup_frame_start": int(warmup_frame_start),
        "frame_start_before": int(infer_cache.get("frame_start", 0)),
        "frame_start_override": None
        if frame_start_override is None
        else int(frame_start_override),
        "frame_start_after": current_frame_start + observed_frames,
        "cache_write_mode": str(cache_spec.write_mode),
        "action_conditioning_mode": rollout_mode.value,
        "generalist_mode_text_token": None
        if generalist_mode is None
        else generalist_mode.value,
        "generalist_mode_text_token_count": int(generalist_mode is not None),
        "rollout_window_size": int(rollout_window_size),
        "rollout_frame_chunk_size": int(rollout_frame_chunk_size),
        "history_stream_visibility": rollout_history_stream_visibility.value,
        "generalist_conditional_history_chunks": int(
            rollout_semantics.history_frame_count
        ),
    }
    return {
        "runtime_mode": "lingbot_exact",
        "cache_name": cache_context.cache_name,
        "cache_backend_name": cache_context.cache_backend_name,
        "cache_initialized": cache_context.cache_initialized
        and inference_config.use_cache,
        "frame_start": current_frame_start + observed_frames,
        "latent_height": cache_context.latent_height,
        "latent_width": cache_context.latent_width,
        "batch_size": cache_context.batch_size,
        "step_index": int(infer_cache.get("step_index", 0)),
        "use_cfg": cache_context.use_cfg,
        "frame_chunk_size": int(rollout_frame_chunk_size),
        "debug_last_warmup": debug,
    }


def _select_warmup_hidden_proprio_history(
    history: torch.Tensor | None,
    *,
    observed_frames: int,
    retained_frames: int,
) -> torch.Tensor | None:
    """Select the frame-aligned state suffix written by cache warmup."""

    if history is None:
        return None
    if history.ndim != 3 or int(history.shape[1]) != int(observed_frames):
        raise ValueError(
            "Cache-warmup hidden proprio history must have shape "
            "[B, observed_frames, state_dim], "
            f"got {tuple(history.shape)} for observed_frames={observed_frames}."
        )
    if retained_frames <= 0 or retained_frames > observed_frames:
        raise ValueError(
            "Cache-warmup retained frame count must be within the observed history, "
            f"got retained_frames={retained_frames}, observed_frames={observed_frames}."
        )
    return history[:, -int(retained_frames) :, :].contiguous()


def commit_initial_observed_video_context(
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
    history_stream_visibility: HistoryStreamVisibility | None = None,
    hidden_proprio_state: torch.Tensor | None = None,
) -> tuple[int, bool]:
    """Commit frame 0 as pure prefix context before generating frame 1.

    The rollout-parity contract is: observed frame 0 is conditioning only, and
    the first denoised chunk starts at frame 1. This helper writes that observed
    video frame into the exact cache without materializing dummy action tokens.
    """

    resolved_frame_chunk_size = max(
        1,
        int(
            inference_config.frame_chunk_size
            if frame_chunk_size is None
            else frame_chunk_size
        ),
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
    if (
        not inference_config.use_cache
        or not startup_plan.is_startup
        or condition_latents is None
    ):
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
        history_stream_visibility=resolved_history_stream_visibility,
        video_hidden_context=video_hidden_context,
    )
    return startup_plan.generation_frame_start, True


__all__ = [
    "commit_initial_observed_video_context",
    "run_parallel_exact_cache_warmup",
]
