from __future__ import annotations

from typing import Any

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    ParallelExactCacheWriteMode,
    ParallelHistoryStreamVisibility,
)
from open_wam.models.common import (
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    build_chunked_temporal_exact_attention_profile,
    cache_backend_uses_slot_pool,
    materialize_cache_backend_entries,
)
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower.exact_runtime import (
    prepare_exact_single_stream_input,
    repeat_exact_single_stream_input_for_cfg,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)

from .exact_cache import (
    ExactCacheInterfaceSpec,
    build_clean_video_action_cache_stream_ids,
    count_single_stream_action_tokens,
    restore_slot_pool_layer_metadata,
    set_slot_pool_layer_metadata,
)


def build_joint_clean_cache_attention_mask(
    *,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_token_count: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
    history_stream_visibility: ParallelHistoryStreamVisibility | str | None = None,
) -> torch.Tensor:
    """Materialize the self-attention mask for one clean joint cache write."""

    profile = build_joint_clean_cache_attention_profile(
        latents=latents,
        actions=actions,
        text_token_count=text_token_count,
        backbone_config=backbone_config,
        chunk_size=chunk_size,
        window_size=window_size,
        current_block_coupling=current_block_coupling,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
        history_stream_visibility=history_stream_visibility,
    )
    if profile.self_attention_mask is None:
        raise ValueError(
            "Joint clean cache attention profile did not materialize a clean self-attention mask."
        )
    return profile.self_attention_mask


def build_joint_clean_cache_attention_profile(
    *,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_token_count: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
    history_stream_visibility: ParallelHistoryStreamVisibility | str | None = None,
) -> PreparedAttentionProfile:
    """Project the dual-slot training profile onto clean cache-write tokens."""

    # The clean-cache writer keeps batch as the real batch dimension. Build a
    # batch-local mask that can broadcast across CFG/batch rows instead of a
    # flattened `[B * tokens, B * tokens]` mask.
    batch_local_latent_shape = (
        1,
        *tuple(int(dim) for dim in latents.shape[1:]),
    )
    batch_local_action_shape = (
        1,
        *tuple(int(dim) for dim in actions.shape[1:]),
    )
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=batch_local_latent_shape,
        action_shape=batch_local_action_shape,
        padded_length=0,
        chunk_size=max(1, int(chunk_size)),
        window_size=max(1, int(window_size)),
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
        text_token_count=int(text_token_count),
        device=latents.device,
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling=CurrentBlockCoupling(
            current_block_coupling
        ).value,
        preserve_video_pretrain_history=bool(
            preserve_video_pretrain_history
        ),
        history_stream_visibility=(
            None
            if history_stream_visibility is None
            else ParallelHistoryStreamVisibility(
                history_stream_visibility
            ).value
        ),
    )
    if (
        profile.self_attention_mask is None
        or profile.cross_attention_mask is None
    ):
        raise ValueError(
            "Joint clean cache attention profile did not materialize dense masks."
        )

    video_token_count = (
        int(latents.shape[2])
        // max(1, int(backbone_config.patch_size_t))
        * (
            int(latents.shape[3])
            // max(1, int(backbone_config.patch_size_h))
        )
        * (
            int(latents.shape[4])
            // max(1, int(backbone_config.patch_size_w))
        )
    )
    action_token_count = (
        int(actions.shape[2])
        * int(actions.shape[3])
        * int(actions.shape[4])
    )
    clean_indices = torch.cat(
        [
            torch.arange(
                video_token_count,
                2 * video_token_count,
                device=profile.self_attention_mask.device,
            ),
            torch.arange(
                2 * video_token_count + action_token_count,
                2 * video_token_count + 2 * action_token_count,
                device=profile.self_attention_mask.device,
            ),
        ],
        dim=0,
    )
    return PreparedAttentionProfile(
        spec=profile.spec,
        self_attention_mask=profile.self_attention_mask.index_select(
            0,
            clean_indices,
        ).index_select(
            1,
            clean_indices,
        ),
        cross_attention_mask=profile.cross_attention_mask.index_select(
            0,
            clean_indices,
        ),
        metadata={
            **profile.metadata,
            "clean_cache_commit": True,
        },
    )


def write_joint_clean_tokens_to_exact_cache(
    *,
    transformer: torch.nn.Module,
    cache_name: str,
    frame_start: int,
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
    update_cache: int,
    backbone_config: SharedVideoTransformerConfig,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling | str,
    preserve_video_pretrain_history: bool,
    history_stream_visibility: ParallelHistoryStreamVisibility | str | None = None,
    video_hidden_context: torch.Tensor | None = None,
    action_hidden_context: torch.Tensor | None = None,
    allow_cache_prefix_during_update_write: bool = False,
) -> None:
    """Embed and write one clean packed video/action chunk to the exact cache."""

    model_dtype = resolve_runtime_module_dtype(transformer)
    video_cache_input = prepare_exact_single_stream_input(
        latents=latents,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=frame_start,
        backbone_config=backbone_config,
        action_mode=False,
    )
    action_cache_input = prepare_exact_single_stream_input(
        latents=actions,
        timestep=0.0,
        text_emb=text_emb,
        frame_st_id=frame_start,
        backbone_config=backbone_config,
        action_mode=True,
        action_channel_mask=action_channel_mask,
    )
    if video_hidden_context is not None:
        video_cache_input["hidden_context"] = video_hidden_context
    if action_hidden_context is not None:
        action_cache_input["hidden_context"] = action_hidden_context
    if use_cfg:
        if negative_text_emb is None:
            raise ValueError(
                "Joint cache commit with CFG requires negative_text_emb."
            )
        video_cache_input = repeat_exact_single_stream_input_for_cfg(
            video_cache_input,
            negative_text_emb=negative_text_emb,
        )
        action_cache_input = repeat_exact_single_stream_input_for_cfg(
            action_cache_input,
            negative_text_emb=negative_text_emb,
        )

    latent_hidden_states = transformer._input_embed(
        video_cache_input["noisy_latents"].to(dtype=model_dtype),
        input_type="latent",
    ).contiguous().clone()
    action_hidden_states = transformer._input_embed(
        action_cache_input["noisy_latents"].to(dtype=model_dtype),
        input_type="action",
    ).contiguous().clone()
    latent_hidden_context = video_cache_input.get("hidden_context")
    if latent_hidden_context is not None:
        if tuple(latent_hidden_context.shape) != tuple(
            latent_hidden_states.shape
        ):
            raise ValueError(
                "Joint clean cache video hidden_context must match embedded hidden states, "
                f"got hidden_context={tuple(latent_hidden_context.shape)}, "
                f"hidden_states={tuple(latent_hidden_states.shape)}."
            )
        latent_hidden_states = (
            latent_hidden_states
            + latent_hidden_context.to(
                device=latent_hidden_states.device,
                dtype=latent_hidden_states.dtype,
            )
        )
    action_hidden_context_input = action_cache_input.get("hidden_context")
    if action_hidden_context_input is not None:
        if tuple(action_hidden_context_input.shape) != tuple(
            action_hidden_states.shape
        ):
            raise ValueError(
                "Joint clean cache action hidden_context must match embedded hidden states, "
                f"got hidden_context={tuple(action_hidden_context_input.shape)}, "
                f"hidden_states={tuple(action_hidden_states.shape)}."
            )
        action_hidden_states = (
            action_hidden_states
            + action_hidden_context_input.to(
                device=action_hidden_states.device,
                dtype=action_hidden_states.dtype,
            )
        )
    hidden_states = torch.cat(
        [latent_hidden_states, action_hidden_states],
        dim=1,
    )
    cache_stream_ids = build_clean_video_action_cache_stream_ids(
        video_token_count=int(latent_hidden_states.shape[1]),
        action_token_count=int(action_hidden_states.shape[1]),
        device=hidden_states.device,
    )

    text_hidden_states = transformer._exact_text_hidden_states(
        video_cache_input["text_emb"],
        dtype=model_dtype,
    ).contiguous().clone()
    latent_grid_id = video_cache_input["grid_id"].contiguous().clone()
    action_grid_id = action_cache_input["grid_id"].contiguous().clone()
    rotary_emb = transformer.rope(
        torch.cat([latent_grid_id, action_grid_id], dim=2)
    )[:, :, None]

    latent_time_steps = video_cache_input["timesteps"].contiguous().clone()
    action_time_steps = action_cache_input["timesteps"].contiguous().clone()
    _, latent_timestep_proj = transformer._time_embed(
        latent_time_steps,
        int(latents.shape[-2]),
        int(latents.shape[-1]),
        dtype=model_dtype,
        action_mode=False,
    )
    _, action_timestep_proj = transformer._time_embed(
        action_time_steps,
        int(actions.shape[-2]),
        int(actions.shape[-1]),
        dtype=model_dtype,
        action_mode=True,
    )
    timestep_proj = torch.cat(
        [latent_timestep_proj, action_timestep_proj],
        dim=1,
    )
    attention_profile = build_joint_clean_cache_attention_profile(
        latents=video_cache_input["noisy_latents"],
        actions=action_cache_input["noisy_latents"],
        text_token_count=int(video_cache_input["text_emb"].shape[1]),
        backbone_config=backbone_config,
        chunk_size=chunk_size,
        window_size=window_size,
        current_block_coupling=current_block_coupling,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
        history_stream_visibility=history_stream_visibility,
    )

    cache_state = transformer._resolve_exact_cache_state(cache_name)
    cache_backend_name = (
        cache_state.backend_name if cache_state is not None else None
    )
    cache_backend_payload = (
        cache_state.backend_payload if cache_state is not None else None
    )
    metadata_previous: list[tuple[Any, dict[str, tuple[bool, Any]]]] = []
    if int(update_cache) != 0 and bool(
        allow_cache_prefix_during_update_write
    ):
        metadata_previous = set_slot_pool_layer_metadata(
            transformer,
            cache_name=cache_name,
            updates={
                SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION: True
            },
        )
    try:
        for layer_index, block in enumerate(transformer.blocks):
            hidden_states, _, _ = block(
                hidden_states,
                encoder_hidden_states=text_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                attention_profile=attention_profile,
                self_attention_cache_backend_name=cache_backend_name,
                self_attention_cache_backend_state=(
                    cache_backend_payload.layer_states[layer_index]
                    if cache_backend_uses_slot_pool(cache_backend_name)
                    and cache_backend_payload is not None
                    and layer_index
                    < len(cache_backend_payload.layer_states)
                    else None
                ),
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=cache_stream_ids,
            )
    finally:
        restore_slot_pool_layer_metadata(metadata_previous)

    if cache_state is not None and cache_backend_uses_slot_pool(
        cache_backend_name
    ):
        materialized_entries = materialize_cache_backend_entries(
            cache_backend_payload
        )
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


def write_exact_cache_chunk(
    *,
    transformer: torch.nn.Module,
    cache_spec: ExactCacheInterfaceSpec,
    cache_name: str,
    frame_start: int,
    backbone_config: SharedVideoTransformerConfig,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    use_cfg: bool,
    action_channel_mask: torch.Tensor | None,
    update_cache: int,
    chunk_size: int,
    window_size: int,
    current_block_coupling: CurrentBlockCoupling
    | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
    preserve_video_pretrain_history: bool = False,
    history_stream_visibility: ParallelHistoryStreamVisibility | str | None = None,
    video_hidden_context: torch.Tensor | None = None,
    action_hidden_context: torch.Tensor | None = None,
    allow_cache_prefix_during_update_write: bool = False,
) -> None:
    """Write one clean chunk through the selected exact-cache interface."""

    if cache_spec.write_mode == ParallelExactCacheWriteMode.JOINT_PACKED:
        write_joint_clean_tokens_to_exact_cache(
            transformer=transformer,
            cache_name=cache_name,
            frame_start=frame_start,
            latents=video_latents,
            actions=action_latents,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            use_cfg=use_cfg,
            action_channel_mask=action_channel_mask,
            update_cache=update_cache,
            backbone_config=backbone_config,
            chunk_size=chunk_size,
            window_size=window_size,
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=preserve_video_pretrain_history,
            history_stream_visibility=history_stream_visibility,
            video_hidden_context=video_hidden_context,
            action_hidden_context=action_hidden_context,
            allow_cache_prefix_during_update_write=(
                allow_cache_prefix_during_update_write
            ),
        )
        return
    if (
        cache_spec.write_mode
        == ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED
    ):
        current_block_coupling = CurrentBlockCoupling(
            current_block_coupling
        )
        chunk_size = max(1, int(chunk_size))
        video_frames = int(video_latents.shape[2])
        action_frames = int(action_latents.shape[2])
        total_frames = max(video_frames, action_frames)

        def _slice_hidden_context(
            hidden_context: torch.Tensor | None,
            *,
            chunk_offset: int,
            frame_count: int,
            tokens_per_frame: int,
        ) -> torch.Tensor | None:
            if hidden_context is None:
                return None
            start = int(chunk_offset) * int(tokens_per_frame)
            end = start + int(frame_count) * int(tokens_per_frame)
            return hidden_context[:, start:end, :]

        def _write_video_cache(
            video_chunk: torch.Tensor,
            *,
            chunk_frame_start: int,
            chunk_offset: int,
        ) -> None:
            video_cache_input = prepare_exact_single_stream_input(
                latents=video_chunk,
                timestep=0.0,
                text_emb=text_emb,
                frame_st_id=chunk_frame_start,
                backbone_config=backbone_config,
                action_mode=False,
            )
            video_context = _slice_hidden_context(
                video_hidden_context,
                chunk_offset=chunk_offset,
                frame_count=int(video_chunk.shape[2]),
                tokens_per_frame=(
                    int(video_chunk.shape[3])
                    // max(1, int(backbone_config.patch_size_h))
                    * (
                        int(video_chunk.shape[4])
                        // max(1, int(backbone_config.patch_size_w))
                    )
                ),
            )
            if video_context is not None:
                video_cache_input["hidden_context"] = video_context
            run_exact_single_stream_forward(
                transformer,
                input_dict=video_cache_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_emb,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )

        def _write_action_cache(
            action_chunk: torch.Tensor,
            *,
            chunk_frame_start: int,
            chunk_offset: int,
        ) -> None:
            action_cache_input = prepare_exact_single_stream_input(
                latents=action_chunk,
                timestep=0.0,
                text_emb=text_emb,
                frame_st_id=chunk_frame_start,
                backbone_config=backbone_config,
                action_mode=True,
                action_channel_mask=action_channel_mask,
            )
            action_context = _slice_hidden_context(
                action_hidden_context,
                chunk_offset=chunk_offset,
                frame_count=int(action_chunk.shape[2]),
                tokens_per_frame=(
                    int(action_chunk.shape[3])
                    * int(action_chunk.shape[4])
                ),
            )
            if action_context is not None:
                action_cache_input["hidden_context"] = action_context
            run_exact_single_stream_forward(
                transformer,
                input_dict=action_cache_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=True,
                guidance_scale=1.0,
                negative_text_emb=negative_text_emb,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )

        for chunk_offset in range(0, total_frames, chunk_size):
            chunk_frame_start = int(frame_start + chunk_offset)
            chunk_end = chunk_offset + chunk_size
            video_chunk = video_latents[
                :,
                :,
                chunk_offset : min(chunk_end, video_frames),
            ]
            action_chunk = action_latents[
                :,
                :,
                chunk_offset : min(chunk_end, action_frames),
            ]
            has_video = int(video_chunk.shape[2]) > 0
            has_action = int(action_chunk.shape[2]) > 0

            if (
                current_block_coupling
                == CurrentBlockCoupling.VIDEO_THEN_ACTION
            ):
                if has_video:
                    _write_video_cache(
                        video_chunk,
                        chunk_frame_start=chunk_frame_start,
                        chunk_offset=chunk_offset,
                    )
                if has_action:
                    _write_action_cache(
                        action_chunk,
                        chunk_frame_start=chunk_frame_start,
                        chunk_offset=chunk_offset,
                    )
            elif (
                current_block_coupling
                == CurrentBlockCoupling.ACTION_THEN_VIDEO
            ):
                if has_action:
                    _write_action_cache(
                        action_chunk,
                        chunk_frame_start=chunk_frame_start,
                        chunk_offset=chunk_offset,
                    )
                if has_video and has_action:
                    metadata_previous = set_slot_pool_layer_metadata(
                        transformer,
                        cache_name=cache_name,
                        updates={
                            SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS: (
                                count_single_stream_action_tokens(
                                    action_chunk
                                )
                            ),
                        },
                    )
                    try:
                        _write_video_cache(
                            video_chunk,
                            chunk_frame_start=chunk_frame_start,
                            chunk_offset=chunk_offset,
                        )
                    finally:
                        restore_slot_pool_layer_metadata(metadata_previous)
                elif has_video:
                    _write_video_cache(
                        video_chunk,
                        chunk_frame_start=chunk_frame_start,
                        chunk_offset=chunk_offset,
                    )
            elif (
                current_block_coupling
                == CurrentBlockCoupling.DECOUPLED_SAME_STEP
            ):
                overlap_frames = min(
                    int(video_chunk.shape[2]),
                    int(action_chunk.shape[2]),
                )
                if overlap_frames > 0:
                    write_joint_clean_tokens_to_exact_cache(
                        transformer=transformer,
                        cache_name=cache_name,
                        frame_start=chunk_frame_start,
                        latents=video_chunk[:, :, :overlap_frames],
                        actions=action_chunk[:, :, :overlap_frames],
                        text_emb=text_emb,
                        negative_text_emb=negative_text_emb,
                        use_cfg=use_cfg,
                        action_channel_mask=action_channel_mask,
                        update_cache=update_cache,
                        backbone_config=backbone_config,
                        chunk_size=chunk_size,
                        window_size=window_size,
                        current_block_coupling=current_block_coupling,
                        preserve_video_pretrain_history=(
                            preserve_video_pretrain_history
                        ),
                        history_stream_visibility=(
                            history_stream_visibility
                        ),
                        video_hidden_context=_slice_hidden_context(
                            video_hidden_context,
                            chunk_offset=chunk_offset,
                            frame_count=overlap_frames,
                            tokens_per_frame=(
                                int(video_chunk.shape[3])
                                // max(
                                    1,
                                    int(backbone_config.patch_size_h),
                                )
                                * (
                                    int(video_chunk.shape[4])
                                    // max(
                                        1,
                                        int(backbone_config.patch_size_w),
                                    )
                                )
                            ),
                        ),
                        action_hidden_context=_slice_hidden_context(
                            action_hidden_context,
                            chunk_offset=chunk_offset,
                            frame_count=overlap_frames,
                            tokens_per_frame=(
                                int(action_chunk.shape[3])
                                * int(action_chunk.shape[4])
                            ),
                        ),
                        allow_cache_prefix_during_update_write=(
                            allow_cache_prefix_during_update_write
                        ),
                    )
                if int(video_chunk.shape[2]) > overlap_frames:
                    _write_video_cache(
                        video_chunk[:, :, overlap_frames:],
                        chunk_frame_start=(
                            chunk_frame_start + overlap_frames
                        ),
                        chunk_offset=chunk_offset + overlap_frames,
                    )
                if int(action_chunk.shape[2]) > overlap_frames:
                    _write_action_cache(
                        action_chunk[:, :, overlap_frames:],
                        chunk_frame_start=(
                            chunk_frame_start + overlap_frames
                        ),
                        chunk_offset=chunk_offset + overlap_frames,
                    )
            else:
                raise ValueError(
                    "Single-stream staged cache writes only support ordered staged "
                    "or decoupled couplings, "
                    f"got {current_block_coupling.value!r}."
                )
        return
    raise ValueError(
        f"Unsupported exact cache write_mode: {cache_spec.write_mode!r}"
    )


def summarize_slot_pool_cache_state(
    transformer: torch.nn.Module,
    cache_name: str,
) -> dict[str, int] | None:
    """Summarize the first slot-pool layer for rollout diagnostics."""

    if not hasattr(transformer, "_resolve_exact_cache_state"):
        return None
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    if cache_state is None or not cache_backend_uses_slot_pool(
        cache_state.backend_name
    ):
        return None
    backend_payload = cache_state.backend_payload
    if backend_payload is None or not getattr(
        backend_payload,
        "layer_states",
        None,
    ):
        return None
    layer_state = backend_payload.layer_states[0]
    if layer_state.slot_mask is None:
        return None
    cached_tokens = int(layer_state.slot_mask.sum().item())
    prediction_tokens = (
        int(
            layer_state.prediction_mask[layer_state.slot_mask]
            .sum()
            .item()
        )
        if layer_state.prediction_mask is not None
        else 0
    )
    return {
        "cached_tokens": cached_tokens,
        "prediction_tokens": prediction_tokens,
        "total_slots": int(layer_state.slot_mask.numel()),
    }


__all__ = [
    "build_joint_clean_cache_attention_mask",
    "build_joint_clean_cache_attention_profile",
    "summarize_slot_pool_cache_state",
    "write_exact_cache_chunk",
    "write_joint_clean_tokens_to_exact_cache",
]
