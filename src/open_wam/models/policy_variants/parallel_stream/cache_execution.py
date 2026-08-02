"""Parallel exact-cache write orchestration and compatibility exports.

Specialized attention, clean-write, and diagnostic behavior lives in role
owners. This module retains generic cache-write dispatch and the historical
import, wildcard, pickle, and monkeypatch surface.
"""

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
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.common.cache_backend_contracts import (
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    cache_backend_uses_slot_pool,
)
from open_wam.models.common.cache_backend_lifecycle import (
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
from .cache_attention import (
    build_joint_clean_cache_attention_mask,
    build_joint_clean_cache_attention_profile,
)
from .cache_diagnostics import summarize_slot_pool_cache_state
from .clean_cache_write import write_joint_clean_tokens_to_exact_cache

(
    Any,
    CacheState,
    CurrentBlockCoupling,
    ExactCacheInterfaceSpec,
    ParallelExactCacheWriteMode,
    ParallelHistoryStreamVisibility,
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    SharedVideoTransformerConfig,
    build_chunked_temporal_exact_attention_profile,
    build_clean_video_action_cache_stream_ids,
    cache_backend_uses_slot_pool,
    count_single_stream_action_tokens,
    materialize_cache_backend_entries,
    prepare_exact_single_stream_input,
    repeat_exact_single_stream_input_for_cfg,
    resolve_runtime_module_dtype,
    restore_slot_pool_layer_metadata,
    run_exact_single_stream_forward,
    set_slot_pool_layer_metadata,
    torch,
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


__all__ = [
    "build_joint_clean_cache_attention_mask",
    "build_joint_clean_cache_attention_profile",
    "summarize_slot_pool_cache_state",
    "write_exact_cache_chunk",
    "write_joint_clean_tokens_to_exact_cache",
]
