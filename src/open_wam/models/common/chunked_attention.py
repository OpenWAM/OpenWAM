"""Chunk-aware dense and FlexAttention profile construction."""

from __future__ import annotations

import torch

from open_wam.configs import HistoryStreamVisibility
from open_wam.models.common.attention_backends import (
    _resolve_compiled_create_block_mask,
    create_block_mask,
)
from open_wam.models.common.attention_contracts import (
    VIDEO_THEN_ACTION_COUPLING,
    AttentionProfileSpec,
    PreparedAttentionProfile,
    chunked_temporal_exact_profile_name_for_coupling,
    normalize_chunked_temporal_exact_coupling,
    normalize_conditional_history_policy,
    normalize_history_stream_visibility,
)
from open_wam.models.common.chunked_attention_visibility import (
    _build_chunked_cross_attention_visibility,
    _build_chunked_self_attention_visibility,
    _effective_frame_ids_for_singleton_cutoff,
)
from open_wam.models.common.packed_token_layout import (
    build_exact_video_action_token_layout,
)


def build_chunked_text_context_cross_attention_mask(
    *,
    query_chunk_ids: torch.Tensor,
    batch_size: int,
    text_token_count: int,
    base_text_token_count: int,
    proprio_context_token_count: int,
    global_suffix_token_count: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Build a query-dependent mask for deprecated text-space proprio tokens.

    Text tokens are visible to every query. Deprecated appended proprio tokens
    are visible only to queries from the matching local chunk. Optional suffix
    tokens, such as learned mode tokens in legacy packed text layouts, are
    visible to every query.
    """

    if query_chunk_ids.ndim != 1:
        raise ValueError(
            "Chunked text context masks expect query_chunk_ids with shape [query_tokens], "
            f"got {tuple(query_chunk_ids.shape)}."
        )
    resolved_batch_size = int(batch_size)
    resolved_text_token_count = int(text_token_count)
    resolved_base_text_token_count = int(base_text_token_count)
    resolved_proprio_context_token_count = int(proprio_context_token_count)
    resolved_global_suffix_token_count = int(global_suffix_token_count)
    if resolved_batch_size <= 0:
        raise ValueError(f"Expected positive batch_size, got {batch_size}.")
    if (
        resolved_base_text_token_count < 0
        or resolved_proprio_context_token_count < 0
        or resolved_global_suffix_token_count < 0
    ):
        raise ValueError(
            "Context token counts must be non-negative, "
            f"got base={base_text_token_count}, proprio={proprio_context_token_count}, "
            f"global_suffix={global_suffix_token_count}."
        )
    if (
        resolved_base_text_token_count
        + resolved_proprio_context_token_count
        + resolved_global_suffix_token_count
        != resolved_text_token_count
    ):
        raise ValueError(
            "Context token counts must sum to text_token_count, "
            f"got base={base_text_token_count}, proprio={proprio_context_token_count}, "
            f"global_suffix={global_suffix_token_count}, "
            f"text={text_token_count}."
        )
    query_chunk_ids = query_chunk_ids.to(device=device, dtype=torch.long)
    text_position = torch.arange(
        resolved_text_token_count, device=device, dtype=torch.long
    )
    base_text_visible = text_position < resolved_base_text_token_count
    proprio_index = text_position - resolved_base_text_token_count
    proprio_visible = (
        (proprio_index[None, :] >= 0)
        & (proprio_index[None, :] < resolved_proprio_context_token_count)
        & (proprio_index[None, :] == query_chunk_ids[:, None])
    )
    global_suffix_start = (
        resolved_base_text_token_count + resolved_proprio_context_token_count
    )
    global_suffix_visible = text_position >= global_suffix_start
    mask = base_text_visible[None, :] | proprio_visible | global_suffix_visible[None, :]
    return mask[None, :, :].expand(resolved_batch_size, -1, -1).contiguous()


def build_chunked_temporal_exact_attention_profile(
    *,
    latent_shape: tuple[int, int, int, int, int],
    action_shape: tuple[int, int, int, int, int],
    padded_length: int,
    chunk_size: int,
    window_size: int,
    patch_size: tuple[int, int, int],
    text_token_count: int,
    base_text_token_count: int | None = None,
    proprio_context_token_count: int = 0,
    chunk_origin_frame: int = 0,
    device: torch.device,
    action_context_mask: torch.Tensor | None = None,
    build_dense_masks: bool = False,
    build_flex_masks: bool = False,
    current_block_coupling: str | None = None,
    history_stream_visibility: HistoryStreamVisibility | str | None = None,
    prefix_condition_frames: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
) -> PreparedAttentionProfile:
    # History visibility applies only to past chunks. Same-chunk cross-stream
    # visibility remains owned by the current-block coupling program.
    if current_block_coupling is None:
        current_block_coupling = VIDEO_THEN_ACTION_COUPLING
    current_block_coupling = normalize_chunked_temporal_exact_coupling(
        current_block_coupling
    )
    resolved_history_stream_visibility = HistoryStreamVisibility(
        normalize_history_stream_visibility(history_stream_visibility)
    )
    resolved_conditional_history_policy = normalize_conditional_history_policy(
        conditional_history_policy
    )
    chunk_origin_frame = int(chunk_origin_frame)
    prefix_condition_frames = max(0, int(prefix_condition_frames))

    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    _, _, action_frames, action_height, action_width = action_shape
    patch_t, patch_h, patch_w = patch_size
    text_token_count = int(text_token_count)
    resolved_base_text_token_count = (
        text_token_count
        if base_text_token_count is None
        else int(base_text_token_count)
    )
    resolved_proprio_context_token_count = int(proprio_context_token_count)
    if resolved_proprio_context_token_count < 0:
        raise ValueError(
            "proprio_context_token_count must be non-negative, "
            f"got {resolved_proprio_context_token_count}."
        )
    if (
        resolved_base_text_token_count < 0
        or resolved_base_text_token_count > text_token_count
    ):
        raise ValueError(
            "base_text_token_count must be within the per-sample text token count, "
            f"got base_text_token_count={resolved_base_text_token_count}, text_token_count={text_token_count}."
        )
    if (
        resolved_base_text_token_count + resolved_proprio_context_token_count
        > text_token_count
    ):
        raise ValueError(
            "base_text_token_count + proprio_context_token_count cannot exceed text_token_count, "
            f"got base={resolved_base_text_token_count}, "
            f"proprio={resolved_proprio_context_token_count}, text={text_token_count}."
        )

    layout = build_exact_video_action_token_layout(
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        action_frames=action_frames,
        action_height=action_height,
        action_width=action_width,
        patch_size=patch_size,
        chunk_size=chunk_size,
        chunk_origin_frame=chunk_origin_frame,
        current_block_coupling=current_block_coupling,
        device=device,
        action_context_mask=action_context_mask,
        prefix_condition_frames=prefix_condition_frames,
        singleton_chunk_frame=singleton_chunk_frame,
    )
    layout = layout.with_padding(padded_length)
    latent_token_count = (
        int(batch_size)
        * int(latent_frames // patch_t)
        * int(latent_height // patch_h)
        * int(latent_width // patch_w)
    )
    action_token_count = (
        int(batch_size) * int(action_frames) * int(action_height) * int(action_width)
    )
    action_token_valid = layout.valid_as_kv[
        2 * latent_token_count : 2 * latent_token_count + action_token_count
    ]
    invalid_action_token_count = int((~action_token_valid).sum().item())
    action_context_valid_tokens: tuple[bool, ...] | None = (
        tuple(bool(value) for value in action_token_valid.detach().cpu().tolist())
        if action_context_mask is not None
        else None
    )

    seq_ids = layout.seq_id
    block_ids = layout.block_id
    chunk_ids = layout.chunk_id
    noise_ids = layout.noise_id
    stream_ids = layout.stream_id
    token_valid_as_query = layout.valid_as_query
    token_valid_as_kv = layout.valid_as_kv

    text_seq_ids = (
        torch.arange(batch_size, device=device)[:, None]
        .expand(-1, text_token_count)
        .flatten()
    )
    text_context_positions = (
        torch.arange(text_token_count, device=device)[None, :]
        .expand(batch_size, -1)
        .flatten()
    )

    self_attention_mask = None
    cross_attention_mask = None
    if build_dense_masks:
        q_seq = seq_ids[:, None]
        kv_seq = seq_ids[None, :]
        q_block_id = block_ids[:, None]
        kv_block_id = block_ids[None, :]
        q_noise = noise_ids[:, None]
        kv_noise = noise_ids[None, :]
        q_stream = stream_ids[:, None]
        kv_stream = stream_ids[None, :]
        q_chunk = chunk_ids[:, None]
        kv_chunk = chunk_ids[None, :]
        q_valid = token_valid_as_query[:, None]
        kv_valid = token_valid_as_kv[None, :]
        effective_frame_ids = _effective_frame_ids_for_singleton_cutoff(
            layout.frame_id,
            stream_ids,
            prefix_condition_frames=prefix_condition_frames,
            singleton_chunk_frame=singleton_chunk_frame,
        )
        q_effective_frame = effective_frame_ids[:, None]
        kv_effective_frame = effective_frame_ids[None, :]
        self_attention_mask = _build_chunked_self_attention_visibility(
            q_seq=q_seq,
            kv_seq=kv_seq,
            q_block_id=q_block_id,
            kv_block_id=kv_block_id,
            q_chunk=q_chunk,
            kv_chunk=kv_chunk,
            q_noise=q_noise,
            kv_noise=kv_noise,
            q_stream=q_stream,
            kv_stream=kv_stream,
            q_effective_frame=q_effective_frame,
            kv_effective_frame=kv_effective_frame,
            q_valid=q_valid,
            kv_valid=kv_valid,
            window_size=window_size,
            chunk_size=chunk_size,
            chunk_origin_frame=chunk_origin_frame,
            prefix_condition_frames=prefix_condition_frames,
            singleton_chunk_frame=singleton_chunk_frame,
            current_block_coupling=current_block_coupling,
            history_stream_visibility=resolved_history_stream_visibility,
            conditional_history_policy=resolved_conditional_history_policy,
        )
        cross_attention_mask = _build_chunked_cross_attention_visibility(
            q_seq=seq_ids[:, None],
            text_seq=text_seq_ids[None, :],
            q_chunk=q_chunk,
            text_position=text_context_positions[None, :],
            q_valid=token_valid_as_query[:, None],
            base_text_token_count=resolved_base_text_token_count,
            proprio_context_token_count=resolved_proprio_context_token_count,
        )

    self_attention_block_mask = None
    cross_attention_block_mask = None
    if build_flex_masks and create_block_mask is not None:
        seq_ids_flex = seq_ids.to(device=device, dtype=torch.long)
        block_ids_flex = block_ids.to(device=device, dtype=torch.long)
        chunk_ids_flex = chunk_ids.to(device=device, dtype=torch.long)
        noise_ids_flex = noise_ids.to(device=device, dtype=torch.long)
        stream_ids_flex = stream_ids.to(device=device, dtype=torch.long)
        effective_frame_ids_flex = _effective_frame_ids_for_singleton_cutoff(
            layout.frame_id,
            stream_ids,
            prefix_condition_frames=prefix_condition_frames,
            singleton_chunk_frame=singleton_chunk_frame,
        ).to(device=device, dtype=torch.long)
        token_valid_as_query_flex = token_valid_as_query.to(
            device=device, dtype=torch.bool
        )
        token_valid_as_kv_flex = token_valid_as_kv.to(device=device, dtype=torch.bool)
        text_seq_ids_flex = text_seq_ids.to(device=device, dtype=torch.long)
        text_context_positions_flex = text_context_positions.to(
            device=device, dtype=torch.long
        )

        def self_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return _build_chunked_self_attention_visibility(
                q_seq=seq_ids_flex[q_idx],
                kv_seq=seq_ids_flex[kv_idx],
                q_block_id=block_ids_flex[q_idx],
                kv_block_id=block_ids_flex[kv_idx],
                q_chunk=chunk_ids_flex[q_idx],
                kv_chunk=chunk_ids_flex[kv_idx],
                q_noise=noise_ids_flex[q_idx],
                kv_noise=noise_ids_flex[kv_idx],
                q_stream=stream_ids_flex[q_idx],
                kv_stream=stream_ids_flex[kv_idx],
                q_effective_frame=effective_frame_ids_flex[q_idx],
                kv_effective_frame=effective_frame_ids_flex[kv_idx],
                q_valid=token_valid_as_query_flex[q_idx],
                kv_valid=token_valid_as_kv_flex[kv_idx],
                window_size=window_size,
                chunk_size=chunk_size,
                chunk_origin_frame=chunk_origin_frame,
                prefix_condition_frames=prefix_condition_frames,
                singleton_chunk_frame=singleton_chunk_frame,
                current_block_coupling=current_block_coupling,
                history_stream_visibility=resolved_history_stream_visibility,
                conditional_history_policy=resolved_conditional_history_policy,
            )

        def cross_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            return _build_chunked_cross_attention_visibility(
                q_seq=seq_ids_flex[q_idx],
                text_seq=text_seq_ids_flex[kv_idx],
                q_chunk=chunk_ids_flex[q_idx],
                text_position=text_context_positions_flex[kv_idx],
                q_valid=token_valid_as_query_flex[q_idx],
                base_text_token_count=resolved_base_text_token_count,
                proprio_context_token_count=resolved_proprio_context_token_count,
            )

        total_seq_len = int(seq_ids.numel())
        total_text_len = int(text_seq_ids.numel())
        compiled_create_block_mask = _resolve_compiled_create_block_mask()
        block_mask_builder = compiled_create_block_mask or create_block_mask
        self_attention_block_mask = block_mask_builder(
            self_mask_mod,
            1,
            1,
            total_seq_len,
            total_seq_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )
        cross_attention_block_mask = block_mask_builder(
            cross_mask_mod,
            1,
            1,
            total_seq_len,
            total_text_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )

    return PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name=chunked_temporal_exact_profile_name_for_coupling(
                current_block_coupling
            ),
            family="chunked_exact",
            backend="flex_or_sdpa",
        ),
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        self_attention_block_mask=self_attention_block_mask,
        cross_attention_block_mask=cross_attention_block_mask,
        metadata={
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "window_size": int(window_size),
            "latent_shape": tuple(int(v) for v in latent_shape),
            "action_shape": tuple(int(v) for v in action_shape),
            "padded_length": int(padded_length),
            "text_token_count": int(text_token_count),
            "base_text_token_count": int(resolved_base_text_token_count),
            "proprio_context_token_count": int(resolved_proprio_context_token_count),
            "chunk_origin_frame": int(chunk_origin_frame),
            "singleton_chunk_frame": None
            if singleton_chunk_frame is None
            else int(singleton_chunk_frame),
            "invalid_action_context_tokens": int(invalid_action_token_count),
            "action_context_valid_tokens": action_context_valid_tokens,
            "current_block_coupling": current_block_coupling,
            "history_stream_visibility": resolved_history_stream_visibility,
            "prefix_condition_frames": int(prefix_condition_frames),
            "conditional_history_policy": resolved_conditional_history_policy,
        },
    )


def build_lingbot_chunked_exact_attention_profile(**kwargs) -> PreparedAttentionProfile:
    return build_chunked_temporal_exact_attention_profile(**kwargs)


__all__ = [
    "build_chunked_temporal_exact_attention_profile",
    "build_chunked_text_context_cross_attention_mask",
    "build_lingbot_chunked_exact_attention_profile",
]
