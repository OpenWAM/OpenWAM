"""Clean packed video/action writes for the parallel exact cache."""

from __future__ import annotations

from typing import Any

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    HistoryStreamVisibility,
)
from open_wam.models.common.cache_backend_contracts import (
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
)

from .cache_attention import build_joint_clean_cache_attention_profile
from .exact_cache import (
    build_clean_video_action_cache_stream_ids,
    restore_slot_pool_layer_metadata,
    set_slot_pool_layer_metadata,
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
    history_stream_visibility: HistoryStreamVisibility | str | None = None,
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
            raise ValueError("Joint cache commit with CFG requires negative_text_emb.")
        video_cache_input = repeat_exact_single_stream_input_for_cfg(
            video_cache_input,
            negative_text_emb=negative_text_emb,
        )
        action_cache_input = repeat_exact_single_stream_input_for_cfg(
            action_cache_input,
            negative_text_emb=negative_text_emb,
        )

    latent_hidden_states = (
        transformer._input_embed(
            video_cache_input["noisy_latents"].to(dtype=model_dtype),
            input_type="latent",
        )
        .contiguous()
        .clone()
    )
    action_hidden_states = (
        transformer._input_embed(
            action_cache_input["noisy_latents"].to(dtype=model_dtype),
            input_type="action",
        )
        .contiguous()
        .clone()
    )
    latent_hidden_context = video_cache_input.get("hidden_context")
    if latent_hidden_context is not None:
        if tuple(latent_hidden_context.shape) != tuple(latent_hidden_states.shape):
            raise ValueError(
                "Joint clean cache video hidden_context must match embedded hidden states, "
                f"got hidden_context={tuple(latent_hidden_context.shape)}, "
                f"hidden_states={tuple(latent_hidden_states.shape)}."
            )
        latent_hidden_states = latent_hidden_states + latent_hidden_context.to(
            device=latent_hidden_states.device,
            dtype=latent_hidden_states.dtype,
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
        action_hidden_states = action_hidden_states + action_hidden_context_input.to(
            device=action_hidden_states.device,
            dtype=action_hidden_states.dtype,
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

    text_hidden_states = (
        transformer._exact_text_hidden_states(
            video_cache_input["text_emb"],
            dtype=model_dtype,
        )
        .contiguous()
        .clone()
    )
    latent_grid_id = video_cache_input["grid_id"].contiguous().clone()
    action_grid_id = action_cache_input["grid_id"].contiguous().clone()
    rotary_emb = transformer.rope(torch.cat([latent_grid_id, action_grid_id], dim=2))[
        :, :, None
    ]

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
        history_stream_visibility=history_stream_visibility,
    )

    cache_state = transformer.get_runtime_cache_state(cache_name)
    cache_backend_name = cache_state.backend_name if cache_state is not None else None
    cache_backend_payload = (
        cache_state.backend_payload if cache_state is not None else None
    )
    metadata_previous: list[tuple[Any, dict[str, tuple[bool, Any]]]] = []
    if int(update_cache) != 0 and bool(allow_cache_prefix_during_update_write):
        metadata_previous = set_slot_pool_layer_metadata(
            transformer,
            cache_name=cache_name,
            updates={SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION: True},
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
                    and layer_index < len(cache_backend_payload.layer_states)
                    else None
                ),
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=cache_stream_ids,
            )
    finally:
        restore_slot_pool_layer_metadata(metadata_previous)

    if cache_state is not None and cache_backend_uses_slot_pool(cache_backend_name):
        materialized_entries = materialize_cache_backend_entries(cache_backend_payload)
        transformer.replace_runtime_cache_state(
            cache_name,
            CacheState(
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
            ),
        )


__all__ = ["write_joint_clean_tokens_to_exact_cache"]
