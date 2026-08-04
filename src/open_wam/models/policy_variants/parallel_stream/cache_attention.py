"""Clean joint-cache attention layout construction."""

from __future__ import annotations

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    HistoryStreamVisibility,
)
from open_wam.models.common import (
    PreparedAttentionProfile,
    build_chunked_temporal_exact_attention_profile,
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
    history_stream_visibility: HistoryStreamVisibility | str | None = None,
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
    history_stream_visibility: HistoryStreamVisibility | str | None = None,
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
            else HistoryStreamVisibility(
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


__all__ = [
    "build_joint_clean_cache_attention_mask",
    "build_joint_clean_cache_attention_profile",
]
