from __future__ import annotations

import torch

from open_wam.models.video_backbone.contracts import TokenGridMetadata


def slice_token_grid_frames(
    token_grid: TokenGridMetadata,
    *,
    num_frames: int,
) -> TokenGridMetadata:
    """Return a frame-sliced view of a full video token grid.

    Joint video+action diffusion variants often predict only the future-video
    suffix while reusing the clean first frame as context. The token geometry
    for those future frames is the same patch grid, but with a shorter temporal
    span and therefore a shorter flattened token sequence.
    """

    if num_frames <= 0 or num_frames > token_grid.num_frames:
        raise ValueError(
            f"Expected `num_frames` in [1, {token_grid.num_frames}], got {num_frames}."
        )
    return TokenGridMetadata(
        num_frames=num_frames,
        latent_height=token_grid.latent_height,
        latent_width=token_grid.latent_width,
        patch_size=token_grid.patch_size,
        patches_per_frame_h=token_grid.patches_per_frame_h,
        patches_per_frame_w=token_grid.patches_per_frame_w,
        tokens_per_frame=token_grid.tokens_per_frame,
        sequence_length=num_frames * token_grid.tokens_per_frame,
    )


def unpatchify_video_tokens(
    token_predictions: torch.Tensor,
    *,
    token_grid: TokenGridMetadata,
    latent_channels: int,
) -> torch.Tensor:
    """Restore `[B, T_tokens, patch_dim]` predictions to `[B, C, F, H, W]`.

    The frontend patchifies video latents frame-major with patch size
    `(patch_t, patch_h, patch_w)`. Joint video+action diffusion variants need
    the inverse map so video flow predictions can be compared to latent-space
    diffusion targets and can be stepped by the scheduler during inference.
    """

    if token_predictions.ndim != 3:
        raise ValueError(
            "Expected token predictions with shape [B, T_video, patch_dim], "
            f"got {tuple(token_predictions.shape)}."
        )
    batch_size, seq_len, patch_dim = token_predictions.shape
    if seq_len != token_grid.sequence_length:
        raise ValueError(
            f"Expected video token sequence length {token_grid.sequence_length}, got {seq_len}."
        )
    patch_t, patch_h, patch_w = token_grid.patch_size
    expected_patch_dim = latent_channels * patch_t * patch_h * patch_w
    if patch_dim != expected_patch_dim:
        raise ValueError(
            f"Expected patch dim {expected_patch_dim}, got {patch_dim}."
        )
    post_patch_frames = token_grid.num_frames // patch_t
    post_patch_height = token_grid.latent_height // patch_h
    post_patch_width = token_grid.latent_width // patch_w
    patches = token_predictions.view(
        batch_size,
        post_patch_frames,
        post_patch_height,
        post_patch_width,
        latent_channels,
        patch_t,
        patch_h,
        patch_w,
    )
    latents = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        batch_size,
        latent_channels,
        token_grid.num_frames,
        token_grid.latent_height,
        token_grid.latent_width,
    )
    return latents
