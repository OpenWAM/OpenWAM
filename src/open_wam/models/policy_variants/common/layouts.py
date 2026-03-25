from __future__ import annotations

import torch
import torch.nn.functional as F

from open_wam.models.video_backbone.contracts import TokenGridMetadata


def tokens_to_frame_major(tokens: torch.Tensor, token_grid: TokenGridMetadata) -> torch.Tensor:
    """Reshape `[B, seq, D]` video tokens into `[B, T, tokens_per_frame, D]`."""

    batch_size, seq_len, hidden_size = tokens.shape
    expected_seq = token_grid.num_frames * token_grid.tokens_per_frame
    if seq_len != expected_seq:
        raise ValueError(
            f"Expected video token sequence length {expected_seq}, got {seq_len}."
        )
    return tokens.view(batch_size, token_grid.num_frames, token_grid.tokens_per_frame, hidden_size)


def pool_frame_tokens(frame_tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool `[B, T, N_patch, D]` into `[B, T, D]`."""

    if mode == "mean":
        return frame_tokens.mean(dim=2)
    if mode == "max":
        return frame_tokens.max(dim=2).values
    raise ValueError(f"Unsupported frame pooling mode '{mode}'.")


def align_sequence_length(features: torch.Tensor, target_length: int) -> torch.Tensor:
    """Interpolate `[B, T, D]` features to `[B, target_length, D]`."""

    if features.shape[1] == target_length:
        return features
    return F.interpolate(
        features.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def expand_previous_action(
    previous_action: torch.Tensor | None,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand optional previous-action context to `[B, H_action, D_action]`."""

    if previous_action is None:
        return torch.zeros(batch_size, action_horizon, action_dim, device=device, dtype=dtype)
    if previous_action.ndim == 3:
        if previous_action.shape[1] != action_horizon or previous_action.shape[2] != action_dim:
            raise ValueError(
                "Expected previous action tensor with shape "
                f"[B, {action_horizon}, {action_dim}], got {tuple(previous_action.shape)}"
            )
        return previous_action.to(device=device, dtype=dtype)
    if previous_action.ndim == 2:
        if previous_action.shape[1] != action_dim:
            raise ValueError(
                f"Expected previous action dim {action_dim}, got {previous_action.shape[1]}"
            )
        return previous_action[:, None, :].to(device=device, dtype=dtype).expand(-1, action_horizon, -1)
    raise ValueError(
        "Expected previous action tensor with shape [B, D_action] or [B, H_action, D_action], "
        f"got {tuple(previous_action.shape)}"
    )
