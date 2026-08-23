from __future__ import annotations

import torch


def inject_deprecated_proprio_text_context(
    transformer: torch.nn.Module,
    *,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    proprio_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append the deprecated text-space proprio token to both CFG branches."""

    if proprio_state is None:
        return text_emb, negative_text_emb
    append = getattr(transformer, "append_proprio_context_tokens", None)
    if not callable(append):
        raise ValueError(
            "Deprecated text-space proprio token mode requires the runtime transformer "
            "to support proprio context appending."
        )
    text_emb = append(text_emb, proprio_state)
    if negative_text_emb is not None:
        negative_text_emb = append(negative_text_emb, proprio_state)
    return text_emb, negative_text_emb


def build_single_stream_hidden_proprio_context(
    transformer: torch.nn.Module,
    *,
    proprio_state: torch.Tensor | None,
    stream_latents: torch.Tensor,
    action_mode: bool,
) -> torch.Tensor | None:
    """Expand one proprio anchor over the tokens in a single stream."""

    if proprio_state is None:
        return None
    if proprio_state.ndim == 3:
        proprio_state = proprio_state[:, -1, :]
    if proprio_state.ndim != 2:
        raise ValueError(
            "Single-stream proprio context expects state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(proprio_state.shape)}."
        )
    batch_size, _, num_frames, _, _ = stream_latents.shape
    if int(proprio_state.shape[0]) != batch_size:
        raise ValueError(
            "Single-stream proprio batch mismatch, "
            f"got proprio batch {proprio_state.shape[0]} and stream batch {batch_size}."
        )
    frame_state = proprio_state[:, None, :].expand(-1, int(num_frames), -1)
    return _encode_single_stream_hidden_proprio_context(
        transformer,
        frame_state=frame_state,
        stream_latents=stream_latents,
        action_mode=action_mode,
    )


def build_single_stream_hidden_proprio_history_context(
    transformer: torch.nn.Module,
    *,
    proprio_history: torch.Tensor | None,
    stream_latents: torch.Tensor,
    action_mode: bool,
) -> torch.Tensor | None:
    """Encode frame-aligned proprio history for one cache-warmup stream."""

    if proprio_history is None:
        return None
    if proprio_history.ndim != 3:
        raise ValueError(
            "Frame-aligned proprio history expects shape [B, T, state_dim], "
            f"got {tuple(proprio_history.shape)}."
        )
    batch_size, _, num_frames, _, _ = stream_latents.shape
    if tuple(proprio_history.shape[:2]) != (batch_size, int(num_frames)):
        raise ValueError(
            "Frame-aligned proprio history must match the stream batch and frame count, "
            f"got history={tuple(proprio_history.shape)} and "
            f"stream={tuple(stream_latents.shape)}."
        )
    return _encode_single_stream_hidden_proprio_context(
        transformer,
        frame_state=proprio_history,
        stream_latents=stream_latents,
        action_mode=action_mode,
    )


def _encode_single_stream_hidden_proprio_context(
    transformer: torch.nn.Module,
    *,
    frame_state: torch.Tensor,
    stream_latents: torch.Tensor,
    action_mode: bool,
) -> torch.Tensor:
    """Project frame states and expand them over one stream's token geometry."""

    encode = getattr(transformer, "encode_proprio_hidden_context", None)
    if not callable(encode):  # pragma: no cover - validated by public entry points
        raise ValueError(
            "Per-chunk proprio mode requires `encode_proprio_hidden_context` on the runtime transformer."
        )
    _, _, _, height, width = stream_latents.shape
    frame_context = encode(
        frame_state,
        device=stream_latents.device,
        dtype=stream_latents.dtype,
    )
    if action_mode:
        tokens_per_frame = int(height) * int(width)
    else:
        patch_t, patch_h, patch_w = transformer.patch_size
        frame_context = frame_context[:, :: int(patch_t), :]
        tokens_per_frame = (int(height) // int(patch_h)) * (int(width) // int(patch_w))
    return frame_context.repeat_interleave(tokens_per_frame, dim=1)


__all__ = [
    "build_single_stream_hidden_proprio_context",
    "build_single_stream_hidden_proprio_history_context",
    "inject_deprecated_proprio_text_context",
]
