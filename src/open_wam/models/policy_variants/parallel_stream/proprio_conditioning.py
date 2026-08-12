from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange


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
    encode = getattr(transformer, "encode_proprio_hidden_context", None)
    if not callable(encode):
        raise ValueError(
            "Per-chunk proprio mode requires `encode_proprio_hidden_context` on the runtime transformer."
        )
    if proprio_state.ndim == 3:
        proprio_state = proprio_state[:, -1, :]
    if proprio_state.ndim != 2:
        raise ValueError(
            "Single-stream proprio context expects state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(proprio_state.shape)}."
        )
    batch_size, _, num_frames, height, width = stream_latents.shape
    if int(proprio_state.shape[0]) != batch_size:
        raise ValueError(
            "Single-stream proprio batch mismatch, "
            f"got proprio batch {proprio_state.shape[0]} and stream batch {batch_size}."
        )
    frame_state = proprio_state[:, None, :].expand(-1, int(num_frames), -1)
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


def apply_parallel_chunk_proprio_context(
    transformer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    split_list: list[int] | tuple[int, ...],
    input_dict: dict[str, Any],
) -> torch.Tensor:
    """Add boundary-aligned proprio context to packed video/action streams.

    Learned state projection remains a VisualTower-core responsibility through
    ``encode_proprio_hidden_context``. This policy-local contract owns only the
    mapping from frame- or chunk-level state to the parallel token layout.
    """

    proprio_state = input_dict.get("per_chunk_proprio_state")
    if proprio_state is None:
        return hidden_states
    if not isinstance(proprio_state, torch.Tensor):
        raise ValueError("`per_chunk_proprio_state` must be a tensor.")
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    if not isinstance(latent_dict, dict) or not isinstance(action_dict, dict):
        raise ValueError("Per-chunk proprio context requires latent_dict and action_dict payloads.")
    latent_shape = tuple(int(dim) for dim in latent_dict["noisy_latents"].shape)
    action_shape = tuple(int(dim) for dim in action_dict["noisy_latents"].shape)
    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    action_batch, _, action_frames, action_height, action_width = action_shape
    context_frame_count = latent_frames
    if batch_size != action_batch:
        raise ValueError(
            "Per-chunk proprio context expects matching video/action batches, "
            f"got {batch_size} and {action_batch}."
        )
    if proprio_state.ndim != 3 or int(proprio_state.shape[0]) != batch_size:
        raise ValueError(
            "Per-chunk proprio context expects state shape [B, frames_or_chunks, state_dim], "
            f"got {tuple(proprio_state.shape)} for batch_size={batch_size}."
        )
    chunk_size = max(1, int(input_dict["chunk_size"]))
    frame_ids = torch.arange(context_frame_count, device=proprio_state.device, dtype=torch.long)
    chunk_origin_frame = int(input_dict.get("chunk_origin_frame", 0) or 0)
    relative_frame_ids = frame_ids - int(chunk_origin_frame)
    boundary_state = torch.zeros(
        batch_size,
        context_frame_count,
        int(proprio_state.shape[-1]),
        device=proprio_state.device,
        dtype=proprio_state.dtype,
    )
    proprio_count = int(proprio_state.shape[1])
    proprio_granularity = str(input_dict.get("per_chunk_proprio_state_granularity", "chunk"))
    if proprio_granularity not in {"chunk", "frame"}:
        raise ValueError(
            "Per-chunk proprio context expects `per_chunk_proprio_state_granularity` to be "
            f"'chunk' or 'frame', got {proprio_granularity!r}."
        )
    prefix_condition_frames = max(0, int(input_dict.get("prefix_condition_frames", 0) or 0))
    if prefix_condition_frames > 0:
        target_frame_count = max(0, latent_frames - prefix_condition_frames)
        if proprio_granularity == "chunk":
            target_chunk_count = max(
                0,
                int(math.ceil(target_frame_count / float(chunk_size))),
            )
            required_proprio_frames = prefix_condition_frames + target_chunk_count
            if proprio_count < required_proprio_frames:
                raise ValueError(
                    "Prefix per-chunk proprio context expects chunk-level state shape "
                    "[B, prefix_plus_target_chunks, state_dim], "
                    f"got {tuple(proprio_state.shape)} for required_chunks={required_proprio_frames}."
                )
            target_frame_ids = torch.arange(
                target_frame_count,
                device=proprio_state.device,
                dtype=torch.long,
            )
            target_chunk_ids = (
                torch.div(target_frame_ids, chunk_size, rounding_mode="floor")
                + prefix_condition_frames
            )
            target_boundary_state = proprio_state.index_select(dim=1, index=target_chunk_ids)
            prefix_state = proprio_state[:, :prefix_condition_frames, :]
            boundary_state = torch.cat([prefix_state, target_boundary_state], dim=1)
        else:
            required_proprio_frames = target_frame_count + prefix_condition_frames
            if proprio_count < required_proprio_frames:
                raise ValueError(
                    "Prefix per-chunk proprio context expects frame-level state shape "
                    "[B, prefix_plus_target_frames, state_dim], "
                    f"got {tuple(proprio_state.shape)} for required_frames={required_proprio_frames}."
                )
            target_frame_ids = torch.arange(
                target_frame_count,
                device=proprio_state.device,
                dtype=torch.long,
            )
            target_boundary_ids = (
                torch.div(target_frame_ids, chunk_size, rounding_mode="floor") * chunk_size
            )
            target_boundary_state = proprio_state.index_select(dim=1, index=target_boundary_ids)
            prefix_state = proprio_state[:, :prefix_condition_frames, :]
            boundary_state = torch.cat([prefix_state, target_boundary_state], dim=1)
    elif proprio_granularity == "frame":
        boundary_frame_ids = (
            torch.div(relative_frame_ids.clamp_min(0), chunk_size, rounding_mode="floor")
            * chunk_size
            + int(chunk_origin_frame)
            - 1
        )
        valid_boundary_mask = boundary_frame_ids >= 0
        if bool(valid_boundary_mask.any()):
            selected_boundary_ids = boundary_frame_ids[valid_boundary_mask].clamp(
                min=0,
                max=proprio_count - 1,
            )
            boundary_state[:, valid_boundary_mask, :] = proprio_state.index_select(
                dim=1,
                index=selected_boundary_ids,
            )
    else:
        chunk_ids = torch.div(
            relative_frame_ids.clamp_min(0),
            chunk_size,
            rounding_mode="floor",
        )
        valid_chunk_mask = (chunk_ids >= 0) & (chunk_ids < proprio_count)
        if bool(valid_chunk_mask.any()):
            selected_chunk_ids = chunk_ids[valid_chunk_mask].clamp(
                min=0,
                max=proprio_count - 1,
            )
            boundary_state[:, valid_chunk_mask, :] = proprio_state.index_select(
                dim=1,
                index=selected_chunk_ids,
            )

    encode = getattr(transformer, "encode_proprio_hidden_context", None)
    if not callable(encode):
        raise ValueError(
            "Per-chunk proprio mode requires `encode_proprio_hidden_context` on the runtime transformer."
        )
    chunk_context = encode(
        boundary_state,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    patch_t, patch_h, patch_w = transformer.patch_size
    video_frames = latent_frames // int(patch_t)
    if int(patch_t) != 1:
        chunk_context = chunk_context[:, :: int(patch_t), :]
    video_tokens_per_frame = (latent_height // int(patch_h)) * (
        latent_width // int(patch_w)
    )
    action_tokens_per_frame = action_height * action_width
    expected_video_frames = action_frames + prefix_condition_frames
    if video_frames != expected_video_frames:
        raise ValueError(
            "Per-chunk proprio context expects patchified video frames to equal action frames plus "
            "prefix condition frames, "
            f"got video_frames={video_frames}, action_frames={action_frames}, "
            f"prefix_condition_frames={prefix_condition_frames}."
        )
    video_context = chunk_context.repeat_interleave(video_tokens_per_frame, dim=1)
    action_chunk_context = (
        chunk_context[:, prefix_condition_frames:, :]
        if prefix_condition_frames > 0
        else chunk_context
    )
    action_context = action_chunk_context.repeat_interleave(
        action_tokens_per_frame,
        dim=1,
    )
    if hidden_states.shape[0] == 1:
        video_context = rearrange(video_context, "b l c -> 1 (b l) c")
        action_context = rearrange(action_context, "b l c -> 1 (b l) c")
    elif hidden_states.shape[0] != batch_size:
        raise ValueError(
            "Unexpected hidden state layout for per-chunk proprio context: expected leading dimension "
            f"1 or batch_size={batch_size}, got {hidden_states.shape[0]}."
        )

    latent_noisy_len, latent_condition_len, action_noisy_len, action_condition_len = (
        int(split_list[0]),
        int(split_list[1]),
        int(split_list[2]),
        int(split_list[3]),
    )
    apply_to_video = bool(input_dict.get("per_chunk_proprio_apply_to_video", True))
    if (
        int(video_context.shape[1]) != latent_noisy_len
        or int(action_context.shape[1]) != action_noisy_len
    ):
        raise ValueError(
            "Per-chunk proprio additive context layout mismatch: "
            f"video_context={tuple(video_context.shape)}, action_context={tuple(action_context.shape)}, "
            f"split_list={tuple(int(value) for value in split_list)}."
        )
    output = hidden_states.clone()
    if apply_to_video:
        output[:, :latent_noisy_len, :] = (
            output[:, :latent_noisy_len, :] + video_context
        )
        if latent_condition_len > 0:
            if int(video_context.shape[1]) != latent_condition_len:
                raise ValueError(
                    "Per-chunk proprio video condition context length mismatch: "
                    f"video_context={tuple(video_context.shape)}, latent_condition_len={latent_condition_len}."
                )
            output[:, latent_noisy_len : latent_noisy_len + latent_condition_len, :] = (
                output[:, latent_noisy_len : latent_noisy_len + latent_condition_len, :]
                + video_context
            )
    action_start = latent_noisy_len + latent_condition_len
    output[:, action_start : action_start + action_noisy_len, :] = (
        output[:, action_start : action_start + action_noisy_len, :] + action_context
    )
    condition_start = action_start + action_noisy_len
    if action_condition_len > 0:
        output[:, condition_start : condition_start + action_condition_len, :] = (
            output[:, condition_start : condition_start + action_condition_len, :]
            + action_context
        )
    return output


__all__ = [
    "apply_parallel_chunk_proprio_context",
    "build_single_stream_hidden_proprio_context",
    "inject_deprecated_proprio_text_context",
]
