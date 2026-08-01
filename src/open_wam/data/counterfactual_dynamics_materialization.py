"""Tensor and temporal materialization for encoded counterfactual dynamics."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from open_wam.configs import DataConfig, WindowSamplingMode

from .latent_temporal import latent_anchor_positions


COUNTERFACTUAL_STATE_KEY = "observation.state"
COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY = "next_latent_source_offset"
COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE = "t0_observation_plus_future"
COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE = "target_only_t0_observation_plus_future"


def _load_latent_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected latent payload dict at {path}, got {type(payload).__name__}.")
    return payload


def _payload_latents(payload: dict[str, Any], *, key: str) -> torch.Tensor:
    if key not in payload:
        raise ValueError(f"Expected key {key!r} in latent payload.")
    tensor = payload[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"Expected {key!r} tensor [C,T,H,W], got {type(tensor)!r}.")
    return tensor.to(dtype=torch.float32).contiguous()


def _counterfactual_latent_state_frames(
    payload: np.lib.npyio.NpzFile,
    *,
    latent_frames: int,
    state_dim: int,
    data_config: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_frames = int(latent_frames)
    state_dim = int(state_dim)
    if state_dim <= 0:
        empty = torch.zeros(latent_frames, 0, dtype=torch.float32)
        return empty, empty.clone()
    if COUNTERFACTUAL_STATE_KEY not in payload.files:
        state = torch.zeros(latent_frames, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    raw_state = np.asarray(payload[COUNTERFACTUAL_STATE_KEY], dtype=np.float32)
    if raw_state.ndim != 2:
        raise ValueError(f"Expected {COUNTERFACTUAL_STATE_KEY!r} with shape [T,D], got {raw_state.shape}.")
    if raw_state.shape[0] <= 0:
        state = torch.zeros(latent_frames, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    anchors = latent_anchor_positions(
        raw_frame_count=int(raw_state.shape[0]),
        latent_num_frames=latent_frames,
        layout=data_config.latent_temporal_layout,
    )
    selected = raw_state[np.asarray(anchors, dtype=np.int64)]
    state = _pack_state(selected, target_dim=state_dim)
    return state, torch.ones_like(state)


def _pack_state(state: np.ndarray, *, target_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(state, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected state array [T,D], got {tuple(tensor.shape)}.")
    if tensor.shape[1] > target_dim:
        raise ValueError(
            f"Counterfactual state dim {tensor.shape[1]} exceeds configured state_dim={target_dim}."
        )
    if tensor.shape[1] == target_dim:
        return tensor.contiguous()
    padded = torch.zeros(tensor.shape[0], target_dim, dtype=torch.float32)
    padded[:, : tensor.shape[1]] = tensor
    return padded


def _pack_actions(actions: np.ndarray, *, target_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(actions, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected action array [T,D], got {tuple(tensor.shape)}.")
    if tensor.shape[1] > target_dim:
        raise ValueError(
            f"Counterfactual action dim {tensor.shape[1]} exceeds configured action_dim={target_dim}."
        )
    if tensor.shape[1] == target_dim:
        return tensor.contiguous()
    padded = torch.zeros(tensor.shape[0], target_dim, dtype=torch.float32)
    padded[:, : tensor.shape[1]] = tensor
    return padded


def _slice_counterfactual_frame_tensor_with_edge_hold(
    tensor: torch.Tensor,
    *,
    latent_start: int,
    segment_length: int,
) -> torch.Tensor:
    source_frames = int(tensor.shape[0])
    if source_frames <= 0:
        raise ValueError("Counterfactual frame tensor requires at least one source frame.")
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    valid_slice = tensor[source_start:source_end]

    parts: list[torch.Tensor] = []
    if int(latent_start) < 0:
        parts.append(tensor[:1].expand(min(-int(latent_start), segment_length), -1))
    parts.append(valid_slice)
    current_frames = sum(int(part.shape[0]) for part in parts)
    if current_frames < segment_length:
        parts.append(tensor[-1:].expand(segment_length - current_frames, -1))
    return torch.cat(parts, dim=0)[:segment_length].contiguous()


def _counterfactual_state_history_from_frames(
    *,
    proprio_context_frames: torch.Tensor,
    proprio_context_frames_mask: torch.Tensor,
    anchor_frame: int,
    state_horizon: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_horizon = int(state_horizon)
    state_dim = int(state_dim)
    if state_horizon <= 0 or state_dim <= 0:
        empty = torch.zeros(max(0, state_horizon), max(0, state_dim), dtype=torch.float32)
        return empty, empty.clone()
    if proprio_context_frames.shape[0] <= 0:
        state = torch.zeros(state_horizon, state_dim, dtype=torch.float32)
        return state, torch.zeros_like(state)
    anchor = max(0, min(int(anchor_frame), int(proprio_context_frames.shape[0]) - 1))
    start = max(0, anchor - state_horizon + 1)
    state = proprio_context_frames[start : anchor + 1]
    mask = proprio_context_frames_mask[start : anchor + 1]
    if state.shape[0] < state_horizon:
        pad_count = state_horizon - int(state.shape[0])
        state = torch.cat([state[:1].expand(pad_count, -1), state], dim=0)
        mask = torch.cat([mask[:1].expand(pad_count, -1), mask], dim=0)
    return state[-state_horizon:].contiguous(), mask[-state_horizon:].contiguous()


def _counterfactual_action_steps_per_frame(
    actions: torch.Tensor,
    *,
    total_frames: int,
    data_config: DataConfig,
) -> int:
    if total_frames <= 0:
        raise ValueError("Counterfactual sample must contain at least one latent frame.")
    configured = _configured_action_steps_per_latent_frame(data_config)
    if configured is not None:
        min_required = max(0, int(total_frames) - 1) * int(configured)
        max_legacy = int(total_frames) * int(configured)
        action_steps = int(actions.shape[0])
        if min_required <= action_steps <= max_legacy:
            return int(configured)

    transition_frames = max(1, int(total_frames) - 1)
    if actions.shape[0] % transition_frames == 0:
        action_per_frame = int(actions.shape[0] // transition_frames)
    elif actions.shape[0] % total_frames == 0:
        action_per_frame = int(actions.shape[0] // total_frames)
    else:
        raise ValueError(
            "Counterfactual action count must be transition- or legacy frame-aligned, "
            f"got actions={actions.shape[0]}, latent_frames={total_frames}."
        )
    if action_per_frame <= 0:
        raise ValueError("Counterfactual sample must contain at least one action per latent frame.")
    return action_per_frame


def _configured_action_steps_per_latent_frame(data_config: DataConfig) -> int | None:
    num_frames = int(getattr(data_config, "num_frames", 0) or 0)
    action_horizon = int(getattr(data_config.action_schema, "action_horizon", 0) or 0)
    if num_frames <= 0 or action_horizon <= 0:
        return None
    if action_horizon % num_frames != 0:
        return None
    action_per_frame = int(action_horizon // num_frames)
    return action_per_frame if action_per_frame > 0 else None


def _counterfactual_condition_latents_from_source(
    video_latents: torch.Tensor,
    *,
    source_frame_offset: int,
) -> torch.Tensor | None:
    if int(source_frame_offset) == 0:
        return None
    source_frames = int(video_latents.shape[1])
    if source_frames <= 0:
        raise ValueError("Counterfactual condition latent synthesis requires at least one source latent frame.")
    source_indices = torch.arange(source_frames, dtype=torch.long, device=video_latents.device)
    source_indices = (source_indices + int(source_frame_offset)).clamp_(0, source_frames - 1)
    return video_latents.index_select(dim=1, index=source_indices).contiguous()


def _validate_counterfactual_condition_latent_manifest(
    manifest: dict[str, Any],
    *,
    encoded_root: Path,
    source_frame_offset: int,
) -> None:
    if int(source_frame_offset) == 0:
        return
    if manifest.get("condition_latents") is True:
        return
    raise ValueError(
        "Counterfactual encoded latents are missing explicit condition latents. "
        f"{encoded_root} has manifest condition_latents={manifest.get('condition_latents')!r}, "
        f"but this config uses condition_source_frame_offset={int(source_frame_offset)}. "
        "Re-encode with scripts/encode_libero_fdm_counterfactual_dataset.py without "
        "`--skip-condition-latents`."
    )


def _counterfactual_target_only_condition_latents_from_payload(
    *,
    target_payload: dict[str, Any],
    fallback_video_latents: torch.Tensor,
    source_frame_offset: int,
) -> tuple[torch.Tensor | None, str]:
    if int(source_frame_offset) == 0:
        return None, "disabled_zero_offset"
    target_condition = _optional_counterfactual_condition_latents(
        target_payload,
        key="target_condition_video_latents",
        source_frame_offset=source_frame_offset,
    )
    if target_condition is not None:
        if tuple(target_condition.shape) != tuple(fallback_video_latents.shape):
            raise ValueError(
                "Encoded target-only counterfactual condition_latents must match target video latents, "
                f"got condition={tuple(target_condition.shape)}, video={tuple(fallback_video_latents.shape)}."
            )
        return target_condition, "encoded_target_single_frame"

    fallback = _counterfactual_condition_latents_from_source(
        fallback_video_latents,
        source_frame_offset=source_frame_offset,
    )
    return fallback, "synthesized_target_shift"


def _optional_counterfactual_condition_latents(
    payload: dict[str, Any],
    *,
    key: str,
    source_frame_offset: int,
) -> torch.Tensor | None:
    if key not in payload:
        return None
    payload_offset = payload.get("condition_source_frame_offset")
    if payload_offset is None or int(payload_offset) != int(source_frame_offset):
        raise ValueError(
            "Encoded counterfactual condition_source_frame_offset mismatch: "
            f"payload={payload_offset!r}, expected={int(source_frame_offset)}."
        )
    payload_policy = payload.get("condition_source_frame_policy")
    if payload_policy != COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY:
        raise ValueError(
            "Encoded counterfactual condition_source_frame_policy mismatch: "
            f"payload={payload_policy!r}, expected={COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY!r}."
        )
    return _payload_latents(payload, key=key)


def _slice_counterfactual_latents_with_edge_hold(
    video_latents: torch.Tensor,
    *,
    latent_start: int,
    segment_length: int,
) -> torch.Tensor:
    source_frames = int(video_latents.shape[1])
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    valid_slice = video_latents[:, source_start:source_end]

    parts: list[torch.Tensor] = []
    if int(latent_start) < 0:
        parts.append(video_latents[:, :1].expand(-1, min(-int(latent_start), segment_length), -1, -1))
    parts.append(valid_slice)
    current_frames = sum(int(part.shape[1]) for part in parts)
    if current_frames < segment_length:
        parts.append(video_latents[:, -1:].expand(-1, segment_length - current_frames, -1, -1))
    return torch.cat(parts, dim=1)[:, :segment_length].contiguous()


def _build_counterfactual_fixed_segment(
    *,
    video_latents: torch.Tensor,
    condition_latents: torch.Tensor | None = None,
    actions: torch.Tensor,
    context_frames: int,
    latent_start: int,
    segment_length: int,
    action_per_frame: int,
    condition_source_frame_offset: int = 0,
    mask_leading_zero_action_context: bool = False,
) -> dict[str, Any]:
    source_frames = int(video_latents.shape[1])
    if source_frames <= 0:
        raise ValueError("Counterfactual segment sampling requires at least one source latent frame.")
    if segment_length <= 0:
        raise ValueError(f"Counterfactual segment_length must be positive, got {segment_length}.")
    if condition_latents is not None and tuple(condition_latents.shape) != tuple(video_latents.shape):
        raise ValueError(
            "Counterfactual condition_latents must match video_latents exactly, "
            f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
        )
    source_start = max(0, int(latent_start))
    source_end = min(source_frames, int(latent_start) + int(segment_length))
    if source_end <= source_start:
        source_end = min(source_frames, source_start + 1)
    pre_start_frames = max(0, min(segment_length, -int(latent_start))) if int(latent_start) < 0 else 0
    valid_latent_frames = max(0, min(segment_length, source_frames - int(latent_start)))
    padded_latent_frames = max(0, segment_length - valid_latent_frames)

    segment_video = _slice_counterfactual_latents_with_edge_hold(
        video_latents,
        latent_start=int(latent_start),
        segment_length=int(segment_length),
    )
    segment_condition = (
        _slice_counterfactual_latents_with_edge_hold(
            condition_latents,
            latent_start=int(latent_start),
            segment_length=int(segment_length),
        )
        if condition_latents is not None
        else None
    )

    segment_actions = torch.zeros(
        segment_length * action_per_frame,
        actions.shape[1],
        dtype=torch.float32,
    )
    action_mask = torch.zeros_like(segment_actions)
    leading_zero_action_frames = int(pre_start_frames) if int(pre_start_frames) > 0 else 1
    leading_zero_action_mask = 0.0 if int(pre_start_frames) > 0 or mask_leading_zero_action_context else 1.0
    for output_frame in range(segment_length):
        dst_start = output_frame * action_per_frame
        dst_end = dst_start + action_per_frame
        if output_frame < leading_zero_action_frames:
            action_mask[dst_start:dst_end] = float(leading_zero_action_mask)
            continue
        source_frame = source_start + output_frame - leading_zero_action_frames
        if source_frame < 0 or source_frame >= source_frames:
            continue
        src_start = source_frame * action_per_frame
        src_end = src_start + action_per_frame
        if src_end > int(actions.shape[0]):
            continue
        segment_actions[dst_start:dst_end] = actions[src_start:src_end]
        action_mask[dst_start:dst_end] = 1.0

    target_observation_frame = int(context_frames) - int(latent_start)
    first_supervised_future_frame = int(target_observation_frame) + 1
    loss_frame_start = max(0, int(pre_start_frames), int(first_supervised_future_frame))
    loss_frame_end = min(int(segment_length), int(valid_latent_frames))
    if loss_frame_end < loss_frame_start:
        loss_frame_end = loss_frame_start
    prefix_state_source_frame = int(source_start)
    if condition_latents is not None:
        prefix_state_source_frame = int(source_start) + int(condition_source_frame_offset)
    prefix_state_source_frame = max(0, min(source_frames - 1, int(prefix_state_source_frame)))
    prefix_state_frame_in_sample: int | None = None
    if int(source_start) <= prefix_state_source_frame < int(source_end):
        prefix_state_frame_in_sample = int(prefix_state_source_frame) - int(source_start) + int(pre_start_frames)
    prefix_state_frame = (
        int(prefix_state_frame_in_sample)
        if prefix_state_frame_in_sample is not None
        else max(0, min(segment_length - 1, int(pre_start_frames)))
    )
    return {
        "video_latents": segment_video,
        "condition_latents": segment_condition,
        "actions": segment_actions.contiguous(),
        "action_mask": action_mask.contiguous(),
        "pre_start_frames": int(pre_start_frames),
        "valid_latent_frames": int(valid_latent_frames),
        "padded_latent_frames": int(padded_latent_frames),
        "valid_source_frames": max(0, int(source_end) - int(source_start)),
        "loss_frame_start": int(loss_frame_start),
        "loss_frame_end": int(loss_frame_end),
        "chunk_origin_frame": int(loss_frame_start),
        "target_observation_frame": int(target_observation_frame),
        "first_supervised_future_frame": int(first_supervised_future_frame),
        "prefix_state_frame": int(prefix_state_frame),
        "prefix_state_source_frame": int(prefix_state_source_frame),
        "prefix_state_frame_in_sample": prefix_state_frame_in_sample,
        "leading_zero_action_frames": int(leading_zero_action_frames),
        "leading_zero_action_mask": float(leading_zero_action_mask),
    }


def _sample_counterfactual_attention_geometry(
    *,
    data_config: DataConfig,
    segment_length: int,
) -> tuple[int, int]:
    """Mirror real uniform-segment chunk/window randomization for counterfactual rows."""

    sample_cfg = data_config.sample_construction
    if sample_cfg.mode != WindowSamplingMode.UNIFORM_SEGMENT:
        return (
            max(1, int(sample_cfg.chunk_size)),
            max(1, int(sample_cfg.window_size)),
        )

    max_chunk_size = max(1, min(int(sample_cfg.chunk_size), int(segment_length)))
    if bool(sample_cfg.randomize_geometry) and max_chunk_size > 1:
        sampled_chunk_size = int(random.randint(1, max_chunk_size))
    else:
        sampled_chunk_size = max_chunk_size

    max_window_size = max(1, int(sample_cfg.window_size))
    if bool(sample_cfg.randomize_geometry) and max_window_size >= 4:
        sampled_window_size = int(random.randint(4, max_window_size))
    else:
        sampled_window_size = max_window_size

    return sampled_chunk_size, sampled_window_size


def _counterfactual_observed_frame_ids(
    *,
    context_start_frame: int,
    latent_start: int,
    segment_length: int,
    source_frames: int,
    action_per_frame: int,
) -> list[int]:
    ids: list[int] = []
    for offset in range(int(segment_length)):
        source_frame = min(max(0, int(latent_start) + offset), int(source_frames) - 1)
        ids.append((int(context_start_frame) + source_frame) * int(action_per_frame))
    return ids


def _load_empty_text_embedding(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected empty text embedding tensor at {path!r}, got {type(payload)!r}.")
    if payload.ndim == 3 and payload.shape[0] == 1:
        payload = payload.squeeze(0)
    return payload.to(dtype=torch.float32).contiguous()


__all__ = [
    "COUNTERFACTUAL_CONDITION_SOURCE_FRAME_POLICY",
    "COUNTERFACTUAL_CONTRACT_T0_PLUS_FUTURE",
    "COUNTERFACTUAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE",
    "COUNTERFACTUAL_STATE_KEY",
]
