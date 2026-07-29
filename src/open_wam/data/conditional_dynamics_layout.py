from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .latent_contracts import LatentWAMSample


GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE = "target_only_t0_observation_plus_future"
GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON = "t0_singleton"
GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY = "previous_boundary_video_only"


def project_real_conditional_sample_to_target_only(sample: LatentWAMSample) -> LatentWAMSample:
    """Project one real FDM/IDM sample to the rollout-style t0 contract."""

    total_frames = int(sample.video_latents.shape[1])
    if total_frames < 2:
        raise ValueError(
            "Real conditional GJD target-only projection requires at least two latent frames, "
            f"got {total_frames}."
        )
    if sample.actions.shape[0] % total_frames != 0:
        raise ValueError(
            "Real conditional GJD target-only projection requires frame-aligned actions, "
            f"actions={sample.actions.shape[0]}, latent_frames={total_frames}."
        )

    boundary, boundary_source = _real_conditional_target_boundary(sample.metadata)
    if boundary is None:
        source_start = 0
        boundary_source = "default_first_frame"
    else:
        boundary = int(boundary)
        if boundary <= 0:
            source_start = 0
        elif boundary >= total_frames:
            raise ValueError(
                "Real conditional GJD target-only projection requires at least one future frame after "
                f"the target boundary, got boundary={boundary}, latent_frames={total_frames}."
            )
        else:
            source_start = boundary - 1

    target_frames = int(total_frames - source_start)
    action_steps_per_frame = int(sample.actions.shape[0] // total_frames)
    video_latents = sample.video_latents[:, source_start:].contiguous()
    actions, action_mask = _target_only_shifted_actions(
        sample.actions,
        sample.action_mask,
        source_start_frame=source_start,
        target_frames=target_frames,
        action_steps_per_frame=action_steps_per_frame,
    )
    condition_latents = _trim_optional_video(
        sample.condition_latents,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    canonical_video = _trim_optional_video(
        sample.canonical_video,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_state = _trim_optional_frame_tensor(
        sample.proprio_context_state,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_state_mask = _trim_optional_frame_tensor(
        sample.proprio_context_state_mask,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_frames = _trim_optional_frame_tensor(
        sample.proprio_context_frames,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    proprio_context_frames_mask = _trim_optional_frame_tensor(
        sample.proprio_context_frames_mask,
        crop_frames=source_start,
        total_frames=total_frames,
    )
    state, state_mask, state_anchor_source_frame = _target_only_prefix_state(
        sample,
        source_start_frame=source_start,
    )
    metadata = _target_only_conditional_metadata(
        sample.metadata,
        source_start_frame=source_start,
        target_frames=target_frames,
        action_steps_per_frame=action_steps_per_frame,
        actions=actions,
        action_mask=action_mask,
        boundary_source=boundary_source,
        state_anchor_source_frame=state_anchor_source_frame,
    )
    return replace(
        sample,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        state=state,
        state_mask=state_mask,
        canonical_video=canonical_video,
        condition_latents=condition_latents,
        proprio_context_state=proprio_context_state,
        proprio_context_state_mask=proprio_context_state_mask,
        proprio_context_frames=proprio_context_frames,
        proprio_context_frames_mask=proprio_context_frames_mask,
        metadata=metadata,
    )


def _target_only_shifted_actions(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    source_start_frame: int,
    target_frames: int,
    action_steps_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_actions = torch.zeros(
        int(target_frames) * int(action_steps_per_frame),
        int(actions.shape[-1]),
        dtype=actions.dtype,
        device=actions.device,
    )
    projected_mask = torch.zeros_like(projected_actions, dtype=torch.float32)
    projected_mask[:action_steps_per_frame] = 0.0
    source_mask = (
        action_mask.to(device=actions.device, dtype=torch.float32)
        if action_mask is not None
        else torch.ones_like(actions, dtype=torch.float32)
    )
    for target_frame in range(1, int(target_frames)):
        source_frame = int(source_start_frame) + int(target_frame) - 1
        src_start = source_frame * int(action_steps_per_frame)
        src_end = src_start + int(action_steps_per_frame)
        dst_start = int(target_frame) * int(action_steps_per_frame)
        dst_end = dst_start + int(action_steps_per_frame)
        if src_end > int(actions.shape[0]):
            continue
        projected_actions[dst_start:dst_end] = actions[src_start:src_end]
        projected_mask[dst_start:dst_end] = source_mask[src_start:src_end]
    return projected_actions.contiguous(), projected_mask.contiguous()


def _target_only_prefix_state(
    sample: LatentWAMSample,
    *,
    source_start_frame: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    state_anchor_source_frame = max(0, int(source_start_frame) - 1)
    if sample.condition_latents is None:
        state_anchor_source_frame = max(0, int(source_start_frame))
    if sample.proprio_context_frames is None:
        return sample.state, sample.state_mask, state_anchor_source_frame
    frames = sample.proprio_context_frames
    if frames.ndim != 2:
        return sample.state, sample.state_mask, state_anchor_source_frame
    mask = sample.proprio_context_frames_mask
    if mask is None:
        mask = torch.ones_like(frames, dtype=torch.float32)
    if mask.shape != frames.shape:
        return sample.state, sample.state_mask, state_anchor_source_frame
    state_horizon = int(sample.state.shape[0]) if sample.state is not None and sample.state.ndim == 2 else 1
    anchor = max(0, min(int(state_anchor_source_frame), int(frames.shape[0]) - 1))
    start = max(0, anchor - max(1, state_horizon) + 1)
    state = frames[start : anchor + 1].contiguous()
    state_mask = mask[start : anchor + 1].to(device=frames.device, dtype=torch.float32).contiguous()
    if int(state.shape[0]) < state_horizon:
        pad_count = state_horizon - int(state.shape[0])
        state = torch.cat([state[:1].expand(pad_count, -1), state], dim=0).contiguous()
        state_mask = torch.cat([state_mask[:1].expand(pad_count, -1), state_mask], dim=0).contiguous()
    return state, state_mask, anchor


def _target_only_conditional_metadata(
    metadata: dict[str, Any],
    *,
    source_start_frame: int,
    target_frames: int,
    action_steps_per_frame: int,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    boundary_source: str,
    state_anchor_source_frame: int,
) -> dict[str, Any]:
    updated = dict(metadata)
    original_observed = _metadata_sequence(updated.get("observation_frame_indices")) or _metadata_sequence(
        updated.get("observed_frame_ids")
    )
    observed_frame_ids = None
    if original_observed is not None and len(original_observed) >= int(source_start_frame) + int(target_frames):
        observed_frame_ids = original_observed[int(source_start_frame) : int(source_start_frame) + int(target_frames)]
        if "observation_frame_indices" in updated:
            updated["observation_frame_indices"] = list(observed_frame_ids)
        if "observed_frame_ids" in updated:
            updated["observed_frame_ids"] = list(observed_frame_ids)

    old_frame_start = _metadata_frame_boundary(
        metadata,
        ("sample_start_frame", "observation_start", "window_start_frame", "frame_shift", "latent_frame_start"),
    )
    if observed_frame_ids:
        sample_start_frame = int(observed_frame_ids[0])
        first_future_frame = int(observed_frame_ids[1]) if int(target_frames) > 1 else sample_start_frame
        sample_end_frame = int(observed_frame_ids[-1]) + int(action_steps_per_frame)
    else:
        sample_start_frame = int(old_frame_start or 0) + int(source_start_frame)
        first_future_frame = sample_start_frame + int(action_steps_per_frame)
        sample_end_frame = sample_start_frame + int(target_frames) * int(action_steps_per_frame)

    valid_action_steps, valid_action_values = _action_validity_stats(actions=actions, action_mask=action_mask)
    old_valid_frames = metadata.get("segment_valid_latent_frames")
    if old_valid_frames is None:
        valid_latent_frames = int(target_frames)
    else:
        valid_latent_frames = max(0, min(int(target_frames), int(old_valid_frames) - int(source_start_frame)))
    padded_latent_frames = max(0, int(target_frames) - int(valid_latent_frames))

    updated.update(
        {
            "generalist_conditional_contract": GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE,
            "generalist_conditional_training_sequence": "target_only",
            "generalist_conditional_context_used_for_training": False,
            "generalist_conditional_boundary_source": str(boundary_source),
            "generalist_conditional_source_t0_frame_in_sample": int(source_start_frame),
            "generalist_conditional_source_future_start_frame_in_sample": int(source_start_frame) + 1,
            "generalist_gjd_chunk_contract": GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON,
            "history_frames": 1,
            "loss_frame_start": 1,
            "loss_frame_end": int(target_frames),
            "latent_loss_frame_start": 1,
            "latent_loss_frame_end": int(target_frames),
            "action_loss_frame_start": 1,
            "action_loss_frame_end": int(target_frames),
            "current_start_frame_in_sample": 1,
            "current_end_frame_in_sample": int(target_frames),
            "supervised_start": 1,
            "supervised_end": int(target_frames),
            "chunk_origin_frame": 1,
            "target_observation_frame_in_sample": 0,
            "target_observation_frame_index": sample_start_frame,
            "conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "generalist_conditional_history_policy": GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
            "first_supervised_future_frame_in_sample": 1,
            "first_supervised_future_frame_index": first_future_frame,
            "supervised_future_latent_frames": max(0, int(target_frames) - 1),
            "sample_start_frame": sample_start_frame,
            "sample_end_frame": sample_end_frame,
            "observation_start": sample_start_frame,
            "window_start_frame": sample_start_frame,
            "window_end_frame": sample_end_frame,
            "anchor_frame_index": sample_start_frame,
            "target_frame_start": first_future_frame,
            "target_frame_end": sample_end_frame,
            "segment_length_frames": int(target_frames),
            "segment_valid_latent_frames": int(valid_latent_frames),
            "segment_padded_latent_frames": int(padded_latent_frames),
            "tail_padding_mode": "none" if padded_latent_frames == 0 else "zero_order_hold",
            "segment_pre_start_frames": 0,
            "start_padding_mode": "none",
            "context_prefix_frames_in_sample": 1,
            "context_prefix_real_frames": 1,
            "context_prefix_truncated_frames": int(max(0, int(metadata.get("context_prefix_frames_requested", 0) or 0) - 1)),
            "singleton_chunk_frame": 0,
            "state_anchor_source_frame": int(state_anchor_source_frame),
            "state_anchor_frame_in_sample": None if int(state_anchor_source_frame) < int(source_start_frame) else 0,
            "valid_action_steps": int(valid_action_steps),
            "valid_action_values": int(valid_action_values),
        }
    )
    for key in ("latent_frame_start", "frame_shift", "effective_start", "effective_frame_start", "logical_frame_start"):
        if key in metadata and metadata[key] is not None:
            updated[key] = int(metadata[key]) + int(source_start_frame)
    for start_key, end_key in (("effective_start", "effective_end"), ("effective_frame_start", "effective_frame_end")):
        if start_key in updated and updated[start_key] is not None:
            updated[end_key] = int(updated[start_key]) + int(target_frames)
    if "logical_frame_start" in updated and updated["logical_frame_start"] is not None:
        updated["logical_frame_end"] = int(updated["logical_frame_start"]) + int(target_frames)
    for key in ("subwindow_latent_start", "virtual_latent_start"):
        updated[key] = sample_start_frame
    updated["subwindow_latent_end"] = sample_end_frame
    original_action_start = metadata.get("subwindow_action_start")
    updated["subwindow_action_start"] = (
        int(original_action_start)
        if original_action_start is not None
        else 0
    ) + int(source_start_frame) * int(action_steps_per_frame)
    updated["subwindow_action_end"] = int(updated["subwindow_action_start"]) + int(actions.shape[0])

    alignment = updated.get("lingbot_window_action_alignment")
    if not isinstance(alignment, dict):
        alignment = {}
    else:
        alignment = dict(alignment)
    alignment.update(
        {
            "latent_num_frames": int(target_frames),
            "prefix_actions": int(action_steps_per_frame),
            "required_action_num": int(actions.shape[0]),
            "leading_zero_action_frames": 1,
            "leading_zero_action_steps": int(action_steps_per_frame),
            "leading_zero_action_mask": 0.0,
        }
    )
    updated["lingbot_window_action_alignment"] = alignment
    return updated


def _metadata_frame_boundary(metadata: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return int(value)
    return None


def _real_conditional_target_boundary(metadata: dict[str, Any]) -> tuple[int | None, str]:
    """Resolve the sampled current/history boundary for real-demo FDM/IDM."""

    for key in ("current_start_frame_in_sample", "history_frames"):
        value = metadata.get(key)
        if value is not None and int(value) > 0:
            return int(value), key
    for key in ("loss_frame_start", "latent_loss_frame_start", "action_loss_frame_start"):
        value = metadata.get(key)
        if value is not None and int(value) > 0:
            return int(value), key
    return None, "default_first_frame"


def _trim_optional_video(
    canonical_video: torch.Tensor | None,
    *,
    crop_frames: int,
    total_frames: int,
) -> torch.Tensor | None:
    if canonical_video is None:
        return None
    if canonical_video.ndim >= 1 and int(canonical_video.shape[0]) == total_frames:
        return canonical_video[crop_frames:].contiguous()
    if canonical_video.ndim >= 2 and int(canonical_video.shape[1]) == total_frames:
        return canonical_video[:, crop_frames:].contiguous()
    return canonical_video


def _trim_optional_frame_tensor(
    tensor: torch.Tensor | None,
    *,
    crop_frames: int,
    total_frames: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim >= 1 and int(tensor.shape[0]) == total_frames:
        return tensor[crop_frames:].contiguous()
    if tensor.ndim >= 2 and int(tensor.shape[1]) == total_frames:
        return tensor[:, crop_frames:].contiguous()
    return tensor


def _metadata_sequence(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return None


def _action_validity_stats(
    *,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> tuple[int, int]:
    if action_mask is None:
        return int(actions.shape[0]), int(actions.numel())
    reduced = action_mask.float().sum(dim=-1)
    return int((reduced > 0).sum().item()), int(action_mask.float().sum().item())


__all__ = [
    "GENERALIST_CONDITIONAL_CONTRACT_TARGET_ONLY_T0_PLUS_FUTURE",
    "GENERALIST_CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY",
    "GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON",
    "project_real_conditional_sample_to_target_only",
]
