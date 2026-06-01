from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MoTVideoLayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MoTVideoCache:
    layers: tuple[MoTVideoLayerCache, ...]
    video_seq_len: int


@dataclass(frozen=True)
class MoTActionLayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MoTActionCache:
    layers: tuple[MoTActionLayerCache, ...]
    action_seq_len: int


@dataclass
class MoTRuntimeState:
    """Typed MoT rollout state stored inside `PolicyInferState.variant_state`."""

    action_device: str | None = None
    text_context: torch.Tensor | None = None
    proprio_state: torch.Tensor | None = None
    hidden_proprio_state: torch.Tensor | None = None
    past_hidden_proprio_states: torch.Tensor | None = None
    video_cache: MoTVideoCache | None = None
    # Persistent per-action-expert-layer K/V cache for past action chunks.
    # Grows by `action_horizon` tokens per chunk when the
    # method-1-aligned non-joint runtime writes the last-step (clean)
    # action K/V back to cache. Mirrors Method 1's shared-transformer
    # cache append for action tokens.
    action_cache: MoTActionCache | None = None
    # Absolute video-frame index of the first frame represented by
    # `action_cache`. Tail trimming advances this value; speculative rewinds
    # use it to convert an absolute rewind frame into a cache-local prefix.
    action_cache_start_frame: int = 0
    # Accumulated clean video latents across rollout chunks. Populated by
    # the method-1-aligned non-joint rollout so each subsequent video
    # denoise can attend the full generated-so-far sequence instead of
    # only the driver's sliding observation window. Shape [B, C, T, H, W].
    past_clean_latents: torch.Tensor | None = None
    # Packed-coupling inference keeps denoised action chunks as clean action
    # history. The native packed path does not use the shared exact slot-pool
    # cache, so this tensor provides the action-side continuation that Method 1
    # gets from its joint video/action cache. Shape [B, T_action, D_action].
    past_clean_actions: torch.Tensor | None = None
    # Number of generated video frames appended to `past_clean_latents` by the
    # last packed inference step. Driver warmup replaces exactly this tail with
    # real env observations; action-only rollout sets it to zero.
    pending_predicted_video_frames: int = 0
    video_tokens_per_frame: int | None = None
    next_condition_frame_start: int = 0
    chunk_advance_frames: int = 0
    # Absolute frame offset used when assigning chunk ids in split-cache
    # rollout masks. Strict one-frame startup uses origin 1 so frames 1..4
    # form the first generated chunk, matching packed/train profiles.
    chunk_origin_frame: int = 0
    # Number of learned GJD mode-context tokens appended to `text_context`.
    generalist_mode_text_token_count: int = 0


@dataclass(frozen=True)
class MoTActionTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    denoised_actions: torch.Tensor
    action_mask: torch.Tensor | None


@dataclass(frozen=True)
class MoTVideoTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    future_loss_mask: torch.Tensor


@dataclass(frozen=True)
class MoTTrainArtifacts:
    action: MoTActionTrainArtifacts
    video: MoTVideoTrainArtifacts | None
    condition_mode: str
    runtime_mode: str
    history_frames: int
    video_cache_seq_len: int | None = None


@dataclass(frozen=True)
class MoTInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor | None
    condition_mode: str
    runtime_mode: str
