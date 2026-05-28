from __future__ import annotations

import math
from contextlib import ExitStack

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from open_wam.configs import CurrentBlockCoupling, MoTConditionMode
from open_wam.models.common.attention_profiles import (
    PreparedAttentionProfile,
    apply_attention_backend,
    select_attention_profile_mask,
)
from open_wam.models.common.coupling_profiles import build_exact_packed_video_action_coupling_profile
from open_wam.models.common.video_geometry import video_token_grid_from_latent_shape
from open_wam.models.visual_tower.grid_ids import build_video_grid_ids
from open_wam.models.policy_variants.parallel_stream.reference_runtime import data_seq_to_patch
from open_wam.models.visual_tower.shared_transformer_support import (
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    select_chunk_slices,
)

from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTVideoCache,
    MoTVideoLayerCache,
)
from .modules import MoTActionExpert, MoTActionPreprocessOutput

try:  # pragma: no cover - import surface depends on torch build
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - CPU-only or non-FSDP env
    FSDP = None


def _video_token_grid_for_latents(visual_tower, video_latents: torch.Tensor):
    return video_token_grid_from_latent_shape(
        video_latents,
        patch_size=visual_tower.core.patch_size,
    )


def move_mot_video_cache(
    video_cache: MoTVideoCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> MoTVideoCache:
    target_device = torch.device(device)
    moved_layers: list[MoTVideoLayerCache] = []
    for layer in video_cache.layers:
        moved_layers.append(
            MoTVideoLayerCache(
                key=layer.key.to(device=target_device, dtype=dtype if dtype is not None else layer.key.dtype),
                value=layer.value.to(device=target_device, dtype=dtype if dtype is not None else layer.value.dtype),
            )
        )
    return MoTVideoCache(
        layers=tuple(moved_layers),
        video_seq_len=video_cache.video_seq_len,
    )


def append_mot_video_cache(
    base_cache: MoTVideoCache,
    appended_cache: MoTVideoCache,
) -> MoTVideoCache:
    if len(base_cache.layers) != len(appended_cache.layers):
        raise ValueError(
            "Cannot append MoT video caches with different layer counts, "
            f"got base_layers={len(base_cache.layers)}, appended_layers={len(appended_cache.layers)}."
        )
    merged_layers: list[MoTVideoLayerCache] = []
    for base_layer, appended_layer in zip(base_cache.layers, appended_cache.layers, strict=True):
        merged_layers.append(
            MoTVideoLayerCache(
                key=torch.cat([base_layer.key, appended_layer.key], dim=2),
                value=torch.cat([base_layer.value, appended_layer.value], dim=2),
            )
        )
    return MoTVideoCache(
        layers=tuple(merged_layers),
        video_seq_len=int(base_cache.video_seq_len + appended_cache.video_seq_len),
    )


def trim_mot_video_cache_tail(
    video_cache: MoTVideoCache,
    *,
    max_video_seq_len: int,
) -> MoTVideoCache:
    if max_video_seq_len <= 0:
        raise ValueError(
            "MoT video cache tail trim requires `max_video_seq_len > 0`, "
            f"got max_video_seq_len={max_video_seq_len}."
        )
    if video_cache.video_seq_len <= max_video_seq_len:
        return video_cache
    trim_start = int(video_cache.video_seq_len - max_video_seq_len)
    trimmed_layers: list[MoTVideoLayerCache] = []
    for layer in video_cache.layers:
        trimmed_layers.append(
            MoTVideoLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return MoTVideoCache(
        layers=tuple(trimmed_layers),
        video_seq_len=int(max_video_seq_len),
    )


def forward_packed_video_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    clean_video_latents: torch.Tensor,
    noisy_timesteps: torch.Tensor,
    clean_timesteps: torch.Tensor | None,
    packed_attention_mask: torch.Tensor,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    cache_name: str = "mot_packed_video_training",
    use_activation_checkpointing: bool = False,
) -> tuple[torch.Tensor, MoTVideoCache]:
    """Run packed ``[V_noisy | V_clean]`` through the video core.

    Returns ``(noisy_flow_pred, v_clean_cache)`` where:

    * ``noisy_flow_pred`` has shape ``[B, C_latent, T, H, W]`` and is
      already sliced to the ``V_noisy`` half (used for the video flow-match
      loss).
    * ``v_clean_cache`` is an ``MoTVideoCache`` that carries the per-layer
      K/V for the ``V_clean`` tokens only. The action stream attends this
      cache to realize Method 1's "noisy queries -> clean keys" teacher
      forcing.

    ``clean_timesteps`` controls the ``V_clean`` copy's per-frame timesteps
    (``None`` means zeros = perfectly clean). Pass the schedule-sampled
    values from ``VideoFlowMatchTrainArtifacts.condition_timesteps`` to
    activate Method 1's ``noisy_video_condition_prob`` augmentation.
    """

    if noisy_video_latents.shape != clean_video_latents.shape:
        raise ValueError(
            "forward_packed_video_denoise expects matching noisy/clean video shapes, "
            f"got noisy={tuple(noisy_video_latents.shape)}, clean={tuple(clean_video_latents.shape)}."
        )
    _, _, num_frames, _, _ = noisy_video_latents.shape
    # Under FSDP + torch.utils.checkpoint(use_reentrant=False), each block's
    # forward normally triggers an all_gather of its sharded params, and
    # backward recompute replays those all_gathers. When the recompute
    # trigger pattern differs across ranks, the replayed all_gathers
    # desynchronize and NCCL deadlocks. We only pay the memory cost of
    # `summon_full_params` (full-unshard of the video core for the whole
    # packed forward, ~7GB/rank for a 5B backbone) when checkpointing is
    # actually on -- otherwise FSDP's per-block shard/gather handles both
    # forward and backward deterministically.
    enable_summon = use_activation_checkpointing and torch.is_grad_enabled()
    summon_ctx: ExitStack | _DummyCtx = (
        _summon_full_params(visual_tower.core) if enable_summon else _DummyCtx()
    )
    with summon_ctx:
        effective_clean_timesteps = (
            torch.zeros_like(noisy_timesteps) if clean_timesteps is None else clean_timesteps
        )
        if noisy_timesteps.shape != effective_clean_timesteps.shape:
            raise ValueError(
                "forward_packed_video_denoise expects matching noisy/clean timestep shapes, "
                f"got noisy={tuple(noisy_timesteps.shape)}, clean={tuple(effective_clean_timesteps.shape)}."
            )
        packed_flow_pred, packed_kv = visual_tower.run_packed_exact_video_forward(
            video_latents=torch.cat([noisy_video_latents, clean_video_latents], dim=2),
            timesteps=torch.cat([noisy_timesteps, effective_clean_timesteps], dim=1),
            text_context=text_context,
            attention_mask=packed_attention_mask,
            frame_start=frame_start,
            cache_name=cache_name,
            packed_copies=2,
            detach_cache=False,
        )
    noisy_flow_pred = packed_flow_pred[:, :, :num_frames].contiguous()

    if not packed_kv:
        raise ValueError("forward_packed_video_denoise expected non-empty per-layer K/V.")
    total_tokens = int(packed_kv[0].key.shape[2])
    if total_tokens % 2 != 0:
        raise ValueError(
            "forward_packed_video_denoise expected even total token count across packed halves, "
            f"got {total_tokens}."
        )
    half = total_tokens // 2
    clean_layers: list[MoTVideoLayerCache] = []
    for layer_index, entry in enumerate(packed_kv):
        if entry.key is None or entry.value is None:
            raise ValueError(
                "Packed video forward produced an empty K/V entry at layer "
                f"{layer_index}."
            )
        # Second half corresponds to the V_clean copy; keep gradients attached
        # so action-loss gradients can flow back into the shared video core.
        clean_layers.append(
            MoTVideoLayerCache(
                key=entry.key[:, :, half:, :].contiguous(),
                value=entry.value[:, :, half:, :].contiguous(),
            )
        )
    v_clean_cache = MoTVideoCache(layers=tuple(clean_layers), video_seq_len=half)
    return noisy_flow_pred, v_clean_cache


def build_chunk_causal_video_mask(
    *,
    video_seq_len: int,
    video_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    chunk_origin_frame: int = 0,
) -> torch.Tensor:
    """Chunk-causal self-attention mask for a video-only forward.

    Tokens within the same chunk attend each other bidirectionally; past
    chunks are visible; future chunks are hidden. Used by MoT non-joint
    training so that:

    * the teacher-forced clean-video prefill (``prefill_video_kv_cache``)
      produces K/V that respect chunk causality, and
    * the standalone noisy-video flow forward (``_build_video_train_rollout``)
      does not leak future chunks into the video loss.

    Without this mask both forwards default to fully bidirectional self-
    attention, which leaks future frames through the video core's own
    activations and breaks alignment with Method 1's chunked_temporal_exact
    ``kv_frame <= q_frame`` rule.
    """

    if video_seq_len <= 0:
        raise ValueError(f"Expected positive video_seq_len, got {video_seq_len}.")
    if video_tokens_per_frame <= 0 or action_chunk_size_frames <= 0:
        raise ValueError(
            "Chunk-causal video mask requires positive geometry, "
            f"got video_tokens_per_frame={video_tokens_per_frame}, "
            f"action_chunk_size_frames={action_chunk_size_frames}."
        )
    token_ids = torch.arange(video_seq_len, device=device)
    frame_ids = torch.div(token_ids, int(video_tokens_per_frame), rounding_mode="floor")
    chunk_ids = torch.div(
        frame_ids - int(chunk_origin_frame),
        int(action_chunk_size_frames),
        rounding_mode="floor",
    )
    q_chunk = chunk_ids[:, None]
    kv_chunk = chunk_ids[None, :]
    mask = kv_chunk <= q_chunk
    if attention_window_size is not None:
        within_window = (q_chunk - kv_chunk).abs() <= int(attention_window_size)
        mask = mask & within_window
    return mask


def build_packed_video_self_attention_mask(
    *,
    num_frames: int,
    video_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
) -> torch.Tensor:
    """Video-only packed self-attention mask for Method-1-style training.

    Sequence layout (along token dim): ``[V_noisy (T*ppF) | V_clean (T*ppF)]``.
    Rules (non-joint, identical to Method 1's chunked_temporal_exact with
    ``allow_joint_noisy_block_attention=False``, projected onto chunk ids):

    * ``clean_to_clean``: ``q_noise=1 & kv_noise=1 & kv_chunk <= q_chunk``
    * ``noise_to_clean``: ``q_noise=0 & kv_noise=1 & kv_chunk < q_chunk``
    * ``noise_to_noisy``: ``q_noise=0 & kv_noise=0 & kv_chunk == q_chunk``

    The ``V_clean`` half serves as teacher-forced conditioning. ``V_noisy``
    queries at chunk B see ``V_clean`` at past chunks ``B' < B`` (this is the
    conditioning signal Method 1 relies on for good video temporal coherence)
    and their own same-chunk ``V_noisy`` (self-block). Returns a ``[2S, 2S]``
    boolean mask with ``S = num_frames * video_tokens_per_frame``.
    """

    if num_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "Packed video mask requires positive num_frames and video_tokens_per_frame, "
            f"got num_frames={num_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if action_chunk_size_frames <= 0:
        raise ValueError(
            f"Packed video mask requires positive action_chunk_size_frames, got {action_chunk_size_frames}."
        )
    single_seq_len = int(num_frames) * int(video_tokens_per_frame)
    token_ids = torch.arange(single_seq_len, device=device)
    frame_ids_single = torch.div(token_ids, int(video_tokens_per_frame), rounding_mode="floor")
    # Block ids = chunk_id * 2 so the window scale matches Method 1's
    # chunked_temporal_exact profile (video frame_id = chunk*2 there), which
    # lets callers reuse the same ``sampled_window_size`` metadata Method 1
    # uses without silently doubling the effective receptive field.
    block_ids_single = (
        torch.div(frame_ids_single, int(action_chunk_size_frames), rounding_mode="floor") * 2
    )
    # Both halves share the same block ids; noise ids distinguish them.
    block_ids = torch.cat([block_ids_single, block_ids_single], dim=0)
    noise_ids = torch.cat(
        [
            torch.zeros(single_seq_len, device=device, dtype=torch.bool),
            torch.ones(single_seq_len, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    q_block = block_ids[:, None]
    kv_block = block_ids[None, :]
    q_is_clean = noise_ids[:, None]
    kv_is_clean = noise_ids[None, :]
    clean_to_clean = q_is_clean & kv_is_clean & (kv_block <= q_block)
    noise_to_clean = (~q_is_clean) & kv_is_clean & (kv_block < q_block)
    noise_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_block == q_block)
    mask = clean_to_clean | noise_to_clean | noise_to_noisy
    if attention_window_size is not None:
        within_window = (q_block - kv_block).abs() <= int(attention_window_size)
        mask = mask & within_window
    return mask


def build_packed_action_attention_mask(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Packed action-expert attention mask (Method-1-style).

    Key/Value layout: ``[V_clean (T_v*ppF_v) | A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``.
    Query layout: ``[A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``. The video
    side contributes only its clean copy -- per Method 1, action queries
    never attend ``V_noisy`` (the frame-id parity prevents it regardless, so
    dropping the row saves memory and compute).

    Block ids follow Method 1: video chunk B -> block 2B (even); action
    chunk B -> block 2B+1 (odd). Rules:

    * ``clean_to_clean``: ``q_noise=1 & kv_noise=1 & kv_block <= q_block``
    * ``noise_to_clean``: ``q_noise=0 & kv_noise=1 & kv_block < q_block``
    * ``noise_to_noisy``: ``q_noise=0 & kv_noise=0 & kv_block == q_block``

    ``decoupled_same_step`` keeps the same layout but removes same-chunk
    cross-stream clean-video visibility from action queries.

    Returns a ``[2T_a*ppF_a, T_v*ppF_v + 2T_a*ppF_a]`` boolean mask.
    """
    coupling = CurrentBlockCoupling(current_block_coupling)

    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "Packed action mask requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "Packed action mask requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if action_chunk_size_frames <= 0:
        raise ValueError(
            f"Packed action mask requires positive action_chunk_size_frames, got {action_chunk_size_frames}."
        )
    video_seq_len = int(num_video_frames) * int(video_tokens_per_frame)
    action_seq_len = int(num_action_frames) * int(action_tokens_per_frame)

    # Video K block ids (all clean, even blocks).
    video_token_ids = torch.arange(video_seq_len, device=device)
    video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
    video_block_ids = (
        torch.div(video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor") * 2
    )
    video_chunk_ids = torch.div(video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
    # Action token ids per single copy.
    action_token_ids = torch.arange(action_seq_len, device=device)
    action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
    action_block_ids_single = (
        torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor") * 2 + 1
    )
    action_chunk_ids_single = torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
    # Key side: V_clean + A_noisy + A_clean. Query side: A_noisy + A_clean.
    kv_block_ids = torch.cat(
        [video_block_ids, action_block_ids_single, action_block_ids_single], dim=0
    )
    kv_chunk_ids = torch.cat(
        [video_chunk_ids, action_chunk_ids_single, action_chunk_ids_single], dim=0
    )
    kv_stream_ids = torch.cat(
        [
            torch.zeros(video_seq_len, device=device, dtype=torch.long),
            torch.ones(action_seq_len, device=device, dtype=torch.long),
            torch.ones(action_seq_len, device=device, dtype=torch.long),
        ],
        dim=0,
    )
    kv_is_clean = torch.cat(
        [
            torch.ones(video_seq_len, device=device, dtype=torch.bool),
            torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            torch.ones(action_seq_len, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    q_block_ids = torch.cat([action_block_ids_single, action_block_ids_single], dim=0)
    q_chunk_ids = torch.cat([action_chunk_ids_single, action_chunk_ids_single], dim=0)
    q_stream_ids = torch.ones(2 * action_seq_len, device=device, dtype=torch.long)
    q_is_clean = torch.cat(
        [
            torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            torch.ones(action_seq_len, device=device, dtype=torch.bool),
        ],
        dim=0,
    )

    q_b = q_block_ids[:, None]
    kv_b = kv_block_ids[None, :]
    q_chunk = q_chunk_ids[:, None]
    kv_chunk = kv_chunk_ids[None, :]
    q_stream = q_stream_ids[:, None]
    kv_stream = kv_stream_ids[None, :]
    q_c = q_is_clean[:, None]
    kv_c = kv_is_clean[None, :]
    if coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        clean_to_clean = q_c & kv_c & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
        )
        noise_to_clean = (~q_c) & kv_c & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_b < q_b))
        )
    else:
        clean_to_clean = q_c & kv_c & (kv_b <= q_b)
        noise_to_clean = (~q_c) & kv_c & (kv_b < q_b)
    noise_to_noisy = (~q_c) & (~kv_c) & (kv_b == q_b)
    mask = clean_to_clean | noise_to_clean | noise_to_noisy
    if attention_window_size is not None:
        within_window = (q_b - kv_b).abs() <= int(attention_window_size)
        mask = mask & within_window
    return mask


def build_mot_packed_coupling_attention_profile(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
    build_dense_masks: bool | None = None,
    build_flex_masks: bool | None = None,
    chunk_origin_frame: int = 0,
    action_context_mask: torch.Tensor | None = None,
    history_stream_visibility: str | None = None,
    prefix_condition_frames: int = 0,
) -> PreparedAttentionProfile:
    """Build the Method-1 exact attention profile for M5 packed coupling.

    Query/key layout is ``[V_noisy, V_clean, A_noisy, A_clean]``. The mask
    semantics are intentionally sourced from Method 1's chunked temporal exact
    profile, so M5's two-expert topology uses the same six coupling contracts
    and preserve-video-pretrain-history rule.
    """
    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "M5 packed coupling mask requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "M5 packed coupling mask requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0:
        raise ValueError(f"M5 packed coupling mask requires positive chunk_size_frames, got {chunk_size_frames}.")

    return build_exact_packed_video_action_coupling_profile(
        num_video_frames=num_video_frames,
        video_tokens_per_frame=video_tokens_per_frame,
        num_action_frames=num_action_frames,
        action_tokens_per_frame=action_tokens_per_frame,
        chunk_size_frames=chunk_size_frames,
        device=device,
        build_dense_masks=build_dense_masks,
        build_flex_masks=build_flex_masks,
        attention_window_size=attention_window_size,
        current_block_coupling=current_block_coupling,
        chunk_origin_frame=int(chunk_origin_frame),
        action_context_mask=action_context_mask,
        preserve_video_pretrain_history=True,
        history_stream_visibility=history_stream_visibility,
        prefix_condition_frames=int(prefix_condition_frames),
    )


def build_mot_packed_coupling_attention_mask(
    **kwargs,
) -> torch.Tensor:
    """Return the dense Method-1 exact mask for M5 packed coupling."""

    kwargs.setdefault("build_dense_masks", True)
    kwargs.setdefault("build_flex_masks", False)
    profile = build_mot_packed_coupling_attention_profile(**kwargs)
    if profile.self_attention_mask is None:
        raise RuntimeError("M5 packed coupling dense profile did not produce a self-attention mask.")
    return profile.self_attention_mask


def build_mot_attention_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    device: torch.device,
    condition_mode: MoTConditionMode | str,
    video_tokens_per_frame: int | None = None,
    video_can_attend_action: bool = False,
    action_tokens_per_frame: int | None = None,
    action_chunk_size_frames: int | None = None,
    clean_video_frames: int | None = None,
    clean_action_frames: int | None = None,
    attention_window_size: int | None = None,
    action_frame_shift: int = 0,
    video_frame_shift: int = 0,
    current_block_coupling: CurrentBlockCoupling | str | None = None,
) -> torch.Tensor:
    """Build a shared MoT mask for the FastWAM conditioning variants.

    ``action_frame_shift`` / ``video_frame_shift`` offset sequential
    token-based frame ids into their actual rotary frame positions when
    constructing block ids. Training leaves both at 0 (packed video and
    action share frame ids 0..T-1). At inference the action cache's first
    entry may sit at a non-zero rotary position (e.g. chunk 0 action lives
    at rotary ``[chunk_frames, 2*chunk_frames)`` because video obs occupies
    ``[0, chunk_frames)`` first), and without the shift the mask
    underestimates action block ids by one block per missing obs chunk,
    preventing action from attending current-chunk clean video through
    ``noise_to_clean: kv_block < q_block``.
    """

    if video_seq_len <= 0 or action_seq_len <= 0:
        raise ValueError(
            "MoT attention mask requires positive video and action lengths, "
            f"got video_seq_len={video_seq_len}, action_seq_len={action_seq_len}."
        )
    resolved_mode = MoTConditionMode(condition_mode)
    resolved_coupling = None if current_block_coupling is None else CurrentBlockCoupling(current_block_coupling)
    if resolved_coupling in {CurrentBlockCoupling.JOINT, CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO}:
        resolved_video_can_attend_action = True
    elif resolved_coupling in {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
    }:
        resolved_video_can_attend_action = False
    else:
        resolved_video_can_attend_action = bool(video_can_attend_action)

    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros(total_seq_len, total_seq_len, device=device, dtype=torch.bool)
    if clean_video_frames is not None and clean_action_frames is not None:
        if video_tokens_per_frame is None:
            raise ValueError("MoT chunked history masking requires `video_tokens_per_frame`.")
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT chunked history masking requires `action_tokens_per_frame` and `action_chunk_size_frames`, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if clean_video_frames < 0 or clean_action_frames < 0:
            raise ValueError(
                "MoT chunked history masking requires non-negative clean spans, "
                f"got clean_video_frames={clean_video_frames}, clean_action_frames={clean_action_frames}."
            )
        video_token_ids = torch.arange(video_seq_len, device=device)
        video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        # Shift sequential token-based frame ids into actual rotary frame
        # positions before computing block ids. Clean/noisy membership still
        # uses unshifted token-position counts (``clean_action_frames`` is a
        # count of past frames in the sequence, not a rotary threshold).
        video_block_source = video_frame_ids + int(video_frame_shift)
        action_block_source = action_frame_ids + int(action_frame_shift)
        video_chunk_ids = torch.div(video_block_source, int(action_chunk_size_frames), rounding_mode="floor")
        action_chunk_ids = torch.div(action_block_source, int(action_chunk_size_frames), rounding_mode="floor")
        video_block_ids = torch.div(video_block_source, int(action_chunk_size_frames), rounding_mode="floor") * 2
        action_block_ids = torch.div(action_block_source, int(action_chunk_size_frames), rounding_mode="floor") * 2 + 1
        full_block_ids = torch.cat([video_block_ids, action_block_ids], dim=0)
        full_chunk_ids = torch.cat([video_chunk_ids, action_chunk_ids], dim=0)
        full_stream_ids = torch.cat(
            [
                torch.zeros(video_seq_len, device=device, dtype=torch.long),
                torch.ones(action_seq_len, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        full_is_clean = torch.cat(
            [
                video_frame_ids < int(clean_video_frames),
                action_frame_ids < int(clean_action_frames),
            ],
            dim=0,
        )
        q_is_clean = full_is_clean[:, None]
        kv_is_clean = full_is_clean[None, :]
        q_block = full_block_ids[:, None]
        kv_block = full_block_ids[None, :]
        q_chunk = full_chunk_ids[:, None]
        kv_chunk = full_chunk_ids[None, :]
        q_stream = full_stream_ids[:, None]
        kv_stream = full_stream_ids[None, :]
        if resolved_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
            clean_to_clean = q_is_clean & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
            )
            noise_to_clean = (~q_is_clean) & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_block < q_block))
            )
        else:
            clean_to_clean = q_is_clean & kv_is_clean & (kv_block <= q_block)
            noise_to_clean = (~q_is_clean) & kv_is_clean & (kv_block < q_block)
        if resolved_coupling == CurrentBlockCoupling.JOINT:
            noise_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        elif resolved_coupling == CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION:
            noise_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
            )
        elif resolved_coupling == CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO:
            noise_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
            )
        else:
            noise_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_block == q_block)
        mask = clean_to_clean | noise_to_clean | noise_to_noisy
        if attention_window_size is not None:
            within_window = (q_block - kv_block).abs() <= int(attention_window_size)
            mask = mask & within_window
        if not resolved_video_can_attend_action:
            mask[:video_seq_len, video_seq_len:] = False
        return mask
    if clean_video_frames is not None:
        if video_tokens_per_frame is None:
            raise ValueError("MoT joint chunk-causal masking requires `video_tokens_per_frame`.")
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT joint chunk-causal masking requires `action_tokens_per_frame` and `action_chunk_size_frames`, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if clean_video_frames < 0:
            raise ValueError(f"Expected non-negative `clean_video_frames`, got {clean_video_frames}.")
        video_token_ids = torch.arange(video_seq_len, device=device)
        video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        action_frame_ids = action_frame_ids + int(clean_video_frames)

        full_frame_ids = torch.cat([video_frame_ids, action_frame_ids], dim=0)
        full_chunk_ids = torch.div(full_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
        full_stream_ids = torch.cat(
            [
                torch.zeros(video_seq_len, device=device, dtype=torch.long),
                torch.ones(action_seq_len, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        full_is_clean = torch.cat(
            [
                video_frame_ids < int(clean_video_frames),
                torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            ],
            dim=0,
        )
        q_is_clean = full_is_clean[:, None]
        kv_is_clean = full_is_clean[None, :]
        q_chunk = full_chunk_ids[:, None]
        kv_chunk = full_chunk_ids[None, :]
        q_stream = full_stream_ids[:, None]
        kv_stream = full_stream_ids[None, :]
        q_frame = full_frame_ids[:, None]
        kv_frame = full_frame_ids[None, :]
        if resolved_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
            clean_to_clean = q_is_clean & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
            )
            noisy_to_clean = (~q_is_clean) & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_frame < q_frame))
            )
        else:
            clean_to_clean = q_is_clean & kv_is_clean & (kv_frame <= q_frame)
            noisy_to_clean = (~q_is_clean) & kv_is_clean & (kv_chunk < q_chunk)
        if resolved_coupling == CurrentBlockCoupling.JOINT:
            noisy_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        elif resolved_coupling == CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION:
            noisy_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
            )
        elif resolved_coupling == CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO:
            noisy_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
            )
        else:
            noisy_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        mask = clean_to_clean | noisy_to_clean | noisy_to_noisy
        if attention_window_size is not None:
            within_window = (q_chunk - kv_chunk).abs() <= int(attention_window_size)
            mask = mask & within_window
        if not resolved_video_can_attend_action:
            mask[:video_seq_len, video_seq_len:] = False
        return mask
    mask[:video_seq_len, :video_seq_len] = True
    if action_tokens_per_frame is not None or action_chunk_size_frames is not None:
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT chunk-causal action masking requires both `action_tokens_per_frame` "
                f"and `action_chunk_size_frames`, got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if action_tokens_per_frame <= 0 or action_chunk_size_frames <= 0:
            raise ValueError(
                "MoT chunk-causal action masking requires positive action geometry, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        action_chunk_ids = torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
        mask[video_seq_len:, video_seq_len:] = action_chunk_ids[:, None] >= action_chunk_ids[None, :]
    else:
        mask[video_seq_len:, video_seq_len:] = True
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        if video_tokens_per_frame is None:
            raise ValueError("MoT first-frame conditioning requires `video_tokens_per_frame`.")
        visible_video = min(video_tokens_per_frame, video_seq_len)
    elif resolved_mode in {MoTConditionMode.FULL_VIDEO, MoTConditionMode.TEACHER_FORCING_COND_VIDEO}:
        visible_video = video_seq_len
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")
    mask[video_seq_len:, :visible_video] = True
    if resolved_video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = True
    return mask


def build_mot_inference_action_attention_mask(
    *,
    video_seq_len: int,
    past_action_seq_len: int,
    current_action_seq_len: int,
    video_tokens_per_frame: int,
    action_tokens_per_frame: int,
    chunk_size_frames: int,
    window_size_frames: int,
    device: torch.device,
    video_can_attend_action: bool = False,
    video_frame_start: int = 0,
    past_action_frame_start: int = 0,
    current_action_frame_start: int | None = None,
    chunk_origin_frame: int = 0,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Inference-only MoT action attention mask (Method-1 byte-aligned).

    Mirrors `build_chunked_temporal_exact_attention_profile` for the
    inference layout `[video_cache; past_action_cache; current_action]`:

      * chunk ids are computed relative to ``chunk_origin_frame``
      * block ids: video at chunk*2, action at chunk*2 + 1 (so the same chunk
        gets adjacent ids and `(q - kv).abs() <= window_size` collapses to the
        same window for both streams)
      * causal between clean: ``kv <= q`` (allows same-frame)
      * noisy queries past clean: ``kv < q``
      * noisy queries same-frame noisy: ``kv == q``
      * within_window on block-id delta, with ``window_size_frames`` passed
        through unchanged from ``training_config.window_size`` (matching
        Method 1's `input_dict["window_size"]`).

    All cached tokens are clean (already-denoised K/V written at past chunks
    or the current chunk's clean obs/pred); only the fresh current-action
    tokens are noisy. There is no "noisy video" stream at inference because
    the action expert never denoises video.

    ``decoupled_same_step`` is a safety mask for ablations where action should
    not read current generated clean-video K/V. The rollout also defers that
    video K/V write, so this branch mainly guards accidental same-step cache
    exposure.
    """
    coupling = CurrentBlockCoupling(current_block_coupling)

    if video_seq_len < 0 or past_action_seq_len < 0 or current_action_seq_len <= 0:
        raise ValueError(
            "MoT inference action mask requires non-negative cache lengths and a positive "
            f"current chunk, got video_seq_len={video_seq_len}, "
            f"past_action_seq_len={past_action_seq_len}, "
            f"current_action_seq_len={current_action_seq_len}."
        )
    if video_tokens_per_frame <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "MoT inference action mask requires positive tokens-per-frame, "
            f"got video_tokens_per_frame={video_tokens_per_frame}, "
            f"action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0 or window_size_frames <= 0:
        raise ValueError(
            "MoT inference action mask requires positive chunk/window sizes, "
            f"got chunk_size_frames={chunk_size_frames}, "
            f"window_size_frames={window_size_frames}."
        )
    # Slot-pool capacity (`(attn_window // 2) * (latent_token_per_chunk +
    # action_token_per_chunk)`) is sized in tokens, not frames. Method 1
    # writes both streams so eviction stays chunk-aligned; Method 5 only
    # writes video, so eviction can leave a partial leading frame in the
    # video cache. Don't reject that — floor-division below assigns the
    # partial frame to the oldest block_id, which is correct for the
    # relative within_window check the mask actually uses.
    if current_action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "MoT inference action mask expects current_action_seq_len divisible by action_tokens_per_frame, "
            f"got current_action_seq_len={current_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
        )

    chunk_origin_frame = int(chunk_origin_frame)
    video_token_ids = torch.arange(video_seq_len, device=device)
    video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor") + int(
        video_frame_start
    )
    video_chunk_ids = torch.div(
        video_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    video_block_ids = video_chunk_ids * 2

    past_action_token_ids = torch.arange(past_action_seq_len, device=device)
    past_action_frame_ids = torch.div(
        past_action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
    ) + int(past_action_frame_start)
    past_action_chunk_ids = torch.div(
        past_action_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    past_action_block_ids = past_action_chunk_ids * 2 + 1

    past_action_frames_count = past_action_seq_len // int(action_tokens_per_frame)
    if current_action_frame_start is None:
        current_action_frame_start = int(past_action_frame_start) + int(past_action_frames_count)
    current_action_token_ids = torch.arange(current_action_seq_len, device=device)
    current_action_frame_ids = torch.div(
        current_action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
    ) + int(current_action_frame_start)
    current_action_chunk_ids = torch.div(
        current_action_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    current_action_block_ids = current_action_chunk_ids * 2 + 1

    block_ids = torch.cat(
        [video_block_ids, past_action_block_ids, current_action_block_ids], dim=0
    ).to(dtype=torch.long)
    chunk_ids = torch.cat(
        [video_chunk_ids, past_action_chunk_ids, current_action_chunk_ids], dim=0
    ).to(dtype=torch.long)
    stream_ids = torch.cat(
        [
            torch.zeros(video_seq_len, dtype=torch.long, device=device),
            torch.ones(past_action_seq_len, dtype=torch.long, device=device),
            torch.ones(current_action_seq_len, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    is_clean = torch.cat(
        [
            torch.ones(video_seq_len, dtype=torch.bool, device=device),
            torch.ones(past_action_seq_len, dtype=torch.bool, device=device),
            torch.zeros(current_action_seq_len, dtype=torch.bool, device=device),
        ],
        dim=0,
    )

    q_frame = block_ids[:, None]
    kv_frame = block_ids[None, :]
    q_chunk = chunk_ids[:, None]
    kv_chunk = chunk_ids[None, :]
    q_stream = stream_ids[:, None]
    kv_stream = stream_ids[None, :]
    q_clean = is_clean[:, None]
    kv_clean = is_clean[None, :]

    if coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        clean_to_clean = q_clean & kv_clean & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
        )
        noise_to_clean = (~q_clean) & kv_clean & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_frame < q_frame))
        )
    else:
        clean_to_clean = q_clean & kv_clean & (kv_frame <= q_frame)
        noise_to_clean = (~q_clean) & kv_clean & (kv_frame < q_frame)
    noise_to_noise = (~q_clean) & (~kv_clean) & (kv_frame == q_frame)

    within_window = (q_frame - kv_frame).abs() <= int(window_size_frames)
    mask = within_window & (clean_to_clean | noise_to_clean | noise_to_noise)

    if not video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = False
    return mask


def prefill_video_kv_cache(
    *,
    visual_tower,
    observed_prefix: torch.Tensor,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    attention_mask: torch.Tensor | None = None,
    cross_attention_mask: torch.Tensor | None = None,
    detach_cache: bool = True,
) -> MoTVideoCache:
    """Run the observed video prefix once and cache per-layer self-attention K/V.

    ``attention_mask``: optional square ``[S, S]`` mask over the post-patch
    video token sequence; used to make the prefill chunk-causal when
    teacher-forcing a full clean video during training. Default ``None``
    preserves the fully bidirectional prefill used at inference over
    observed history.

    ``detach_cache``: when True (inference default), K/V are detached so they
    can cross device/dtype boundaries without participating in autograd. Set
    to False during training if you want action-loss gradients to flow back
    through the shared video core (matches Method 1 shared-backbone behavior).
    """

    if observed_prefix.ndim != 5:
        raise ValueError(
            "MoT video prefill expects observed_prefix with shape [B, C, T, H, W], "
            f"got {tuple(observed_prefix.shape)}."
        )
    cache_state = visual_tower.prefill_exact_video_cache(
        observed_prefix=observed_prefix,
        text_context=text_context,
        frame_start=frame_start,
        cache_name="mot_video_prefill",
        attention_mask=attention_mask,
        cross_attention_mask=cross_attention_mask,
        detach_cache=detach_cache,
    )
    cache_layers: list[MoTVideoLayerCache] = []
    for layer_index, entry in enumerate(cache_state.self_attention_kv):
        if entry.key is None or entry.value is None:
            raise ValueError(
                "MoT video prefill expected materialized self-attention K/V entries, "
                f"but layer {layer_index} was empty."
            )
        cache_layers.append(
            MoTVideoLayerCache(
                key=entry.key.detach() if detach_cache else entry.key,
                value=entry.value.detach() if detach_cache else entry.value,
            )
        )
    if not cache_layers:
        raise ValueError("MoT video prefill did not materialize any layer cache entries.")
    return MoTVideoCache(layers=tuple(cache_layers), video_seq_len=int(cache_layers[0].key.shape[2]))


def resolve_mot_condition_latents(
    *,
    video_latents: torch.Tensor,
    condition_mode: MoTConditionMode | str,
    video_prefix_frames: int,
    teacher_forcing_video_noise_prob: float,
    training: bool,
    scheduler=None,
) -> torch.Tensor:
    """Select the video branch used to condition the MoT action expert."""

    resolved_mode = MoTConditionMode(condition_mode)
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        return video_latents[:, :, :1]
    if resolved_mode == MoTConditionMode.FULL_VIDEO:
        return video_latents
    if resolved_mode == MoTConditionMode.TEACHER_FORCING_COND_VIDEO:
        cond_latents = video_latents[:, :, : max(1, video_prefix_frames)].clone()
        if (
            training
            and scheduler is not None
            and teacher_forcing_video_noise_prob > 0.0
            and torch.rand(1, device=video_latents.device).item() < teacher_forcing_video_noise_prob
        ):
            batch_size = cond_latents.shape[0]
            timestep_ids = torch.randint(
                low=0,
                high=len(scheduler.timesteps),
                size=(batch_size, cond_latents.shape[2]),
                device=video_latents.device,
            )
            timesteps = scheduler.timesteps.to(device=video_latents.device)[timestep_ids]
            noise = torch.randn_like(cond_latents)
            cond_latents = scheduler.add_noise(cond_latents, noise, timesteps, t_dim=2)
        return cond_latents
    raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")


def move_mot_action_cache(
    action_cache: MoTActionCache,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> MoTActionCache:
    target_device = torch.device(device)
    moved_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        moved_layers.append(
            MoTActionLayerCache(
                key=layer.key.to(
                    device=target_device,
                    dtype=dtype if dtype is not None else layer.key.dtype,
                ),
                value=layer.value.to(
                    device=target_device,
                    dtype=dtype if dtype is not None else layer.value.dtype,
                ),
            )
        )
    return MoTActionCache(
        layers=tuple(moved_layers),
        action_seq_len=action_cache.action_seq_len,
    )


def append_mot_action_cache(
    base_cache: MoTActionCache,
    appended_cache: MoTActionCache,
) -> MoTActionCache:
    if len(base_cache.layers) != len(appended_cache.layers):
        raise ValueError(
            "Cannot append MoT action caches with different layer counts, "
            f"got base_layers={len(base_cache.layers)}, appended_layers={len(appended_cache.layers)}."
        )
    merged_layers: list[MoTActionLayerCache] = []
    for base_layer, appended_layer in zip(base_cache.layers, appended_cache.layers, strict=True):
        merged_layers.append(
            MoTActionLayerCache(
                key=torch.cat([base_layer.key, appended_layer.key], dim=2),
                value=torch.cat([base_layer.value, appended_layer.value], dim=2),
            )
        )
    return MoTActionCache(
        layers=tuple(merged_layers),
        action_seq_len=int(base_cache.action_seq_len + appended_cache.action_seq_len),
    )


def trim_mot_action_cache_tail(
    action_cache: MoTActionCache,
    *,
    max_action_seq_len: int,
) -> MoTActionCache:
    """Drop the oldest action K/V tokens so the cache stays bounded.

    Mirrors `trim_mot_video_cache_tail`. Method-1-aligned inference uses
    this to keep the action lookback window in sync with the video cache's
    sliding window (which the shared transformer's reference cache backend
    enforces via `attn_window`). Without this trim the action cache grows
    unboundedly while video stays capped, which puts the action expert
    well past its training-time `window_size` distribution after ~10
    chunks and corrupts late-rollout action predictions.
    """

    if max_action_seq_len <= 0:
        raise ValueError(
            "MoT action cache tail trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trim_start = int(action_cache.action_seq_len - max_action_seq_len)
    trimmed_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            MoTActionLayerCache(
                key=layer.key[:, :, trim_start:, :].contiguous(),
                value=layer.value[:, :, trim_start:, :].contiguous(),
            )
        )
    return MoTActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def trim_mot_action_cache_prefix(
    action_cache: MoTActionCache,
    *,
    max_action_seq_len: int,
) -> MoTActionCache:
    """Keep the oldest action K/V tokens when rewinding a speculative tail."""

    if max_action_seq_len <= 0:
        raise ValueError(
            "MoT action cache prefix trim requires `max_action_seq_len > 0`, "
            f"got max_action_seq_len={max_action_seq_len}."
        )
    if action_cache.action_seq_len <= max_action_seq_len:
        return action_cache
    trimmed_layers: list[MoTActionLayerCache] = []
    for layer in action_cache.layers:
        trimmed_layers.append(
            MoTActionLayerCache(
                key=layer.key[:, :, :max_action_seq_len, :].contiguous(),
                value=layer.value[:, :, :max_action_seq_len, :].contiguous(),
            )
        )
    return MoTActionCache(
        layers=tuple(trimmed_layers),
        action_seq_len=int(max_action_seq_len),
    )


def forward_action_with_video_and_action_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    action_cache: MoTActionCache | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, MoTActionCache]:
    """Run the action expert against cached video AND cached past-action K/V.

    Method-1-aligned variant of `forward_action_with_video_cache`. Past
    action chunks contribute per-layer clean K/V via `action_cache`; the
    current (fresh, noisy) chunk's K/V is recomputed from `action_pre`
    each call. Returns the action hidden states plus the per-layer fresh
    K/V, so the caller can append to the running `MoTActionCache` at the
    last denoise step (matching Method 1's `update_cache=1` semantics).
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got video_cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    if action_cache is not None and len(action_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached action K/V pair per action layer, "
            f"got action_cache_layers={len(action_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    fresh_action_seq_len = int(action_pre.tokens.shape[1])
    action_cache_seq_len = int(action_cache.action_seq_len) if action_cache is not None else 0
    expected_total = video_cache.video_seq_len + action_cache_seq_len + fresh_action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching video+action_cache+fresh length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    fresh_q_start = video_cache.video_seq_len + action_cache_seq_len
    action_attention_mask = attention_mask[fresh_q_start:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    fresh_kv_layers: list[MoTActionLayerCache] = []
    with _summon_full_params(action_expert):
        for layer_index, (block, vid_layer) in enumerate(
            zip(action_expert.blocks, video_cache.layers, strict=True)
        ):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            key_parts: list[torch.Tensor] = [vid_layer.key]
            value_parts: list[torch.Tensor] = [vid_layer.value]
            if action_cache is not None:
                act_layer = action_cache.layers[layer_index]
                key_parts.append(act_layer.key)
                value_parts.append(act_layer.value)
            key_parts.append(attn_inputs["key"])
            value_parts.append(attn_inputs["value"])
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat(key_parts, dim=2),
                torch.cat(value_parts, dim=2),
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=action_pre.cross_attention_mask,
            )
            fresh_kv_layers.append(
                MoTActionLayerCache(
                    key=attn_inputs["key"].detach(),
                    value=attn_inputs["value"].detach(),
                )
            )
    fresh_cache = MoTActionCache(
        layers=tuple(fresh_kv_layers),
        action_seq_len=int(fresh_kv_layers[0].key.shape[2]),
    )
    return hidden_states, fresh_cache


def forward_mot_packed_coupling_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    clean_video_latents: torch.Tensor,
    noisy_video_timesteps: torch.Tensor,
    clean_video_timesteps: torch.Tensor | None,
    action_expert: MoTActionExpert,
    packed_action_pre: MoTActionPreprocessOutput,
    attention_profile: PreparedAttentionProfile,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    use_activation_checkpointing: bool = False,
    packed_block_stack=None,
    prefer_flex_attention: bool = True,
    video_cross_attention_mask: torch.Tensor | None = None,
    video_hidden_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run M5's native four-stream packed coupling forward.

    Video/action experts execute separate blocks, but every block attends over
    concatenated K/V from ``[V_noisy, V_clean, A_noisy, A_clean]`` using the
    supplied coupling mask. Returns the V_noisy flow and packed action hidden
    states; callers take loss on the first action half.
    """

    if noisy_video_latents.shape != clean_video_latents.shape:
        raise ValueError(
            "M5 packed coupling expects matching noisy/clean video shapes, "
            f"got noisy={tuple(noisy_video_latents.shape)}, clean={tuple(clean_video_latents.shape)}."
        )
    batch_size = noisy_video_latents.shape[0]
    _, _, num_frames, latent_height, latent_width = noisy_video_latents.shape
    effective_clean_video_timesteps = (
        torch.zeros_like(noisy_video_timesteps) if clean_video_timesteps is None else clean_video_timesteps
    )
    if noisy_video_timesteps.shape != effective_clean_video_timesteps.shape:
        raise ValueError(
            "M5 packed coupling expects matching noisy/clean video timestep shapes, "
            f"got noisy={tuple(noisy_video_timesteps.shape)}, clean={tuple(effective_clean_video_timesteps.shape)}."
        )

    resolved_text = (
        torch.zeros(
            batch_size,
            visual_tower.config.max_text_tokens,
            visual_tower.config.text_dim,
            device=noisy_video_latents.device,
            dtype=noisy_video_latents.dtype,
        )
        if text_context is None
        else text_context.to(device=noisy_video_latents.device, dtype=noisy_video_latents.dtype)
    )
    packed_video_latents = torch.cat([noisy_video_latents, clean_video_latents], dim=2)
    packed_video_timesteps = torch.cat([noisy_video_timesteps, effective_clean_video_timesteps], dim=1)
    video_prepared = visual_tower.core.prepare_exact_single_stream_inputs(
        {
            "noisy_latents": packed_video_latents,
            "text_emb": resolved_text,
            "grid_id": torch.cat(
                [
                    build_video_grid_ids(
                        _video_token_grid_for_latents(visual_tower, noisy_video_latents),
                        device=noisy_video_latents.device,
                        frame_shift=float(frame_start),
                    ),
                    build_video_grid_ids(
                        _video_token_grid_for_latents(visual_tower, clean_video_latents),
                        device=clean_video_latents.device,
                        frame_shift=float(frame_start),
                    ),
                ],
                dim=1,
            )[None].expand(batch_size, -1, -1),
            "timesteps": packed_video_timesteps,
        },
        action_mode=False,
    )
    video_hidden_states = video_prepared["hidden_states"]
    if video_hidden_context is not None:
        if tuple(video_hidden_context.shape) != tuple(video_hidden_states.shape):
            raise ValueError(
                "M5 packed video hidden_context must match embedded video hidden states, "
                f"got hidden_context={tuple(video_hidden_context.shape)}, "
                f"hidden_states={tuple(video_hidden_states.shape)}."
            )
        video_hidden_states = video_hidden_states + video_hidden_context.to(
            device=video_hidden_states.device,
            dtype=video_hidden_states.dtype,
        )
    video_text_hidden_states = video_prepared["text_hidden_states"]
    video_rotary_emb = video_prepared["rotary_emb"]
    video_temb = video_prepared["temb"]
    video_timestep_proj = video_prepared["timestep_proj"]

    action_hidden_states = packed_action_pre.tokens
    action_rotary_emb = packed_action_pre.freqs[:, :, None]
    video_seq_len = int(video_hidden_states.shape[1])
    action_seq_len = int(action_hidden_states.shape[1])
    expected_total = video_seq_len + action_seq_len
    profile_attention_mask, profile_block_mask = select_attention_profile_mask(
        attention_profile,
        device=video_hidden_states.device,
        prefer_flex=prefer_flex_attention,
        is_cross_attention=False,
    )
    if profile_block_mask is None:
        if profile_attention_mask is None or profile_attention_mask.shape != (expected_total, expected_total):
            raise ValueError(
                "M5 packed coupling requires a dense or flex attention profile matching packed video+action length, "
                f"got dense_mask={None if profile_attention_mask is None else tuple(profile_attention_mask.shape)}, "
                f"expected=({expected_total}, {expected_total})."
            )
        video_attention_mask = profile_attention_mask[:video_seq_len, :expected_total][None, None, :, :]
        action_attention_mask = profile_attention_mask[video_seq_len:, :expected_total][None, None, :, :]
    else:
        video_attention_mask = None
        action_attention_mask = None

    def _packed_block_step(
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        video_block,
        action_block,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attn_inputs = video_block.prepare_self_attention_inputs(
            video_hidden_states,
            temb=video_timestep_proj,
            rotary_emb=video_rotary_emb,
        )
        action_attn_inputs = action_block.prepare_self_attention_inputs(
            action_hidden_states,
            temb=packed_action_pre.t_mod,
            rotary_emb=action_rotary_emb,
        )
        joint_query = torch.cat([video_attn_inputs["query"], action_attn_inputs["query"]], dim=2)
        joint_key = torch.cat([video_attn_inputs["key"], action_attn_inputs["key"]], dim=2)
        joint_value = torch.cat([video_attn_inputs["value"], action_attn_inputs["value"]], dim=2)
        if profile_block_mask is not None:
            mixed = apply_attention_backend(
                query=joint_query,
                key=joint_key,
                value=joint_value,
                block_mask=profile_block_mask,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )
            mixed_video, mixed_action = torch.split(mixed, [video_seq_len, action_seq_len], dim=2)
            mixed_video = mixed_video.transpose(1, 2).flatten(2, 3)
            mixed_action = mixed_action.transpose(1, 2).flatten(2, 3)
        else:
            mixed_video = F.scaled_dot_product_attention(
                video_attn_inputs["query"],
                joint_key,
                joint_value,
                attn_mask=video_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            mixed_action = F.scaled_dot_product_attention(
                action_attn_inputs["query"],
                joint_key,
                joint_value,
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
        new_video, _ = video_block.apply_post_attention(
            video_attn_inputs["hidden_states"],
            mixed_attn_output=video_block.attn1.to_out[1](
                linear_with_materialized_params(video_block.attn1.to_out[0], mixed_video)
            ),
            encoder_hidden_states=video_text_hidden_states,
            gate_msa=video_attn_inputs["gate_msa"],
            c_shift_msa=video_attn_inputs["c_shift_msa"],
            c_scale_msa=video_attn_inputs["c_scale_msa"],
            c_gate_msa=video_attn_inputs["c_gate_msa"],
            cross_attention_mask=video_cross_attention_mask,
        )
        new_action, _ = action_block.apply_post_attention(
            action_attn_inputs["hidden_states"],
            mixed_attn_output=action_block.attn1.to_out[1](
                linear_with_materialized_params(action_block.attn1.to_out[0], mixed_action)
            ),
            encoder_hidden_states=packed_action_pre.context,
            gate_msa=action_attn_inputs["gate_msa"],
            c_shift_msa=action_attn_inputs["c_shift_msa"],
            c_scale_msa=action_attn_inputs["c_scale_msa"],
            c_gate_msa=action_attn_inputs["c_gate_msa"],
            cross_attention_mask=packed_action_pre.cross_attention_mask,
        )
        return new_video, new_action

    checkpoint_active = bool(use_activation_checkpointing) and torch.is_grad_enabled()
    if packed_block_stack is not None:
        # FSDP-friendly path: each MoTPackedBlock owns its (video_block,
        # action_block) pair and is its own FSDP unit. Calling the wrapper's
        # forward triggers FSDP's standard pre/post-forward hooks; backward
        # gather is also FSDP-managed. No manual `_summon_full_params` /
        # `linear_with_materialized_params` calls in the per-block path,
        # so backward no longer hits "setStorage out of bounds" from
        # resharded buffers. Both the dense (video/action_attention_mask) and
        # flex (profile_block_mask) profile paths are supported.
        flex_kernel_options = (
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            }
            if profile_block_mask is not None
            else None
        )
        kwargs = dict(
            video_timestep_proj=video_timestep_proj,
            video_rotary_emb=video_rotary_emb,
            action_temb=packed_action_pre.t_mod,
            action_rotary_emb=action_rotary_emb,
            video_attention_mask=video_attention_mask,
            action_attention_mask=action_attention_mask,
            video_text_hidden_states=video_text_hidden_states,
            action_text_hidden_states=packed_action_pre.context,
            video_cross_attention_mask=video_cross_attention_mask,
            action_cross_attention_mask=packed_action_pre.cross_attention_mask,
            block_mask=profile_block_mask,
            flex_kernel_options=flex_kernel_options,
        )
        for packed_block in packed_block_stack.packed_blocks:
            if checkpoint_active:
                video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                    packed_block,
                    video_hidden_states,
                    action_hidden_states,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                video_hidden_states, action_hidden_states = packed_block(
                    video_hidden_states,
                    action_hidden_states,
                    **kwargs,
                )
    else:
        for video_block, action_block in zip(visual_tower.core.blocks, action_expert.blocks, strict=True):
            if checkpoint_active:
                video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                    _packed_block_step,
                    video_hidden_states,
                    action_hidden_states,
                    video_block,
                    action_block,
                    use_reentrant=False,
                    context_fn=lambda vb=video_block, ab=action_block: _checkpoint_summon_context(vb, ab),
                )
            else:
                with _unshard_runtime_params(video_block, action_block):
                    video_hidden_states, action_hidden_states = _packed_block_step(
                        video_hidden_states,
                        action_hidden_states,
                        video_block,
                        action_block,
                    )

    shift, scale = select_chunk_slices(
        materialize_runtime_parameter(
            visual_tower.core.scale_shift_table,
            device=video_temb.device,
            dtype=video_temb.dtype,
        )[None]
        + video_temb[:, :, None, ...],
        2,
    )
    shift = shift.to(video_hidden_states.device)
    scale = scale.to(video_hidden_states.device)
    video_hidden_states = (
        layer_norm_with_materialized_params(visual_tower.core.norm_out, video_hidden_states.float())
        * (1.0 + scale)
        + shift
    ).type_as(video_hidden_states)
    packed_video_flow = linear_with_materialized_params(visual_tower.core.proj_out, video_hidden_states)
    packed_video_flow = data_seq_to_patch(
        visual_tower.core.patch_size,
        packed_video_flow,
        num_frames * 2,
        latent_height,
        latent_width,
        batch_size=batch_size,
    )
    video_flow = packed_video_flow[:, :, :num_frames].contiguous()
    return video_flow, action_hidden_states


def forward_action_with_video_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the action expert against a cached video prefix.

    This mirrors the FastWAM `forward_action_with_video_cache` contract:
    action queries are recomputed every denoise step, while video K/V come from
    a prefilled observed-prefix cache. The helper deliberately leaves video
    cache construction to a later stage because that requires exposing
    additional blockwise helpers from the shared video core.
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    action_seq_len = int(action_pre.tokens.shape[1])
    expected_total = video_cache.video_seq_len + action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching cache+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    action_attention_mask = attention_mask[video_cache.video_seq_len:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    with _summon_full_params(action_expert):
        for block, layer_cache in zip(action_expert.blocks, video_cache.layers, strict=True):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat([layer_cache.key, attn_inputs["key"]], dim=2),
                torch.cat([layer_cache.value, attn_inputs["value"]], dim=2),
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=action_pre.cross_attention_mask,
            )
    return hidden_states


def forward_packed_action_with_video_cache(
    *,
    action_expert: MoTActionExpert,
    packed_action_pre: MoTActionPreprocessOutput,
    v_clean_cache: MoTVideoCache,
    action_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run packed ``[A_noisy | A_clean]`` through the action expert.

    Keys/values at each layer are the concatenation of ``v_clean_cache``
    (teacher-forced clean video K/V produced by ``forward_packed_video_denoise``)
    and the packed action's own K/V. Queries are the packed action tokens --
    the caller slices the first ``T_a*ppF_a`` outputs to get the
    ``A_noisy`` flow prediction for the action loss.

    ``action_attention_mask`` shape: ``[2 * A_tokens, V_clean_tokens + 2 * A_tokens]``
    (use ``build_packed_action_attention_mask``).
    """

    if len(v_clean_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "Packed action runtime requires one cached V_clean K/V pair per action layer, "
            f"got cache_layers={len(v_clean_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    packed_action_seq_len = int(packed_action_pre.tokens.shape[1])
    expected_q = packed_action_seq_len
    expected_kv = v_clean_cache.video_seq_len + packed_action_seq_len
    if action_attention_mask.shape != (expected_q, expected_kv):
        raise ValueError(
            "Packed action runtime requires an attention mask shaped "
            f"[{expected_q}, {expected_kv}], got {tuple(action_attention_mask.shape)}."
        )

    hidden_states = packed_action_pre.tokens
    broadcast_mask = action_attention_mask[None, None, :, :]
    action_rotary_emb = packed_action_pre.freqs[:, :, None]
    with _summon_full_params(action_expert):
        for block, layer_cache in zip(action_expert.blocks, v_clean_cache.layers, strict=True):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=packed_action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat([layer_cache.key, attn_inputs["key"]], dim=2),
                torch.cat([layer_cache.value, attn_inputs["value"]], dim=2),
                attn_mask=broadcast_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=packed_action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=packed_action_pre.cross_attention_mask,
            )
    return hidden_states


def forward_joint_video_action_denoise(
    *,
    visual_tower,
    noisy_video_latents: torch.Tensor,
    video_timesteps: torch.Tensor,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    text_context: torch.Tensor | None,
    attention_mask: torch.Tensor,
    frame_start: int = 0,
    use_activation_checkpointing: bool = False,
    video_cross_attention_mask: torch.Tensor | None = None,
    video_hidden_context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run true joint video+action denoising with cross-stream attention on every layer.

    When `use_activation_checkpointing` is true and grad is enabled, each per-block
    (video, action) joint step is wrapped in `torch.utils.checkpoint.checkpoint`
    so the intermediate activations are freed after forward and recomputed at
    backward time. Trades ~1.3x forward compute for a large activation-memory win
    (useful on the two-stream training path where both experts are trainable).
    """

    batch_size = noisy_video_latents.shape[0]
    _, _, num_frames, latent_height, latent_width = noisy_video_latents.shape
    resolved_text = (
        torch.zeros(
            batch_size,
            visual_tower.config.max_text_tokens,
            visual_tower.config.text_dim,
            device=noisy_video_latents.device,
            dtype=noisy_video_latents.dtype,
        )
        if text_context is None
        else text_context.to(device=noisy_video_latents.device, dtype=noisy_video_latents.dtype)
    )
    video_prepared = visual_tower.core.prepare_exact_single_stream_inputs(
        {
            "noisy_latents": noisy_video_latents,
            "text_emb": resolved_text,
            "grid_id": build_video_grid_ids(
                _video_token_grid_for_latents(visual_tower, noisy_video_latents),
                device=noisy_video_latents.device,
                frame_shift=float(frame_start),
            )[None].expand(batch_size, -1, -1),
            "timesteps": video_timesteps,
        },
        action_mode=False,
    )
    video_hidden_states = video_prepared["hidden_states"]
    if video_hidden_context is not None:
        if tuple(video_hidden_context.shape) != tuple(video_hidden_states.shape):
            raise ValueError(
                "MoT joint video hidden_context must match embedded video hidden states, "
                f"got hidden_context={tuple(video_hidden_context.shape)}, "
                f"hidden_states={tuple(video_hidden_states.shape)}."
            )
        video_hidden_states = video_hidden_states + video_hidden_context.to(
            device=video_hidden_states.device,
            dtype=video_hidden_states.dtype,
        )
    video_text_hidden_states = video_prepared["text_hidden_states"]
    video_rotary_emb = video_prepared["rotary_emb"]
    video_temb = video_prepared["temb"]
    video_timestep_proj = video_prepared["timestep_proj"]

    action_hidden_states = action_pre.tokens
    action_rotary_emb = action_pre.freqs[:, :, None]
    video_seq_len = int(video_hidden_states.shape[1])
    action_seq_len = int(action_hidden_states.shape[1])
    expected_total = video_seq_len + action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT joint runtime requires a square joint attention mask matching video+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )
    video_attention_mask = attention_mask[:video_seq_len, :expected_total][None, None, :, :]
    action_attention_mask = attention_mask[video_seq_len:, :expected_total][None, None, :, :]

    def _joint_block_step(
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        video_block,
        action_block,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attn_inputs = video_block.prepare_self_attention_inputs(
            video_hidden_states,
            temb=video_timestep_proj,
            rotary_emb=video_rotary_emb,
        )
        action_attn_inputs = action_block.prepare_self_attention_inputs(
            action_hidden_states,
            temb=action_pre.t_mod,
            rotary_emb=action_rotary_emb,
        )
        joint_key = torch.cat([video_attn_inputs["key"], action_attn_inputs["key"]], dim=2)
        joint_value = torch.cat([video_attn_inputs["value"], action_attn_inputs["value"]], dim=2)

        mixed_video = F.scaled_dot_product_attention(
            video_attn_inputs["query"],
            joint_key,
            joint_value,
            attn_mask=video_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).flatten(2, 3)
        mixed_action = F.scaled_dot_product_attention(
            action_attn_inputs["query"],
            joint_key,
            joint_value,
            attn_mask=action_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).flatten(2, 3)

        new_video, _ = video_block.apply_post_attention(
            video_attn_inputs["hidden_states"],
            mixed_attn_output=video_block.attn1.to_out[1](
                linear_with_materialized_params(video_block.attn1.to_out[0], mixed_video)
            ),
            encoder_hidden_states=video_text_hidden_states,
            gate_msa=video_attn_inputs["gate_msa"],
            c_shift_msa=video_attn_inputs["c_shift_msa"],
            c_scale_msa=video_attn_inputs["c_scale_msa"],
            c_gate_msa=video_attn_inputs["c_gate_msa"],
            cross_attention_mask=video_cross_attention_mask,
        )
        new_action, _ = action_block.apply_post_attention(
            action_attn_inputs["hidden_states"],
            mixed_attn_output=action_block.attn1.to_out[1](
                linear_with_materialized_params(action_block.attn1.to_out[0], mixed_action)
            ),
            encoder_hidden_states=action_pre.context,
            gate_msa=action_attn_inputs["gate_msa"],
            c_shift_msa=action_attn_inputs["c_shift_msa"],
            c_scale_msa=action_attn_inputs["c_scale_msa"],
            c_gate_msa=action_attn_inputs["c_gate_msa"],
            cross_attention_mask=action_pre.cross_attention_mask,
        )
        return new_video, new_action

    checkpoint_active = bool(use_activation_checkpointing) and torch.is_grad_enabled()
    for video_block, action_block in zip(visual_tower.core.blocks, action_expert.blocks, strict=True):
        if checkpoint_active:
            video_hidden_states, action_hidden_states = torch.utils.checkpoint.checkpoint(
                _joint_block_step,
                video_hidden_states,
                action_hidden_states,
                video_block,
                action_block,
                use_reentrant=False,
                context_fn=lambda vb=video_block, ab=action_block: _checkpoint_summon_context(vb, ab),
            )
        else:
            with _unshard_runtime_params(video_block, action_block):
                video_hidden_states, action_hidden_states = _joint_block_step(
                    video_hidden_states,
                    action_hidden_states,
                    video_block,
                    action_block,
                )

    shift, scale = select_chunk_slices(
        materialize_runtime_parameter(
            visual_tower.core.scale_shift_table,
            device=video_temb.device,
            dtype=video_temb.dtype,
        )[None]
        + video_temb[:, :, None, ...],
        2,
    )
    shift = shift.to(video_hidden_states.device)
    scale = scale.to(video_hidden_states.device)
    video_hidden_states = (
        layer_norm_with_materialized_params(visual_tower.core.norm_out, video_hidden_states.float())
        * (1.0 + scale)
        + shift
    ).type_as(video_hidden_states)
    video_flow = linear_with_materialized_params(visual_tower.core.proj_out, video_hidden_states)
    video_flow = data_seq_to_patch(
        visual_tower.core.patch_size,
        video_flow,
        num_frames,
        latent_height,
        latent_width,
        batch_size=batch_size,
    )
    return video_flow, action_hidden_states


class _DummyCtx:
    """No-op context manager used when we want to skip ``summon_full_params``."""

    def __enter__(self) -> "_DummyCtx":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def _summon_full_params(*modules):
    stack = ExitStack()
    if FSDP is None:
        return stack
    seen_ids: set[int] = set()
    for module in modules:
        fsdp_modules = tuple(FSDP.fsdp_modules(module, root_only=False))
        if not fsdp_modules:
            continue
        for fsdp_module in fsdp_modules:
            module_id = id(fsdp_module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            stack.enter_context(FSDP.summon_full_params(fsdp_module, recurse=False, writeback=False))
    return stack


class _FSDP2UnshardCtx:
    def __init__(self, *modules) -> None:
        self._modules = modules
        self._unsharded: list[object] = []

    def __enter__(self) -> "_FSDP2UnshardCtx":
        seen_ids: set[int] = set()
        for module in self._modules:
            for submodule in module.modules():
                module_id = id(submodule)
                if module_id in seen_ids:
                    continue
                seen_ids.add(module_id)
                unshard = getattr(submodule, "unshard", None)
                reshard = getattr(submodule, "reshard", None)
                if not callable(unshard) or not callable(reshard):
                    continue
                unshard()
                self._unsharded.append(submodule)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for submodule in reversed(self._unsharded):
            reshard = getattr(submodule, "reshard", None)
            if callable(reshard):
                reshard()
        self._unsharded.clear()
        return False


def _unshard_runtime_params(*modules):
    stack = ExitStack()
    stack.enter_context(_summon_full_params(*modules))
    stack.enter_context(_FSDP2UnshardCtx(*modules))
    return stack


def _checkpoint_summon_context(video_block, action_block):
    return (
        _unshard_runtime_params(video_block, action_block),
        _unshard_runtime_params(video_block, action_block),
    )
