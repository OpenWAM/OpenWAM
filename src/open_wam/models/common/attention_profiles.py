from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common.packed_token_layout import build_exact_video_action_token_layout

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:  # pragma: no cover - older torch builds may not expose FlexAttention
    BlockMask = Any  # type: ignore[misc,assignment]
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]


_COMPILED_FLEX_ATTENTION = None
_COMPILED_CREATE_BLOCK_MASK = None


def _resolve_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        return None
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


def _resolve_compiled_create_block_mask():
    global _COMPILED_CREATE_BLOCK_MASK
    if create_block_mask is None:
        return None
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


@dataclass(frozen=True)
class AttentionProfileSpec:
    """Declarative description of a reusable attention visibility profile."""

    name: str
    family: str
    backend: str


@dataclass
class PreparedAttentionProfile:
    """Backend-ready attention visibility state.

    The profile can carry either dense boolean masks, FlexAttention block masks,
    or both. Callers choose the best representation for the current runtime.
    """

    spec: AttentionProfileSpec
    self_attention_mask: torch.Tensor | None = None
    cross_attention_mask: torch.Tensor | None = None
    self_attention_block_mask: BlockMask | None = None
    cross_attention_block_mask: BlockMask | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


VIDEO_THEN_ACTION_COUPLING = "video_then_action"
JOINT_COUPLING = "joint"
ACTION_THEN_VIDEO_COUPLING = "action_then_video"
DECOUPLED_SAME_STEP_COUPLING = "decoupled_same_step"
VIDEO_NOISY_TO_ACTION_COUPLING = "video_noisy_to_action"
ACTION_NOISY_TO_VIDEO_COUPLING = "action_noisy_to_video"
HISTORY_STREAM_VISIBILITY_FULL = "full"
HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY = "video_queries_video_only"
HISTORY_STREAM_VISIBILITY_VIDEO_ONLY = "video_only"
_HISTORY_STREAM_VISIBILITY_VALUES = {
    HISTORY_STREAM_VISIBILITY_FULL,
    HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY,
    HISTORY_STREAM_VISIBILITY_VIDEO_ONLY,
}

_CHUNKED_EXACT_PROFILE_BY_COUPLING: dict[str, str] = {
    VIDEO_THEN_ACTION_COUPLING: "chunked_temporal_exact",
    JOINT_COUPLING: "chunked_temporal_exact_joint",
    ACTION_THEN_VIDEO_COUPLING: "chunked_temporal_exact_action_then_video",
    DECOUPLED_SAME_STEP_COUPLING: "chunked_temporal_exact_decoupled_same_step",
    VIDEO_NOISY_TO_ACTION_COUPLING: "chunked_temporal_exact_video_noisy_to_action",
    ACTION_NOISY_TO_VIDEO_COUPLING: "chunked_temporal_exact_action_noisy_to_video",
}
_CHUNKED_EXACT_COUPLING_BY_PROFILE = {
    profile_name: coupling for coupling, profile_name in _CHUNKED_EXACT_PROFILE_BY_COUPLING.items()
}

_ATTENTION_PROFILE_ALIASES: dict[str, str] = {
    "chunked_temporal_exact": "chunked_temporal_exact",
    "chunked_temporal_exact_joint": "chunked_temporal_exact_joint",
    "chunked_temporal_exact_action_then_video": "chunked_temporal_exact_action_then_video",
    "chunked_temporal_exact_decoupled_same_step": "chunked_temporal_exact_decoupled_same_step",
    "chunked_temporal_exact_video_noisy_to_action": "chunked_temporal_exact_video_noisy_to_action",
    "chunked_temporal_exact_action_noisy_to_video": "chunked_temporal_exact_action_noisy_to_video",
    "lingbot_chunked_exact": "chunked_temporal_exact",
    "none": "none",
}


def normalize_attention_profile_name(name: str | None) -> str | None:
    if name is None:
        return None
    try:
        return _ATTENTION_PROFILE_ALIASES[name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported attention profile {name!r}. Expected one of {tuple(_ATTENTION_PROFILE_ALIASES)}."
        ) from exc


def normalize_chunked_temporal_exact_coupling(coupling: str | None) -> str:
    """Normalize exact method-1 current-block coupling names."""

    if coupling is None:
        return VIDEO_THEN_ACTION_COUPLING
    value = str(getattr(coupling, "value", coupling))
    if value in _CHUNKED_EXACT_PROFILE_BY_COUPLING:
        return value
    try:
        normalized_profile = normalize_attention_profile_name(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported exact current-block coupling {coupling!r}. "
            f"Expected one of {tuple(_CHUNKED_EXACT_PROFILE_BY_COUPLING)}."
        ) from exc
    if normalized_profile in _CHUNKED_EXACT_COUPLING_BY_PROFILE:
        return _CHUNKED_EXACT_COUPLING_BY_PROFILE[normalized_profile]
    raise ValueError(
        f"Unsupported exact current-block coupling {coupling!r}. "
        f"Expected one of {tuple(_CHUNKED_EXACT_PROFILE_BY_COUPLING)}."
    )


def normalize_parallel_history_stream_visibility(
    visibility: str | None,
    *,
    preserve_video_pretrain_history: bool = False,
) -> str:
    """Normalize exact Method-1 clean-history stream visibility."""

    if visibility is None:
        return (
            HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY
            if preserve_video_pretrain_history
            else HISTORY_STREAM_VISIBILITY_FULL
        )
    value = str(getattr(visibility, "value", visibility))
    if value in _HISTORY_STREAM_VISIBILITY_VALUES:
        return value
    raise ValueError(
        f"Unsupported history stream visibility {visibility!r}. "
        f"Expected one of {tuple(sorted(_HISTORY_STREAM_VISIBILITY_VALUES))}."
    )
def chunked_temporal_exact_profile_name_for_coupling(coupling: str | None) -> str:
    """Return the attention-profile name for an exact method-1 coupling mode."""

    return _CHUNKED_EXACT_PROFILE_BY_COUPLING[normalize_chunked_temporal_exact_coupling(coupling)]


def chunked_temporal_exact_coupling_from_profile_name(name: str) -> str:
    """Return the exact method-1 coupling represented by an attention-profile name."""

    normalized_profile = normalize_attention_profile_name(name)
    if normalized_profile not in _CHUNKED_EXACT_COUPLING_BY_PROFILE:
        raise ValueError(f"Attention profile {name!r} is not a chunked exact profile.")
    return _CHUNKED_EXACT_COUPLING_BY_PROFILE[normalized_profile]


def resolve_attention_profile_backend(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> str:
    if profile is None:
        return "none"
    if prefer_flex and device.type == "cuda":
        if is_cross_attention and profile.cross_attention_block_mask is not None:
            return "lingbot_flex"
        if not is_cross_attention and profile.self_attention_block_mask is not None:
            return "lingbot_flex"
    return "sdpa" if (
        (is_cross_attention and profile.cross_attention_mask is not None)
        or (not is_cross_attention and profile.self_attention_mask is not None)
    ) else "none"


def select_attention_profile_mask(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> tuple[torch.Tensor | None, BlockMask | None]:
    backend = resolve_attention_profile_backend(
        profile,
        device=device,
        prefer_flex=prefer_flex,
        is_cross_attention=is_cross_attention,
    )
    if profile is None or backend == "none":
        return None, None
    if backend == "lingbot_flex":
        return (
            None,
            profile.cross_attention_block_mask if is_cross_attention else profile.self_attention_block_mask,
        )
    return (
        (profile.cross_attention_mask if is_cross_attention else profile.self_attention_mask),
        None,
    )


def apply_attention_backend(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    block_mask: BlockMask | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    if attention_mask is not None:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
    if block_mask is not None:
        if flex_attention is None:
            raise RuntimeError("FlexAttention is not available in this torch build.")
        compiled_flex_attention = _resolve_compiled_flex_attention()
        if compiled_flex_attention is not None:
            return compiled_flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)
        return flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def build_chunked_text_context_cross_attention_mask(
    *,
    query_chunk_ids: torch.Tensor,
    batch_size: int,
    text_token_count: int,
    base_text_token_count: int,
    proprio_context_token_count: int,
    global_suffix_token_count: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Build a query-dependent mask for deprecated text-space proprio tokens.

    Text tokens are visible to every query. Deprecated appended proprio tokens
    are visible only to queries from the matching local chunk. Optional suffix
    tokens, such as learned mode tokens in legacy packed text layouts, are
    visible to every query.
    """

    if query_chunk_ids.ndim != 1:
        raise ValueError(
            "Chunked text context masks expect query_chunk_ids with shape [query_tokens], "
            f"got {tuple(query_chunk_ids.shape)}."
        )
    resolved_batch_size = int(batch_size)
    resolved_text_token_count = int(text_token_count)
    resolved_base_text_token_count = int(base_text_token_count)
    resolved_proprio_context_token_count = int(proprio_context_token_count)
    resolved_global_suffix_token_count = int(global_suffix_token_count)
    if resolved_batch_size <= 0:
        raise ValueError(f"Expected positive batch_size, got {batch_size}.")
    if (
        resolved_base_text_token_count < 0
        or resolved_proprio_context_token_count < 0
        or resolved_global_suffix_token_count < 0
    ):
        raise ValueError(
            "Context token counts must be non-negative, "
            f"got base={base_text_token_count}, proprio={proprio_context_token_count}, "
            f"global_suffix={global_suffix_token_count}."
        )
    if (
        resolved_base_text_token_count
        + resolved_proprio_context_token_count
        + resolved_global_suffix_token_count
        != resolved_text_token_count
    ):
        raise ValueError(
            "Context token counts must sum to text_token_count, "
            f"got base={base_text_token_count}, proprio={proprio_context_token_count}, "
            f"global_suffix={global_suffix_token_count}, "
            f"text={text_token_count}."
        )
    query_chunk_ids = query_chunk_ids.to(device=device, dtype=torch.long)
    text_position = torch.arange(resolved_text_token_count, device=device, dtype=torch.long)
    base_text_visible = text_position < resolved_base_text_token_count
    proprio_index = text_position - resolved_base_text_token_count
    proprio_visible = (
        (proprio_index[None, :] >= 0)
        & (proprio_index[None, :] < resolved_proprio_context_token_count)
        & (proprio_index[None, :] == query_chunk_ids[:, None])
    )
    global_suffix_start = resolved_base_text_token_count + resolved_proprio_context_token_count
    global_suffix_visible = text_position >= global_suffix_start
    mask = base_text_visible[None, :] | proprio_visible | global_suffix_visible[None, :]
    return mask[None, :, :].expand(resolved_batch_size, -1, -1).contiguous()


def build_chunked_temporal_exact_attention_profile(
    *,
    latent_shape: tuple[int, int, int, int, int],
    action_shape: tuple[int, int, int, int, int],
    padded_length: int,
    chunk_size: int,
    window_size: int,
    patch_size: tuple[int, int, int],
    text_token_count: int,
    base_text_token_count: int | None = None,
    proprio_context_token_count: int = 0,
    chunk_origin_frame: int = 0,
    device: torch.device,
    action_context_mask: torch.Tensor | None = None,
    build_dense_masks: bool = False,
    build_flex_masks: bool = False,
    allow_joint_noisy_block_attention: bool | None = None,
    current_block_coupling: str | None = None,
    preserve_video_pretrain_history: bool = False,
    history_stream_visibility: str | None = None,
    prefix_condition_frames: int = 0,
) -> PreparedAttentionProfile:
    # When preserve_video_pretrain_history=True, restrict the noise_to_clean
    # rule on PAST CHUNKS so that the video stream's K/V context matches the
    # video-only pretrain distribution: current V_n attends only history
    # V_clean (no history A_clean), while current A_n keeps full history
    # access. Same-chunk cross-stream visibility is unchanged so all 6
    # coupling modes still behave as before within the current chunk.
    if current_block_coupling is None:
        current_block_coupling = JOINT_COUPLING if allow_joint_noisy_block_attention else VIDEO_THEN_ACTION_COUPLING
    elif allow_joint_noisy_block_attention is not None:
        legacy_coupling = JOINT_COUPLING if allow_joint_noisy_block_attention else VIDEO_THEN_ACTION_COUPLING
        normalized_coupling = normalize_chunked_temporal_exact_coupling(current_block_coupling)
        if normalized_coupling != legacy_coupling:
            raise ValueError(
                "`current_block_coupling` conflicts with legacy "
                "`allow_joint_noisy_block_attention`."
            )
    current_block_coupling = normalize_chunked_temporal_exact_coupling(current_block_coupling)
    resolved_history_stream_visibility = normalize_parallel_history_stream_visibility(
        history_stream_visibility,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
    )
    chunk_origin_frame = int(chunk_origin_frame)
    prefix_condition_frames = max(0, int(prefix_condition_frames))

    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    _, _, action_frames, action_height, action_width = action_shape
    patch_t, patch_h, patch_w = patch_size
    text_token_count = int(text_token_count)
    resolved_base_text_token_count = (
        text_token_count if base_text_token_count is None else int(base_text_token_count)
    )
    resolved_proprio_context_token_count = int(proprio_context_token_count)
    if resolved_proprio_context_token_count < 0:
        raise ValueError(
            "proprio_context_token_count must be non-negative, "
            f"got {resolved_proprio_context_token_count}."
        )
    if resolved_base_text_token_count < 0 or resolved_base_text_token_count > text_token_count:
        raise ValueError(
            "base_text_token_count must be within the per-sample text token count, "
            f"got base_text_token_count={resolved_base_text_token_count}, text_token_count={text_token_count}."
        )
    if resolved_base_text_token_count + resolved_proprio_context_token_count > text_token_count:
        raise ValueError(
            "base_text_token_count + proprio_context_token_count cannot exceed text_token_count, "
            f"got base={resolved_base_text_token_count}, "
            f"proprio={resolved_proprio_context_token_count}, text={text_token_count}."
        )

    layout = build_exact_video_action_token_layout(
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        action_frames=action_frames,
        action_height=action_height,
        action_width=action_width,
        patch_size=patch_size,
        chunk_size=chunk_size,
        chunk_origin_frame=chunk_origin_frame,
        current_block_coupling=current_block_coupling,
        device=device,
        action_context_mask=action_context_mask,
        prefix_condition_frames=prefix_condition_frames,
    )
    layout = layout.with_padding(padded_length)
    latent_token_count = int(batch_size) * int(latent_frames // patch_t) * int(latent_height // patch_h) * int(latent_width // patch_w)
    action_token_count = int(batch_size) * int(action_frames) * int(action_height) * int(action_width)
    action_token_valid = layout.valid_as_kv[
        2 * latent_token_count : 2 * latent_token_count + action_token_count
    ]
    invalid_action_token_count = int((~action_token_valid).sum().item())
    action_context_valid_tokens: tuple[bool, ...] | None = (
        tuple(bool(value) for value in action_token_valid.detach().cpu().tolist())
        if action_context_mask is not None
        else None
    )

    seq_ids = layout.seq_id
    block_ids = layout.block_id
    chunk_ids = layout.chunk_id
    noise_ids = layout.noise_id
    stream_ids = layout.stream_id
    token_valid_as_query = layout.valid_as_query
    token_valid_as_kv = layout.valid_as_kv

    text_seq_ids = torch.arange(batch_size, device=device)[:, None].expand(-1, text_token_count).flatten()
    text_context_positions = torch.arange(text_token_count, device=device)[None, :].expand(batch_size, -1).flatten()

    self_attention_mask = None
    cross_attention_mask = None
    if build_dense_masks:
        q_seq = seq_ids[:, None]
        kv_seq = seq_ids[None, :]
        q_block_id = block_ids[:, None]
        kv_block_id = block_ids[None, :]
        q_noise = noise_ids[:, None]
        kv_noise = noise_ids[None, :]
        q_stream = stream_ids[:, None]
        kv_stream = stream_ids[None, :]
        q_chunk = chunk_ids[:, None]
        kv_chunk = chunk_ids[None, :]
        q_valid = token_valid_as_query[:, None]
        kv_valid = token_valid_as_kv[None, :]

        same_seq = (q_seq == kv_seq) & (q_seq >= 0) & (kv_seq >= 0) & q_valid & kv_valid
        if resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_FULL:
            history_stream_ok = torch.ones_like(q_seq, dtype=torch.bool)
        elif resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY:
            history_stream_ok = (q_stream == kv_stream) | (q_stream == 1)
        elif resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_ONLY:
            history_stream_ok = kv_stream == 0
        else:  # pragma: no cover - normalized above
            raise ValueError(f"Unsupported history stream visibility {resolved_history_stream_visibility!r}.")
        if prefix_condition_frames > 0 or current_block_coupling == DECOUPLED_SAME_STEP_COUPLING:
            clean_to_clean = (
                (q_noise == 1)
                & (kv_noise == 1)
                & (
                    ((kv_chunk < q_chunk) & history_stream_ok)
                    | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
                )
            )
        else:
            clean_to_clean = (
                (q_noise == 1)
                & (kv_noise == 1)
                & (
                    ((kv_chunk < q_chunk) & history_stream_ok)
                    | ((kv_chunk == q_chunk) & (kv_block_id <= q_block_id))
                )
            )
        joint_like_couplings = {
            JOINT_COUPLING,
            DECOUPLED_SAME_STEP_COUPLING,
            VIDEO_NOISY_TO_ACTION_COUPLING,
            ACTION_NOISY_TO_VIDEO_COUPLING,
        }
        prefix_action_then_video = (
            prefix_condition_frames > 0
            and current_block_coupling == ACTION_THEN_VIDEO_COUPLING
        )
        # History stream filter: when preserve_video_pretrain_history is on,
        # current video queries see only same-stream (V) past clean; action
        # queries keep full visibility.
        if current_block_coupling in joint_like_couplings or prefix_action_then_video:
            # Joint-like: noise_to_clean only fires on past chunks.
            noise_to_clean = (
                (q_noise == 0) & (kv_noise == 1) & (kv_chunk < q_chunk) & history_stream_ok
            )
            if prefix_action_then_video:
                noise_to_clean = noise_to_clean | (
                    (q_noise == 0)
                    & (q_stream == 0)
                    & (kv_noise == 1)
                    & (kv_stream == 1)
                    & (kv_chunk == q_chunk)
                )
        else:
            # Staged: split history (filtered) from current-chunk earlier-stage
            # clean (unfiltered) so V_THEN_A's "A reads current Vc" and
            # A_THEN_V's "V reads current Ac" still work after we tighten
            # history visibility.
            in_history = kv_chunk < q_chunk
            in_current_chunk_earlier = (kv_chunk == q_chunk) & (kv_block_id < q_block_id)
            noise_to_clean = (
                (q_noise == 0)
                & (kv_noise == 1)
                & ((in_history & history_stream_ok) | in_current_chunk_earlier)
            )
        if current_block_coupling == JOINT_COUPLING:
            noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_chunk == q_chunk)
        elif current_block_coupling == VIDEO_NOISY_TO_ACTION_COUPLING:
            noise_to_noise = (
                (q_noise == 0)
                & (kv_noise == 0)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
            )
        elif current_block_coupling == ACTION_NOISY_TO_VIDEO_COUPLING:
            noise_to_noise = (
                (q_noise == 0)
                & (kv_noise == 0)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
            )
        else:
            if prefix_condition_frames > 0:
                noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_chunk == q_chunk) & (q_stream == kv_stream)
            else:
                noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_block_id == q_block_id)
        within_window = (q_block_id - kv_block_id).abs() <= int(window_size)
        self_attention_mask = same_seq & within_window & (clean_to_clean | noise_to_clean | noise_to_noise)
        same_text_sample = (
            (seq_ids[:, None] == text_seq_ids[None, :])
            & (seq_ids[:, None] >= 0)
            & (text_seq_ids[None, :] >= 0)
            & token_valid_as_query[:, None]
        )
        if resolved_proprio_context_token_count > 0:
            text_position = text_context_positions[None, :]
            base_text_visible = text_position < resolved_base_text_token_count
            proprio_index = text_position - resolved_base_text_token_count
            proprio_visible = (
                (proprio_index >= 0)
                & (proprio_index < resolved_proprio_context_token_count)
                & (proprio_index == q_chunk)
            )
            cross_attention_mask = same_text_sample & (base_text_visible | proprio_visible)
        else:
            cross_attention_mask = same_text_sample

    self_attention_block_mask = None
    cross_attention_block_mask = None
    if build_flex_masks and create_block_mask is not None:
        seq_ids_flex = seq_ids.to(device=device, dtype=torch.long)
        block_ids_flex = block_ids.to(device=device, dtype=torch.long)
        chunk_ids_flex = chunk_ids.to(device=device, dtype=torch.long)
        noise_ids_flex = noise_ids.to(device=device, dtype=torch.long)
        stream_ids_flex = stream_ids.to(device=device, dtype=torch.long)
        token_valid_as_query_flex = token_valid_as_query.to(device=device, dtype=torch.bool)
        token_valid_as_kv_flex = token_valid_as_kv.to(device=device, dtype=torch.bool)
        text_seq_ids_flex = text_seq_ids.to(device=device, dtype=torch.long)
        text_context_positions_flex = text_context_positions.to(device=device, dtype=torch.long)

        def self_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            same_seq = (
                (seq_ids_flex[q_idx] == seq_ids_flex[kv_idx])
                & (seq_ids_flex[q_idx] >= 0)
                & (seq_ids_flex[kv_idx] >= 0)
                & token_valid_as_query_flex[q_idx]
                & token_valid_as_kv_flex[kv_idx]
            )
            q_chunk = chunk_ids_flex[q_idx]
            kv_chunk = chunk_ids_flex[kv_idx]
            q_block_id = block_ids_flex[q_idx]
            kv_block_id = block_ids_flex[kv_idx]
            if resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_FULL:
                history_stream_ok = torch.ones((), dtype=torch.bool, device=q_idx.device)
            elif resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY:
                history_stream_ok = (stream_ids_flex[q_idx] == stream_ids_flex[kv_idx]) | (
                    stream_ids_flex[q_idx] == 1
                )
            elif resolved_history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_ONLY:
                history_stream_ok = stream_ids_flex[kv_idx] == 0
            else:  # pragma: no cover - normalized above
                raise ValueError(f"Unsupported history stream visibility {resolved_history_stream_visibility!r}.")
            if prefix_condition_frames > 0 or current_block_coupling == DECOUPLED_SAME_STEP_COUPLING:
                clean_to_clean = (
                    (noise_ids_flex[q_idx] == 1)
                    & (noise_ids_flex[kv_idx] == 1)
                    & (
                        ((kv_chunk < q_chunk) & history_stream_ok)
                        | ((kv_chunk == q_chunk) & (stream_ids_flex[kv_idx] == stream_ids_flex[q_idx]))
                    )
                )
            else:
                clean_to_clean = (
                    (noise_ids_flex[q_idx] == 1)
                    & (noise_ids_flex[kv_idx] == 1)
                    & (
                        ((kv_chunk < q_chunk) & history_stream_ok)
                        | ((kv_chunk == q_chunk) & (block_ids_flex[kv_idx] <= block_ids_flex[q_idx]))
                    )
                )
            joint_like_couplings = {
                JOINT_COUPLING,
                DECOUPLED_SAME_STEP_COUPLING,
                VIDEO_NOISY_TO_ACTION_COUPLING,
                ACTION_NOISY_TO_VIDEO_COUPLING,
            }
            prefix_action_then_video = (
                prefix_condition_frames > 0
                and current_block_coupling == ACTION_THEN_VIDEO_COUPLING
            )
            if current_block_coupling in joint_like_couplings or prefix_action_then_video:
                noise_to_clean = (
                    (noise_ids_flex[q_idx] == 0)
                    & (noise_ids_flex[kv_idx] == 1)
                    & (kv_chunk < q_chunk)
                    & history_stream_ok
                )
                if prefix_action_then_video:
                    noise_to_clean = noise_to_clean | (
                        (noise_ids_flex[q_idx] == 0)
                        & (stream_ids_flex[q_idx] == 0)
                        & (noise_ids_flex[kv_idx] == 1)
                        & (stream_ids_flex[kv_idx] == 1)
                        & (kv_chunk == q_chunk)
                    )
            else:
                in_history = kv_chunk < q_chunk
                in_current_chunk_earlier = (kv_chunk == q_chunk) & (kv_block_id < q_block_id)
                noise_to_clean = (
                    (noise_ids_flex[q_idx] == 0)
                    & (noise_ids_flex[kv_idx] == 1)
                    & ((in_history & history_stream_ok) | in_current_chunk_earlier)
                )
            if current_block_coupling == JOINT_COUPLING:
                noise_to_noise = (noise_ids_flex[q_idx] == 0) & (noise_ids_flex[kv_idx] == 0) & (kv_chunk == q_chunk)
            elif current_block_coupling == VIDEO_NOISY_TO_ACTION_COUPLING:
                noise_to_noise = (
                    (noise_ids_flex[q_idx] == 0)
                    & (noise_ids_flex[kv_idx] == 0)
                    & (kv_chunk == q_chunk)
                    & (
                        (stream_ids_flex[q_idx] == stream_ids_flex[kv_idx])
                        | ((stream_ids_flex[q_idx] == 1) & (stream_ids_flex[kv_idx] == 0))
                    )
                )
            elif current_block_coupling == ACTION_NOISY_TO_VIDEO_COUPLING:
                noise_to_noise = (
                    (noise_ids_flex[q_idx] == 0)
                    & (noise_ids_flex[kv_idx] == 0)
                    & (kv_chunk == q_chunk)
                    & (
                        (stream_ids_flex[q_idx] == stream_ids_flex[kv_idx])
                        | ((stream_ids_flex[q_idx] == 0) & (stream_ids_flex[kv_idx] == 1))
                    )
                )
            else:
                if prefix_condition_frames > 0:
                    noise_to_noise = (
                        (noise_ids_flex[q_idx] == 0)
                        & (noise_ids_flex[kv_idx] == 0)
                        & (kv_chunk == q_chunk)
                        & (stream_ids_flex[q_idx] == stream_ids_flex[kv_idx])
                    )
                else:
                    noise_to_noise = (
                        (noise_ids_flex[q_idx] == 0)
                        & (noise_ids_flex[kv_idx] == 0)
                        & (block_ids_flex[kv_idx] == block_ids_flex[q_idx])
                    )
            within_window = (q_block_id - kv_block_id).abs() <= int(window_size)
            return same_seq & within_window & (clean_to_clean | noise_to_clean | noise_to_noise)

        def cross_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del b, h
            same_text_sample = (
                (seq_ids_flex[q_idx] == text_seq_ids_flex[kv_idx])
                & (seq_ids_flex[q_idx] >= 0)
                & (text_seq_ids_flex[kv_idx] >= 0)
                & token_valid_as_query_flex[q_idx]
            )
            if resolved_proprio_context_token_count <= 0:
                return same_text_sample
            text_position = text_context_positions_flex[kv_idx]
            base_text_visible = text_position < resolved_base_text_token_count
            proprio_index = text_position - resolved_base_text_token_count
            proprio_visible = (
                (proprio_index >= 0)
                & (proprio_index < resolved_proprio_context_token_count)
                & (proprio_index == chunk_ids_flex[q_idx])
            )
            return same_text_sample & (base_text_visible | proprio_visible)

        total_seq_len = int(seq_ids.numel())
        total_text_len = int(text_seq_ids.numel())
        compiled_create_block_mask = _resolve_compiled_create_block_mask()
        block_mask_builder = compiled_create_block_mask or create_block_mask
        self_attention_block_mask = block_mask_builder(
            self_mask_mod,
            1,
            1,
            total_seq_len,
            total_seq_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )
        cross_attention_block_mask = block_mask_builder(
            cross_mask_mod,
            1,
            1,
            total_seq_len,
            total_text_len,
            device=str(device),
            _compile=compiled_create_block_mask is not None,
        )

    return PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name=chunked_temporal_exact_profile_name_for_coupling(current_block_coupling),
            family="chunked_exact",
            backend="flex_or_sdpa",
        ),
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        self_attention_block_mask=self_attention_block_mask,
        cross_attention_block_mask=cross_attention_block_mask,
        metadata={
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
            "window_size": int(window_size),
            "latent_shape": tuple(int(v) for v in latent_shape),
            "action_shape": tuple(int(v) for v in action_shape),
            "padded_length": int(padded_length),
            "text_token_count": int(text_token_count),
            "base_text_token_count": int(resolved_base_text_token_count),
            "proprio_context_token_count": int(resolved_proprio_context_token_count),
            "chunk_origin_frame": int(chunk_origin_frame),
            "invalid_action_context_tokens": int(invalid_action_token_count),
            "action_context_valid_tokens": action_context_valid_tokens,
            "allow_joint_noisy_block_attention": current_block_coupling == JOINT_COUPLING,
            "current_block_coupling": current_block_coupling,
            "preserve_video_pretrain_history": bool(preserve_video_pretrain_history),
            "history_stream_visibility": resolved_history_stream_visibility,
            "prefix_condition_frames": int(prefix_condition_frames),
        },
    )


def build_lingbot_chunked_exact_attention_profile(**kwargs) -> PreparedAttentionProfile:
    return build_chunked_temporal_exact_attention_profile(**kwargs)
