from __future__ import annotations

from open_wam.configs import LatentTemporalLayout


WAN_CAUSAL_LATENT_STRIDE_FRAMES = 4
CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET = "next_latent_source_offset"


def raw_window_frames_for_latents(
    latent_frames: int,
    *,
    layout: LatentTemporalLayout | str = LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
    latent_stride_frames: int = WAN_CAUSAL_LATENT_STRIDE_FRAMES,
    action_per_frame: int | None = None,
) -> int:
    """Return raw frames cleanly consumed by `latent_frames` video latents."""

    latent_frames = int(latent_frames)
    if latent_frames <= 0:
        raise ValueError(f"Expected positive latent frame count, got {latent_frames}.")
    layout = _coerce_layout(layout)
    if action_per_frame is not None:
        # Backward-compatible alias for call sites that used the old name.
        # This value is a VAE temporal stride, not a policy action grouping.
        latent_stride_frames = int(action_per_frame)
    latent_stride_frames = int(latent_stride_frames)
    if latent_stride_frames <= 0:
        raise ValueError(f"Expected positive latent_stride_frames, got {latent_stride_frames}.")
    return 1 + latent_stride_frames * (latent_frames - 1)


def latent_raw_boundaries(
    *,
    raw_frame_count: int,
    latent_num_frames: int,
    layout: LatentTemporalLayout | str,
) -> list[int]:
    """Return raw-frame positions delimiting the source span of each latent."""

    raw_frame_count = int(raw_frame_count)
    latent_num_frames = int(latent_num_frames)
    if raw_frame_count <= 0 or latent_num_frames <= 0:
        raise ValueError(
            "Expected positive raw frame and latent frame counts, "
            f"got raw_frame_count={raw_frame_count}, latent_num_frames={latent_num_frames}."
        )
    layout = _coerce_layout(layout)
    if _should_use_explicit_latent_frame_ids(raw_frame_count, latent_num_frames):
        return _explicit_latent_frame_boundaries(raw_frame_count=raw_frame_count, latent_num_frames=latent_num_frames)

    boundaries = [0]
    stride = WAN_CAUSAL_LATENT_STRIDE_FRAMES
    for latent_index in range(1, latent_num_frames + 1):
        boundaries.append(min(raw_frame_count, 1 + stride * (latent_index - 1)))
    boundaries[-1] = min(raw_frame_count, max(boundaries[-1], boundaries[-2] if len(boundaries) > 1 else 0))
    return boundaries


def latent_anchor_positions(
    *,
    raw_frame_count: int,
    latent_num_frames: int,
    layout: LatentTemporalLayout | str,
) -> list[int]:
    """Return the raw-frame position that should anchor each latent slot."""

    raw_frame_count = int(raw_frame_count)
    latent_num_frames = int(latent_num_frames)
    if raw_frame_count <= 0 or latent_num_frames <= 0:
        raise ValueError(
            "Expected positive raw frame and latent frame counts, "
            f"got raw_frame_count={raw_frame_count}, latent_num_frames={latent_num_frames}."
        )
    layout = _coerce_layout(layout)
    if _should_use_explicit_latent_frame_ids(raw_frame_count, latent_num_frames):
        return [min(latent_index, raw_frame_count - 1) for latent_index in range(latent_num_frames)]

    stride = WAN_CAUSAL_LATENT_STRIDE_FRAMES
    return [0 if latent_index == 0 else min(stride * latent_index, raw_frame_count - 1) for latent_index in range(latent_num_frames)]


def observed_frame_ids_for_latent_segment(
    *,
    raw_frame_ids: list[int],
    source_latent_frames: int,
    latent_start: int,
    segment_length: int,
    layout: LatentTemporalLayout | str,
) -> list[int]:
    """Resolve raw-frame anchors for a contiguous latent segment."""

    if not raw_frame_ids:
        raise ValueError("Expected non-empty raw_frame_ids.")
    anchors = latent_anchor_positions(
        raw_frame_count=len(raw_frame_ids),
        latent_num_frames=source_latent_frames,
        layout=layout,
    )
    observed_frame_ids: list[int] = []
    for latent_index in range(int(latent_start), int(latent_start) + int(segment_length)):
        if latent_index < 0:
            observed_frame_ids.append(int(raw_frame_ids[0]))
        elif latent_index >= int(source_latent_frames):
            observed_frame_ids.append(int(raw_frame_ids[-1]))
        else:
            observed_frame_ids.append(int(raw_frame_ids[anchors[latent_index]]))
    return observed_frame_ids


def raw_span_for_latent_range(
    *,
    raw_frame_ids: list[int],
    source_latent_frames: int,
    latent_start: int,
    latent_end: int,
    layout: LatentTemporalLayout | str,
) -> tuple[int, int, int, int]:
    """Return `(start_pos, end_pos, start_frame, end_frame_exclusive)` for a latent range."""

    if not raw_frame_ids:
        raise ValueError("Expected non-empty raw_frame_ids.")
    boundaries = latent_raw_boundaries(
        raw_frame_count=len(raw_frame_ids),
        latent_num_frames=source_latent_frames,
        layout=layout,
    )
    source_start = max(0, min(int(latent_start), int(source_latent_frames)))
    source_end = max(source_start, min(int(latent_end), int(source_latent_frames)))
    start_pos = min(boundaries[source_start], len(raw_frame_ids) - 1)
    end_pos = min(boundaries[source_end], len(raw_frame_ids))
    if end_pos <= start_pos:
        end_pos = min(len(raw_frame_ids), start_pos + 1)
    start_frame = int(raw_frame_ids[start_pos])
    end_frame = int(raw_frame_ids[max(start_pos, end_pos - 1)]) + 1
    return start_pos, end_pos, start_frame, end_frame


def _explicit_latent_frame_boundaries(*, raw_frame_count: int, latent_num_frames: int) -> list[int]:
    boundaries = [min(latent_index, int(raw_frame_count)) for latent_index in range(int(latent_num_frames) + 1)]
    boundaries[-1] = int(raw_frame_count)
    return boundaries


def _should_use_explicit_latent_frame_ids(raw_frame_count: int, latent_num_frames: int) -> bool:
    return int(raw_frame_count) <= int(latent_num_frames)


def _coerce_layout(layout: LatentTemporalLayout | str) -> LatentTemporalLayout:
    if isinstance(layout, LatentTemporalLayout):
        resolved = layout
    else:
        resolved = LatentTemporalLayout(str(layout))
    if resolved is LatentTemporalLayout.EQUAL_BUCKET_LEGACY:
        raise ValueError(
            "`equal_bucket_legacy` latent temporal layout is deprecated and unsupported. "
            "It equal-splits raw frame ids across latent slots and can silently misalign Wan/LingBot "
            "video latents with action groups, including dropping early actions such as u0/u1. "
            "Use `wan_causal_stride4` and rebuild metadata/latents rather than using the legacy layout."
        )
    return resolved
