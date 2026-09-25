"""Observation/action window geometry, independent of dataset storage."""

from __future__ import annotations


def observation_window_starts(
    *,
    episode_length: int,
    num_frames: int,
    frame_stride: int,
    action_horizon: int,
    sample_stride: int,
) -> range:
    """Enumerate complete windows with actions anchored at the last observation.

    The last required index is ``start + (num_frames - 1) * frame_stride
    + action_horizon - 1``. Episode ordering and window record types belong to
    the dataset adapter.
    """

    required_span = (num_frames - 1) * frame_stride + action_horizon
    max_start = episode_length - required_span
    if max_start < 0:
        return range(0)
    return range(0, max_start + 1, sample_stride)
