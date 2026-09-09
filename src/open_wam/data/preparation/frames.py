"""Decode, resample, and resize RGB frames without invoking a model."""

from __future__ import annotations

import io
import math
from typing import NamedTuple

import av
import numpy as np
import torch
import torch.nn.functional as F

from open_wam.configs.data_mixed_video import (
    default_mixed_video_resize_bins,
)
from open_wam.data.mixed_video_decode_backends import (
    _TIMESTAMP_BOUNDARY_EPSILON_SECONDS,
)
from open_wam.data.preparation.video_sources import Clip


class Decoded(NamedTuple):
    """A clip's frames, plus what is needed to place them on the source timeline.

    Two rates, and they are not interchangeable. `fps` is the rate `frames` is
    already at, and is what a caller must resample against -- resampling against
    the source rate decimates a second time. `source_fps` is the rate of the
    footage itself, and is what the payload has to record, because `end_frame`
    and `source_ids` are counted in source frames: a reader that divides one by
    the other to recover a duration gets it wrong unless both are in source
    units.
    """

    frames: np.ndarray
    fps: float
    source_fps: float
    source_frames: int
    source_ids: list[int]


def decode_video(
    data: bytes,
    clip: Clip | None = None,
    *,
    target_fps: float | None = None,
    declared_fps: float | None = None,
) -> Decoded:
    """Decode an in-memory mp4 to uint8 [N, H, W, 3] RGB, optionally one time range.

    Seeking matters here: a v3.0 file can hold hundreds of episodes, and decoding
    from the start for each one would cost far more than the encode itself.
    """

    start = None if clip is None else clip.from_timestamp
    stop = None if clip is None else clip.to_timestamp

    container_fps = 0.0
    kept: list[np.ndarray] = []
    source_ids: list[int] = []
    position = 0
    total = 0

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        container_fps = float(rate) if rate else 0.0
        # A repo that declares a rate is the more trustworthy source; the
        # container is all there is otherwise. Whichever it is, the SAME rate has
        # to drive both the decimation below and the rate reported back, or the
        # two disagree and the caller resamples against a rate the frames are
        # not at.
        source_fps = (
            float(declared_fps) if declared_fps and declared_fps > 0 else container_fps
        )
        if start:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)

        # Materialise only the frames the target rate keeps. A minutes-long clip
        # decoded in full is gigabytes of RGB held at once, and most of it is
        # discarded a moment later by resampling — with many processes on one
        # node that surplus is what runs the machine out of memory.
        step = 1.0
        if target_fps and source_fps > target_fps > 0:
            step = source_fps / target_fps

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            moment = float(frame.pts * stream.time_base)
            if (
                start is not None
                and moment < start - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
            ):
                continue
            if (
                stop is not None
                and moment >= stop - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
            ):
                break
            if total >= position:
                kept.append(frame.to_ndarray(format="rgb24"))
                source_ids.append(total)
                position += step
            total += 1

    if not kept:
        raise ValueError("decoded zero frames")
    if source_fps <= 0:
        raise ValueError("neither metadata nor container declares a frame rate")
    # The effective rate of `kept`, not the rate of the source. Returning the
    # source rate here is what made the caller decimate an already-decimated
    # array and write latents at half the requested rate.
    return Decoded(np.stack(kept), source_fps / step, source_fps, total, source_ids)


def resample_indices(num_frames: int, src_fps: float, dst_fps: float) -> list[int]:
    """Nearest-source-frame resampling. No blending: every output frame is a real frame.

    Upsampling (dst > src) duplicates frames rather than failing, which keeps a
    mixed-fps corpus on one timeline.
    """

    if num_frames <= 0:
        return []
    if abs(src_fps - dst_fps) < 1e-6:
        return list(range(num_frames))
    duration = (num_frames - 1) / src_fps
    count = int(math.floor(duration * dst_fps)) + 1
    step = src_fps / dst_fps
    return [min(num_frames - 1, int(round(i * step))) for i in range(count)]


def truncate_to_temporal_stride(count: int, stride: int) -> int:
    """Largest n <= count with (n - 1) divisible by stride.

    The causal VAE consumes frames as `1 + stride * k`; anything past the last
    full group is silently dropped inside diffusers, so drop it here instead and
    record the honest count in the payload.
    """

    if count < 1:
        return 0
    return 1 + ((count - 1) // stride) * stride


FIT_MODES = ("letterbox_pad", "center_crop", "stretch")


DEFAULT_BINS = default_mixed_video_resize_bins()


def fit_frames(
    frames: np.ndarray, target_h: int, target_w: int, mode: str
) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float32 [N, 3, target_h, target_w] in [0, 1].

    letterbox_pad  keeps geometry and the whole field of view, pads the rest
    center_crop    keeps geometry, discards whatever falls outside the target aspect
    stretch        keeps the field of view, distorts geometry
    """

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    _, _, height, width = video.shape
    resize = lambda x, h, w: F.interpolate(
        x, size=(h, w), mode="bilinear", align_corners=False, antialias=True
    )

    if mode == "letterbox_pad":
        scale = min(target_h / height, target_w / width)
        inner_h = max(1, min(target_h, int(round(height * scale))))
        inner_w = max(1, min(target_w, int(round(width * scale))))
        canvas = video.new_zeros((video.shape[0], 3, target_h, target_w))
        top = (target_h - inner_h) // 2
        left = (target_w - inner_w) // 2
        canvas[:, :, top : top + inner_h, left : left + inner_w] = resize(
            video, inner_h, inner_w
        )
        return canvas.clamp_(0.0, 1.0)

    if mode == "center_crop":
        target_ratio = target_w / target_h
        crop_w = min(width, int(round(height * target_ratio)))
        crop_h = min(height, int(round(width / target_ratio)))
        video = video[
            :,
            :,
            (height - crop_h) // 2 : (height - crop_h) // 2 + crop_h,
            (width - crop_w) // 2 : (width - crop_w) // 2 + crop_w,
        ]
    elif mode != "stretch":
        raise ValueError(f"unknown fit mode {mode!r}")

    if video.shape[-2:] != (target_h, target_w):
        video = resize(video, target_h, target_w)
    return video.clamp_(0.0, 1.0)
