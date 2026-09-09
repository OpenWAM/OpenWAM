"""Frame geometry and resize transforms for mixed-video decode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from open_wam.configs import (
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoResizeBinConfig,
)


@dataclass(frozen=True)
class MixedVideoResolvedDecodeSize:
    """Resolved resize target for one mixed-video stream."""

    height: int
    width: int
    bin_name: str
    source_height: int | None
    source_width: int | None


def resolve_mixed_video_decode_size(
    data_config: MixedVideoDataConfig,
    *,
    source_height: int | None,
    source_width: int | None,
) -> MixedVideoResolvedDecodeSize:
    """Resolve the VAE input size for one mixed-video stream."""

    if data_config.decode_size_mode == MixedVideoDecodeSizeMode.FIXED:
        return MixedVideoResolvedDecodeSize(
            height=int(data_config.decode_height),
            width=int(data_config.decode_width),
            bin_name="fixed",
            source_height=source_height,
            source_width=source_width,
        )
    if source_height is None or source_width is None:
        return MixedVideoResolvedDecodeSize(
            height=int(data_config.decode_height),
            width=int(data_config.decode_width),
            bin_name="fixed_missing_source_size",
            source_height=source_height,
            source_width=source_width,
        )
    bin_config = select_mixed_video_resize_bin(
        data_config.decode_resize_bins,
        source_height=int(source_height),
        source_width=int(source_width),
    )
    return MixedVideoResolvedDecodeSize(
        height=int(bin_config.target_height),
        width=int(bin_config.target_width),
        bin_name=str(bin_config.name),
        source_height=int(source_height),
        source_width=int(source_width),
    )


def _batch_resize_frames(
    frames: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
    fit_mode: MixedVideoFrameFitMode | str | None = None,
    center_crop: bool = False,
) -> np.ndarray:
    """Resize an NHWC RGB frame batch with one interpolation call."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [N,H,W,3] uint8 frames, got {frames.shape}.")
    n, height, width, _ = frames.shape
    if n == 0:
        return frames
    resolved_fit_mode = _resolve_frame_fit_mode(
        fit_mode,
        center_crop=center_crop,
    )

    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        target_aspect = target_width / target_height
        current_aspect = width / height
        if abs(current_aspect - target_aspect) > 1e-6:
            if current_aspect > target_aspect:
                crop_width = max(1, int(round(height * target_aspect)))
                left = max(0, (width - crop_width) // 2)
                frames = frames[
                    :,
                    :,
                    left : left + crop_width,
                    :,
                ]
            else:
                crop_height = max(1, int(round(width / target_aspect)))
                top = max(0, (height - crop_height) // 2)
                frames = frames[
                    :,
                    top : top + crop_height,
                    :,
                    :,
                ]
        n, height, width, _ = frames.shape
        if not allow_upscale and (height < target_height or width < target_width):
            return frames
        if height == target_height and width == target_width:
            return frames
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return tensor.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        if height <= 0 or width <= 0 or target_height <= 0 or target_width <= 0:
            raise ValueError(
                "Letterbox requires positive dims, "
                f"got input=({height},{width}) "
                f"target=({target_height},{target_width})."
            )
        scale = min(
            float(target_width) / float(width),
            float(target_height) / float(height),
        )
        if not allow_upscale:
            scale = min(scale, 1.0)
        resized_height = max(
            1,
            min(
                target_height,
                int(round(float(height) * scale)),
            ),
        )
        resized_width = max(
            1,
            min(
                target_width,
                int(round(float(width) * scale)),
            ),
        )
        if resized_height == height and resized_width == width:
            resized = frames
        else:
            tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            tensor = F.interpolate(
                tensor,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = tensor.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        canvas = np.zeros(
            (n, target_height, target_width, 3),
            dtype=np.uint8,
        )
        top_pad = max(0, (target_height - resized_height) // 2)
        left_pad = max(0, (target_width - resized_width) // 2)
        canvas[
            :,
            top_pad : top_pad + resized_height,
            left_pad : left_pad + resized_width,
        ] = resized[..., :3]
        return canvas

    raise ValueError(f"Unsupported fit mode: {resolved_fit_mode}")


def transform_frame(
    frame: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    center_crop: bool,
    allow_upscale: bool,
    fit_mode: MixedVideoFrameFitMode | str | None = None,
) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB frame [H,W,3+], got {array.shape}.")
    array = np.ascontiguousarray(array[..., :3])
    resolved_fit_mode = _resolve_frame_fit_mode(
        fit_mode,
        center_crop=center_crop,
    )
    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        array = _center_crop_to_aspect(
            array,
            target_height=target_height,
            target_width=target_width,
        )
        return _resize_frame(
            array,
            target_height=target_height,
            target_width=target_width,
            allow_upscale=allow_upscale,
        )
    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        return _letterbox_pad_to_target(
            array,
            target_height=target_height,
            target_width=target_width,
            allow_upscale=allow_upscale,
        )
    raise ValueError(f"Unsupported mixed-video frame fit mode: {resolved_fit_mode}")


def _resolve_frame_fit_mode(
    fit_mode: MixedVideoFrameFitMode | str | None,
    *,
    center_crop: bool,
) -> MixedVideoFrameFitMode:
    if fit_mode is not None:
        return (
            fit_mode
            if isinstance(fit_mode, MixedVideoFrameFitMode)
            else MixedVideoFrameFitMode(str(fit_mode))
        )
    return (
        MixedVideoFrameFitMode.CENTER_CROP
        if center_crop
        else MixedVideoFrameFitMode.LETTERBOX_PAD
    )


def _resize_frame(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
) -> np.ndarray:
    input_height, input_width = int(array.shape[0]), int(array.shape[1])
    if not allow_upscale and (
        input_height < target_height or input_width < target_width
    ):
        return array
    if input_height == target_height and input_width == target_width:
        return array
    image = Image.fromarray(array)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((target_width, target_height), resampling)
    return np.asarray(resized, dtype=np.uint8)


def _letterbox_pad_to_target(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
) -> np.ndarray:
    input_height, input_width = int(array.shape[0]), int(array.shape[1])
    if input_height <= 0 or input_width <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError(
            "Mixed-video letterbox resize expects positive dimensions, "
            f"got input=({input_height}, {input_width}) "
            f"target=({target_height}, {target_width})."
        )
    scale = min(
        float(target_width) / float(input_width),
        float(target_height) / float(input_height),
    )
    if not allow_upscale:
        scale = min(scale, 1.0)
    resized_height = max(
        1,
        min(
            int(target_height),
            int(round(float(input_height) * scale)),
        ),
    )
    resized_width = max(
        1,
        min(
            int(target_width),
            int(round(float(input_width) * scale)),
        ),
    )
    if resized_height == input_height and resized_width == input_width:
        resized = array
    else:
        image = Image.fromarray(array)
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        resized = np.asarray(
            image.resize((resized_width, resized_height), resampling),
            dtype=np.uint8,
        )
    canvas = np.zeros(
        (int(target_height), int(target_width), 3),
        dtype=np.uint8,
    )
    top = max(0, (int(target_height) - int(resized_height)) // 2)
    left = max(0, (int(target_width) - int(resized_width)) // 2)
    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized[..., :3]
    return canvas


def _center_crop_to_aspect(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    height, width = int(array.shape[0]), int(array.shape[1])
    target_aspect = target_width / target_height
    current_aspect = width / height
    if abs(current_aspect - target_aspect) < 1e-6:
        return array
    if current_aspect > target_aspect:
        crop_width = max(1, int(round(height * target_aspect)))
        left = max(0, (width - crop_width) // 2)
        return array[:, left : left + crop_width]
    crop_height = max(1, int(round(width / target_aspect)))
    top = max(0, (height - crop_height) // 2)
    return array[top : top + crop_height, :]


def select_mixed_video_resize_bin(
    bins: Sequence[MixedVideoResizeBinConfig],
    *,
    source_height: int,
    source_width: int,
) -> MixedVideoResizeBinConfig:
    """Choose the closest log-aspect ratio, then the first fitting pixel tier."""
    if source_height <= 0 or source_width <= 0:
        raise ValueError(
            "Mixed-video source dimensions must be positive, "
            f"got height={source_height}, width={source_width}."
        )
    if not bins:
        raise ValueError("At least one mixed-video resize bin is required.")
    source_ratio = float(source_width) / float(source_height)
    ranked = sorted(
        bins,
        key=lambda bin_config: (
            abs(math.log(source_ratio / bin_config.aspect_ratio)),
            (
                float("inf")
                if bin_config.max_pixels is None
                else float(bin_config.max_pixels)
            ),
        ),
    )
    best_distance = abs(math.log(source_ratio / ranked[0].aspect_ratio))
    aspect_candidates = [
        bin_config
        for bin_config in ranked
        if abs(math.log(source_ratio / bin_config.aspect_ratio)) <= best_distance + 1e-6
    ]
    source_pixels = int(source_height) * int(source_width)
    for bin_config in aspect_candidates:
        if bin_config.max_pixels is None or source_pixels <= int(bin_config.max_pixels):
            return bin_config
    return aspect_candidates[-1]


__all__ = [
    "MixedVideoResolvedDecodeSize",
    "resolve_mixed_video_decode_size",
    "select_mixed_video_resize_bin",
    "transform_frame",
]
