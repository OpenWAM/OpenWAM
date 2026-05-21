from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from open_wam.configs.data import DataConfig, ViewLayoutConfig
from open_wam.configs.enums import MixedVideoDecodeSizeMode


@dataclass(frozen=True)
class ViewPlacement:
    """Placement of a resized camera view inside the canonical canvas."""

    source_name: str
    canonical_name: str
    top: int
    left: int
    height: int
    width: int


@dataclass
class CanonicalVideoBatch:
    """Canonical multi-view video tensor consumed by the shared backbone.

    Attributes:
        video:
            Canonical RGB video of shape [B, 3, T, H, W].
        placements:
            Camera placements inside the canonical canvas.
        metadata:
            Extra shape and layout metadata used by later pipeline stages.
    """

    video: torch.Tensor
    placements: tuple[ViewPlacement, ...]
    metadata: dict[str, Any]


class ConfiguredCanonicalVideoPreprocessor(torch.nn.Module):
    """Build a fixed multi-view RGB canvas from arbitrary raw camera streams.

    The shared video backbone must always receive the same RGB canvas geometry,
    even when the source dataset exposes different camera sets. The data layer
    owns the layout decision by resizing each source view into a declared slot
    inside the canonical canvas.
    """

    def __init__(
        self,
        placements: tuple[ViewPlacement, ...],
        canvas_height: int,
        canvas_width: int,
    ) -> None:
        super().__init__()
        self.placements = placements
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width

    def forward(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        resolved_keys = {
            placement.source_name: self._resolve_view_key(views, placement.source_name)
            for placement in self.placements
        }
        missing = [name for name, key in resolved_keys.items() if key is None]
        if missing:
            raise KeyError(f"Missing camera views for canonical layout: {missing}")

        canonical_views: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        num_frames: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None

        for placement in self.placements:
            canonical = self._to_bcthw(views[resolved_keys[placement.source_name]])
            if batch_size is None:
                batch_size = canonical.shape[0]
                num_frames = canonical.shape[2]
                device = canonical.device
                dtype = canonical.dtype
            else:
                if canonical.shape[0] != batch_size or canonical.shape[2] != num_frames:
                    raise ValueError(
                        "All camera views must share the same batch and time dimensions: "
                        f"expected [B={batch_size}, T={num_frames}], got {canonical.shape}"
                    )
            canonical_views[placement.source_name] = self._resize_video(
                canonical,
                target_height=placement.height,
                target_width=placement.width,
            )

        assert batch_size is not None
        assert num_frames is not None
        assert device is not None
        assert dtype is not None

        canvas = torch.zeros(
            batch_size,
            3,
            num_frames,
            self.canvas_height,
            self.canvas_width,
            device=device,
            dtype=dtype,
        )

        # Write each resized view into its fixed canvas location so the video
        # backbone always sees the same multi-view geometry regardless of source.
        for placement in self.placements:
            canvas[
                :,
                :,
                :,
                placement.top : placement.top + placement.height,
                placement.left : placement.left + placement.width,
            ] = canonical_views[placement.source_name]

        metadata = {
            "canvas_height": self.canvas_height,
            "canvas_width": self.canvas_width,
            "num_frames": num_frames,
            "view_names": tuple(placement.source_name for placement in self.placements),
            "canonical_view_names": tuple(placement.canonical_name for placement in self.placements),
            "resolved_view_names": tuple(resolved_keys[placement.source_name] for placement in self.placements),
        }
        return CanonicalVideoBatch(video=canvas, placements=self.placements, metadata=metadata)

    def _resolve_view_key(self, views: Mapping[str, torch.Tensor], name: str) -> str | None:
        if name in views:
            return name
        suffix_matches = [key for key in views if key.endswith(f".{name}") or key.split(".")[-1] == name]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    def _resize_video(
        self,
        video: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Resize [B, 3, T, H, W] video by flattening batch and time together."""

        batch_size, channels, num_frames, height, width = video.shape
        if (height, width) == (target_height, target_width):
            return video

        flattened = video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        resized = F.interpolate(
            flattened,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, num_frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)

    def _to_bcthw(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a raw RGB view into [B, 3, T, H, W].

        Supported inputs:
        - [T, H, W, C]
        - [B, T, H, W, C]
        - [T, C, H, W]
        - [B, C, T, H, W]
        """

        if tensor.ndim == 4 and tensor.shape[-1] == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 5 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 4, 1, 2, 3)
        elif tensor.ndim == 4 and tensor.shape[1] == 3:
            tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        elif tensor.ndim == 5 and tensor.shape[1] == 3:
            tensor = tensor
        else:
            raise ValueError(
                "Unsupported video tensor shape. Expected one of "
                "[T,H,W,3], [B,T,H,W,3], [T,3,H,W], or [B,3,T,H,W], "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()

        return tensor


class RobotWinCanonicalVideoPreprocessor(ConfiguredCanonicalVideoPreprocessor):
    """Canonical RobotWin layout that mirrors the LingBot RGB composition."""

    def __init__(self) -> None:
        super().__init__(
            placements=(
                ViewPlacement("cam_high", "cam_high", top=0, left=0, height=256, width=320),
                ViewPlacement("cam_left_wrist", "cam_left_wrist", top=256, left=0, height=128, width=160),
                ViewPlacement("cam_right_wrist", "cam_right_wrist", top=256, left=160, height=128, width=160),
            ),
            canvas_height=384,
            canvas_width=320,
        )


class AdaptiveSingleViewCanonicalVideoPreprocessor(ConfiguredCanonicalVideoPreprocessor):
    """Use the already-decoded single view as the canonical VAE input canvas."""

    def __init__(self, *, source_name: str, canonical_name: str) -> None:
        super().__init__(
            placements=(ViewPlacement(source_name, canonical_name, top=0, left=0, height=1, width=1),),
            canvas_height=1,
            canvas_width=1,
        )
        self.source_name = source_name
        self.canonical_name = canonical_name

    def forward(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        resolved_key = self._resolve_view_key(views, self.source_name)
        if resolved_key is None:
            raise KeyError(f"Missing camera view for adaptive canonical layout: {self.source_name!r}")
        canonical = self._to_bcthw(views[resolved_key])
        height = int(canonical.shape[-2])
        width = int(canonical.shape[-1])
        placement = ViewPlacement(
            self.source_name,
            self.canonical_name,
            top=0,
            left=0,
            height=height,
            width=width,
        )
        return CanonicalVideoBatch(
            video=canonical,
            placements=(placement,),
            metadata={
                "canvas_height": height,
                "canvas_width": width,
                "num_frames": int(canonical.shape[2]),
                "view_names": (self.source_name,),
                "canonical_view_names": (self.canonical_name,),
                "resolved_view_names": (resolved_key,),
                "adaptive_canvas": True,
            },
        )


def build_canonical_video_preprocessor(data_config: DataConfig) -> ConfiguredCanonicalVideoPreprocessor:
    """Construct the canonicalizer from the experiment data config.

    The resulting preprocessor is the only component that should translate
    dataset-specific camera names and placements into the fixed RGB canvas seen
    by the shared video backbone.
    """

    if getattr(data_config, "decode_size_mode", None) == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS:
        if len(data_config.view_layout) != 1:
            raise ValueError(
                "`decode_size_mode=aspect_ratio_bins` currently supports one mixed-video view per sample. "
                f"Got {len(data_config.view_layout)} view placements."
            )
        view = data_config.view_layout[0]
        return AdaptiveSingleViewCanonicalVideoPreprocessor(
            source_name=view.source_name,
            canonical_name=view.canonical_name,
        )

    placements = tuple(_view_layout_to_placement(view) for view in data_config.view_layout)
    return ConfiguredCanonicalVideoPreprocessor(
        placements=placements,
        canvas_height=data_config.canonical_height,
        canvas_width=data_config.canonical_width,
    )


def _view_layout_to_placement(view: ViewLayoutConfig) -> ViewPlacement:
    return ViewPlacement(
        source_name=view.source_name,
        canonical_name=view.canonical_name,
        top=view.top,
        left=view.left,
        height=view.height,
        width=view.width,
    )
