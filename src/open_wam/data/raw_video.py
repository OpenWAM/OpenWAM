from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ViewPlacement:
    """Placement of a resized camera view inside the canonical canvas."""

    name: str
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


class RobotWinCanonicalVideoPreprocessor(torch.nn.Module):
    """Build the canonical RobotWin multi-view canvas from raw RGB views.

    The layout mirrors the LingBot view composition in RGB space:

    - `cam_high` occupies the full top row at 256x320
    - `cam_left_wrist` occupies the bottom-left at 128x160
    - `cam_right_wrist` occupies the bottom-right at 128x160

    The resulting canonical canvas is 384x320 before latentization.
    With a 16x spatial latentizer this maps to 24x20 latent tokens, which is
    the same geometry expected by the LingBot-style backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.placements = (
            ViewPlacement("cam_high", top=0, left=0, height=256, width=320),
            ViewPlacement("cam_left_wrist", top=256, left=0, height=128, width=160),
            ViewPlacement("cam_right_wrist", top=256, left=160, height=128, width=160),
        )
        self.canvas_height = 384
        self.canvas_width = 320

    def forward(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        missing = [placement.name for placement in self.placements if placement.name not in views]
        if missing:
            raise KeyError(f"Missing camera views for canonical layout: {missing}")

        canonical_views: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        num_frames: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None

        for placement in self.placements:
            canonical = self._to_bcthw(views[placement.name])
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
            canonical_views[placement.name] = self._resize_video(
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
            ] = canonical_views[placement.name]

        metadata = {
            "canvas_height": self.canvas_height,
            "canvas_width": self.canvas_width,
            "num_frames": num_frames,
            "view_names": tuple(placement.name for placement in self.placements),
        }
        return CanonicalVideoBatch(video=canvas, placements=self.placements, metadata=metadata)

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

