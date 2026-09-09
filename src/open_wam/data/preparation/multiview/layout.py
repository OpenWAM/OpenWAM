"""Aspect-preserving RGB multi view layouts with a bounded, VAE-compatible canvas."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

POLICY = "rgb_multi_view_65k_portable_v2"


@dataclass(frozen=True)
class Placement:
    camera: str
    source_width: int
    source_height: int
    scale: float
    left: float
    top: float
    rotation_degrees: int = 0


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    kind: str
    placements: tuple[Placement, ...]

    def to_dict(self):
        return {"policy": POLICY, **asdict(self)}


def make_layout(
    views: list[tuple[str, int, int]],
    target_pixels=256 * 256,
    pair_orientation="auto",
    camera_rotations=None,
) -> Layout:
    """Views are ordered (head,left,right), or canonical pair order.

    Each placement uses one real-valued scale for BOTH axes. The rasterizer
    applies a uniform affine transform, avoiding separate rounded width/height
    resize factors. Alignment padding is outside the complete source images.
    """
    if len(views) not in (2, 3) or len({v[0] for v in views}) != len(views):
        raise ValueError("Exactly two or three distinct cameras are required")
    if any(w < 1 or h < 1 for _, w, h in views):
        raise ValueError("Invalid source geometry")
    ratios = [w / h for _, w, h in views]
    boxes = []
    if pair_orientation not in ("auto", "horizontal", "vertical"):
        raise ValueError("Unknown pair orientation")
    if len(views) == 2 and (
        pair_orientation == "vertical"
        or (pair_orientation == "auto" and all(r > 1 for r in ratios))
    ):
        kind = "pair_vertical"
        width, height = 1.0, sum(1 / r for r in ratios)
        y = 0.0
        for r in ratios:
            boxes.append((0.0, y, 1.0, 1 / r))
            y += 1 / r
    elif len(views) == 2:
        # Portrait and square pairs share a common height. Mixed orientation
        # also uses this deterministic rule, preserving both image ratios.
        kind = "pair_horizontal"
        width, height = sum(ratios), 1.0
        x = 0.0
        for r in ratios:
            boxes.append((x, 0.0, r, 1.0))
            x += r
    else:
        kind = "head_above_left_right"
        width = ratios[1] + ratios[2]
        head_height = width / ratios[0]
        height = head_height + 1.0
        boxes = [
            (0.0, 0.0, width, head_height),
            (0.0, head_height, ratios[1], 1.0),
            (ratios[1], head_height, ratios[2], 1.0),
        ]
    choices = []
    for out_h in range(64, 1025, 32):
        for out_w in range(64, 1025, 32):
            area = out_h * out_w
            if 0.8 * target_pixels <= area <= 1.2 * target_pixels:
                scale = min(out_w / width, out_h / height)
                padding = 1 - width * height * scale * scale / area
                score = padding + 0.05 * abs(math.log(area / target_pixels))
                choices.append((score, abs(area - target_pixels), out_h, out_w))
    _, _, out_h, out_w = min(choices)
    total_scale = min(out_w / width, out_h / height)
    dx, dy = (out_w - width * total_scale) / 2, (out_h - height * total_scale) / 2
    placements = []
    for (camera, w, h), (x, y, bw, bh) in zip(views, boxes):
        scale = bw * total_scale / w
        assert math.isclose(scale, bh * total_scale / h, rel_tol=1e-10)
        rotation = (camera_rotations or {}).get(camera, 0)
        if rotation not in (0, 180):
            raise ValueError(
                "Only audited 180-degree orientation corrections are supported"
            )
        placements.append(
            Placement(
                camera,
                w,
                h,
                scale,
                dx + x * total_scale,
                dy + y * total_scale,
                rotation,
            )
        )
    return Layout(out_w, out_h, kind, tuple(placements))


def render_frame(frames, layout: Layout):
    import cv2
    import numpy as np

    output = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    for p in layout.placements:
        frame = frames[p.camera]
        if (
            frame.shape != (p.source_height, p.source_width, 3)
            or frame.dtype != np.uint8
        ):
            raise ValueError(f"Camera geometry or dtype changed: {p.camera}")
        if p.rotation_degrees == 180:
            frame = np.ascontiguousarray(frame[::-1, ::-1])
        # Source pixel centers map using a single uniform scale. Warp each
        # camera only into its own rectangle so adjacent views cannot overlap.
        affine = np.array(
            [
                [p.scale, 0, p.left + (p.scale - 1) / 2],
                [0, p.scale, p.top + (p.scale - 1) / 2],
            ],
            dtype=np.float64,
        )
        warped = cv2.warpAffine(
            frame,
            affine,
            (layout.width, layout.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        x0 = max(0, math.ceil(p.left - 0.5))
        y0 = max(0, math.ceil(p.top - 0.5))
        x1 = min(layout.width, math.ceil(p.left + p.source_width * p.scale - 0.5))
        y1 = min(layout.height, math.ceil(p.top + p.source_height * p.scale - 0.5))
        output[y0:y1, x0:x1] = warped[y0:y1, x0:x1]
    return output
