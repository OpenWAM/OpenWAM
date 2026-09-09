"""Trim only near-black outer columns while preserving the full native height."""

import numpy as np


def inner_box(frames):
    h, w = frames[0].shape[:2]
    assert (w, h) == (1280, 720), "Unreviewed FastUMI native geometry"
    visible = np.maximum.reduce([f.max(axis=2) for f in frames]) > 32
    columns = np.flatnonzero(visible.sum(axis=0) > 8)
    if len(columns):
        # Keep a 24-pixel guard and at least 70 percent of the native width.
        # Never crop the top/bottom to hide circular-lens corners.
        x0 = max(0, min(192, int(columns[0]) - 24)) // 2 * 2
        x1 = min(w, max(w - 192, int(columns[-1]) + 25))
        x1 = min(w, (x1 + 1) // 2 * 2)
    else:
        x0, x1 = 0, w
    return [x0, 0, x1, h], {
        "retained_native_area_fraction": (x1 - x0) / w,
        "full_height_preserved": True,
        "max_side_trim_pixels": 192,
        "guard_pixels": 24,
        "policy": "full native height; near-black outer columns only; circular lens corners retained",
    }
