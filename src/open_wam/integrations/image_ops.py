"""Small image transforms shared by simulator renderers."""

from __future__ import annotations

import numpy as np


def resize_nearest_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    """Resize to a height with nearest-neighbor sampling and rounded width."""

    frame = np.asarray(frame)
    if frame.shape[0] == target_h:
        return frame
    scale = target_h / frame.shape[0]
    target_w = max(1, int(round(frame.shape[1] * scale)))
    y_indices = np.clip((np.arange(target_h) / scale).astype(np.int64), 0, frame.shape[0] - 1)
    x_indices = np.clip((np.arange(target_w) / scale).astype(np.int64), 0, frame.shape[1] - 1)
    return frame[y_indices][:, x_indices]
