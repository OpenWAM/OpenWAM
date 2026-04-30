from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)

_CHUNK_COLORS = (
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (230, 126, 34),
    (155, 89, 182),
    (26, 188, 156),
    (231, 76, 60),
)


def save_libero_rollout_video(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    save_path: Path,
    fps: float,
    frame_chunk_ids: list[int] | None = None,
    title: str = "LingBot-VA LIBERO Rollout",
) -> None:
    if not real_obs_list:
        return

    final_frames: list[np.ndarray] = []
    for index, obs in enumerate(real_obs_list):
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        frame = np.hstack([agentview, wrist]).astype(np.uint8)
        chunk_id = frame_chunk_ids[index] if frame_chunk_ids and index < len(frame_chunk_ids) else None
        frame = _add_title_bar(frame, _format_title(title, index=index, chunk_id=chunk_id))
        if chunk_id is not None:
            frame = _add_chunk_strip(frame, chunk_id)
        final_frames.append(np.ascontiguousarray(frame))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), final_frames, fps=fps, macro_block_size=1)


def _format_title(title: str, *, index: int, chunk_id: int | None) -> str:
    chunk_label = "init" if chunk_id == -1 else ("unknown" if chunk_id is None else str(chunk_id))
    return f"{title} | frame={index} | chunk={chunk_label}"


def _add_title_bar(img: np.ndarray, text: str, font_scale: float = 0.45, thickness: int = 1) -> np.ndarray:
    _, width, _ = img.shape
    bar_height = 30
    title_bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    cv2.putText(
        title_bar,
        text[:96],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, img])


def _add_chunk_strip(img: np.ndarray, chunk_id: int) -> np.ndarray:
    height, width, _ = img.shape
    strip_height = 8
    color = (140, 140, 140) if chunk_id < 0 else _CHUNK_COLORS[chunk_id % len(_CHUNK_COLORS)]
    strip = np.zeros((strip_height, width, 3), dtype=np.uint8)
    strip[:] = color
    del height
    return np.vstack([img, strip])
