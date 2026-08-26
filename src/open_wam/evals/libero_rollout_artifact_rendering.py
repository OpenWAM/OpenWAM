"""Frame composition and latent decoding for LIBERO rollout artifacts."""

# Preserve exact renderer ASTs and frame-index rounding semantics.
# ruff: noqa: RUF046

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from open_wam.evals.video_artifacts import (
    to_uint8,
)
from open_wam.integrations import LIBERO_ROLLOUT_VIEW_KEYS

LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS


def append_predicted_latent_chunk(
    predicted_latent_chunks: list[torch.Tensor],
    predicted_latents: torch.Tensor,
    *,
    max_imagined_latent_frames: int | None,
) -> None:
    """Append a detached CPU latent chunk without exceeding an artifact cap."""

    if predicted_latents.ndim != 5:
        raise ValueError(
            "Predicted latent chunks must have shape [B, C, T, H, W], "
            f"got {tuple(predicted_latents.shape)}."
        )
    if max_imagined_latent_frames is not None:
        cap = int(max_imagined_latent_frames)
        if cap <= 0:
            return
        retained_frames = sum(int(chunk.shape[2]) for chunk in predicted_latent_chunks)
        if retained_frames >= cap:
            return
        predicted_latents = predicted_latents[:, :, : cap - retained_frames]
    if int(predicted_latents.shape[2]) <= 0:
        return
    predicted_latent_chunks.append(predicted_latents.detach().cpu())


def extract_predicted_latents(infer_output: Any) -> torch.Tensor | None:
    """Read optional imagined latents from decoder or policy diagnostics."""

    predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
    return predicted_latents if isinstance(predicted_latents, torch.Tensor) else None


def iter_rollout_video_frames(
    *,
    real_observations: Sequence[Mapping[str, np.ndarray]],
) -> Iterable[np.ndarray]:
    """Yield titled agent-view/wrist rows for the real rollout."""

    for observation in real_observations:
        agentview = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(observation[LIBERO_OBS_KEYS[1]])
        real_row = np.ascontiguousarray(np.hstack([agentview, wrist]))
        titled = with_title(
            Image.fromarray(real_row),
            "DualExpert Rollout (AgentView / Wrist)",
        )
        yield np.ascontiguousarray(np.array(titled, copy=True))


def build_libero_realtime_video_frames(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    target_action_hz: float,
    action_per_frame: int,
) -> list[np.ndarray]:
    """Render the established live-observation realtime video layout."""

    frames: list[np.ndarray] = []
    for record in action_video_records:
        obs = record["obs"]
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        titled = with_title(
            Image.fromarray(np.ascontiguousarray(row_real)),
            "Live LIBERO (AgentView / Wrist)",
        )
        info_panel = Image.new("RGB", (titled.width, 108), color=(0, 0, 0))
        draw = ImageDraw.Draw(info_panel)
        header = (
            f"Action {int(record['action_index']) + 1} | "
            f"Frame {int(record['absolute_frame_index'])} "
            f"[{int(record['action_offset']) + 1}/{action_per_frame}]"
        )
        source = str(record["source"])
        lag_text = (
            "fallback"
            if record["generation_lag_frames"] is None
            else str(int(record["generation_lag_frames"]))
        )
        lines = [
            header,
            (
                f"Source: {source} | Target: {target_action_hz:.1f} Hz | "
                f"Lateness: {1000.0 * float(record['lateness_s']):.1f} ms"
            ),
            (
                f"Env step: {1000.0 * float(record['env_step_s']):.1f} ms | "
                f"Generation lag: {lag_text} frame(s)"
            ),
        ]
        text_color = (255, 255, 255) if source == "policy" else (255, 180, 120)
        for index, line in enumerate(lines):
            draw.text((10, 10 + index * 28), line, fill=text_color)
        full_frame = np.vstack(
            [np.array(titled, copy=True), np.array(info_panel, copy=True)]
        )
        frames.append(np.ascontiguousarray(full_frame))
    return frames


def build_libero_fallback_timeline_video_frames(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    target_action_hz: float,
    action_per_frame: int,
) -> list[np.ndarray]:
    """Render the established fallback-aware realtime timeline layout."""

    frames: list[np.ndarray] = []
    for record_index, record in enumerate(action_video_records):
        obs = record["obs"]
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        titled = with_title(
            Image.fromarray(np.ascontiguousarray(row_real)),
            "Live LIBERO fallback timeline (AgentView / Wrist)",
        )
        source_color = _fallback_timeline_record_color(record)
        bordered = Image.new(
            "RGB",
            (titled.width + 12, titled.height + 12),
            color=source_color,
        )
        bordered.paste(titled, (6, 6))

        info_panel = Image.new("RGB", (bordered.width, 164), color=(0, 0, 0))
        draw = ImageDraw.Draw(info_panel)
        source = str(record.get("source", "unknown"))
        history_decision = str(record.get("frame_history_decision", "not_recorded"))
        generation_lag = (
            "fallback"
            if record.get("generation_lag_frames") is None
            else f"{int(record['generation_lag_frames'])} frame(s)"
        )
        generation_frame = (
            "NA"
            if record.get("generation_frame_start") is None
            else str(record["generation_frame_start"])
        )
        ready_delay = (
            "NA"
            if record.get("plan_ready_delay_s") is None
            else f"{float(record['plan_ready_delay_s']):.2f}s"
        )
        lines = [
            (
                f"Action {int(record['action_index']) + 1} | "
                f"Frame {int(record['absolute_frame_index'])} "
                f"[{int(record['action_offset']) + 1}/{action_per_frame}] | "
                f"Target {target_action_hz:.1f} Hz"
            ),
            (
                f"Source: {source} | History decision: {history_decision} | "
                "Frame has fallback: "
                f"{bool(record.get('frame_contains_fallback_action', source.startswith('fallback_')))}"
            ),
            (
                f"Gen frame: {generation_frame} | Gen lag: {generation_lag} | "
                f"Ready delay: {ready_delay} | "
                f"Late: {1000.0 * float(record['lateness_s']):.1f} ms"
            ),
            _format_fallback_timeline_action(record.get("action")),
            (
                "Timeline colors: red fallback, orange hidden/washout, "
                "blue history, green extension, violet startup."
            ),
        ]
        for index, line in enumerate(lines):
            fill = source_color if index == 1 else (255, 255, 255)
            draw.text((10, 10 + index * 28), line, fill=fill)

        timeline = _build_fallback_timeline_strip(
            action_video_records=action_video_records,
            width=bordered.width,
            height=42,
            current_index=record_index,
        )
        full_frame = np.vstack(
            [
                np.array(bordered, copy=True),
                np.array(info_panel, copy=True),
                np.array(timeline, copy=True),
            ]
        )
        frames.append(np.ascontiguousarray(full_frame))
    return frames


def iter_comparison_video_frames(
    *,
    real_observations: Sequence[Mapping[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> Iterable[np.ndarray]:
    """Yield the maintained real/imagined comparison layout lazily."""

    panel_height = 300
    target_length = len(real_observations)
    for frame_index, real_observation in enumerate(real_observations):
        imagined_frame = _imagined_frame_for_rollout_index(
            imagined_video=imagined_video,
            frame_index=frame_index,
            target_length=target_length,
        )
        agentview = np.ascontiguousarray(real_observation[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(real_observation[LIBERO_OBS_KEYS[1]])
        real_row = np.ascontiguousarray(np.hstack([agentview, wrist]))
        real_row = np.array(
            with_title(
                Image.fromarray(real_row),
                f"Real Rollout Frame {frame_index}",
            ),
            copy=True,
        )
        target_width = real_row.shape[1]
        if imagined_frame is None:
            imagined_row = Image.new(
                "RGB",
                (target_width, panel_height),
                color=(0, 0, 0),
            )
            draw = ImageDraw.Draw(imagined_row)
            draw.text(
                (10, panel_height // 2),
                "No imagined frame",
                fill=(120, 120, 120),
            )
        else:
            image = Image.fromarray(to_uint8(imagined_frame))
            scale = min(
                target_width / image.width,
                panel_height / image.height,
            )
            resized = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
            )
            imagined_row = Image.new(
                "RGB",
                (target_width, panel_height),
                color=(0, 0, 0),
            )
            imagined_row.paste(
                resized,
                (
                    (target_width - resized.width) // 2,
                    (panel_height - resized.height) // 2,
                ),
            )
        imagined_row = with_title(
            imagined_row,
            f"Imagined Frame {frame_index}",
        )
        yield np.ascontiguousarray(
            np.vstack([real_row, np.array(imagined_row, copy=True)])
        )


def with_title(image: Image.Image, title: str) -> Image.Image:
    """Add a fixed-height title bar without resizing the source image."""

    title_height = 36
    canvas = Image.new(
        "RGB",
        (image.width, image.height + title_height),
        color=(0, 0, 0),
    )
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas


def _imagined_frame_for_rollout_index(
    *,
    imagined_video: np.ndarray | None,
    frame_index: int,
    target_length: int,
) -> np.ndarray | None:
    if imagined_video is None or target_length <= 0:
        return None
    imagined_frame_count = len(imagined_video)
    if imagined_frame_count <= 0:
        return None
    if imagined_frame_count == 1 or target_length == 1:
        imagined_index = 0
    elif imagined_frame_count == target_length:
        imagined_index = frame_index
    else:
        imagined_index = int(
            round(frame_index * (imagined_frame_count - 1) / (target_length - 1))
        )
    imagined_index = max(
        0,
        min(imagined_frame_count - 1, imagined_index),
    )
    return np.array(imagined_video[imagined_index], copy=True)


def _fallback_timeline_record_color(
    record: Mapping[str, Any],
) -> tuple[int, int, int]:
    source = str(record.get("source", ""))
    history_decision = str(record.get("frame_history_decision", ""))
    if source.startswith("fallback_"):
        return (230, 50, 40)
    if history_decision in {"fallback", "washout"}:
        return (235, 165, 35)
    if source == "history_replan":
        return (70, 150, 255)
    if source == "open_loop_extension":
        return (70, 210, 120)
    if source == "startup_plan":
        return (175, 150, 255)
    return (180, 180, 180)


def _format_fallback_timeline_action(action: Any) -> str:
    if action is None:
        return "Action: NA"
    values = [float(value) for value in action]
    delta_values = " ".join(f"{value:+.2f}" for value in values[:6])
    tail_values = " ".join(f"{value:+.2f}" for value in values[6:])
    return f"Action delta[0:6]: {delta_values} | absolute[6:]: {tail_values or 'NA'}"


def _build_fallback_timeline_strip(
    *,
    action_video_records: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    current_index: int,
) -> Image.Image:
    strip = Image.new("RGB", (int(width), int(height)), color=(18, 18, 18))
    draw = ImageDraw.Draw(strip)
    total = max(1, len(action_video_records))
    bar_top = 8
    bar_bottom = int(height) - 10
    for index, record in enumerate(action_video_records):
        x0 = int(index * int(width) / total)
        x1 = max(x0 + 1, int((index + 1) * int(width) / total))
        draw.rectangle(
            [x0, bar_top, min(int(width) - 1, x1), bar_bottom],
            fill=_fallback_timeline_record_color(record),
        )
    current_x = int(current_index * int(width) / total)
    draw.line(
        [(current_x, 0), (current_x, int(height) - 1)],
        fill=(255, 255, 255),
        width=3,
    )
    draw.text(
        (10, int(height) - 10),
        f"{current_index + 1}/{total}",
        fill=(255, 255, 255),
    )
    return strip
