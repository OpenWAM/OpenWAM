"""Render and persist LIBERO rollout artifacts independently from execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

from open_wam.integrations import LIBERO_ROLLOUT_VIEW_KEYS
from open_wam.pipelines import VariantPipeline


LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS

__all__ = [
    "LiberoRolloutArtifactIdentity",
    "LiberoRolloutArtifactOptions",
    "LiberoRolloutArtifactOutput",
    "LiberoRolloutArtifactPayload",
    "append_predicted_latent_chunk",
    "build_libero_rollout_output_path",
    "decode_latent_video_chunks",
    "extract_predicted_latents",
    "iter_comparison_video_frames",
    "iter_rollout_video_frames",
    "persist_libero_rollout_artifacts",
    "to_uint8",
    "with_title",
    "write_video_frames",
]


@dataclass(frozen=True)
class LiberoRolloutArtifactIdentity:
    """Stable coordinates used to derive one episode's artifact paths."""

    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    success: bool
    suffix: str


@dataclass(frozen=True)
class LiberoRolloutArtifactOptions:
    """Output and video choices independent from simulator execution."""

    output_root: Path
    video_fps: float
    save_rollout_video: bool = False
    skip_comparison_video: bool = False


@dataclass(frozen=True)
class LiberoRolloutArtifactPayload:
    """Episode traces consumed only by artifact rendering and persistence."""

    real_observations: Sequence[Mapping[str, np.ndarray]]
    predicted_latent_chunks: Sequence[torch.Tensor]
    action_trace: Sequence[np.ndarray]
    chunk_events: Sequence[Mapping[str, object]]
    component_report: Mapping[str, object]


@dataclass(frozen=True)
class LiberoRolloutArtifactOutput:
    """Persisted artifact paths plus the path-enriched legacy summary."""

    summary: dict[str, object]
    summary_path: Path
    action_trace_path: Path
    chunk_events_path: Path
    component_report_path: Path
    comparison_video_path: Path | None
    rollout_video_path: Path | None


def persist_libero_rollout_artifacts(
    *,
    pipeline: VariantPipeline,
    identity: LiberoRolloutArtifactIdentity,
    options: LiberoRolloutArtifactOptions,
    payload: LiberoRolloutArtifactPayload,
    summary: Mapping[str, object],
    decode_device: torch.device,
) -> LiberoRolloutArtifactOutput:
    """Render and persist one rollout while preserving the legacy file schema."""

    output_path = build_libero_rollout_output_path(
        root=options.output_root,
        identity=identity,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_video_path: Path | None = None
    if not options.skip_comparison_video:
        imagined_video = decode_latent_video_chunks(
            pipeline,
            payload.predicted_latent_chunks,
            decode_device=decode_device,
            restore_vae=False,
        )
        write_video_frames(
            output_path,
            iter_comparison_video_frames(
                real_observations=payload.real_observations,
                imagined_video=imagined_video,
            ),
            fps=options.video_fps,
        )
        comparison_video_path = output_path.resolve()

    rollout_video_path: Path | None = None
    if options.save_rollout_video:
        rollout_video_path = output_path.with_name(
            f"{output_path.stem}_rollout.mp4"
        )
        write_video_frames(
            rollout_video_path,
            iter_rollout_video_frames(
                real_observations=payload.real_observations,
            ),
            fps=options.video_fps,
        )
        rollout_video_path = rollout_video_path.resolve()

    resolved_summary = dict(summary)
    resolved_comparison_path = (
        None
        if comparison_video_path is None
        else str(comparison_video_path)
    )
    resolved_summary["video_path"] = resolved_comparison_path
    resolved_summary["comparison_video_path"] = resolved_comparison_path
    resolved_summary["rollout_video_path"] = (
        None if rollout_video_path is None else str(rollout_video_path)
    )

    summary_path = output_path.with_suffix(".json")
    action_trace_path = output_path.with_name(
        f"{output_path.stem}_actions.jsonl"
    )
    chunk_events_path = output_path.with_name(
        f"{output_path.stem}_chunks.json"
    )
    component_report_path = output_path.with_name(
        f"{output_path.stem}_load_report.json"
    )
    resolved_summary["action_trace_path"] = str(action_trace_path.resolve())

    summary_path.write_text(
        json.dumps(resolved_summary, indent=2),
        encoding="utf-8",
    )
    _write_action_trace(action_trace_path, payload.action_trace)
    chunk_events_path.write_text(
        json.dumps(list(payload.chunk_events), indent=2, default=str),
        encoding="utf-8",
    )
    component_report_path.write_text(
        json.dumps(dict(payload.component_report), indent=2),
        encoding="utf-8",
    )
    return LiberoRolloutArtifactOutput(
        summary=resolved_summary,
        summary_path=summary_path,
        action_trace_path=action_trace_path,
        chunk_events_path=chunk_events_path,
        component_report_path=component_report_path,
        comparison_video_path=comparison_video_path,
        rollout_video_path=rollout_video_path,
    )


def build_libero_rollout_output_path(
    *,
    root: Path,
    identity: LiberoRolloutArtifactIdentity,
) -> Path:
    """Resolve the maintained per-task, per-episode artifact path."""

    safe_prompt = identity.prompt.replace(" ", "_")
    return (
        root
        / identity.benchmark
        / f"{identity.task_id}_{safe_prompt}"
        / (
            f"{identity.episode_idx}_{identity.success}_"
            f"{identity.suffix}.mp4"
        )
    )


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
        retained_frames = sum(
            int(chunk.shape[2]) for chunk in predicted_latent_chunks
        )
        if retained_frames >= cap:
            return
        predicted_latents = predicted_latents[
            :, :, : cap - retained_frames
        ]
    if int(predicted_latents.shape[2]) <= 0:
        return
    predicted_latent_chunks.append(predicted_latents.detach().cpu())


def extract_predicted_latents(infer_output: Any) -> torch.Tensor | None:
    """Read optional imagined latents from decoder or policy diagnostics."""

    predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        predicted_latents = infer_output.policy_output.aux.get(
            "predicted_latents"
        )
    return (
        predicted_latents
        if isinstance(predicted_latents, torch.Tensor)
        else None
    )


def write_video_frames(
    output_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
) -> None:
    """Stream contiguous frames to one imageio video writer."""

    wrote_frame = False
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
            wrote_frame = True
    if not wrote_frame:
        raise ValueError(
            f"No frames were produced for video output {output_path}."
        )


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
            "MoT Rollout (AgentView / Wrist)",
        )
        yield np.ascontiguousarray(np.array(titled, copy=True))


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
        agentview = np.ascontiguousarray(
            real_observation[LIBERO_OBS_KEYS[0]]
        )
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


def decode_latent_video_chunks(
    pipeline: VariantPipeline,
    latent_chunks: Sequence[torch.Tensor],
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    """Decode accumulated imagined latents when the frontend has a VAE."""

    if not latent_chunks:
        return None
    return _decode_latent_video(
        pipeline,
        torch.cat(tuple(latent_chunks), dim=2),
        decode_device=decode_device,
        restore_vae=restore_vae,
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


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize an RGB array to contiguous display-ready uint8 values."""

    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


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
            round(
                frame_index
                * (imagined_frame_count - 1)
                / (target_length - 1)
            )
        )
    imagined_index = max(
        0,
        min(imagined_frame_count - 1, imagined_index),
    )
    return np.array(imagined_video[imagined_index], copy=True)


def _decode_latent_video(
    pipeline: VariantPipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = (
        torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    )
    if original_device != decode_device or original_dtype != target_dtype:
        vae = vae.to(device=decode_device, dtype=target_dtype)
    latents = latents.to(device=decode_device, dtype=target_dtype)
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    imagined_video = video_processor.postprocess_video(
        decoded,
        output_type="np",
    )[0]
    if (
        restore_vae
        and (
            next(assets.vae.parameters()).device != original_device
            or next(assets.vae.parameters()).dtype != original_dtype
        )
    ):
        assets.vae = assets.vae.to(
            device=original_device,
            dtype=original_dtype,
        )
    return imagined_video


def _write_action_trace(
    path: Path,
    actions: Sequence[np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for action_index, action in enumerate(actions):
            handle.write(
                json.dumps(
                    {
                        "action_index": int(action_index),
                        "action": np.asarray(
                            action,
                            dtype=np.float32,
                        ).tolist(),
                    }
                )
                + "\n"
            )
