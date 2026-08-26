"""Benchmark-independent latent decoding and video persistence helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers.video_processor import VideoProcessor

from open_wam.pipelines import VariantPipeline


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
        raise ValueError(f"No frames were produced for video output {output_path}.")


def decode_latent_video_chunks(
    pipeline: VariantPipeline,
    latent_chunks: Sequence[torch.Tensor],
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    """Decode accumulated latent chunks when the frontend has a VAE."""

    if not latent_chunks:
        return None
    return _decode_latent_video(
        pipeline,
        torch.cat(tuple(latent_chunks), dim=2),
        decode_device=decode_device,
        restore_vae=restore_vae,
    )


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize an RGB array to contiguous display-ready uint8 values."""

    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _decode_latent_video(
    pipeline: VariantPipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
    restore_vae: bool,
) -> np.ndarray | None:
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    try:
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
        return video_processor.postprocess_video(decoded, output_type="np")[0]
    finally:
        current_parameter = next(assets.vae.parameters())
        if restore_vae and (
            current_parameter.device != original_device
            or current_parameter.dtype != original_dtype
        ):
            assets.vae = assets.vae.to(
                device=original_device,
                dtype=original_dtype,
            )


__all__ = ["decode_latent_video_chunks", "to_uint8", "write_video_frames"]
