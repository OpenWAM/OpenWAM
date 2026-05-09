from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch


def decode_latent_video(
    pipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
) -> np.ndarray | None:
    """Decode `[B, C, F, H, W]` reference latents to `[F, H, W, C]` RGB floats."""

    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    from diffusers.video_processor import VideoProcessor

    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    try:
        if original_device != decode_device or original_dtype != target_dtype:
            vae = vae.to(device=decode_device, dtype=target_dtype)
        decode_latents = latents.to(device=decode_device, dtype=target_dtype)
        decode_latents = _denormalize_reference_latents(decode_latents, vae)
        with torch.inference_mode():
            decoded = vae.decode(decode_latents, return_dict=False)[0]
        return video_processor.postprocess_video(decoded, output_type="np")[0]
    finally:
        if next(assets.vae.parameters()).device != original_device or next(assets.vae.parameters()).dtype != original_dtype:
            assets.vae = assets.vae.to(device=original_device, dtype=original_dtype)


def write_prediction_video(
    *,
    output_path: Path,
    target_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
    title: str,
    fps: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_frames = [_to_uint8(frame) for frame in target_rgb]
    predicted_frames = [_to_uint8(frame) for frame in predicted_rgb]
    frames: list[np.ndarray] = []
    for index in range(max(len(target_frames), len(predicted_frames))):
        target = target_frames[index] if index < len(target_frames) else np.zeros_like(predicted_frames[0])
        predicted = predicted_frames[index] if index < len(predicted_frames) else np.zeros_like(target_frames[0])
        if target.shape[:2] != predicted.shape[:2]:
            predicted = np.asarray(Image.fromarray(predicted).resize((target.shape[1], target.shape[0])))
        combined = np.concatenate([target, predicted], axis=1)
        frame_title = f"{title} | frame {index} | left=gt right=pred"
        frames.append(np.asarray(_add_title_bar(Image.fromarray(combined), frame_title)))
    imageio.mimsave(output_path, frames, fps=float(fps), macro_block_size=1)
    return output_path


def _denormalize_reference_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        mean = torch.tensor(latents_mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return (latents.float() * std + mean).to(dtype=latents.dtype)
    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is not None:
        return latents / float(scaling_factor)
    return latents


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    value = np.asarray(frame)
    if value.size and float(np.nanmax(value)) <= 1.0001:
        return (np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(value, 0.0, 255.0).astype(np.uint8)


def _add_title_bar(image: Image.Image, title: str) -> Image.Image:
    title_height = 36
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(0, 0, 0))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas
