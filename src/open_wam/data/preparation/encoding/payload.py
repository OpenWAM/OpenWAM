"""Stored latent tensor and provenance contract shared by encoding adapters."""

from pathlib import PurePosixPath

import torch

from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_payload(
    *,
    latent: torch.Tensor,
    camera: str,
    episode_index: int,
    indices: list[int],
    source_frames: int,
    source_fps: float,
    bin_config: MixedVideoResizeBinConfig,
    fps: float,
    store_dtype: str,
    fit_mode: str,
    normalize_latents: bool,
    vae_path: str,
) -> dict:
    tensor = latent.to(DTYPES[store_dtype])
    return {
        # Consumed by the training-time loader.
        "latent": tensor,
        "latent_layout": "THWC",
        "latent_num_frames": int(tensor.shape[0]),
        "latent_height": int(tensor.shape[1]),
        "latent_width": int(tensor.shape[2]),
        "frame_ids": [int(i) for i in indices],
        # Source-timeline span, so the latent stays aligned with the action rows.
        "start_frame": 0,
        "end_frame": int(source_frames),
        "video_num_frames": int(len(indices)),
        "fps": float(fps),
        "ori_fps": float(source_fps),
        "video_height": int(bin_config.target_height),
        "video_width": int(bin_config.target_width),
        "resize_bin": bin_config.name,
        "camera": camera,
        "episode_index": int(episode_index),
        "fit_mode": fit_mode,
        "latents_normalized": normalize_latents,
        # Identify the VAE without embedding a machine-specific location.
        "vae_id": PurePosixPath(str(vae_path).rstrip("/")).name,
    }
