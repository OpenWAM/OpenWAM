from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-layer config shared by all future action heads."""

    video_num_inference_steps: int = 25
    action_num_inference_steps: int = 50
    frame_chunk_size: int = 2
    use_cache: bool = True
    guidance_scale: float = 1.0

