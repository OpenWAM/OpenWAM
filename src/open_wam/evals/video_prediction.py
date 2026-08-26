"""Evaluation contracts for causal video-only prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from open_wam.contracts import CanonicalViewLayout, ViewPlacement
from open_wam.data import LatentWAMBatch
from open_wam.evals.video_artifacts import decode_latent_video_chunks
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.pipelines import VariantPipeline, VariantRolloutRunner


@dataclass(frozen=True)
class CausalVideoPredictionRollout:
    """Latent outputs and alignment metadata for one open-loop rollout."""

    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    observed_latent_frames: int
    future_latent_frames: int
    context_latent_frames: tuple[int, ...]
    first_chunk_future_mse: float


def rollout_causal_video_prediction(
    pipeline: VariantPipeline,
    batch: LatentWAMBatch,
    *,
    num_chunks: int = 1,
) -> CausalVideoPredictionRollout:
    """Generate fixed-width future chunks from one adapter-produced sample.

    The first chunk exactly matches the sample's latent prefix/suffix contract.
    Additional chunks re-feed the complete generated latent history and are
    open-loop extrapolations without dataset targets.
    """

    if int(num_chunks) <= 0:
        raise ValueError(f"`num_chunks` must be positive, got {num_chunks}.")
    if int(batch.video_latents.shape[0]) != 1 or len(batch.metadata) != 1:
        raise ValueError("Causal video rollout currently requires batch size 1.")

    metadata = batch.metadata[0]
    observed_frames = _positive_frame_count(metadata, "observed_prefix_frames")
    future_frames = _positive_frame_count(metadata, "future_suffix_frames")
    total_frames = observed_frames + future_frames
    valid_frames = int(metadata.get("valid_video_frames", total_frames))
    if valid_frames != total_frames:
        raise ValueError(
            "Causal video sample metadata must define one exact prefix/suffix "
            f"window: observed={observed_frames}, future={future_frames}, "
            f"valid={valid_frames}."
        )
    if total_frames > int(batch.video_latents.shape[2]):
        raise ValueError(
            "Causal video sample exceeds its latent tensor: "
            f"required={total_frames}, available={batch.video_latents.shape[2]}."
        )

    target_latents = batch.video_latents[:, :, :total_frames]
    context_latents = target_latents[:, :, :observed_frames]
    context_sizes = [observed_frames]
    first_chunk_future_mse: float | None = None
    runner = VariantRolloutRunner(pipeline)
    session = runner.reset(
        task_text=batch.task_text,
        text_context=batch.text_context,
        negative_text_context=batch.negative_text_context,
    )

    with torch.inference_mode():
        for chunk_index in range(int(num_chunks)):
            context_frames = int(context_latents.shape[2])
            placeholder = context_latents.new_zeros(
                context_latents.shape[0],
                context_latents.shape[1],
                future_frames,
                context_latents.shape[3],
                context_latents.shape[4],
            )
            model_input = torch.cat([context_latents, placeholder], dim=2)
            step = runner.infer_step(
                session=session,
                context=PolicyInferContext(
                    extra={
                        "task_text": batch.task_text,
                        "metadata": (
                            {
                                "observed_prefix_frames": context_frames,
                                "future_suffix_frames": future_frames,
                            },
                        ),
                    }
                ),
                video_latents=model_input,
            )
            session = step.session
            output = step.infer_output
            predicted = output.decoder_output.aux.get("predicted_latents")
            if not isinstance(predicted, torch.Tensor):
                raise TypeError(
                    "Causal video decoder did not emit `predicted_latents`."
                )
            expected_frames = context_frames + future_frames
            if int(predicted.shape[2]) != expected_frames:
                raise ValueError(
                    "Causal video decoder returned the wrong latent horizon: "
                    f"expected={expected_frames}, actual={predicted.shape[2]}."
                )
            torch.testing.assert_close(
                predicted[:, :, :context_frames],
                context_latents,
                rtol=0.0,
                atol=0.0,
                msg="Causal video rollout modified its clean observed prefix.",
            )
            if chunk_index == 0:
                target_future = target_latents[:, :, observed_frames:total_frames]
                predicted_future = predicted[:, :, observed_frames:total_frames]
                first_chunk_future_mse = float(
                    (predicted_future.float() - target_future.float())
                    .square()
                    .mean()
                    .item()
                )
            context_latents = predicted
            context_sizes.append(int(context_latents.shape[2]))

    assert first_chunk_future_mse is not None
    return CausalVideoPredictionRollout(
        predicted_latents=context_latents,
        target_latents=target_latents,
        observed_latent_frames=observed_frames,
        future_latent_frames=future_frames,
        context_latent_frames=tuple(context_sizes),
        first_chunk_future_mse=first_chunk_future_mse,
    )


def decode_canonical_latent_views(
    pipeline: VariantPipeline,
    latents: torch.Tensor,
    *,
    latent_layout: Mapping[str, Any] | None,
    decode_device: torch.device,
) -> np.ndarray | None:
    """Decode independently encoded views and reassemble their canonical canvas."""

    if latent_layout is None:
        return decode_latent_video_chunks(
            pipeline,
            [latents],
            decode_device=decode_device,
        )
    layout = CanonicalViewLayout.from_metadata(latent_layout)
    latent_canvas_shape = (int(latents.shape[-2]), int(latents.shape[-1]))
    expected_canvas_shape = (layout.canvas_height, layout.canvas_width)
    if latent_canvas_shape != expected_canvas_shape:
        raise ValueError(
            "Canonical latent layout canvas does not match the latent tensor: "
            f"layout={expected_canvas_shape}, tensor={latent_canvas_shape}."
        )

    decoded: list[tuple[ViewPlacement, np.ndarray]] = []
    for placement in layout.placements:
        top = placement.top
        left = placement.left
        height = placement.height
        width = placement.width
        view = latents[:, :, :, top : top + height, left : left + width]
        video = decode_latent_video_chunks(
            pipeline,
            [view],
            decode_device=decode_device,
        )
        if video is None:
            return None
        decoded.append((placement, video))

    first_placement, first_video = decoded[0]
    if (
        int(first_video.shape[1]) % first_placement.height != 0
        or int(first_video.shape[2]) % first_placement.width != 0
    ):
        raise ValueError("Decoded view has a non-integral spatial scale.")
    scale_h = int(first_video.shape[1]) // first_placement.height
    scale_w = int(first_video.shape[2]) // first_placement.width
    if scale_h <= 0 or scale_w <= 0:
        raise ValueError("Decoded view has invalid spatial scale.")
    canvas_height = layout.canvas_height * scale_h
    canvas_width = layout.canvas_width * scale_w
    canvas = np.zeros(
        (first_video.shape[0], canvas_height, canvas_width, first_video.shape[3]),
        dtype=first_video.dtype,
    )
    for placement, video in decoded:
        if int(video.shape[0]) != int(first_video.shape[0]):
            raise ValueError("Decoded latent views have different frame counts.")
        top = placement.top * scale_h
        left = placement.left * scale_w
        height = placement.height * scale_h
        width = placement.width * scale_w
        if tuple(video.shape[1:3]) != (height, width):
            raise ValueError(
                "Decoded latent view does not match its canonical placement: "
                f"decoded={tuple(video.shape[1:3])}, expected={(height, width)}."
            )
        canvas[:, top : top + height, left : left + width] = video
    return canvas


def _positive_frame_count(metadata: dict[str, object], key: str) -> int:
    value = int(metadata.get(key, 0))
    if value <= 0:
        raise ValueError(f"Causal video sample requires positive `{key}`, got {value}.")
    return value


__all__ = [
    "CausalVideoPredictionRollout",
    "decode_canonical_latent_views",
    "rollout_causal_video_prediction",
]
