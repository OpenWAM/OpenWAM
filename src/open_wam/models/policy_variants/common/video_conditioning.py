from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from open_wam.configs import VideoConditionInputSpace
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..contracts import PolicyTrainBatch, VideoConditionWindowContext
from .layouts import tokens_to_frame_major


_FRAME_START_METADATA_KEYS = (
    "action_start_index",
    "subwindow_action_start",
    "window_start_frame",
    "sample_start_frame",
    "observation_start",
    "segment_start_frame",
    "frame_shift",
)

_OBSERVED_PREFIX_ANCHOR_METADATA_KEYS = (
    "video_condition_observed_prefix_anchor",
    "observed_prefix_anchor",
)

_VIDEO_CONDITION_SEED_METADATA_KEYS = (
    "video_condition_seed",
    "sample_seed",
)


def resolve_video_condition_frame_start(batch: PolicyTrainBatch) -> int:
    """Resolve a scalar absolute frame start for generated condition-window training."""

    metadata = batch.extra.get("metadata")
    if not isinstance(metadata, (tuple, list)):
        return 0
    frame_starts: list[int] = []
    for sample_metadata in metadata:
        if not isinstance(sample_metadata, Mapping):
            continue
        for key in _FRAME_START_METADATA_KEYS:
            value = sample_metadata.get(key)
            if value is not None:
                frame_starts.append(int(value))
                break
    if not frame_starts:
        return 0
    first_frame_start = frame_starts[0]
    if any(frame_start != first_frame_start for frame_start in frame_starts):
        raise ValueError(
            "Generated video-condition training currently requires every sample in a batch to share one "
            "absolute frame start because the shared visual runtime accepts a scalar frame offset. "
            f"Got frame_starts={frame_starts!r}."
        )
    return int(first_frame_start)


def resolve_video_condition_observed_prefix_anchor(batch: PolicyTrainBatch) -> str:
    """Resolve the observed-prefix anchor used by generated condition-window training."""

    metadata = batch.extra.get("metadata")
    if not isinstance(metadata, (tuple, list)):
        return "start"
    anchors: list[str] = []
    for sample_metadata in metadata:
        if not isinstance(sample_metadata, Mapping):
            continue
        for key in _OBSERVED_PREFIX_ANCHOR_METADATA_KEYS:
            value = sample_metadata.get(key)
            if value is not None:
                anchors.append(str(value))
                break
    if not anchors:
        return "start"
    first_anchor = anchors[0]
    if any(anchor != first_anchor for anchor in anchors):
        raise ValueError(
            "Generated video-condition training currently requires every sample in a batch to share one "
            "observed-prefix anchor because the shared visual runtime accepts one conditioning convention per call. "
            f"Got anchors={anchors!r}."
        )
    if first_anchor not in {"start", "end"}:
        raise ValueError(
            "Generated video-condition training expected observed-prefix anchor to be 'start' or 'end', "
            f"got {first_anchor!r}."
        )
    return first_anchor


def derive_video_condition_sample_seed(sample_metadata: Mapping[str, Any]) -> int | None:
    """Derive one stable generated-video sample seed from rollout/sample metadata."""

    for key in _VIDEO_CONDITION_SEED_METADATA_KEYS:
        value = sample_metadata.get(key)
        if value is not None:
            return int(value)

    seed_components: list[int] = []
    for key in _FRAME_START_METADATA_KEYS:
        value = sample_metadata.get(key)
        if value is not None:
            seed_components.append(int(value))
            break
    if not seed_components:
        for key in ("episode_index", "task_index", "anchor_frame_index"):
            value = sample_metadata.get(key)
            if value is not None:
                seed_components.append(int(value))
    if not seed_components:
        return None

    seed = 0x45D9F3B
    for value in seed_components:
        seed = ((seed * 1000003) ^ (int(value) + 0x9E3779B9)) & 0x7FFFFFFF
    return int(seed)


def resolve_video_condition_sample_seed(batch: PolicyTrainBatch) -> int | None:
    """Resolve one deterministic seed for generated video conditioning."""

    metadata = batch.extra.get("metadata")
    if not isinstance(metadata, (tuple, list)):
        return None
    seeds: list[int] = []
    for sample_metadata in metadata:
        if not isinstance(sample_metadata, Mapping):
            continue
        seed = derive_video_condition_sample_seed(sample_metadata)
        if seed is not None:
            seeds.append(int(seed))
    if not seeds:
        return None
    first_seed = seeds[0]
    if any(seed != first_seed for seed in seeds):
        raise ValueError(
            "Generated video-condition training currently requires every sample in a batch to share one "
            "deterministic conditioning seed because the shared visual runtime denoises one batched future window. "
            f"Got seeds={seeds!r}."
        )
    return int(first_seed)


def build_local_video_condition_window(
    *,
    visual_outputs: VisualStageOutputs,
    input_space: str,
    local_window_frames: int,
    current_frame_index: int,
    action_chunk_anchor_mode: str,
    source_stage: str,
    observed_frame_count: int = 1,
) -> VideoConditionWindowContext:
    """Build a typed local frame-token window for decoder-side video conditioning."""

    input_space = VideoConditionInputSpace(str(input_space))
    if input_space == VideoConditionInputSpace.VIDEO_LATENT:
        frame_tokens = tokens_to_frame_major(
            visual_outputs.frontend.video_tokens,
            visual_outputs.frontend.token_grid,
        )
        source_family = "frontend_video_tokens"
        source_metadata = {
            "encoded_from": visual_outputs.frontend.input_source,
        }
    elif input_space == VideoConditionInputSpace.RGB_VIDEO:
        if visual_outputs.frontend.input_source != "canonical_rgb":
            raise ValueError(
                "Method-4 `rgb_video` conditioning requires an RGB-backed frontend pass. "
                "This run entered the frontend from precomputed latents instead. "
                "Use `video_latent` conditioning for latent-first runs, or execute the method-4 path "
                "from raw RGB views so the shared frontend/VAE encodes the condition window."
            )
        frame_tokens = tokens_to_frame_major(
            visual_outputs.frontend.video_tokens,
            visual_outputs.frontend.token_grid,
        )
        source_family = "encoded_rgb_frontend_video_tokens"
        source_metadata = {
            "encoded_from": "canonical_rgb",
            "rgb_encoder": "shared_frontend_vae",
        }
    else:  # pragma: no cover - enum validation should prevent this
        raise ValueError(f"Unsupported method-4 video condition input space {input_space!r}.")
    if int(current_frame_index) != 0:
        raise ValueError(
            "Method-4 local video-condition windows currently support only "
            "`current_frame_index = 0` for rollout-window decoding. Non-zero sliding-window "
            "alignment is not implemented yet."
        )
    if local_window_frames > frame_tokens.shape[1]:
        raise ValueError(
            "Local video condition window requires enough frontend frames, "
            f"got local_window_frames={local_window_frames}, available_frames={frame_tokens.shape[1]}."
        )
    local_tokens = frame_tokens[:, :local_window_frames]
    return VideoConditionWindowContext(
        local_window_tokens=local_tokens,
        token_grid=visual_outputs.frontend.token_grid,
        source_stage=source_stage,
        input_space=str(input_space),
        local_window_frames=int(local_window_frames),
        current_frame_index=int(current_frame_index),
        current_action_index=0,
        action_chunk_anchor_mode=str(action_chunk_anchor_mode),
        observed_frame_count=int(observed_frame_count),
        previous_context_frames=0,
        metadata={
            "source_family": source_family,
            "deferred_previous_context": True,
            **source_metadata,
        },
    )


def build_generated_video_condition_window(
    *,
    visual_tower: VisualTower,
    visual_outputs: VisualStageOutputs,
    input_space: str,
    local_window_frames: int,
    current_frame_index: int,
    action_chunk_anchor_mode: str,
    frame_start: int,
    num_inference_steps: int,
    num_train_timesteps: int,
    sigma_shift: float,
    guidance_scale: float,
    cache_name: str,
    source_stage: str = "generated_future",
    observed_frame_count: int = 1,
    observed_prefix_anchor: str = "start",
    sample_seed: int | None = None,
) -> tuple[VideoConditionWindowContext, dict[str, Any]]:
    """Build a method-4 video-condition window from observed RGB/latents plus predicted future video."""

    input_space = VideoConditionInputSpace(str(input_space))
    if int(current_frame_index) != 0:
        raise ValueError(
            "Generated method-4 video-condition windows currently support only "
            "`current_frame_index = 0`. Non-zero rollout-window alignment is not implemented yet."
        )
    local_window_frames = int(local_window_frames)
    observed_frame_count = int(observed_frame_count)
    if local_window_frames <= 0:
        raise ValueError(f"Expected local_window_frames > 0, got {local_window_frames}.")
    if observed_frame_count <= 0:
        raise ValueError(f"Expected observed_frame_count > 0, got {observed_frame_count}.")
    if observed_frame_count > local_window_frames:
        raise ValueError(
            "Observed prefix cannot be longer than the local video-condition window, "
            f"got observed_frame_count={observed_frame_count}, local_window_frames={local_window_frames}."
        )

    if input_space == VideoConditionInputSpace.RGB_VIDEO and visual_outputs.frontend.input_source != "canonical_rgb":
        raise ValueError(
            "Generated method-4 `rgb_video` conditioning requires an RGB-backed frontend prefix. "
            "This run entered the frontend from precomputed latents instead. Use `video_latent` "
            "conditioning for latent-first inference."
        )

    video_latents = visual_outputs.frontend.video_latents
    if video_latents.ndim != 5:
        raise ValueError(
            "Expected frontend video latents with shape [B, C, T, H, W], "
            f"got {tuple(video_latents.shape)}."
        )
    if video_latents.shape[2] < observed_frame_count:
        raise ValueError(
            "Generated method-4 video conditioning requires enough observed frontend frames, "
            f"got observed_frame_count={observed_frame_count}, available_frames={video_latents.shape[2]}."
        )

    if observed_prefix_anchor == "start":
        observed_start = 0
    elif observed_prefix_anchor == "end":
        observed_start = int(video_latents.shape[2]) - observed_frame_count
    else:
        raise ValueError(
            "Generated method-4 video conditioning expected observed_prefix_anchor to be 'start' or 'end', "
            f"got {observed_prefix_anchor!r}."
        )
    observed_prefix = video_latents[:, :, observed_start : observed_start + observed_frame_count]
    future_frame_count = local_window_frames - observed_frame_count
    predicted_future_latents: torch.Tensor | None
    if future_frame_count > 0:
        future_template = video_latents.new_zeros(
            video_latents.shape[0],
            video_latents.shape[1],
            future_frame_count,
            video_latents.shape[3],
            video_latents.shape[4],
        )
        predicted_future_latents = visual_tower.generate_conditioned_future_latents(
            observed_prefix=observed_prefix,
            future_template=future_template,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            frame_start=int(frame_start),
            num_inference_steps=int(num_inference_steps),
            num_train_timesteps=int(num_train_timesteps),
            sigma_shift=float(sigma_shift),
            guidance_scale=float(guidance_scale),
            cache_name=str(cache_name),
            sample_seed=None if sample_seed is None else int(sample_seed),
        )
        condition_latents = torch.cat([observed_prefix, predicted_future_latents], dim=2)
    else:
        predicted_future_latents = None
        condition_latents = observed_prefix

    condition_tokens, token_grid = visual_tower.frontend.tokenize_video_latents(condition_latents)
    local_tokens = tokens_to_frame_major(condition_tokens, token_grid)
    metadata = {
        "source_family": "generated_future_video_tokens",
        "encoded_prefix_from": visual_outputs.frontend.input_source,
        "generator": "shared_visual_tower",
        "frame_start": int(frame_start),
        "observed_prefix_frames": observed_frame_count,
        "observed_prefix_anchor": str(observed_prefix_anchor),
        "observed_prefix_start_index": int(observed_start),
        "generated_future_frames": future_frame_count,
        "uses_future_ground_truth": False,
    }
    if sample_seed is not None:
        metadata["sample_seed"] = int(sample_seed)
    window = VideoConditionWindowContext(
        local_window_tokens=local_tokens,
        token_grid=token_grid,
        source_stage=source_stage,
        input_space=str(input_space),
        local_window_frames=local_window_frames,
        current_frame_index=int(current_frame_index),
        current_action_index=0,
        action_chunk_anchor_mode=str(action_chunk_anchor_mode),
        observed_frame_count=observed_frame_count,
        previous_context_frames=0,
        metadata=metadata,
    )
    aux: dict[str, Any] = {
        "video_condition_source": metadata["source_family"],
        "video_condition_uses_future_ground_truth": False,
        "observed_video_prefix_latents": observed_prefix.detach(),
    }
    if predicted_future_latents is not None:
        aux["predicted_latents"] = predicted_future_latents.detach()
        aux["predicted_video_latents"] = predicted_future_latents.detach()
    return window, aux
