from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from open_wam.configs import ExperimentConfig, PolicyVariantName, VideoConditionSource
from open_wam.models.policy_variants.common import derive_video_condition_sample_seed

if TYPE_CHECKING:
    from open_wam.pipelines import VariantPipeline


class RolloutObservationInputs(TypedDict):
    """Frontend tensors passed into one recurrent policy step."""

    video_latents: torch.Tensor
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None


def apply_rollout_chunk_steps_override(
    config: ExperimentConfig,
    rollout_chunk_steps: int | None,
) -> None:
    if rollout_chunk_steps is None:
        return
    decoder = getattr(config, "action_decoder", None)
    if decoder is None or not hasattr(decoder, "rollout_chunk_steps"):
        raise ValueError(
            "Requested rollout chunk override, but the config action decoder "
            "has no rollout chunk steps."
        )
    object.__setattr__(decoder, "rollout_chunk_steps", int(rollout_chunk_steps))


def resolve_runtime_devices(
    raw: str | None,
    *,
    fallback: torch.device,
) -> tuple[torch.device, ...]:
    if raw is None or not raw.strip():
        return (fallback,)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return (fallback,)
    return tuple(torch.device(part) for part in parts)


def uses_zero_based_generation_start(config: ExperimentConfig) -> bool:
    policy_variant = getattr(config, "policy_variant", None)
    if policy_variant is None:
        return False
    policy_name = getattr(policy_variant, "name", None)
    if policy_name == PolicyVariantName.MOT:
        return True
    train_source = getattr(policy_variant, "train_video_condition_source", None)
    return train_source == VideoConditionSource.GENERATED_FUTURE


def resolve_initial_generation_action_start(
    initial_observations: Sequence[Mapping[str, Any]],
    *,
    initial_generation_action_start: int | None,
    rollout_starts_at_action_zero: bool = False,
) -> int:
    if initial_generation_action_start is not None:
        return max(0, int(initial_generation_action_start))
    if rollout_starts_at_action_zero:
        return 0
    return max(0, len(initial_observations))


def build_sequence_rollout_infer_extra(
    *,
    config: ExperimentConfig,
    prompt: str,
    generation_action_start: int,
    runtime_device: torch.device | None = None,
    task_id: int | None = None,
    episode_idx: int | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {"task_text": (prompt,)}
    policy_name = getattr(config.policy_variant, "name", None)
    if policy_name in {
        PolicyVariantName.POST_LATENT,
        PolicyVariantName.POST_DECODED,
    }:
        extra["video_condition_frame_start"] = int(generation_action_start)
        extra["video_condition_observed_prefix_anchor"] = "end"
        sample_seed = derive_video_condition_sample_seed(
            {
                "task_index": task_id,
                "episode_index": episode_idx,
                "anchor_frame_index": int(generation_action_start),
                "action_start_index": int(generation_action_start),
            }
        )
        if sample_seed is not None:
            extra["video_condition_sample_seed"] = int(sample_seed)
    if policy_name == PolicyVariantName.MOT and runtime_device is not None:
        extra["action_device"] = str(runtime_device)
    return extra


def prepare_rollout_observation_inputs(
    pipeline: VariantPipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
) -> RolloutObservationInputs:
    """Canonicalize and encode one observed view window for recurrent inference."""

    canonical_batch = pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend = pipeline.visual_tower.frontend
    assets = frontend.reference_assets

    if assets.has_vae:
        video_latents = assets.encode_video(
            canonical_video,
            placements=canonical_batch.placements,
            reset_cache=True,
        )
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = assets.encode_text(
                task_text,
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if (
            resolved_negative_text_context is None
            and resolved_text_context is not None
        ):
            resolved_negative_text_context = assets.encode_blank_text(
                batch_size=canonical_video.shape[0],
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
    else:
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=(
                None
                if text_context is None
                else text_context.to(device=frontend_device)
            ),
            negative_text_context=(
                None
                if negative_text_context is None
                else negative_text_context.to(device=frontend_device)
            ),
            preserve_stream_cache=False,
        )
        video_latents = frontend_output.video_latents
        resolved_text_context = frontend_output.conditioning.text_context
        resolved_negative_text_context = (
            frontend_output.conditioning.negative_text_context
        )
    return {
        "video_latents": video_latents.to(device=runtime_device),
        "text_context": (
            None
            if resolved_text_context is None
            else resolved_text_context.to(device=runtime_device)
        ),
        "negative_text_context": (
            None
            if resolved_negative_text_context is None
            else resolved_negative_text_context.to(device=runtime_device)
        ),
    }
