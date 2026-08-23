from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from open_wam.models.policy_variants import PolicyGenerationActionOrigin

if TYPE_CHECKING:
    from open_wam.models.policy_variants import PolicyVariant
    from open_wam.pipelines import VariantPipeline


class RolloutObservationInputs(TypedDict):
    """Frontend tensors passed into one recurrent policy step."""

    video_latents: torch.Tensor
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None


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


def uses_zero_based_generation_start(policy_variant: PolicyVariant) -> bool:
    return (
        policy_variant.rollout_contract.generation_action_origin
        == PolicyGenerationActionOrigin.ZERO
    )


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
    policy_variant: PolicyVariant,
    prompt: str,
    runtime_device: torch.device | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {"task_text": (prompt,)}
    extra.update(
        policy_variant.build_rollout_infer_extra(runtime_device=runtime_device)
    )
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
