"""Model-input preparation for checkpoint-backed DualExpert LIBERO rollouts."""

from __future__ import annotations

import numpy as np
import torch

from open_wam.integrations import build_libero_state_history
from open_wam.models.policy_variants import (
    PolicyInferContext,
    PolicyInferenceOutputRequest,
    PolicyVideoGenerationRequest,
)


def _build_infer_context(
    prompt: str,
    *,
    action_device: torch.device,
    model_obs_window: list[dict[str, np.ndarray]],
    config,
    runtime_device: torch.device,
    dual_expert_inference_window_size: int | None,
    dual_expert_action_only_rollout: bool,
    dual_expert_rollout_frame_chunk_size: int | None = None,
    output_request: PolicyInferenceOutputRequest | None = None,
    video_generation: PolicyVideoGenerationRequest | None = None,
):
    extra: dict[str, object] = {
        "task_text": (prompt,),
        "action_device": str(action_device),
    }
    if dual_expert_inference_window_size is not None:
        extra["dual_expert_inference_window_size"] = int(dual_expert_inference_window_size)
    if dual_expert_rollout_frame_chunk_size is not None:
        extra["dual_expert_rollout_frame_chunk_size"] = int(dual_expert_rollout_frame_chunk_size)
    if dual_expert_action_only_rollout:
        extra["dual_expert_action_only_rollout"] = True
    state_horizon = int(config.data.action_schema.state_horizon)
    state = None
    if state_horizon > 0:
        state = (
            build_libero_state_history(
                model_obs_window,
                state_horizon=state_horizon,
                state_encoding=config.data.action_target.state_encoding,
            )
            .unsqueeze(0)
            .to(device=runtime_device)
        )
    return PolicyInferContext(
        state=state,
        output_request=output_request,
        video_generation=video_generation,
        extra=extra,
    )


def _select_model_obs_window(
    frame_window: list[dict[str, np.ndarray]],
    *,
    chunk_index: int,
    startup_model_obs_frames: int,
) -> list[dict[str, np.ndarray]]:
    if not frame_window:
        raise ValueError("DualExpert visualization requires at least one observation frame.")
    if chunk_index == 0:
        if startup_model_obs_frames <= 0:
            raise ValueError(
                f"Expected positive startup_model_obs_frames, got {startup_model_obs_frames}."
            )
        if startup_model_obs_frames > len(frame_window):
            raise ValueError(
                "startup_model_obs_frames cannot exceed the available startup window, "
                f"got startup_model_obs_frames={startup_model_obs_frames}, window={len(frame_window)}."
            )
        return list(frame_window[-startup_model_obs_frames:])
    return list(frame_window)


def _prepare_dual_expert_visual_outputs(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    use_streaming_frontend: bool,
    preserve_stream_cache: bool = False,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    if use_streaming_frontend:
        canonical_batch = pipeline.canonicalize(views)
        canonical_video = canonical_batch.video.to(device=frontend_device)
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype
        return pipeline.prepare_visual_outputs_from_latents(
            frontend_output.video_latents.to(
                device=runtime_device, dtype=runtime_dtype
            ),
            task_text=task_text,
            text_context=(
                None
                if frontend_output.conditioning.text_context is None
                else frontend_output.conditioning.text_context.to(
                    device=runtime_device, dtype=runtime_dtype
                )
            ),
            negative_text_context=(
                None
                if frontend_output.conditioning.negative_text_context is None
                else frontend_output.conditioning.negative_text_context.to(
                    device=runtime_device,
                    dtype=runtime_dtype,
                )
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )
    return _prepare_visual_outputs_offline(
        pipeline,
        views=views,
        task_text=task_text,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _prepare_visual_outputs_offline(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    canonical_batch = pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend = pipeline.visual_tower.frontend
    assets = frontend.reference_assets
    runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype

    if assets.has_vae:
        video_latents = _encode_video_window_offline(
            assets,
            canonical_video=canonical_video,
            placements=canonical_batch.placements,
            device=frontend_device,
        ).to(device=runtime_device, dtype=runtime_dtype)
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = assets.encode_text(
                task_text,
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if resolved_negative_text_context is None and resolved_text_context is not None:
            resolved_negative_text_context = assets.encode_blank_text(
                batch_size=canonical_video.shape[0],
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        return pipeline.prepare_visual_outputs_from_latents(
            video_latents,
            task_text=task_text,
            text_context=(
                None
                if resolved_text_context is None
                else resolved_text_context.to(
                    device=runtime_device, dtype=runtime_dtype
                )
            ),
            negative_text_context=(
                None
                if resolved_negative_text_context is None
                else resolved_negative_text_context.to(
                    device=runtime_device, dtype=runtime_dtype
                )
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )

    return pipeline.prepare_visual_outputs(
        views,
        task_text=task_text,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _encode_video_window_offline(
    assets,
    *,
    canonical_video: torch.Tensor,
    placements,
    device: torch.device,
) -> torch.Tensor:
    if not assets.has_vae:
        raise RuntimeError("Wan VAE assets are not loaded for offline video encoding.")
    del device
    return assets.encode_video(canonical_video, placements=placements, reset_cache=True)


__all__: list[str] = []
