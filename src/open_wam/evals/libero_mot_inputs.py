"""Model-input preparation for checkpoint-backed MoT LIBERO rollouts."""

from __future__ import annotations

import numpy as np
import torch

from open_wam.evals.libero_mot_runtime import (
    _validate_live_sim_mot_generalist_rollout_mode,
)
from open_wam.integrations import build_libero_state_history
from open_wam.models.policy_variants import PolicyInferContext


def _build_infer_context(
    prompt: str,
    *,
    action_device: torch.device,
    model_obs_window: list[dict[str, np.ndarray]],
    config,
    runtime_device: torch.device,
    mot_inference_window_size: int | None,
    mot_action_only_rollout: bool,
    mot_generalist_rollout_mode: str | None,
    mot_rollout_frame_chunk_size: int | None = None,
):
    extra: dict[str, object] = {
        "task_text": (prompt,),
        "action_device": str(action_device),
    }
    if mot_inference_window_size is not None:
        extra["mot_inference_window_size"] = int(mot_inference_window_size)
    if mot_rollout_frame_chunk_size is not None:
        extra["mot_rollout_frame_chunk_size"] = int(mot_rollout_frame_chunk_size)
    if mot_action_only_rollout:
        extra["mot_action_only_rollout"] = True
    if mot_generalist_rollout_mode is not None:
        _validate_live_sim_mot_generalist_rollout_mode(mot_generalist_rollout_mode)
        extra["action_conditioning_mode"] = str(mot_generalist_rollout_mode)
        extra["mot_generalist_rollout_mode"] = str(mot_generalist_rollout_mode)
    return PolicyInferContext(
        state=build_libero_state_history(
            model_obs_window,
            state_horizon=int(config.data.action_schema.state_horizon),
            state_encoding=config.data.action_target.state_encoding,
        )
        .unsqueeze(0)
        .to(device=runtime_device),
        extra=extra,
    )


def _select_model_obs_window(
    frame_window: list[dict[str, np.ndarray]],
    *,
    chunk_index: int,
    startup_model_obs_frames: int,
) -> list[dict[str, np.ndarray]]:
    if not frame_window:
        raise ValueError("MoT visualization requires at least one observation frame.")
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


def _prepare_mot_visual_outputs(
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
