"""Exact-startup diagnostics for LIBERO rollout execution."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from einops import rearrange

from open_wam.evals.libero_rollout_artifact_contracts import (
    LiberoExactStartupDebugOptions,
    LiberoExactStartupDebugPayload,
)


def capture_torch_rng_debug_state() -> dict[str, Any]:
    """Capture hash-oriented CPU and CUDA RNG summaries without changing RNG."""

    return {
        "torch_cpu": _debug_tensor_summary(torch.get_rng_state()),
        "torch_cuda": (
            [_debug_tensor_summary(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def build_libero_exact_startup_debug_report(
    *,
    options: LiberoExactStartupDebugOptions,
    payload: LiberoExactStartupDebugPayload,
) -> dict[str, Any]:
    """Build the canonical first-chunk diagnostic report for exact rollouts."""

    cuda_device_name = None
    if options.runtime_device.type == "cuda" and torch.cuda.is_available():
        device_index = (
            torch.cuda.current_device()
            if options.runtime_device.index is None
            else int(options.runtime_device.index)
        )
        cuda_device_name = torch.cuda.get_device_name(device_index)
    return {
        "schema_version": 1,
        "purpose": "startup_first_chunk_cross_gpu_debug",
        "prompt": str(options.prompt),
        "seed": int(options.seed),
        "torch_version": str(torch.__version__),
        "cuda_device_name": cuda_device_name,
        "runtime_device": str(options.runtime_device),
        "frontend_device": str(options.frontend_device),
        "decode_device": str(options.decode_device),
        "reference_assets_device_policy": str(options.reference_assets_device_policy),
        "runtime_mode": str(options.runtime_mode),
        "video_num_inference_steps": int(options.video_num_inference_steps),
        "action_num_inference_steps": int(options.action_num_inference_steps),
        "guidance_scale": float(options.guidance_scale),
        "action_guidance_scale": float(options.action_guidance_scale),
        "exact_startup_bootstrap_padding": bool(
            options.exact_startup_bootstrap_padding
        ),
        "startup_warmup_debug": (
            None
            if options.startup_warmup_debug is None
            else dict(options.startup_warmup_debug)
        ),
        "first_obs": {
            key: _debug_array_summary(value)
            for key, value in sorted(payload.first_observation.items())
        },
        "initial_inputs": {
            "video_latents": _debug_tensor_summary(payload.video_latents),
            "text_context": _debug_tensor_summary(payload.text_context),
            "negative_text_context": _debug_tensor_summary(
                payload.negative_text_context
            ),
        },
        "session_text_context": _debug_tensor_summary(payload.session_text_context),
        "session_negative_text_context": _debug_tensor_summary(
            payload.session_negative_text_context
        ),
        "rng_before_startup_infer": payload.rng_before_startup_infer,
        "rng_after_startup_infer": payload.rng_after_startup_infer,
        "first_chunk": {
            "debug": dict(payload.first_chunk_debug),
            "chunk_action_pred": _debug_tensor_summary(payload.chunk_action_pred),
            "raw_chunk_action_pred": _debug_tensor_summary(
                payload.raw_chunk_action_pred
            ),
            "predicted_latents": _debug_tensor_summary(payload.predicted_latents),
            "raw_action_grid": _debug_raw_action_grid(
                raw_chunk_action_pred=payload.raw_chunk_action_pred,
                generation_frame_start=int(
                    payload.first_chunk_debug.get("generation_frame_start", 0)
                ),
                frame_chunk_size=options.frame_chunk_size,
                action_per_frame=options.action_per_frame,
            ),
        },
    }


def _debug_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _debug_array_summary(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    flat = array.reshape(-1)
    numeric = flat.astype(np.float64, copy=False) if flat.size else flat
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "sha256": _debug_sha256_bytes(array.tobytes()),
        "preview": flat[:12].tolist(),
        "mean": None if flat.size == 0 else float(numeric.mean()),
        "std": None if flat.size == 0 else float(numeric.std()),
    }


def _debug_tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach().contiguous().cpu()
    byte_tensor = tensor.view(torch.uint8)
    flat = tensor.reshape(-1)
    numeric = flat.to(dtype=torch.float32) if flat.numel() else flat
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(value.device),
        "sha256": _debug_sha256_bytes(byte_tensor.numpy().tobytes()),
        "preview": flat[:12].to(dtype=torch.float32).tolist(),
        "mean": None if flat.numel() == 0 else float(numeric.mean().item()),
        "std": None if flat.numel() == 0 else float(numeric.std(unbiased=False).item()),
    }


def _debug_raw_action_grid(
    *,
    raw_chunk_action_pred: torch.Tensor | None,
    generation_frame_start: int,
    frame_chunk_size: int,
    action_per_frame: int,
) -> dict[str, Any] | None:
    if raw_chunk_action_pred is None:
        return None
    raw_actions = rearrange(
        raw_chunk_action_pred[0].detach().to(dtype=torch.float32).cpu(),
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    executable: list[list[float]] = []
    for frame_offset in range(raw_actions.shape[0]):
        if int(generation_frame_start) + frame_offset < 1:
            continue
        for action_offset in range(raw_actions.shape[1]):
            executable.append(
                [
                    float(value)
                    for value in raw_actions[frame_offset, action_offset].tolist()
                ]
            )
    return {
        "generation_frame_start": int(generation_frame_start),
        "all_gripper_by_frame": [
            [
                float(raw_actions[frame_offset, action_offset, 6].item())
                for action_offset in range(raw_actions.shape[1])
            ]
            for frame_offset in range(raw_actions.shape[0])
        ],
        "first_executable_actions": executable[:16],
    }
