"""History-record copying and model-input materialization for realtime evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from open_wam.evals import libero_visualization as exact_viz


def copy_history_record_for_worker(record: dict[str, Any]) -> dict[str, Any]:
    copied = {
        "absolute_frame_index": int(record["absolute_frame_index"]),
        "obs": {
            key: np.array(value, copy=True)
            for key, value in record["obs"].items()
        },
    }
    if "raw_actions" in record:
        copied["raw_actions"] = np.array(record["raw_actions"], copy=True)
    if "raw_actions_valid" in record:
        copied["raw_actions_valid"] = bool(record["raw_actions_valid"])
    if "raw_action_dim" in record:
        copied["raw_action_dim"] = int(record["raw_action_dim"])
    if "obs_sequence" in record:
        copied["obs_sequence"] = [
            {
                key: np.array(value, copy=True)
                for key, value in obs.items()
            }
            for obs in record["obs_sequence"]
        ]
    if isinstance(record.get("video_latents"), torch.Tensor):
        copied["video_latents"] = record["video_latents"].detach().clone()
    if record.get("proprio_state") is not None:
        copied["proprio_state"] = np.array(record["proprio_state"], dtype=np.float32, copy=True)
    return copied


def _prepare_history_runtime_inputs(
    runner,
    *,
    history_views: list[dict[str, np.ndarray]],
    task_text: tuple[str | None, ...] | None,
    text_context: torch.Tensor | None,
    negative_text_context: torch.Tensor | None,
    config,
    frontend_device: torch.device,
    runtime_device: torch.device,
    precomputed_video_latents: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if not history_views and precomputed_video_latents is None:
        raise ValueError("Expected observed frames or precomputed latents for history preparation.")
    prepared = None
    resolved_text_context = text_context
    resolved_negative_text_context = negative_text_context
    if history_views:
        prepared = exact_viz.prepare_exact_runtime_inputs(
            runner,
            views=exact_viz.observations_to_views(history_views, device=frontend_device),
            task_text=task_text,
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            preserve_stream_cache=True,
        )
        resolved_text_context = prepared["text_context"]
        resolved_negative_text_context = prepared["negative_text_context"]
    latent_chunks: list[torch.Tensor] = []
    if precomputed_video_latents is not None:
        latent_chunks.append(precomputed_video_latents.to(device=runtime_device))
    if prepared is not None:
        latent_chunks.append(prepared["video_latents"])
    return {
        "video_latents": torch.cat(latent_chunks, dim=2),
        "text_context": resolved_text_context,
        "negative_text_context": resolved_negative_text_context,
    }


def _history_records_to_obs_sequence(
    history_records: list[dict[str, Any]],
) -> list[dict[str, np.ndarray]]:
    history_views: list[dict[str, np.ndarray]] = []
    for record in history_records:
        obs_sequence = record.get("obs_sequence")
        if obs_sequence is not None:
            history_views.extend(
                {
                    key: np.array(value, copy=True)
                    for key, value in obs.items()
                }
                for obs in obs_sequence
            )
            continue
        history_views.append(
            {
                key: np.array(value, copy=True)
                for key, value in record["obs"].items()
            }
        )
    return history_views


def _count_history_raw_observations(history_records: list[dict[str, Any]]) -> int:
    raw_count = 0
    for record in history_records:
        obs_sequence = record.get("obs_sequence")
        if obs_sequence is not None:
            raw_count += len(obs_sequence)
        elif "obs" in record and not isinstance(record.get("video_latents"), torch.Tensor):
            raw_count += 1
    return int(raw_count)


def _history_records_to_action_history(
    history_records: list[dict[str, Any]],
    *,
    config,
) -> np.ndarray:
    action_rows: list[np.ndarray] = []
    inferred_dim: int | None = None
    for record in history_records:
        if "raw_action_dim" in record:
            inferred_dim = int(record["raw_action_dim"])
        raw_actions = record.get("raw_actions")
        if raw_actions is None or record.get("raw_actions_valid") is False:
            continue
        raw_array = np.asarray(raw_actions, dtype=np.float32)
        if raw_array.ndim != 2:
            raise ValueError(f"History raw_actions must be [T, D], got {raw_array.shape}.")
        inferred_dim = int(raw_array.shape[-1])
        if int(raw_array.shape[0]) > 0:
            action_rows.append(raw_array)
    if action_rows:
        return np.concatenate(action_rows, axis=0)
    if inferred_dim is None:
        action_schema = getattr(getattr(config, "data", None), "action_schema", None)
        inferred_dim = int(getattr(action_schema, "action_dim", 0) or 0)
    if inferred_dim is None or inferred_dim <= 0:
        raise ValueError("Unable to infer raw action dimension for empty exact rollout history.")
    return np.zeros((0, int(inferred_dim)), dtype=np.float32)


def _history_records_to_precomputed_video_latents(
    history_records: list[dict[str, Any]],
) -> torch.Tensor | None:
    latent_chunks: list[torch.Tensor] = []
    for record in history_records:
        video_latents = record.get("video_latents")
        if isinstance(video_latents, torch.Tensor):
            latent_chunks.append(video_latents.detach())
    if not latent_chunks:
        return None
    return torch.cat(latent_chunks, dim=2)


def _history_records_to_proprio_state(
    history_records: list[dict[str, Any]],
    *,
    config,
    device: torch.device,
) -> torch.Tensor | None:
    if not exact_viz.proprio_context_enabled(config):
        return None
    for record in reversed(history_records):
        proprio_state = record.get("proprio_state")
        if proprio_state is not None:
            return torch.as_tensor(proprio_state, device=device, dtype=torch.float32).reshape(1, -1)
    return None


__all__ = ["copy_history_record_for_worker"]
