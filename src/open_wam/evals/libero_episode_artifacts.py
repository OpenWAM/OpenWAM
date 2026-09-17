"""LIBERO artifact collection; no model state or control scheduling."""

from __future__ import annotations

import numpy as np
import torch

from open_wam.evals.libero_policy_runtime import print_rollout_event
from open_wam.evals.libero_rollout_artifacts import append_predicted_latent_chunk
from open_wam.models.policy_variants.contracts import PolicyInferOutput


class LiberoEpisodeArtifacts:
    """Retain images, predictions and the existing per-chunk diagnostic schema."""

    def __init__(
        self,
        observations: list[dict[str, np.ndarray]],
        *,
        coordinates: dict[str, int],
        skip_comparison_video: bool,
        max_imagined_latent_frames: int | None,
    ) -> None:
        self.observations = [self.copy_observation(obs) for obs in observations]
        self.coordinates = dict(coordinates)
        self.skip_comparison_video = skip_comparison_video
        self.max_imagined_latent_frames = max_imagined_latent_frames
        self.predictions: list[torch.Tensor] = []
        self.actions: list[np.ndarray] = []
        self.events: list[dict[str, object]] = []
        self._active_inference: dict[str, object] | None = None
        self._executed = 0

    @staticmethod
    def copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.array(value, copy=True) for key, value in observation.items()}

    def event(
        self, chunk_index: int, phase: str, **values: object
    ) -> dict[str, object]:
        record = {
            **self.coordinates,
            "chunk_index": chunk_index,
            "phase": phase,
            **values,
        }
        label = (
            f"task_{self.coordinates['task_id']}_episode_{self.coordinates['episode_idx']}_chunk_{chunk_index}"
            if self.coordinates
            else f"chunk_{chunk_index}"
        )
        print_rollout_event(label, record)
        self.events.append(record)
        return record

    def inferred(
        self, event: dict[str, object], predicted_latents: torch.Tensor | None
    ) -> None:
        if self._active_inference is not None:
            raise RuntimeError(
                "Cannot record a new prediction before finishing the executed plan."
            )
        if not self.skip_comparison_video and isinstance(
            predicted_latents, torch.Tensor
        ):
            append_predicted_latent_chunk(
                self.predictions,
                predicted_latents,
                max_imagined_latent_frames=self.max_imagined_latent_frames,
            )
        record = dict(event)
        chunk_index = record.pop("chunk_index")
        record.pop("phase")
        self._active_inference = self.event(chunk_index, "infer", **record)
        self._executed = 0

    def executed(
        self,
        action: np.ndarray,
        observation: dict[str, np.ndarray],
        *,
        timestep: int,
        terminal: bool,
        success: bool,
    ) -> None:
        self.actions.append(np.array(action, copy=True))
        self.observations.append(self.copy_observation(observation))
        self._executed += 1
        if terminal or self._executed == self._active_inference["execute_action_steps"]:
            self.finish(timestep=timestep, terminal=terminal, success=success)

    def finish(self, *, timestep: int, terminal: bool, success: bool) -> None:
        record = self._active_inference
        if record is None:
            return
        self.event(
            record["chunk_index"],
            "env_rollout",
            env_timestep_after=int(timestep),
            executed_actions=self._executed,
            **{
                name: record[name]
                for name in (
                    "execute_action_steps",
                    "configured_frame_chunk_size",
                    "rollout_frame_chunk_size",
                    "execute_frame_chunk_size",
                )
            },
            done_after_chunk=bool(terminal),
            success_after_chunk=bool(success),
        )
        self._active_inference = None


def _summarize_policy_debug(policy_output: PolicyInferOutput) -> dict[str, object]:
    """Keep rollout logs readable by replacing large tensors with metadata."""

    summary: dict[str, object] = {}
    for key, value in policy_output.aux.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            continue
        if isinstance(value, dict):
            summary[key] = value
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            summary[key] = value
            continue
        summary[key] = type(value).__name__

    envelope = policy_output.decoder_artifacts
    if envelope is not None:
        summary["decoder_artifact_contract"] = envelope.contract
        summary["decoder_artifact_payload_type"] = type(envelope.payload).__name__
    if policy_output.generated_video is not None:
        summary["generated_video"] = {
            "shape": list(policy_output.generated_video.latents.shape),
            "frame_start": policy_output.generated_video.frame_start,
        }
    return summary
