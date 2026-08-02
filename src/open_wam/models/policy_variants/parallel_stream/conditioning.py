from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from open_wam.configs import (
    JointDenoiseTrainingMode,
    ParallelRuntimeMode,
    ProprioContextMode,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.visual_tower import VisualTower

from ..contracts import PolicyTrainBatch


class ParallelConditioningTrainArtifacts(Protocol):
    """Train-artifact surface mutated by policy-level conditioning."""

    @property
    def input_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ParallelStreamConditioning:
    """Resolve M1 video, text, mode, and proprio conditioning inputs."""

    config: ParallelStreamPolicyConfig

    _CHUNK_GRANULARITY = "chunk"
    _FRAME_GRANULARITY = "frame"

    def uses_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) != ProprioContextMode.NONE

    def uses_text_proprio_context(self) -> bool:
        # Deprecated compatibility path; new proprio runs use per-chunk additive context.
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.TEXT_CONTEXT_TOKEN

    def uses_per_chunk_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.PER_CHUNK_ADDITIVE

    def uses_generalist_mode_text_token(self) -> bool:
        return bool(self.config.generalist_mode_text_token)

    @staticmethod
    def select_anchor_state(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state[:, -1, :]
        raise ValueError(
            "Proprio context expects batch state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(state.shape)}."
        )

    def select_rollout_proprio_state(
        self,
        state: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            self.config.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME
            and state is not None
            and state.ndim == 3
        ):
            return state[:, 0, :]
        return self.select_anchor_state(state)

    def resolve_required_proprio_state(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
    ) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
            return None
        selected = self.select_rollout_proprio_state(state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def resolve_train_proprio_context(
        self,
        batch: PolicyTrainBatch,
    ) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
            return None
        proprio_context_state = batch.extra.get("proprio_context_state")
        if isinstance(proprio_context_state, torch.Tensor):
            if proprio_context_state.ndim != 3:
                raise ValueError(
                    "Per-chunk proprio context expects shape [B, chunks, state_dim], "
                    f"got {tuple(proprio_context_state.shape)}."
                )
            proprio_context_state_mask = batch.extra.get("proprio_context_state_mask")
            if isinstance(proprio_context_state_mask, torch.Tensor):
                if tuple(proprio_context_state_mask.shape) != tuple(proprio_context_state.shape):
                    raise ValueError(
                        "Per-chunk proprio context mask must match proprio_context_state shape, "
                        f"got mask={tuple(proprio_context_state_mask.shape)}, "
                        f"state={tuple(proprio_context_state.shape)}."
                    )
                proprio_context_state = proprio_context_state * proprio_context_state_mask.to(
                    device=proprio_context_state.device,
                    dtype=proprio_context_state.dtype,
                )
            return proprio_context_state
        return self.resolve_required_proprio_state(
            batch.state,
            label="parallel-stream training",
        )

    def resolve_train_hidden_proprio_context(
        self,
        batch: PolicyTrainBatch,
        *,
        label: str,
    ) -> tuple[torch.Tensor, str] | None:
        if not self.uses_per_chunk_proprio_context():
            return None
        prefer_chunk_state = self.config.runtime_mode in {
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        }
        if prefer_chunk_state:
            value = batch.extra.get("proprio_context_state")
            mask = batch.extra.get("proprio_context_state_mask")
            granularity = self._CHUNK_GRANULARITY
            if not isinstance(value, torch.Tensor):
                value = batch.extra.get("proprio_context_frames")
                mask = batch.extra.get("proprio_context_frames_mask")
                granularity = self._FRAME_GRANULARITY
        else:
            value = batch.extra.get("proprio_context_frames")
            mask = batch.extra.get("proprio_context_frames_mask")
            granularity = self._FRAME_GRANULARITY
            if not isinstance(value, torch.Tensor):
                value = batch.extra.get("proprio_context_state")
                mask = batch.extra.get("proprio_context_state_mask")
                granularity = self._CHUNK_GRANULARITY
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"proprio_context_mode=per_chunk_additive requires proprio additive context for {label}.")
        if value.ndim != 3:
            raise ValueError(
                "Per-chunk proprio context expects state with shape [B, frames, state_dim], "
                f"got {tuple(value.shape)}."
            )
        if isinstance(mask, torch.Tensor):
            if tuple(mask.shape) != tuple(value.shape):
                raise ValueError(
                    "Per-chunk proprio context mask must match state shape, "
                    f"got mask={tuple(mask.shape)}, state={tuple(value.shape)}."
                )
            value = value * mask.to(device=value.device, dtype=value.dtype)
        return value, granularity

    def resolve_infer_proprio_context(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        infer_cache: dict | None = None,
    ) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
            return None
        selected = self.select_anchor_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = self.select_anchor_state(cached_state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def resolve_infer_hidden_proprio_context(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        infer_cache: dict | None = None,
    ) -> torch.Tensor | None:
        if not self.uses_per_chunk_proprio_context():
            return None
        selected = self.select_anchor_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = self.select_anchor_state(cached_state)
        if selected is None:
            raise ValueError(f"Per-chunk proprio mode is enabled but no state was provided for {label}.")
        return selected

    def cache_infer_proprio_state(
        self,
        cache: dict,
        state: torch.Tensor | None,
    ) -> None:
        if self.uses_proprio_context() and state is not None:
            cache["last_proprio_state"] = state.detach().clone()

    def configure_visual_tower(self, visual_tower: VisualTower) -> None:
        if self.uses_generalist_mode_text_token():
            configure_mode = getattr(visual_tower.core, "configure_generalist_mode_context_encoder", None)
            if not callable(configure_mode):
                raise ValueError("Generalist mode text-token ablation requires a shared transformer core.")
            configure_mode(enabled=True)
        if not self.uses_proprio_context():
            return
        configure = (
            getattr(visual_tower.core, "configure_proprio_context_encoder", None)
            if self.uses_text_proprio_context()
            else getattr(visual_tower.core, "configure_proprio_hidden_context_encoder", None)
        )
        if not callable(configure):
            raise ValueError("Proprio context mode requires a shared transformer core.")
        state_dim = int(visual_tower.state_dim or 0)
        if state_dim <= 0:
            raise ValueError("Proprio context mode requires positive data.action_schema.state_dim.")
        configure(enabled=True, state_dim=state_dim)

    def resolve_train_condition_latents(
        self,
        batch: PolicyTrainBatch,
        *,
        video_latents: torch.Tensor,
    ) -> torch.Tensor | None:
        if not bool(self.config.use_condition_latents):
            return None
        condition_latents = batch.extra.get("condition_latents")
        if condition_latents is None:
            if bool(self.config.require_condition_latents):
                raise ValueError(
                    "Parallel-stream training was configured with `require_condition_latents=true`, "
                    "but the latent batch did not provide `condition_latents`."
                )
            return None
        if not isinstance(condition_latents, torch.Tensor):
            raise ValueError(
                "Parallel-stream `condition_latents` must be a tensor when provided, "
                f"got {type(condition_latents).__name__}."
            )
        if condition_latents.ndim != 5:
            raise ValueError(
                "Parallel-stream `condition_latents` must have shape `[B, C, T, H, W]`, "
                f"got {tuple(condition_latents.shape)}."
            )
        if tuple(condition_latents.shape[:2]) != tuple(video_latents.shape[:2]) or tuple(
            condition_latents.shape[-2:]
        ) != tuple(video_latents.shape[-2:]):
            raise ValueError(
                "Parallel-stream `condition_latents` batch/channel/spatial dimensions must match video_latents, "
                f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
            )
        return condition_latents.to(device=video_latents.device, dtype=video_latents.dtype)

    @staticmethod
    def resolve_generalist_training_metadata(
        batch: PolicyTrainBatch,
    ) -> dict[str, object | None]:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            return {"mode_override": None, "drop_text": None, "source": None}
        return {
            "mode_override": sample_metadata.generalist.mode_override,
            "drop_text": sample_metadata.generalist.drop_text_conditioning,
            "source": sample_metadata.generalist.source,
        }

    def attach_train_hidden_proprio_context(
        self,
        artifacts: ParallelConditioningTrainArtifacts,
        *,
        batch: PolicyTrainBatch,
        video_latents: torch.Tensor,
        payload: tuple[torch.Tensor, str] | None,
    ) -> None:
        if payload is None:
            return
        per_chunk_proprio_state, per_chunk_proprio_granularity = payload
        if artifacts.input_dict.get("prefix_condition_frames"):
            prefix_state = self.select_anchor_state(batch.state)
            if prefix_state is None:
                raise ValueError(
                    "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` prefix "
                    "conditioning requires batch.state for the condition frame."
                )
            if per_chunk_proprio_granularity == self._CHUNK_GRANULARITY:
                per_chunk_proprio_state = torch.cat(
                    [
                        prefix_state[:, None, :].to(
                            device=per_chunk_proprio_state.device,
                            dtype=per_chunk_proprio_state.dtype,
                        ),
                        per_chunk_proprio_state,
                    ],
                    dim=1,
                )
            else:
                frame_count = int(video_latents.shape[2])
                if int(per_chunk_proprio_state.shape[1]) < frame_count:
                    raise ValueError(
                        "Prefix per-chunk proprio frame context expects at least one state per target frame, "
                        f"got {tuple(per_chunk_proprio_state.shape)} for target_frames={frame_count}."
                    )
                per_chunk_proprio_state = torch.cat(
                    [
                        prefix_state[:, None, :].to(
                            device=per_chunk_proprio_state.device,
                            dtype=per_chunk_proprio_state.dtype,
                        ),
                        per_chunk_proprio_state[:, :frame_count, :],
                    ],
                    dim=1,
                )
                per_chunk_proprio_granularity = self._FRAME_GRANULARITY
        artifacts.input_dict["per_chunk_proprio_state"] = per_chunk_proprio_state.to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        artifacts.input_dict["per_chunk_proprio_state_granularity"] = per_chunk_proprio_granularity

    def append_generalist_mode_text_token(
        self,
        transformer: torch.nn.Module,
        artifacts: ParallelConditioningTrainArtifacts,
    ) -> int:
        if not self.uses_generalist_mode_text_token():
            return 0
        raw_mode = artifacts.input_dict.get("joint_denoise_training_mode")
        if raw_mode is None:
            raise ValueError(
                "`generalist_mode_text_token = true` requires `joint_denoise_training_mode` "
                "in parallel-stream train artifacts."
            )
        mode = JointDenoiseTrainingMode(raw_mode).value
        latent_dict = artifacts.input_dict["latent_dict"]
        action_dict = artifacts.input_dict["action_dict"]
        text_emb = latent_dict["text_emb"]
        if action_dict["text_emb"].shape != text_emb.shape:
            raise ValueError(
                "Generalist mode text-token appending expects latent/action text embeddings "
                f"to share shape, got latent={tuple(text_emb.shape)} "
                f"and action={tuple(action_dict['text_emb'].shape)}."
            )
        append = getattr(transformer, "append_generalist_mode_context_token", None)
        if not callable(append):
            raise ValueError(
                "Generalist mode text-token ablation requires the runtime transformer "
                "to support mode-token appending."
            )
        appended_text = append(text_emb, mode)
        token_count = int(appended_text.shape[1] - text_emb.shape[1])
        if token_count != 1:
            raise ValueError(
                "Generalist mode text-token ablation expects exactly one appended token, "
                f"got {token_count}."
            )
        latent_dict["text_emb"] = appended_text
        action_dict["text_emb"] = appended_text
        artifacts.input_dict["generalist_mode_text_token"] = mode
        artifacts.input_dict["generalist_mode_text_token_count"] = token_count
        return token_count

    @staticmethod
    def append_train_proprio_text_context(
        transformer: torch.nn.Module,
        artifacts: ParallelConditioningTrainArtifacts,
    ) -> None:
        proprio_state = artifacts.input_dict.get("proprio_state")
        if proprio_state is None:
            return
        latent_dict = artifacts.input_dict["latent_dict"]
        action_dict = artifacts.input_dict["action_dict"]
        text_emb = latent_dict["text_emb"]
        append = getattr(transformer, "append_proprio_context_tokens", None)
        if not callable(append):
            raise ValueError(
                "Deprecated text-space proprio token mode requires the runtime transformer "
                "to support proprio appending."
            )
        base_text_token_count = int(text_emb.shape[1])
        appended_text = append(text_emb, proprio_state)
        latent_dict["text_emb"] = appended_text
        action_dict["text_emb"] = appended_text
        artifacts.input_dict["base_text_token_count"] = base_text_token_count
        artifacts.input_dict["proprio_context_token_count"] = int(
            appended_text.shape[1] - base_text_token_count
        )


__all__ = [
    "ParallelConditioningTrainArtifacts",
    "ParallelStreamConditioning",
]
