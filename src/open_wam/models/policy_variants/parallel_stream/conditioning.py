from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from open_wam.configs import (
    ContextConditionLatentSource,
    DynamicsObjective,
    ProprioContextMode,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.dynamics_conditioning import (
    append_dynamics_mode_context_token,
)
from open_wam.models.common.dynamics_objectives import DynamicsSamplePlan
from open_wam.models.common.proprio_conditioning import (
    HiddenProprioContext,
    prepend_hidden_proprio_context,
    resolve_hidden_proprio_context,
    select_latest_proprio_state,
)
from open_wam.models.common.video_conditioning import (
    resolve_video_condition_latents,
)

from ..contracts import PolicyTrainBatch


class ParallelConditioningTrainArtifacts(Protocol):
    """Train-artifact surface mutated by policy-level conditioning."""

    @property
    def input_dict(self) -> dict[str, Any]: ...

    @property
    def dynamics_objective(self) -> DynamicsObjective | None: ...


@dataclass(frozen=True, slots=True)
class ParallelStreamConditioning:
    """Resolve parallel-stream video, text, mode, and proprio conditioning inputs."""

    config: ParallelStreamPolicyConfig

    def uses_proprio_context(self) -> bool:
        return (
            ProprioContextMode(self.config.proprio_context_mode)
            != ProprioContextMode.NONE
        )

    def uses_text_proprio_context(self) -> bool:
        # Deprecated compatibility path; new proprio runs use per-chunk additive context.
        return (
            ProprioContextMode(self.config.proprio_context_mode)
            == ProprioContextMode.TEXT_CONTEXT_TOKEN
        )

    def uses_per_chunk_proprio_context(self) -> bool:
        return (
            ProprioContextMode(self.config.proprio_context_mode)
            == ProprioContextMode.PER_CHUNK_ADDITIVE
        )

    def uses_generalist_mode_text_token(self) -> bool:
        return bool(self.config.generalist_mode_text_token)

    def uses_external_condition_prefix(
        self,
        *,
        context_prefix_frames_in_sample: int | None,
        dynamics_sample_plan: DynamicsSamplePlan | None = None,
    ) -> bool:
        """Return whether a separate condition frame precedes the sample sequence."""

        if (
            ContextConditionLatentSource(self.config.context_condition_latent_source)
            != ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        ):
            return False
        if (
            dynamics_sample_plan is not None
            and dynamics_sample_plan.uses_in_sequence_condition
        ):
            return False
        return int(context_prefix_frames_in_sample or 0) == 0

    def select_rollout_proprio_state(
        self,
        state: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return select_latest_proprio_state(state)

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
            raise ValueError(
                f"Proprio context mode is enabled but no state was provided for {label}."
            )
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
                if tuple(proprio_context_state_mask.shape) != tuple(
                    proprio_context_state.shape
                ):
                    raise ValueError(
                        "Per-chunk proprio context mask must match proprio_context_state shape, "
                        f"got mask={tuple(proprio_context_state_mask.shape)}, "
                        f"state={tuple(proprio_context_state.shape)}."
                    )
                proprio_context_state = (
                    proprio_context_state
                    * proprio_context_state_mask.to(
                        device=proprio_context_state.device,
                        dtype=proprio_context_state.dtype,
                    )
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
    ) -> HiddenProprioContext | None:
        if not self.uses_per_chunk_proprio_context():
            return None
        return resolve_hidden_proprio_context(
            batch.extra,
            require_frame_aligned=self.config.requires_frame_aligned_proprio_context,
            label=label,
        )

    def resolve_infer_proprio_context(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        infer_cache: dict | None = None,
    ) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
            return None
        selected = select_latest_proprio_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = select_latest_proprio_state(cached_state)
        if selected is None:
            raise ValueError(
                f"Proprio context mode is enabled but no state was provided for {label}."
            )
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
        selected = select_latest_proprio_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = select_latest_proprio_state(cached_state)
        if selected is None:
            raise ValueError(
                f"Per-chunk proprio mode is enabled but no state was provided for {label}."
            )
        return selected

    def cache_infer_proprio_state(
        self,
        cache: dict,
        state: torch.Tensor | None,
    ) -> None:
        if self.uses_proprio_context() and state is not None:
            cache["last_proprio_state"] = state.detach().clone()

    def resolve_train_condition_latents(
        self,
        batch: PolicyTrainBatch,
        *,
        video_latents: torch.Tensor,
        dynamics_sample_plan: DynamicsSamplePlan | None = None,
    ) -> torch.Tensor | None:
        if (
            dynamics_sample_plan is not None
            and dynamics_sample_plan.uses_in_sequence_condition
        ):
            return None
        return resolve_video_condition_latents(
            video_latents,
            batch.extra.get("condition_latents"),
            enabled=bool(self.config.use_condition_latents),
            required=bool(self.config.require_condition_latents),
            label="Training",
        )

    def attach_train_hidden_proprio_context(
        self,
        artifacts: ParallelConditioningTrainArtifacts,
        *,
        batch: PolicyTrainBatch,
        video_latents: torch.Tensor,
        payload: HiddenProprioContext | None,
    ) -> None:
        if payload is None:
            return
        if artifacts.input_dict.get("prefix_condition_frames"):
            payload = prepend_hidden_proprio_context(
                payload,
                prefix_state=batch.state,
                target_frame_count=int(video_latents.shape[2]),
                label="Parallel Stream external condition prefix",
            )
        artifacts.input_dict["per_chunk_proprio_state"] = payload.values.to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        artifacts.input_dict["per_chunk_proprio_state_granularity"] = (
            payload.granularity.value
        )

    def append_generalist_mode_text_token(
        self,
        transformer: torch.nn.Module,
        artifacts: ParallelConditioningTrainArtifacts,
    ) -> int:
        if not self.uses_generalist_mode_text_token():
            return 0
        objective = artifacts.dynamics_objective
        if objective is None:
            raise ValueError(
                "`generalist_mode_text_token = true` requires a resolved dynamics objective."
            )
        latent_dict = artifacts.input_dict["latent_dict"]
        action_dict = artifacts.input_dict["action_dict"]
        text_emb = latent_dict["text_emb"]
        if action_dict["text_emb"].shape != text_emb.shape:
            raise ValueError(
                "Generalist mode text-token appending expects latent/action text embeddings "
                f"to share shape, got latent={tuple(text_emb.shape)} "
                f"and action={tuple(action_dict['text_emb'].shape)}."
            )
        appended_text, token_count = append_dynamics_mode_context_token(
            transformer,
            text_emb,
            objective,
        )
        latent_dict["text_emb"] = appended_text
        action_dict["text_emb"] = appended_text
        artifacts.input_dict["generalist_mode_text_token"] = objective.value
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
            raise TypeError(
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
