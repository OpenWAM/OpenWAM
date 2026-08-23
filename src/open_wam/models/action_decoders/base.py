from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.models.policy_variants.contracts import (
    PolicyInferOutput,
    PolicyPipelineRequirements,
    PolicyTrainBatch,
    PolicyTrainOutput,
)

_DecoderArtifactT = TypeVar("_DecoderArtifactT")


def require_decoder_artifact_payload(
    policy_output: PolicyTrainOutput | PolicyInferOutput,
    *,
    contract: str,
    payload_type: type[_DecoderArtifactT],
) -> _DecoderArtifactT:
    """Resolve a typed payload crossing the policy-to-decoder boundary."""

    if policy_output.decoder_artifacts is not None:
        return policy_output.decoder_artifacts.require(
            contract=contract,
            payload_type=payload_type,
        )
    raise ValueError(
        f"Decoder requires artifact contract {contract!r} with payload "
        f"{payload_type.__name__}."
    )


@dataclass
class ActionDecoderTrainOutput:
    """Common train-time action-decoder outputs."""

    action_pred: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecoderInferOutput:
    """Common inference-time action-decoder outputs."""

    action_pred: torch.Tensor
    next_state: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecoderRolloutPlan:
    """Decoder-owned model-space actions committed by one rollout step.

    `actions` is a detached float32 CPU tensor with shape `[steps, action_dim]`.
    The optional commit range lets a sequence decoder advance cached state past
    every action released to the environment, not merely the action sampled by
    the latest decoder call.
    """

    actions: torch.Tensor
    source: str
    rollout_chunk_steps: int | None = None
    commit_start_index: int | None = None
    commit_end_index: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the stable rollout trace fields used by integrations."""

        return {
            "action_plan_source": str(self.source),
            "decoder_rollout_chunk_steps": self.rollout_chunk_steps,
            "decoder_rollout_commit_start_index": self.commit_start_index,
            "decoder_rollout_commit_end_index": self.commit_end_index,
            "decoder_rollout_committed_actions": int(self.actions.shape[0]),
        }


def align_policy_features(
    policy_features: torch.Tensor, target_length: int
) -> torch.Tensor:
    """Interpolate `[B, T, D]` features to the action horizon."""

    if policy_features.shape[1] == target_length:
        return policy_features
    return F.interpolate(
        policy_features.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class ActionDecoder(nn.Module, ABC):
    """Action decoder interface shared across policy variants."""

    @property
    def dynamics_metric_namespace(self) -> str | None:
        """Return the decoder-native routed-dynamics metric namespace."""

        return None

    @property
    def decoder_artifact_contract(self) -> str | None:
        """Return the typed policy payload contract consumed by this decoder."""

        return None

    def configure_pipeline_requirements(
        self,
        requirements: PolicyPipelineRequirements,
    ) -> None:
        """Consume optional policy-declared decoder requirements at assembly."""

        del requirements

    def build_rollout_plan(
        self,
        output: ActionDecoderInferOutput,
    ) -> ActionDecoderRolloutPlan:
        """Project one inference output into actions released to a rollout.

        The default handles both full-horizon decoders and sequence decoders
        that expose a cached `current_action`. A custom decoder can override
        this method without adding method-specific logic to an integration.
        """

        action_chunk = output.action_pred[0].detach().to(dtype=torch.float32).cpu()
        current_action = output.aux.get("current_action")
        if not isinstance(current_action, torch.Tensor):
            return ActionDecoderRolloutPlan(
                actions=action_chunk,
                source="decoder_action_chunk",
            )

        current_action_index = _resolve_current_action_index(output)
        rollout_chunk_steps = _resolve_rollout_chunk_steps(output)
        if current_action_index < 0:
            raise ValueError(
                "Decoder current action index must be non-negative, "
                f"got {current_action_index}."
            )
        commit_end_index = min(
            int(action_chunk.shape[0]),
            max(int(current_action_index) + 1, int(rollout_chunk_steps)),
        )
        planned_chunk = action_chunk[int(current_action_index) : commit_end_index]
        source = "decoder_current_action_rollout_chunk"
        if int(planned_chunk.shape[0]) == 0:
            planned_chunk = _current_action_tensor_to_chunk(current_action)
            commit_end_index = int(current_action_index) + int(planned_chunk.shape[0])
            source = "decoder_current_action"
        return ActionDecoderRolloutPlan(
            actions=planned_chunk,
            source=source,
            rollout_chunk_steps=int(rollout_chunk_steps),
            commit_start_index=int(current_action_index),
            commit_end_index=int(commit_end_index),
        )

    def commit_rollout_plan(
        self,
        state: Any | None,
        plan: ActionDecoderRolloutPlan,
    ) -> None:
        """Advance decoder-owned state past actions released by `plan`."""

        if plan.commit_end_index is None:
            return
        if state is None or not hasattr(state, "step_within_chunk"):
            return
        state.step_within_chunk = max(
            int(state.step_within_chunk),
            int(plan.commit_end_index),
        )

    def configure_action_sampler_mask(
        self,
        sampler_mask: torch.Tensor | None,
        *,
        inactive_value: float = 0.0,
    ) -> None:
        """Configure optional inference-time channel pinning for mapped action spaces."""

        if sampler_mask is None:
            self._buffers.pop("_action_sampler_mask", None)
            self._action_sampler_inactive_value = float(inactive_value)
            return
        if sampler_mask.ndim != 2:
            raise ValueError(
                f"Action sampler mask must have shape [H, D], got {tuple(sampler_mask.shape)}."
            )
        mask = sampler_mask.detach().to(dtype=torch.float32).unsqueeze(0)
        if "_action_sampler_mask" in self._buffers:
            self._buffers["_action_sampler_mask"] = mask
        else:
            self.register_buffer("_action_sampler_mask", mask, persistent=False)
        self._action_sampler_inactive_value = float(inactive_value)

    def _apply_action_sampler_mask(
        self, actions: torch.Tensor, *, start_index: int = 0
    ) -> torch.Tensor:
        sampler_mask = getattr(self, "_action_sampler_mask", None)
        if sampler_mask is None:
            return actions
        if actions.ndim not in {2, 3}:
            raise ValueError(
                f"Action sampler mask supports [B, D] or [B, H, D], got {tuple(actions.shape)}."
            )
        if actions.shape[-1] != sampler_mask.shape[-1]:
            raise ValueError(
                f"Action sampler mask dim {sampler_mask.shape[-1]} does not match action dim {actions.shape[-1]}."
            )
        horizon = actions.shape[-2] if actions.ndim == 3 else 1
        end_index = int(start_index) + int(horizon)
        if end_index > sampler_mask.shape[1]:
            raise ValueError(
                "Action sampler mask horizon is shorter than the requested action slice, "
                f"got mask_horizon={sampler_mask.shape[1]}, start_index={start_index}, horizon={horizon}."
            )
        mask = sampler_mask[:, int(start_index) : end_index].to(
            device=actions.device, dtype=actions.dtype
        )
        if actions.ndim == 2:
            mask = mask[:, 0]
        inactive = actions.new_full(
            (), float(getattr(self, "_action_sampler_inactive_value", 0.0))
        )
        return actions * mask + inactive * (1.0 - mask)

    @abstractmethod
    def forward_train(
        self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch
    ) -> ActionDecoderTrainOutput:
        """Decode actions and compute loss."""

    @abstractmethod
    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        """Decode actions for one inference step."""


def _current_action_tensor_to_chunk(current_action: torch.Tensor) -> torch.Tensor:
    current_action = current_action.detach().to(dtype=torch.float32).cpu()
    if current_action.ndim == 1:
        return current_action.unsqueeze(0)
    if current_action.ndim == 2:
        return current_action[:1]
    raise ValueError(
        "Expected current_action shape [D] or [B, D], "
        f"got {tuple(current_action.shape)}."
    )


def _resolve_current_action_index(output: ActionDecoderInferOutput) -> int:
    current_action_index = output.aux.get("current_action_index")
    if current_action_index is not None:
        return int(_python_value(current_action_index))
    next_state = output.next_state
    if next_state is not None and hasattr(next_state, "step_within_chunk"):
        return max(0, int(next_state.step_within_chunk) - 1)
    return 0


def _resolve_rollout_chunk_steps(output: ActionDecoderInferOutput) -> int:
    rollout_chunk_steps = output.aux.get("rollout_chunk_steps")
    if rollout_chunk_steps is None:
        next_state = output.next_state
        rollout_chunk_steps = (
            getattr(next_state, "aux", {}).get("rollout_chunk_steps")
            if next_state is not None
            else None
        )
    if rollout_chunk_steps is None:
        return 1
    return max(1, int(_python_value(rollout_chunk_steps)))


def _python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value
