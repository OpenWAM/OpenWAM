from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs.enums import ActionSpace

from open_wam.models.policy_variants.contracts import (
    PolicyInferOutput,
    PolicyPipelineRequirements,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyTemporalSpan,
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
    frame_span: PolicyTemporalSpan | None = None
    action_space: ActionSpace = ActionSpace.MODEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_space", ActionSpace(self.action_space))
        if self.action_space is not ActionSpace.MODEL:
            raise ValueError("Decoder plans must be model-space; integration adapters own source conversion.")
        if self.actions.ndim != 2:
            raise ValueError("Executable actions must have shape [steps, channels].")

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
        """Return model-space controls; custom decoders may override slicing."""
        if output.action_pred.ndim != 3 or output.action_pred.shape[0] != 1:
            raise ValueError("An executable plan requires one environment's actions [1, steps, channels].")
        return ActionDecoderRolloutPlan(
            actions=output.action_pred[0].detach().to(dtype=torch.float32).cpu(),
            source="decoder_action_chunk",
        )

    def commit_rollout_plan(
        self, state: Any | None, plan: ActionDecoderRolloutPlan,
    ) -> Any | None:
        """Return decoder execution state; overrides must not mutate the input."""
        return state


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
