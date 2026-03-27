from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common import RolloutCursor


@dataclass
class PolicyTrainBatch:
    """Structured policy training inputs independent from attachment site."""

    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyPreparedInputs:
    """Prepared variant-specific inputs."""

    batch: PolicyTrainBatch
    variant_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyTrainOutput:
    """Train-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferState:
    """Per-variant inference state."""

    step_index: int = 0
    cursor: RolloutCursor = field(default_factory=RolloutCursor)
    cache: Any = field(default_factory=dict)


@dataclass
class PolicyInferContext:
    """Inputs required for one policy inference step."""

    state: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyInferOutput:
    """Inference-time features emitted by a policy variant."""

    policy_features: torch.Tensor
    next_state: PolicyInferState
    aux: dict[str, Any] = field(default_factory=dict)
