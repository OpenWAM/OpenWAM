"""Outputs shared by parallel-stream inference strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..contracts import PolicyInferState


@dataclass
class ParallelInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor
    next_state: PolicyInferState
    generation_frame_start: int
    debug: dict[str, Any]
