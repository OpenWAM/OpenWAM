"""Outputs shared by parallel-stream inference strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ParallelInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor
    next_cache: dict[str, Any]
    debug: dict[str, Any]


# Checkpoint-era import compatibility.
LingbotParallelInferArtifacts = ParallelInferArtifacts
