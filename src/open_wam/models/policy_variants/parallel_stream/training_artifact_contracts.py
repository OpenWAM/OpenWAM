"""Typed outputs shared by parallel-stream training artifact builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from open_wam.configs.enums import DynamicsObjective
from open_wam.models.common.flow_schedule import FlowMatchScheduler


@dataclass
class ParallelTrainArtifacts:
    """Prepared parallel-stream inputs and flow schedulers."""

    input_dict: dict[str, Any]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler
    dynamics_objective: DynamicsObjective | None = None


# Checkpoint-era import compatibility.
LingbotParallelTrainArtifacts = ParallelTrainArtifacts


__all__ = ["LingbotParallelTrainArtifacts", "ParallelTrainArtifacts"]
