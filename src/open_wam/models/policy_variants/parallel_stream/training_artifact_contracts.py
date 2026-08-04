"""Typed outputs shared by parallel-stream training artifact builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.models.common.flow_schedule import FlowMatchScheduler


@dataclass
class ParallelTrainArtifacts:
    """Prepared parallel-stream inputs and flow schedulers."""

    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler


# Checkpoint-era import compatibility.
LingbotParallelTrainArtifacts = ParallelTrainArtifacts


__all__ = ["LingbotParallelTrainArtifacts", "ParallelTrainArtifacts"]
