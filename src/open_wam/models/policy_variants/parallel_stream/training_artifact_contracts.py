"""Typed outputs shared by parallel-stream training artifact builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.models.common.flow_schedule import FlowMatchScheduler


@dataclass
class LingbotParallelTrainArtifacts:
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler


# Generic public name; retain the LingBot name for checkpoint-era import compatibility.
ParallelTrainArtifacts = LingbotParallelTrainArtifacts


__all__ = ["LingbotParallelTrainArtifacts", "ParallelTrainArtifacts"]
