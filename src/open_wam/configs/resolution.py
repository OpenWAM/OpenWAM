"""Canonical resolution for typed experiment configurations."""

from __future__ import annotations

from .experiment import ExperimentConfig
from .sequence_contracts import materialize_video_action_sequence_contract


def resolve_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    """Materialize fields owned by typed experiment contracts."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            f"Expected ExperimentConfig, got {type(config).__name__}."
        )
    return materialize_video_action_sequence_contract(config)


__all__ = ["resolve_experiment_config"]
