"""Canonical serialization for public OpenWAM configuration artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .enums import serialize_enum_values
from .experiment import ExperimentConfig
from .resolution import resolve_experiment_config


def serialize_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize an experiment using canonical enum values and field names."""

    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass config, got {type(config).__name__}.")
    return serialize_enum_values(asdict(resolve_experiment_config(config)))


__all__ = ["serialize_experiment_config"]
