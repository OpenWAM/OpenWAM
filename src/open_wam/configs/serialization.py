"""Canonical serialization for public Open-WAM configuration artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .enums import serialize_enum_values
from .experiment import ExperimentConfig
from .policy_compatibility import LEGACY_VIDEO_ACTION_POLICY_FIELDS


def serialize_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize an experiment without constructor-only compatibility fields."""

    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass config, got {type(config).__name__}.")
    payload = serialize_enum_values(asdict(config))
    policy_payload = payload.get("policy_variant")
    if isinstance(policy_payload, dict):
        for field_name in LEGACY_VIDEO_ACTION_POLICY_FIELDS:
            policy_payload.pop(field_name, None)
    return payload


__all__ = ["serialize_experiment_config"]
