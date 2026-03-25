from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionDecoderConfig:
    """Final action-decoder config independent from policy attachment."""

    name: str
    hidden_size: int
    action_dim: int
    action_horizon: int
    dropout: float = 0.0


@dataclass(frozen=True)
class MLPActionDecoderConfig(ActionDecoderConfig):
    name: str = "mlp_decoder"
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class RegisterActionDecoderConfig(ActionDecoderConfig):
    name: str = "register_decoder"
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class DecodedFeatureActionDecoderConfig(ActionDecoderConfig):
    name: str = "decoded_feature_decoder"
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
