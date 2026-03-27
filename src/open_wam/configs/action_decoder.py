from __future__ import annotations

from dataclasses import dataclass

from .enums import ActionDecoderName, coerce_fields


@dataclass(frozen=True)
class ActionDecoderConfig:
    """Final action-decoder config independent from policy attachment."""

    name: ActionDecoderName
    hidden_size: int
    action_dim: int
    action_horizon: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        coerce_fields(self, enum_fields={"name": ActionDecoderName})


@dataclass(frozen=True)
class MLPActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.MLP
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class RegisterActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.REGISTER
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class DecodedFeatureActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.DECODED_FEATURE
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


@dataclass(frozen=True)
class LingbotParallelActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.LINGBOT_PARALLEL
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
