from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import enums as config_enums
from .coercion import coerce_enum as _coerce_enum
from .data_contracts import DataConfig
from .enums import ActionDecoderName, coerce_fields
from .policy_compatibility import normalize_video_action_decoder_fields
from .policy_contracts import PolicyVariantConfig


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
        if self.name == ActionDecoderName.LINGBOT_PARALLEL:
            object.__setattr__(self, "name", ActionDecoderName.PARALLEL_STREAM)


@dataclass(frozen=True)
class ExtensionActionDecoderConfig(ActionDecoderConfig):
    """Config envelope for an application-owned action decoder."""

    name: ActionDecoderName = ActionDecoderName.EXTENSION
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    extension_type: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != ActionDecoderName.EXTENSION:
            raise ValueError("Extension action decoder requires `name = extension`.")
        if not isinstance(self.extension_type, str) or not self.extension_type.strip():
            raise ValueError(
                "Extension action decoder requires a non-empty `extension_type` string."
            )
        if self.extension_type != self.extension_type.strip():
            raise ValueError(
                "Extension action decoder `extension_type` must not have surrounding whitespace."
            )
        if not isinstance(self.options, Mapping):
            raise TypeError("Extension action decoder `options` must be a mapping.")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("Extension action decoder `options` keys must be strings.")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class ParallelStreamActionDecoderConfig(ActionDecoderConfig):
    """Decoder config for outputs produced by a parallel-stream policy."""

    name: ActionDecoderName = ActionDecoderName.PARALLEL_STREAM
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    recovered_osc_loss_weight: float = 0.0
    recovered_osc_position_scale: float = 0.010576533139391671
    recovered_osc_rotation_scale: float = 0.1136411594890211


@dataclass(frozen=True)
class DualExpertActionDecoderConfig(ActionDecoderConfig):
    """DualExpert decoder config for action/video flow supervision and infer packaging."""

    name: ActionDecoderName = ActionDecoderName.DUAL_EXPERT
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


# Deprecated Python config type alias.
MoTActionDecoderConfig = DualExpertActionDecoderConfig


@dataclass(frozen=True)
class VideoOnlyActionDecoderConfig(ActionDecoderConfig):
    """Video-only decoder config for future-latent supervision without action loss."""

    name: ActionDecoderName = ActionDecoderName.VIDEO_ONLY
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


def parse_action_decoder_config(
    action_decoder_raw: Mapping[str, Any],
    policy_variant_config: PolicyVariantConfig,
    data_config: DataConfig,
) -> ActionDecoderConfig:
    resolved_raw = normalize_video_action_decoder_fields(action_decoder_raw)
    if not resolved_raw:
        default_decoder = policy_variant_config.default_action_decoder
        if default_decoder is None:
            raise ValueError(
                "Experiment config requires an explicit `action_decoder` mapping for this policy."
            )
        resolved_raw["name"] = default_decoder

    name = _coerce_enum(config_enums.ActionDecoderName, resolved_raw["name"])
    hidden_size = resolved_raw.get("hidden_size", policy_variant_config.hidden_size)
    action_dim = resolved_raw.get("action_dim", data_config.action_schema.action_dim)
    action_horizon = resolved_raw.get(
        "action_horizon", data_config.action_schema.action_horizon
    )
    dropout = resolved_raw.get("dropout", 0.0)

    if name == config_enums.ActionDecoderName.PARALLEL_STREAM:
        return ParallelStreamActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            recovered_osc_loss_weight=resolved_raw.get(
                "recovered_osc_loss_weight", 0.0
            ),
            recovered_osc_position_scale=resolved_raw.get(
                "recovered_osc_position_scale",
                ParallelStreamActionDecoderConfig.recovered_osc_position_scale,
            ),
            recovered_osc_rotation_scale=resolved_raw.get(
                "recovered_osc_rotation_scale",
                ParallelStreamActionDecoderConfig.recovered_osc_rotation_scale,
            ),
        )
    if name == config_enums.ActionDecoderName.DUAL_EXPERT:
        return DualExpertActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.VIDEO_ONLY:
        return VideoOnlyActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.EXTENSION:
        return ExtensionActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            extension_type=resolved_raw.get("extension_type", ""),
            options=resolved_raw.get("options", {}),
        )
    raise ValueError(f"Unsupported action decoder '{name}'.")


# Deprecated Python config type alias.
LingbotParallelActionDecoderConfig = ParallelStreamActionDecoderConfig
