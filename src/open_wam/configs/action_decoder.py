from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from . import enums as config_enums
from .backbone import SharedVideoTransformerConfig
from .coercion import coerce_enum as _coerce_enum
from .data import DataConfig
from .enums import (
    ActionDecoderName,
    ActionChunkAnchorMode,
    ActionExpertInitMode,
    VideoConditionInputSpace,
    VideoConditionTrainMode,
    coerce_fields,
)
from .policy_variant import (
    ParallelStreamPolicyConfig,
    PolicyVariantConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
)


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
            raise ValueError("Extension action decoder requires a non-empty `extension_type` string.")
        if self.extension_type != self.extension_type.strip():
            raise ValueError("Extension action decoder `extension_type` must not have surrounding whitespace.")
        if not isinstance(self.options, Mapping):
            raise ValueError("Extension action decoder `options` must be a mapping.")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("Extension action decoder `options` keys must be strings.")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class MLPActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.MLP
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
class VideoConditionedActionDecoderConfig(ActionDecoderConfig):
    """Generic current-action decoder over a local video-conditioned window."""

    name: ActionDecoderName = ActionDecoderName.VIDEO_CONDITIONED
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    context_dim: int = 256
    text_context_dim: int = 0
    state_dim: int = 0
    freq_dim: int = 256
    num_layers: int = 1
    num_heads: int = 8
    attention_head_dim: int = 32
    ffn_dim: int = 1024
    cross_attn_norm: bool = True
    eps: float = 1e-6
    input_space: VideoConditionInputSpace = VideoConditionInputSpace.VIDEO_LATENT
    train_mode: VideoConditionTrainMode = VideoConditionTrainMode.ROLLOUT_WINDOW_DIFFUSION
    action_chunk_anchor_mode: ActionChunkAnchorMode = ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
    action_expert_init_mode: ActionExpertInitMode = ActionExpertInitMode.VIDEO_WEIGHT_COPY
    rollout_chunk_steps: int = 1
    direct_latent_channels: int = 48
    direct_rgb_patch_size: int = 16
    use_text_conditioning: bool = True
    use_state_conditioning: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "input_space": VideoConditionInputSpace,
                "train_mode": VideoConditionTrainMode,
                "action_chunk_anchor_mode": ActionChunkAnchorMode,
                "action_expert_init_mode": ActionExpertInitMode,
            },
        )
        if (
            self.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
            and self.action_chunk_anchor_mode != ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
        ):
            raise ValueError(
                "Current-frame regression mode requires `action_chunk_anchor_mode = current_plus_future`, "
                f"got action_chunk_anchor_mode={self.action_chunk_anchor_mode!r}."
            )
        if int(self.rollout_chunk_steps) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `rollout_chunk_steps > 0`, "
                f"got rollout_chunk_steps={self.rollout_chunk_steps!r}."
            )
        if int(self.direct_latent_channels) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `direct_latent_channels > 0`, "
                f"got direct_latent_channels={self.direct_latent_channels!r}."
            )
        if int(self.direct_rgb_patch_size) <= 0:
            raise ValueError(
                "Video-conditioned action decoder requires `direct_rgb_patch_size > 0`, "
                f"got direct_rgb_patch_size={self.direct_rgb_patch_size!r}."
            )


@dataclass(frozen=True)
class LingbotParallelActionDecoderConfig(ActionDecoderConfig):
    name: ActionDecoderName = ActionDecoderName.LINGBOT_PARALLEL
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0
    recovered_osc_loss_weight: float = 0.0
    recovered_osc_position_scale: float = 0.010576533139391671
    recovered_osc_rotation_scale: float = 0.1136411594890211


@dataclass(frozen=True)
class MoTActionDecoderConfig(ActionDecoderConfig):
    """MoT decoder config for action/video flow supervision and infer packaging."""

    name: ActionDecoderName = ActionDecoderName.MOT
    hidden_size: int = 256
    action_dim: int = 0
    action_horizon: int = 0


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
    backbone_config: SharedVideoTransformerConfig,
) -> ActionDecoderConfig:
    resolved_raw = dict(action_decoder_raw)
    if not resolved_raw:
        if policy_variant_config.name == config_enums.PolicyVariantName.MOT:
            resolved_raw["name"] = config_enums.ActionDecoderName.MOT
        elif policy_variant_config.name == config_enums.PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_ONLY
        elif policy_variant_config.name == config_enums.PolicyVariantName.POST_DECODED:
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_CONDITIONED
        elif (
            policy_variant_config.name == config_enums.PolicyVariantName.POST_LATENT
            and isinstance(policy_variant_config, PostLatentPolicyConfig)
            and not policy_variant_config.compatibility_mode
        ):
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_CONDITIONED
        elif (
            policy_variant_config.name == config_enums.PolicyVariantName.PARALLEL_STREAM
            and isinstance(policy_variant_config, ParallelStreamPolicyConfig)
            and policy_variant_config.runtime_mode
            in {
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT,
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
                config_enums.ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
                config_enums.ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            }
        ):
            resolved_raw["name"] = config_enums.ActionDecoderName.LINGBOT_PARALLEL
        else:
            resolved_raw["name"] = config_enums.ActionDecoderName.MLP

    name = _coerce_enum(config_enums.ActionDecoderName, resolved_raw["name"])
    if name == config_enums.ActionDecoderName.MLP and policy_variant_config.name == config_enums.PolicyVariantName.MOT:
        # Compatibility path for early MoT YAMLs that used `mlp_decoder` as a placeholder.
        name = config_enums.ActionDecoderName.MOT
    hidden_size = resolved_raw.get(
        "hidden_size",
        (
            backbone_config.hidden_size
            if name == config_enums.ActionDecoderName.VIDEO_CONDITIONED
            else policy_variant_config.hidden_size
        ),
    )
    action_dim = resolved_raw.get("action_dim", data_config.action_schema.action_dim)
    action_horizon = resolved_raw.get("action_horizon", data_config.action_schema.action_horizon)
    dropout = resolved_raw.get("dropout", 0.0)

    if name == config_enums.ActionDecoderName.MLP:
        return MLPActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.DECODED_FEATURE:
        return DecodedFeatureActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.VIDEO_CONDITIONED:
        if isinstance(policy_variant_config, (PostLatentPolicyConfig, PostDecodedPolicyConfig)):
            input_space = policy_variant_config.video_condition_input_space
            anchor_mode = policy_variant_config.action_chunk_anchor_mode
        else:
            input_space = config_enums.VideoConditionInputSpace.VIDEO_LATENT
            anchor_mode = config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
        return VideoConditionedActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            context_dim=resolved_raw.get("context_dim", backbone_config.hidden_size),
            text_context_dim=resolved_raw.get("text_context_dim", backbone_config.text_dim),
            state_dim=resolved_raw.get("state_dim", data_config.action_schema.state_dim),
            freq_dim=resolved_raw.get("freq_dim", backbone_config.freq_dim),
            num_layers=resolved_raw.get("num_layers", backbone_config.num_layers),
            num_heads=resolved_raw.get("num_heads", backbone_config.num_heads),
            attention_head_dim=resolved_raw.get(
                "attention_head_dim",
                backbone_config.attention_head_dim or (backbone_config.hidden_size // backbone_config.num_heads),
            ),
            ffn_dim=resolved_raw.get(
                "ffn_dim",
                backbone_config.ffn_dim or (backbone_config.hidden_size * backbone_config.mlp_ratio),
            ),
            cross_attn_norm=resolved_raw.get("cross_attn_norm", backbone_config.cross_attn_norm),
            eps=resolved_raw.get("eps", backbone_config.latent_norm_eps),
            input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("input_space", input_space),
            ),
            train_mode=_coerce_enum(
                config_enums.VideoConditionTrainMode,
                resolved_raw.get(
                    "train_mode",
                    config_enums.VideoConditionTrainMode.ROLLOUT_WINDOW_DIFFUSION,
                ),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get("action_chunk_anchor_mode", anchor_mode),
            ),
            action_expert_init_mode=_coerce_enum(
                config_enums.ActionExpertInitMode,
                resolved_raw.get(
                    "action_expert_init_mode",
                    config_enums.ActionExpertInitMode.VIDEO_WEIGHT_COPY,
                ),
            ),
            rollout_chunk_steps=resolved_raw.get("rollout_chunk_steps", 1),
            direct_latent_channels=resolved_raw.get("direct_latent_channels", backbone_config.latent_channels),
            direct_rgb_patch_size=resolved_raw.get("direct_rgb_patch_size", 16),
            use_text_conditioning=resolved_raw.get("use_text_conditioning", True),
            use_state_conditioning=resolved_raw.get("use_state_conditioning", True),
        )
    if name == config_enums.ActionDecoderName.LINGBOT_PARALLEL:
        return LingbotParallelActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            recovered_osc_loss_weight=resolved_raw.get("recovered_osc_loss_weight", 0.0),
            recovered_osc_position_scale=resolved_raw.get(
                "recovered_osc_position_scale",
                LingbotParallelActionDecoderConfig.recovered_osc_position_scale,
            ),
            recovered_osc_rotation_scale=resolved_raw.get(
                "recovered_osc_rotation_scale",
                LingbotParallelActionDecoderConfig.recovered_osc_rotation_scale,
            ),
        )
    if name == config_enums.ActionDecoderName.MOT:
        return MoTActionDecoderConfig(
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
