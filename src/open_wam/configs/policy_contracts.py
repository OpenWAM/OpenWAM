"""Shared and attachment-specific policy configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ActionDecoderName,
    AttachSite,
    BackboneImplementation,
    BatchingMode,
    CausalVideoProgram,
    DynamicsObjective,
    PolicyVariantName,
    ProprioContextMode,
    TextConditioningMode,
    coerce_fields,
)


@dataclass(frozen=True, slots=True)
class PolicyConditioningRequirements:
    """Shared visual-stack adapters requested by a policy configuration."""

    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    dynamics_mode_context_enabled: bool = False
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proprio_context_mode",
            ProprioContextMode(self.proprio_context_mode),
        )
        object.__setattr__(
            self,
            "text_conditioning_mode",
            TextConditioningMode(self.text_conditioning_mode),
        )


@dataclass(frozen=True)
class PolicyVariantConfig:
    """Base config shared by all policy variants."""

    name: PolicyVariantName
    hidden_size: int
    attach_site: AttachSite

    @property
    def supported_batching_modes(self) -> tuple[BatchingMode, ...]:
        """Execution capabilities, available before model allocation."""
        return (BatchingMode.STRICT,)

    @property
    def supported_backbone_implementations(
        self,
    ) -> tuple[BackboneImplementation, ...]:
        """Return an empty tuple for no policy-level backbone restriction."""

        return ()

    @property
    def default_action_decoder(self) -> ActionDecoderName | None:
        """Return the decoder selected when a config omits that component."""

        return None

    @property
    def fixed_conditioning_mode(self) -> DynamicsObjective | None:
        """Return a fixed dynamics objective, when the policy declares one."""

        return None

    @property
    def conditioning_requirements(self) -> PolicyConditioningRequirements:
        """Declare shared visual adapters before policy modules are allocated."""

        return PolicyConditioningRequirements()

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "name": PolicyVariantName,
                "attach_site": AttachSite,
            },
        )


@dataclass(frozen=True)
class ExtensionPolicyConfig(PolicyVariantConfig):
    """Config envelope for an application-owned policy variant."""

    name: PolicyVariantName = PolicyVariantName.EXTENSION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    extension_type: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    dynamics_mode_context_enabled: bool = False
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT

    @property
    def conditioning_requirements(self) -> PolicyConditioningRequirements:
        return PolicyConditioningRequirements(
            proprio_context_mode=self.proprio_context_mode,
            dynamics_mode_context_enabled=bool(self.dynamics_mode_context_enabled),
            text_conditioning_mode=self.text_conditioning_mode,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "proprio_context_mode": ProprioContextMode,
                "text_conditioning_mode": TextConditioningMode,
            },
        )
        if self.name != PolicyVariantName.EXTENSION:
            raise ValueError("Extension policy requires `name = extension`.")
        if not isinstance(self.extension_type, str) or not self.extension_type.strip():
            raise ValueError(
                "Extension policy requires a non-empty `extension_type` string."
            )
        if self.extension_type != self.extension_type.strip():
            raise ValueError(
                "Extension policy `extension_type` must not have surrounding whitespace."
            )
        if not isinstance(self.options, Mapping):
            raise TypeError("Extension policy `options` must be a mapping.")
        if not all(isinstance(key, str) for key in self.options):
            raise ValueError("Extension policy `options` keys must be strings.")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class CausalVideoPredictionPolicyConfig(PolicyVariantConfig):
    """Standalone causal video-only pretraining variant."""

    name: PolicyVariantName = PolicyVariantName.CAUSAL_VIDEO_PREDICTION
    hidden_size: int = 256
    attach_site: AttachSite = AttachSite.POST_VISUAL_CORE
    program: CausalVideoProgram | None = None
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT
    noisy_video_condition_prob: float | None = None
    # Recompute shared video blocks during backward for long-segment training.
    use_activation_checkpointing: bool = False

    @property
    def default_action_decoder(self) -> ActionDecoderName:
        return ActionDecoderName.VIDEO_ONLY

    @property
    def supported_backbone_implementations(
        self,
    ) -> tuple[BackboneImplementation, ...]:
        return (BackboneImplementation.SHARED_TRANSFORMER,)

    @property
    def conditioning_requirements(self) -> PolicyConditioningRequirements:
        return PolicyConditioningRequirements(
            text_conditioning_mode=self.text_conditioning_mode,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={"text_conditioning_mode": TextConditioningMode},
            optional_enum_fields={"program": CausalVideoProgram},
        )
        if self.program is None:
            raise ValueError(
                "Causal video prediction requires an explicit `policy_variant.program`."
            )
        if self.program == CausalVideoProgram.PREFIX_SUFFIX:
            if self.noisy_video_condition_prob is not None:
                raise ValueError(
                    "`noisy_video_condition_prob` is not part of the prefix/suffix "
                    "video program; omit it."
                )
            if self.use_activation_checkpointing:
                raise ValueError(
                    "`use_activation_checkpointing` is only implemented by the "
                    "chunked conditioned-video program."
                )
        else:
            if self.noisy_video_condition_prob is None:
                raise ValueError(
                    "Chunked conditioned video requires an explicit "
                    "`noisy_video_condition_prob`."
                )
            if not 0.0 <= float(self.noisy_video_condition_prob) <= 1.0:
                raise ValueError(
                    "Chunked conditioned video requires "
                    "`0 <= noisy_video_condition_prob <= 1`, got "
                    f"{self.noisy_video_condition_prob!r}."
                )
            if not isinstance(self.use_activation_checkpointing, bool):
                raise TypeError(
                    "Chunked conditioned video requires "
                    "`use_activation_checkpointing` to be boolean."
                )
        if self.attach_site != AttachSite.POST_VISUAL_CORE:
            raise ValueError(
                "Causal video prediction requires `attach_site = post_visual_core`, "
                f"got attach_site={self.attach_site!r}."
            )


__all__ = [
    "CausalVideoPredictionPolicyConfig",
    "ExtensionPolicyConfig",
    "PolicyConditioningRequirements",
    "PolicyVariantConfig",
]
