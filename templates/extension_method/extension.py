"""Registration hook for the extension template."""

from __future__ import annotations

from open_wam.sdk.config import (
    ExperimentConfig,
    ExtensionActionDecoderConfig,
    ExtensionPolicyConfig,
)
from open_wam.sdk.policy import register_action_decoder, register_policy_variant

from .action_decoder import TemplateActionDecoder
from .policy_variant import TemplatePolicyVariant


def _build_policy(experiment: ExperimentConfig) -> TemplatePolicyVariant:
    config = experiment.policy_variant
    if not isinstance(config, ExtensionPolicyConfig):
        raise TypeError("Template policy requires ExtensionPolicyConfig.")
    return TemplatePolicyVariant(config)


def _build_decoder(experiment: ExperimentConfig) -> TemplateActionDecoder:
    config = experiment.action_decoder
    if not isinstance(config, ExtensionActionDecoderConfig):
        raise TypeError("Template decoder requires ExtensionActionDecoderConfig.")
    return TemplateActionDecoder(config)


def register_open_wam() -> None:
    register_policy_variant("example.policy", _build_policy)
    register_action_decoder("example.decoder", _build_decoder)
