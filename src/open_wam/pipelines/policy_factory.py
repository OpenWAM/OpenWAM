"""Configured policy-variant construction and extension dispatch."""

from __future__ import annotations

from open_wam.configs import ExperimentConfig
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
)
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.policy_variants import (
    CausalVideoPredictionPolicyVariant,
    MoTPolicyVariant,
    ParallelStreamPolicyVariant,
    PolicyVariant,
    PostDecodedPolicyVariant,
    PostLatentPolicyVariant,
)

from .factory_validation import _resolve_parallel_stream_model_action_dim
from .registries import POLICY_VARIANT_BUILDERS, _EXTENSION_POLICY_VARIANT_BUILDERS


def _build_post_latent_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    assert isinstance(policy_config, PostLatentPolicyConfig)
    return PostLatentPolicyVariant(
        config=policy_config,
        training_config=config.training,
        inference_config=config.inference,
        action_horizon=action_schema.action_horizon,
        state_dim=action_schema.state_dim,
    )


def _build_post_decoded_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    assert isinstance(policy_config, PostDecodedPolicyConfig)
    return PostDecodedPolicyVariant(
        config=policy_config,
        training_config=config.training,
        inference_config=config.inference,
        action_horizon=action_schema.action_horizon,
        state_dim=action_schema.state_dim,
    )


def _build_causal_video_prediction_policy_variant(config: ExperimentConfig):
    policy_config = config.policy_variant
    assert isinstance(policy_config, CausalVideoPredictionPolicyConfig)
    return CausalVideoPredictionPolicyVariant(
        config=policy_config,
        training_config=config.training,
        inference_config=config.inference,
    )


def _build_mot_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    assert isinstance(policy_config, MoTPolicyConfig)
    return MoTPolicyVariant(
        config=policy_config,
        backbone_config=config.backbone,
        training_config=config.training,
        inference_config=config.inference,
        action_dim=action_schema.action_dim,
        action_horizon=action_schema.action_horizon,
        state_dim=action_schema.state_dim,
    )


def _build_parallel_stream_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    assert isinstance(policy_config, ParallelStreamPolicyConfig)
    return ParallelStreamPolicyVariant(
        config=policy_config,
        backbone_config=config.backbone,
        training_config=config.training,
        inference_config=config.inference,
        action_dim=_resolve_parallel_stream_model_action_dim(config),
        action_horizon=action_schema.action_horizon,
        num_frames=config.data.num_frames,
    )


def _build_extension_policy_variant(config: ExperimentConfig):
    policy_config = config.policy_variant
    assert isinstance(policy_config, ExtensionPolicyConfig)
    builder = _EXTENSION_POLICY_VARIANT_BUILDERS.get(policy_config.extension_type)
    if builder is None:
        registered = ", ".join(_EXTENSION_POLICY_VARIANT_BUILDERS.keys()) or "<none>"
        raise ValueError(
            f"Unsupported policy variant extension {policy_config.extension_type!r}. "
            f"Registered extension types: {registered}. "
            "Load its module with `--extension module[:hook]` before constructing the experiment."
        )
    return builder(config)


def build_policy_variant(config: ExperimentConfig):
    builder = POLICY_VARIANT_BUILDERS.get(type(config.policy_variant))
    if builder is None:
        raise ValueError(f"Unsupported policy variant config '{type(config.policy_variant).__name__}'.")
    policy_variant = builder(config)
    if not isinstance(policy_variant, PolicyVariant):
        raise TypeError(
            f"Policy variant builder returned {type(policy_variant).__name__}; "
            "expected an `open_wam.models.policy_variants.PolicyVariant`."
        )
    return policy_variant


__all__ = ["build_policy_variant"]
