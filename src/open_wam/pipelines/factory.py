"""Stable pipeline composition and historical factory import surface."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    ActionNormalizationMode,
    BackboneImplementation,
    BatchAdapterName,
    ExperimentConfig,
    ExtensionActionDecoderConfig,
    MoTRuntimeMode,
    ParallelRuntimeMode,
    ProprioContextMode,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
)
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
)
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.data import build_canonical_video_preprocessor
from open_wam.data.action_mapping import (
    build_action_sampler_mask,
    validate_action_mapping_preflight,
)
from open_wam.models.action_decoders import (
    ActionDecoder,
    DecodedFeatureActionDecoder,
    LingbotParallelActionDecoder,
    MLPActionDecoder,
    MoTActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
)
from open_wam.models.policy_variants import (
    CausalVideoPredictionPolicyVariant,
    MoTPolicyVariant,
    ParallelStreamPolicyVariant,
    PolicyVariant,
    PostDecodedPolicyVariant,
    PostLatentPolicyVariant,
)
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    build_action_adapter_spec,
)
from open_wam.models.video_backbone import normalize_backbone_implementation
from open_wam.models.visual_tower import VisualTower

from .lingbot_exact import LingbotExactRunner
from .registries import (
    ACTION_DECODER_BUILDERS,
    POLICY_VARIANT_BUILDERS,
    _EXTENSION_ACTION_DECODER_BUILDERS,
    _EXTENSION_POLICY_VARIANT_BUILDERS,
)
from .variant_pipeline import VariantPipeline

from .action_decoder_factory import (
    _build_decoded_feature_action_decoder,
    _build_extension_action_decoder,
    _build_lingbot_parallel_action_decoder,
    _build_mlp_action_decoder,
    _build_mot_action_decoder,
    _build_video_conditioned_action_decoder,
    _build_video_only_action_decoder,
    build_action_decoder,
)
from .factory_validation import (
    _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES,
    _resolve_parallel_stream_model_action_dim,
    _resolve_proprio_context_state_dim,
    _resolve_proprio_hidden_context_state_dim,
    validate_experiment_config,
)
from .policy_factory import (
    _build_causal_video_prediction_policy_variant,
    _build_extension_policy_variant,
    _build_mot_policy_variant,
    _build_parallel_stream_policy_variant,
    _build_post_decoded_policy_variant,
    _build_post_latent_policy_variant,
    build_policy_variant,
)


# Preserve the historical direct/wildcard import surface without making these
# implementation dependencies of the composition owner.
_COMPATIBILITY_EXPORTS = (
    ActionNormalizationMode,
    BackboneImplementation,
    BatchAdapterName,
    ExtensionActionDecoderConfig,
    MoTRuntimeMode,
    ParallelRuntimeMode,
    ProprioContextMode,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
    validate_action_mapping_preflight,
    ActionDecoder,
    DecodedFeatureActionDecoder,
    LingbotParallelActionDecoder,
    MLPActionDecoder,
    MoTActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
    CausalVideoPredictionPolicyVariant,
    MoTPolicyVariant,
    ParallelStreamPolicyVariant,
    PolicyVariant,
    PostDecodedPolicyVariant,
    PostLatentPolicyVariant,
    build_action_adapter_spec,
    normalize_backbone_implementation,
    _EXTENSION_ACTION_DECODER_BUILDERS,
    _EXTENSION_POLICY_VARIANT_BUILDERS,
    _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES,
)


def _register_builtin_pipeline_builders() -> None:
    POLICY_VARIANT_BUILDERS.register(
        PostLatentPolicyConfig,
        _build_post_latent_policy_variant,
        description="Post-latent policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        PostDecodedPolicyConfig,
        _build_post_decoded_policy_variant,
        description="Post-decoded policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        CausalVideoPredictionPolicyConfig,
        _build_causal_video_prediction_policy_variant,
        description="Video-only causal prediction policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        MoTPolicyConfig,
        _build_mot_policy_variant,
        description="Mixture-of-transformers policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        ParallelStreamPolicyConfig,
        _build_parallel_stream_policy_variant,
        description="Parallel-stream LingBot-compatible policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        ExtensionPolicyConfig,
        _build_extension_policy_variant,
        description="Application-owned policy variant.",
        replace=True,
    )

    ACTION_DECODER_BUILDERS.register(ActionDecoderName.MLP, _build_mlp_action_decoder, replace=True)
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.DECODED_FEATURE,
        _build_decoded_feature_action_decoder,
        replace=True,
    )
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.VIDEO_CONDITIONED,
        _build_video_conditioned_action_decoder,
        replace=True,
    )
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.LINGBOT_PARALLEL,
        _build_lingbot_parallel_action_decoder,
        replace=True,
    )
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.MOT, _build_mot_action_decoder, replace=True)
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.VIDEO_ONLY, _build_video_only_action_decoder, replace=True)
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.EXTENSION,
        _build_extension_action_decoder,
        replace=True,
    )


_register_builtin_pipeline_builders()


def build_variant_pipeline_from_config(config: ExperimentConfig) -> VariantPipeline:
    validate_experiment_config(config)
    policy_action_dim = _resolve_parallel_stream_model_action_dim(config)
    proprio_context_state_dim = _resolve_proprio_context_state_dim(config)
    proprio_hidden_context_state_dim = _resolve_proprio_hidden_context_state_dim(config)
    visual_tower = VisualTower(
        config.backbone,
        action_dim=policy_action_dim,
        state_dim=config.data.action_schema.state_dim,
        proprio_context_state_dim=proprio_context_state_dim,
        proprio_hidden_context_state_dim=proprio_hidden_context_state_dim,
        generalist_mode_context_enabled=bool(
            getattr(config.policy_variant, "generalist_mode_text_token", False)
        ),
    )
    policy_variant = build_policy_variant(config)
    # Pipeline-time hook for variants that need cross-module surgery (e.g. MoT
    # packed coupling transfers video core/action expert blocks into one
    # MoTPackedBlockStack). Must run before FSDP sharding.
    if hasattr(policy_variant, "attach_visual_tower"):
        policy_variant.attach_visual_tower(visual_tower)
    action_decoder = build_action_decoder(config)
    if (
        config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
        and hasattr(action_decoder, "initialize_from_video_core")
    ):
        action_decoder.initialize_from_video_core(visual_tower.core)
    action_sampler_mask = build_action_sampler_mask(
        config.data.action_mapping,
        action_horizon=config.action_decoder.action_horizon,
        target_dim=config.action_decoder.action_dim,
    )
    return VariantPipeline(
        visual_tower=visual_tower,
        policy_variant=policy_variant,
        action_decoder=action_decoder,
        preprocessor=build_canonical_video_preprocessor(config.data),
        action_sampler_mask=action_sampler_mask,
        action_sampler_inactive_value=config.data.action_mapping.inactive_value,
    )


def build_lingbot_exact_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return LingbotExactRunner(build_variant_pipeline_from_config(config))


def build_exact_runtime_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return build_lingbot_exact_runner_from_config(config)
