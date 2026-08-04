"""Stable pipeline composition and historical factory import surface."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    ActionNormalizationMode,
    BackboneImplementation,
    BatchAdapterName,
    DualExpertRuntimeMode,
    ExperimentConfig,
    ExtensionActionDecoderConfig,
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
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.data import build_canonical_video_preprocessor
from open_wam.data.action_mapping import (
    build_action_sampler_mask,
    validate_action_mapping_preflight,
)
from open_wam.models.action_decoders import (
    ActionDecoder,
    DecodedFeatureActionDecoder,
    DualExpertActionDecoder,
    MLPActionDecoder,
    ParallelStreamActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
)
from open_wam.models.policy_variants import (
    CausalVideoPredictionPolicyVariant,
    DualExpertPolicyVariant,
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

from .action_decoder_factory import (
    _build_decoded_feature_action_decoder,
    _build_dual_expert_action_decoder,
    _build_extension_action_decoder,
    _build_mlp_action_decoder,
    _build_parallel_stream_action_decoder,
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
from .lingbot_exact import LingbotExactRunner
from .policy_factory import (
    _build_causal_video_prediction_policy_variant,
    _build_dual_expert_policy_variant,
    _build_extension_policy_variant,
    _build_parallel_stream_policy_variant,
    _build_post_decoded_policy_variant,
    _build_post_latent_policy_variant,
    build_policy_variant,
)
from .registries import (
    _EXTENSION_ACTION_DECODER_BUILDERS,
    _EXTENSION_POLICY_VARIANT_BUILDERS,
    ACTION_DECODER_BUILDERS,
    POLICY_VARIANT_BUILDERS,
)
from .variant_pipeline import VariantPipeline

# Preserve the historical direct/wildcard import surface without making these
# implementation dependencies of the composition owner.
_COMPATIBILITY_EXPORTS = (
    ActionNormalizationMode,
    BackboneImplementation,
    BatchAdapterName,
    ExtensionActionDecoderConfig,
    DualExpertRuntimeMode,
    ParallelRuntimeMode,
    ProprioContextMode,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
    validate_action_mapping_preflight,
    ActionDecoder,
    DecodedFeatureActionDecoder,
    ParallelStreamActionDecoder,
    MLPActionDecoder,
    DualExpertActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
    CausalVideoPredictionPolicyVariant,
    DualExpertPolicyVariant,
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
        DualExpertPolicyConfig,
        _build_dual_expert_policy_variant,
        description="Dual-expert video/action policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        ParallelStreamPolicyConfig,
        _build_parallel_stream_policy_variant,
        description="Parallel-stream video/action policy variant.",
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
        ActionDecoderName.PARALLEL_STREAM,
        _build_parallel_stream_action_decoder,
        replace=True,
    )
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.DUAL_EXPERT, _build_dual_expert_action_decoder, replace=True)
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
    # Pipeline-time hook for variants that need cross-module surgery (e.g. DualExpert
    # packed coupling transfers video core/action expert blocks into one
    # DualExpertPackedBlockStack). Must run before FSDP sharding.
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
