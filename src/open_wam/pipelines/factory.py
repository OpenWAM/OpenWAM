"""Stable pipeline composition and historical factory import surface."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    ActionNormalizationMode,
    BackboneImplementation,
    BatchAdapterName,
    ExperimentConfig,
    ExtensionActionDecoderConfig,
    ParallelRuntimeMode,
    ProprioContextMode,
)
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.resolution import (
    resolve_experiment_config as _resolve_experiment_config,
)
from open_wam.data import build_canonical_video_preprocessor
from open_wam.data.action_mapping import (
    build_action_sampler_mask,
    validate_action_mapping_preflight,
)
from open_wam.models.action_decoders import (
    ActionDecoder,
    DualExpertActionDecoder,
    ParallelStreamActionDecoder,
    VideoOnlyActionDecoder,
)
from open_wam.models.policy_variants import (
    CausalVideoPredictionPolicyVariant,
    DualExpertPolicyVariant,
    ParallelStreamPolicyVariant,
    PolicyVariant,
)
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    build_action_adapter_spec,
)
from open_wam.models.video_backbone import normalize_backbone_implementation
from open_wam.models.visual_tower import VisualTower

from .action_decoder_factory import (
    _build_dual_expert_action_decoder,
    _build_extension_action_decoder,
    _build_parallel_stream_action_decoder,
    _build_video_only_action_decoder,
    build_action_decoder,
)
from .factory_validation import (
    validate_experiment_config,
)
from .lingbot_exact import LingbotExactRunner
from .policy_factory import (
    _build_causal_video_prediction_policy_variant,
    _build_dual_expert_policy_variant,
    _build_extension_policy_variant,
    _build_parallel_stream_policy_variant,
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
    ParallelRuntimeMode,
    ProprioContextMode,
    validate_action_mapping_preflight,
    ActionDecoder,
    ParallelStreamActionDecoder,
    DualExpertActionDecoder,
    VideoOnlyActionDecoder,
    CausalVideoPredictionPolicyVariant,
    DualExpertPolicyVariant,
    ParallelStreamPolicyVariant,
    PolicyVariant,
    build_action_adapter_spec,
    normalize_backbone_implementation,
    _EXTENSION_ACTION_DECODER_BUILDERS,
    _EXTENSION_POLICY_VARIANT_BUILDERS,
)


def _register_builtin_pipeline_builders() -> None:
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

    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.PARALLEL_STREAM,
        _build_parallel_stream_action_decoder,
        replace=True,
    )
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.DUAL_EXPERT, _build_dual_expert_action_decoder, replace=True
    )
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.VIDEO_ONLY, _build_video_only_action_decoder, replace=True
    )
    ACTION_DECODER_BUILDERS.register(
        ActionDecoderName.EXTENSION,
        _build_extension_action_decoder,
        replace=True,
    )


_register_builtin_pipeline_builders()


def build_variant_pipeline_from_config(config: ExperimentConfig) -> VariantPipeline:
    config = _resolve_experiment_config(config)
    validate_experiment_config(config)
    action_schema = config.data.action_schema
    visual_tower = VisualTower(
        config.backbone,
        action_dim=config.action_decoder.action_dim,
        state_dim=action_schema.state_dim,
    )
    conditioning_requirements = config.policy_variant.conditioning_requirements
    visual_tower.configure_policy_conditioning(
        proprio_context_mode=conditioning_requirements.proprio_context_mode,
        dynamics_mode_context_enabled=(
            conditioning_requirements.dynamics_mode_context_enabled
        ),
        text_conditioning_mode=conditioning_requirements.text_conditioning_mode,
    )
    visual_tower.initialize_configured_weights()

    policy_variant = build_policy_variant(config)
    pipeline_requirements = policy_variant.pipeline_requirements(
        default_action_dim=config.action_decoder.action_dim,
        default_action_horizon=config.action_decoder.action_horizon,
        default_state_dim=action_schema.state_dim,
    )
    pipeline_requirements.validate_source_action_shape(
        action_dim=action_schema.action_dim,
        action_horizon=action_schema.action_horizon,
    )
    pipeline_requirements.validate_action_decoder(
        action_dim=config.action_decoder.action_dim,
        action_horizon=config.action_decoder.action_horizon,
    )
    pipeline_requirements.validate_visual_tower(
        action_dim=visual_tower.action_dim,
        state_dim=visual_tower.state_dim,
    )
    pipeline_requirements.validate_conditioning(conditioning_requirements)
    policy_variant.validate_pipeline_assembly(
        data_action_dim=action_schema.action_dim,
        num_frames=config.data.num_frames,
        backbone_num_layers=config.backbone.num_layers,
    )
    # Finalize variant-owned module attachment before distributed wrapping.
    policy_variant.attach_visual_tower(visual_tower)
    action_decoder = build_action_decoder(config)
    action_decoder.configure_pipeline_requirements(pipeline_requirements)
    action_sampler_mask = build_action_sampler_mask(
        config.data.action_mapping,
        action_horizon=config.action_decoder.action_horizon,
        target_dim=pipeline_requirements.action_dim,
    )
    return VariantPipeline(
        visual_tower=visual_tower,
        policy_variant=policy_variant,
        action_decoder=action_decoder,
        preprocessor=build_canonical_video_preprocessor(config.data),
        action_sampler_mask=action_sampler_mask,
        action_sampler_inactive_value=config.data.action_mapping.inactive_value,
    )


def build_lingbot_exact_runner_from_config(
    config: ExperimentConfig,
) -> LingbotExactRunner:
    return LingbotExactRunner(build_variant_pipeline_from_config(config))


def build_exact_runtime_runner_from_config(
    config: ExperimentConfig,
) -> LingbotExactRunner:
    return build_lingbot_exact_runner_from_config(config)
