from __future__ import annotations

from open_wam.configs import (
    ExperimentConfig,
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterAttachedPolicyConfig,
)
from open_wam.data import build_canonical_video_preprocessor
from open_wam.models.action_decoders import DecodedFeatureActionDecoder, MLPActionDecoder, RegisterActionDecoder
from open_wam.models.action_decoders import LingbotParallelActionDecoder
from open_wam.models.policy_variants import (
    ParallelStreamPolicyVariant,
    PostDecodedPolicyVariant,
    PostLatentPolicyVariant,
    RegisterAttachedPolicyVariant,
)
from open_wam.models.visual_tower import VisualTower

from .variant_pipeline import VariantPipeline
from .lingbot_exact import LingbotExactMethod1Runner


def build_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    if isinstance(policy_config, PostLatentPolicyConfig):
        return PostLatentPolicyVariant(
            config=policy_config,
            training_config=config.training,
            inference_config=config.inference,
            action_horizon=action_schema.action_horizon,
            state_dim=action_schema.state_dim,
        )
    if isinstance(policy_config, PostDecodedPolicyConfig):
        return PostDecodedPolicyVariant(
            config=policy_config,
            training_config=config.training,
            inference_config=config.inference,
            action_horizon=action_schema.action_horizon,
            state_dim=action_schema.state_dim,
        )
    if isinstance(policy_config, RegisterAttachedPolicyConfig):
        return RegisterAttachedPolicyVariant(
            config=policy_config,
            training_config=config.training,
            inference_config=config.inference,
            action_dim=action_schema.action_dim,
            action_horizon=action_schema.action_horizon,
            state_dim=action_schema.state_dim,
            state_horizon=action_schema.state_horizon,
        )
    if isinstance(policy_config, ParallelStreamPolicyConfig):
        return ParallelStreamPolicyVariant(
            config=policy_config,
            backbone_config=config.backbone,
            training_config=config.training,
            inference_config=config.inference,
            action_dim=action_schema.action_dim,
            action_horizon=action_schema.action_horizon,
            num_frames=config.data.num_frames,
        )
    raise ValueError(f"Unsupported policy variant config '{type(policy_config).__name__}'.")


def build_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    if decoder_config.name == "mlp_decoder":
        return MLPActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == "register_decoder":
        return RegisterActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == "decoded_feature_decoder":
        return DecodedFeatureActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == "lingbot_parallel_decoder":
        return LingbotParallelActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            dropout=decoder_config.dropout,
        )
    raise ValueError(f"Unsupported action decoder '{decoder_config.name}'.")


def build_variant_pipeline_from_config(config: ExperimentConfig) -> VariantPipeline:
    return VariantPipeline(
        visual_tower=VisualTower(config.backbone),
        policy_variant=build_policy_variant(config),
        action_decoder=build_action_decoder(config),
        preprocessor=build_canonical_video_preprocessor(config.data),
    )


def build_lingbot_exact_runner_from_config(config: ExperimentConfig) -> LingbotExactMethod1Runner:
    return LingbotExactMethod1Runner(build_variant_pipeline_from_config(config))
