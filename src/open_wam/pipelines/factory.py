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
from open_wam.models.policy_variants.parallel_stream.action_adapter import build_action_adapter_spec
from open_wam.models.visual_tower import VisualTower

from .variant_pipeline import VariantPipeline
from .lingbot_exact import LingbotExactRunner


def _resolve_parallel_stream_model_action_dim(config: ExperimentConfig) -> int:
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig) and config.policy_variant.runtime_mode == "lingbot_exact":
        return config.action_decoder.action_dim
    return config.data.action_schema.action_dim


def validate_experiment_config(config: ExperimentConfig) -> None:
    action_schema = config.data.action_schema
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        expected_horizon = config.data.num_frames * config.policy_variant.action_per_frame
        if action_schema.action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream config requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_schema.action_horizon}, num_frames={config.data.num_frames}, "
                f"action_per_frame={config.policy_variant.action_per_frame}."
            )
        if config.policy_variant.runtime_mode == "lingbot_exact":
            if config.action_decoder.name != "lingbot_parallel_decoder":
                raise ValueError(
                    "Exact LingBot runtime requires `action_decoder.name = lingbot_parallel_decoder`."
                )
            if config.action_decoder.action_horizon != action_schema.action_horizon:
                raise ValueError(
                    "Exact LingBot runtime requires `action_decoder.action_horizon` to match "
                    "`data.action_schema.action_horizon`, "
                    f"got decoder={config.action_decoder.action_horizon}, data={action_schema.action_horizon}."
                )
            model_action_dim = _resolve_parallel_stream_model_action_dim(config)
            adapter_spec = build_action_adapter_spec(config.policy_variant, model_action_dim=model_action_dim)
            if adapter_spec is None and action_schema.action_dim != model_action_dim:
                raise ValueError(
                    "Exact LingBot runtime needs an action adapter when dataset and model action dims differ, "
                    f"got data action_dim={action_schema.action_dim} and model action_dim={model_action_dim}."
                )
            if (
                adapter_spec is not None
                and action_schema.action_dim != model_action_dim
                and adapter_spec.raw_action_dim != action_schema.action_dim
            ):
                raise ValueError(
                    "Exact LingBot action adapter raw action dim must match `data.action_schema.action_dim` "
                    "when dataset and model action dims differ, "
                    f"got adapter raw_action_dim={adapter_spec.raw_action_dim}, "
                    f"data action_dim={action_schema.action_dim}, model action_dim={model_action_dim}."
                )


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
            action_dim=_resolve_parallel_stream_model_action_dim(config),
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
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == "register_decoder":
        return RegisterActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == "decoded_feature_decoder":
        return DecodedFeatureActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
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
    validate_experiment_config(config)
    policy_action_dim = _resolve_parallel_stream_model_action_dim(config)
    return VariantPipeline(
        visual_tower=VisualTower(
            config.backbone,
            action_dim=policy_action_dim,
        ),
        policy_variant=build_policy_variant(config),
        action_decoder=build_action_decoder(config),
        preprocessor=build_canonical_video_preprocessor(config.data),
    )


def build_lingbot_exact_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return LingbotExactRunner(build_variant_pipeline_from_config(config))
