from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    BackboneImplementation,
    ExperimentConfig,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterAttachedPolicyConfig,
    VideoSequencePolicyConfig,
)
from open_wam.data import build_canonical_video_preprocessor
from open_wam.models.action_decoders import DecodedFeatureActionDecoder, MLPActionDecoder, RegisterActionDecoder
from open_wam.models.action_decoders import LingbotParallelActionDecoder, VPPSequenceActionDecoder
from open_wam.models.policy_variants import (
    ParallelStreamPolicyVariant,
    PostDecodedPolicyVariant,
    PostLatentPolicyVariant,
    RegisterAttachedPolicyVariant,
    VideoSequencePolicyVariant,
)
from open_wam.models.policy_variants.parallel_stream.action_adapter import build_action_adapter_spec
from open_wam.models.visual_tower import VisualTower
from open_wam.models.video_backbone import normalize_backbone_implementation

from .variant_pipeline import VariantPipeline
from .lingbot_exact import LingbotExactRunner


def _resolve_parallel_stream_model_action_dim(config: ExperimentConfig) -> int:
    if (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT
    ):
        return config.action_decoder.action_dim
    return config.data.action_schema.action_dim


def validate_experiment_config(config: ExperimentConfig) -> None:
    action_schema = config.data.action_schema
    if isinstance(
        config.policy_variant,
        (
            ParallelStreamPolicyConfig,
            RegisterAttachedPolicyConfig,
            PostLatentPolicyConfig,
            PostDecodedPolicyConfig,
            VideoSequencePolicyConfig,
        ),
    ):
        if normalize_backbone_implementation(config.backbone.implementation) != BackboneImplementation.SHARED_TRANSFORMER:
            raise ValueError(
                "Policy variants in the current repo all require the shared transformer backbone so they run "
                f"through the same LingBot-compatible visual core, got "
                f"backbone.implementation={config.backbone.implementation!r}."
            )
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        if config.policy_variant.runtime_mode != ParallelRuntimeMode.LINGBOT_EXACT:
            raise ValueError(
                "Parallel-stream method 1 now only supports LingBot-exact semantics, "
                f"got policy_variant.runtime_mode={config.policy_variant.runtime_mode!r}."
            )
        expected_horizon = config.data.num_frames * config.policy_variant.action_per_frame
        if action_schema.action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream config requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_schema.action_horizon}, num_frames={config.data.num_frames}, "
                f"action_per_frame={config.policy_variant.action_per_frame}."
            )
        if config.action_decoder.name != ActionDecoderName.LINGBOT_PARALLEL:
            raise ValueError(
                "Parallel-stream method 1 requires `action_decoder.name = lingbot_parallel_decoder`."
            )
        if config.action_decoder.action_horizon != action_schema.action_horizon:
            raise ValueError(
                "Parallel-stream method 1 requires `action_decoder.action_horizon` to match "
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
    if isinstance(config.policy_variant, RegisterAttachedPolicyConfig):
        num_frames = config.data.num_frames
        if (num_frames - 1) % config.policy_variant.num_frame_per_block != 0:
            raise ValueError(
                "Register-attached config requires `(num_frames - 1)` to be divisible by "
                "`policy_variant.num_frame_per_block`, "
                f"got num_frames={num_frames}, "
                f"num_frame_per_block={config.policy_variant.num_frame_per_block}."
            )
        if action_schema.action_horizon % config.policy_variant.num_action_per_block != 0:
            raise ValueError(
                "Register-attached config requires `action_horizon` to be divisible by "
                "`policy_variant.num_action_per_block`, "
                f"got action_horizon={action_schema.action_horizon}, "
                f"num_action_per_block={config.policy_variant.num_action_per_block}."
            )
        if action_schema.state_horizon % config.policy_variant.num_state_per_block != 0:
            raise ValueError(
                "Register-attached config requires `state_horizon` to be divisible by "
                "`policy_variant.num_state_per_block`, "
                f"got state_horizon={action_schema.state_horizon}, "
                f"num_state_per_block={config.policy_variant.num_state_per_block}."
            )
        image_block_count = (num_frames - 1) // config.policy_variant.num_frame_per_block
        action_block_count = action_schema.action_horizon // config.policy_variant.num_action_per_block
        state_block_count = action_schema.state_horizon // config.policy_variant.num_state_per_block
        if image_block_count != action_block_count or image_block_count != state_block_count:
            raise ValueError(
                "Register-attached config requires image, action, and state block counts to match, "
                f"got image={image_block_count}, action={action_block_count}, state={state_block_count}. "
                "For raw LIBERO this usually means increasing `data.action_schema.state_horizon` so the "
                "state register blocks align with the future image/action blocks."
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
    if isinstance(policy_config, VideoSequencePolicyConfig):
        return VideoSequencePolicyVariant(
            config=policy_config,
            training_config=config.training,
            inference_config=config.inference,
            action_horizon=action_schema.action_horizon,
            state_dim=action_schema.state_dim,
        )
    if isinstance(policy_config, RegisterAttachedPolicyConfig):
        return RegisterAttachedPolicyVariant(
            config=policy_config,
            backbone_config=config.backbone,
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
    if decoder_config.name == ActionDecoderName.MLP:
        return MLPActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == ActionDecoderName.REGISTER:
        return RegisterActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == ActionDecoderName.DECODED_FEATURE:
        return DecodedFeatureActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == ActionDecoderName.LINGBOT_PARALLEL:
        return LingbotParallelActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == ActionDecoderName.VPP:
        return VPPSequenceActionDecoder(
            decoder_config,
            training_config=config.training,
            inference_config=config.inference,
            state_dim=config.data.action_schema.state_dim,
        )
    raise ValueError(f"Unsupported action decoder '{decoder_config.name}'.")


def build_variant_pipeline_from_config(config: ExperimentConfig) -> VariantPipeline:
    validate_experiment_config(config)
    policy_action_dim = _resolve_parallel_stream_model_action_dim(config)
    return VariantPipeline(
        visual_tower=VisualTower(
            config.backbone,
            action_dim=policy_action_dim,
            state_dim=config.data.action_schema.state_dim,
        ),
        policy_variant=build_policy_variant(config),
        action_decoder=build_action_decoder(config),
        preprocessor=build_canonical_video_preprocessor(config.data),
    )


def build_lingbot_exact_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return LingbotExactRunner(build_variant_pipeline_from_config(config))


def build_exact_runtime_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return build_lingbot_exact_runner_from_config(config)
