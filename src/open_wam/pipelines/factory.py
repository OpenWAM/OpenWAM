from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    BatchAdapterName,
    BackboneImplementation,
    CausalVideoPredictionPolicyConfig,
    ExperimentConfig,
    MoTPolicyConfig,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterAttachedPolicyConfig,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
    VideoSequencePolicyConfig,
)
from open_wam.data import build_canonical_video_preprocessor
from open_wam.models.action_decoders import DecodedFeatureActionDecoder, MLPActionDecoder, RegisterActionDecoder
from open_wam.models.action_decoders import (
    LingbotParallelActionDecoder,
    MoTActionDecoder,
    VPPSequenceActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
)
from open_wam.models.policy_variants import (
    CausalVideoPredictionPolicyVariant,
    MoTPolicyVariant,
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
        and config.policy_variant.runtime_mode
        in {
            ParallelRuntimeMode.LINGBOT_EXACT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        }
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
            CausalVideoPredictionPolicyConfig,
            MoTPolicyConfig,
        ),
    ):
        if normalize_backbone_implementation(config.backbone.implementation) != BackboneImplementation.SHARED_TRANSFORMER:
            raise ValueError(
                "Policy variants in the current repo all require the shared transformer backbone so they run "
                f"through the same LingBot-compatible visual core, got "
                f"backbone.implementation={config.backbone.implementation!r}."
            )
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        if config.policy_variant.runtime_mode not in {
            ParallelRuntimeMode.LINGBOT_EXACT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        }:
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
    if isinstance(config.policy_variant, MoTPolicyConfig):
        if action_schema.action_horizon <= 0:
            raise ValueError("MoT method 5 requires `data.action_schema.action_horizon > 0`.")
        if config.policy_variant.video_prefix_frames >= config.data.num_frames:
            raise ValueError(
                "MoT method 5 requires `video_prefix_frames < data.num_frames`, "
                f"got video_prefix_frames={config.policy_variant.video_prefix_frames}, "
                f"data.num_frames={config.data.num_frames}."
            )
        if config.policy_variant.num_action_layers != config.backbone.num_layers:
            raise ValueError(
                "MoT method 5 currently requires `policy_variant.num_action_layers == backbone.num_layers` "
                "so the action expert stays layer-aligned with the video expert, "
                f"got num_action_layers={config.policy_variant.num_action_layers}, "
                f"backbone.num_layers={config.backbone.num_layers}."
            )
    if (
        isinstance(config.policy_variant, (PostLatentPolicyConfig, PostDecodedPolicyConfig))
        and config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    ):
        direct_train_mode = config.action_decoder.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
        if config.policy_variant.local_video_window_frames > config.data.num_frames:
            raise ValueError(
                "Method-4 video-conditioned decoding requires `policy_variant.local_video_window_frames <= data.num_frames`, "
                f"got local_video_window_frames={config.policy_variant.local_video_window_frames}, "
                f"data.num_frames={config.data.num_frames}."
            )
        if config.action_decoder.action_horizon <= 0:
            raise ValueError("Method-4 video-conditioned decoding requires `action_horizon > 0`.")
        if not direct_train_mode and int(config.policy_variant.current_video_frame_index) != 0:
            raise ValueError(
                "Method-4 rollout-window decoding currently supports only `current_video_frame_index = 0`. "
                "Non-zero sliding-window alignment is not implemented yet."
            )
        if (
            direct_train_mode
            and config.policy_variant.train_video_condition_source == VideoConditionSource.GENERATED_FUTURE
        ):
            raise ValueError(
                "Method-4 generated-future video conditioning is only supported for rollout-window diffusion "
                "training. `current_frame_regression` bypasses the policy-variant window builder."
            )
        if direct_train_mode:
            if config.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT:
                if config.trainer.batch_adapter != BatchAdapterName.LATENTS:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `video_latent` input requires the latent batch "
                        "adapter so training sees dataset video latents directly, "
                        f"got trainer.batch_adapter={config.trainer.batch_adapter!r}."
                    )
                if config.data.dataset_type != "lerobot_v2_latent_local":
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `video_latent` input is currently maintained "
                        "only for latent-local datasets, "
                        f"got data.dataset_type={config.data.dataset_type!r}."
                    )
            if config.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO:
                if config.trainer.batch_adapter != BatchAdapterName.VIEWS:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input requires the view batch "
                        "adapter so training sees raw RGB frames directly, "
                        f"got trainer.batch_adapter={config.trainer.batch_adapter!r}."
                    )
                if config.data.dataset_type == "lerobot_v2_latent_local":
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input requires raw RGB dataset "
                        "windows. Latent-local datasets enter through precomputed latents, so use "
                        "`video_latent` there."
                    )
                if config.action_decoder.use_text_conditioning:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input and the view batch adapter "
                        "does not currently provide text embeddings. Set `action_decoder.use_text_conditioning=false` "
                        "for this mode."
                    )
        elif (
            config.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO
            and config.data.dataset_type == "lerobot_v2_latent_local"
        ):
            raise ValueError(
                "Method-4 `rgb_video` conditioning requires raw RGB to enter through the shared frontend/VAE path. "
                "Latent-local datasets enter from precomputed latents, so use `video_latent` conditioning there."
            )
        if not direct_train_mode and config.data.dataset_type in {"lerobot_v2", "libero_hdf5"}:
            raise ValueError(
                "Method-4 video-conditioned current-action decoding is currently aligned only for latent-local "
                "`standard_policy_window` style data. Raw LIBERO / raw LeRobot adapters anchor actions at the "
                "last observed frame, so apples-to-apples current-action method-4 runs need a deliberate raw-data "
                "alignment pass first. Use the latent-local method-4 configs or keep the explicit legacy "
                "method-4 decoders on raw LIBERO for now."
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
    if isinstance(policy_config, CausalVideoPredictionPolicyConfig):
        return CausalVideoPredictionPolicyVariant(
            config=policy_config,
            training_config=config.training,
            inference_config=config.inference,
        )
    if isinstance(policy_config, MoTPolicyConfig):
        return MoTPolicyVariant(
            config=policy_config,
            backbone_config=config.backbone,
            training_config=config.training,
            inference_config=config.inference,
            action_dim=action_schema.action_dim,
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
    mot_compat_decoder = (
        isinstance(config.policy_variant, MoTPolicyConfig)
        and decoder_config.name == ActionDecoderName.MLP
    )
    if decoder_config.name == ActionDecoderName.MLP:
        if mot_compat_decoder:
            return MoTActionDecoder(
                hidden_size=decoder_config.hidden_size,
                action_dim=decoder_config.action_dim,
                action_horizon=decoder_config.action_horizon,
                training_config=config.training,
                inference_config=config.inference,
                dropout=decoder_config.dropout,
            )
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
    if decoder_config.name == ActionDecoderName.VIDEO_CONDITIONED:
        return VideoConditionedActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            context_dim=decoder_config.context_dim,
            text_context_dim=decoder_config.text_context_dim,
            state_dim=decoder_config.state_dim,
            freq_dim=decoder_config.freq_dim,
            num_layers=decoder_config.num_layers,
            num_heads=decoder_config.num_heads,
            attention_head_dim=decoder_config.attention_head_dim,
            ffn_dim=decoder_config.ffn_dim,
            cross_attn_norm=decoder_config.cross_attn_norm,
            eps=decoder_config.eps,
            input_space=decoder_config.input_space,
            train_mode=decoder_config.train_mode,
            action_chunk_anchor_mode=decoder_config.action_chunk_anchor_mode,
            action_expert_init_mode=decoder_config.action_expert_init_mode,
            rollout_chunk_steps=decoder_config.rollout_chunk_steps,
            direct_latent_channels=decoder_config.direct_latent_channels,
            direct_rgb_patch_size=decoder_config.direct_rgb_patch_size,
            use_text_conditioning=decoder_config.use_text_conditioning,
            use_state_conditioning=decoder_config.use_state_conditioning,
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
    if decoder_config.name == ActionDecoderName.MOT:
        return MoTActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    if decoder_config.name == ActionDecoderName.VPP:
        return VPPSequenceActionDecoder(
            decoder_config,
            training_config=config.training,
            inference_config=config.inference,
            state_dim=config.data.action_schema.state_dim,
            observation_token_dim=config.backbone.hidden_size,
            goal_feature_dim=config.backbone.text_dim,
        )
    if decoder_config.name == ActionDecoderName.VIDEO_ONLY:
        return VideoOnlyActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    raise ValueError(f"Unsupported action decoder '{decoder_config.name}'.")


def build_variant_pipeline_from_config(config: ExperimentConfig) -> VariantPipeline:
    validate_experiment_config(config)
    policy_action_dim = _resolve_parallel_stream_model_action_dim(config)
    visual_tower = VisualTower(
        config.backbone,
        action_dim=policy_action_dim,
        state_dim=config.data.action_schema.state_dim,
    )
    policy_variant = build_policy_variant(config)
    action_decoder = build_action_decoder(config)
    if (
        config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
        and hasattr(action_decoder, "initialize_from_video_core")
    ):
        action_decoder.initialize_from_video_core(visual_tower.core)
    return VariantPipeline(
        visual_tower=visual_tower,
        policy_variant=policy_variant,
        action_decoder=action_decoder,
        preprocessor=build_canonical_video_preprocessor(config.data),
    )


def build_lingbot_exact_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return LingbotExactRunner(build_variant_pipeline_from_config(config))


def build_exact_runtime_runner_from_config(config: ExperimentConfig) -> LingbotExactRunner:
    return build_lingbot_exact_runner_from_config(config)
