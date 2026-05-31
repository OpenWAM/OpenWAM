from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    ActionNormalizationMode,
    BatchAdapterName,
    BackboneImplementation,
    CausalVideoPredictionPolicyConfig,
    ExperimentConfig,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    ProprioContextMode,
    RegisterAttachedPolicyConfig,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
    VideoSequencePolicyConfig,
)
from open_wam.data import build_canonical_video_preprocessor
from open_wam.data.action_mapping import build_action_sampler_mask, validate_action_mapping_preflight
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
from open_wam.models.policy_variants.register_attached.deprecation import (
    raise_register_attached_obsolete,
)
from open_wam.models.visual_tower import VisualTower
from open_wam.models.video_backbone import normalize_backbone_implementation

from .registries import ACTION_DECODER_BUILDERS, POLICY_VARIANT_BUILDERS
from .variant_pipeline import VariantPipeline
from .lingbot_exact import LingbotExactRunner


_PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES = {
    ParallelRuntimeMode.LINGBOT_EXACT,
    ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
    ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
}


def _resolve_parallel_stream_model_action_dim(config: ExperimentConfig) -> int:
    if (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.runtime_mode in _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES
    ):
        return config.action_decoder.action_dim
    return config.data.action_schema.action_dim


def _resolve_proprio_context_state_dim(config: ExperimentConfig) -> int | None:
    if not isinstance(config.policy_variant, (MoTPolicyConfig, ParallelStreamPolicyConfig)):
        return None
    mode = ProprioContextMode(config.policy_variant.proprio_context_mode)
    if mode != ProprioContextMode.TEXT_CONTEXT_TOKEN:
        return None
    state_dim = int(config.data.action_schema.state_dim)
    if state_dim <= 0:
        raise ValueError(
            "Deprecated proprio_context_mode=text_context_token requires positive data.action_schema.state_dim."
        )
    return state_dim


def _resolve_proprio_hidden_context_state_dim(config: ExperimentConfig) -> int | None:
    if not isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        return None
    mode = ProprioContextMode(config.policy_variant.proprio_context_mode)
    if mode != ProprioContextMode.PER_CHUNK_ADDITIVE:
        return None
    state_dim = int(config.data.action_schema.state_dim)
    if state_dim <= 0:
        raise ValueError("proprio_context_mode=per_chunk_additive requires positive data.action_schema.state_dim.")
    return state_dim


def validate_experiment_config(config: ExperimentConfig) -> None:
    action_schema = config.data.action_schema
    if isinstance(config.policy_variant, RegisterAttachedPolicyConfig):
        raise_register_attached_obsolete(stacklevel=3)
    validate_action_mapping_preflight(
        config.data.action_mapping,
        action_schema_dim=action_schema.action_dim,
    )
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
        if config.policy_variant.runtime_mode not in _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES:
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
        if (
            config.policy_variant.runtime_mode
            in {MoTRuntimeMode.JOINT_DENOISE, MoTRuntimeMode.NON_JOINT_TWO_STREAM}
            and config.policy_variant.video_prefix_frames >= config.data.num_frames
        ):
            raise ValueError(
                "MoT two-stream method 5 requires `video_prefix_frames < data.num_frames`, "
                f"got video_prefix_frames={config.policy_variant.video_prefix_frames}, "
                f"data.num_frames={config.data.num_frames}, "
                f"runtime_mode={config.policy_variant.runtime_mode!r}."
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


def _build_video_sequence_policy_variant(config: ExperimentConfig):
    action_schema = config.data.action_schema
    policy_config = config.policy_variant
    assert isinstance(policy_config, VideoSequencePolicyConfig)
    return VideoSequencePolicyVariant(
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


def _build_register_attached_policy_variant(config: ExperimentConfig):
    policy_config = config.policy_variant
    assert isinstance(policy_config, RegisterAttachedPolicyConfig)
    raise_register_attached_obsolete(stacklevel=3)


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


def build_policy_variant(config: ExperimentConfig):
    builder = POLICY_VARIANT_BUILDERS.get(type(config.policy_variant))
    if builder is None:
        raise ValueError(f"Unsupported policy variant config '{type(config.policy_variant).__name__}'.")
    return builder(config)


def _build_mlp_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    mot_compat_decoder = (
        isinstance(config.policy_variant, MoTPolicyConfig)
        and decoder_config.name == ActionDecoderName.MLP
    )
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


def _build_register_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return RegisterActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_decoded_feature_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return DecodedFeatureActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_video_conditioned_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
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


def _build_lingbot_parallel_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    source_action_channel_ids: tuple[int, ...] = ()
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        adapter_spec = build_action_adapter_spec(
            config.policy_variant,
            model_action_dim=decoder_config.action_dim,
        )
        if adapter_spec is not None:
            source_action_channel_ids = adapter_spec.used_action_channel_ids
    action_normalization = config.data.action_target.normalization
    source_action_mean: tuple[float, ...] = ()
    source_action_std: tuple[float, ...] = ()
    if action_normalization.mode == ActionNormalizationMode.GAUSSIAN:
        source_action_mean = action_normalization.mean
        source_action_std = action_normalization.std
    return LingbotParallelActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        dropout=decoder_config.dropout,
        recovered_osc_loss_weight=decoder_config.recovered_osc_loss_weight,
        recovered_osc_position_scale=decoder_config.recovered_osc_position_scale,
        recovered_osc_rotation_scale=decoder_config.recovered_osc_rotation_scale,
        source_action_channel_ids=source_action_channel_ids,
        source_action_mean=source_action_mean,
        source_action_std=source_action_std,
    )


def _build_mot_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return MoTActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_vpp_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return VPPSequenceActionDecoder(
        decoder_config,
        training_config=config.training,
        inference_config=config.inference,
        state_dim=config.data.action_schema.state_dim,
        observation_token_dim=config.backbone.hidden_size,
        goal_feature_dim=config.backbone.text_dim,
    )


def _build_video_only_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return VideoOnlyActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def build_action_decoder(config: ExperimentConfig):
    builder = ACTION_DECODER_BUILDERS.get(config.action_decoder.name)
    if builder is None:
        raise ValueError(f"Unsupported action decoder '{config.action_decoder.name}'.")
    return builder(config)


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
        VideoSequencePolicyConfig,
        _build_video_sequence_policy_variant,
        description="Sequence-native video-policy policy variant.",
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
        RegisterAttachedPolicyConfig,
        _build_register_attached_policy_variant,
        description="OBSOLETE traditional Method 2 register-attached policy variant.",
        replace=True,
    )
    POLICY_VARIANT_BUILDERS.register(
        ParallelStreamPolicyConfig,
        _build_parallel_stream_policy_variant,
        description="Parallel-stream LingBot-compatible policy variant.",
        replace=True,
    )

    ACTION_DECODER_BUILDERS.register(ActionDecoderName.MLP, _build_mlp_action_decoder, replace=True)
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.REGISTER, _build_register_action_decoder, replace=True)
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
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.VPP, _build_vpp_action_decoder, replace=True)
    ACTION_DECODER_BUILDERS.register(ActionDecoderName.VIDEO_ONLY, _build_video_only_action_decoder, replace=True)


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
