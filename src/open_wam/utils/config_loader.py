from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from open_wam.configs import (
    ActionDecoderConfig,
    DecodedFeatureActionDecoderConfig,
    LingbotParallelActionDecoderConfig,
    MLPActionDecoderConfig,
    ParallelStreamPolicyConfig,
    PolicyVariantConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterActionDecoderConfig,
    RegisterAttachedPolicyConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    ExperimentConfig,
    GenericDataConfig,
    LiberoDataConfig,
    RobotWinDataConfig,
    TrainerConfig,
    ViewLayoutConfig,
)
import open_wam.configs.enums as config_enums
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, normalize_backbone_implementation

EnumT = TypeVar("EnumT", bound=StrEnum)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def _coerce_enum(enum_cls: type[EnumT], value: EnumT | str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _coerce_optional_enum(enum_cls: type[EnumT], value: EnumT | str | None) -> EnumT | None:
    if value is None:
        return None
    return _coerce_enum(enum_cls, value)


def _coerce_enum_tuple(enum_cls: type[EnumT], values: tuple[EnumT | str, ...] | list[EnumT | str]) -> tuple[EnumT, ...]:
    return tuple(_coerce_enum(enum_cls, value) for value in values)


def _load_policy_variant_config(
    policy_variant_raw: dict[str, Any],
    action_head_raw: dict[str, Any],
    data_config: GenericDataConfig | LiberoDataConfig | RobotWinDataConfig,
    backbone_config: SharedVideoTransformerConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> PolicyVariantConfig:
    resolved_raw = dict(policy_variant_raw)
    compatibility_mode = False
    if not resolved_raw:
        compatibility_mode = True
        resolved_raw = {
            "name": config_enums.PolicyVariantName.POST_LATENT,
            "hidden_size": action_head_raw.get("hidden_size", backbone_config.hidden_size),
            "attach_site": config_enums.AttachSite.POST_VISUAL_CORE,
            "pooling_mode": config_enums.PoolingMode.COMPAT_GLOBAL_MEAN,
            "use_state_projection": True,
            "compatibility_mode": True,
        }

    name = _coerce_enum(
        config_enums.PolicyVariantName,
        resolved_raw.get("name", config_enums.PolicyVariantName.POST_LATENT),
    )
    hidden_size = resolved_raw.get("hidden_size", backbone_config.hidden_size)
    if name == config_enums.PolicyVariantName.POST_LATENT:
        return PostLatentPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get(
                    "pooling_mode",
                    (
                        config_enums.PoolingMode.COMPAT_GLOBAL_MEAN
                        if compatibility_mode
                        else config_enums.PoolingMode.PER_FRAME_MEAN
                    ),
                ),
            ),
            query_count=resolved_raw.get("query_count", 0),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            compatibility_mode=resolved_raw.get("compatibility_mode", compatibility_mode),
        )
    if name == config_enums.PolicyVariantName.POST_DECODED:
        return PostDecodedPolicyConfig(
            hidden_size=hidden_size,
            decode_feature_mode=_coerce_enum(
                config_enums.DecodeFeatureMode,
                resolved_raw.get("decode_feature_mode", config_enums.DecodeFeatureMode.FRAME_TOKEN_SEQUENCE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get("pooling_mode", config_enums.PoolingMode.PER_FRAME_MEAN),
            ),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
        )
    if name == config_enums.PolicyVariantName.REGISTER_ATTACHED:
        return RegisterAttachedPolicyConfig(
            hidden_size=hidden_size,
            num_frame_per_block=resolved_raw.get("num_frame_per_block", 1),
            num_action_per_block=resolved_raw.get("num_action_per_block", 1),
            num_state_per_block=resolved_raw.get("num_state_per_block", 1),
            max_chunk_size=resolved_raw.get("max_chunk_size", inference_config.frame_chunk_size),
            register_layout=_coerce_enum(
                config_enums.RegisterLayout,
                resolved_raw.get("register_layout", config_enums.RegisterLayout.ACTION_THEN_STATE),
            ),
            mask_mode=_coerce_enum(
                config_enums.RegisterMaskMode,
                resolved_raw.get("mask_mode", config_enums.RegisterMaskMode.DREAMZERO_BLOCKWISE),
            ),
            use_state_encoder=resolved_raw.get("use_state_encoder", True),
            action_encoder_type=_coerce_enum(
                config_enums.StreamEncoderType,
                resolved_raw.get("action_encoder_type", config_enums.StreamEncoderType.MLP),
            ),
            state_encoder_type=_coerce_enum(
                config_enums.StreamEncoderType,
                resolved_raw.get("state_encoder_type", config_enums.StreamEncoderType.MLP),
            ),
            couple_action_to_video_blocks=resolved_raw.get("couple_action_to_video_blocks", True),
            structured_block_mode=_coerce_enum(
                config_enums.StructuredBlockMode,
                resolved_raw.get("structured_block_mode", config_enums.StructuredBlockMode.REGISTER_EXPLICIT),
            ),
            structured_time_layout=_coerce_enum(
                config_enums.StructuredTimeLayout,
                resolved_raw.get("structured_time_layout", config_enums.StructuredTimeLayout.VIDEO_ACTION_STATE),
            ),
            structured_frequency_mode=_coerce_enum(
                config_enums.StructuredFrequencyMode,
                resolved_raw.get("structured_frequency_mode", config_enums.StructuredFrequencyMode.STREAM_LOCAL),
            ),
            structured_teacher_forcing_layout=_coerce_enum(
                config_enums.StructuredTeacherForcingLayout,
                resolved_raw.get(
                    "structured_teacher_forcing_layout",
                    config_enums.StructuredTeacherForcingLayout.CLEAN_PREFIX,
                ),
            ),
            structured_attention_kernel=_coerce_enum(
                config_enums.StructuredAttentionKernel,
                resolved_raw.get(
                    "structured_attention_kernel",
                    config_enums.StructuredAttentionKernel.BRANCHWISE_EXPLICIT,
                ),
            ),
            structured_cache_kernel=_coerce_enum(
                config_enums.StructuredCacheKernel,
                resolved_raw.get(
                    "structured_cache_kernel",
                    config_enums.StructuredCacheKernel.BRANCHWISE_ROLLOUT_EXPLICIT,
                ),
            ),
            stream_input_adapter_family=_coerce_enum(
                config_enums.StreamInputAdapterFamily,
                resolved_raw.get(
                    "stream_input_adapter_family",
                    config_enums.StreamInputAdapterFamily.STRUCTURED_REGISTER_STREAMS,
                ),
            ),
            stream_output_head_family=_coerce_enum(
                config_enums.StreamOutputHeadFamily,
                resolved_raw.get(
                    "stream_output_head_family",
                    config_enums.StreamOutputHeadFamily.STRUCTURED_JOINT_FLOW,
                ),
            ),
        )
    if name == config_enums.PolicyVariantName.PARALLEL_STREAM:
        default_action_per_frame = max(
            1,
            data_config.action_schema.action_horizon // max(1, data_config.num_frames),
        )
        sequence_order = tuple(
            resolved_raw.get(
                "sequence_order",
                (
                    config_enums.ParallelSequenceComponent.VIDEO_NOISY,
                    config_enums.ParallelSequenceComponent.VIDEO_CONDITION,
                    config_enums.ParallelSequenceComponent.ACTION_NOISY,
                    config_enums.ParallelSequenceComponent.ACTION_CONDITION,
                ),
            )
        )
        return ParallelStreamPolicyConfig(
            hidden_size=hidden_size,
            runtime_mode=_coerce_enum(
                config_enums.ParallelRuntimeMode,
                resolved_raw.get("runtime_mode", config_enums.ParallelRuntimeMode.LINGBOT_EXACT),
            ),
            reference_profile=resolved_raw.get("reference_profile"),
            frame_chunk_size=resolved_raw.get("frame_chunk_size", inference_config.frame_chunk_size),
            action_per_frame=resolved_raw.get("action_per_frame", default_action_per_frame),
            attn_window=resolved_raw.get("attn_window", training_config.window_size),
            sequence_order=_coerce_enum_tuple(config_enums.ParallelSequenceComponent, sequence_order),
            mask_mode=_coerce_enum(
                config_enums.ParallelMaskMode,
                resolved_raw.get("mask_mode", config_enums.ParallelMaskMode.LINGBOT_CHUNKED),
            ),
            cache_mode=_coerce_enum(
                config_enums.ParallelCacheMode,
                resolved_raw.get("cache_mode", config_enums.ParallelCacheMode.METADATA_ONLY),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            used_action_channel_ids=tuple(resolved_raw.get("used_action_channel_ids", ())),
            inverse_used_action_channel_ids=tuple(resolved_raw.get("inverse_used_action_channel_ids", ())),
            action_norm_method=_coerce_enum(
                config_enums.ActionNormMethod,
                resolved_raw.get("action_norm_method", config_enums.ActionNormMethod.NONE),
            ),
            norm_q01=tuple(resolved_raw.get("norm_q01", ())),
            norm_q99=tuple(resolved_raw.get("norm_q99", ())),
        )
    raise ValueError(f"Unsupported policy variant '{name}'.")


def _load_action_decoder_config(
    action_decoder_raw: dict[str, Any],
    policy_variant_config: PolicyVariantConfig,
    data_config: GenericDataConfig | LiberoDataConfig | RobotWinDataConfig,
) -> ActionDecoderConfig:
    resolved_raw = dict(action_decoder_raw)
    if not resolved_raw:
        if policy_variant_config.name == config_enums.PolicyVariantName.REGISTER_ATTACHED:
            resolved_raw["name"] = config_enums.ActionDecoderName.REGISTER
        elif policy_variant_config.name == config_enums.PolicyVariantName.POST_DECODED:
            resolved_raw["name"] = config_enums.ActionDecoderName.DECODED_FEATURE
        elif (
            policy_variant_config.name == config_enums.PolicyVariantName.PARALLEL_STREAM
            and isinstance(policy_variant_config, ParallelStreamPolicyConfig)
            and policy_variant_config.runtime_mode == config_enums.ParallelRuntimeMode.LINGBOT_EXACT
        ):
            resolved_raw["name"] = config_enums.ActionDecoderName.LINGBOT_PARALLEL
        else:
            resolved_raw["name"] = config_enums.ActionDecoderName.MLP

    name = _coerce_enum(config_enums.ActionDecoderName, resolved_raw["name"])
    hidden_size = resolved_raw.get("hidden_size", policy_variant_config.hidden_size)
    action_dim = resolved_raw.get("action_dim", data_config.action_schema.action_dim)
    action_horizon = resolved_raw.get("action_horizon", data_config.action_schema.action_horizon)
    dropout = resolved_raw.get("dropout", 0.0)

    if name == config_enums.ActionDecoderName.MLP:
        return MLPActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.REGISTER:
        return RegisterActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.DECODED_FEATURE:
        return DecodedFeatureActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.LINGBOT_PARALLEL:
        return LingbotParallelActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported action decoder '{name}'.")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one root experiment YAML into the typed config boundary."""

    raw = _read_yaml(path)
    data_raw = raw.get("data", {})
    action_schema_raw = data_raw.get("action_schema", {})
    action_target_raw = data_raw.get("action_target", {})
    dataset_name = data_raw.get("dataset_name", "robotwin")
    dataset_type = data_raw.get("dataset_type")

    # Resolve defaults in two stages:
    # 1. known benchmark presets such as RobotWin and LIBERO
    # 2. a fully generic multiview fallback for custom sources
    #
    # This keeps new-source onboarding mostly declarative. A collaborator can
    # often add a new dataset config by specifying `dataset_type`, camera names,
    # layouts, and action/state schema in YAML without editing the loader.
    if dataset_name == "libero":
        if dataset_type == "libero_hdf5":
            data_defaults = LiberoDataConfig(dataset_type="libero_hdf5", repo_id=None)
        else:
            data_defaults = LiberoDataConfig()
        data_config_cls = LiberoDataConfig
    elif dataset_name == "robotwin":
        data_defaults = RobotWinDataConfig()
        data_config_cls = RobotWinDataConfig
    elif dataset_type == "lerobot_v2":
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type="lerobot_v2")
        data_config_cls = GenericDataConfig
    else:
        resolved_type = dataset_type or "synthetic_multiview"
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type=resolved_type)
        data_config_cls = GenericDataConfig

    default_view_layout = [
        {
            "source_name": view.source_name,
            "canonical_name": view.canonical_name,
            "top": view.top,
            "left": view.left,
            "height": view.height,
            "width": view.width,
        }
        for view in data_defaults.view_layout
    ]
    view_layout_raw = data_raw.get("view_layout", default_view_layout)
    view_layout = tuple(
        ViewLayoutConfig(
            source_name=view["source_name"],
            canonical_name=view.get("canonical_name", view["source_name"]),
            top=view["top"],
            left=view["left"],
            height=view["height"],
            width=view["width"],
        )
        for view in view_layout_raw
    )

    # `data_config_cls` may be a benchmark-specific preset or the generic
    # fallback. In both cases, the instantiated object carries the exact view
    # layout and action/state schema that the rest of the code should trust.
    data_config = data_config_cls(
        dataset_name=dataset_name,
        dataset_type=data_raw.get("dataset_type", data_defaults.dataset_type),
        repo_id=data_raw.get("repo_id", data_defaults.repo_id),
        local_root=data_raw.get("local_root", data_defaults.local_root),
        latent_root=data_raw.get("latent_root", data_defaults.latent_root),
        latent_subdir=data_raw.get("latent_subdir", data_defaults.latent_subdir),
        split=_coerce_enum(config_enums.DataSplit, data_raw.get("split", data_defaults.split)),
        cache_dir=data_raw.get("cache_dir", data_defaults.cache_dir),
        camera_names=tuple(data_raw.get("camera_names", data_defaults.camera_names)),
        latent_camera_names=tuple(data_raw.get("latent_camera_names", data_defaults.latent_camera_names)),
        canonical_height=data_raw.get("canonical_height", data_defaults.canonical_height),
        canonical_width=data_raw.get("canonical_width", data_defaults.canonical_width),
        view_layout=view_layout,
        num_frames=data_raw.get("num_frames", data_defaults.num_frames),
        frame_stride=data_raw.get("frame_stride", data_defaults.frame_stride),
        sample_stride=data_raw.get("sample_stride", data_defaults.sample_stride),
        episode_cache_size=data_raw.get("episode_cache_size", data_defaults.episode_cache_size),
        train_fraction=data_raw.get("train_fraction", data_defaults.train_fraction),
        split_seed=data_raw.get("split_seed", data_defaults.split_seed),
        max_train_episodes=data_raw.get("max_train_episodes", data_defaults.max_train_episodes),
        max_val_episodes=data_raw.get("max_val_episodes", data_defaults.max_val_episodes),
        train_batch_size=data_raw.get("train_batch_size", data_defaults.train_batch_size),
        val_batch_size=data_raw.get("val_batch_size", data_defaults.val_batch_size),
        num_workers=data_raw.get("num_workers", data_defaults.num_workers),
        action_schema=ActionSchemaConfig(
            action_dim=action_schema_raw.get("action_dim", data_defaults.action_schema.action_dim),
            action_horizon=action_schema_raw.get("action_horizon", data_defaults.action_schema.action_horizon),
            state_dim=action_schema_raw.get("state_dim", data_defaults.action_schema.state_dim),
            state_horizon=action_schema_raw.get("state_horizon", data_defaults.action_schema.state_horizon),
        ),
        action_target=ActionTargetConfig(
            representation=_coerce_enum(
                config_enums.ActionTargetRepresentation,
                action_target_raw.get("representation", data_defaults.action_target.representation),
            ),
            source_key=action_target_raw.get("source_key", data_defaults.action_target.source_key),
            pose_source_key=action_target_raw.get("pose_source_key", data_defaults.action_target.pose_source_key),
            state_encoding=_coerce_enum(
                config_enums.ActionTargetStateEncoding,
                action_target_raw.get("state_encoding", data_defaults.action_target.state_encoding),
            ),
            reference_source=_coerce_enum(
                config_enums.ActionTargetReferenceSource,
                action_target_raw.get("reference_source", data_defaults.action_target.reference_source),
            ),
            rotation_representation=_coerce_enum(
                config_enums.RotationRepresentation,
                action_target_raw.get("rotation_representation", data_defaults.action_target.rotation_representation),
            ),
            include_gripper=action_target_raw.get("include_gripper", data_defaults.action_target.include_gripper),
            gripper_representation=_coerce_enum(
                config_enums.GripperRepresentation,
                action_target_raw.get("gripper_representation", data_defaults.action_target.gripper_representation),
            ),
            gripper_action_index=action_target_raw.get(
                "gripper_action_index",
                data_defaults.action_target.gripper_action_index,
            ),
        ),
    )

    backbone_raw = raw.get("backbone", {})
    backbone_defaults = SharedVideoTransformerConfig()
    pretrained_model_name_or_path = backbone_raw.get("pretrained_model_name_or_path")
    load_wan_vae_frontend = backbone_raw.get("load_wan_vae_frontend")
    if load_wan_vae_frontend is None:
        load_wan_vae_frontend = pretrained_model_name_or_path is not None
    load_text_conditioning = backbone_raw.get("load_text_conditioning")
    if load_text_conditioning is None:
        load_text_conditioning = pretrained_model_name_or_path is not None
    load_reference_core_weights = backbone_raw.get("load_reference_core_weights")
    if load_reference_core_weights is None:
        load_reference_core_weights = False

    backbone_config = SharedVideoTransformerConfig(
        input_channels=backbone_raw.get("input_channels", backbone_defaults.input_channels),
        latent_channels=backbone_raw.get("latent_channels", backbone_defaults.latent_channels),
        latent_stride=backbone_raw.get("latent_stride", backbone_defaults.latent_stride),
        patch_size_t=backbone_raw.get("patch_size_t", backbone_defaults.patch_size_t),
        patch_size_h=backbone_raw.get("patch_size_h", backbone_defaults.patch_size_h),
        patch_size_w=backbone_raw.get("patch_size_w", backbone_defaults.patch_size_w),
        implementation=normalize_backbone_implementation(
            backbone_raw.get("implementation", backbone_defaults.implementation)
        ),
        hidden_size=backbone_raw.get("hidden_size", backbone_defaults.hidden_size),
        num_layers=backbone_raw.get("num_layers", backbone_defaults.num_layers),
        num_heads=backbone_raw.get("num_heads", backbone_defaults.num_heads),
        attention_head_dim=backbone_raw.get("attention_head_dim"),
        mlp_ratio=backbone_raw.get("mlp_ratio", backbone_defaults.mlp_ratio),
        ffn_dim=backbone_raw.get("ffn_dim"),
        text_dim=backbone_raw.get("text_dim", backbone_defaults.text_dim),
        freq_dim=backbone_raw.get("freq_dim", backbone_defaults.freq_dim),
        cross_attn_norm=backbone_raw.get("cross_attn_norm", backbone_defaults.cross_attn_norm),
        rope_max_seq_len=backbone_raw.get("rope_max_seq_len", backbone_defaults.rope_max_seq_len),
        latent_norm_eps=backbone_raw.get("latent_norm_eps", backbone_defaults.latent_norm_eps),
        attn_mode=_coerce_enum(config_enums.AttentionMode, backbone_raw.get("attn_mode", backbone_defaults.attn_mode)),
        train_attn_mode=_coerce_optional_enum(
            config_enums.AttentionMode,
            backbone_raw.get("train_attn_mode", backbone_defaults.train_attn_mode),
        ),
        infer_attn_mode=_coerce_optional_enum(
            config_enums.AttentionMode,
            backbone_raw.get("infer_attn_mode", backbone_defaults.infer_attn_mode),
        ),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer_subdir=backbone_raw.get("transformer_subdir", backbone_defaults.transformer_subdir),
        vae_subdir=backbone_raw.get("vae_subdir", backbone_defaults.vae_subdir),
        text_encoder_subdir=backbone_raw.get("text_encoder_subdir", backbone_defaults.text_encoder_subdir),
        tokenizer_subdir=backbone_raw.get("tokenizer_subdir", backbone_defaults.tokenizer_subdir),
        max_text_tokens=backbone_raw.get("max_text_tokens", backbone_defaults.max_text_tokens),
        load_wan_vae_frontend=load_wan_vae_frontend,
        load_text_conditioning=load_text_conditioning,
        load_reference_core_weights=load_reference_core_weights,
        reference_assets_device_policy=_coerce_enum(
            config_enums.ReferenceAssetsDevicePolicy,
            backbone_raw.get(
                "reference_assets_device_policy",
                backbone_defaults.reference_assets_device_policy,
            ),
        ),
        reference_model_path=backbone_raw.get("reference_model_path"),
    )

    training_raw = raw.get("training", {})
    training_config = TrainingConfig(
        video_num_train_timesteps=training_raw.get("video_num_train_timesteps", 1000),
        action_num_train_timesteps=training_raw.get("action_num_train_timesteps", 1000),
        video_sigma_shift=training_raw.get("video_sigma_shift", 5.0),
        action_sigma_shift=training_raw.get("action_sigma_shift", 1.0),
        use_teacher_forcing=training_raw.get("use_teacher_forcing", False),
        chunk_size=training_raw.get("chunk_size", 2),
        window_size=training_raw.get("window_size", 8),
        optimizer_name=_coerce_enum(
            config_enums.OptimizerName,
            training_raw.get("optimizer_name", "adamw"),
        ),
        scheduler_name=_coerce_enum(
            config_enums.SchedulerName,
            training_raw.get("scheduler_name", "constant"),
        ),
        learning_rate=training_raw.get("learning_rate", 1e-4),
        beta1=training_raw.get("beta1", 0.9),
        beta2=training_raw.get("beta2", 0.999),
        weight_decay=training_raw.get("weight_decay", 0.0),
        warmup_steps=training_raw.get("warmup_steps", 0),
        gradient_accumulation_steps=training_raw.get("gradient_accumulation_steps", 1),
        max_grad_norm=training_raw.get("max_grad_norm"),
        num_steps=training_raw.get("num_steps"),
        text_condition_dropout_prob=training_raw.get("text_condition_dropout_prob", 0.0),
        enabled_objectives=tuple(training_raw.get("enabled_objectives", ("action", "latent"))),
        latent_loss_weight=training_raw.get("latent_loss_weight", 1.0),
        action_loss_weight=training_raw.get("action_loss_weight", 1.0),
        trainable_components=tuple(training_raw.get("trainable_components", ("all",))),
        frozen_components=tuple(training_raw.get("frozen_components", ())),
    )

    inference_raw = raw.get("inference", {})
    joint_cfg_application = _coerce_optional_enum(
        config_enums.JointCfgApplication,
        inference_raw.get("joint_cfg_application"),
    )
    if joint_cfg_application == config_enums.JointCfgApplication.VIDEO_ONLY:
        video_cfg_mode = config_enums.CFGMode.GUIDED
        action_cfg_mode = config_enums.CFGMode.CONDITIONED
    elif joint_cfg_application == config_enums.JointCfgApplication.JOINT:
        video_cfg_mode = config_enums.CFGMode.GUIDED
        action_cfg_mode = config_enums.CFGMode.GUIDED
    else:
        video_cfg_mode = _coerce_enum(
            config_enums.CFGMode,
            inference_raw.get("video_cfg_mode", "guided"),
        )
        action_cfg_mode = _coerce_enum(
            config_enums.CFGMode,
            inference_raw.get("action_cfg_mode", "conditioned"),
        )

    joint_cache_warmup_source = inference_raw.get("joint_cache_warmup_source")
    if joint_cache_warmup_source == "dreamzero_reference_block":
        resolved_warmup_source = config_enums.CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source == "reference_video":
        resolved_warmup_source = config_enums.CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "full"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames")
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "full"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source in {None, "none"}:
        resolved_warmup_source = (
            config_enums.CacheWarmupSource.REFERENCE_VIDEO
            if joint_cache_warmup_source is None
            else config_enums.CacheWarmupSource.NONE
        )
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    else:
        resolved_warmup_source = _coerce_enum(config_enums.CacheWarmupSource, joint_cache_warmup_source)
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")

    inference_config = InferenceConfig(
        video_num_inference_steps=inference_raw.get("video_num_inference_steps", 25),
        action_num_inference_steps=inference_raw.get("action_num_inference_steps", 50),
        joint_num_inference_steps=inference_raw.get("joint_num_inference_steps"),
        joint_sampler=_coerce_enum(
            config_enums.JointSampler,
            inference_raw.get("joint_sampler", "unipc"),
        ),
        video_cfg_mode=video_cfg_mode,
        action_cfg_mode=action_cfg_mode,
        joint_cfg_application=joint_cfg_application,
        joint_cache_update_mode=_coerce_enum(
            config_enums.CacheUpdateMode,
            inference_raw.get("joint_cache_update_mode", "warmup_only"),
        ),
        joint_cache_warmup_source=resolved_warmup_source,
        joint_cache_initial_warmup_anchor=initial_warmup_anchor,
        joint_cache_initial_warmup_frames=initial_warmup_frames,
        joint_cache_rollout_warmup_anchor=rollout_warmup_anchor,
        joint_cache_rollout_warmup_frames=rollout_warmup_frames,
        joint_observed_video_prefix_frames=inference_raw.get("joint_observed_video_prefix_frames", 1),
        frame_chunk_size=inference_raw.get("frame_chunk_size", 2),
        use_cache=inference_raw.get("use_cache", True),
        guidance_scale=inference_raw.get("guidance_scale", 1.0),
        action_guidance_scale=inference_raw.get("action_guidance_scale", 1.0),
        video_exec_step=inference_raw.get("video_exec_step", -1),
    )

    # Legacy configs may still provide an `action_head` block. The current
    # runtime no longer instantiates a separate head stack, but we still read
    # those fields here to derive the equivalent variant/decoder defaults.
    action_head_raw = raw.get("action_head", {})
    policy_variant_config = _load_policy_variant_config(
        policy_variant_raw=raw.get("policy_variant", {}),
        action_head_raw=action_head_raw,
        data_config=data_config,
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    action_decoder_config = _load_action_decoder_config(
        action_decoder_raw=raw.get("action_decoder", {}),
        policy_variant_config=policy_variant_config,
        data_config=data_config,
    )

    trainer_raw = raw.get("trainer", {})
    trainer_config = TrainerConfig(
        max_epochs=trainer_raw.get("max_epochs", 1),
        limit_train_batches=trainer_raw.get("limit_train_batches", 2),
        limit_val_batches=trainer_raw.get("limit_val_batches", 1),
        log_every_n_steps=trainer_raw.get("log_every_n_steps", 1),
        accelerator=_coerce_enum(
            config_enums.TrainerAccelerator,
            trainer_raw.get("accelerator", "cpu"),
        ),
        devices=trainer_raw.get("devices", 1),
        precision=_coerce_enum(
            config_enums.TrainerPrecision,
            trainer_raw.get("precision", "32-true"),
        ),
        enable_checkpointing=trainer_raw.get("enable_checkpointing", False),
        enable_model_summary=trainer_raw.get("enable_model_summary", False),
        runtime=_coerce_enum(
            config_enums.TrainerRuntimeName,
            trainer_raw.get("runtime", "lightning"),
        ),
        batch_adapter=_coerce_enum(
            config_enums.BatchAdapterName,
            trainer_raw.get("batch_adapter", "views"),
        ),
        loop_policy=_coerce_enum(
            config_enums.LoopPolicyName,
            trainer_raw.get("loop_policy", "epochs"),
        ),
        strategy=_coerce_enum(
            config_enums.StrategyName,
            trainer_raw.get("strategy", "lightning"),
        ),
        default_root_dir=trainer_raw.get("default_root_dir"),
        checkpoint_dir=trainer_raw.get("checkpoint_dir"),
        save_interval=trainer_raw.get("save_interval"),
        checkpoint_mode=_coerce_enum(
            config_enums.CheckpointMode,
            trainer_raw.get("checkpoint_mode", "full_training_state"),
        ),
        export_runtime_backbone=trainer_raw.get("export_runtime_backbone", False),
        resume_from=trainer_raw.get("resume_from"),
        enable_jsonl_logging=trainer_raw.get("enable_jsonl_logging", False),
        metrics_filename=trainer_raw.get("metrics_filename", "metrics.jsonl"),
        enable_wandb=trainer_raw.get("enable_wandb", False),
        wandb_project=trainer_raw.get("wandb_project"),
        wandb_entity=trainer_raw.get("wandb_entity"),
        wandb_mode=_coerce_enum(
            config_enums.WandBMode,
            trainer_raw.get("wandb_mode", "disabled"),
        ),
        run_name=trainer_raw.get("run_name"),
    )

    return ExperimentConfig(
        name=raw.get("name", "unnamed_experiment"),
        data=data_config,
        backbone=backbone_config,
        policy_variant=policy_variant_config,
        action_decoder=action_decoder_config,
        training=training_config,
        inference=inference_config,
        trainer=trainer_config,
    )
