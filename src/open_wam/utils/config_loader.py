from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import (
    ActionDecoderConfig,
    DecodedFeatureActionDecoderConfig,
    ActionHeadConfig,
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
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def _load_policy_variant_config(
    policy_variant_raw: dict[str, Any],
    action_head_raw: dict[str, Any],
    data_config: GenericDataConfig | LiberoDataConfig | RobotWinDataConfig,
    backbone_config: LingbotCompatibleVideoBackboneConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> PolicyVariantConfig:
    resolved_raw = dict(policy_variant_raw)
    compatibility_mode = False
    if not resolved_raw:
        compatibility_mode = True
        resolved_raw = {
            "name": "post_latent",
            "hidden_size": action_head_raw.get("hidden_size", backbone_config.hidden_size),
            "attach_site": "post_visual_core",
            "pooling_mode": "compat_global_mean",
            "use_state_projection": True,
            "compatibility_mode": True,
        }

    name = resolved_raw.get("name", "post_latent")
    hidden_size = resolved_raw.get("hidden_size", backbone_config.hidden_size)
    if name == "post_latent":
        return PostLatentPolicyConfig(
            hidden_size=hidden_size,
            attach_site=resolved_raw.get("attach_site", "post_visual_core"),
            pooling_mode=resolved_raw.get(
                "pooling_mode",
                "compat_global_mean" if compatibility_mode else "per_frame_mean",
            ),
            query_count=resolved_raw.get("query_count", 0),
            temporal_projection=resolved_raw.get("temporal_projection", "interpolate"),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            compatibility_mode=resolved_raw.get("compatibility_mode", compatibility_mode),
        )
    if name == "post_decoded":
        return PostDecodedPolicyConfig(
            hidden_size=hidden_size,
            decode_feature_mode=resolved_raw.get("decode_feature_mode", "frame_token_sequence"),
            pooling_mode=resolved_raw.get("pooling_mode", "per_frame_mean"),
            temporal_projection=resolved_raw.get("temporal_projection", "interpolate"),
            use_state_projection=resolved_raw.get("use_state_projection", True),
        )
    if name == "register_attached":
        return RegisterAttachedPolicyConfig(
            hidden_size=hidden_size,
            num_frame_per_block=resolved_raw.get("num_frame_per_block", 1),
            num_action_per_block=resolved_raw.get("num_action_per_block", 1),
            num_state_per_block=resolved_raw.get("num_state_per_block", 1),
            max_chunk_size=resolved_raw.get("max_chunk_size", inference_config.frame_chunk_size),
            register_layout=resolved_raw.get("register_layout", "action_then_state"),
            mask_mode=resolved_raw.get("mask_mode", "dreamzero_blockwise"),
            use_state_encoder=resolved_raw.get("use_state_encoder", True),
            action_encoder_type=resolved_raw.get("action_encoder_type", "mlp"),
            state_encoder_type=resolved_raw.get("state_encoder_type", "mlp"),
        )
    if name == "parallel_stream":
        default_action_per_frame = max(
            1,
            data_config.action_schema.action_horizon // max(1, data_config.num_frames),
        )
        sequence_order = tuple(
            resolved_raw.get(
                "sequence_order",
                ("video_noisy", "video_condition", "action_noisy", "action_condition"),
            )
        )
        return ParallelStreamPolicyConfig(
            hidden_size=hidden_size,
            runtime_mode=resolved_raw.get("runtime_mode", "approx"),
            reference_profile=resolved_raw.get("reference_profile"),
            frame_chunk_size=resolved_raw.get("frame_chunk_size", inference_config.frame_chunk_size),
            action_per_frame=resolved_raw.get("action_per_frame", default_action_per_frame),
            attn_window=resolved_raw.get("attn_window", training_config.window_size),
            sequence_order=sequence_order,
            mask_mode=resolved_raw.get("mask_mode", "lingbot_chunked"),
            cache_mode=resolved_raw.get("cache_mode", "metadata_only"),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            used_action_channel_ids=tuple(resolved_raw.get("used_action_channel_ids", ())),
            inverse_used_action_channel_ids=tuple(resolved_raw.get("inverse_used_action_channel_ids", ())),
            action_norm_method=resolved_raw.get("action_norm_method", "none"),
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
        if policy_variant_config.name == "register_attached":
            resolved_raw["name"] = "register_decoder"
        elif policy_variant_config.name == "post_decoded":
            resolved_raw["name"] = "decoded_feature_decoder"
        elif (
            policy_variant_config.name == "parallel_stream"
            and isinstance(policy_variant_config, ParallelStreamPolicyConfig)
            and policy_variant_config.runtime_mode == "lingbot_exact"
        ):
            resolved_raw["name"] = "lingbot_parallel_decoder"
        else:
            resolved_raw["name"] = "mlp_decoder"

    name = resolved_raw["name"]
    hidden_size = resolved_raw.get("hidden_size", policy_variant_config.hidden_size)
    action_dim = resolved_raw.get("action_dim", data_config.action_schema.action_dim)
    action_horizon = resolved_raw.get("action_horizon", data_config.action_schema.action_horizon)
    dropout = resolved_raw.get("dropout", 0.0)

    if name == "mlp_decoder":
        return MLPActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == "register_decoder":
        return RegisterActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == "decoded_feature_decoder":
        return DecodedFeatureActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == "lingbot_parallel_decoder":
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
        split=data_raw.get("split", data_defaults.split),
        cache_dir=data_raw.get("cache_dir", data_defaults.cache_dir),
        camera_names=tuple(data_raw.get("camera_names", data_defaults.camera_names)),
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
            representation=action_target_raw.get("representation", data_defaults.action_target.representation),
            source_key=action_target_raw.get("source_key", data_defaults.action_target.source_key),
            pose_source_key=action_target_raw.get("pose_source_key", data_defaults.action_target.pose_source_key),
            state_encoding=action_target_raw.get("state_encoding", data_defaults.action_target.state_encoding),
            reference_source=action_target_raw.get("reference_source", data_defaults.action_target.reference_source),
            rotation_representation=action_target_raw.get(
                "rotation_representation",
                data_defaults.action_target.rotation_representation,
            ),
            include_gripper=action_target_raw.get("include_gripper", data_defaults.action_target.include_gripper),
            gripper_representation=action_target_raw.get(
                "gripper_representation",
                data_defaults.action_target.gripper_representation,
            ),
            gripper_action_index=action_target_raw.get(
                "gripper_action_index",
                data_defaults.action_target.gripper_action_index,
            ),
        ),
    )

    backbone_raw = raw.get("backbone", {})
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        input_channels=backbone_raw.get("input_channels", 3),
        latent_channels=backbone_raw.get("latent_channels", 48),
        latent_stride=backbone_raw.get("latent_stride", 16),
        patch_size_t=backbone_raw.get("patch_size_t", 1),
        patch_size_h=backbone_raw.get("patch_size_h", 2),
        patch_size_w=backbone_raw.get("patch_size_w", 2),
        implementation=backbone_raw.get("implementation", "dummy"),
        hidden_size=backbone_raw.get("hidden_size", 3072),
        num_layers=backbone_raw.get("num_layers", 0),
        num_heads=backbone_raw.get("num_heads", 8),
        attention_head_dim=backbone_raw.get("attention_head_dim"),
        mlp_ratio=backbone_raw.get("mlp_ratio", 4),
        ffn_dim=backbone_raw.get("ffn_dim"),
        text_dim=backbone_raw.get("text_dim", 4096),
        freq_dim=backbone_raw.get("freq_dim", 256),
        cross_attn_norm=backbone_raw.get("cross_attn_norm", True),
        rope_max_seq_len=backbone_raw.get("rope_max_seq_len", 1024),
        latent_norm_eps=backbone_raw.get("latent_norm_eps", 1e-6),
        attn_mode=backbone_raw.get("attn_mode", "torch"),
        pretrained_model_name_or_path=backbone_raw.get("pretrained_model_name_or_path"),
        transformer_subdir=backbone_raw.get("transformer_subdir", "transformer"),
        vae_subdir=backbone_raw.get("vae_subdir", "vae"),
        text_encoder_subdir=backbone_raw.get("text_encoder_subdir", "text_encoder"),
        tokenizer_subdir=backbone_raw.get("tokenizer_subdir", "tokenizer"),
        max_text_tokens=backbone_raw.get("max_text_tokens", 512),
        load_wan_vae_frontend=backbone_raw.get("load_wan_vae_frontend", False),
        load_text_conditioning=backbone_raw.get("load_text_conditioning", False),
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
    )

    inference_raw = raw.get("inference", {})
    inference_config = InferenceConfig(
        video_num_inference_steps=inference_raw.get("video_num_inference_steps", 25),
        action_num_inference_steps=inference_raw.get("action_num_inference_steps", 50),
        frame_chunk_size=inference_raw.get("frame_chunk_size", 2),
        use_cache=inference_raw.get("use_cache", True),
        guidance_scale=inference_raw.get("guidance_scale", 1.0),
        action_guidance_scale=inference_raw.get("action_guidance_scale", 1.0),
        video_exec_step=inference_raw.get("video_exec_step", -1),
    )

    action_head_raw = raw.get("action_head", {})
    action_head_config = ActionHeadConfig(
        name=action_head_raw.get("name", "contract_only"),
        hidden_size=action_head_raw.get("hidden_size", backbone_config.hidden_size),
        action_dim=action_head_raw.get("action_dim", data_config.action_schema.action_dim),
        action_horizon=action_head_raw.get("action_horizon", data_config.action_schema.action_horizon),
        state_dim=action_head_raw.get("state_dim", data_config.action_schema.state_dim),
    )
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
        accelerator=trainer_raw.get("accelerator", "cpu"),
        devices=trainer_raw.get("devices", 1),
        precision=trainer_raw.get("precision", "32-true"),
        enable_checkpointing=trainer_raw.get("enable_checkpointing", False),
        enable_model_summary=trainer_raw.get("enable_model_summary", False),
    )

    return ExperimentConfig(
        name=raw.get("name", "unnamed_experiment"),
        data=data_config,
        backbone=backbone_config,
        policy_variant=policy_variant_config,
        action_decoder=action_decoder_config,
        action_head=action_head_config,
        training=training_config,
        inference=inference_config,
        trainer=trainer_config,
    )
