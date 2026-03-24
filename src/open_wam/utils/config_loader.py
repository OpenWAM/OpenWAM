from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import (
    ActionHeadConfig,
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
        hidden_size=backbone_raw.get("hidden_size", 3072),
        num_layers=backbone_raw.get("num_layers", 0),
        num_heads=backbone_raw.get("num_heads", 8),
        mlp_ratio=backbone_raw.get("mlp_ratio", 4),
        latent_norm_eps=backbone_raw.get("latent_norm_eps", 1e-6),
    )

    action_head_raw = raw.get("action_head", {})
    action_head_config = ActionHeadConfig(
        name=action_head_raw.get("name", "contract_only"),
        hidden_size=action_head_raw.get("hidden_size", backbone_config.hidden_size),
        action_dim=action_head_raw.get("action_dim", data_config.action_schema.action_dim),
        action_horizon=action_head_raw.get("action_horizon", data_config.action_schema.action_horizon),
        state_dim=action_head_raw.get("state_dim", data_config.action_schema.state_dim),
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
        action_head=action_head_config,
        training=training_config,
        inference=inference_config,
        trainer=trainer_config,
    )
