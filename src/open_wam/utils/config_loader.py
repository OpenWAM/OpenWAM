from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import ActionHeadConfig, ActionSchemaConfig, ExperimentConfig, RobotWinDataConfig, TrainerConfig
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
    data_config = RobotWinDataConfig(
        dataset_name=data_raw.get("dataset_name", "robotwin"),
        camera_names=tuple(data_raw.get("camera_names", ("cam_high", "cam_left_wrist", "cam_right_wrist"))),
        canonical_height=data_raw.get("canonical_height", 384),
        canonical_width=data_raw.get("canonical_width", 320),
        num_frames=data_raw.get("num_frames", 2),
        action_schema=ActionSchemaConfig(
            action_dim=action_schema_raw.get("action_dim", 30),
            action_horizon=action_schema_raw.get("action_horizon", 32),
            state_dim=action_schema_raw.get("state_dim", 30),
            state_horizon=action_schema_raw.get("state_horizon", 1),
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

