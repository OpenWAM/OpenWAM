from __future__ import annotations

import pytest

from open_wam.configs import (
    ActionSchemaConfig,
    ExperimentConfig,
    InferenceConfig,
    MLPActionDecoderConfig,
    ParallelStreamPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.pipelines import build_variant_pipeline_from_config


def test_exact_parallel_stream_requires_reference_model_path() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="lingbot_replica"),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    with pytest.raises(ValueError, match="reference_model_path"):
        build_variant_pipeline_from_config(config)


def test_reference_core_weight_loading_requires_reference_model_path() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            implementation="lingbot_replica",
            load_reference_core_weights=True,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="approx",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    with pytest.raises(ValueError, match="reference_model_path"):
        build_variant_pipeline_from_config(config)
