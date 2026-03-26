from __future__ import annotations

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ExperimentConfig,
    InferenceConfig,
    LingbotParallelActionDecoderConfig,
    LiberoDataConfig,
    MLPActionDecoderConfig,
    ParallelStreamPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.pipelines import build_variant_pipeline_from_config


def test_exact_parallel_stream_uses_vendored_reference_model_by_default() -> None:
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
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.policy_variant.config.runtime_mode == "lingbot_exact"


def test_reference_core_weight_loading_uses_vendored_reference_model_by_default() -> None:
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

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.visual_tower.reference_core_load_report is not None


def test_exact_parallel_stream_uses_decoder_action_dim_when_dataset_stays_raw() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=16, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            max_text_tokens=512,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=30, action_horizon=16),
        training=TrainingConfig(chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0),
        inference=InferenceConfig(
            frame_chunk_size=4,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            video_exec_step=-1,
        ),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.visual_tower.action_dim == 30
    assert pipeline.policy_variant.action_dim == 30
    assert pipeline.policy_variant.exact_action_adapter.spec is not None
    assert pipeline.policy_variant.exact_action_adapter.spec.raw_action_dim == 7


def test_exact_parallel_stream_reference_profile_rejects_mismatched_text_length() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=16, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            max_text_tokens=226,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=30, action_horizon=16),
        training=TrainingConfig(chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0),
        inference=InferenceConfig(
            frame_chunk_size=4,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            video_exec_step=-1,
        ),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "max_text_tokens" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected LingBot exact profile validation to reject mismatched max_text_tokens.")
