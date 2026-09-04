from __future__ import annotations

import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ExperimentConfig,
    InferenceConfig,
    LiberoDataConfig,
    ParallelStreamActionDecoderConfig,
    ParallelStreamPolicyConfig,
    TrainingConfig,
    VideoActionProgram,
)
from open_wam.data import build_synthetic_batch
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.pipelines import build_variant_pipeline_from_config


def test_exact_libero_train_inputs_expand_raw_actions_to_model_space() -> None:
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
            text_dim=16,
            freq_dim=8,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(hidden_size=32, action_dim=30, action_horizon=16),
        training=TrainingConfig(chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0),
        inference=InferenceConfig(
            frame_chunk_size=4,
            attention_window_size=30,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            video_exec_step=-1,
        ),
    )

    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
    )

    visual_outputs = pipeline.prepare_visual_outputs(batch.views, task_text=batch.task_text)
    prepared_inputs = pipeline.policy_variant.prepare_train_inputs(visual_outputs, train_batch)
    train_artifacts = prepared_inputs.variant_inputs["parallel_train_artifacts"]
    action_targets = train_artifacts.input_dict["action_dict"]["targets"]
    action_mask = train_artifacts.input_dict["action_dict"]["actions_mask"]

    assert batch.actions.shape == (2, 16, 7)
    assert action_targets.shape == (2, 30, 4, 4, 1)
    assert action_mask.shape == (2, 30, 4, 4, 1)
    assert torch.all(action_mask[:, :6] == 1)
    assert torch.all(action_mask[:, 6:28] == 0)
    assert torch.all(action_mask[:, 28:29] == 1)
    assert torch.all(action_mask[:, 29:30] == 0)
