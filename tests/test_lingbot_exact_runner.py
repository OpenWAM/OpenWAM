from __future__ import annotations

from pathlib import Path

import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ExperimentConfig,
    InferenceConfig,
    LingbotParallelActionDecoderConfig,
    ParallelStreamPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.data import build_synthetic_batch
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class
from open_wam.pipelines import build_exact_runtime_runner_from_config

from .reference_model_test_utils import reference_model_path_or_skip


def test_lingbot_exact_runner_supports_warmup_and_chunk_generation(tmp_path: Path) -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        pretrained_model_name_or_path=str(tmp_path / "lingbot_ckpt"),
        reference_model_path=reference_model_path_or_skip(),
    )
    model_cls = load_wan_transformer_class(backbone_config)
    reference_model = model_cls(
        patch_size=[1, 2, 2],
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=48,
        out_channels=48,
        action_dim=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        attn_mode="torch",
    ).to(dtype=torch.bfloat16)
    transformer_dir = tmp_path / "lingbot_ckpt" / "transformer"
    reference_model.save_pretrained(transformer_dir)

    config = ExperimentConfig(
        name="lingbot_exact_runner_test",
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=backbone_config,
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            used_action_channel_ids=(0, 3),
            inverse_used_action_channel_ids=(0, 2, 2, 1),
            action_norm_method="quantiles",
            norm_q01=(0.0, 0.0, 0.0, 0.0),
            norm_q99=(1.0, 1.0, 1.0, 1.0),
        ),
        action_decoder=LingbotParallelActionDecoderConfig(
            hidden_size=32,
            action_dim=4,
            action_horizon=4,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    batch = build_synthetic_batch(config.data, batch_size=2)
    runner = build_exact_runtime_runner_from_config(config)
    shared_transformer = runner.pipeline.visual_tower.get_runtime_backbone(action_dim=4)
    assert shared_transformer is runner.pipeline.visual_tower.get_runtime_backbone(action_dim=4)
    assert shared_transformer is runner.pipeline.visual_tower.core
    assert not hasattr(runner.policy_variant, "reference_transformer")
    session = runner.reset(task_text=batch.task_text)

    alias_views = {f"observation.images.{name}": value for name, value in batch.views.items()}
    raw_action_history = torch.rand(2, 4, 2)
    warmup = runner.warmup_cache(
        session=session,
        views=alias_views,
        action_history=raw_action_history,
        action_space="raw",
    )

    assert warmup.session.policy_state.cache["cache_initialized"] is True
    assert warmup.session.policy_state.cache["frame_start"] == 2

    infer_chunk = runner.infer_chunk(session=warmup.session)
    second_chunk = runner.infer_chunk(session=infer_chunk.session)

    assert infer_chunk.chunk_action_pred.shape == (2, 4, 4)
    assert infer_chunk.raw_chunk_action_pred is not None
    assert infer_chunk.raw_chunk_action_pred.shape == (2, 4, 2)
    assert infer_chunk.decoder_output.action_pred.shape == (2, 4, 4)
    assert infer_chunk.decoder_output.aux["action_space"] == "model"
    assert infer_chunk.decoder_output.aux["raw_action_pred"].shape == (2, 4, 2)
    assert infer_chunk.predicted_latents.shape[:3] == (2, 48, 2)
    assert second_chunk.chunk_action_pred.shape == (2, 4, 4)
    assert second_chunk.session.policy_state.step_index == 2
