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
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype
from open_wam.pipelines import build_variant_pipeline_from_config

from reference_model_test_utils import reference_model_path_or_skip


def test_lingbot_reference_transformer_weights_load_as_is(tmp_path: Path) -> None:
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
        action_dim=30,
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
        name="lingbot_exact_load_test",
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=30, action_horizon=8, state_dim=30, state_horizon=1),
        ),
        backbone=backbone_config,
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(
            hidden_size=32,
            action_dim=30,
            action_horizon=8,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    reference_transformer = pipeline.visual_tower.get_runtime_backbone(action_dim=30)
    assert reference_transformer is pipeline.visual_tower.core
    loaded_state_dict = reference_transformer.state_dict()
    preferred_dtype = preferred_reference_dtype(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    expected_dtype = loaded_state_dict["patch_embedding_mlp.weight"].dtype

    assert expected_dtype in {torch.float32, preferred_dtype}
    assert torch.equal(
        loaded_state_dict["patch_embedding_mlp.weight"],
        reference_model.state_dict()["patch_embedding_mlp.weight"].to(dtype=expected_dtype),
    )
    assert torch.equal(
        loaded_state_dict["blocks.0.attn1.to_q.weight"],
        reference_model.state_dict()["blocks.0.attn1.to_q.weight"].to(dtype=expected_dtype),
    )
    assert torch.equal(
        loaded_state_dict["action_proj_out.weight"],
        reference_model.state_dict()["action_proj_out.weight"].to(dtype=expected_dtype),
    )

    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )
    second_infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(
            state=batch.state,
            previous_action=infer_output.decoder_output.action_pred,
            extra={"task_text": batch.task_text},
        ),
        infer_state=infer_output.policy_output.next_state,
    )

    assert train_output.decoder_output.action_pred.shape == (2, 8, 30)
    assert infer_output.decoder_output.action_pred.shape == (2, 8, 30)
    assert second_infer_output.decoder_output.action_pred.shape == (2, 8, 30)
    assert second_infer_output.policy_output.next_state.step_index == 2
    assert second_infer_output.policy_output.next_state.cache["cache_initialized"] is True
