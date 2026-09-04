from __future__ import annotations

from pathlib import Path

import torch

from open_wam.configs import (
    ParallelStreamPolicyConfig,
    TrainingConfig,
    VideoActionProgram,
)
from open_wam.models.policy_variants.parallel_stream.reference_runtime import run_parallel_exact_train
from open_wam.models.policy_variants.parallel_stream.training_artifacts import (
    prepare_parallel_exact_train_artifacts,
)
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower import VisualTower
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class

from .reference_model_test_utils import reference_model_path_or_skip


def _clone_train_input_dict(input_dict: dict[str, object]) -> dict[str, object]:
    return {
        key: {inner_key: inner_value.clone() for inner_key, inner_value in value.items()} if isinstance(value, dict) else value
        for key, value in input_dict.items()
    }
def test_exact_train_runtime_matches_reference_transformer_forward_train(tmp_path: Path) -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="lingbot_replica",
        attn_mode="torch",
        train_attn_mode="torch",
        infer_attn_mode="torch",
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

    tower = VisualTower(
        LingbotCompatibleVideoBackboneConfig(
            implementation="lingbot_replica",
            attn_mode="torch",
            train_attn_mode="torch",
            infer_attn_mode="torch",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            pretrained_model_name_or_path=str(tmp_path / "lingbot_ckpt"),
            load_reference_core_weights=True,
            reference_model_path=reference_model_path_or_skip(),
        ),
        action_dim=4,
    )
    transformer = tower.get_runtime_backbone(action_dim=4).to(dtype=torch.bfloat16)
    policy_config = ParallelStreamPolicyConfig(
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        hidden_size=32,
        frame_chunk_size=2,
        action_per_frame=2,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    video_latents = torch.randn(2, 48, 2, 24, 20)
    actions = torch.randn(2, 4, 4)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    text_emb = torch.randn(2, 512, 16)

    train_artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
    )

    ours_input = _clone_train_input_dict(train_artifacts.input_dict)
    reference_input = _clone_train_input_dict(train_artifacts.input_dict)
    latent_pred, action_pred = run_parallel_exact_train(transformer, ours_input)
    reference_dtype = next(reference_model.parameters()).dtype
    for stream_name in ("latent_dict", "action_dict"):
        stream = reference_input[stream_name]
        assert isinstance(stream, dict)
        for key, value in tuple(stream.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                stream[key] = value.to(dtype=reference_dtype)
    assert reference_input["latent_dict"]["noisy_latents"].dtype == reference_dtype
    reference_module = __import__(reference_model.__class__.__module__, fromlist=["FlexAttnFunc"])
    original_init_mask = reference_module.FlexAttnFunc.init_mask
    reference_module.FlexAttnFunc.init_mask = staticmethod(lambda *args, **kwargs: None)
    reference_latent_pred, reference_action_pred = reference_model.forward_train(reference_input)
    reference_module.FlexAttnFunc.init_mask = original_init_mask

    assert torch.allclose(latent_pred.float(), reference_latent_pred.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(action_pred.float(), reference_action_pred.float(), atol=1e-2, rtol=1e-2)
