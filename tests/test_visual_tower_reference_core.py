from __future__ import annotations

from pathlib import Path

import torch

from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower import VisualCoreInput, VisualTower
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class

from reference_model_test_utils import reference_model_path_or_skip


def test_visual_tower_can_initialize_replica_core_from_reference_weights(tmp_path: Path) -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
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

    tower = VisualTower(backbone_config, action_dim=4)

    assert tower.reference_core_load_report is not None
    assert torch.equal(
        tower.core.time_conditioner.time_embedder.linear_1.weight,
        reference_model.state_dict()["condition_embedder.time_embedder.linear_1.weight"],
    )
    assert torch.equal(
        tower.core.action_time_conditioner.time_embedder.linear_1.weight,
        reference_model.state_dict()["condition_embedder_action.time_embedder.linear_1.weight"],
    )
    assert torch.equal(
        tower.core.blocks[0].attn1.to_q.weight,
        reference_model.state_dict()["blocks.0.attn1.to_q.weight"],
    )
    assert torch.equal(
        tower.core.patch_embedding_mlp.weight,
        reference_model.state_dict()["patch_embedding_mlp.weight"],
    )
    assert torch.equal(
        tower.core.action_proj_out.weight,
        reference_model.state_dict()["action_proj_out.weight"],
    )

    output = tower.run_core(
        VisualCoreInput(
            tokens=torch.randn(2, 12, 32),
            stream_ids=torch.tensor([[0] * 6 + [1] * 6, [0] * 6 + [1] * 6]),
            timestep_values=torch.zeros(2, 12, dtype=torch.float32),
        )
    )
    assert output.aux["weight_source"] == "reference_initialized"
    assert output.aux["used_action_conditioner"] is True
