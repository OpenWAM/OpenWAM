from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower.reference_transformer import build_reference_transformer
from open_wam.third_party.lingbot import WanTransformer3DModel


def _build_small_transformer(*, attn_mode: str) -> WanTransformer3DModel:
    return WanTransformer3DModel(
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
        attn_mode=attn_mode,
    )


def _assert_loaded_matches(
    *,
    checkpoint_root: Path,
    expected_model: WanTransformer3DModel,
    expected_attn_mode: str,
) -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="lingbot_replica",
        pretrained_model_name_or_path=str(checkpoint_root),
    )
    loaded = build_reference_transformer(backbone_config, action_dim=30)
    assert loaded.config.attn_mode == expected_attn_mode

    expected_weight = expected_model.state_dict()["patch_embedding_mlp.weight"]
    actual_weight = loaded.state_dict()["patch_embedding_mlp.weight"]
    assert torch.equal(actual_weight, expected_weight.to(dtype=actual_weight.dtype))


def test_vendored_lingbot_import_installs_flash_attn_shims() -> None:
    assert WanTransformer3DModel.__name__ == "WanTransformer3DModel"


def test_open_wam_loads_single_file_external_trainer_checkpoint_layout(tmp_path: Path) -> None:
    model = _build_small_transformer(attn_mode="flashattn")
    checkpoint_root = tmp_path / "checkpoint_step_600"
    transformer_dir = checkpoint_root / "transformer"
    transformer_dir.mkdir(parents=True)

    state_dict_bf16 = {
        key: value.to(torch.bfloat16) if torch.is_floating_point(value) else value
        for key, value in model.state_dict().items()
    }
    save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")
    config_dict = dict(model.config)
    config_dict.pop("_name_or_path", None)
    with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2)

    _assert_loaded_matches(
        checkpoint_root=checkpoint_root,
        expected_model=model,
        expected_attn_mode="flashattn",
    )


def test_open_wam_loads_sharded_external_base_model_layout(tmp_path: Path) -> None:
    model = _build_small_transformer(attn_mode="flex")
    checkpoint_root = tmp_path / "lingbot_va_base"
    transformer_dir = checkpoint_root / "transformer"
    model.save_pretrained(transformer_dir, max_shard_size="50KB")

    _assert_loaded_matches(
        checkpoint_root=checkpoint_root,
        expected_model=model,
        expected_attn_mode="flex",
    )
