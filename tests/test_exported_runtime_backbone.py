from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from open_wam.configs import ExportedRuntimeActionInitMode
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.exported_runtime_backbone import (
    is_action_runtime_target_key,
    is_allowed_runtime_missing_key,
    load_exported_runtime_backbone_into_replica_core,
)
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def _tiny_backbone_config(
    tmp_path: Path,
    *,
    action_init_mode: ExportedRuntimeActionInitMode,
) -> SharedVideoTransformerConfig:
    return SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        attention_head_dim=4,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
        pretrained_model_name_or_path=str(tmp_path / "exported_runtime"),
        transformer_subdir="transformer",
        exported_runtime_action_init_mode=action_init_mode,
    )


def _write_exported_runtime_checkpoint(
    config: SharedVideoTransformerConfig,
    *,
    action_dim: int,
) -> None:
    source_core = SharedVideoTransformerCore(config, action_dim=action_dim)
    state = {key: value.detach().cpu().clone() for key, value in source_core.state_dict().items()}
    state["patch_embedding_mlp.bias"].fill_(11.0)
    state["action_embedder.weight"].fill_(13.0)
    state["action_time_conditioner.time_proj.bias"].fill_(17.0)
    state["runtime_stream_adapters.action_register_adapter.2.bias"].fill_(19.0)
    state["action_proj_out.bias"].fill_(23.0)

    transformer_dir = Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    transformer_dir.mkdir(parents=True)
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")


def test_exported_runtime_random_action_init_skips_action_runtime_weights(tmp_path: Path) -> None:
    config = _tiny_backbone_config(tmp_path, action_init_mode=ExportedRuntimeActionInitMode.RANDOM)
    _write_exported_runtime_checkpoint(config, action_dim=4)

    target_core = SharedVideoTransformerCore(config, action_dim=4)
    initial_action_embedder = target_core.action_embedder.weight.detach().clone()
    initial_action_time_bias = target_core.action_time_conditioner.time_proj.bias.detach().clone()
    initial_action_adapter_bias = target_core.runtime_stream_adapters.action_register_adapter[2].bias.detach().clone()
    initial_action_output_bias = target_core.action_proj_out.bias.detach().clone()

    report = load_exported_runtime_backbone_into_replica_core(target_core, backbone_config=config)

    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )
    assert torch.equal(target_core.action_embedder.weight, initial_action_embedder)
    assert torch.equal(target_core.action_time_conditioner.time_proj.bias, initial_action_time_bias)
    assert torch.equal(
        target_core.runtime_stream_adapters.action_register_adapter[2].bias,
        initial_action_adapter_bias,
    )
    assert torch.equal(target_core.action_proj_out.bias, initial_action_output_bias)
    assert "patch_embedding_mlp.bias" in report.loaded_keys
    assert "action_embedder.weight" not in report.loaded_keys
    assert "action_time_conditioner.time_proj.bias" not in report.loaded_keys
    assert "runtime_stream_adapters.action_register_adapter.2.bias" not in report.loaded_keys
    assert "action_proj_out.bias" not in report.loaded_keys
    assert "action_embedder.weight" in report.missing_reference_keys
    assert "action_time_conditioner.time_proj.bias" in report.missing_reference_keys
    assert "runtime_stream_adapters.action_register_adapter.2.bias" in report.missing_reference_keys
    assert "action_proj_out.bias" in report.missing_reference_keys


def test_exported_runtime_load_from_checkpoint_keeps_action_runtime_weights(tmp_path: Path) -> None:
    config = _tiny_backbone_config(tmp_path, action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT)
    _write_exported_runtime_checkpoint(config, action_dim=4)

    target_core = SharedVideoTransformerCore(config, action_dim=4)
    report = load_exported_runtime_backbone_into_replica_core(target_core, backbone_config=config)

    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )
    assert torch.equal(target_core.action_embedder.weight, torch.full_like(target_core.action_embedder.weight, 13.0))
    assert torch.equal(
        target_core.action_time_conditioner.time_proj.bias,
        torch.full_like(target_core.action_time_conditioner.time_proj.bias, 17.0),
    )
    assert torch.equal(
        target_core.runtime_stream_adapters.action_register_adapter[2].bias,
        torch.full_like(target_core.runtime_stream_adapters.action_register_adapter[2].bias, 19.0),
    )
    assert torch.equal(target_core.action_proj_out.bias, torch.full_like(target_core.action_proj_out.bias, 23.0))
    assert "action_embedder.weight" in report.loaded_keys
    assert "action_time_conditioner.time_proj.bias" in report.loaded_keys
    assert "runtime_stream_adapters.action_register_adapter.2.bias" in report.loaded_keys
    assert "action_proj_out.bias" in report.loaded_keys


def test_exported_runtime_action_missing_key_policy_matches_skip_predicate() -> None:
    action_keys = (
        "action_embedder.weight",
        "action_time_conditioner.time_proj.bias",
        "action_text_proj.linear_1.weight",
        "runtime_stream_adapters.action_register_adapter.2.weight",
        "action_proj_out.bias",
    )
    for key in action_keys:
        assert is_action_runtime_target_key(key)
        assert is_allowed_runtime_missing_key(key, allow_random_action=True)
        assert not is_allowed_runtime_missing_key(key, allow_random_action=False)

    assert not is_action_runtime_target_key("runtime_stream_adapters.state_register_adapter.0.weight")
    assert not is_allowed_runtime_missing_key(
        "runtime_stream_adapters.state_register_adapter.0.weight",
        allow_random_action=True,
    )
    assert is_allowed_runtime_missing_key("proprio_context_encoder.input_proj.weight", allow_random_action=False)
    assert is_allowed_runtime_missing_key(
        "generalist_mode_context_encoder.embedding.weight",
        allow_random_action=False,
    )
