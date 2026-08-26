from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from open_wam.configs import ExportedRuntimeActionInitMode
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.component_ownership import owned_state_dict_keys
from open_wam.models.visual_tower.exported_runtime_backbone import (
    is_action_runtime_target_key,
    is_allowed_runtime_missing_key,
    is_open_wam_exported_runtime_backbone_dir,
    load_exported_runtime_backbone_into_replica_core,
)
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore
from open_wam.runtime.runtime_backbone_manifest import (
    RuntimeBackboneManifest,
    write_runtime_backbone_manifest,
)


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
    state = {
        key: value.detach().cpu().clone()
        for key, value in source_core.state_dict().items()
    }
    state["patch_embedding_mlp.bias"].fill_(11.0)
    state["action_embedder.weight"].fill_(13.0)
    state["action_time_conditioner.time_proj.bias"].fill_(17.0)
    state["runtime_stream_adapters.action_register_adapter.2.bias"].fill_(19.0)
    state["action_proj_out.bias"].fill_(23.0)

    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    transformer_dir.mkdir(parents=True)
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")


def test_exported_runtime_random_action_init_skips_action_runtime_weights(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path, action_init_mode=ExportedRuntimeActionInitMode.RANDOM
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)

    target_core = SharedVideoTransformerCore(config, action_dim=4)
    initial_action_embedder = target_core.action_embedder.weight.detach().clone()
    initial_action_time_bias = (
        target_core.action_time_conditioner.time_proj.bias.detach().clone()
    )
    initial_action_adapter_bias = (
        target_core.runtime_stream_adapters.action_register_adapter[2]
        .bias.detach()
        .clone()
    )
    initial_action_output_bias = target_core.action_proj_out.bias.detach().clone()

    report = load_exported_runtime_backbone_into_replica_core(
        target_core, backbone_config=config
    )

    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )
    assert torch.equal(target_core.action_embedder.weight, initial_action_embedder)
    assert torch.equal(
        target_core.action_time_conditioner.time_proj.bias, initial_action_time_bias
    )
    assert torch.equal(
        target_core.runtime_stream_adapters.action_register_adapter[2].bias,
        initial_action_adapter_bias,
    )
    assert torch.equal(target_core.action_proj_out.bias, initial_action_output_bias)
    assert "patch_embedding_mlp.bias" in report.loaded_keys
    assert "action_embedder.weight" not in report.loaded_keys
    assert "action_time_conditioner.time_proj.bias" not in report.loaded_keys
    assert (
        "runtime_stream_adapters.action_register_adapter.2.bias"
        not in report.loaded_keys
    )
    assert "action_proj_out.bias" not in report.loaded_keys
    assert "action_embedder.weight" in report.missing_reference_keys
    assert "action_time_conditioner.time_proj.bias" in report.missing_reference_keys
    assert (
        "runtime_stream_adapters.action_register_adapter.2.bias"
        in report.missing_reference_keys
    )
    assert "action_proj_out.bias" in report.missing_reference_keys


def test_detached_runtime_backbone_does_not_require_a_pretrained_root(
    tmp_path: Path,
) -> None:
    bundled_config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(bundled_config, action_dim=4)
    transformer_dir = (
        Path(bundled_config.pretrained_model_name_or_path)
        / bundled_config.transformer_subdir
    )
    detached_config = replace(
        bundled_config,
        pretrained_model_name_or_path=None,
        runtime_backbone_artifact_path=str(transformer_dir),
    )
    target_core = SharedVideoTransformerCore(detached_config, action_dim=4)

    report = load_exported_runtime_backbone_into_replica_core(
        target_core,
        backbone_config=detached_config,
    )

    assert "patch_embedding_mlp.bias" in report.loaded_keys
    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )


def test_exported_runtime_load_from_checkpoint_keeps_action_runtime_weights(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path, action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)

    target_core = SharedVideoTransformerCore(config, action_dim=4)
    report = load_exported_runtime_backbone_into_replica_core(
        target_core, backbone_config=config
    )

    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )
    assert torch.equal(
        target_core.action_embedder.weight,
        torch.full_like(target_core.action_embedder.weight, 13.0),
    )
    assert torch.equal(
        target_core.action_time_conditioner.time_proj.bias,
        torch.full_like(target_core.action_time_conditioner.time_proj.bias, 17.0),
    )
    assert torch.equal(
        target_core.runtime_stream_adapters.action_register_adapter[2].bias,
        torch.full_like(
            target_core.runtime_stream_adapters.action_register_adapter[2].bias, 19.0
        ),
    )
    assert torch.equal(
        target_core.action_proj_out.bias,
        torch.full_like(target_core.action_proj_out.bias, 23.0),
    )
    assert "action_embedder.weight" in report.loaded_keys
    assert "action_time_conditioner.time_proj.bias" in report.loaded_keys
    assert (
        "runtime_stream_adapters.action_register_adapter.2.bias" in report.loaded_keys
    )
    assert "action_proj_out.bias" in report.loaded_keys


def test_scoped_video_export_initializes_omitted_components_from_model_defaults(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    source_core = SharedVideoTransformerCore(config, action_dim=4)
    source_core.patch_embedding_mlp.bias.data.fill_(11.0)
    source_core.action_embedder.weight.data.fill_(13.0)
    video_keys = owned_state_dict_keys(
        source_core,
        source_core.component_topology().shared_video_backbone,
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in source_core.state_dict().items()
        if key in video_keys
    }
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    transformer_dir.mkdir(parents=True)
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.shared_video_backbone",),
            state_keys=tuple(sorted(state)),
        ),
    )

    target_core = SharedVideoTransformerCore(config, action_dim=4)
    initial_action = target_core.action_embedder.weight.detach().clone()
    report = load_exported_runtime_backbone_into_replica_core(
        target_core,
        backbone_config=config,
    )

    assert torch.equal(
        target_core.patch_embedding_mlp.bias,
        torch.full_like(target_core.patch_embedding_mlp.bias, 11.0),
    )
    assert torch.equal(target_core.action_embedder.weight, initial_action)
    assert report.missing_reference_keys == ()


def test_complete_manifest_matches_the_active_runtime_backbone(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    state = load_file(transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.runtime_backbone",),
            state_keys=tuple(state),
        ),
    )

    report = load_exported_runtime_backbone_into_replica_core(
        SharedVideoTransformerCore(config, action_dim=4),
        backbone_config=config,
    )

    assert set(report.loaded_keys) == set(state)
    assert report.missing_reference_keys == ()


def test_complete_manifest_allows_optional_source_components(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    source_core = SharedVideoTransformerCore(config, action_dim=4, state_dim=3)
    source_core.configure_proprio_context_encoder(enabled=True)
    source_core.configure_generalist_mode_context_encoder(enabled=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in source_core.state_dict().items()
    }
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    transformer_dir.mkdir(parents=True)
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.runtime_backbone",),
            state_keys=tuple(state),
        ),
    )

    target_core = SharedVideoTransformerCore(config, action_dim=4, state_dim=3)
    report = load_exported_runtime_backbone_into_replica_core(
        target_core,
        backbone_config=config,
    )

    assert "patch_embedding_mlp.weight" in report.loaded_keys
    assert not any(
        key.startswith(
            ("proprio_context_encoder.", "generalist_mode_context_encoder.")
        )
        for key in report.loaded_keys
    )


def test_complete_manifest_allows_optional_target_components(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    state = load_file(transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.runtime_backbone",),
            state_keys=tuple(state),
        ),
    )

    target_core = SharedVideoTransformerCore(config, action_dim=4, state_dim=3)
    target_core.configure_proprio_context_encoder(enabled=True)
    target_core.configure_generalist_mode_context_encoder(enabled=True)
    report = load_exported_runtime_backbone_into_replica_core(
        target_core,
        backbone_config=config,
    )

    assert "proprio_context_encoder.proj.weight" in report.missing_reference_keys
    assert (
        "generalist_mode_context_encoder.embedding.weight"
        in report.missing_reference_keys
    )


def test_manifest_identifies_scoped_export_without_legacy_key_prefixes(
    tmp_path: Path,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    state = {"runtime_stream_adapters.test.weight": torch.ones(1)}
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.shared_runtime_adapters",),
            state_keys=tuple(state),
        ),
    )

    assert is_open_wam_exported_runtime_backbone_dir(transformer_dir)


def test_runtime_backbone_manifest_rejects_weight_inventory_drift(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.shared_video_backbone",),
            state_keys=("declared.but.missing",),
        ),
    )

    with pytest.raises(ValueError, match="do not match their manifest"):
        load_exported_runtime_backbone_into_replica_core(
            SharedVideoTransformerCore(config, action_dim=4),
            backbone_config=config,
        )


def test_runtime_backbone_manifest_rejects_unknown_model_tensors(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    state = load_file(weights_path)
    state["unknown.weight"] = torch.ones(1)
    save_file(state, weights_path)
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.shared_video_backbone",),
            state_keys=tuple(state),
        ),
    )

    with pytest.raises(ValueError, match="absent from the active model"):
        load_exported_runtime_backbone_into_replica_core(
            SharedVideoTransformerCore(config, action_dim=4),
            backbone_config=config,
        )


def test_complete_manifest_rejects_missing_required_active_tensor(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    _write_exported_runtime_checkpoint(config, action_dim=4)
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    state = load_file(weights_path)
    state.pop("patch_embedding_mlp.bias")
    save_file(state, weights_path)
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.runtime_backbone",),
            state_keys=tuple(state),
        ),
    )

    with pytest.raises(ValueError, match="omits tensors required by the active model"):
        load_exported_runtime_backbone_into_replica_core(
            SharedVideoTransformerCore(config, action_dim=4),
            backbone_config=config,
        )


def test_runtime_backbone_manifest_enforces_declared_component_ownership(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path,
        action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT,
    )
    source_core = SharedVideoTransformerCore(config, action_dim=4)
    video_keys = owned_state_dict_keys(
        source_core,
        source_core.component_topology().shared_video_backbone,
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in source_core.state_dict().items()
        if key in video_keys or key == "action_embedder.weight"
    }
    transformer_dir = (
        Path(config.pretrained_model_name_or_path) / config.transformer_subdir
    )
    transformer_dir.mkdir(parents=True)
    save_file(state, transformer_dir / "diffusion_pytorch_model.safetensors")
    write_runtime_backbone_manifest(
        transformer_dir,
        RuntimeBackboneManifest(
            components=("visual_tower.shared_video_backbone",),
            state_keys=tuple(state),
        ),
    )

    with pytest.raises(ValueError, match="outside_components=.*action_embedder.weight"):
        load_exported_runtime_backbone_into_replica_core(
            SharedVideoTransformerCore(config, action_dim=4),
            backbone_config=config,
        )


def test_runtime_backbone_manifest_rejects_redundant_component_scopes() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        RuntimeBackboneManifest(
            components=(
                "visual_tower.runtime_backbone",
                "visual_tower.shared_video_backbone",
            ),
            state_keys=("patch_embedding_mlp.weight",),
        )


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

    assert not is_action_runtime_target_key(
        "runtime_stream_adapters.state_register_adapter.0.weight"
    )
    assert is_allowed_runtime_missing_key(
        "runtime_stream_adapters.state_register_adapter.0.weight",
        allow_random_action=True,
    )
    assert is_allowed_runtime_missing_key(
        "runtime_stream_adapters.state_register_adapter.0.weight",
        allow_random_action=False,
    )
    assert is_allowed_runtime_missing_key(
        "proprio_context_encoder.input_proj.weight", allow_random_action=False
    )
    assert is_allowed_runtime_missing_key(
        "generalist_mode_context_encoder.embedding.weight",
        allow_random_action=False,
    )


def test_checkpoint_compatibility_parameter_keys_keep_legacy_order(
    tmp_path: Path,
) -> None:
    config = _tiny_backbone_config(
        tmp_path, action_init_mode=ExportedRuntimeActionInitMode.LOAD_FROM_CHECKPOINT
    )
    core = SharedVideoTransformerCore(config, action_dim=4, state_dim=3)

    compatibility_keys = tuple(
        key for key in core.state_dict() if key.startswith("runtime_stream_adapters.")
    )

    assert compatibility_keys == (
        "runtime_stream_adapters.action_register_adapter.0.weight",
        "runtime_stream_adapters.action_register_adapter.0.bias",
        "runtime_stream_adapters.action_register_adapter.2.weight",
        "runtime_stream_adapters.action_register_adapter.2.bias",
        "runtime_stream_adapters.state_register_adapter.0.weight",
        "runtime_stream_adapters.state_register_adapter.0.bias",
        "runtime_stream_adapters.state_register_adapter.2.weight",
        "runtime_stream_adapters.state_register_adapter.2.bias",
        "runtime_stream_adapters.role_embedding.weight",
    )
