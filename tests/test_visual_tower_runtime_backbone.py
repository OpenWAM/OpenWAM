from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from open_wam.configs import (
    ExportedRuntimeActionInitMode,
    SharedVideoTransformerConfig,
)
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower import VisualTower
from open_wam.models.visual_tower import runtime_backbone as runtime_backbone_module
from open_wam.models.visual_tower.reference_core_weights import BackboneLoadReport
from open_wam.models.visual_tower.runtime_backbone import (
    ensure_runtime_module_device,
    initialize_runtime_backbone,
    log_runtime_backbone_missing_keys,
    reset_runtime_module_cache,
    validate_runtime_backbone_request,
)


def _backbone_config() -> SharedVideoTransformerConfig:
    return SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        attention_head_dim=4,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
        pretrained_model_name_or_path=None,
        load_reference_core_weights=False,
    )


def test_runtime_backbone_initialization_is_idempotent_and_optional() -> None:
    core = nn.Linear(2, 2)
    existing = BackboneLoadReport(
        loaded_keys=("weight",),
        missing_reference_keys=tuple(),
    )

    assert (
        initialize_runtime_backbone(
            current_report=existing,
            core=core,
            config=_backbone_config(),
            action_dim=4,
        )
        is existing
    )
    assert (
        initialize_runtime_backbone(
            current_report=None,
            core=core,
            config=_backbone_config(),
            action_dim=4,
        )
        is None
    )


def test_required_runtime_backbone_initialization_rejects_missing_location() -> None:
    config = replace(_backbone_config(), load_reference_core_weights=True)

    with pytest.raises(ValueError, match="an absolute `backbone.transformer_subdir`"):
        initialize_runtime_backbone(
            current_report=None,
            core=nn.Linear(2, 2),
            config=config,
            action_dim=4,
        )


def test_runtime_backbone_initialization_rejects_missing_artifact(tmp_path) -> None:
    config = replace(
        _backbone_config(),
        pretrained_model_name_or_path=str(tmp_path / "missing-model"),
        load_reference_core_weights=True,
    )

    with pytest.raises(FileNotFoundError, match="existing transformer artifact"):
        initialize_runtime_backbone(
            current_report=None,
            core=nn.Linear(2, 2),
            config=config,
            action_dim=4,
        )


def test_runtime_backbone_initialization_loads_absolute_component_without_model_root(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = nn.Linear(2, 2)
    transformer_dir = tmp_path / "detached-transformer"
    transformer_dir.mkdir()
    expected = BackboneLoadReport(
        loaded_keys=("weight",),
        missing_reference_keys=(),
    )
    config = replace(
        _backbone_config(),
        transformer_subdir=str(transformer_dir),
    )

    monkeypatch.setattr(
        runtime_backbone_module,
        "is_open_wam_exported_runtime_backbone_dir",
        lambda path: False,
    )

    def load_reference(core_arg, *, backbone_config, action_dim):
        assert core_arg is core
        assert backbone_config is config
        assert action_dim == 4
        return expected

    monkeypatch.setattr(
        runtime_backbone_module,
        "load_reference_weights_into_replica_core",
        load_reference,
    )

    assert (
        initialize_runtime_backbone(
            current_report=None,
            core=core,
            config=config,
            action_dim=4,
        )
        is expected
    )


def test_runtime_backbone_request_validation_preserves_access_contract() -> None:
    config = _backbone_config()

    validate_runtime_backbone_request(
        config=config,
        configured_action_dim=4,
        requested_action_dim=4,
    )
    with pytest.raises(ValueError, match="configured action_dim"):
        validate_runtime_backbone_request(
            config=config,
            configured_action_dim=None,
            requested_action_dim=4,
        )
    with pytest.raises(ValueError, match="requested=5, tower_action_dim=4"):
        validate_runtime_backbone_request(
            config=config,
            configured_action_dim=4,
            requested_action_dim=5,
        )
    with pytest.raises(ValueError, match="implementation = shared_transformer"):
        validate_runtime_backbone_request(
            config=replace(config, implementation="dummy"),
            configured_action_dim=4,
            requested_action_dim=4,
        )


def test_runtime_backbone_device_normalizes_floating_state_in_place() -> None:
    module = nn.Linear(2, 2, dtype=torch.float64)
    module.register_buffer("floating_buffer", torch.ones(2, dtype=torch.float64))
    module.register_buffer("integer_buffer", torch.ones(2, dtype=torch.int64))

    resolved = ensure_runtime_module_device(module, device="cpu")

    assert resolved is module
    assert module.weight.dtype == torch.float32
    assert module.floating_buffer.dtype == torch.float32
    assert module.integer_buffer.dtype == torch.int64


def test_runtime_backbone_device_preserves_sharded_parameter_dtype() -> None:
    module = nn.Linear(2, 2, dtype=torch.float64)
    module.weight.full_tensor = lambda: module.weight
    module.register_buffer("floating_buffer", torch.ones(2, dtype=torch.float64))

    resolved = ensure_runtime_module_device(module, device="cpu")

    assert resolved is module
    assert module.weight.dtype == torch.float64
    assert module.bias.dtype == torch.float64
    assert module.floating_buffer.dtype == torch.float32


class _ModernCacheModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def clear_runtime_prediction_cache(self, cache_name: str) -> None:
        self.calls.append(("prediction", cache_name))

    def clear_runtime_cache_state(self, cache_name: str) -> None:
        self.calls.append(("state", cache_name))


class _LegacyCacheModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def clear_runtime_prediction_cache(self, cache_name: str) -> None:
        self.calls.append(("runtime_prediction", cache_name))
        raise AttributeError("legacy prediction cache")

    def clear_pred_cache(self, cache_name: str) -> None:
        self.calls.append(("prediction", cache_name))

    def clear_runtime_cache_state(self, cache_name: str) -> None:
        self.calls.append(("runtime_state", cache_name))
        raise AttributeError("legacy state cache")

    def clear_cache(self, cache_name: str) -> None:
        self.calls.append(("state", cache_name))


def test_runtime_backbone_cache_reset_supports_current_and_legacy_apis() -> None:
    modern = _ModernCacheModule()
    legacy = _LegacyCacheModule()

    reset_runtime_module_cache(modern, cache_name="modern")
    reset_runtime_module_cache(legacy, cache_name="legacy")

    assert modern.calls == [("prediction", "modern"), ("state", "modern")]
    assert legacy.calls == [
        ("runtime_prediction", "legacy"),
        ("prediction", "legacy"),
        ("runtime_state", "legacy"),
        ("state", "legacy"),
    ]


def test_runtime_backbone_missing_key_diagnostics_classify_gaps(capsys) -> None:
    config = replace(
        _backbone_config(),
        exported_runtime_action_init_mode=ExportedRuntimeActionInitMode.RANDOM,
    )
    report = BackboneLoadReport(
        loaded_keys=("patch_embedding_mlp.weight",),
        missing_reference_keys=(
            "proprio_context_encoder.input_proj.weight",
            "action_embedder.weight",
            "blocks.0.attn1.to_q.weight",
        ),
    )

    log_runtime_backbone_missing_keys(report, config=config)

    assert capsys.readouterr().out.splitlines() == [
        "[runtime_backbone_load] "
        "allowed_missing_keys=['proprio_context_encoder.input_proj.weight', "
        "'action_embedder.weight']",
        "[runtime_backbone_load] unexpected_missing_keys_count=1 "
        "unexpected_missing_keys_preview=['blocks.0.attn1.to_q.weight']",
    ]


def test_visual_tower_owns_copied_frontend_and_named_cache_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tower = VisualTower(_backbone_config(), action_dim=4)
    frontend_state = [torch.tensor([2.0])]

    monkeypatch.setattr(
        tower.frontend,
        "snapshot_runtime_state",
        lambda: copy.deepcopy(frontend_state),
    )

    def restore_frontend(snapshot) -> None:
        frontend_state[:] = copy.deepcopy(snapshot)

    monkeypatch.setattr(tower.frontend, "restore_runtime_state", restore_frontend)
    tower.core._exact_runtime_caches["planner"] = CacheState(
        supported=True,
        current_start_frame=1,
        cached_frames=2,
        chunk_size=1,
        payload={"value": torch.tensor([1.0])},
    )

    snapshot = tower.snapshot_runtime_state(cache_name="planner")
    assert snapshot is not None
    frontend_state[0][0] = 8.0
    tower.core._exact_runtime_caches["planner"].payload["value"][0] = 9.0

    tower.restore_runtime_state(snapshot)

    assert float(frontend_state[0][0]) == 2.0
    assert float(tower.core._exact_runtime_caches["planner"].payload["value"][0]) == 1.0


def test_visual_tower_snapshot_removes_cache_created_by_speculation() -> None:
    tower = VisualTower(_backbone_config(), action_dim=4)

    snapshot = tower.snapshot_runtime_state(cache_name="new_planner")
    assert snapshot is not None
    assert not snapshot.runtime_cache_existed
    tower.core._exact_runtime_caches["new_planner"] = CacheState(
        supported=True,
        current_start_frame=1,
        cached_frames=1,
        chunk_size=1,
    )

    tower.restore_runtime_state(snapshot)

    assert "new_planner" not in tower.core._exact_runtime_caches


def test_visual_tower_named_cache_snapshot_requires_action_dimension() -> None:
    tower = VisualTower(_backbone_config(), action_dim=None)

    with pytest.raises(ValueError, match="require a configured action_dim"):
        tower.snapshot_runtime_state(cache_name="planner")
