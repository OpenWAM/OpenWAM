from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from open_wam.configs import (
    ExportedRuntimeActionInitMode,
    SharedVideoTransformerConfig,
)
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
