from __future__ import annotations

from pathlib import Path

import pytest
import torch

from open_wam.configs import ReferenceCoreInitMode
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.models.visual_tower.reference_core_weights import (
    load_reference_weights_into_replica_core,
)
from open_wam.models.visual_tower.reference_transformer import build_reference_transformer
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore

from .reference_model_test_utils import reference_model_path_or_skip


def _tiny_backbone_config(
    *,
    pretrained_model_name_or_path: str | None,
) -> SharedVideoTransformerConfig:
    return SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer_subdir="transformer",
        load_reference_core_weights=True,
        reference_model_path=reference_model_path_or_skip(),
    )


def test_missing_export_for_requested_checkpoint_is_a_hard_error(tmp_path: Path) -> None:
    """A requested warm start that is not on disk must stop the run.

    This is the regression that motivated the guard. A checkpoint written with
    `training.export_runtime_backbone: false` has `model_state.pt` and no
    `transformer/`, so `resolve_pretrained_component_dir` hands back a path
    that does not exist. The old code fell through to the fresh-construct
    branch and returned a healthy randomly initialised transformer, which is
    exactly what a correct warm start also looks like from the outside.
    """

    checkpoint_root = tmp_path / "checkpoint_step_4250"
    checkpoint_root.mkdir()
    # Everything a real un-exported checkpoint has, and nothing it lacks.
    (checkpoint_root / "model_state.pt").write_bytes(b"")
    (checkpoint_root / ".checkpoint_complete").write_text("ok\n", encoding="utf-8")

    config = _tiny_backbone_config(pretrained_model_name_or_path=str(checkpoint_root))

    with pytest.raises(FileNotFoundError) as excinfo:
        build_reference_transformer(config, action_dim=4)

    # The operator has to be told which directory to go create; a bare
    # "checkpoint not found" would send them looking at `model_state.pt`,
    # which is present.
    message = str(excinfo.value)
    assert str(checkpoint_root / "transformer") in message
    assert "export_runtime_backbone" in message


def test_missing_export_is_not_reported_as_a_successful_warm_start(tmp_path: Path) -> None:
    """The silent path used to reach all the way out to a clean load report.

    `load_reference_weights_into_replica_core` copies tensor-by-tensor from
    whatever `build_reference_transformer` returns, so a from-scratch model
    satisfied every key it asked for: `loaded_keys` came back full and
    `missing_reference_keys` came back empty. Pin the failure at the boundary
    the caller actually uses, not just at the builder.
    """

    config = _tiny_backbone_config(pretrained_model_name_or_path=str(tmp_path / "never_exported"))
    replica_core = SharedVideoTransformerCore(config, action_dim=4)

    with pytest.raises(FileNotFoundError):
        load_reference_weights_into_replica_core(
            replica_core,
            backbone_config=config,
            action_dim=4,
        )


def test_unloadable_export_is_a_hard_error(tmp_path: Path) -> None:
    """A `transformer/` that exists but cannot load is the same failure.

    A half-written export -- an interrupted copy, a truncated safetensors file
    -- passes the `exists()` check. Before the guard, the exception from
    `from_pretrained` was not caught here at all, but the directory-shaped
    variants of it (empty dir, foreign `config.json`) resolved to the same
    silent-construct outcome. Treat anything unloadable as a stop.
    """

    checkpoint_root = tmp_path / "checkpoint_step_4250"
    transformer_dir = checkpoint_root / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{ this is not json", encoding="utf-8")

    config = _tiny_backbone_config(pretrained_model_name_or_path=str(checkpoint_root))

    with pytest.raises(RuntimeError) as excinfo:
        build_reference_transformer(config, action_dim=4)

    assert str(transformer_dir) in str(excinfo.value)


def test_no_requested_checkpoint_still_builds_from_scratch() -> None:
    """The from-scratch path is legitimate and must stay silent.

    Configs that never set `backbone.pretrained_model_name_or_path` are asking
    for random init on purpose; the guard must not make those an error.
    """

    config = _tiny_backbone_config(pretrained_model_name_or_path=None)

    model = build_reference_transformer(config, action_dim=4)

    assert isinstance(model, torch.nn.Module)
