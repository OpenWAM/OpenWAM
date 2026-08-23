from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from open_wam.configs.enums import (
    RolloutContextPolicy,
    SampleTargetAlignment,
    WindowSamplingMode,
)
from open_wam.runtime.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointCompatibilityPolicy,
    load_pipeline_checkpoint,
    normalize_checkpoint_state_dict,
    resolve_checkpoint_file,
)
from open_wam.utils import (
    find_checkpoint_resolved_config,
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_checkpoint_file_accepts_checkpoint_step_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_12"
    checkpoint_dir.mkdir(parents=True)
    model_state = checkpoint_dir / "model_state.pt"
    model_state.write_bytes(b"test")

    assert resolve_checkpoint_file(checkpoint_dir) == model_state


def test_normalize_checkpoint_state_dict_accepts_pipeline_prefix() -> None:
    tensor = torch.ones(1)

    normalized = normalize_checkpoint_state_dict(
        {"state_dict": {"pipeline.layer.weight": tensor, "metadata": "ignored"}}
    )

    assert normalized == {"layer.weight": tensor}


def test_normalize_checkpoint_state_dict_rejects_prefix_collision() -> None:
    with pytest.raises(ValueError, match="ambiguous keys"):
        normalize_checkpoint_state_dict(
            {
                "state_dict": {
                    "layer.weight": torch.ones(1),
                    "pipeline.layer.weight": torch.zeros(1),
                }
            }
        )


def test_load_pipeline_checkpoint_notifies_supported_lifecycle(tmp_path: Path) -> None:
    class Pipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, 1))
            self.loaded_state_keys: frozenset[str] | None = None
            self.missing_state_keys: frozenset[str] | None = None

        def on_checkpoint_loaded(
            self,
            *,
            loaded_state_keys: frozenset[str],
            missing_state_keys: frozenset[str],
        ) -> None:
            self.loaded_state_keys = loaded_state_keys
            self.missing_state_keys = missing_state_keys

    pipeline = Pipeline()
    checkpoint_path = tmp_path / "model_state.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.full((1, 1), 3.0)}
        },
        checkpoint_path,
    )

    report = load_pipeline_checkpoint(pipeline, checkpoint_path)

    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert pipeline.loaded_state_keys == frozenset({"weight"})
    assert pipeline.missing_state_keys == frozenset()
    torch.testing.assert_close(
        pipeline.weight,
        torch.full((1, 1), 3.0),
    )


def test_load_pipeline_checkpoint_rejects_partial_state_by_default(
    tmp_path: Path,
) -> None:
    pipeline = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    original_state = {
        key: value.detach().clone() for key, value in pipeline.state_dict().items()
    }
    checkpoint_path = tmp_path / "model_state.pt"
    state = pipeline.state_dict()
    state["0.weight"] = torch.full_like(state["0.weight"], 7.0)
    del state["1.bias"]
    state["retired.weight"] = torch.ones(1)
    torch.save(state, checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        load_pipeline_checkpoint(pipeline, checkpoint_path)

    assert exc_info.value.report.missing_keys == ("1.bias",)
    assert exc_info.value.report.unexpected_keys == ("retired.weight",)
    for key, expected in original_state.items():
        torch.testing.assert_close(pipeline.state_dict()[key], expected, rtol=0, atol=0)


def test_load_pipeline_checkpoint_rejects_shape_mismatch_before_mutation(
    tmp_path: Path,
) -> None:
    pipeline = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    original_state = {
        key: value.detach().clone() for key, value in pipeline.state_dict().items()
    }
    checkpoint_path = tmp_path / "model_state.pt"
    state = pipeline.state_dict()
    state["0.bias"] = torch.full_like(state["0.bias"], 7.0)
    state["1.weight"] = torch.ones(1, 3)
    torch.save(state, checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        load_pipeline_checkpoint(pipeline, checkpoint_path)

    assert exc_info.value.report.shape_mismatches == (
        "1.weight (checkpoint=(1, 3), runtime=(1, 2))",
    )
    for key, expected in original_state.items():
        torch.testing.assert_close(pipeline.state_dict()[key], expected, rtol=0, atol=0)


def test_load_pipeline_checkpoint_allows_explicit_migration_diagnostic(
    tmp_path: Path,
) -> None:
    pipeline = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    checkpoint_path = tmp_path / "model_state.pt"
    state = pipeline.state_dict()
    del state["1.bias"]
    torch.save(state, checkpoint_path)

    report = load_pipeline_checkpoint(
        pipeline,
        checkpoint_path,
        compatibility=CheckpointCompatibilityPolicy.ALLOW_PARTIAL,
    )

    assert report.missing_keys == ("1.bias",)
    assert report.unexpected_keys == ()


def test_load_pipeline_checkpoint_allows_complete_checkpoint_superset(
    tmp_path: Path,
) -> None:
    pipeline = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "model_state.pt"
    state = pipeline.state_dict()
    state["retired_component.proj.weight"] = torch.ones(1)
    torch.save(state, checkpoint_path)

    report = load_pipeline_checkpoint(
        pipeline,
        checkpoint_path,
        compatibility=CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET,
    )

    assert report.missing_keys == ()
    assert report.unexpected_keys == ("retired_component.proj.weight",)


def test_complete_checkpoint_superset_policy_rejects_missing_runtime_state(
    tmp_path: Path,
) -> None:
    pipeline = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "model_state.pt"
    state = pipeline.state_dict()
    del state["bias"]
    state["retired_component.proj.weight"] = torch.ones(1)
    torch.save(state, checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        load_pipeline_checkpoint(
            pipeline,
            checkpoint_path,
            compatibility=(
                CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET
            ),
        )

    assert exc_info.value.report.missing_keys == ("bias",)
    assert exc_info.value.report.unexpected_keys == (
        "retired_component.proj.weight",
    )


def test_find_checkpoint_resolved_config_uses_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_123"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_state.pt").write_bytes(b"")
    (checkpoint_dir / "resolved_config.yaml").write_text("name: placeholder\n", encoding="utf-8")

    resolved_config_path = find_checkpoint_resolved_config(checkpoint_dir)

    assert resolved_config_path == (checkpoint_dir / "resolved_config.yaml").resolve()


def test_merge_runtime_config_from_checkpoint_keeps_data_sources_but_restores_runtime_contract(tmp_path: Path) -> None:
    base_config_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    base_config = load_experiment_config(base_config_path)
    base_config = replace(
        base_config,
        data=replace(
            base_config.data,
            local_root="/tmp/custom-libero-root",
            train_batch_size=99,
            val_batch_size=77,
        ),
    )
    checkpoint_dir = tmp_path / "checkpoint_step_400"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"")
    checkpoint_pretrained = tmp_path / "checkpoint-pretrained"
    checkpoint_pretrained.mkdir()
    checkpoint_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    checkpoint_config["backbone"]["pretrained_model_name_or_path"] = str(checkpoint_pretrained)
    checkpoint_config["policy_variant"]["attn_window"] = 31
    checkpoint_config["inference"]["action_num_inference_steps"] = 37
    (checkpoint_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(checkpoint_config, sort_keys=False),
        encoding="utf-8",
    )

    merged_config, resolved_config_path = merge_runtime_config_from_checkpoint(base_config, checkpoint_dir)

    assert resolved_config_path == (checkpoint_dir / "resolved_config.yaml").resolve()
    assert merged_config.data.local_root == "/tmp/custom-libero-root"
    assert merged_config.data.train_batch_size == 99
    assert merged_config.data.val_batch_size == 77
    assert str(merged_config.backbone.pretrained_model_name_or_path) == str(checkpoint_pretrained)
    assert int(merged_config.policy_variant.attn_window) == 31
    assert int(merged_config.inference.action_num_inference_steps) == 37


def test_merge_runtime_config_from_checkpoint_accepts_legacy_resolved_sample_fields(tmp_path: Path) -> None:
    base_config_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    base_config = load_experiment_config(base_config_path)
    checkpoint_dir = tmp_path / "checkpoint_step_1000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"")

    checkpoint_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    checkpoint_config["policy_variant"]["sequence_contract"] = "default"
    sample_config = checkpoint_config["data"]["sample_construction"]
    sample_config.update(
        {
            "mode": "hierarchical_fixed_segment",
            "sample_order_mode": "epoch_order",
            "randomize_geometry": False,
            "target_alignment": "next_after_context",
            "rollout_context_policy": "one_frame",
            "context_prefix_policy": "none",
            "context_prefix_frames": 0,
            "start_padding_frames": 0,
            "segment_frames": 128,
            "segment_min_frames": None,
            "segment_max_frames": None,
            "randomize_segment_length": False,
            "randomize_segment_start": False,
            "require_full_segment": False,
            "sample_weight_mode": "uniform",
            "sample_weight_length_power": 1.0,
        }
    )
    (checkpoint_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(checkpoint_config, sort_keys=False),
        encoding="utf-8",
    )

    merged_config, resolved_config_path = merge_runtime_config_from_checkpoint(base_config, checkpoint_dir)

    assert resolved_config_path == (checkpoint_dir / "resolved_config.yaml").resolve()
    assert merged_config.data.sample_construction.mode == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    assert merged_config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert merged_config.data.sample_construction.rollout_context_policy == RolloutContextPolicy.ONE_FRAME
    assert merged_config.data.sample_construction.segment_frames == 128


def test_merge_runtime_config_from_checkpoint_rehomes_nonportable_backbone_paths(tmp_path: Path) -> None:
    base_config_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    base_config = load_experiment_config(base_config_path)
    base_pretrained = tmp_path / "local_lingbot_va_base"
    base_pretrained.mkdir()
    base_config = replace(
        base_config,
        backbone=replace(base_config.backbone, pretrained_model_name_or_path=str(base_pretrained)),
    )

    checkpoint_dir = tmp_path / "checkpoint_step_1000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"")
    checkpoint_transformer = checkpoint_dir / "transformer"
    checkpoint_transformer.mkdir()
    (checkpoint_transformer / "config.json").write_text("{}", encoding="utf-8")

    checkpoint_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    checkpoint_config["backbone"]["pretrained_model_name_or_path"] = "/missing/remote/lingbot-va-base"
    checkpoint_config["backbone"]["transformer_subdir"] = "/missing/remote/transformer"
    (checkpoint_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(checkpoint_config, sort_keys=False),
        encoding="utf-8",
    )

    merged_config, _ = merge_runtime_config_from_checkpoint(base_config, checkpoint_dir)

    assert str(merged_config.backbone.pretrained_model_name_or_path) == str(base_pretrained.resolve())
    assert str(merged_config.backbone.transformer_subdir) == str(checkpoint_transformer.resolve())
