from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from open_wam.configs.enums import RolloutContextPolicy, SampleTargetAlignment, WindowSamplingMode
from open_wam.utils import (
    find_checkpoint_resolved_config,
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_find_checkpoint_resolved_config_uses_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_123"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_state.pt").write_bytes(b"")
    (checkpoint_dir / "resolved_config.yaml").write_text("name: placeholder\n", encoding="utf-8")

    resolved_config_path = find_checkpoint_resolved_config(checkpoint_dir)

    assert resolved_config_path == (checkpoint_dir / "resolved_config.yaml").resolve()


def test_merge_runtime_config_from_checkpoint_keeps_data_sources_but_restores_runtime_contract(tmp_path: Path) -> None:
    base_config_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
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
    base_config_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_full_segment_non_joint_action_only.yaml"
    base_config = load_experiment_config(base_config_path)
    checkpoint_dir = tmp_path / "checkpoint_step_1000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"")

    checkpoint_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    sample_config = checkpoint_config["data"]["sample_construction"]
    sample_config.update(
        {
            "mode": "hierarchical_fixed_segment",
            "target_alignment": "next_after_context",
            "rollout_context_policy": "one_frame",
            "context_prefix_policy": "none",
            "context_prefix_frames": 0,
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
    base_config_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_full_segment_non_joint_action_only.yaml"
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
