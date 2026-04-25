from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

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
    checkpoint_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    checkpoint_config["backbone"]["pretrained_model_name_or_path"] = "/tmp/checkpoint-pretrained"
    checkpoint_config["policy_variant"]["frame_chunk_size"] = 8
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
    assert str(merged_config.backbone.pretrained_model_name_or_path) == "/tmp/checkpoint-pretrained"
    assert int(merged_config.policy_variant.frame_chunk_size) == 8
    assert int(merged_config.inference.action_num_inference_steps) == 37
