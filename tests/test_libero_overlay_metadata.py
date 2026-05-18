from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str):
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ABS_JOINT_MODULE = _load_script_module("materialize_libero_absolute_joint_lerobot_overlay.py")
EEF6D_MODULE = _load_script_module("materialize_libero_integrated_eef6d_overlay.py")


def _write_overlay_meta(root: Path, episode_indices: list[int]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 999,
                "features": {},
            }
        ),
        encoding="utf-8",
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode_index in episode_indices:
            handle.write(json.dumps({"episode_index": episode_index, "length": 4}) + "\n")


def test_absolute_joint_overlay_reindexes_episode_metadata_contiguously(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_overlay_meta(source_root, [3, 9])

    ABS_JOINT_MODULE._copy_meta(
        dataset_root=source_root,
        output_root=output_root,
        episode_index_map={3: 0, 9: 1},
    )
    info_path = output_root / "meta" / "info.json"
    ABS_JOINT_MODULE._update_info_features(info_path, action_dim=8)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_rows = [
        json.loads(line)
        for line in (output_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["episode_index"] for row in episode_rows] == [0, 1]
    assert [row["source_dataset_episode_index"] for row in episode_rows] == [3, 9]
    assert info["episode_index_policy"] == "contiguous_reindexed"
    assert info["selected_episode_count"] == 2
    assert info["episode_indices"] == [0, 1]
    assert info["total_episodes"] == 2
    assert "total_episodes_semantics" not in info
    assert "absolute_joint_action" in info["features"]


@pytest.mark.parametrize("module", [ABS_JOINT_MODULE, EEF6D_MODULE])
def test_overlay_materializers_refuse_existing_output_root_without_overwrite(module, tmp_path: Path) -> None:
    output_root = tmp_path / "existing_output"
    output_root.mkdir()
    (output_root / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output root already exists"):
        module._prepare_output_root(output_root, overwrite=False)

    assert (output_root / "sentinel.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("module", [ABS_JOINT_MODULE, EEF6D_MODULE])
def test_overlay_materializers_overwrite_existing_output_root(module, tmp_path: Path) -> None:
    output_root = tmp_path / "existing_output"
    output_root.mkdir()
    (output_root / "sentinel.txt").write_text("remove", encoding="utf-8")

    module._prepare_output_root(output_root, overwrite=True)

    assert output_root.is_dir()
    assert not (output_root / "sentinel.txt").exists()


def test_integrated_eef6d_overlay_reindexes_episode_assets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_overlay_meta(source_root, [3])
    source_asset = source_root / "latents" / "chunk-000" / "cam" / "episode_000003_0_4.pth"
    source_asset.parent.mkdir(parents=True)
    source_asset.write_text("latent", encoding="utf-8")

    EEF6D_MODULE._copy_meta(
        dataset_root=source_root,
        output_root=output_root,
        episode_index_map={3: 0},
    )
    EEF6D_MODULE._materialize_reindexed_episode_assets(
        source_root / "latents",
        output_root / "latents",
        episode_index_map={3: 0},
        chunk_size=1000,
    )
    info_path = output_root / "meta" / "info.json"

    EEF6D_MODULE._update_info_features(info_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_rows = [
        json.loads(line)
        for line in (output_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["episode_index"] for row in episode_rows] == [0]
    assert [row["source_dataset_episode_index"] for row in episode_rows] == [3]
    assert (output_root / "latents" / "chunk-000" / "cam" / "episode_000000_0_4.pth").is_symlink()
    assert info["episode_index_policy"] == "contiguous_reindexed"
    assert info["selected_episode_count"] == 1
    assert info["episode_indices"] == [0]
    assert info["total_episodes"] == 1
    assert "total_episodes_semantics" not in info
    assert "integrated_eef6d_action" in info["features"]


def test_integrated_eef6d_replay_loads_reindexed_overlay_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_root = tmp_path / "overlay"
    meta = output_root / "meta"
    data = output_root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}),
        encoding="utf-8",
    )
    (data / "episode_000000.parquet").write_text("target", encoding="utf-8")
    loaded_episode_indices: list[int] = []

    monkeypatch.setattr(EEF6D_MODULE, "load_experiment_config", lambda _path: SimpleNamespace(data=object()))

    class FakeAdapter:
        def __init__(self, *, config) -> None:
            self.config = config

        def reset(self, _spec) -> None:
            return None

        def set_integrated_eef6d_previous_target_from_state(self, _state) -> None:
            return None

        def action_from_model_action(self, target, *, data_config) -> object:
            return target

        def step(self, _action):
            return type("Step", (), {"success": True, "done": True})()

        def close(self) -> None:
            return None

    def fake_load_targets(_output_root: Path, *, dataset_info: dict[str, object], episode_index: int):
        loaded_episode_indices.append(int(episode_index))
        assert (_output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet").is_file()
        return [[0.0] * 10]

    monkeypatch.setattr(EEF6D_MODULE, "LiberoBenchmarkAdapter", FakeAdapter)
    monkeypatch.setattr(EEF6D_MODULE, "_load_overlay_targets", fake_load_targets)
    monkeypatch.setattr(EEF6D_MODULE, "_load_source_initial_state", lambda *_args, **_kwargs: [0.0] * 8)
    row = {
        "dataset_episode_index": 3,
        "upstream_task_id": 0,
        "resolved_init_state_index": 0,
        "attempts": [{"init_state_index": 0, "reset_seed": 1}],
    }

    summary = EEF6D_MODULE._run_adapter_replay_validation(
        [row],
        dataset_root=tmp_path / "source",
        dataset_info={"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"},
        output_root=output_root,
        episode_index_map={3: 0},
        rollout_config=tmp_path / "config.yaml",
        position_scale=1.0,
        rotation_scale=1.0,
        camera_height=8,
        camera_width=8,
        replay_env_backend="control",
        replay_use_camera_obs=False,
        replay_has_offscreen_renderer=False,
        resume=False,
    )

    report_rows = [
        json.loads(line)
        for line in (meta / "integrated_eef6d_adapter_replay_status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert loaded_episode_indices == [0]
    assert summary["success_count"] == 1
    assert report_rows[0]["dataset_episode_index"] == 0
    assert report_rows[0]["source_dataset_episode_index"] == 3


def test_integrated_eef6d_replay_report_loader_prefers_source_episode_ids(tmp_path: Path) -> None:
    report_path = tmp_path / "integrated_eef6d_adapter_replay_status.jsonl"
    report_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_episode_index": 0,
                        "source_dataset_episode_index": 3,
                        "reused_report_dataset_episode_index": 99,
                        "success": True,
                    }
                ),
                json.dumps(
                    {
                        "dataset_episode_index": 1,
                        "source_dataset_episode_index": 9,
                        "success": False,
                    }
                ),
                json.dumps(
                    {
                        "dataset_episode_index": 2,
                        "success": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = EEF6D_MODULE._load_success_records_from_adapter_report(report_path)

    assert [record["source_dataset_episode_index"] for record in records] == [2, 3]
    assert records[0]["reused_report_dataset_episode_index"] == 2
    assert records[1]["reused_report_dataset_episode_index"] == 0


def test_integrated_eef6d_attached_replay_report_rewrites_to_new_overlay_namespace(tmp_path: Path) -> None:
    output_root = tmp_path / "overlay"
    source_report = tmp_path / "old_report.jsonl"
    records = (
        {
            "dataset_episode_index": 0,
            "source_dataset_episode_index": 3,
            "reused_report_dataset_episode_index": 99,
            "success": True,
        },
        {
            "dataset_episode_index": 2,
            "source_dataset_episode_index": 9,
            "success": True,
        },
    )

    EEF6D_MODULE._write_attached_adapter_replay_report(
        output_root=output_root,
        records=records,
        episode_index_map={3: 0, 9: 1},
        source_report=source_report,
    )

    report_rows = [
        json.loads(line)
        for line in (output_root / "meta" / "integrated_eef6d_adapter_replay_status.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["dataset_episode_index"] for row in report_rows] == [0, 1]
    assert [row["source_dataset_episode_index"] for row in report_rows] == [3, 9]
    assert [row["reused_report_dataset_episode_index"] for row in report_rows] == [0, 2]
    assert all(row["reused_from_report"] == str(source_report) for row in report_rows)
