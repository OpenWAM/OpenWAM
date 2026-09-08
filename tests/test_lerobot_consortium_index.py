from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from open_wam.data import (
    build_lerobot_consortium_contract_catalog,
    build_lerobot_consortium_contract_catalog_from_inventory_rows,
    build_lerobot_consortium_inventory_row,
    infer_lerobot_consortium_source_group,
    load_lerobot_consortium_inventory_rows,
    load_lerobot_consortium_repo_targets,
    render_lerobot_consortium_inventory_markdown,
    write_lerobot_consortium_contract_catalog,
    write_lerobot_consortium_inventory_csv,
)


def test_consortium_index_facade_preserves_public_and_pickle_identity() -> None:
    from open_wam import data as public_data
    from open_wam.data import lerobot_consortium_index as index
    from open_wam.data import (
        lerobot_consortium_inventory_contracts as contracts,
    )
    from open_wam.data import lerobot_consortium_inventory_io as inventory_io
    from open_wam.data import lerobot_consortium_targets as targets

    moved_owners = {
        contracts: (
            "LeRobotConsortiumInventoryRow",
            "LeRobotConsortiumRepoTarget",
        ),
        inventory_io: (
            "load_lerobot_consortium_inventory_rows",
            "render_lerobot_consortium_inventory_markdown",
            "write_lerobot_consortium_inventory_csv",
            "write_lerobot_consortium_inventory_json",
            "write_lerobot_consortium_inventory_markdown",
        ),
        targets: (
            "infer_lerobot_consortium_source_group",
            "load_lerobot_consortium_repo_targets",
            "write_lerobot_consortium_repo_targets",
        ),
    }
    for owner, names in moved_owners.items():
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(index, name) is canonical
            assert getattr(public_data, name) is canonical
            legacy_global = (
                "copen_wam.data.lerobot_consortium_index\n" f"{name}\n."
            ).encode("ascii")
            assert pickle.loads(legacy_global) is canonical

    assert (
        index.build_lerobot_consortium_inventory_row.__globals__
        is index.__dict__
    )
    assert (
        index.build_lerobot_consortium_inventory_row.__globals__[
            "hf_hub_download"
        ]
        is index.hf_hub_download
    )

    expected_wildcard_names = {
        "Any",
        "HfApi",
        "Iterable",
        "LeRobotConsortiumInventoryRow",
        "LeRobotConsortiumRepoTarget",
        "Path",
        "ThreadPoolExecutor",
        "annotations",
        "as_completed",
        "asdict",
        "build_lerobot_consortium_inventory",
        "build_lerobot_consortium_inventory_row",
        "csv",
        "dataclass",
        "hf_hub_download",
        "infer_lerobot_consortium_source_group",
        "json",
        "load_lerobot_consortium_inventory_rows",
        "load_lerobot_consortium_repo_targets",
        "pq",
        "render_lerobot_consortium_inventory_markdown",
        "write_lerobot_consortium_inventory_csv",
        "write_lerobot_consortium_inventory_json",
        "write_lerobot_consortium_inventory_markdown",
        "write_lerobot_consortium_repo_targets",
    }
    wildcard_namespace: dict[str, object] = {}
    exec(
        "from open_wam.data.lerobot_consortium_index import *",
        wildcard_namespace,
    )
    assert set(index.__all__) == expected_wildcard_names
    assert set(wildcard_namespace) - {"__builtins__"} == (
        expected_wildcard_names
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _build_fake_hf_repo(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "fake_repo"
    _write_json(
        repo_root / "meta" / "info.json",
        {
            "fps": 30,
            "observation_fps": 30,
            "action_fps": 10,
            "total_episodes": 12,
            "total_frames": 3600,
            "total_tasks": 2,
            "robot_type": "aloha",
            "features": {
                "observation.images.cam_high": {"dtype": "video", "shape": [480, 640, 3]},
                "observation.images.cam_left": {"dtype": "video", "shape": [480, 640, 3]},
                "action": {"dtype": "float32", "shape": [14]},
                "observation.state": {"dtype": "float32", "shape": [14]},
                "task": {"dtype": "string"},
            },
        },
    )
    (repo_root / "README.md").write_text("Bimanual ALOHA real-world dataset.", encoding="utf-8")
    task_table = pa.Table.from_pylist(
        [
            {"task_index": 0, "task": "fold towel"},
            {"task_index": 1, "task": "place towel"},
        ]
    )
    task_parquet = repo_root / "meta" / "tasks.parquet"
    task_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(task_table, task_parquet)
    _write_json(repo_root / "meta" / "temporal_proportions_dense.json", {"dummy": 1.0})
    return {
        "repo_root": repo_root,
        "info": repo_root / "meta" / "info.json",
        "readme": repo_root / "README.md",
        "tasks": task_parquet,
        "dense": repo_root / "meta" / "temporal_proportions_dense.json",
    }


def test_load_repo_targets_from_text_and_infers_source_groups(tmp_path: Path) -> None:
    repo_list = tmp_path / "repos.txt"
    repo_list.write_text(
        "\n".join(
            [
                "# comment",
                "lerobot/aloha_static_towel",
                "DaivdYuan/hub-flip-bagel-lerobot",
                "custom_group,other-org/custom-set",
            ]
        ),
        encoding="utf-8",
    )
    targets = load_lerobot_consortium_repo_targets(repo_list, default_source_group="manual")
    assert tuple((item.source_group, item.repo_id) for item in targets) == (
        ("official_lerobot", "lerobot/aloha_static_towel"),
        ("nmotion_current", "DaivdYuan/hub-flip-bagel-lerobot"),
        ("custom_group", "other-org/custom-set"),
    )
    assert infer_lerobot_consortium_source_group("someone/foo", default_source_group="manual") == "manual"


def test_domain_and_embodiment_inference_accepts_generic_metadata() -> None:
    from open_wam.data import lerobot_consortium_index as index

    assert index._infer_domain_type("example/collection", "Real-world robot data") == "real"
    assert index._infer_domain_type("example/collection", "Recorded in simulation") == "sim"
    assert index._infer_domain_type("example/collection", None) == "unknown"
    embodiment, confidence, _ = index._infer_embodiment(
        repo_id="example/collection",
        robot_type=None,
        action_dim=16,
        state_dim=16,
        readme_text="Dexterous hand manipulation",
    )
    assert (embodiment, confidence) == ("dexterous_hand", "high")


def test_load_repo_targets_from_csv(tmp_path: Path) -> None:
    repo_csv = tmp_path / "repos.csv"
    with repo_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repo_id", "source_group"])
        writer.writeheader()
        writer.writerow({"repo_id": "lerobot/aloha_static_towel", "source_group": ""})
        writer.writerow({"repo_id": "other-org/custom", "source_group": "custom"})
    targets = load_lerobot_consortium_repo_targets(repo_csv, default_source_group="manual")
    assert tuple((item.source_group, item.repo_id) for item in targets) == (
        ("official_lerobot", "lerobot/aloha_static_towel"),
        ("custom", "other-org/custom"),
    )


def test_load_repo_targets_dedupes_by_repo_id_and_prefers_explicit_source_group(tmp_path: Path) -> None:
    repo_csv = tmp_path / "repos.csv"
    with repo_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repo_id", "source_group"])
        writer.writeheader()
        writer.writerow({"repo_id": "lerobot/aloha_static_towel", "source_group": ""})
        writer.writerow({"repo_id": "lerobot/aloha_static_towel", "source_group": "manual_override"})
        writer.writerow({"repo_id": "other-org/custom", "source_group": "first"})
        writer.writerow({"repo_id": "other-org/custom", "source_group": "second"})
    targets = load_lerobot_consortium_repo_targets(repo_csv, default_source_group="manual")
    assert tuple((item.source_group, item.repo_id) for item in targets) == (
        ("manual_override", "lerobot/aloha_static_towel"),
        ("second", "other-org/custom"),
    )


def test_load_repo_targets_from_text_dedupes_by_repo_id(tmp_path: Path) -> None:
    repo_list = tmp_path / "repos.txt"
    repo_list.write_text(
        "\n".join(
            [
                "lerobot/aloha_static_towel",
                "custom_group,lerobot/aloha_static_towel",
                "first_group,other-org/custom",
                "second_group,other-org/custom",
            ]
        ),
        encoding="utf-8",
    )
    targets = load_lerobot_consortium_repo_targets(repo_list, default_source_group="manual")
    assert tuple((item.source_group, item.repo_id) for item in targets) == (
        ("custom_group", "lerobot/aloha_static_towel"),
        ("second_group", "other-org/custom"),
    )


def test_build_inventory_row_from_live_like_repo_metadata(monkeypatch, tmp_path: Path) -> None:
    files = _build_fake_hf_repo(tmp_path)
    repo_id = "lerobot/aloha_static_towel"

    class _FakeApi:
        def repo_info(self, *, repo_id: str, repo_type: str, token: str | None = None, files_metadata: bool = False):
            assert repo_type == "dataset"
            assert files_metadata is True
            return SimpleNamespace(
                private=False,
                siblings=[
                    SimpleNamespace(rfilename="README.md", size=500),
                    SimpleNamespace(rfilename="meta/info.json", size=600),
                    SimpleNamespace(rfilename="meta/tasks.parquet", size=700),
                    SimpleNamespace(rfilename="meta/temporal_proportions_dense.json", size=200),
                    SimpleNamespace(rfilename="videos/observation.images.cam_high/chunk-000/file-000.mp4", size=4 * 1024 * 1024),
                    SimpleNamespace(rfilename="data/chunk-000/episode_000000.parquet", size=512 * 1024),
                ],
            )

    def _fake_hf_hub_download(*, repo_id: str, repo_type: str, filename: str, token: str | None = None):
        assert repo_id == "lerobot/aloha_static_towel"
        assert repo_type == "dataset"
        mapping = {
            "meta/info.json": files["info"],
            "README.md": files["readme"],
            "meta/tasks.parquet": files["tasks"],
            "meta/temporal_proportions_dense.json": files["dense"],
        }
        return str(mapping[filename])

    monkeypatch.setattr("open_wam.data.lerobot_consortium_index.hf_hub_download", _fake_hf_hub_download)

    row = build_lerobot_consortium_inventory_row(
        api=_FakeApi(),
        repo_id=repo_id,
        source_group="official_lerobot",
    )

    assert row.repo_id == repo_id
    assert row.domain_type == "real"
    assert row.visual_stream_count == 2
    assert row.visual_stream_keys == "observation.images.cam_high | observation.images.cam_left"
    assert row.action_dim == 14
    assert row.state_dim == 14
    assert row.embodiment_type == "dual_arm"
    assert row.task_text_present is True
    assert row.task_text_examples == "fold towel | place towel"
    assert row.temporal_dense_present is True
    assert row.generation_error is None


def test_load_inventory_rows_and_build_contract_catalog(tmp_path: Path) -> None:
    inventory_csv = tmp_path / "inventory.csv"
    inventory_csv.write_text(
        "source_group,repo_id,private,domain_type,total_size_mb,data_size_mb,video_size_mb,total_episodes,total_frames,total_tasks,total_hours,avg_seconds_per_episode,fps,observation_fps,action_fps,robot_type,embodiment_type,embodiment_confidence,embodiment_reason,action_dim,action_shape,state_dim,state_shape,visual_stream_count,visual_stream_keys,visual_dimensions,visual_dtypes,text_annotation_extent,task_text_present,task_text_count,task_text_examples,temporal_dense_present,temporal_sparse_present,language_feature_keys,readme_url,dataset_url,generation_error\n"
        "official_lerobot,repo_a,False,real,120.0,20.0,100.0,10,1000,1,0.1,36.0,30.0,30.0,10.0,aloha,dual_arm,high,reason,14,14,14,14,2,cam0 | cam1,cam0:224x224x3 | cam1:640x480x3,video | video,multi_task_instruction,True,2,task a | task b,False,True,language,readme,dataset,\n",
        encoding="utf-8",
    )

    rows = load_lerobot_consortium_inventory_rows(inventory_csv)
    assert len(rows) == 1
    assert rows[0].repo_id == "repo_a"

    contracts = build_lerobot_consortium_contract_catalog(inventory_csv)
    assert contracts["contract_version"] == "hf_dataset_contracts.v1"
    assert contracts["dataset_count"] == 1
    dataset = contracts["datasets"][0]
    assert dataset["modalities"]["visual_stream_count"] == 2
    assert dataset["modalities"]["visual_streams"][1]["width"] == 480
    assert dataset["text_annotations"]["task_text_examples"] == ["task a", "task b"]
    assert dataset["video_contract"]["episode_routing_available"] is True
    assert dataset["video_contract"]["manifest_total_stream_rows"] == 20


def test_inventory_markdown_and_contract_writers_round_trip(tmp_path: Path) -> None:
    rows = [
        # Use the public dataclass so the round-trip checks stay stable.
        load_lerobot_consortium_inventory_rows(
            _write_inventory_fixture(
                tmp_path / "fixture.csv",
                [
                    {
                        "source_group": "official_lerobot",
                        "repo_id": "lerobot/example",
                        "private": "False",
                        "domain_type": "real",
                        "total_size_mb": "1024.0",
                        "data_size_mb": "100.0",
                        "video_size_mb": "924.0",
                        "total_episodes": "10",
                        "total_frames": "300",
                        "total_tasks": "1",
                        "total_hours": "0.01",
                        "avg_seconds_per_episode": "3.0",
                        "fps": "30.0",
                        "observation_fps": "30.0",
                        "action_fps": "30.0",
                        "robot_type": "aloha",
                        "embodiment_type": "dual_arm",
                        "embodiment_confidence": "high",
                        "embodiment_reason": "test",
                        "action_dim": "14",
                        "action_shape": "14",
                        "state_dim": "14",
                        "state_shape": "14",
                        "visual_stream_count": "2",
                        "visual_stream_keys": "cam_a | cam_b",
                        "visual_dimensions": "cam_a:224x224x3 | cam_b:224x224x3",
                        "visual_dtypes": "video | video",
                        "text_annotation_extent": "single_task_instruction",
                        "task_text_present": "True",
                        "task_text_count": "1",
                        "task_text_examples": "pick up towel",
                        "temporal_dense_present": "False",
                        "temporal_sparse_present": "False",
                        "language_feature_keys": "task",
                        "readme_url": "https://huggingface.co/datasets/lerobot/example/blob/main/README.md",
                        "dataset_url": "https://huggingface.co/datasets/lerobot/example",
                        "generation_error": "",
                    }
                ],
            )
        )[0]
    ]
    md = render_lerobot_consortium_inventory_markdown(rows)
    assert "official_lerobot: `1` repos, `10` episodes" in md
    assert "| official_lerobot | lerobot/example | real | 1.00 | 10 | 0.01 | 30.0 | 30.0 |" in md

    out_csv = tmp_path / "roundtrip.csv"
    write_lerobot_consortium_inventory_csv(out_csv, rows)
    reloaded = load_lerobot_consortium_inventory_rows(out_csv)
    assert reloaded[0].repo_id == "lerobot/example"

    contracts = build_lerobot_consortium_contract_catalog_from_inventory_rows(reloaded)
    out_json = tmp_path / "contracts.json"
    write_lerobot_consortium_contract_catalog(out_json, contracts)
    assert json.loads(out_json.read_text())["dataset_count"] == 1


def _write_inventory_fixture(path: Path, records: list[dict[str, str]]) -> Path:
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path
