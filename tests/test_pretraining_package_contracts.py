"""Preparation entry points, explicit sources, and generic consumer boundaries."""

import ast
import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.artifacts.object_cache import ObjectCache
from open_wam.artifacts.resolver import ArtifactResolver
from open_wam.configs.asset_cache import ArtifactCacheConfig, PromptCacheConfig
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.data.preparation.multiview.prepare import make_plan


def test_multi_view_plan_is_source_and_camera_name_agnostic():
    row = dict(
        source_id="unseen_dataset",
        repo_id="episode_group",
        episode_index=0,
        physical_episode_key="stable_identity",
        cameras=["x", "y", "z"],
        task="move",
        text_provenance="native",
        text_status="native",
        native_end_frames=dict(x=18, y=18, z=18),
        raw_source=dict(type="lerobot"),
    )
    plan, _ = make_plan(row, cameras=["z", "x"], pair_orientation="horizontal")
    assert plan["cameras"] == ["z", "x"]
    assert plan["physical_episode_key"] == row["physical_episode_key"]
    assert plan["camera_rotations_degrees"] == {}
    with pytest.raises(ValueError, match="distinct"):
        make_plan(row, cameras=["z", "z"])


def test_explicit_catalogs_are_isolated_and_survive_worker_serialization(
    tmp_path, monkeypatch
):
    from open_wam.artifacts import resolver as module
    from tests.test_pretraining_storage_roundtrip import MemoryS3

    remote = MemoryS3()

    def cache(root, max_bytes, min_free_bytes):
        return ObjectCache(
            root, max_bytes, min_free_bytes, client_factory=lambda: remote
        )

    monkeypatch.setattr(module, "ObjectCache", cache)
    monkeypatch.setenv("OPENWAM_PRETRAIN_CATALOG_SPEC", "/invalid/ambient/catalog")
    logical = tmp_path / "logical.pth"
    resolvers = []
    for i, contents in enumerate((b"first tensor", b"second tensor")):
        tensor_spec = dict(
            bucket="test",
            key=f"tensor{i}",
            bytes=len(contents),
            sha256=hashlib.sha256(contents).hexdigest(),
        )
        remote.objects["test", f"tensor{i}"] = contents
        catalog = tmp_path / f"catalog{i}.sqlite"
        with sqlite3.connect(catalog) as db:
            db.execute("CREATE TABLE paths (path TEXT PRIMARY KEY, spec TEXT)")
            db.execute(
                "INSERT INTO paths VALUES (?, ?)",
                (str(logical), json.dumps(tensor_spec)),
            )
        data = catalog.read_bytes()
        remote.objects["test", f"catalog{i}"] = data
        spec = tmp_path / f"catalog{i}.json"
        spec.write_text(
            json.dumps(
                dict(
                    bucket="test",
                    key=f"catalog{i}",
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
        )
        resolver = ArtifactResolver(
            ArtifactCacheConfig(str(spec), str(tmp_path / f"cache{i}"), 100000, 0)
        )
        assert resolver.contains(logical)
        with resolver.materialize(logical) as materialized:
            assert materialized.read_bytes() == contents
        restored = pickle.loads(pickle.dumps(resolver))
        with restored.materialize(logical) as materialized:
            assert materialized.read_bytes() == contents
        restored.close()
        resolvers.append(resolver)
    local = tmp_path / "unlisted.txt"
    local.write_text("local file")
    for resolver in resolvers:
        with resolver.materialize(local) as path:
            assert path == local
        resolver.close()
    assert not ArtifactResolver().contains(logical)


def test_prompt_encoder_cli_produces_a_cache_readable_by_model_assets(
    tmp_path, monkeypatch
):
    from open_wam.cli import encode_prompt_cache
    from open_wam.configs import loader
    from open_wam.models.visual_tower.prompt_cache import OfflinePromptCache
    from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "custom_dataset.csv").write_text("task\nmove\n")
    assets_root = tmp_path / "assets"
    for name in ("text_encoder", "tokenizer"):
        folder = assets_root / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
    config = SharedVideoTransformerConfig(
        text_dim=4, max_text_tokens=8, prompt_cache=PromptCacheConfig("/old/cache")
    )
    monkeypatch.setattr(
        loader, "load_experiment_config", lambda _: SimpleNamespace(backbone=config)
    )
    seen = []

    def load(backbone):
        assert backbone.prompt_cache is None
        assert backbone.load_text_conditioning
        seen.append(backbone)
        return SimpleNamespace(
            has_text_encoder=True,
            text_embedding_cache={},
            encode_prompts=lambda texts, **kw: torch.ones(
                len(texts), 8, 4, dtype=torch.bfloat16
            ),
            tokenizer=lambda texts, **kw: SimpleNamespace(
                attention_mask=torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]] * len(texts))
            ),
        )

    monkeypatch.setattr(LingbotReferenceAssets, "maybe_load", load)
    out = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encode_prompt_cache",
            "--cfg",
            "unused.yaml",
            "--assets",
            str(assets_root),
            "--manifests",
            str(manifests),
            "--out",
            str(out),
            "--allow-subset",
            "--device",
            "cpu",
        ],
    )
    encode_prompt_cache.main()
    cache = OfflinePromptCache(out, max_text_tokens=8, text_dim=4)
    result = cache.encode_prompts(
        ("", "move"), device=torch.device("cpu"), dtype=torch.float32
    )
    assert len(seen) == 1
    assert torch.equal(result[:, :2], torch.ones(2, 2, 4))
    assert not result[:, 2:].any()


def test_preparation_package_never_imports_checkout_scripts_or_mutates_module_search_path():
    root = Path(__file__).parents[1] / "src/open_wam"
    paths = [
        *root.joinpath("data/preparation").rglob("*.py"),
        *root.joinpath("artifacts").rglob("*.py"),
        *root.joinpath("cli").glob("*pretraining*.py"),
    ]
    for path in paths:
        code = path.read_text()
        assert "sys.path" not in code, path
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("scripts"), path
