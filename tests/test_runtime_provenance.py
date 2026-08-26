from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from open_wam.runtime.provenance import (
    OPEN_WAM_PROVENANCE_SCHEMA_V1,
    _git_identity,
    collect_artifact_identity,
    collect_runtime_provenance,
)


def test_standard_provenance_hashes_configs_but_not_large_artifacts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    checkpoint_path = tmp_path / "model_state.pt"
    config_path.write_text("name: fixture\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")

    provenance = collect_runtime_provenance(
        config_path=config_path,
        resolved_config={"name": "fixture", "seed": 7},
        checkpoint_path=checkpoint_path,
        mode="standard",
        argv=("open-wam-eval", "--cfg", str(config_path)),
    )

    assert provenance["schema_version"] == OPEN_WAM_PROVENANCE_SCHEMA_V1
    assert provenance["mode"] == "standard"
    assert (
        provenance["config"]["sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert provenance["config"]["resolved_sha256"]
    assert provenance["checkpoint"]["sha256"] is None
    assert provenance["command_argv"][0] == "open-wam-eval"


def test_full_provenance_hashes_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model_state.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    provenance = collect_runtime_provenance(
        config_path=None,
        checkpoint_path=checkpoint_path,
        mode="full",
        argv=(),
    )

    assert (
        provenance["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    )


def test_directory_artifact_identity_hashes_metadata_but_not_large_shards(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "transformer"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    weights = checkpoint / "diffusion_pytorch_model.safetensors"
    config.write_text('{"hidden_size": 8}\n', encoding="utf-8")
    weights.write_bytes(b"w" * (1024 * 1024 + 1))

    standard = collect_artifact_identity(checkpoint)
    provenance = collect_runtime_provenance(
        config_path=None,
        checkpoint_path=checkpoint,
        argv=(),
    )
    standard_files = {item["relative_path"]: item for item in standard["files"]}

    assert standard["exists"] is True
    assert standard["kind"] == "directory"
    assert standard["sha256"] is None
    assert provenance["checkpoint"] == standard
    assert (
        standard_files["config.json"]["sha256"]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert standard_files[weights.name]["sha256"] is None

    full = collect_artifact_identity(checkpoint, mode="full")
    full_files = {item["relative_path"]: item for item in full["files"]}
    assert (
        full_files[weights.name]["sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )


def test_directory_artifact_identity_supports_in_root_file_symlinks(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (artifact / "metadata.json").symlink_to(outside)

    identity = collect_artifact_identity(artifact)

    assert identity["files"] == [
        {
            "path": str(outside.resolve()),
            "exists": True,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            "size_bytes": outside.stat().st_size,
            "mtime_ns": outside.stat().st_mtime_ns,
            "relative_path": "metadata.json",
            "symlink_target": str(outside.resolve()),
        }
    ]


def test_directory_artifact_identity_rejects_lexical_root_escape(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes its root"):
        collect_artifact_identity(
            artifact,
            relative_paths=("../outside.json",),
        )


def test_artifact_identity_treats_an_empty_path_as_unconfigured() -> None:
    assert collect_artifact_identity("") == {
        "path": None,
        "exists": False,
        "sha256": None,
    }


def test_git_identity_counts_untracked_sources_as_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    (repo / "src" / "open_wam").mkdir(parents=True)
    package_file = repo / "src" / "open_wam" / "runtime" / "provenance.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("# fixture\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "open-wam"\n',
        encoding="utf-8",
    )
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Open-WAM Test",
            "-c",
            "user.email=open-wam@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )

    assert _git_identity(package_file)["dirty"] is False
    (repo / "src" / "open_wam" / "untracked.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    assert _git_identity(package_file)["dirty"] is True


def test_git_identity_rejects_an_unrelated_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "src" / "open_wam").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "open-wam"\n',
        encoding="utf-8",
    )
    (checkout / ".git").mkdir()
    installed_file = (
        tmp_path / "site-packages" / "open_wam" / "runtime" / "provenance.py"
    )
    installed_file.parent.mkdir(parents=True)
    installed_file.write_text("# installed fixture\n", encoding="utf-8")
    monkeypatch.chdir(checkout)

    assert _git_identity(installed_file) == {
        "root": None,
        "commit": None,
        "dirty": None,
    }
