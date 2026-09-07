from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import tarfile
import tomllib

import pytest

from scripts.check_release_metadata import (
    private_distribution_violations,
    validate_public_consortium_snapshot,
    validate_project_metadata,
    validate_public_model_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_public_project_metadata_is_complete() -> None:
    validate_project_metadata(_pyproject())


@pytest.mark.unit
def test_public_consortium_snapshot_has_no_private_repositories() -> None:
    validate_public_consortium_snapshot(REPO_ROOT)


@pytest.mark.unit
def test_model_extras_exclude_incompatible_wan_vae_normalization() -> None:
    optional_dependencies = _pyproject()["project"]["optional-dependencies"]
    expected = "diffusers>=0.35.0,<0.38"

    for extra in ("torch", "train", "eval", "sim", "full"):
        requirements = optional_dependencies[extra]
        assert [item for item in requirements if item.startswith("diffusers")] == [expected]


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    (
        "license",
        "license-files",
        "authors",
        "urls",
        "classifiers",
        "keywords",
        "requires-python",
    ),
)
def test_public_project_metadata_rejects_missing_fields(field: str) -> None:
    pyproject = deepcopy(_pyproject())
    project = pyproject["project"]
    assert isinstance(project, dict)
    project.pop(field)

    with pytest.raises(ValueError, match="Project metadata"):
        validate_project_metadata(pyproject)


@pytest.mark.unit
def test_release_accepts_a_downloadable_licensed_model() -> None:
    manifest = {
        "artifacts": [
            {
                "architecture": "dual_expert",
                "download_url": "https://example.test/model.safetensors",
                "checksum": "sha256:" + "a" * 64,
                "license": "Apache-2.0",
            }
        ]
    }

    validate_public_model_artifacts(manifest)


@pytest.mark.unit
def test_release_allows_unpublished_model_artifacts() -> None:
    validate_public_model_artifacts(
        {
            "artifacts": [
                {
                    "architecture": "dual_expert",
                    "download_url": None,
                    "checksum": None,
                    "license": None,
                }
            ]
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact",
    [
        {
            "architecture": "dual_expert",
            "download_url": None,
            "checksum": "a" * 64,
            "license": "MIT",
        },
        {
            "architecture": "dual_expert",
            "download_url": "https://example.test/model.safetensors",
            "checksum": "pending",
            "license": "MIT",
        },
        {
            "architecture": "dual_expert",
            "download_url": "https://example.test/model.safetensors",
            "checksum": "a" * 64,
            "license": None,
        },
    ],
)
def test_release_rejects_incomplete_model_artifacts(artifact: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="public model artifact"):
        validate_public_model_artifacts({"artifacts": [artifact]})


@pytest.mark.unit
def test_built_distribution_check_reads_archive_contents(tmp_path: Path) -> None:
    import zipfile

    source = tmp_path / "open_wam-0.1.0.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        payload = b"public source"
        member = tarfile.TarInfo("open_wam-0.1.0/README.md")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    wheel = tmp_path / "open_wam-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("open_wam/example.py", 'root = "/home/private-user/data"')

    violations = private_distribution_violations(tmp_path)

    assert violations == (f"{wheel.name}:open_wam/example.py: /home/",)
