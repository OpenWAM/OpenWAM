from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from scripts.check_release_metadata import (
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
def test_release_requires_a_downloadable_licensed_model() -> None:
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
@pytest.mark.parametrize(
    "artifact",
    [
        {
            "architecture": "fixture",
            "download_url": "https://example.test/model.safetensors",
            "checksum": "a" * 64,
            "license": "MIT",
        },
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
    with pytest.raises(ValueError, match="non-fixture model artifact"):
        validate_public_model_artifacts({"artifacts": [artifact]})
