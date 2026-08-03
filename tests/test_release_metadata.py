from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from scripts.check_release_metadata import validate_project_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_public_project_metadata_is_complete() -> None:
    validate_project_metadata(_pyproject())


@pytest.mark.unit
@pytest.mark.parametrize(
    "field", ("license", "license-files", "authors", "urls", "classifiers", "keywords")
)
def test_public_project_metadata_rejects_missing_fields(field: str) -> None:
    pyproject = deepcopy(_pyproject())
    project = pyproject["project"]
    assert isinstance(project, dict)
    project.pop(field)

    with pytest.raises(ValueError, match="Project metadata"):
        validate_project_metadata(pyproject)
