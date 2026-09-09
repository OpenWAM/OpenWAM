from __future__ import annotations

from copy import deepcopy
from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import tomllib
import venv
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
import pytest
import yaml

from scripts.build_pypi_distributions import (
    INSTALLATION_ALIASES,
    alias_pyproject,
    validate_release_tag,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]


@pytest.mark.unit
def test_installation_names_are_distinct_and_cover_requested_spellings() -> None:
    assert {canonicalize_name(name) for name in (
        "openwam", "open_wam", "openwam_sdk", "open_wam_sdk"
    )} == {PROJECT["name"], *INSTALLATION_ALIASES}


@pytest.mark.unit
@pytest.mark.parametrize("name", INSTALLATION_ALIASES)
def test_alias_metadata_tracks_canonical_version_and_extras(name: str) -> None:
    original = deepcopy(PYPROJECT)
    project = alias_pyproject(PYPROJECT, name)["project"]
    assert project["name"] == name
    for key in ("version", "requires-python", "license", "authors", "urls", "classifiers"):
        assert project[key] == PROJECT[key]
    assert project["dependencies"] == [f"openwam=={PROJECT['version']}"]
    assert project["optional-dependencies"] == {
        extra: [f"openwam[{extra}]=={PROJECT['version']}"]
        for extra in PROJECT["optional-dependencies"]
    }
    assert "scripts" not in project
    assert PYPROJECT == original


@pytest.mark.unit
def test_aliases_follow_future_release_versions_and_new_extras() -> None:
    pyproject = deepcopy(PYPROJECT)
    pyproject["project"]["version"] = "0.2.0a1"
    pyproject["project"]["optional-dependencies"]["new-runtime"] = ["example>=1"]
    project = alias_pyproject(pyproject, "openwam-sdk")["project"]
    assert project["dependencies"] == ["openwam==0.2.0a1"]
    assert project["optional-dependencies"]["new-runtime"] == ["openwam[new-runtime]==0.2.0a1"]


@pytest.mark.unit
@pytest.mark.parametrize("ref", ("refs/heads/main", "refs/tags/v9.9.9", "v0.1.0", ""))
def test_publishing_rejects_nonmatching_release_refs(ref: str) -> None:
    with pytest.raises(ValueError, match="Publishing requires"):
        validate_release_tag(ref, PROJECT["version"])


@pytest.mark.unit
def test_publishing_accepts_matching_alpha_tag() -> None:
    validate_release_tag("refs/tags/v0.2.0a1", "0.2.0a1")


@pytest.mark.unit
def test_publishing_is_manual_production_only_and_separated_from_builds() -> None:
    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/publish-pypi.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert "github.repository == 'OpenWAM/OpenWAM'" in jobs["verify-ref"]["if"]
    assert "refs/tags/v" in jobs["verify-ref"]["if"]
    assert "git merge-base --is-ancestor HEAD origin/main" in str(jobs["verify-ref"])
    assert "--check-tag" in str(jobs["verify-ref"])
    assert jobs["checks"]["needs"] == "verify-ref"
    assert jobs["checks"]["uses"] == "./.github/workflows/release-checks.yml"
    assert jobs["checks"]["with"]["release"] == "true"
    publish = jobs["publish"]
    assert publish["needs"] == "checks"
    assert publish["environment"]["name"] == "${{ inputs.index }}"
    assert publish["permissions"] == {"id-token": "write"}
    assert all("checkout" not in step.get("uses", "") for step in publish["steps"])
    uploads = [step for step in publish["steps"] if "pypi-publish" in step.get("uses", "")]
    assert [step["with"]["packages-dir"] for step in uploads] == ["dist/core/", "dist/aliases/"]
    assert all(step["with"]["skip-existing"] == "true" for step in uploads)


@pytest.mark.unit
def test_release_checks_keep_dependency_audit_and_exercise_built_aliases() -> None:
    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/release-checks.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert "workflow_call" in workflow["on"]
    assert "pull_request" in workflow["on"]
    assert "check_dependency_audit.py" in str(workflow["jobs"]["dependency-audit"])
    package = workflow["jobs"]["package"]
    assert "build_pypi_distributions.py" in str(package)
    assert "test_pypi_distributions.py" in str(package)
    assert "OPENWAM_DISTRIBUTIONS_DIR" in str(package)


@pytest.fixture(scope="module")
def distributions() -> Path:
    root = os.environ.get("OPENWAM_DISTRIBUTIONS_DIR")
    if not root:
        pytest.skip("Set OPENWAM_DISTRIBUTIONS_DIR to exercise built release artifacts.")
    path = Path(root).resolve()
    assert path.is_dir()
    return path


@pytest.mark.integration
def test_built_alias_wheels_are_metadata_only_and_forward_all_extras(distributions: Path) -> None:
    assert len(list((distributions / "core").glob("*.whl"))) == 1
    assert len(list((distributions / "core").glob("*.tar.gz"))) == 1
    assert len(list((distributions / "aliases").glob("*.tar.gz"))) == len(INSTALLATION_ALIASES)
    wheels = list((distributions / "aliases").glob("*.whl"))
    assert len(wheels) == len(INSTALLATION_ALIASES)
    observed = set()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            metadata_name, = [name for name in members if name.endswith(".dist-info/METADATA")]
            prefix = metadata_name.removesuffix("METADATA")
            assert all(name.startswith(prefix) for name in members)
            assert not any(name.endswith("entry_points.txt") for name in members)
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
        name = canonicalize_name(metadata["Name"])
        assert name in INSTALLATION_ALIASES
        observed.add(name)
        assert metadata["Version"] == PROJECT["version"]
        assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(PROJECT["requires-python"])
        assert metadata["License-Expression"] == PROJECT["license"]
        assert set(metadata.get_all("License-File")) == {"LICENSE", "NOTICE"}
        assert set(metadata.get_all("Provides-Extra")) == set(PROJECT["optional-dependencies"])
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist")]
        assert len(requirements) == 1 + len(PROJECT["optional-dependencies"])
        for requirement in requirements:
            assert requirement.name == PROJECT["name"]
            assert str(requirement.specifier) == f"=={PROJECT['version']}"
        for extra in ("", *PROJECT["optional-dependencies"]):
            active = [r for r in requirements if r.marker is None or r.marker.evaluate({"extra": extra})]
            assert {frozenset(r.extras) for r in active} == (
                {frozenset(), frozenset({extra})} if extra else {frozenset()}
            )
    assert observed == set(INSTALLATION_ALIASES)


@pytest.mark.integration
@pytest.mark.parametrize("requested", (
    ("openwam",), ("open_wam",), ("openwam_sdk",), ("open_wam_sdk",),
    ("openwam", "open_wam", "openwam_sdk", "open_wam_sdk"),
))
def test_fresh_installs_share_one_implementation(
    distributions: Path, tmp_path: Path, requested: tuple[str, ...]
) -> None:
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    command = [str(python), "-m", "pip", "install", "--no-index", "--only-binary=:all:"]
    for directory in ("core", "aliases", "dependencies"):
        command.extend(("--find-links", str(distributions / directory)))
    subprocess.run(
        [*command, *(f"{name}=={PROJECT['version']}" for name in requested)],
        check=True, cwd=tmp_path,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=tmp_path)
    subprocess.run([
        str(python), "-I", "-c",
        "import importlib.util, importlib.metadata as m, pathlib, sys; "
        "import open_wam, open_wam.sdk.config, open_wam.sdk.results; "
        "from open_wam.sdk.config import resolve_config_reference; "
        "assert open_wam.__version__ == sys.argv[1]; "
        "assert pathlib.Path(open_wam.__file__).is_relative_to(sys.prefix); "
        "assert importlib.util.find_spec('torch') is None; "
        "assert all(m.version(n) == sys.argv[1] for n in sys.argv[2:]); "
        "assert resolve_config_reference('configs/examples/public_tiny_synthetic_contract.yaml').is_file()",
        PROJECT["version"], *requested,
    ], check=True, cwd=tmp_path)
    for command in PROJECT["scripts"]:
        subprocess.run([str(environment / "bin" / command), "--help"], check=True, cwd=tmp_path)
    if len(requested) > 1:
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", *INSTALLATION_ALIASES],
            check=True, cwd=tmp_path,
        )
        subprocess.run(
            [str(python), "-I", "-c", "import open_wam.sdk.config"], check=True, cwd=tmp_path
        )
