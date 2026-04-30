"""Open-WAM research package.

The stable runtime boundary is:

``ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder``
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in RoboTwin's Python 3.10 env.
    import tomli as tomllib


def _resolve_version() -> str:
    try:
        return version("open-wam")
    except PackageNotFoundError:
        pass

    for root in Path(__file__).resolve().parents:
        pyproject_path = root / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        project = pyproject.get("project", {})
        if project.get("name") == "open-wam" and isinstance(project.get("version"), str):
            return project["version"]
    return "0+unknown"


__version__ = _resolve_version()

__all__ = ["__version__"]
