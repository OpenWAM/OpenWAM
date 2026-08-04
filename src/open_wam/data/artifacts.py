from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DatasetArtifactKind(str, Enum):
    """Filesystem shape required by a dataset adapter."""

    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class DatasetArtifactRequirement:
    """One adapter-owned filesystem dependency checked before model creation."""

    name: str
    path: str | Path | None
    kind: DatasetArtifactKind
    required: bool
    config_path: str
    purpose: str
    remediation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DatasetArtifactKind(self.kind))
        if not self.name.strip():
            raise ValueError("Dataset artifact name must be non-empty.")
        if not self.config_path.strip():
            raise ValueError("Dataset artifact config_path must be non-empty.")
        if not self.purpose.strip():
            raise ValueError("Dataset artifact purpose must be non-empty.")
        if not isinstance(self.required, bool):
            raise TypeError("Dataset artifact required must be a boolean.")

    @property
    def resolved_path(self) -> Path | None:
        if self.path is None:
            return None
        if isinstance(self.path, str) and not self.path.strip():
            return None
        return Path(self.path).expanduser()


@dataclass(frozen=True)
class DatasetArtifactStatus:
    """Result of checking one artifact requirement."""

    requirement: DatasetArtifactRequirement
    available: bool

    def to_dict(self) -> dict[str, str | bool | None]:
        path = self.requirement.resolved_path
        return {
            "name": self.requirement.name,
            "path": None if path is None else str(path),
            "kind": self.requirement.kind.value,
            "required": self.requirement.required,
            "available": self.available,
            "config_path": self.requirement.config_path,
            "purpose": self.requirement.purpose,
            "remediation": self.requirement.remediation,
        }


class DatasetArtifactPreflightError(FileNotFoundError):
    """Raised when an adapter's required filesystem contract is unsatisfied."""


def check_dataset_artifacts(
    requirements: Sequence[DatasetArtifactRequirement],
) -> tuple[DatasetArtifactStatus, ...]:
    statuses: list[DatasetArtifactStatus] = []
    for requirement in requirements:
        path = requirement.resolved_path
        available = False
        if path is not None:
            available = (
                path.is_file()
                if requirement.kind is DatasetArtifactKind.FILE
                else path.is_dir()
            )
        statuses.append(
            DatasetArtifactStatus(
                requirement=requirement,
                available=available,
            )
        )
    return tuple(statuses)


def require_dataset_artifacts(
    requirements: Sequence[DatasetArtifactRequirement],
    *,
    dataset_type: str,
) -> tuple[DatasetArtifactStatus, ...]:
    statuses = check_dataset_artifacts(requirements)
    missing = [
        status
        for status in statuses
        if status.requirement.required and not status.available
    ]
    if not missing:
        return statuses

    details: list[str] = []
    for status in missing:
        requirement = status.requirement
        path = requirement.resolved_path
        location = "not configured" if path is None else str(path)
        detail = (
            f"- {requirement.name} ({requirement.kind.value}) at {location}; "
            f"configured by `{requirement.config_path}`; {requirement.purpose}"
        )
        if requirement.remediation:
            detail += f". {requirement.remediation}"
        details.append(detail)
    raise DatasetArtifactPreflightError(
        f"Dataset artifact preflight failed for dataset_type={dataset_type!r}:\n"
        + "\n".join(details)
    )


__all__ = [
    "DatasetArtifactKind",
    "DatasetArtifactPreflightError",
    "DatasetArtifactRequirement",
    "DatasetArtifactStatus",
    "check_dataset_artifacts",
    "require_dataset_artifacts",
]
