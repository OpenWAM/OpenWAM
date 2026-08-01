"""Dependency-light static-validation issue and report contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticConfigIssue:
    level: str
    path: str
    message: str


@dataclass(frozen=True)
class StaticConfigReport:
    source_path: Path
    errors: tuple[StaticConfigIssue, ...]
    warnings: tuple[StaticConfigIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class _IssueBuilder:
    def __init__(self, *, source_path: Path, repo_root: Path) -> None:
        self.source_path = source_path
        self.repo_root = repo_root
        self.errors: list[StaticConfigIssue] = []
        self.warnings: list[StaticConfigIssue] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(StaticConfigIssue(level="error", path=path or "<root>", message=message))

    def warning(self, path: str, message: str) -> None:
        self.warnings.append(StaticConfigIssue(level="warning", path=path or "<root>", message=message))


__all__ = ["StaticConfigIssue", "StaticConfigReport"]
