from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / ".github" / "dependency-audit-exceptions.toml"


@dataclass(frozen=True, order=True)
class AuditFinding:
    package: str
    version: str
    vulnerability_id: str
    fix_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditException:
    package: str
    version: str
    vulnerability_id: str
    expires: date
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.package, self.version, self.vulnerability_id


@dataclass(frozen=True)
class AuditValidation:
    findings: tuple[AuditFinding, ...]
    accepted_exceptions: tuple[AuditException, ...]


def load_audit_exceptions(path: Path) -> tuple[AuditException, ...]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Dependency audit exception policy requires schema_version = 1.")
    exceptions: list[AuditException] = []
    for index, raw in enumerate(payload.get("exceptions", ())):
        if not isinstance(raw, dict):
            raise ValueError(f"Dependency audit exception {index} must be a TOML table.")
        try:
            expires = raw["expires"]
            exception = AuditException(
                package=_normalize_package_name(str(raw["package"])),
                version=str(raw["version"]),
                vulnerability_id=str(raw["id"]),
                expires=expires,
                reason=str(raw["reason"]).strip(),
            )
        except KeyError as exc:
            raise ValueError(f"Dependency audit exception {index} is missing {exc.args[0]!r}.") from exc
        if not isinstance(exception.expires, date):
            raise ValueError(f"Dependency audit exception {index} requires a TOML date for `expires`.")
        if not exception.reason:
            raise ValueError(f"Dependency audit exception {index} requires a rationale.")
        exceptions.append(exception)
    keys = [exception.key for exception in exceptions]
    if len(keys) != len(set(keys)):
        raise ValueError("Dependency audit exception keys must be unique.")
    return tuple(exceptions)


def parse_audit_findings(payload: dict[str, Any]) -> tuple[AuditFinding, ...]:
    findings: set[AuditFinding] = set()
    for dependency in payload.get("dependencies", ()):
        package = _normalize_package_name(str(dependency["name"]))
        version = str(dependency["version"])
        for vulnerability in dependency.get("vulns", ()):
            findings.add(
                AuditFinding(
                    package=package,
                    version=version,
                    vulnerability_id=str(vulnerability["id"]),
                    fix_versions=tuple(sorted(str(item) for item in vulnerability.get("fix_versions", ()))),
                )
            )
    return tuple(sorted(findings))


def validate_audit_report(
    payload: dict[str, Any],
    exceptions: tuple[AuditException, ...],
    *,
    today: date | None = None,
) -> AuditValidation:
    current_date = today or date.today()
    expired = tuple(exception for exception in exceptions if exception.expires < current_date)
    if expired:
        raise ValueError(
            "Dependency audit exceptions have expired: "
            + ", ".join(_format_exception(exception) for exception in expired)
        )

    findings = parse_audit_findings(payload)
    exception_by_key = {exception.key: exception for exception in exceptions}
    accepted: list[AuditException] = []
    unaccepted: list[AuditFinding] = []
    for finding in findings:
        key = finding.package, finding.version, finding.vulnerability_id
        exception = exception_by_key.get(key)
        if exception is None:
            unaccepted.append(finding)
        else:
            accepted.append(exception)
    if unaccepted:
        raise ValueError(
            "Dependency audit found unaccepted vulnerabilities: "
            + ", ".join(_format_finding(finding) for finding in unaccepted)
        )

    accepted_keys = {exception.key for exception in accepted}
    stale = tuple(exception for exception in exceptions if exception.key not in accepted_keys)
    if stale:
        raise ValueError(
            "Dependency audit exceptions are stale and must be removed: "
            + ", ".join(_format_exception(exception) for exception in stale)
        )
    return AuditValidation(
        findings=findings,
        accepted_exceptions=tuple(sorted(accepted, key=lambda item: item.key)),
    )


def run_dependency_audit(*, policy_path: Path = DEFAULT_POLICY) -> AuditValidation:
    exceptions = load_audit_exceptions(policy_path)
    with tempfile.TemporaryDirectory(prefix="open-wam-dependency-audit-") as temporary_dir:
        requirements_path = Path(temporary_dir) / "requirements.txt"
        export = subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--extra",
                "full",
                "--no-dev",
                "--format",
                "requirements-txt",
                "--no-emit-project",
                "--output-file",
                str(requirements_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if export.returncode != 0:
            raise RuntimeError(f"uv export failed:\n{export.stderr or export.stdout}")
        audit = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                str(requirements_path),
                "--no-deps",
                "--disable-pip",
                "--format",
                "json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if audit.returncode not in {0, 1}:
            raise RuntimeError(f"pip-audit failed:\n{audit.stderr or audit.stdout}")
        try:
            payload = json.loads(audit.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"pip-audit did not emit valid JSON:\n{audit.stdout}\n{audit.stderr}") from exc
    return validate_audit_report(payload, exceptions)


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _format_finding(finding: AuditFinding) -> str:
    fixes = f" (fix: {', '.join(finding.fix_versions)})" if finding.fix_versions else ""
    return f"{finding.package}=={finding.version}:{finding.vulnerability_id}{fixes}"


def _format_exception(exception: AuditException) -> str:
    return (
        f"{exception.package}=={exception.version}:{exception.vulnerability_id} "
        f"(expires {exception.expires.isoformat()})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Open-WAM full environment.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args(argv)
    try:
        validation = run_dependency_audit(policy_path=args.policy)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "dependency audit ok: "
        f"{len(validation.findings)} accepted findings, "
        f"{len(validation.accepted_exceptions)} reviewed exceptions"
    )


if __name__ == "__main__":
    main()
