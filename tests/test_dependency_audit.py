from __future__ import annotations

from datetime import date

import pytest

from scripts.check_dependency_audit import (
    DEFAULT_POLICY,
    AuditException,
    load_audit_exceptions,
    validate_audit_report,
)


def _report(*, vulnerability_id: str = "PYSEC-test") -> dict[str, object]:
    return {
        "dependencies": [
            {
                "name": "Example__Package",
                "version": "1.2.3",
                "vulns": [
                    {
                        "id": vulnerability_id,
                        "fix_versions": ["1.2.4"],
                    }
                ],
            }
        ]
    }


def _exception(*, vulnerability_id: str = "PYSEC-test", expires: date = date(2026, 9, 1)) -> AuditException:
    return AuditException(
        package="example-package",
        version="1.2.3",
        vulnerability_id=vulnerability_id,
        expires=expires,
        reason="Maintained code does not expose the affected API.",
    )


def test_dependency_audit_accepts_exact_reviewed_exception() -> None:
    validation = validate_audit_report(
        _report(),
        (_exception(),),
        today=date(2026, 8, 6),
    )

    assert len(validation.findings) == 1
    assert validation.accepted_exceptions == (_exception(),)


def test_dependency_audit_rejects_new_vulnerability() -> None:
    with pytest.raises(ValueError, match="unaccepted vulnerabilities"):
        validate_audit_report(_report(), (), today=date(2026, 8, 6))


def test_dependency_audit_rejects_expired_exception() -> None:
    with pytest.raises(ValueError, match="expired"):
        validate_audit_report(
            _report(),
            (_exception(expires=date(2026, 8, 5)),),
            today=date(2026, 8, 6),
        )


def test_dependency_audit_rejects_stale_exception() -> None:
    with pytest.raises(ValueError, match="stale"):
        validate_audit_report(
            {"dependencies": []},
            (_exception(),),
            today=date(2026, 8, 6),
        )


@pytest.mark.parametrize("version,today,error", (
    ("1.13.0", date(2026, 9, 9), None),
    ("1.14.0", date(2026, 9, 9), "unaccepted vulnerabilities"),
    ("1.13.0", date(2026, 10, 10), "expired"),
))
def test_accelerate_release_exception_is_version_and_time_bounded(
    version: str, today: date, error: str | None,
) -> None:
    exception, = [
        item for item in load_audit_exceptions(DEFAULT_POLICY)
        if item.vulnerability_id == "CVE-2026-69112"
    ]
    report = {"dependencies": [{
        "name": "accelerate", "version": version,
        "vulns": [{"id": "CVE-2026-69112", "fix_versions": []}],
    }]}
    if error is not None:
        with pytest.raises(ValueError, match=error):
            validate_audit_report(report, (exception,), today=today)
    else:
        result = validate_audit_report(report, (exception,), today=today)
        assert result.accepted_exceptions == (exception,)
