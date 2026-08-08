from __future__ import annotations

from datetime import date

import pytest

from scripts.check_dependency_audit import AuditException, validate_audit_report


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
