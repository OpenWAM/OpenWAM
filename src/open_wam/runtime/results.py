from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from open_wam import __version__


OPEN_WAM_RESULT_SCHEMA_V1 = "open_wam.result.v1"
RESERVED_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "open_wam_version",
        "created_at",
        "command",
        "config",
        "checkpoint",
        "benchmark",
        "device",
        "seed",
        "metrics",
        "artifacts",
    }
)


def build_result_envelope(
    *,
    command: str,
    config: str | None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    checkpoint: str | None = None,
    benchmark: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable result envelope used by new runtime outputs."""

    envelope: dict[str, Any] = {
        "schema_version": OPEN_WAM_RESULT_SCHEMA_V1,
        "open_wam_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "checkpoint": checkpoint,
        "benchmark": benchmark,
        "device": device,
        "seed": seed,
        "metrics": dict(metrics or {}),
        "artifacts": dict(artifacts or {}),
    }
    if extra:
        extra_dict = dict(extra)
        collisions = RESERVED_RESULT_KEYS.intersection(extra_dict)
        for key, value in extra_dict.items():
            if key not in RESERVED_RESULT_KEYS:
                envelope[key] = value
        if collisions:
            envelope["legacy"] = extra_dict
            envelope["legacy_key_collisions"] = sorted(collisions)
    return envelope
