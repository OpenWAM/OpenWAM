from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
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
        "provenance",
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
    provenance: Mapping[str, Any] | None = None,
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
        "provenance": dict(provenance or {}),
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


def write_result_json(
    path: str | Path,
    result: Mapping[str, Any],
) -> Path:
    """Atomically persist one JSON-serializable result envelope."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(result), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            temporary_path = Path(handle.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


__all__ = [
    "OPEN_WAM_RESULT_SCHEMA_V1",
    "RESERVED_RESULT_KEYS",
    "build_result_envelope",
    "write_result_json",
]
