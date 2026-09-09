"""Read the manifest inventory sealed by snapshot publication."""

import hashlib
import json
from pathlib import Path

from open_wam.artifacts.files import sha256_file


def load_verified_snapshot(root: str | Path) -> dict:
    """Verify snapshot identity and manifest bytes, without imposing a mixture."""
    root = Path(root).resolve()
    record = json.loads((root / "snapshot.json").read_text())
    if record.get("format_version") != 1 or record.get("status") != "complete":
        raise ValueError("Snapshot must be complete format_version=1")
    manifests = record.get("manifests_sha256")
    sources = record.get("sources")
    if (
        not isinstance(manifests, dict)
        or not manifests
        or not isinstance(sources, dict)
        or set(manifests) != {f"{source}.csv" for source in sources}
        or any(Path(name).name != name for name in manifests)
    ):
        raise ValueError("Snapshot manifest inventory does not match its sources")
    digest = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    if record.get("snapshot_sha256") != digest:
        raise ValueError("Snapshot inventory checksum mismatch")
    for name, expected in manifests.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Snapshot manifest changed: {root / name}")
    return record
