from __future__ import annotations


def build_metadata_cache(kind: str, **kwargs) -> dict[str, object]:
    """Create a simple metadata-only cache payload."""

    payload: dict[str, object] = {"kind": kind}
    payload.update(kwargs)
    return payload
