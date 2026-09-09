"""Explicit optional artifact sources, independent of sequence semantics."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactCacheConfig:
    catalog_spec_path: str
    cache_dir: str = "~/.cache/openwam/objects"
    max_bytes: int = 200 * 1024**3
    min_free_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        if not self.catalog_spec_path.strip() or not self.cache_dir.strip():
            raise ValueError("Artifact catalog and cache paths must be nonempty.")
        if self.max_bytes <= 0 or self.min_free_bytes < 0:
            raise ValueError(
                "Artifact cache requires positive capacity and nonnegative reserve."
            )


@dataclass(frozen=True)
class PromptCacheConfig:
    root: str
    encoder_fingerprint: str | None = None
    artifact_cache: ArtifactCacheConfig | None = None

    def __post_init__(self) -> None:
        if not self.root.strip():
            raise ValueError("Prompt cache root must be nonempty.")


def parse_artifact_cache(raw: Mapping[str, Any] | None) -> ArtifactCacheConfig | None:
    return None if raw is None else ArtifactCacheConfig(**raw)


def parse_prompt_cache(raw: Mapping[str, Any] | None) -> PromptCacheConfig | None:
    if raw is None:
        return None
    return PromptCacheConfig(
        **{**raw, "artifact_cache": parse_artifact_cache(raw.get("artifact_cache"))}
    )
