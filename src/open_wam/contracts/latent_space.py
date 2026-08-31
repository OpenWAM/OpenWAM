from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_LATENT_SPACE_SCHEMA_V1 = "open_wam.video_latent_space.v1"
_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".safetensors"})


@dataclass(frozen=True)
class VideoLatentSpaceIdentity:
    """Path-independent identity of an encoded video latent coordinate space."""

    encoder_family: str
    encoding_contract: str
    config_sha256: str
    weights_sha256: str
    artifact_sha256: str
    schema_version: str = VIDEO_LATENT_SPACE_SCHEMA_V1

    def __post_init__(self) -> None:
        for field_name in ("encoder_family", "encoding_contract"):
            value = str(getattr(self, field_name))
            if not value:
                raise ValueError(f"Video latent-space {field_name} cannot be empty.")
        for field_name in ("config_sha256", "weights_sha256", "artifact_sha256"):
            _validate_sha256(str(getattr(self, field_name)), field_name=field_name)
        if self.schema_version != VIDEO_LATENT_SPACE_SCHEMA_V1:
            raise ValueError(
                "Unsupported video latent-space identity schema "
                f"{self.schema_version!r}; expected {VIDEO_LATENT_SPACE_SCHEMA_V1!r}."
            )
        expected_artifact_sha256 = _mapping_sha256(
            {
                "schema_version": self.schema_version,
                "encoder_family": self.encoder_family,
                "encoding_contract": self.encoding_contract,
                "config_sha256": self.config_sha256,
                "weights_sha256": self.weights_sha256,
            }
        )
        if self.artifact_sha256 != expected_artifact_sha256:
            raise ValueError(
                "Video latent-space artifact digest does not match its semantic "
                "components."
            )

    def to_mapping(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "encoder_family": self.encoder_family,
            "encoding_contract": self.encoding_contract,
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


def identify_video_latent_space(
    artifact_root: str | Path,
    *,
    encoder_family: str,
    encoding_contract: str,
) -> VideoLatentSpaceIdentity:
    """Fingerprint the configuration and weights that define one latent space."""

    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video latent-space artifact is not a directory: {root}.")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Video latent-space artifact is missing config.json: {root}."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Video latent-space config is not valid JSON: {config_path}."
        ) from exc
    if not isinstance(config, dict):
        raise TypeError(
            f"Video latent-space config must be a JSON object: {config_path}."
        )

    weight_paths = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if not weight_paths:
        raise FileNotFoundError(
            f"Video latent-space artifact contains no recognized weight files: {root}."
        )
    weight_manifest = [
        _weight_file_identity(path, root=root) for path in weight_paths
    ]
    config_sha256 = _mapping_sha256(config)
    weights_sha256 = _mapping_sha256({"files": weight_manifest})
    semantic_identity = {
        "schema_version": VIDEO_LATENT_SPACE_SCHEMA_V1,
        "encoder_family": str(encoder_family),
        "encoding_contract": str(encoding_contract),
        "config_sha256": config_sha256,
        "weights_sha256": weights_sha256,
    }
    return VideoLatentSpaceIdentity(
        encoder_family=str(encoder_family),
        encoding_contract=str(encoding_contract),
        config_sha256=config_sha256,
        weights_sha256=weights_sha256,
        artifact_sha256=_mapping_sha256(semantic_identity),
    )


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"Video latent-space {field_name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"Video latent-space {field_name} must be a hexadecimal SHA-256 digest."
        ) from exc


def _mapping_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weight_file_identity(path: Path, *, root: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    # Filesystem timestamps are not a content identity: rapid same-size rewrites
    # can retain every stat field on network filesystems. This is a startup-only
    # compatibility gate, so hash the artifact directly rather than risking a
    # stale process cache.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "VIDEO_LATENT_SPACE_SCHEMA_V1",
    "VideoLatentSpaceIdentity",
    "identify_video_latent_space",
]
