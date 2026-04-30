from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_MODEL_SUBDIRS = ("vae", "text_encoder", "tokenizer", "transformer")


@dataclass(frozen=True)
class PreparedModel:
    """Resolved vanilla LingBot-VA model root used by the baseline runner."""

    name: str
    model_root: Path
    hf_repo_id: str | None = None
    hf_revision: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_root": str(self.model_root),
            "hf_repo_id": self.hf_repo_id,
            "hf_revision": self.hf_revision,
        }


def prepare_model_root(
    *,
    name: str,
    model_root: Path,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> PreparedModel:
    resolved = model_root.expanduser().resolve()
    for subdir in REQUIRED_MODEL_SUBDIRS:
        require_dir(resolved / subdir)
    require_file(resolved / "transformer" / "config.json")
    return PreparedModel(
        name=name,
        model_root=resolved,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required LingBot-VA model directory not found: {path}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required LingBot-VA model file not found: {path}")


def build_component_report(model: Any, prepared: PreparedModel) -> dict[str, Any]:
    root = prepared.model_root
    transformer_dir = root / "transformer"
    vae_dir = root / "vae"
    text_encoder_dir = root / "text_encoder"
    tokenizer_dir = root / "tokenizer"
    transformer_config = getattr(model.transformer, "config", None)
    return {
        "pipeline": "lingbot_va_vanilla_baseline",
        "runtime_device": str(model.device),
        "model_root": str(root),
        "hf_repo_id": prepared.hf_repo_id,
        "hf_revision": prepared.hf_revision,
        "transformer_dir": str(transformer_dir),
        "transformer_config_sha256": sha256_if_exists(transformer_dir / "config.json"),
        "transformer_index_sha256": sha256_if_exists(transformer_dir / "diffusion_pytorch_model.safetensors.index.json"),
        "transformer_weight_files": file_manifest(transformer_dir.glob("*.safetensors")),
        "vae_dir": str(vae_dir),
        "vae_config_sha256": sha256_if_exists(vae_dir / "config.json"),
        "vae_weight_files": file_manifest(vae_dir.glob("*.safetensors")),
        "text_encoder_dir": str(text_encoder_dir),
        "text_encoder_index_sha256": sha256_if_exists(text_encoder_dir / "model.safetensors.index.json"),
        "text_encoder_weight_files": file_manifest(text_encoder_dir.glob("*.safetensors")),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_json_sha256": sha256_if_exists(tokenizer_dir / "tokenizer.json"),
        "spiece_sha256": sha256_if_exists(tokenizer_dir / "spiece.model"),
        "transformer_class": model.transformer.__class__.__name__,
        "transformer_num_layers": getattr(transformer_config, "num_layers", None),
        "transformer_action_dim": getattr(transformer_config, "action_dim", None),
        "transformer_attn_mode": getattr(transformer_config, "attn_mode", None),
        "transformer_patch_size": list(getattr(model.transformer, "patch_size", ()) or ()),
        "frame_chunk_size": int(model.job_config.frame_chunk_size),
        "action_per_frame": int(model.job_config.action_per_frame),
        "action_snr_shift": float(model.job_config.action_snr_shift),
        "used_action_channel_ids": list(model.job_config.used_action_channel_ids),
        "enable_offload": bool(model.enable_offload),
        "runtime_mode": "upstream_va_server_vanilla_loop",
    }


def sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(item) for item in paths):
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def read_transformer_attn_mode(model_root: Path) -> str | None:
    config_path = model_root.expanduser().resolve() / "transformer" / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    value = raw.get("attn_mode")
    return None if value is None else str(value)
