from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedCheckpoint:
    name: str
    pretrained_root: Path
    source_transformer_dir: Path
    transformer_dir: Path
    model_root: Path

    @property
    def converted_for_heng(self) -> bool:
        return self.source_transformer_dir.resolve() != self.transformer_dir.resolve()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretrained_root": str(self.pretrained_root),
            "source_transformer_dir": str(self.source_transformer_dir),
            "transformer_dir": str(self.transformer_dir),
            "model_root": str(self.model_root),
            "converted_for_heng": self.converted_for_heng,
        }


def stage_checkpoint(
    *,
    name: str,
    save_root: Path,
    pretrained_root: Path,
    transformer_dir: Path | None,
) -> PreparedCheckpoint:
    pretrained_root = pretrained_root.expanduser().resolve()
    source_transformer_dir = (transformer_dir.expanduser().resolve() if transformer_dir else pretrained_root / "transformer")
    resolved_transformer_dir = ensure_heng_compatible_transformer_dir(
        save_root=save_root,
        pretrained_root=pretrained_root,
        transformer_dir=source_transformer_dir,
    )
    model_root = prepare_model_root(
        save_root=save_root,
        pretrained_root=pretrained_root,
        transformer_dir=resolved_transformer_dir,
    )
    return PreparedCheckpoint(
        name=name,
        pretrained_root=pretrained_root,
        source_transformer_dir=source_transformer_dir,
        transformer_dir=resolved_transformer_dir,
        model_root=model_root,
    )


def ensure_heng_compatible_transformer_dir(
    *,
    save_root: Path,
    pretrained_root: Path,
    transformer_dir: Path,
) -> Path:
    transformer_dir = transformer_dir.expanduser().resolve()
    config_path = transformer_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Transformer config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        source_config = json.load(handle)
    if source_config.get("_class_name") == "WanTransformer3DModel":
        return transformer_dir

    weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Only unsharded Open-WAM transformer exports are supported: {weights_path}")
    digest = hashlib.sha256(str(transformer_dir).encode("utf-8")).hexdigest()[:12]
    converted_dir = save_root.resolve() / "_heng_transformer_converted" / digest
    converted_config_path = converted_dir / "config.json"
    converted_weights_path = converted_dir / "diffusion_pytorch_model.safetensors"
    if converted_config_path.is_file() and converted_weights_path.is_file():
        return converted_dir

    base_config_path = pretrained_root.expanduser().resolve() / "transformer" / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Heng transformer base config not found: {base_config_path}")
    converted_dir.mkdir(parents=True, exist_ok=True)
    with base_config_path.open("r", encoding="utf-8") as handle:
        heng_config = json.load(handle)
    if "attn_mode" in source_config:
        heng_config["attn_mode"] = source_config["attn_mode"]

    from safetensors.torch import load_file, save_file

    source_state = load_file(str(weights_path), device="cpu")
    converted_state = {}
    for key, value in source_state.items():
        converted_key = to_heng_transformer_key(key)
        if converted_key is None:
            continue
        converted_state[converted_key] = value

    with converted_config_path.open("w", encoding="utf-8") as handle:
        json.dump(heng_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_file(converted_state, str(converted_weights_path))
    return converted_dir


def to_heng_transformer_key(key: str) -> str | None:
    if key.startswith("runtime_stream_adapters."):
        return None
    prefix_pairs = (
        ("time_conditioner.", "condition_embedder."),
        ("text_proj.", "condition_embedder.text_embedder."),
        ("action_time_conditioner.", "condition_embedder_action."),
        ("action_text_proj.", "condition_embedder_action.text_embedder."),
    )
    for source_prefix, target_prefix in prefix_pairs:
        if key.startswith(source_prefix):
            return f"{target_prefix}{key[len(source_prefix):]}"
    return key


def prepare_model_root(*, save_root: Path, pretrained_root: Path, transformer_dir: Path) -> Path:
    pretrained_root = pretrained_root.expanduser().resolve()
    transformer_dir = transformer_dir.expanduser().resolve()
    require_dir(pretrained_root / "vae")
    require_dir(pretrained_root / "text_encoder")
    require_dir(pretrained_root / "tokenizer")
    require_dir(transformer_dir)

    digest = hashlib.sha256(f"{pretrained_root}|{transformer_dir}".encode("utf-8")).hexdigest()[:12]
    model_root = save_root.resolve() / "_heng_model_roots" / digest
    model_root.mkdir(parents=True, exist_ok=True)
    safe_link_dir(pretrained_root / "vae", model_root / "vae")
    safe_link_dir(pretrained_root / "text_encoder", model_root / "text_encoder")
    safe_link_dir(pretrained_root / "tokenizer", model_root / "tokenizer")
    safe_link_dir(transformer_dir, model_root / "transformer")
    return model_root


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required LingBot model directory not found: {path}")


def safe_link_dir(source: Path, dest: Path) -> None:
    source = source.expanduser().resolve()
    if dest.is_symlink() and dest.resolve() == source:
        return
    if dest.exists():
        if dest.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace existing LingBot model path: {dest}")
    dest.symlink_to(source, target_is_directory=True)


def build_component_report(model: Any, prepared: PreparedCheckpoint) -> dict[str, Any]:
    pretrained_root = prepared.pretrained_root
    transformer_dir = prepared.transformer_dir
    vae_dir = pretrained_root / "vae"
    text_encoder_dir = pretrained_root / "text_encoder"
    tokenizer_dir = pretrained_root / "tokenizer"
    transformer_config = getattr(model.transformer, "config", None)
    return {
        "pipeline": "lingbot_va_baseline",
        "runtime_device": str(model.device),
        "source_transformer_dir": str(prepared.source_transformer_dir),
        "transformer_dir": str(transformer_dir),
        "transformer_converted_for_heng": prepared.converted_for_heng,
        "model_root": str(prepared.model_root),
        "backbone_pretrained_root": str(pretrained_root),
        "transformer_config_sha256": sha256_if_exists(transformer_dir / "config.json"),
        "transformer_weights_sha256": sha256_if_exists(transformer_dir / "diffusion_pytorch_model.safetensors"),
        "vae_dir": str(vae_dir),
        "vae_config_sha256": sha256_if_exists(vae_dir / "config.json"),
        "vae_weights_sha256": sha256_if_exists(vae_dir / "diffusion_pytorch_model.safetensors"),
        "text_encoder_dir": str(text_encoder_dir),
        "text_encoder_index_sha256": sha256_if_exists(text_encoder_dir / "model.safetensors.index.json"),
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
        "baseline_action_config": getattr(model.job_config, "lingbot_va_baseline_action_config", None),
        "enable_offload": bool(model.enable_offload),
        "runtime_mode": "heng_server_exact_chunk_by_chunk",
    }


def sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
