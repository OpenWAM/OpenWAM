#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from open_wam.models.visual_tower.reference_loader import load_internal_wan_transformer_class


TransformFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _identity(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    del target
    return value


def _conv3d_to_linear(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reshaped = value.reshape(value.shape[0], -1)
    if reshaped.shape != target.shape:
        raise ValueError(
            "Unable to reshape Wan patch_embedding.weight into LingBot patch_embedding_mlp.weight: "
            f"raw {tuple(value.shape)} -> {tuple(reshaped.shape)}, expected {tuple(target.shape)}."
        )
    return reshaped


def build_wan22_to_lingbot_map(num_layers: int) -> dict[str, tuple[str, TransformFn]]:
    mapping: dict[str, tuple[str, TransformFn]] = {
        "patch_embedding.weight": ("patch_embedding_mlp.weight", _conv3d_to_linear),
        "patch_embedding.bias": ("patch_embedding_mlp.bias", _identity),
        "text_embedding.0.weight": ("condition_embedder.text_embedder.linear_1.weight", _identity),
        "text_embedding.0.bias": ("condition_embedder.text_embedder.linear_1.bias", _identity),
        "text_embedding.2.weight": ("condition_embedder.text_embedder.linear_2.weight", _identity),
        "text_embedding.2.bias": ("condition_embedder.text_embedder.linear_2.bias", _identity),
        "time_embedding.0.weight": ("condition_embedder.time_embedder.linear_1.weight", _identity),
        "time_embedding.0.bias": ("condition_embedder.time_embedder.linear_1.bias", _identity),
        "time_embedding.2.weight": ("condition_embedder.time_embedder.linear_2.weight", _identity),
        "time_embedding.2.bias": ("condition_embedder.time_embedder.linear_2.bias", _identity),
        "time_projection.1.weight": ("condition_embedder.time_proj.weight", _identity),
        "time_projection.1.bias": ("condition_embedder.time_proj.bias", _identity),
        "head.head.weight": ("proj_out.weight", _identity),
        "head.head.bias": ("proj_out.bias", _identity),
        "head.modulation": ("scale_shift_table", _identity),
    }

    per_block = {
        "self_attn.q.weight": "attn1.to_q.weight",
        "self_attn.q.bias": "attn1.to_q.bias",
        "self_attn.k.weight": "attn1.to_k.weight",
        "self_attn.k.bias": "attn1.to_k.bias",
        "self_attn.v.weight": "attn1.to_v.weight",
        "self_attn.v.bias": "attn1.to_v.bias",
        "self_attn.o.weight": "attn1.to_out.0.weight",
        "self_attn.o.bias": "attn1.to_out.0.bias",
        "self_attn.norm_q.weight": "attn1.norm_q.weight",
        "self_attn.norm_k.weight": "attn1.norm_k.weight",
        "cross_attn.q.weight": "attn2.to_q.weight",
        "cross_attn.q.bias": "attn2.to_q.bias",
        "cross_attn.k.weight": "attn2.to_k.weight",
        "cross_attn.k.bias": "attn2.to_k.bias",
        "cross_attn.v.weight": "attn2.to_v.weight",
        "cross_attn.v.bias": "attn2.to_v.bias",
        "cross_attn.o.weight": "attn2.to_out.0.weight",
        "cross_attn.o.bias": "attn2.to_out.0.bias",
        "cross_attn.norm_q.weight": "attn2.norm_q.weight",
        "cross_attn.norm_k.weight": "attn2.norm_k.weight",
        "ffn.0.weight": "ffn.net.0.proj.weight",
        "ffn.0.bias": "ffn.net.0.proj.bias",
        "ffn.2.weight": "ffn.net.2.weight",
        "ffn.2.bias": "ffn.net.2.bias",
        "norm3.weight": "norm2.weight",
        "norm3.bias": "norm2.bias",
        "modulation": "scale_shift_table",
    }
    for layer_index in range(num_layers):
        for raw_suffix, lingbot_suffix in per_block.items():
            mapping[f"blocks.{layer_index}.{raw_suffix}"] = (
                f"blocks.{layer_index}.{lingbot_suffix}",
                _identity,
            )
    return mapping


def build_lingbot_config_from_wan_config(wan_config: dict[str, object], *, action_dim: int, attn_mode: str) -> dict[str, object]:
    dim = int(wan_config["dim"])
    num_heads = int(wan_config["num_heads"])
    if dim % num_heads != 0:
        raise ValueError(f"Wan dim={dim} is not divisible by num_heads={num_heads}.")
    return {
        "patch_size": [1, 2, 2],
        "num_attention_heads": num_heads,
        "attention_head_dim": dim // num_heads,
        "in_channels": int(wan_config["in_dim"]),
        "out_channels": int(wan_config["out_dim"]),
        "action_dim": action_dim,
        "text_dim": 4096,
        "freq_dim": int(wan_config["freq_dim"]),
        "ffn_dim": int(wan_config["ffn_dim"]),
        "num_layers": int(wan_config["num_layers"]),
        "cross_attn_norm": True,
        "eps": float(wan_config["eps"]),
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
        "attn_mode": attn_mode,
        "_class_name": "WanTransformer3DModel",
        "_diffusers_version": "0.35.0.dev0",
    }


def _load_selected_safetensors(root: Path, keys: set[str]) -> dict[str, torch.Tensor]:
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    single_path = root / "diffusion_pytorch_model.safetensors"
    state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = payload.get("weight_map", {})
        shard_to_keys: dict[str, list[str]] = {}
        for key in sorted(keys):
            shard_name = weight_map.get(key)
            if shard_name is not None:
                shard_to_keys.setdefault(shard_name, []).append(key)
        for shard_name, shard_keys in shard_to_keys.items():
            with safe_open(str(root / shard_name), framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    state[key] = handle.get_tensor(key)
        return state
    if single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in sorted(keys & available):
                state[key] = handle.get_tensor(key)
        return state
    raise FileNotFoundError(
        "Unable to find Wan2.2 safetensors. Expected either "
        f"{index_path} or {single_path}."
    )


def _parse_torch_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(choices)}.") from exc


def _copy_auxiliary_components(source_root: Path, output_root: Path) -> None:
    for name in ("vae", "text_encoder", "tokenizer"):
        source = source_root / name
        if source.exists():
            target = output_root / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, symlinks=True)


def convert_wan22_to_lingbot_init(
    *,
    wan_root: Path,
    output_root: Path,
    action_dim: int = 30,
    attn_mode: str = "flex",
    max_shard_size: str = "5GB",
    seed: int = 0,
    dtype: str = "bfloat16",
    copy_auxiliary_components: bool = False,
) -> dict[str, object]:
    wan_config_path = wan_root / "config.json"
    if not wan_config_path.exists():
        raise FileNotFoundError(f"Missing Wan2.2 config: {wan_config_path}")
    wan_config = json.loads(wan_config_path.read_text(encoding="utf-8"))
    lingbot_config = build_lingbot_config_from_wan_config(wan_config, action_dim=action_dim, attn_mode=attn_mode)
    num_layers = int(lingbot_config["num_layers"])
    mapping = build_wan22_to_lingbot_map(num_layers)

    target_dtype = _parse_torch_dtype(dtype)
    torch.manual_seed(seed)
    model_cls = load_internal_wan_transformer_class()
    model = model_cls(
        patch_size=lingbot_config["patch_size"],
        num_attention_heads=lingbot_config["num_attention_heads"],
        attention_head_dim=lingbot_config["attention_head_dim"],
        in_channels=lingbot_config["in_channels"],
        out_channels=lingbot_config["out_channels"],
        action_dim=lingbot_config["action_dim"],
        text_dim=lingbot_config["text_dim"],
        freq_dim=lingbot_config["freq_dim"],
        ffn_dim=lingbot_config["ffn_dim"],
        num_layers=lingbot_config["num_layers"],
        cross_attn_norm=lingbot_config["cross_attn_norm"],
        eps=lingbot_config["eps"],
        rope_max_seq_len=lingbot_config["rope_max_seq_len"],
        pos_embed_seq_len=lingbot_config["pos_embed_seq_len"],
        attn_mode=lingbot_config["attn_mode"],
    ).to(dtype=target_dtype)
    target_state = model.state_dict()
    raw_state = _load_selected_safetensors(wan_root, set(mapping))

    missing_raw = sorted(set(mapping) - set(raw_state))
    if missing_raw:
        raise ValueError(f"Wan2.2 checkpoint is missing {len(missing_raw)} mapped keys. First keys: {missing_raw[:10]}")

    loaded: list[str] = []
    for raw_key, (target_key, transform) in mapping.items():
        if target_key not in target_state:
            raise KeyError(f"Mapped LingBot target key is not in model state_dict: {target_key}")
        value = transform(raw_state[raw_key], target_state[target_key]).to(dtype=target_state[target_key].dtype)
        if value.shape != target_state[target_key].shape:
            raise ValueError(
                f"Shape mismatch for {raw_key} -> {target_key}: "
                f"got {tuple(value.shape)}, expected {tuple(target_state[target_key].shape)}."
            )
        target_state[target_key] = value
        loaded.append(target_key)

    model.load_state_dict(target_state, strict=True)
    transformer_dir = output_root / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformer_dir, safe_serialization=True, max_shard_size=max_shard_size)
    if copy_auxiliary_components:
        _copy_auxiliary_components(wan_root, output_root)

    report = {
        "wan_root": str(wan_root),
        "output_root": str(output_root),
        "transformer_dir": str(transformer_dir),
        "num_layers": num_layers,
        "mapped_raw_keys": len(mapping),
        "loaded_lingbot_keys": len(loaded),
        "left_random_lingbot_keys": sorted(set(target_state) - set(loaded)),
        "dtype": dtype,
        "copy_auxiliary_components": copy_auxiliary_components,
    }
    (output_root / "wan22_to_lingbot_init_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official Wan2.2-TI2V-5B transformer weights into LingBot/Open-WAM format."
    )
    parser.add_argument("--wan-root", required=True, type=Path, help="Official Wan2.2-TI2V-5B model root.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output model root to create.")
    parser.add_argument("--action-dim", default=30, type=int, help="LingBot action_dim for randomly initialized action layers.")
    parser.add_argument("--attn-mode", default="flex", help="LingBot transformer attn_mode written to config.json.")
    parser.add_argument("--max-shard-size", default="5GB", help="Shard size passed to save_pretrained().")
    parser.add_argument("--seed", default=0, type=int, help="Seed for LingBot-only random action layers.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32", "float16"),
        help="Dtype used for the saved LingBot-format checkpoint.",
    )
    parser.add_argument(
        "--copy-auxiliary-components",
        action="store_true",
        help="Copy vae/text_encoder/tokenizer directories if present next to the Wan2.2 root.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = convert_wan22_to_lingbot_init(
        wan_root=args.wan_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        action_dim=args.action_dim,
        attn_mode=args.attn_mode,
        max_shard_size=args.max_shard_size,
        seed=args.seed,
        dtype=args.dtype,
        copy_auxiliary_components=args.copy_auxiliary_components,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
