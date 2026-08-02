#!/usr/bin/env python
"""Qualitatively compare explicit LingBot-format WAN checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors import safe_open
from transformers import AutoTokenizer, UMT5EncoderModel


OPENWAM_TO_DIFFUSERS_REMAP = {
    "time_conditioner.time_embedder.linear_1.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_conditioner.time_embedder.linear_1.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_conditioner.time_embedder.linear_2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_conditioner.time_embedder.linear_2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_conditioner.time_proj.weight": "condition_embedder.time_proj.weight",
    "time_conditioner.time_proj.bias": "condition_embedder.time_proj.bias",
    "text_proj.linear_1.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_proj.linear_1.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_proj.linear_2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_proj.linear_2.bias": "condition_embedder.text_embedder.linear_2.bias",
}


def _parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_transformer_dir(path: Path) -> Path:
    if (path / "transformer").is_dir():
        return path / "transformer"
    return path


def _safetensor_shards(directory: Path) -> list[Path]:
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return [directory / name for name in sorted(set(payload.get("weight_map", {}).values()))]
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {directory}")
    return shards


def _load_state_for_diffusers_transformer(checkpoint_dir: Path, target_keys: set[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_dir = _resolve_transformer_dir(checkpoint_dir)
    direct_state: dict[str, torch.Tensor] = {}
    fallback_patch_weight: torch.Tensor | None = None
    fallback_patch_bias: torch.Tensor | None = None
    raw_key_count = 0
    ignored_keys: list[str] = []
    remapped_count = 0

    for shard in _safetensor_shards(checkpoint_dir):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                raw_key_count += 1
                value = handle.get_tensor(key)
                mapped_key = OPENWAM_TO_DIFFUSERS_REMAP.get(key, key)
                if mapped_key != key:
                    remapped_count += 1
                if mapped_key in target_keys:
                    direct_state[mapped_key] = value
                    continue
                if key == "patch_embedding_mlp.weight" and "patch_embedding.weight" in target_keys:
                    fallback_patch_weight = value
                    continue
                if key == "patch_embedding_mlp.bias" and "patch_embedding.bias" in target_keys:
                    fallback_patch_bias = value
                    continue
                ignored_keys.append(key)

    patch_fallback_used = False
    if "patch_embedding.weight" not in direct_state and fallback_patch_weight is not None:
        direct_state["patch_embedding.weight"] = fallback_patch_weight.reshape(fallback_patch_weight.shape[0], 48, 1, 2, 2)
        patch_fallback_used = True
        remapped_count += 1
    if "patch_embedding.bias" not in direct_state and fallback_patch_bias is not None:
        direct_state["patch_embedding.bias"] = fallback_patch_bias
        patch_fallback_used = True
        remapped_count += 1

    missing_keys = sorted(target_keys - set(direct_state))
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "raw_key_count": raw_key_count,
        "loaded_key_count": len(direct_state),
        "target_key_count": len(target_keys),
        "missing_key_count": len(missing_keys),
        "missing_keys_first": missing_keys[:20],
        "ignored_key_count": len(ignored_keys),
        "ignored_keys_first": ignored_keys[:20],
        "remapped_count": remapped_count,
        "patch_embedding_mlp_fallback_used": patch_fallback_used,
    }
    return direct_state, summary


def _load_transformer_from_checkpoint(
    checkpoint_dir: Path,
    template_dir: Path,
    dtype: torch.dtype,
    *,
    allow_partial_load: bool = False,
) -> tuple[WanTransformer3DModel, dict[str, Any]]:
    config = WanTransformer3DModel.load_config(str(template_dir))
    transformer = WanTransformer3DModel.from_config(config).to(dtype=dtype)
    target_keys = set(transformer.state_dict().keys())
    state, summary = _load_state_for_diffusers_transformer(checkpoint_dir, target_keys)
    # Default to strict load so a broken mapping / wrong checkpoint cannot
    # silently fall through to random target weights and still emit a video.
    load_result = transformer.load_state_dict(state, strict=not allow_partial_load)
    summary["load_missing_count"] = len(load_result.missing_keys)
    summary["load_missing_keys_first"] = list(load_result.missing_keys[:20])
    summary["load_unexpected_count"] = len(load_result.unexpected_keys)
    summary["load_unexpected_keys_first"] = list(load_result.unexpected_keys[:20])
    summary["allow_partial_load"] = bool(allow_partial_load)
    return transformer, summary


def _parse_checkpoint_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            label, value = spec.split("=", 1)
        else:
            value = spec
            label = Path(value).name or "checkpoint"
        label = label.strip().replace("/", "_")
        if not label:
            raise ValueError(f"Invalid checkpoint spec label: {spec!r}")
        parsed.append((label, Path(value).expanduser()))
    return parsed


def _frames_to_uint8(frames: Any) -> np.ndarray:
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame) for frame in frames], axis=0)
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames
    frames = frames.astype(np.float32)
    if frames.max(initial=0.0) <= 1.5:
        frames = frames * 255.0
    return np.clip(frames, 0, 255).astype(np.uint8)


def _save_mid_frame(frames: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    Image.fromarray(frames[len(frames) // 2]).save(path)


def _run_one(
    *,
    label: str,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    tokenizer: Any,
    text_encoder: UMT5EncoderModel,
    vae: AutoencoderKLWan,
    dtype: torch.dtype,
) -> dict[str, Any]:
    transformer, load_summary = _load_transformer_from_checkpoint(
        checkpoint_dir,
        args.transformer_template,
        dtype,
        allow_partial_load=args.allow_partial_load,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=args.shift,
        num_train_timesteps=1000,
    )
    pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    if args.cpu_offload:
        # Explicit `device` so `--device cuda:N` is honored under cpu offload
        # instead of silently offloading to the default accelerator.
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
    )
    frames = _frames_to_uint8(output.frames[0])
    label_dir = args.output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = label_dir / f"{label}.mp4"
    mid_path = label_dir / f"{label}_mid.png"
    imageio.mimsave(str(mp4_path), list(frames), fps=args.fps)
    _save_mid_frame(frames, mid_path)

    metrics = {
        "label": label,
        "checkpoint": str(checkpoint_dir),
        "mp4": str(mp4_path),
        "mid_frame": str(mid_path),
        "frame_shape": list(frames.shape),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
        "min": int(frames.min()),
        "max": int(frames.max()),
        "load": load_summary,
    }
    (label_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    del pipe, transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LingBot-format Wan init checkpoints with Wan text-to-video generation.")
    parser.add_argument(
        "--base-root",
        type=Path,
        required=True,
        help="Model root containing the tokenizer, text_encoder, and VAE.",
    )
    parser.add_argument(
        "--transformer-template",
        type=Path,
        required=True,
        help="Diffusers WanTransformer3DModel config directory used to instantiate the text-to-video transformer.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec, either LABEL=PATH or PATH. Repeat to compare multiple checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text2video_wan_lingbot_init_compare"))
    parser.add_argument("--prompt", default="a robotic arm picking up a red cube on a wooden table, cinematic, smooth motion")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help=(
            "Permit non-strict transformer.load_state_dict so missing/unexpected keys "
            "are reported but do not fail. Default off: a broken mapping or wrong "
            "checkpoint must surface as a load error, not silently fall through to "
            "random target weights."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.transformer_template = _resolve_transformer_dir(args.transformer_template)
    checkpoints = _parse_checkpoint_specs(args.checkpoint)
    dtype = _parse_dtype(args.dtype)

    print(f"[load] assets from {args.base_root}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_root / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(args.base_root / "text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(args.base_root / "vae", torch_dtype=dtype)

    all_metrics = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "seed": args.seed,
        "checkpoints": [],
    }
    for label, checkpoint_dir in checkpoints:
        print(f"[run] {label}: {checkpoint_dir}", flush=True)
        metrics = _run_one(
            label=label,
            checkpoint_dir=checkpoint_dir,
            args=args,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            dtype=dtype,
        )
        all_metrics["checkpoints"].append(metrics)
        print(f"[done] {label}: std={metrics['std']:.2f} mean={metrics['mean']:.2f} mp4={metrics['mp4']}", flush=True)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"[done] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
