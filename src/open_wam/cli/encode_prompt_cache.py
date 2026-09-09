#!/usr/bin/env python3
"""Encode unique, recovered dataset instructions once with the reference UMT5."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from open_wam.artifacts.files import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a pilot snapshot with fewer than nine source manifests",
    )
    args = parser.parse_args()

    # Encoding must use the native UMT5 even when an old training cache is exported.
    import csv

    import torch

    from open_wam.configs.loader import load_experiment_config
    from open_wam.data.mixed_video_manifest import _parse_tasks
    from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets

    cfg = load_experiment_config(args.cfg)
    max_tokens, dim = cfg.backbone.max_text_tokens, cfg.backbone.text_dim
    prompts = {""}
    manifest_sha = {}
    coverage = {}
    for path in sorted(args.manifests.glob("*.csv")):
        total = labelled = 0
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                tasks = _parse_tasks(row)
                prompt = tasks[0] if tasks else ""
                prompts.add(prompt)
                total += 1
                labelled += bool(prompt)
        manifest_sha[path.name] = sha256_file(path)
        coverage[path.stem] = {"rows": total, "labelled": labelled}
    expected = {
        f"VPT-{source}.csv"
        for source in ("01", "04", "05", "06", "07", "08", "09", "10R", "10S")
    }
    if not args.allow_subset and set(manifest_sha) != expected:
        raise RuntimeError(
            f"Expected all nine manifest snapshots; missing {expected - set(manifest_sha)}"
        )
    if len(prompts) < 2:
        raise RuntimeError(
            "No recovered nonempty text; refusing to encode an empty corpus."
        )
    prompts = sorted(prompts)
    print(
        json.dumps({"unique_prompts": len(prompts), "coverage": coverage}), flush=True
    )
    args.out.mkdir(parents=True, exist_ok=True)
    embeddings = args.out / "embeddings"
    embeddings.mkdir(exist_ok=True)
    hashes = {}
    for folder in ("text_encoder", "tokenizer"):
        for path in sorted((args.assets / folder).iterdir()):
            if path.is_file():
                hashes[f"{folder}/{path.name}"] = sha256_file(path)
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "max_text_tokens": max_tokens,
        "text_dim": dim,
        "dtype": "bfloat16",
        "storage": "unpadded",
        "encoder_fingerprint": fingerprint,
        "encoder_files_sha256": hashes,
        "manifest_sha256": manifest_sha,
        "coverage": coverage,
        "unique_prompts": len(prompts),
        "complete": False,
    }
    known = [args.out / "index.json", args.out / "index.pending.json"]
    existing = [p for p in known if p.exists()]
    if not existing and next(embeddings.glob("*.pt"), None) is not None:
        raise RuntimeError("Orphaned cache embeddings have no encoder provenance.")
    for previous in existing:
        old = json.loads(previous.read_text())
        for key in ("encoder_fingerprint", "max_text_tokens", "text_dim", "storage"):
            if old.get(key) != metadata[key]:
                raise RuntimeError(f"Existing cache has incompatible {key}.")
    atomic_json(args.out / "index.pending.json", metadata)

    backbone = replace(
        cfg.backbone,
        pretrained_model_name_or_path=str(args.assets),
        load_text_conditioning=True,
        load_wan_vae_frontend=False,
    )
    assets = LingbotReferenceAssets.maybe_load(replace(backbone, prompt_cache=None))
    if not assets.has_text_encoder:
        raise RuntimeError("Reference text encoder failed to load.")
    device = torch.device(args.device)
    entries, written, started = [], 0, time.monotonic()
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        pending = []
        for prompt in batch:
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = embeddings / f"{key}.pt"
            if path.exists():
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    tensor.ndim != 2
                    or not 0 < tensor.shape[0] <= max_tokens
                    or tensor.shape[1] != dim
                ):
                    raise RuntimeError(f"Bad cached shape {path}")
                if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Bad cached dtype/values {path}")
                entries.append(
                    {"sha256": key, "prompt": prompt, "tokens": tensor.shape[0]}
                )
            else:
                pending.append((prompt, key, path))
        if not pending:
            continue
        texts = [item[0] for item in pending]
        encoded = assets.encode_prompts(texts, device=device, dtype=torch.bfloat16)
        if encoded is None or encoded.shape != (len(texts), max_tokens, dim):
            raise RuntimeError("Reference encoder returned an unexpected shape.")
        lengths = (
            assets.tokenizer(
                texts,
                padding="max_length",
                max_length=max_tokens,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            .attention_mask.sum(1)
            .tolist()
        )
        for idx, (prompt, key, path) in enumerate(pending):
            tensor = encoded[idx, : int(lengths[idx])].detach().to("cpu").clone()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Nonfinite embedding for {key}")
            tmp = path.with_suffix(".pt.tmp")
            torch.save(tensor, tmp)
            tmp.replace(path)
            entries.append({"sha256": key, "prompt": prompt, "tokens": tensor.shape[0]})
            written += 1
        assets.text_embedding_cache.clear()
        del encoded
        if start == 0 or (start // args.batch_size) % 20 == 0:
            print(
                json.dumps(
                    {
                        "processed": min(start + args.batch_size, len(prompts)),
                        "total": len(prompts),
                        "written": written,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    entries.sort(key=lambda row: row["sha256"])
    target = args.out / "prompts.jsonl"
    temp = target.with_suffix(".jsonl.tmp")
    with temp.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    temp.replace(target)
    metadata.update(
        complete=True,
        cached_prompts=len(entries),
        prompts_sha256=sha256_file(target),
        embedding_bytes=sum(p.stat().st_size for p in embeddings.glob("*.pt")),
    )
    atomic_json(args.out / "index.json", metadata)
    (args.out / "index.pending.json").unlink(missing_ok=True)
    print(
        json.dumps({"cache_complete": True, **metadata}, ensure_ascii=False), flush=True
    )


if __name__ == "__main__":
    main()
