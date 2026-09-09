"""Resolve the portable video-pretraining recipe against a sealed snapshot."""

import argparse
import json
from pathlib import Path

import yaml

from open_wam.configs.config_paths import EXAMPLE_CONFIG_ROOT
from open_wam.data.preparation.snapshot import load_verified_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--model-assets", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument(
        "--batching", choices=("strict", "padded", "bucket", "packed"), default="bucket"
    )
    parser.add_argument("--bucket-pool-size", type=int, default=1152)
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="Absolute target optimizer step"
    )
    parser.add_argument("--weights", choices=("hours", "balanced"), default="hours")
    text_source = parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument(
        "--online-text",
        action="store_true",
        help="Load UMT5 instead of a prepared prompt cache",
    )
    text_source.add_argument(
        "--text-cache",
        type=Path,
        help="Explicit offline prompt cache root (may be encoded after config generation)",
    )
    parser.add_argument("--text-encoder-fingerprint")
    parser.add_argument(
        "--catalog-spec",
        type=Path,
        help="Optional pinned catalog for latent and prompt tensors",
    )
    parser.add_argument(
        "--object-cache-dir", type=Path, default=Path("~/.cache/openwam/objects")
    )
    parser.add_argument("--object-cache-gib", type=float, default=200)
    parser.add_argument("--object-cache-min-free-gib", type=float, default=8)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument(
        "--require-multiview",
        action="store_true",
        help="Require both original single-view and verified RGB multi view clips",
    )
    args = parser.parse_args()
    record = load_verified_snapshot(args.snapshot)
    expected = {
        f"VPT-{s}" for s in ("01", "04", "05", "06", "07", "08", "09", "10R", "10S")
    }
    if not args.allow_subset and set(record["sources"]) != expected:
        raise ValueError(
            "Expected all nine sources; --allow-subset is for explicit pilot runs"
        )
    unlabelled = {
        source: stats["clips"] - stats["labelled"]
        for source, stats in record["sources"].items()
        if stats["labelled"] != stats["clips"]
    }
    if unlabelled:
        raise ValueError(
            f"Task-prompt pretraining requires native labels for every clip; recover labels or create a separate unconditional recipe: {unlabelled}"
        )
    cfg = yaml.safe_load((EXAMPLE_CONFIG_ROOT / "video_pretraining.yaml").read_text())
    sources = []
    for name in sorted(record["manifests_sha256"]):
        manifest = (args.snapshot / name).resolve()
        source = manifest.stem
        weight = (
            record["sources"][source]["encoded_view_hours"]
            if args.weights == "hours"
            else 1.0
        )
        if weight <= 0:
            raise ValueError("Source has no positive duration")
        sources.append(
            dict(
                source_id=source,
                manifest_csv=str(manifest),
                source_format="latent",
                latent_key="latent",
                sampling_weight=weight,
            )
        )
    from open_wam.data.preparation.view_mixture import validate_view_mixture

    view_mixture = validate_view_mixture(
        args.snapshot, record, require_multiview=args.require_multiview
    )
    if args.require_multiview:
        cfg["name"] = "openwam_single_and_multiview_pretraining"
    cfg["data"]["video_sources"] = sources
    cfg["data"]["train_batch_size"] = args.batch_size
    cfg["data"]["batching"].update(
        mode=args.batching, bucket_pool_size=args.bucket_pool_size
    )
    cfg["backbone"].update(
        pretrained_model_name_or_path=str(args.model_assets.resolve()),
        load_text_conditioning=args.online_text,
    )
    if args.text_encoder_fingerprint and not args.text_cache:
        parser.error("--text-encoder-fingerprint requires --text-cache")
    artifact_cache = None
    if args.catalog_spec:
        artifact_cache = dict(
            catalog_spec_path=str(args.catalog_spec.resolve()),
            cache_dir=str(args.object_cache_dir.expanduser().resolve()),
            max_bytes=int(args.object_cache_gib * 1024**3),
            min_free_bytes=int(args.object_cache_min_free_gib * 1024**3),
        )
        cfg["data"]["artifact_cache"] = artifact_cache
    if args.text_cache:
        cfg["backbone"]["prompt_cache"] = dict(
            root=str(args.text_cache.resolve()),
            encoder_fingerprint=args.text_encoder_fingerprint,
            artifact_cache=artifact_cache,
        )
    cfg["training"]["num_steps"] = args.num_steps
    cfg["trainer"]["default_root_dir"] = str(args.run_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    from open_wam.configs import load_experiment_config

    load_experiment_config(args.out)
    print(
        json.dumps(
            dict(
                config=str(args.out),
                snapshot_sha256=record["snapshot_sha256"],
                source_weights={s["source_id"]: s["sampling_weight"] for s in sources},
                view_mixture=view_mixture,
            )
        )
    )


if __name__ == "__main__":
    main()
