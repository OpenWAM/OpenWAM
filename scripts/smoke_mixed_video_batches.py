from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import BatchAdapterName, MixedVideoDataConfig, MixedVideoDecodeSizeMode
from open_wam.data import (
    build_canonical_video_preprocessor,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
)
from open_wam.utils.config_loader import load_experiment_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview mixed-video batches before launching video-only training.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        required=True,
        help="Experiment YAML using data.dataset_type=mixed_video.",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-train-sampler",
        action="store_true",
        help="Use the dataset-provided train sampler when previewing the train split.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    if not isinstance(config.data, MixedVideoDataConfig):
        raise ValueError("smoke_mixed_video_batches.py requires `data.dataset_type: mixed_video`.")

    use_latents = config.trainer.batch_adapter == BatchAdapterName.LATENTS
    if use_latents:
        train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
    else:
        train_dataset, val_dataset = build_train_val_datasets(config.data)
    dataset = train_dataset if args.split == "train" else val_dataset
    sampler = None
    if args.use_train_sampler and args.split == "train":
        build_train_sampler = getattr(dataset, "build_train_sampler", None)
        if callable(build_train_sampler):
            sampler = build_train_sampler(world_size=1, rank=0)
    batch_size = args.batch_size or config.data.train_batch_size
    if config.data.decode_size_mode == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS and batch_size != 1:
        raise ValueError(
            "Mixed-video aspect-ratio-bin mode currently requires --batch-size 1 because samples can have "
            f"different decoded heights/widths; got batch_size={batch_size}."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_latent_wam_samples if use_latents else collate_wam_samples,
    )
    batch = next(iter(loader))
    canonical = None if use_latents else build_canonical_video_preprocessor(config.data)(batch.views)
    source_counts: dict[str, int] = {}
    for metadata in batch.metadata:
        source_id = str(metadata.get("source_id", "unknown"))
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
    report = {
        "split": args.split,
        "dataset_length": len(dataset),
        "used_train_sampler": sampler is not None,
        "source_counts": source_counts,
        "batch_adapter": config.trainer.batch_adapter.value,
        "view_shapes": {} if use_latents else {name: list(value.shape) for name, value in batch.views.items()},
        "video_latents_shape": list(batch.video_latents.shape) if use_latents else None,
        "canonical_video_shape": None if canonical is None else list(canonical.video.shape),
        "metadata_preview": batch.metadata[0] if batch.metadata else {},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
