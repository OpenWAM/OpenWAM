from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (  # noqa: E402
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
)
from open_wam.configs import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mimic step-based train-dataloader draws for hierarchical fixed-segment "
            "configs and report observed trajectory/start coverage."
        )
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Number of global sampler draws to inspect. Defaults to one dataset-length pass per epoch.",
    )
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--materialize-batches", type=int, default=0)
    parser.add_argument(
        "--require-no-invalid",
        action="store_true",
        help="Exit non-zero if any emitted sample key is outside the eligible training-data range.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.draws is not None and args.draws <= 0:
        raise SystemExit("--draws must be positive when provided.")
    if args.world_size <= 0:
        raise SystemExit("--world-size must be positive.")
    if args.materialize_batches < 0:
        raise SystemExit("--materialize-batches must be non-negative.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    train_dataset, _ = build_train_val_latent_datasets(config.data)

    eligible_iter = getattr(train_dataset, "iter_hierarchical_eligible_start_keys", None)
    resolve_key = getattr(train_dataset, "resolve_hierarchical_sample_key", None)
    if not callable(eligible_iter) or not callable(resolve_key):
        raise SystemExit(
            "The selected train dataset does not expose hierarchical sampler coverage hooks. "
            f"dataset_type={config.data.dataset_type!r}"
        )

    eligible_keys = set(eligible_iter())
    if not eligible_keys:
        raise SystemExit("No eligible hierarchical sampler keys were discovered.")

    batch_size = int(args.batch_size or config.data.train_batch_size)
    report = build_coverage_report(
        train_dataset=train_dataset,
        eligible_keys=eligible_keys,
        epochs=args.epochs,
        draws=args.draws,
        world_size=args.world_size,
        batch_size=batch_size,
        resolve_key=resolve_key,
    )
    if args.materialize_batches:
        report["materialized_batches"] = materialize_dataloader_batches(
            train_dataset=train_dataset,
            batch_size=batch_size,
            max_batches=args.materialize_batches,
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    if args.require_no_invalid and not report["ok"]:
        raise SystemExit(1)


def build_coverage_report(
    *,
    train_dataset: Any,
    eligible_keys: set[tuple[int, ...]],
    epochs: int,
    draws: int | None,
    world_size: int,
    batch_size: int,
    resolve_key: Any,
) -> dict[str, Any]:
    covered_keys: set[tuple[int, ...]] = set()
    task_counts: Counter[str] = Counter()
    window_counts: Counter[int] = Counter()
    start_counts: Counter[int] = Counter()
    out_of_range: list[dict[str, Any]] = []
    batch_task_diversity: list[int] = []
    batch_window_diversity: list[int] = []
    effective_frame_counts: list[int] = []
    supervised_frame_counts: list[int] = []
    loss_frame_counts: list[int] = []
    head_padded_frame_counts: list[int] = []
    tail_padded_frame_counts: list[int] = []
    context_prefix_requested_counts: list[int] = []
    context_prefix_in_sample_counts: list[int] = []
    context_prefix_real_counts: list[int] = []
    context_prefix_truncated_counts: list[int] = []
    sampled_chunk_counts: Counter[int] = Counter()
    rank_sample_counts: dict[str, int] = {}

    per_rank_samples = int(math.ceil(len(train_dataset) / float(world_size)))
    sampler_pass_draws = per_rank_samples * int(world_size)
    max_draws = int(draws) if draws is not None else sampler_pass_draws * int(epochs)
    consumed_draws = 0
    epoch_limit = max(int(epochs), int(math.ceil(max_draws / max(1, sampler_pass_draws))))
    for epoch in range(epoch_limit):
        epoch_start = int(epoch) * len(train_dataset)
        epoch_draw_count = min(sampler_pass_draws, max_draws - consumed_draws)
        if epoch_draw_count <= 0:
            break
        rank_indices: dict[int, list[int]] = {rank: [] for rank in range(int(world_size))}
        for local_index in range(epoch_draw_count):
            global_index = epoch_start + local_index
            rank_indices[local_index % int(world_size)].append(global_index)
        for rank, indices in rank_indices.items():
            rank_sample_counts[f"epoch{epoch}/rank{rank}"] = len(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_records = [resolve_key(index) for index in batch_indices]
                batch_task_diversity.append(len({str(record["task_text"]) for record in batch_records}))
                batch_window_diversity.append(
                    len({int(record["trajectory_window_index"]) for record in batch_records})
                )
                for record in batch_records:
                    key = _coverage_key_from_record(record)
                    covered_keys.add(key)
                    task_counts[str(record["task_text"])] += 1
                    window_counts[int(record["trajectory_window_index"])] += 1
                    start_counts[int(record["latent_start"])] += 1
                    if not (int(record["start_min"]) <= int(record["latent_start"]) <= int(record["start_max"])):
                        out_of_range.append(record)
                    if "effective_segment_frames" in record:
                        effective_frame_counts.append(int(record["effective_segment_frames"]))
                    if "supervised_frame_start" in record and "supervised_frame_end" in record:
                        supervised_frame_counts.append(
                            max(0, int(record["supervised_frame_end"]) - int(record["supervised_frame_start"]))
                        )
                    if "loss_frame_start" in record and "loss_frame_end" in record:
                        loss_frame_counts.append(
                            max(0, int(record["loss_frame_end"]) - int(record["loss_frame_start"]))
                        )
                    if "head_padded_frame_count" in record:
                        head_padded_frame_counts.append(int(record["head_padded_frame_count"]))
                    if "tail_padded_frame_count" in record:
                        tail_padded_frame_counts.append(int(record["tail_padded_frame_count"]))
                    if "context_prefix_frames_requested" in record:
                        context_prefix_requested_counts.append(int(record["context_prefix_frames_requested"]))
                    if "context_prefix_frames_in_sample" in record:
                        context_prefix_in_sample_counts.append(int(record["context_prefix_frames_in_sample"]))
                    if "context_prefix_real_frames" in record:
                        context_prefix_real_counts.append(int(record["context_prefix_real_frames"]))
                    if "context_prefix_truncated_frames" in record:
                        context_prefix_truncated_counts.append(int(record["context_prefix_truncated_frames"]))
                    if "sampled_chunk_size" in record:
                        sampled_chunk_counts[int(record["sampled_chunk_size"])] += 1
        consumed_draws += epoch_draw_count

    missing_keys = sorted(eligible_keys.difference(covered_keys))
    extra_keys = sorted(covered_keys.difference(eligible_keys))
    return {
        "ok": not extra_keys and not out_of_range,
        "dataset_length": len(train_dataset),
        "sampler_pass_draw_count": int(sampler_pass_draws),
        "sampler_per_rank_sample_count": int(per_rank_samples),
        "eligible_key_total": len(eligible_keys),
        "eligible_start_total": len(eligible_keys),
        "requested_epochs": int(epochs),
        "epochs_checked": int(epoch_limit),
        "requested_draws": int(max_draws),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "draw_count": sum(rank_sample_counts.values()),
        "unique_covered_key_total": len(covered_keys),
        "unique_covered_start_total": len(covered_keys),
        "unique_key_coverage_rate": len(covered_keys) / float(len(eligible_keys)),
        "unique_start_coverage_rate": len(covered_keys) / float(len(eligible_keys)),
        "missing_key_count": len(missing_keys),
        "missing_start_count": len(missing_keys),
        "missing_start_examples": [_format_coverage_key(key) for key in missing_keys[:20]],
        "extra_key_count": len(extra_keys),
        "extra_start_count": len(extra_keys),
        "extra_start_examples": [_format_coverage_key(key) for key in extra_keys[:20]],
        "out_of_range_count": len(out_of_range),
        "out_of_range_examples": out_of_range[:20],
        "task_counts": dict(sorted(task_counts.items())),
        "window_count": len(window_counts),
        "window_count_min": min(window_counts.values()) if window_counts else 0,
        "window_count_max": max(window_counts.values()) if window_counts else 0,
        "start_min": min(start_counts) if start_counts else None,
        "start_max": max(start_counts) if start_counts else None,
        "negative_start_draw_count": sum(count for start, count in start_counts.items() if int(start) < 0),
        "effective_frame_counts": summarize_ints(effective_frame_counts),
        "supervised_frame_counts": summarize_ints(supervised_frame_counts),
        "loss_frame_counts": summarize_ints(loss_frame_counts),
        "head_padded_frame_counts": summarize_ints(head_padded_frame_counts),
        "tail_padded_frame_counts": summarize_ints(tail_padded_frame_counts),
        "context_prefix_requested_counts": summarize_ints(context_prefix_requested_counts),
        "context_prefix_in_sample_counts": summarize_ints(context_prefix_in_sample_counts),
        "context_prefix_real_counts": summarize_ints(context_prefix_real_counts),
        "context_prefix_truncated_counts": summarize_ints(context_prefix_truncated_counts),
        "sampled_chunk_counts": dict(sorted(sampled_chunk_counts.items())),
        "rank_sample_counts": rank_sample_counts,
        "batch_task_diversity": summarize_ints(batch_task_diversity),
        "batch_window_diversity": summarize_ints(batch_window_diversity),
    }


def _coverage_key_from_record(record: dict[str, Any]) -> tuple[int, ...]:
    key = (int(record["trajectory_window_index"]), int(record["latent_start"]))
    if "sampled_chunk_size" not in record:
        return key
    return (*key, int(record["sampled_chunk_size"]))


def _format_coverage_key(key: tuple[int, ...]) -> dict[str, int]:
    row = {
        "trajectory_window_index": int(key[0]),
        "latent_start": int(key[1]),
    }
    if len(key) >= 3:
        row["sampled_chunk_size"] = int(key[2])
    return row


def materialize_dataloader_batches(
    *,
    train_dataset: Any,
    batch_size: int,
    max_batches: int,
) -> list[dict[str, Any]]:
    sampler = train_dataset.build_train_sampler(world_size=1, rank=0)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(0)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler,
        collate_fn=collate_latent_wam_samples,
    )
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        rows.append(
            {
                "batch_index": int(batch_index),
                "batch_size": len(batch.metadata),
                "task_text": [metadata.get("hierarchical_task_text") for metadata in batch.metadata],
                "trajectory_window_index": [
                    int(metadata.get("trajectory_window_index", -1)) for metadata in batch.metadata
                ],
                "latent_start": [int(metadata.get("virtual_latent_start", 0)) for metadata in batch.metadata],
                "effective_segment_frames": [
                    int(metadata.get("effective_segment_frames", batch.video_latents.shape[2]))
                    for metadata in batch.metadata
                ],
                "sampled_chunk_size": [int(metadata.get("sampled_chunk_size", 0)) for metadata in batch.metadata],
                "sampled_window_size": [int(metadata.get("sampled_window_size", 0)) for metadata in batch.metadata],
                "loss_frame_start": [int(metadata.get("loss_frame_start", 0)) for metadata in batch.metadata],
                "loss_frame_end": [
                    int(metadata.get("loss_frame_end", batch.video_latents.shape[2])) for metadata in batch.metadata
                ],
                "history_frames": [int(metadata.get("history_frames", 0)) for metadata in batch.metadata],
                "context_prefix_frames_requested": [
                    int(metadata.get("context_prefix_frames_requested", 0)) for metadata in batch.metadata
                ],
                "context_prefix_frames_in_sample": [
                    int(metadata.get("context_prefix_frames_in_sample", 0)) for metadata in batch.metadata
                ],
                "valid_action_steps": [int(metadata.get("valid_action_steps", 0)) for metadata in batch.metadata],
                "video_latents_shape": tuple(int(dim) for dim in batch.video_latents.shape),
                "actions_shape": tuple(int(dim) for dim in batch.actions.shape),
            }
        )
    return rows


def summarize_ints(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / float(len(values)),
    }


if __name__ == "__main__":
    main()
