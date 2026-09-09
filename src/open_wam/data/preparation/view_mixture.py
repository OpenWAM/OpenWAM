"""Validate the single-view and RGB multi view composition of a pretraining snapshot."""

import csv
from pathlib import Path


def summarize_views(record):
    sources = record.get("sources", {})
    return {
        "single_clips": sum(int(row["single_clips"]) for row in sources.values()),
        "multi_view_clips": sum(int(row["multi_view_clips"]) for row in sources.values()),
        "multiview_sources": sorted(
            name for name, row in sources.items() if int(row["multi_view_clips"]) > 0
        ),
    }


def validate_view_mixture(snapshot, record, *, require_multiview=False):
    """For a mixed run, verify actual CSV rows rather than trusting summary counts."""
    summary = summarize_views(record)
    summary["mixed_pretraining_required"] = bool(require_multiview)
    if not require_multiview:
        return summary
    manifest_sources = {Path(name).stem for name in record["manifests_sha256"]}
    if manifest_sources != set(record["sources"]):
        raise ValueError(
            "Snapshot source summaries differ from actual manifest sources"
        )
    if summary["single_clips"] <= 0 or summary["multi_view_clips"] <= 0:
        raise ValueError(
            "Mixed pretraining requires both single-view and RGB multi view clips; "
            "publish a snapshot containing both validated manifest sets."
        )
    root = Path(snapshot)
    for name in sorted(record["manifests_sha256"]):
        source = Path(name).stem
        counts = {"single_clips": 0, "multi_view_clips": 0}
        with (root / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_id") != source:
                    raise ValueError(f"Manifest source identity differs from {source}")
                augmentation = row.get("augmentation")
                if augmentation == "single_view":
                    counts["single_clips"] += 1
                elif augmentation == "multi_view":
                    counts["multi_view_clips"] += 1
                else:
                    raise ValueError(
                        f"Unknown pretraining view representation: {augmentation!r}"
                    )
        expected = record["sources"][source]
        if (
            counts["single_clips"] != int(expected["single_clips"])
            or counts["multi_view_clips"] != int(expected["multi_view_clips"])
            or sum(counts.values()) != int(expected["clips"])
        ):
            raise ValueError(
                f"Actual manifest view counts differ from snapshot: {source}"
            )
    summary["manifest_view_counts_verified"] = True
    return summary
