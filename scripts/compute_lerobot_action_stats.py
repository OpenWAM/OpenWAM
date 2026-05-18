from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    summary = compute_lerobot_action_stats(dataset_root, column=args.column)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def compute_lerobot_action_stats(dataset_root: Path, *, column: str) -> dict[str, Any]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No LeRobot parquet files found under {dataset_root / 'data'}.")

    count = 0
    sum_values: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    min_values: np.ndarray | None = None
    max_values: np.ndarray | None = None
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=[column])
        values = np.asarray(table.column(0).combine_chunks().to_pylist(), dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected column {column!r} in {parquet_path} to be 2D, got shape {values.shape}.")
        if values.shape[0] == 0:
            continue
        if sum_values is None:
            sum_values = values.sum(axis=0)
            sum_squares = (values * values).sum(axis=0)
            min_values = values.min(axis=0)
            max_values = values.max(axis=0)
        else:
            if values.shape[1] != sum_values.shape[0]:
                raise ValueError(
                    f"Action dim changed while reading {column!r}: expected {sum_values.shape[0]}, "
                    f"got {values.shape[1]} in {parquet_path}."
                )
            sum_values += values.sum(axis=0)
            assert sum_squares is not None
            sum_squares += (values * values).sum(axis=0)
            assert min_values is not None and max_values is not None
            min_values = np.minimum(min_values, values.min(axis=0))
            max_values = np.maximum(max_values, values.max(axis=0))
        count += int(values.shape[0])

    if count <= 0 or sum_values is None or sum_squares is None or min_values is None or max_values is None:
        raise ValueError(f"No rows found for column {column!r} under {dataset_root}.")

    mean = sum_values / float(count)
    variance = np.maximum(sum_squares / float(count) - mean * mean, 1e-12)
    std = np.sqrt(variance)
    return {
        "source_root": str(dataset_root),
        "column": column,
        "episodes": len(parquet_paths),
        "rows": int(count),
        "mean": [float(value) for value in mean],
        "variance": [float(value) for value in variance],
        "std": [float(value) for value in std],
        "min": [float(value) for value in min_values],
        "max": [float(value) for value in max_values],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-channel Gaussian stats for a LeRobot action column.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--column", default="action")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
