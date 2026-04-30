from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from .external import _bootstrap_libero_config_without_prompt


BENCHMARK = "libero_10"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic/random LingBot-VA LIBERO-10 manifests.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--episodes-per-kind", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260430)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument(
        "--sample-kinds",
        type=str,
        default="deterministic,random",
        help="Comma-separated subset of deterministic,random.",
    )
    args = parser.parse_args()

    sample_kinds = tuple(part.strip() for part in args.sample_kinds.split(",") if part.strip())
    manifest = build_manifest(
        sample_kinds=sample_kinds,
        episodes_per_kind=args.episodes_per_kind,
        random_seed=args.random_seed,
        episode_seed=args.episode_seed,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(json.dumps(_summary(manifest), indent=2, sort_keys=True))


def build_manifest(
    *,
    sample_kinds: tuple[str, ...] = ("deterministic", "random"),
    episodes_per_kind: int = 100,
    random_seed: int = 20260430,
    episode_seed: int = 0,
) -> dict[str, Any]:
    if episodes_per_kind <= 0:
        raise ValueError("episodes_per_kind must be positive.")
    unsupported = set(sample_kinds) - {"deterministic", "random"}
    if unsupported:
        raise ValueError(f"Unsupported sample kinds: {sorted(unsupported)}")

    _bootstrap_libero_config_without_prompt()
    from libero.libero import benchmark  # type: ignore

    benchmark_instance = benchmark.get_benchmark_dict()[BENCHMARK]()
    task_count = _task_count(benchmark_instance)
    init_counts = _init_counts(benchmark_instance, task_count)
    min_init_count = min(init_counts.values())
    total_pairs = task_count * min_init_count
    if episodes_per_kind > total_pairs:
        raise ValueError(
            f"Cannot sample {episodes_per_kind} unique episodes from {BENCHMARK}; "
            f"only {total_pairs} task/init pairs are available."
        )

    episodes: list[dict[str, Any]] = []
    if "deterministic" in sample_kinds:
        for sample_index, (task_id, episode_idx) in enumerate(
            _deterministic_pairs(task_count, min_init_count, episodes_per_kind)
        ):
            episodes.append(
                _episode_row(
                    benchmark_instance,
                    task_id,
                    episode_idx,
                    sample_kind="deterministic",
                    sample_index=sample_index,
                    episode_seed=episode_seed,
                )
            )
    if "random" in sample_kinds:
        for sample_index, (task_id, episode_idx) in enumerate(
            _random_pairs(task_count, min_init_count, episodes_per_kind, seed=random_seed)
        ):
            episodes.append(
                _episode_row(
                    benchmark_instance,
                    task_id,
                    episode_idx,
                    sample_kind="random",
                    sample_index=sample_index,
                    episode_seed=episode_seed,
                )
            )

    return {
        "metadata": {
            "generator": "baselines.lingbot_va.generate_libero10_manifest",
            "benchmark": BENCHMARK,
            "sample_kinds": list(sample_kinds),
            "episodes_per_kind": episodes_per_kind,
            "random_seed": random_seed,
            "episode_seed": episode_seed,
            "task_count": task_count,
            "init_count_min": min_init_count,
            "init_count_max": max(init_counts.values()),
            "total_unique_pairs_at_min_init_count": total_pairs,
        },
        "episodes": episodes,
    }


def _task_count(benchmark_instance) -> int:
    task_count = getattr(benchmark_instance, "n_tasks", None)
    if task_count is not None:
        return int(task_count)
    return len(benchmark_instance.tasks)


def _init_counts(benchmark_instance, task_count: int) -> dict[int, int]:
    return {task_id: len(benchmark_instance.get_task_init_states(task_id)) for task_id in range(task_count)}


def _deterministic_pairs(task_count: int, init_count: int, count: int) -> list[tuple[int, int]]:
    return [(index % task_count, (index // task_count) % init_count) for index in range(count)]


def _random_pairs(task_count: int, init_count: int, count: int, *, seed: int) -> list[tuple[int, int]]:
    pairs = [(task_id, episode_idx) for task_id in range(task_count) for episode_idx in range(init_count)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:count]


def _episode_row(
    benchmark_instance,
    task_id: int,
    episode_idx: int,
    *,
    sample_kind: str,
    sample_index: int,
    episode_seed: int,
) -> dict[str, Any]:
    task = benchmark_instance.get_task(task_id)
    return {
        "benchmark": BENCHMARK,
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "seed": int(episode_seed),
        "sample_kind": sample_kind,
        "sample_index": int(sample_index),
        "sample_id": f"{BENCHMARK}_{sample_kind}_{sample_index:03d}",
        "task_name": task.name,
        "prompt": task.language,
    }


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in manifest["episodes"]:
        sample_kind = row["sample_kind"]
        counts[sample_kind] = counts.get(sample_kind, 0) + 1
    return {
        "episodes": len(manifest["episodes"]),
        "counts": counts,
        "metadata": manifest["metadata"],
    }


if __name__ == "__main__":
    main()
