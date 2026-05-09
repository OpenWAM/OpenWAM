from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/check_hierarchical_sampler_coverage.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_hierarchical_sampler_coverage", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TinyDataset:
    def __len__(self) -> int:
        return 5


def test_coverage_report_uses_distributed_sampler_padding() -> None:
    script = _load_script_module()

    def resolve_key(index: int) -> dict[str, object]:
        latent_start = int(index) % len(_TinyDataset())
        return {
            "task_text": "task",
            "trajectory_window_index": 0,
            "latent_start": latent_start,
            "start_min": 0,
            "start_max": 4,
        }

    report = script.build_coverage_report(
        train_dataset=_TinyDataset(),
        eligible_keys={(0, index) for index in range(5)},
        epochs=1,
        draws=None,
        world_size=4,
        batch_size=1,
        resolve_key=resolve_key,
    )

    assert report["sampler_pass_draw_count"] == 8
    assert report["rank_sample_counts"] == {
        "epoch0/rank0": 2,
        "epoch0/rank1": 2,
        "epoch0/rank2": 2,
        "epoch0/rank3": 2,
    }
    assert report["draw_count"] == 8
    assert report["ok"] is True


def test_coverage_report_tracks_sampled_chunk_as_key_dimension() -> None:
    script = _load_script_module()

    def resolve_key(index: int) -> dict[str, object]:
        latent_start = int(index) % len(_TinyDataset())
        sampled_chunk_size = 1 if int(index) % 2 == 0 else 2
        return {
            "task_text": "task",
            "trajectory_window_index": 0,
            "latent_start": latent_start,
            "sampled_chunk_size": sampled_chunk_size,
            "start_min": 0,
            "start_max": 4,
        }

    report = script.build_coverage_report(
        train_dataset=_TinyDataset(),
        eligible_keys={(0, index, chunk) for index in range(5) for chunk in (1, 2)},
        epochs=1,
        draws=5,
        world_size=1,
        batch_size=1,
        resolve_key=resolve_key,
    )

    assert report["eligible_key_total"] == 10
    assert report["unique_covered_key_total"] == 5
    assert report["missing_key_count"] == 5
    assert report["sampled_chunk_counts"] == {1: 3, 2: 2}
