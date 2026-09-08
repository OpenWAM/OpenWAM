from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/summarize_video_marginal_artifacts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_video_marginal_artifacts",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_sharded_records_requires_complete_global_population(tmp_path: Path) -> None:
    module = _load_script_module()
    common = {
        "config": "config.yaml",
        "data_root": "data",
        "split": "train",
        "segment_frames": 32,
        "rollout_frame_chunk_size": 4,
        "rollout_chunk_horizons": [1, 2],
        "geometry_mode": "fixed",
        "train_window_size": 30,
        "seed": 1729,
        "global_sample_indices": [10, 20, 30],
    }
    paths = []
    for shard, records in enumerate(
        ([{"ordinal": 0}, {"ordinal": 2}], [{"ordinal": 1}])
    ):
        path = tmp_path / f"shard{shard}.json"
        path.write_text(json.dumps({**common, "records": records}))
        paths.append(path)

    records, _ = module._load_sharded_records(paths)

    assert [record["ordinal"] for record in records] == [0, 1, 2]


def test_load_sharded_records_rejects_missing_ordinal(tmp_path: Path) -> None:
    module = _load_script_module()
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "config": "config.yaml",
                "data_root": "data",
                "split": "train",
                "segment_frames": 32,
                "rollout_frame_chunk_size": 4,
                "rollout_chunk_horizons": [1],
                "geometry_mode": "fixed",
                "train_window_size": 30,
                "seed": 1729,
                "global_sample_indices": [10, 20],
                "records": [{"ordinal": 1}],
            }
        )
    )

    with pytest.raises(ValueError, match="full global sample population"):
        module._load_sharded_records([path])


def test_derive_latent_layout_preserves_view_slots() -> None:
    module = _load_script_module()

    @dataclass(frozen=True)
    class Placement:
        source_name: str
        canonical_name: str
        top: int
        left: int
        height: int
        width: int

    config = SimpleNamespace(
        data=SimpleNamespace(
            canonical_height=128,
            canonical_width=256,
            view_layout=(
                Placement("agent", "image", 0, 0, 128, 128),
                Placement("wrist", "wrist_image", 0, 128, 128, 128),
            ),
        )
    )

    layout = module._derive_latent_layout(
        config=config,
        latent_height=8,
        latent_width=16,
    )

    assert layout["canvas_height"] == 8
    assert layout["canvas_width"] == 16
    assert layout["placements"][1]["left"] == 8
    assert layout["placements"][1]["width"] == 8


def test_rgb_metrics_are_identity_for_matching_videos() -> None:
    module = _load_script_module()
    video = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 2, 2, 3)

    metrics = module._rgb_metrics(video, video)

    assert metrics["rgb_mse"] == pytest.approx(0.0)
    assert metrics["rgb_psnr_db"] == float("inf")
    assert metrics["rgb_ssim"] == pytest.approx(1.0)


@pytest.mark.parametrize("field", [
    "transformer_dir", "text_source", "guidance_scale", "max_train_chunk_size",
    "skip_train_metrics", "num_shards", "schema_version",
])
def test_shards_reject_different_experiment_contracts(tmp_path: Path, field: str) -> None:
    module = _load_script_module()
    paths = []
    for ordinal in range(2):
        path = tmp_path / f"shard{ordinal}.json"
        path.write_text(json.dumps({
            "global_sample_indices": [10, 20],
            "records": [{"ordinal": ordinal}],
            field: ordinal,
        }))
        paths.append(path)
    with pytest.raises(ValueError, match=field):
        module._load_sharded_records(paths)


def test_shards_compare_provenance_but_allow_different_commands(tmp_path: Path) -> None:
    module = _load_script_module()
    paths = []
    for ordinal in range(2):
        path = tmp_path / f"shard{ordinal}.json"
        path.write_text(json.dumps({
            "global_sample_indices": [10, 20],
            "records": [{"ordinal": ordinal}],
            "provenance": {
                "command_argv": ["eval", f"--shard-index={ordinal}"],
                "source": {"commit": "same", "dirty": False},
                "checkpoint": {"path": "weights", "size_bytes": 123},
            },
        }))
        paths.append(path)
    assert len(module._load_sharded_records(paths)[0]) == 2
    changed = json.loads(paths[1].read_text())
    changed["provenance"]["checkpoint"]["size_bytes"] = 456
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance 'checkpoint'"):
        module._load_sharded_records(paths)


def test_rgb_only_does_not_request_fvd_clips() -> None:
    module = _load_script_module()
    args = SimpleNamespace(uva_root=None, i3d_checkpoint=None, fvd_horizons="1")
    assert module._requested_fvd_horizons(args, (3,)) == ()
    args.uva_root = "uva"
    with pytest.raises(ValueError, match="requires both"):
        module._requested_fvd_horizons(args, (3,))
    args.i3d_checkpoint = "i3d.pt"
    with pytest.raises(ValueError, match="not present"):
        module._requested_fvd_horizons(args, (3,))
    args.fvd_horizons = "3"
    assert module._requested_fvd_horizons(args, (3,)) == (3,)


def test_perfect_psnr_has_valid_json_and_undefined_population_std() -> None:
    module = _load_script_module()
    summary = module._summary([30.0, float("inf")])
    assert summary["mean"] == float("inf")
    assert summary["std"] is None
    serialized = json.loads(json.dumps(module._json_metrics(summary), allow_nan=False))
    assert serialized["mean"] == "Infinity"
    assert serialized["min"] == 30.0
    with pytest.raises(ValueError, match="NaN"):
        module._summary([float("nan")])
