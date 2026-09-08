from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/evaluate_video_marginal_checkpoint.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_video_marginal_checkpoint",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evenly_spaced_indices_cover_dataset_in_order() -> None:
    module = _load_script_module()

    assert module._evenly_spaced_indices(10, 4) == (1, 3, 6, 8)
    assert module._evenly_spaced_indices(3, 10) == (0, 1, 2)


def test_cli_keeps_training_and_fixed_geometry_explicit() -> None:
    module = _load_script_module()
    parser = module.build_argument_parser()
    required = [
        "--cfg",
        "config.yaml",
        "--checkpoint",
        "checkpoint",
        "--data-root",
        "data",
        "--empty-text-embedding",
        "empty.pt",
        "--reference-assets-root",
        "assets",
        "--output-json",
        "report.json",
    ]

    assert parser.parse_args(required).geometry_mode == "training_random"
    assert (
        parser.parse_args([*required, "--geometry-mode", "fixed"]).geometry_mode
        == "fixed"
    )


def test_fixed_geometry_is_a_dataset_view_not_a_model_contract() -> None:
    module = _load_script_module()

    @dataclass(frozen=True)
    class SampleConfig:
        randomize_geometry: bool = True

    @dataclass(frozen=True)
    class DataConfig:
        sample_construction: SampleConfig

    model_data = DataConfig(sample_construction=SampleConfig())
    fixed_data = module._diagnostic_data_config(
        model_data,
        module.GeometryMode.FIXED,
    )

    assert model_data.sample_construction.randomize_geometry is True
    assert fixed_data.sample_construction.randomize_geometry is False
    assert (
        module._diagnostic_data_config(
            model_data,
            module.GeometryMode.TRAINING_RANDOM,
        )
        is model_data
    )


def test_train_video_flow_regions_separates_startup_from_later_frames() -> None:
    module = _load_script_module()
    flow_pred = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1, 1)
    video = SimpleNamespace(
        flow_pred=flow_pred,
        targets=torch.zeros_like(flow_pred),
        future_loss_mask=torch.ones_like(flow_pred, dtype=torch.bool),
    )
    output = SimpleNamespace(
        policy_output=SimpleNamespace(
            decoder_artifacts=SimpleNamespace(payload=SimpleNamespace(video=video))
        )
    )

    metrics = module._train_video_flow_regions(output, startup_frames=2)

    assert metrics["startup_frames_train_video_flow_loss"] == pytest.approx(2.5)
    assert metrics["later_frames_train_video_flow_loss"] == pytest.approx(12.5)


def test_rollout_metrics_compare_motion_and_static_baseline() -> None:
    module = _load_script_module()
    rollout = SimpleNamespace(
        observed_latent_frames=1,
        future_latent_frames=2,
        predicted_latents=torch.tensor([0.0, 1.0, 2.0]).reshape(1, 1, 3, 1, 1),
        target_latents=torch.tensor([0.0, 1.0, 3.0]).reshape(1, 1, 3, 1, 1),
    )

    metrics = module._rollout_metrics(rollout)

    assert metrics["full_denoise_future_latent_mse"] == pytest.approx(0.5)
    assert metrics["static_baseline_future_latent_mse"] == pytest.approx(5.0)
    assert metrics["model_vs_static_mse_ratio"] == pytest.approx(0.1)
    assert metrics["temporal_delta_latent_mse"] == pytest.approx(0.5)
    assert metrics["endpoint_displacement_latent_mse"] == pytest.approx(1.0)
    assert metrics["endpoint_displacement_cosine"] == pytest.approx(1.0)
    assert metrics["predicted_motion_energy"] == pytest.approx(1.0)
    assert metrics["target_motion_energy"] == pytest.approx(2.5)


def test_chunk_horizon_metrics_score_nested_recurrent_prefixes() -> None:
    module = _load_script_module()
    rollout = SimpleNamespace(
        observed_latent_frames=1,
        future_latent_frames=2,
        predicted_latents=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(
            1, 1, 5, 1, 1
        ),
        target_latents=torch.tensor([0.0, 1.0, 3.0, 3.0, 5.0]).reshape(
            1, 1, 5, 1, 1
        ),
    )

    metrics = module._horizon_metrics(
        rollout,
        chunk_horizons=(1, 2),
        frame_chunk_size=2,
    )

    assert metrics["h1_full_denoise_future_latent_mse"] == pytest.approx(0.5)
    assert metrics["h2_full_denoise_future_latent_mse"] == pytest.approx(0.5)
    assert metrics["h1_endpoint_displacement_latent_mse"] == pytest.approx(1.0)
    assert metrics["h2_endpoint_displacement_latent_mse"] == pytest.approx(1.0)


def test_chunk_horizons_are_positive_sorted_and_unique() -> None:
    module = _load_script_module()

    assert module._parse_chunk_horizons("8,1,4,2,4") == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="positive integers"):
        module._parse_chunk_horizons("1,0")


def test_sample_sharding_preserves_global_ordinals() -> None:
    module = _load_script_module()
    indices = (10, 20, 30, 40, 50)

    assert module._shard_indexed_samples(
        indices,
        shard_index=1,
        num_shards=2,
    ) == ((1, 20), (3, 40))
