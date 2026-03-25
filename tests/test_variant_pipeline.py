from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.data import build_synthetic_batch
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_pipeline(config_path: Path) -> tuple:
    config = load_experiment_config(config_path)
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state)
    return config, pipeline, batch, train_batch


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("post_latent_robotwin.yaml", 6),
        ("post_decoded_robotwin.yaml", 6),
        ("register_attached_robotwin.yaml", 6),
        ("parallel_stream_robotwin.yaml", 8),
    ],
)
def test_variant_pipeline_train_and_infer_shapes(config_name: str, expected_horizon: int) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.data.action_schema.action_dim,
    )
    assert infer_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.data.action_schema.action_dim,
    )
