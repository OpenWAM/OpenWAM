from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from open_wam.configs import ParallelStreamPolicyConfig, load_experiment_config
from open_wam.data import build_synthetic_batch, build_synthetic_latent_batch
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_pipeline(config_path: Path) -> tuple:
    config = load_experiment_config(config_path)
    # Joint diffusion variants now run the shared core at every denoising step.
    # Keep the pipeline test on a short smoke rollout so it stays focused on
    # shape/contract coverage rather than full reference-timing parity.
    if not (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.reference_profile is not None
    ):
        config = replace(
            config,
            inference=replace(
                config.inference,
                video_num_inference_steps=min(
                    config.inference.video_num_inference_steps, 2
                ),
                action_num_inference_steps=min(
                    config.inference.action_num_inference_steps, 2
                ),
                joint_num_inference_steps=(
                    min(config.inference.joint_num_inference_steps, 2)
                    if config.inference.joint_num_inference_steps is not None
                    else None
                ),
            ),
        )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=batch.actions, action_mask=batch.action_mask, state=batch.state
    )
    return config, pipeline, batch, train_batch


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("parallel_stream_robotwin_smoke.yaml", 8),
        ("dual_expert_robotwin_smoke.yaml", 8),
    ],
)
def test_variant_pipeline_train_and_infer_shapes(
    config_name: str, expected_horizon: int
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )
    assert infer_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("parallel_stream_libero_raw_smoke.yaml", 16),
    ],
)
def test_raw_libero_variant_pipeline_train_and_infer_shapes(
    config_name: str, expected_horizon: int
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )
    assert infer_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("parallel_stream_robotwin_smoke.yaml", 8),
        ("dual_expert_robotwin_smoke.yaml", 8),
    ],
)
def test_variant_pipeline_train_from_latents_shapes(
    config_name: str, expected_horizon: int
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config = load_experiment_config(config_path)
    pipeline = build_variant_pipeline_from_config(config)
    latent_batch = build_synthetic_latent_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=latent_batch.actions,
        action_mask=latent_batch.action_mask,
        state=latent_batch.state,
        extra={
            "task_text": latent_batch.task_text,
            "metadata": latent_batch.metadata,
            "state_mask": latent_batch.state_mask,
        },
    )

    train_output = pipeline.forward_train_from_latents(
        latent_batch.video_latents,
        train_batch,
        canonical_video=latent_batch.canonical_video,
        text_context=latent_batch.text_context,
        negative_text_context=latent_batch.negative_text_context,
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


def test_causal_video_prediction_pipeline_trains_from_latents(tmp_path: Path) -> None:
    del tmp_path
    config_path = (
        REPO_ROOT / "configs/experiments/causal_video_prediction_robotwin_smoke.yaml"
    )
    config = load_experiment_config(config_path)
    pipeline = build_variant_pipeline_from_config(config)
    latent_batch = build_synthetic_latent_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=latent_batch.actions,
        action_mask=latent_batch.action_mask,
        state=latent_batch.state,
        extra={
            "task_text": latent_batch.task_text,
            "metadata": latent_batch.metadata,
            "state_mask": latent_batch.state_mask,
        },
    )

    train_output = pipeline.forward_train_from_latents(
        latent_batch.video_latents,
        train_batch,
        canonical_video=latent_batch.canonical_video,
        text_context=latent_batch.text_context,
        negative_text_context=latent_batch.negative_text_context,
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        0,
        config.action_decoder.action_dim,
    )
    assert "latent_mse" in train_output.decoder_output.metrics
    assert "predicted_latents" in train_output.decoder_output.aux

    infer_batch = build_synthetic_latent_batch(config.data, batch_size=1)
    infer_output = pipeline.forward_infer_step_from_latents(
        infer_batch.video_latents,
        PolicyInferContext(
            state=infer_batch.state,
            extra={
                "task_text": infer_batch.task_text,
                "metadata": infer_batch.metadata,
            },
        ),
        canonical_video=infer_batch.canonical_video,
        text_context=infer_batch.text_context,
        negative_text_context=infer_batch.negative_text_context,
    )

    assert infer_output.decoder_output.action_pred.shape == (
        1,
        0,
        config.action_decoder.action_dim,
    )
    assert "predicted_latents" in infer_output.decoder_output.aux


def test_variant_pipeline_rejects_unknown_visual_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    monkeypatch.setattr(
        type(pipeline.policy_variant),
        "required_visual_stages",
        lambda _self: ("typo",),
    )

    with pytest.raises(ValueError, match="unsupported visual stage"):
        pipeline._complete_visual_outputs(object())
