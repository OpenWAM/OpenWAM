from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from open_wam.data import build_synthetic_latent_batch
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training import apply_training_component_controls
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apply_training_component_controls_can_train_decoder_only() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("action_decoder",),
        ),
    )
    pipeline = build_variant_pipeline_from_config(config)

    report = apply_training_component_controls(pipeline, config.training)

    assert report.trainable_components == ("action_decoder",)
    assert report.trainable_parameters > 0
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.frontend.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.action_decoder.parameters())


def test_parallel_stream_enabled_objectives_can_disable_action_loss() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            enabled_objectives=("latent",),
            latent_loss_weight=1.0,
            action_loss_weight=1.0,
        ),
    )
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

    assert train_output.decoder_output.metrics["weighted_action_loss"].item() == pytest.approx(0.0)
    assert train_output.decoder_output.loss.item() == pytest.approx(
        train_output.decoder_output.metrics["weighted_latent_loss"].item()
    )


def test_apply_training_component_controls_supports_mot_action_expert_selector() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    pipeline = build_variant_pipeline_from_config(config)

    report = apply_training_component_controls(pipeline, config.training)

    assert report.trainable_components == ("policy_variant.action_expert",)
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.frontend.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.policy_variant.action_expert.parameters())
