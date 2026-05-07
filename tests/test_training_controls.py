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


def _build_mot_packed_smoke_pipeline():
    """Build the MoT smoke pipeline with packed coupling on (CPU-only).

    ``mot_robotwin_smoke.yaml`` keeps the backbone/action-expert tiny (1
    layer, hidden=256), so the pipeline is cheap to construct and we can
    exercise the ownership-transfer surgery and selector resolution without
    spinning up the full LingBot-scale stack.
    """

    from dataclasses import replace as _replace

    from open_wam.configs.enums import (
        CurrentBlockCoupling,
        MoTRuntimeMode,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    policy_variant_config = _replace(
        config.policy_variant,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
    )
    config = _replace(config, policy_variant=policy_variant_config)
    pipeline = build_variant_pipeline_from_config(config)
    return config, pipeline


def test_packed_coupling_action_expert_selector_only_trains_action_side() -> None:
    config, pipeline = _build_mot_packed_smoke_pipeline()
    assert pipeline.policy_variant.packed_block_stack is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("policy_variant.action_expert",),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    # action_expert (embedder/conditioner/proj) is trainable.
    assert all(
        parameter.requires_grad
        for parameter in pipeline.policy_variant.action_expert.action_embedder.parameters()
    )
    # Each packed_block.action_block is trainable.
    for packed_block in pipeline.policy_variant.packed_block_stack.packed_blocks:
        assert all(parameter.requires_grad for parameter in packed_block.action_block.parameters())
        # And video_block is NOT trainable.
        assert all(not parameter.requires_grad for parameter in packed_block.video_block.parameters())
    # Visual tower core (non-block parts) is NOT trainable.
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())


def test_packed_coupling_runtime_backbone_selector_only_trains_video_side() -> None:
    config, pipeline = _build_mot_packed_smoke_pipeline()
    assert pipeline.policy_variant.packed_block_stack is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("visual_tower.runtime_backbone",),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    # Visual tower core (non-block parts: patch_embed, norm_out, scale_shift_table) is trainable.
    assert any(
        parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters()
    )
    # Each packed_block.video_block is trainable.
    for packed_block in pipeline.policy_variant.packed_block_stack.packed_blocks:
        assert all(parameter.requires_grad for parameter in packed_block.video_block.parameters())
        # And action_block is NOT trainable.
        assert all(not parameter.requires_grad for parameter in packed_block.action_block.parameters())
    # action_expert (embedder/conditioner/proj) is NOT trainable.
    assert all(
        not parameter.requires_grad
        for parameter in pipeline.policy_variant.action_expert.action_embedder.parameters()
    )


def test_packed_coupling_combined_selector_trains_both_sides() -> None:
    config, pipeline = _build_mot_packed_smoke_pipeline()
    assert pipeline.policy_variant.packed_block_stack is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=(
                "policy_variant.action_expert",
                "visual_tower.runtime_backbone",
            ),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    for packed_block in pipeline.policy_variant.packed_block_stack.packed_blocks:
        assert all(parameter.requires_grad for parameter in packed_block.video_block.parameters())
        assert all(parameter.requires_grad for parameter in packed_block.action_block.parameters())
    assert any(
        parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in pipeline.policy_variant.action_expert.action_embedder.parameters()
    )


def test_packed_coupling_freeze_video_train_action() -> None:
    config, pipeline = _build_mot_packed_smoke_pipeline()
    assert pipeline.policy_variant.packed_block_stack is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("policy_variant.action_expert",),
            frozen_components=("visual_tower.runtime_backbone",),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    # Video side fully frozen even though the previous resolver collision
    # would otherwise have re-frozen action blocks.
    for packed_block in pipeline.policy_variant.packed_block_stack.packed_blocks:
        assert all(not parameter.requires_grad for parameter in packed_block.video_block.parameters())
        assert all(parameter.requires_grad for parameter in packed_block.action_block.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())
    assert all(
        parameter.requires_grad
        for parameter in pipeline.policy_variant.action_expert.action_embedder.parameters()
    )


def test_apply_training_component_controls_supports_action_decoder_adapter_selector() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("action_decoder.adapters",),
        ),
    )
    pipeline = build_variant_pipeline_from_config(config)

    report = apply_training_component_controls(pipeline, config.training)

    assert report.trainable_components == ("action_decoder.adapters",)
    assert report.trainable_parameters > 0
    assert report.trainable_parameters < report.total_parameters
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.action_decoder.action_expert.blocks.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.action_decoder.action_expert.action_embedder.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.action_decoder.action_expert.context_proj.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.action_decoder.action_expert.action_proj_out.parameters())
