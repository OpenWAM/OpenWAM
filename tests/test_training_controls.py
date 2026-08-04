from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from torch import nn

from open_wam.configs import (
    DualExpertPolicyConfig,
    ParallelStreamPolicyConfig,
    TrainingConfig,
    load_experiment_config,
)
from open_wam.configs.enums import (
    AttachSite,
    CurrentBlockCoupling,
    DualExpertRuntimeMode,
    GeneralistDenoisingMode,
    ParallelRuntimeMode,
    TrainingComponentSelector,
)
from open_wam.data import build_synthetic_latent_batch
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.policy_variants.dual_expert.packed_block import (
    DualExpertPackedBlockStack,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training import apply_training_component_controls

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
        policy_variant=replace(config.policy_variant, action_norm_method="none"),
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


def test_apply_training_component_controls_supports_dual_expert_action_expert_selector() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    pipeline = build_variant_pipeline_from_config(config)

    report = apply_training_component_controls(pipeline, config.training)

    assert report.trainable_components == ("policy_variant.action_expert",)
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.frontend.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.parameters())
    assert all(parameter.requires_grad for parameter in pipeline.policy_variant.action_expert.parameters())


def _build_dual_expert_packed_smoke_pipeline():
    """Build the DualExpert smoke pipeline with packed coupling on (CPU-only).

    ``dual_expert_robotwin_smoke.yaml`` keeps the backbone/action-expert tiny (1
    layer, hidden=256), so the pipeline is cheap to construct and we can
    exercise the ownership-transfer surgery and selector resolution without
    spinning up the full LingBot-scale stack.
    """

    from dataclasses import replace as _replace

    from open_wam.configs.enums import (
        CurrentBlockCoupling,
        DualExpertRuntimeMode,
        ProprioContextMode,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    policy_variant_config = _replace(
        config.policy_variant,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        runtime_mode=DualExpertRuntimeMode.NON_JOINT_TWO_STREAM,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
    )
    config = _replace(config, policy_variant=policy_variant_config)
    pipeline = build_variant_pipeline_from_config(config)
    return config, pipeline


def _non_proprio_core_parameters(pipeline):
    return [
        parameter
        for name, parameter in pipeline.visual_tower.core.named_parameters()
        if not name.startswith("proprio_context_encoder.")
        and not name.startswith("proprio_hidden_context_encoder.")
    ]


def test_packed_coupling_action_expert_selector_only_trains_action_side() -> None:
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
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
    # Visual tower core (non-block parts) is NOT trainable except the zero-init
    # proprio adapter, which must learn even in action-only runs.
    assert all(not parameter.requires_grad for parameter in _non_proprio_core_parameters(pipeline))
    assert all(
        parameter.requires_grad
        for parameter in pipeline.visual_tower.core.proprio_hidden_context_encoder.parameters()
    )


def test_packed_coupling_runtime_backbone_selector_only_trains_video_side() -> None:
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
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
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
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
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
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
    assert all(not parameter.requires_grad for parameter in _non_proprio_core_parameters(pipeline))
    assert all(
        parameter.requires_grad
        for parameter in pipeline.visual_tower.core.proprio_hidden_context_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in pipeline.policy_variant.action_expert.action_embedder.parameters()
    )


def test_packed_legacy_restore_transfers_block_ownership_once() -> None:
    _, pipeline = _build_dual_expert_packed_smoke_pipeline()
    packed_stack = pipeline.policy_variant.packed_block_stack
    assert packed_stack is not None
    video_block_ids = [id(packed_block.video_block) for packed_block in packed_stack.packed_blocks]
    action_block_ids = [id(packed_block.action_block) for packed_block in packed_stack.packed_blocks]
    assert len(pipeline.visual_tower.core.blocks) == 0
    assert len(pipeline.policy_variant.action_expert.blocks) == 0

    restored = pipeline.policy_variant.restore_packed_blocks_for_legacy_inference(pipeline.visual_tower)

    assert restored is True
    assert pipeline.policy_variant.packed_block_stack is None
    assert [id(block) for block in pipeline.visual_tower.core.blocks] == video_block_ids
    assert [id(block) for block in pipeline.policy_variant.action_expert.blocks] == action_block_ids
    assert not any(isinstance(module, DualExpertPackedBlockStack) for module in pipeline.modules())
    assert not any("packed_block_stack" in key for key in pipeline.state_dict())
    assert pipeline.policy_variant.restore_packed_blocks_for_legacy_inference(pipeline.visual_tower) is False


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


def test_additive_proprio_context_encoder_trains_with_action_only_selector() -> None:
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
    encoder = pipeline.visual_tower.core.proprio_hidden_context_encoder
    assert encoder is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("policy_variant.action_expert",),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.patch_embedding_mlp.parameters())


def test_additive_proprio_context_encoder_can_be_explicitly_frozen() -> None:
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
    encoder = pipeline.visual_tower.core.proprio_hidden_context_encoder
    assert encoder is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("policy_variant.action_expert",),
            frozen_components=("visual_tower.proprio_context_encoder",),
        ),
    )

    apply_training_component_controls(pipeline, config.training)

    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_additive_proprio_context_encoder_respects_broad_visual_core_freeze() -> None:
    config, pipeline = _build_dual_expert_packed_smoke_pipeline()
    encoder = pipeline.visual_tower.core.proprio_hidden_context_encoder
    assert encoder is not None
    config = replace(
        config,
        training=replace(
            config.training,
            trainable_components=("policy_variant.action_expert",),
            frozen_components=("visual_tower.core",),
        ),
    )

    report = apply_training_component_controls(pipeline, config.training)

    assert TrainingComponentSelector.VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER not in report.trainable_components
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


class _TinyGeneralistModePipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        core = SharedVideoTransformerCore(
            SharedVideoTransformerConfig(
                hidden_size=16,
                num_layers=1,
                num_heads=2,
                attention_head_dim=8,
                text_dim=8,
                freq_dim=4,
                patch_size_t=1,
                patch_size_h=1,
                patch_size_w=1,
            ),
            action_dim=2,
        )
        core.configure_generalist_mode_context_encoder(enabled=True)
        self.visual_tower = nn.Module()
        self.visual_tower.core = core
        self.policy_variant = nn.Module()
        self.policy_variant.config = ParallelStreamPolicyConfig(
            hidden_size=16,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            variant_profile="generalist_joint_denoising",
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_condition_on_action=True,
            generalist_mode_text_token=True,
        )
        self.policy_variant.proj = nn.Linear(1, 1)
        self.action_decoder = nn.Linear(1, 1)


def test_generalist_mode_context_encoder_trains_with_frozen_backbone_selector() -> None:
    pipeline = _TinyGeneralistModePipeline()
    encoder = pipeline.visual_tower.core.generalist_mode_context_encoder
    assert encoder is not None

    report = apply_training_component_controls(
        pipeline,
        TrainingConfig(trainable_components=("policy_variant",)),
    )

    assert TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER in report.trainable_components
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.patch_embedding_mlp.parameters())


def test_dual_expert_generalist_mode_context_encoder_trains_with_frozen_backbone_selector() -> None:
    pipeline = _TinyGeneralistModePipeline()
    pipeline.policy_variant.config = DualExpertPolicyConfig(
        hidden_size=16,
        attach_site=AttachSite.POST_VISUAL_CORE,
        runtime_mode=DualExpertRuntimeMode.NON_JOINT_TWO_STREAM,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        generalist_denoising_mode_probs={GeneralistDenoisingMode.JOINT: 1.0},
        generalist_mode_text_token=True,
    )
    encoder = pipeline.visual_tower.core.generalist_mode_context_encoder
    assert encoder is not None

    report = apply_training_component_controls(
        pipeline,
        TrainingConfig(trainable_components=("policy_variant",)),
    )

    assert TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER in report.trainable_components
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in pipeline.visual_tower.core.patch_embedding_mlp.parameters())


def test_generalist_mode_context_encoder_can_be_explicitly_frozen() -> None:
    pipeline = _TinyGeneralistModePipeline()
    encoder = pipeline.visual_tower.core.generalist_mode_context_encoder
    assert encoder is not None

    report = apply_training_component_controls(
        pipeline,
        TrainingConfig(
            trainable_components=("policy_variant",),
            frozen_components=("visual_tower.generalist_mode_context_encoder",),
        ),
    )

    assert TrainingComponentSelector.VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER not in report.trainable_components
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
