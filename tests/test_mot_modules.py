from __future__ import annotations

import pytest
import torch

import open_wam.configs  # Ensure config/video-backbone modules finish initialization before policy-variant imports.
from open_wam.configs import (
    ActionSchemaConfig,
    ExperimentConfig,
    InferenceConfig,
    MLPActionDecoderConfig,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.policy_variants.mot.contracts import MoTRuntimeState
from open_wam.models.policy_variants.mot.modules import (
    MoTActionExpert,
    init_action_expert_from_video_core,
)
from open_wam.models.policy_variants.mot.runtime import (
    build_mot_attention_mask,
    resolve_mot_condition_latents,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore
from open_wam.pipelines import build_variant_pipeline_from_config


def test_mot_action_expert_pre_and_post_shapes() -> None:
    expert = MoTActionExpert(
        hidden_size=32,
        action_dim=4,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
    )
    actions = torch.randn(2, 6, 4)
    timestep = torch.randint(low=0, high=1000, size=(2, 6))
    context = torch.randn(2, 5, 16)
    context_mask = torch.ones(2, 5, dtype=torch.bool)

    pre = expert.pre_dit(
        action_tokens=actions,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
    )
    pred = expert.post_dit(pre.tokens, pre)

    assert pre.tokens.shape == (2, 6, 32)
    assert pre.freqs.shape[0] == 2
    assert pre.t_mod.shape == (2, 6, 6, 32)
    assert pre.context.shape == (2, 5, 32)
    assert pred.shape == (2, 6, 4)


def test_mot_action_expert_can_copy_shared_video_blocks() -> None:
    video_core = SharedVideoTransformerCore(
        SharedVideoTransformerConfig(
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=4,
    )
    expert = MoTActionExpert(
        hidden_size=32,
        action_dim=4,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
    )

    init_action_expert_from_video_core(action_expert=expert, video_core=video_core)

    for action_block, video_block in zip(expert.blocks, video_core.blocks, strict=True):
        assert torch.allclose(action_block.scale_shift_table, video_block.scale_shift_table)


def test_mot_action_expert_can_interpolate_smaller_ffn_from_shared_video_blocks() -> None:
    video_core = SharedVideoTransformerCore(
        SharedVideoTransformerConfig(
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=4,
    )
    expert = MoTActionExpert(
        hidden_size=32,
        action_dim=4,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
    )

    init_action_expert_from_video_core(
        action_expert=expert,
        video_core=video_core,
        mode="video_weight_interpolate",
    )

    assert expert.blocks[0].ffn.net[0].proj.weight.shape[-1] == 32
    assert torch.isfinite(expert.blocks[0].ffn.net[0].proj.weight).all()


@pytest.mark.parametrize(
    ("condition_mode", "video_prefix_frames", "expected_frames"),
    [
        (MoTConditionMode.FIRST_FRAME, 3, 1),
        (MoTConditionMode.FULL_VIDEO, 3, 4),
        (MoTConditionMode.TEACHER_FORCING_COND_VIDEO, 2, 2),
    ],
)
def test_resolve_mot_condition_latents_selects_expected_frames(
    condition_mode: MoTConditionMode,
    video_prefix_frames: int,
    expected_frames: int,
) -> None:
    video_latents = torch.randn(2, 48, 4, 8, 8)

    selected = resolve_mot_condition_latents(
        video_latents=video_latents,
        condition_mode=condition_mode,
        video_prefix_frames=video_prefix_frames,
        teacher_forcing_video_noise_prob=0.0,
        training=True,
        scheduler=None,
    )

    assert selected.shape == (2, 48, expected_frames, 8, 8)


def test_build_mot_attention_mask_respects_first_frame_visibility() -> None:
    mask = build_mot_attention_mask(
        video_seq_len=8,
        action_seq_len=4,
        device=torch.device("cpu"),
        condition_mode=MoTConditionMode.FIRST_FRAME,
        video_tokens_per_frame=2,
    )

    assert mask.shape == (12, 12)
    assert mask[8:, :2].all()
    assert not mask[8:, 2:8].any()


def test_build_mot_attention_mask_respects_full_video_visibility() -> None:
    mask = build_mot_attention_mask(
        video_seq_len=8,
        action_seq_len=4,
        device=torch.device("cpu"),
        condition_mode=MoTConditionMode.FULL_VIDEO,
        video_tokens_per_frame=2,
    )

    assert mask.shape == (12, 12)
    assert mask[8:, :8].all()


@pytest.mark.parametrize(
    ("condition_mode", "video_prefix_frames", "teacher_forcing_video_noise_prob"),
    [
        (MoTConditionMode.FIRST_FRAME, 1, 0.0),
        (MoTConditionMode.FULL_VIDEO, 1, 0.0),
        (MoTConditionMode.TEACHER_FORCING_COND_VIDEO, 2, 0.0),
    ],
)
def test_mot_variant_train_forward_from_latents_smoke(
    condition_mode: MoTConditionMode,
    video_prefix_frames: int,
    teacher_forcing_video_noise_prob: float,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=MoTPolicyConfig(
            hidden_size=32,
            condition_mode=condition_mode,
            video_prefix_frames=video_prefix_frames,
            teacher_forcing_video_noise_prob=teacher_forcing_video_noise_prob,
            num_action_layers=2,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(actions=torch.randn(2, 4, 4))
    video_latents = torch.randn(2, 48, 4, 8, 8)
    text_context = torch.randn(2, 5, 16)

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=text_context,
    )

    assert output.decoder_output.action_pred.shape == (2, 4, 4)
    assert torch.isfinite(output.decoder_output.loss)


@pytest.mark.parametrize(
    ("condition_mode", "video_prefix_frames"),
    [
        (MoTConditionMode.FIRST_FRAME, 1),
        (MoTConditionMode.FULL_VIDEO, 1),
        (MoTConditionMode.TEACHER_FORCING_COND_VIDEO, 2),
    ],
)
def test_mot_variant_infer_from_latents_smoke(
    condition_mode: MoTConditionMode,
    video_prefix_frames: int,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=MoTPolicyConfig(
            hidden_size=32,
            condition_mode=condition_mode,
            video_prefix_frames=video_prefix_frames,
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=2,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2, action_num_inference_steps=3),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(
        video_latents,
        text_context=text_context,
    )
    output = pipeline._forward_infer_with_visual_outputs(
        visual_outputs,
        context=PolicyInferContext(),
    )

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert torch.isfinite(output.decoder_output.action_pred).all()
    assert isinstance(output.policy_output.next_state.variant_state, MoTRuntimeState)


def test_mot_variant_train_from_latents_supports_joint_action_and_video_objectives() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=MoTPolicyConfig(
            hidden_size=32,
            condition_mode=MoTConditionMode.FIRST_FRAME,
            video_prefix_frames=1,
            num_action_layers=2,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(actions=torch.randn(2, 4, 4))
    video_latents = torch.randn(2, 48, 4, 8, 8)
    text_context = torch.randn(2, 5, 16)

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=text_context,
    )

    assert torch.isfinite(output.decoder_output.loss)
    assert torch.isfinite(output.decoder_output.metrics["weighted_action_diffusion_loss"])
    assert torch.isfinite(output.decoder_output.metrics["weighted_video_diffusion_loss"])
    assert output.decoder_output.aux["predicted_latents"].shape == video_latents.shape


def test_mot_variant_builds_with_interpolated_action_expert_ffn() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        policy_variant=MoTPolicyConfig(
            hidden_size=32,
            action_expert_init_mode=MoTActionExpertInitMode.VIDEO_WEIGHT_INTERPOLATE,
            action_hidden_size=24,
            action_ffn_dim=32,
            num_action_layers=2,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(enabled_objectives=("action",)),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(
        video_latents,
        text_context=text_context,
    )

    pipeline.policy_variant.prepare_infer_state(
        visual_tower=pipeline.visual_tower,
        visual_outputs=visual_outputs,
        context=PolicyInferContext(),
    )

    assert pipeline.policy_variant.action_expert.hidden_size == 24
    first_ffn_proj = pipeline.policy_variant.action_expert.blocks[0].ffn.net[0].proj.weight
    second_ffn_proj = pipeline.policy_variant.action_expert.blocks[0].ffn.net[2].weight
    assert first_ffn_proj.shape[0] == 32
    assert second_ffn_proj.shape[-1] == 32
    assert torch.isfinite(first_ffn_proj).all()
