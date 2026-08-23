from __future__ import annotations

from pathlib import Path

import pytest
import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    CausalVideoPredictionPolicyConfig,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    InferenceConfig,
    JointTimestepCoupling,
    LiberoDataConfig,
    ParallelStreamActionDecoderConfig,
    ParallelStreamPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoOnlyActionDecoderConfig,
    load_experiment_config,
)
from open_wam.models.action_decoders import DualExpertActionDecoder
from open_wam.models.policy_variants import DualExpertPolicyVariant
from open_wam.models.video_backbone.config import (
    LingbotCompatibleVideoBackboneConfig,
    SharedVideoTransformerConfig,
)
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class
from open_wam.pipelines import build_variant_pipeline_from_config

from .reference_model_test_utils import reference_model_path_or_skip

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exact_parallel_stream_uses_vendored_reference_model_by_default() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="lingbot_replica"),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.policy_variant.config.runtime_mode == "lingbot_exact"


def test_action_conditioned_libero_smoke_config_builds() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_action_conditioned_smoke.yaml"
    )

    pipeline = build_variant_pipeline_from_config(config)

    assert (
        pipeline.policy_variant.config.runtime_mode
        == "lingbot_exact_action_conditioned"
    )
    assert pipeline.policy_variant.reference_profile is not None
    assert pipeline.policy_variant.reference_profile.name == "libero_joint"


def test_action_conditioned_parallel_stream_builds_with_shared_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="lingbot_replica"),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.JOINT,
            hidden_size=32,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            video_action_condition_source="noisy_action",
            video_action_attention_scope="block_local",
            joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2, use_cache=False),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert (
        pipeline.policy_variant.config.runtime_mode
        == "lingbot_exact_action_conditioned"
    )
    assert pipeline.policy_variant.config.video_condition_on_action is True


def test_reference_core_weight_loading_uses_vendored_reference_model_by_default(
    tmp_path: Path,
) -> None:
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        pretrained_model_name_or_path=str(tmp_path / "lingbot_ckpt"),
        load_reference_core_weights=True,
        reference_model_path=reference_model_path_or_skip(),
    )
    model_cls = load_wan_transformer_class(backbone_config)
    reference_model = model_cls(
        patch_size=[1, 2, 2],
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=48,
        out_channels=48,
        action_dim=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        attn_mode="torch",
    ).to(dtype=torch.bfloat16)
    transformer_dir = tmp_path / "lingbot_ckpt" / "transformer"
    reference_model.save_pretrained(transformer_dir)

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=backbone_config,
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.visual_tower.reference_core_load_report is not None


def test_exact_parallel_stream_uses_decoder_action_dim_when_dataset_stays_raw() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=7, action_horizon=16, state_dim=8, state_horizon=1
            ),
            action_target=ActionTargetConfig(representation="raw"),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            max_text_tokens=512,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=30, action_horizon=16
        ),
        training=TrainingConfig(
            chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0
        ),
        inference=InferenceConfig(
            frame_chunk_size=4,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            video_exec_step=-1,
        ),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.visual_tower.action_dim == 30
    assert pipeline.policy_variant.action_dim == 30
    assert pipeline.policy_variant.exact_action_adapter.spec is not None
    assert pipeline.policy_variant.exact_action_adapter.spec.raw_action_dim == 7
    requirements = pipeline.policy_variant.pipeline_requirements(
        default_action_dim=config.action_decoder.action_dim,
        default_action_horizon=config.action_decoder.action_horizon,
        default_state_dim=config.data.action_schema.state_dim,
    )
    assert requirements.accepted_source_action_shapes == ((7, 16), (30, 16))
    assert (
        pipeline.action_decoder.source_action_channel_ids
        == pipeline.policy_variant.exact_action_adapter.spec.used_action_channel_ids
    )


def test_exact_parallel_stream_reference_profile_rejects_mismatched_text_length() -> (
    None
):
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=7, action_horizon=16, state_dim=8, state_horizon=1
            ),
            action_target=ActionTargetConfig(representation="raw"),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            max_text_tokens=226,
        ),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=30, action_horizon=16
        ),
        training=TrainingConfig(
            chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0
        ),
        inference=InferenceConfig(
            frame_chunk_size=4,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            video_exec_step=-1,
        ),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "max_text_tokens" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError(
            "Expected LingBot exact profile validation to reject mismatched max_text_tokens."
        )


def test_parallel_stream_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            hidden_size=32,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=ParallelStreamActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "requires one of the following backbone implementations" in str(exc)
        assert "shared_transformer" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError(
            "Expected parallel-stream validation to reject a non-shared backbone."
        )


def test_dual_expert_policy_builds_with_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
        ),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=2,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert isinstance(pipeline.policy_variant, DualExpertPolicyVariant)
    assert isinstance(pipeline.action_decoder, DualExpertActionDecoder)
    assert (
        pipeline.policy_variant.config.program == VideoActionProgram.VIDEO_THEN_ACTION
    )
    assert pipeline.policy_variant.action_expert.num_layers == 2
    assert pipeline.policy_variant.action_expert.action_dim == 4


@pytest.mark.parametrize(
    ("policy_variant", "action_decoder"),
    [
        (
            DualExpertPolicyConfig(
                hidden_size=32,
                program=VideoActionProgram.VIDEO_THEN_ACTION,
                video_prefix_frames=1,
                num_action_layers=1,
            ),
            DualExpertActionDecoderConfig(
                hidden_size=32,
                action_dim=4,
                action_horizon=3,
            ),
        ),
        (
            ParallelStreamPolicyConfig(
                hidden_size=32,
                program=VideoActionProgram.VIDEO_THEN_ACTION,
                frame_chunk_size=2,
                action_per_frame=2,
                attn_window=8,
            ),
            ParallelStreamActionDecoderConfig(
                hidden_size=32,
                action_dim=4,
                action_horizon=3,
            ),
        ),
    ],
)
def test_video_action_policies_share_source_horizon_validation(
    policy_variant: DualExpertPolicyConfig | ParallelStreamPolicyConfig,
    action_decoder: DualExpertActionDecoderConfig | ParallelStreamActionDecoderConfig,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4,
                action_horizon=4,
                state_dim=4,
                state_horizon=1,
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
        ),
        policy_variant=policy_variant,
        action_decoder=action_decoder,
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    with pytest.raises(
        ValueError,
        match="Dataset action geometry is not accepted",
    ):
        build_variant_pipeline_from_config(config)


def test_dual_expert_rejects_unadapted_source_action_width() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(
                action_dim=4,
                action_horizon=4,
                state_dim=4,
                state_horizon=1,
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
        ),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32,
            action_dim=5,
            action_horizon=4,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    with pytest.raises(
        ValueError,
        match="Dataset action geometry is not accepted",
    ):
        build_variant_pipeline_from_config(config)


def test_dual_expert_policy_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=4, state_dim=4, state_horizon=1
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=DualExpertPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=2,
        ),
        action_decoder=DualExpertActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=4
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "requires one of the following backbone implementations" in str(exc)
        assert "shared_transformer" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError(
            "Expected DualExpert validation to reject a non-shared backbone."
        )


def test_causal_video_prediction_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(
                action_dim=4, action_horizon=0, state_dim=4, state_horizon=0
            ),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=CausalVideoPredictionPolicyConfig(hidden_size=32),
        action_decoder=VideoOnlyActionDecoderConfig(
            hidden_size=32, action_dim=4, action_horizon=0
        ),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("latent",),
            action_loss_weight=0.0,
        ),
        inference=InferenceConfig(frame_chunk_size=1),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "requires one of the following backbone implementations" in str(exc)
        assert "shared_transformer" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError(
            "Expected causal-video validation to reject a non-shared backbone."
        )
