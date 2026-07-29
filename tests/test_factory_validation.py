from __future__ import annotations

from pathlib import Path

import pytest
import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ActionExpertInitMode,
    CausalVideoPredictionPolicyConfig,
    ExperimentConfig,
    InferenceConfig,
    JointTimestepCoupling,
    LingbotParallelActionDecoderConfig,
    LiberoDataConfig,
    MLPActionDecoderConfig,
    MoTPolicyConfig,
    ParallelStreamPolicyConfig,
    PostLatentPolicyConfig,
    PostDecodedPolicyConfig,
    RegisterActionDecoderConfig,
    RegisterAttachedPolicyConfig,
    RobotWinDataConfig,
    TrainerConfig,
    TrainingConfig,
    VideoConditionedActionDecoderConfig,
    VideoOnlyActionDecoderConfig,
)
from open_wam.models.action_decoders import MoTActionDecoder
from open_wam.models.policy_variants import MoTPolicyVariant
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig, SharedVideoTransformerConfig
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config

from .reference_model_test_utils import reference_model_path_or_skip


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_exact_parallel_stream_uses_vendored_reference_model_by_default() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="lingbot_replica"),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.policy_variant.config.runtime_mode == "lingbot_exact"


def test_action_conditioned_libero_smoke_config_builds() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_action_conditioned_smoke.yaml")

    pipeline = build_variant_pipeline_from_config(config)

    assert pipeline.policy_variant.config.runtime_mode == "lingbot_exact_action_conditioned"
    assert pipeline.policy_variant.reference_profile is not None
    assert pipeline.policy_variant.reference_profile.name == "libero_joint"


def test_action_conditioned_parallel_stream_builds_with_shared_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="lingbot_replica"),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact_action_conditioned",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            video_condition_on_action=True,
            video_action_condition_source="noisy_action",
            video_action_attention_scope="block_local",
            joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2, use_cache=False),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.policy_variant.config.runtime_mode == "lingbot_exact_action_conditioned"
    assert pipeline.policy_variant.config.video_condition_on_action is True


def test_reference_core_weight_loading_uses_vendored_reference_model_by_default(tmp_path: Path) -> None:
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
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=backbone_config,
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.visual_tower.reference_core_load_report is not None


def test_exact_parallel_stream_uses_decoder_action_dim_when_dataset_stays_raw() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=16, state_dim=8, state_horizon=1),
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
            hidden_size=32,
            runtime_mode="lingbot_exact",
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=30, action_horizon=16),
        training=TrainingConfig(chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0),
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


def test_exact_parallel_stream_reference_profile_rejects_mismatched_text_length() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=16, state_dim=8, state_horizon=1),
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
            hidden_size=32,
            runtime_mode="lingbot_exact",
            reference_profile="libero",
            frame_chunk_size=4,
            action_per_frame=4,
            attn_window=30,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=30, action_horizon=16),
        training=TrainingConfig(chunk_size=4, window_size=30, video_sigma_shift=5.0, action_sigma_shift=1.0),
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
        raise AssertionError("Expected LingBot exact profile validation to reject mismatched max_text_tokens.")


def test_method4_video_conditioned_decoder_rejects_window_longer_than_num_frames() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=6, state_dim=4, state_horizon=1),
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
        ),
        policy_variant=PostLatentPolicyConfig(
            hidden_size=32,
            local_video_window_frames=4,
            current_video_frame_index=0,
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=4,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=4,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "local_video_window_frames" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected method-4 video-conditioned validation to reject oversized local window.")


def test_method4_video_conditioned_decoder_warm_starts_from_video_core() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=6, state_dim=4, state_horizon=1),
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
        ),
        policy_variant=PostLatentPolicyConfig(hidden_size=32),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=4,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=4,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            action_expert_init_mode=ActionExpertInitMode.VIDEO_WEIGHT_COPY,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert torch.allclose(
        pipeline.action_decoder.action_expert.scale_shift_table,
        pipeline.visual_tower.core.scale_shift_table,
    )
    assert torch.allclose(
        pipeline.action_decoder.action_expert.blocks[0].attn1.to_q.weight,
        pipeline.visual_tower.core.blocks[0].attn1.to_q.weight,
    )


def test_method4_video_conditioned_decoder_rejects_raw_libero_current_anchor_until_aligned() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            dataset_type="lerobot_v2",
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="eef_pose_relative_to_reference"),
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
        ),
        policy_variant=PostLatentPolicyConfig(hidden_size=32),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
        ),
        training=TrainingConfig(chunk_size=4, window_size=8),
        inference=InferenceConfig(frame_chunk_size=4),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "latent-local" in str(exc)
        assert "last observed frame" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected raw LIBERO current-anchor validation to reject method-4 video-conditioned runs.")


def test_method4_rgb_video_conditioning_rejects_latent_local_configs() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            dataset_type="lerobot_v2_latent_local",
            local_root="/tmp/fake_latents",
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
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
        ),
        policy_variant=PostDecodedPolicyConfig(
            hidden_size=32,
            video_condition_input_space="rgb_video",
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
        ),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "rgb_video" in str(exc)
        assert "Latent-local" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected rgb-video conditioning to reject latent-local method-4 configs.")


def test_method4_current_frame_regression_allows_raw_rgb_configs() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
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
        ),
        policy_variant=PostDecodedPolicyConfig(
            hidden_size=32,
            video_condition_input_space="rgb_video",
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            train_mode="current_frame_regression",
            input_space="rgb_video",
            use_text_conditioning=False,
        ),
        trainer=TrainerConfig(batch_adapter="views"),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert pipeline.action_decoder.supports_direct_train_inputs() is True


def test_method4_current_frame_regression_video_latent_requires_latent_batch_adapter() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            dataset_type="lerobot_v2_latent_local",
            local_root="/tmp/fake_latents",
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
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
        ),
        policy_variant=PostLatentPolicyConfig(
            hidden_size=32,
            video_condition_input_space="video_latent",
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            train_mode="current_frame_regression",
            input_space="video_latent",
        ),
        trainer=TrainerConfig(batch_adapter="views"),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "latent batch adapter" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError(
            "Expected current-frame latent regression to reject the view batch adapter."
        )


def test_method4_rollout_window_rejects_nonzero_current_frame_index() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            dataset_type="lerobot_v2_latent_local",
            local_root="/tmp/fake_latents",
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
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
        ),
        policy_variant=PostLatentPolicyConfig(
            hidden_size=32,
            current_video_frame_index=1,
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
        ),
        trainer=TrainerConfig(batch_adapter="latents"),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "current_video_frame_index = 0" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected non-zero current_video_frame_index to be rejected for rollout-window decoding.")


def test_method4_current_frame_regression_rgb_views_rejects_text_conditioning() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            dataset_type="lerobot_v2",
            local_root="/tmp/fake_views",
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="raw"),
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
        ),
        policy_variant=PostDecodedPolicyConfig(
            hidden_size=32,
            video_condition_input_space="rgb_video",
        ),
        action_decoder=VideoConditionedActionDecoderConfig(
            hidden_size=32,
            action_dim=7,
            action_horizon=6,
            context_dim=32,
            text_context_dim=16,
            state_dim=8,
            freq_dim=8,
            num_layers=2,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            train_mode="current_frame_regression",
            input_space="rgb_video",
            use_text_conditioning=True,
        ),
        trainer=TrainerConfig(batch_adapter="views"),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "use_text_conditioning=false" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected RGB current-frame regression with text conditioning to be rejected.")


def test_post_latent_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=PostLatentPolicyConfig(hidden_size=32),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected post-latent validation to reject a non-shared backbone.")


def test_post_decoded_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=PostDecodedPolicyConfig(hidden_size=32),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected post-decoded validation to reject a non-shared backbone.")


def test_parallel_stream_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode="lingbot_exact",
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        action_decoder=LingbotParallelActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected parallel-stream validation to reject a non-shared backbone.")


def test_register_attached_pipeline_build_is_obsolete() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=2, state_dim=4, state_horizon=2),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=RegisterAttachedPolicyConfig(
            hidden_size=32,
            num_frame_per_block=1,
            num_action_per_block=1,
            num_state_per_block=1,
        ),
        action_decoder=RegisterActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=2),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    with pytest.warns(RuntimeWarning, match="Traditional Method 2 `register_attached` is obsolete"):
        with pytest.raises(RuntimeError, match="Traditional Method 2 `register_attached` is obsolete"):
            build_variant_pipeline_from_config(config)


def test_mot_policy_builds_with_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
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
        policy_variant=MoTPolicyConfig(hidden_size=32, video_prefix_frames=1, num_action_layers=2),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    pipeline = build_variant_pipeline_from_config(config)
    assert isinstance(pipeline.policy_variant, MoTPolicyVariant)
    assert isinstance(pipeline.action_decoder, MoTActionDecoder)
    assert pipeline.policy_variant.config.runtime_mode == "video_prefill_action_denoise"
    assert pipeline.policy_variant.action_expert.num_layers == 2
    assert pipeline.policy_variant.action_expert.action_dim == 4


def test_mot_policy_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=MoTPolicyConfig(hidden_size=32, video_prefix_frames=1, num_action_layers=2),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected MoT validation to reject a non-shared backbone.")


def test_causal_video_prediction_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=0, state_dim=4, state_horizon=0),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=CausalVideoPredictionPolicyConfig(hidden_size=32),
        action_decoder=VideoOnlyActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=0),
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
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected causal-video validation to reject a non-shared backbone.")


def test_post_latent_rejects_frontend_only_attachment() -> None:
    try:
        PostLatentPolicyConfig(hidden_size=32, attach_site="post_frontend_latents")
    except ValueError as exc:
        assert "post_visual_core" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected post-latent config to reject frontend-only attachment.")


def test_post_decoded_rejects_non_decode_attachment() -> None:
    try:
        PostDecodedPolicyConfig(hidden_size=32, attach_site="post_visual_core")
    except ValueError as exc:
        assert "post_visual_decode" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected post-decoded config to reject non-decode attachment.")


def test_register_attached_libero_style_config_is_obsolete_before_block_validation() -> None:
    config = ExperimentConfig(
        data=LiberoDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=7, action_horizon=6, state_dim=8, state_horizon=1),
            action_target=ActionTargetConfig(representation="eef_pose_relative_to_reference"),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
        ),
        policy_variant=RegisterAttachedPolicyConfig(
            hidden_size=32,
            num_frame_per_block=1,
            num_action_per_block=2,
            num_state_per_block=1,
        ),
        action_decoder=RegisterActionDecoderConfig(hidden_size=32, action_dim=7, action_horizon=6),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=1),
    )

    with pytest.warns(RuntimeWarning, match="Traditional Method 2 `register_attached` is obsolete"):
        with pytest.raises(RuntimeError, match="Traditional Method 2 `register_attached` is obsolete"):
            build_variant_pipeline_from_config(config)
