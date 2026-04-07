from __future__ import annotations

from pathlib import Path

import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ExperimentConfig,
    InferenceConfig,
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
    TrainingConfig,
    VideoSequencePolicyConfig,
)
from open_wam.models.action_decoders import MoTActionDecoder
from open_wam.models.policy_variants import MoTPolicyVariant
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig, SharedVideoTransformerConfig
from open_wam.models.visual_tower.reference_loader import load_wan_transformer_class
from open_wam.pipelines import build_variant_pipeline_from_config

from reference_model_test_utils import reference_model_path_or_skip


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
            couple_action_to_video_timesteps=True,
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
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=1, state_dim=4, state_horizon=1),
        ),
        backbone=backbone_config,
        policy_variant=RegisterAttachedPolicyConfig(
            hidden_size=32,
            num_frame_per_block=1,
            num_action_per_block=1,
            num_state_per_block=1,
        ),
        action_decoder=RegisterActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=1),
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


def test_video_sequence_policy_requires_shared_transformer_backbone() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=LingbotCompatibleVideoBackboneConfig(implementation="dummy"),
        policy_variant=VideoSequencePolicyConfig(hidden_size=32),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8),
        inference=InferenceConfig(frame_chunk_size=2),
    )

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "shared transformer backbone" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected video-sequence validation to reject a non-shared backbone.")


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


def test_register_attached_rejects_misaligned_libero_style_block_counts() -> None:
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

    try:
        build_variant_pipeline_from_config(config)
    except ValueError as exc:
        assert "block counts to match" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected register-attached validation to reject misaligned raw-LIBERO block counts.")
