from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.action_decoders import VideoConditionedActionDecoder
from open_wam.models.action_decoders.base import DecoderRolloutState, DirectActionDecoderTrainInputs
from open_wam.models.action_decoders.video_conditioned_expert import (
    VideoConditionedActionExpert,
    init_conditioned_action_expert_from_video_core,
)
from open_wam.models.policy_variants.contracts import (
    DecoderSequenceContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyTrainBatch,
    PolicyTrainOutput,
    VideoConditionWindowContext,
)


def _build_decoder(
    *,
    rollout_chunk_steps: int = 1,
    input_space: str = "video_latent",
    train_mode: str = "rollout_window_diffusion",
    dropout: float = 0.0,
) -> VideoConditionedActionDecoder:
    return VideoConditionedActionDecoder(
        hidden_size=32,
        action_dim=3,
        action_horizon=6,
        context_dim=32,
        text_context_dim=16,
        state_dim=4,
        freq_dim=8,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        cross_attn_norm=True,
        eps=1e-6,
        input_space=input_space,
        train_mode=train_mode,
        action_chunk_anchor_mode="current_plus_future",
        action_expert_init_mode="random",
        rollout_chunk_steps=rollout_chunk_steps,
        direct_latent_channels=48,
        direct_rgb_patch_size=8,
        use_text_conditioning=True,
        use_state_conditioning=True,
        training_config=TrainingConfig(action_num_train_timesteps=8),
        inference_config=InferenceConfig(action_num_inference_steps=2),
        dropout=dropout,
    )


def _build_sequence_context() -> DecoderSequenceContext:
    return DecoderSequenceContext(
        sequence_tokens=torch.randn(1, 4, 2, 32),
        sequence_layout={"family": "video_feature_policy"},
        frame_count=4,
        source_stage="frontend",
        state_sequence=torch.randn(1, 1, 4),
        goal_features=torch.randn(1, 2, 16),
        video_condition_window=VideoConditionWindowContext(
            local_window_tokens=torch.randn(1, 4, 2, 32),
            source_stage="frontend",
            input_space="video_latent",
            local_window_frames=4,
            current_frame_index=0,
            current_action_index=0,
            action_chunk_anchor_mode="current_plus_future",
            observed_frame_count=1,
        ),
    )


def test_video_conditioned_decoder_train_reports_current_anchor_metadata() -> None:
    decoder = _build_decoder()
    sequence_context = _build_sequence_context()
    policy_output = PolicyTrainOutput(
        policy_features=torch.zeros(1, 0, 32),
        metrics={},
        decoder_sequence_context=sequence_context,
    )
    batch = PolicyTrainBatch(actions=torch.randn(1, 6, 3), state=torch.randn(1, 1, 4))

    output = decoder.forward_train(policy_output, batch)

    assert output.action_pred.shape == (1, 6, 3)
    assert output.aux["action_chunk_anchor_mode"] == "current_plus_future"
    assert output.aux["current_frame_index"].item() == 0.0
    assert output.aux["current_action_index"].item() == 0.0
    assert output.aux["local_video_window_frames"].item() == 4.0


def test_video_conditioned_decoder_infer_reuses_cached_chunk() -> None:
    decoder = _build_decoder(rollout_chunk_steps=2)
    sequence_context = _build_sequence_context()
    policy_output = PolicyInferOutput(
        policy_features=torch.zeros(1, 0, 32),
        next_state=PolicyInferState(),
        decoder_sequence_context=sequence_context,
    )

    first = decoder.forward_infer(policy_output, previous_state=None)
    second = decoder.forward_infer(policy_output, previous_state=first.next_state)

    assert isinstance(first.next_state, DecoderRolloutState)
    assert first.aux["sampled_new_chunk"] is True
    assert second.aux["sampled_new_chunk"] is False
    assert torch.allclose(second.action_pred, first.action_pred)
    assert second.aux["current_action_index"].item() == 1.0
    assert torch.allclose(second.aux["current_action"], first.action_pred[:, 1])


def test_video_conditioned_decoder_rollout_chunk_step_one_resamples_next_call() -> None:
    decoder = _build_decoder(rollout_chunk_steps=1)
    sequence_context = _build_sequence_context()
    policy_output = PolicyInferOutput(
        policy_features=torch.zeros(1, 0, 32),
        next_state=PolicyInferState(),
        decoder_sequence_context=sequence_context,
    )

    first = decoder.forward_infer(policy_output, previous_state=None)
    second = decoder.forward_infer(policy_output, previous_state=first.next_state)

    assert isinstance(first.next_state, DecoderRolloutState)
    assert first.next_state.step_within_chunk == 1
    assert first.aux["sampled_new_chunk"] is True
    assert second.aux["sampled_new_chunk"] is True


def test_video_conditioned_decoder_direct_latent_current_frame_regression() -> None:
    decoder = _build_decoder(train_mode="current_frame_regression")
    batch = PolicyTrainBatch(
        actions=torch.randn(2, 6, 3),
        action_mask=torch.ones(2, 6, 3),
        state=torch.randn(2, 1, 4),
    )
    direct_inputs = DirectActionDecoderTrainInputs(
        current_frame=torch.randn(2, 48, 8, 8),
        input_space="video_latent",
        current_action_index=0,
        state=batch.state,
    )

    output = decoder.forward_train_direct(direct_inputs, batch)

    assert output.action_pred.shape == (2, 1, 3)
    assert output.aux["train_mode"] == "current_frame_regression"
    assert output.aux["video_condition_input_space"] == "video_latent"
    assert output.aux["current_action_index"].item() == 0.0


def test_video_conditioned_decoder_direct_rgb_current_frame_regression() -> None:
    decoder = _build_decoder(input_space="rgb_video", train_mode="current_frame_regression")
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 6, 3),
        state=torch.randn(1, 1, 4),
    )
    direct_inputs = DirectActionDecoderTrainInputs(
        current_frame=torch.randn(1, 3, 64, 64),
        input_space="rgb_video",
        current_action_index=0,
        state=batch.state,
    )

    output = decoder.forward_train_direct(direct_inputs, batch)

    assert output.action_pred.shape == (1, 1, 3)
    assert output.aux["video_condition_input_space"] == "rgb_video"


def test_video_conditioned_decoder_current_frame_regression_rejects_infer() -> None:
    decoder = _build_decoder(train_mode="current_frame_regression")
    policy_output = PolicyInferOutput(
        policy_features=torch.zeros(1, 0, 32),
        next_state=PolicyInferState(),
        decoder_sequence_context=_build_sequence_context(),
    )

    try:
        decoder.forward_infer(policy_output, previous_state=None)
    except ValueError as exc:
        assert "train-only" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected current-frame regression mode to reject inference.")


def test_video_conditioned_decoder_uses_configured_dropout() -> None:
    decoder = _build_decoder(dropout=0.25)

    assert isinstance(decoder.context_dropout, nn.Dropout)
    assert decoder.context_dropout.p == 0.25


def test_video_conditioned_action_expert_initializes_time_conditioning_from_video_side() -> None:
    source_core = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )
    source_core.config = SimpleNamespace(num_heads=4, attention_head_dim=8, hidden_size=32)
    source_core.action_time_conditioner = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    ).time_conditioner
    target_expert = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )

    with torch.no_grad():
        for parameter in source_core.time_conditioner.parameters():
            parameter.fill_(0.25)
        for parameter in source_core.action_time_conditioner.parameters():
            parameter.fill_(0.75)

    init_conditioned_action_expert_from_video_core(
        action_expert=target_expert,
        video_core=source_core,
        mode="video_weight_copy",
    )

    for target, source in zip(
        target_expert.time_conditioner.parameters(),
        source_core.time_conditioner.parameters(),
        strict=True,
    ):
        assert torch.allclose(target, source)


def test_video_conditioned_action_expert_can_interpolate_from_deeper_video_core() -> None:
    source_core = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=3,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )
    source_core.config = SimpleNamespace(num_heads=4, attention_head_dim=8, hidden_size=32)
    target_expert = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )

    with torch.no_grad():
        for layer_index, block in enumerate(source_core.blocks):
            for parameter in block.parameters():
                parameter.fill_(float(layer_index + 1))

    init_conditioned_action_expert_from_video_core(
        action_expert=target_expert,
        video_core=source_core,
        mode="video_weight_interpolate",
    )

    first_target_weight = next(target_expert.blocks[0].parameters())
    last_target_weight = next(target_expert.blocks[1].parameters())
    assert torch.allclose(first_target_weight, torch.full_like(first_target_weight, 1.0))
    assert torch.allclose(last_target_weight, torch.full_like(last_target_weight, 3.0))


def test_video_conditioned_action_expert_rejects_deeper_action_expert_interpolation() -> None:
    source_core = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )
    source_core.config = SimpleNamespace(num_heads=4, attention_head_dim=8, hidden_size=32)
    target_expert = VideoConditionedActionExpert(
        hidden_size=32,
        action_dim=3,
        num_layers=3,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        freq_dim=8,
        context_dim=32,
    )

    with pytest.raises(ValueError, match="shallower action experts only"):
        init_conditioned_action_expert_from_video_core(
            action_expert=target_expert,
            video_core=source_core,
            mode="video_weight_interpolate",
        )
