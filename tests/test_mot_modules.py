from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import open_wam.configs  # Ensure config/video-backbone modules finish initialization before policy-variant imports.
from open_wam.configs import (
    ActionSchemaConfig,
    CurrentBlockCoupling,
    ExperimentConfig,
    InferenceConfig,
    MLPActionDecoderConfig,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.models.common.attention_profiles import build_chunked_temporal_exact_attention_profile
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.policy_variants.mot.contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTRuntimeState,
)
from open_wam.models.policy_variants.mot.modules import (
    MoTActionExpert,
    init_action_expert_from_video_core,
)
from open_wam.models.policy_variants.mot.runtime import (
    build_mot_inference_action_attention_mask,
    build_mot_attention_mask,
    build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask,
    resolve_mot_condition_latents,
    trim_mot_action_cache_prefix,
)
from open_wam.models.policy_variants.mot.variant import (
    MoTPolicyVariant,
    _rewind_runtime_action_cache_to_frame,
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


def test_build_mot_inference_action_mask_uses_absolute_frame_starts() -> None:
    mask = build_mot_inference_action_attention_mask(
        video_seq_len=8,
        past_action_seq_len=0,
        current_action_seq_len=4,
        video_tokens_per_frame=2,
        action_tokens_per_frame=2,
        chunk_size_frames=2,
        window_size_frames=8,
        device=torch.device("cpu"),
        video_frame_start=0,
        current_action_frame_start=2,
    )

    current_action_query = 8
    current_video_frame_token = 4
    assert mask[current_action_query, current_video_frame_token]


def test_build_mot_inference_action_mask_decouples_same_step_video() -> None:
    mask = build_mot_inference_action_attention_mask(
        video_seq_len=8,
        past_action_seq_len=0,
        current_action_seq_len=4,
        video_tokens_per_frame=2,
        action_tokens_per_frame=2,
        chunk_size_frames=2,
        window_size_frames=8,
        device=torch.device("cpu"),
        video_frame_start=0,
        current_action_frame_start=2,
        current_block_coupling="decoupled_same_step",
    )

    current_action_query = 8
    current_video_frame_token = 4
    previous_video_frame_token = 2
    assert not mask[current_action_query, current_video_frame_token]
    assert mask[current_action_query, previous_video_frame_token]


def test_build_packed_action_mask_decouples_same_step_clean_video() -> None:
    mask = build_packed_action_attention_mask(
        num_video_frames=2,
        video_tokens_per_frame=2,
        num_action_frames=2,
        action_tokens_per_frame=2,
        action_chunk_size_frames=1,
        device=torch.device("cpu"),
        current_block_coupling="decoupled_same_step",
    )

    action_noisy_frame_0_query = 0
    action_noisy_frame_1_query = 2
    video_clean_frame_0_key = 0
    video_clean_frame_1_key = 2
    assert not mask[action_noisy_frame_0_query, video_clean_frame_0_key]
    assert not mask[action_noisy_frame_1_query, video_clean_frame_1_key]
    assert mask[action_noisy_frame_1_query, video_clean_frame_0_key]


@pytest.mark.parametrize(
    ("coupling", "video_reads_action", "action_reads_video"),
    [
        ("joint", True, True),
        ("decoupled_same_step", False, False),
        ("video_noisy_to_action", False, True),
        ("action_noisy_to_video", True, False),
    ],
)
def test_build_mot_attention_mask_same_step_coupling_visibility(
    coupling: str,
    video_reads_action: bool,
    action_reads_video: bool,
) -> None:
    mask = build_mot_attention_mask(
        video_seq_len=4,
        action_seq_len=4,
        device=torch.device("cpu"),
        condition_mode=MoTConditionMode.FIRST_FRAME,
        video_tokens_per_frame=2,
        action_tokens_per_frame=2,
        action_chunk_size_frames=1,
        clean_video_frames=0,
        clean_action_frames=0,
        current_block_coupling=coupling,
    )

    video_query_frame_0 = 0
    action_query_frame_0 = 4
    video_key_frame_0 = 0
    action_key_frame_0 = 4
    assert bool(mask[video_query_frame_0, action_key_frame_0]) is video_reads_action
    assert bool(mask[action_query_frame_0, video_key_frame_0]) is action_reads_video


@pytest.mark.parametrize(
    ("coupling", "video_reads_action", "action_reads_video"),
    [
        (CurrentBlockCoupling.VIDEO_THEN_ACTION, False, True),
        (CurrentBlockCoupling.JOINT, True, True),
        (CurrentBlockCoupling.ACTION_THEN_VIDEO, True, False),
        (CurrentBlockCoupling.DECOUPLED_SAME_STEP, False, False),
        (CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION, False, True),
        (CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO, True, False),
    ],
)
def test_build_mot_packed_coupling_mask_six_mode_visibility(
    coupling: CurrentBlockCoupling,
    video_reads_action: bool,
    action_reads_video: bool,
) -> None:
    mask = build_mot_packed_coupling_attention_mask(
        num_video_frames=1,
        video_tokens_per_frame=1,
        num_action_frames=1,
        action_tokens_per_frame=1,
        chunk_size_frames=1,
        device=torch.device("cpu"),
        current_block_coupling=coupling,
    )

    video_noisy_query = 0
    video_clean_key = 1
    action_noisy_query = 2
    action_noisy_key = 2
    action_clean_key = 3
    video_key = (
        action_noisy_key
        if coupling in {CurrentBlockCoupling.JOINT, CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO}
        else action_clean_key
    )
    action_key = (
        video_noisy_query
        if coupling in {CurrentBlockCoupling.JOINT, CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION}
        else video_clean_key
    )
    assert bool(mask[video_noisy_query, video_key]) is video_reads_action
    assert bool(mask[action_noisy_query, action_key]) is action_reads_video


@pytest.mark.parametrize(
    "coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
def test_build_mot_packed_coupling_mask_preserves_video_history_for_all_modes(
    coupling: CurrentBlockCoupling,
) -> None:
    mask = build_mot_packed_coupling_attention_mask(
        num_video_frames=2,
        video_tokens_per_frame=1,
        num_action_frames=2,
        action_tokens_per_frame=1,
        chunk_size_frames=1,
        device=torch.device("cpu"),
        current_block_coupling=coupling,
    )

    # Layout for two frames: V_noisy [0:2], V_clean [2:4], A_noisy [4:6], A_clean [6:8].
    video_noisy_chunk1 = 1
    action_noisy_chunk1 = 5
    video_clean_history = 2
    action_clean_history = 6

    assert bool(mask[video_noisy_chunk1, video_clean_history]) is True
    assert bool(mask[video_noisy_chunk1, action_clean_history]) is False
    assert bool(mask[action_noisy_chunk1, video_clean_history]) is True
    assert bool(mask[action_noisy_chunk1, action_clean_history]) is True


@pytest.mark.parametrize(
    "coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
@pytest.mark.parametrize(
    ("num_frames", "video_tokens_per_frame", "action_tokens_per_frame", "chunk_size"),
    [
        (2, 1, 1, 1),
        (4, 2, 1, 2),
    ],
)
def test_build_mot_packed_coupling_profile_matches_method1_dense_mask(
    coupling: CurrentBlockCoupling,
    num_frames: int,
    video_tokens_per_frame: int,
    action_tokens_per_frame: int,
    chunk_size: int,
) -> None:
    m5_profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=num_frames,
        video_tokens_per_frame=video_tokens_per_frame,
        num_action_frames=num_frames,
        action_tokens_per_frame=action_tokens_per_frame,
        chunk_size_frames=chunk_size,
        attention_window_size=8,
        device=torch.device("cpu"),
        current_block_coupling=coupling,
    )
    method1_profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, num_frames, 1, video_tokens_per_frame),
        action_shape=(1, 1, num_frames, 1, action_tokens_per_frame),
        padded_length=0,
        chunk_size=chunk_size,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling=coupling.value,
        preserve_video_pretrain_history=True,
    )

    assert m5_profile.self_attention_mask is not None
    assert method1_profile.self_attention_mask is not None
    assert torch.equal(m5_profile.self_attention_mask, method1_profile.self_attention_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex block mask requires CUDA in this setup")
def test_build_mot_packed_coupling_profile_uses_flex_on_cuda() -> None:
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=4,
        video_tokens_per_frame=2,
        num_action_frames=4,
        action_tokens_per_frame=1,
        chunk_size_frames=2,
        attention_window_size=8,
        device=torch.device("cuda"),
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )

    assert profile.self_attention_mask is None
    assert profile.self_attention_block_mask is not None


def test_trim_mot_action_cache_prefix_keeps_oldest_tokens() -> None:
    key = torch.arange(1 * 1 * 6 * 1, dtype=torch.float32).reshape(1, 1, 6, 1)
    value = key + 100
    cache = MoTActionCache(
        layers=(MoTActionLayerCache(key=key, value=value),),
        action_seq_len=6,
    )

    trimmed = trim_mot_action_cache_prefix(cache, max_action_seq_len=4)

    assert trimmed.action_seq_len == 4
    assert torch.equal(trimmed.layers[0].key.flatten(), torch.arange(4, dtype=torch.float32))
    assert torch.equal(trimmed.layers[0].value.flatten(), torch.arange(100, 104, dtype=torch.float32))


def test_runtime_action_cache_rewind_uses_absolute_cache_start_frame() -> None:
    key = torch.arange(1 * 1 * 12 * 1, dtype=torch.float32).reshape(1, 1, 12, 1)
    state = MoTRuntimeState(
        action_cache=MoTActionCache(
            layers=(MoTActionLayerCache(key=key, value=key + 100),),
            action_seq_len=12,
        ),
        action_cache_start_frame=10,
    )

    _rewind_runtime_action_cache_to_frame(
        state,
        absolute_frame_start=14,
        action_tokens_per_frame=2,
    )

    assert state.action_cache_start_frame == 10
    assert state.action_cache is not None
    assert state.action_cache.action_seq_len == 8
    assert torch.equal(state.action_cache.layers[0].key.flatten(), torch.arange(8, dtype=torch.float32))


def test_runtime_action_cache_rewind_clears_cache_before_window() -> None:
    key = torch.arange(1 * 1 * 12 * 1, dtype=torch.float32).reshape(1, 1, 12, 1)
    state = MoTRuntimeState(
        action_cache=MoTActionCache(
            layers=(MoTActionLayerCache(key=key, value=key + 100),),
            action_seq_len=12,
        ),
        action_cache_start_frame=10,
    )

    _rewind_runtime_action_cache_to_frame(
        state,
        absolute_frame_start=8,
        action_tokens_per_frame=2,
    )

    assert state.action_cache is None
    assert state.action_cache_start_frame == 8


def test_mot_train_loss_masks_use_objective_specific_metadata() -> None:
    variant = MoTPolicyVariant(
        config=MoTPolicyConfig(),
        backbone_config=SharedVideoTransformerConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
        action_dim=1,
        action_horizon=8,
        state_dim=4,
    )
    batch = PolicyTrainBatch(
        actions=torch.ones(1, 8, 1),
        action_mask=torch.ones(1, 8, 1),
        extra={
            "metadata": (
                {
                    "loss_frame_start": 1,
                    "loss_frame_end": 3,
                    "latent_loss_frame_start": 2,
                    "latent_loss_frame_end": 4,
                    "action_loss_frame_start": 1,
                    "action_loss_frame_end": 2,
                },
            )
        },
    )

    action_mask = variant._build_effective_action_mask(batch=batch, observed_num_frames=4)
    assert action_mask is not None
    assert torch.equal(action_mask[:, :, 0], torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))

    video_mask = variant._build_effective_video_loss_mask(
        video_latents=torch.ones(1, 2, 4, 1, 1),
        batch=batch,
        default_history_frames=1,
    )
    assert torch.equal(video_mask.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_mot_train_video_cache_detach_decision_is_cached_per_core() -> None:
    class CountingCore:
        def __init__(self) -> None:
            self.calls = 0
            self.parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def parameters(self):
            self.calls += 1
            return iter((self.parameter,))

    variant = MoTPolicyVariant(
        config=MoTPolicyConfig(),
        backbone_config=SharedVideoTransformerConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
        action_dim=4,
        action_horizon=4,
        state_dim=4,
    )
    core = CountingCore()
    visual_tower = SimpleNamespace(core=core)

    assert variant._should_detach_train_video_cache(visual_tower)
    core.parameter.requires_grad_(True)
    assert variant._should_detach_train_video_cache(visual_tower)
    assert core.calls == 1


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


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
def test_mot_joint_denoise_train_supports_same_step_couplings(
    current_block_coupling: CurrentBlockCoupling,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
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
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=current_block_coupling,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            enabled_objectives=("action", "latent"),
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(actions=torch.randn(1, 4, 4))
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)

    output = pipeline.forward_train_from_latents(video_latents, batch, text_context=text_context)

    assert torch.isfinite(output.decoder_output.loss)
    assert output.policy_output.aux["current_block_coupling"] == current_block_coupling.value


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
def test_mot_joint_denoise_infer_supports_same_step_couplings(
    current_block_coupling: CurrentBlockCoupling,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
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
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=current_block_coupling,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(video_latents, text_context=text_context)

    output = pipeline._forward_infer_with_visual_outputs(visual_outputs, context=PolicyInferContext())

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert output.policy_output.aux["current_block_coupling"] == current_block_coupling.value


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
def test_mot_packed_infer_six_modes_keep_two_chunk_history(
    current_block_coupling: CurrentBlockCoupling,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
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
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=current_block_coupling,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    text_context = torch.randn(1, 5, 16)
    first_latents = torch.randn(1, 48, 1, 8, 8)
    first = pipeline.forward_infer_step_from_latents(
        first_latents,
        context=PolicyInferContext(),
        text_context=text_context,
    )
    first_state = first.policy_output.next_state.variant_state
    assert isinstance(first_state, MoTRuntimeState)
    assert first_state.past_clean_latents is not None
    assert first_state.past_clean_actions is not None

    second_latents = torch.randn(1, 48, 2, 8, 8)
    second = pipeline.forward_infer_step_from_latents(
        second_latents,
        context=PolicyInferContext(),
        infer_state=first.policy_output.next_state,
        text_context=text_context,
    )

    debug = second.policy_output.aux["mot_packed_history_debug"]
    assert debug["shared_history_frames"] >= 1
    assert debug["past_clean_latent_frames"] >= 1
    assert debug["past_clean_action_frames"] >= 1
    assert debug["packed_video_frames"] == debug["shared_history_frames"] + 2
    assert debug["packed_action_frames"] == debug["shared_history_frames"] + 2
    second_state = second.policy_output.next_state.variant_state
    assert isinstance(second_state, MoTRuntimeState)
    assert second_state.past_clean_latents is not None
    assert second_state.past_clean_actions is not None
    assert second_state.past_clean_actions.shape[1] % 2 == 0


def test_mot_packed_infer_chunk0_matches_method1_first_step_bootstrap() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=4,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=4, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
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
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 1, 8, 8)
    text_context = torch.randn(1, 5, 16)

    first = pipeline.forward_infer_step_from_latents(
        video_latents,
        context=PolicyInferContext(),
        text_context=text_context,
    )

    assert first.policy_output.aux["mot_first_step_bootstrap"] is True
    assert first.policy_output.aux["mot_action_cond_tokens"] == 2
    assert torch.allclose(first.policy_output.aux["predicted_latents"][:, :, 0:1], video_latents)
    assert torch.allclose(first.decoder_output.action_pred[:, :2], torch.zeros_like(first.decoder_output.action_pred[:, :2]))

    packed_state = first.policy_output.next_state.variant_state
    assert isinstance(packed_state, MoTRuntimeState)
    history_anchor = packed_state.past_clean_latents[:, :, -1:].clone()
    second_latents = torch.randn(1, 48, 2, 8, 8)
    second = pipeline.forward_infer_step_from_latents(
        second_latents,
        context=PolicyInferContext(),
        infer_state=first.policy_output.next_state,
        text_context=text_context,
    )

    assert second.policy_output.aux["mot_first_step_bootstrap"] is False
    assert second.policy_output.aux["mot_action_cond_tokens"] == 0
    assert second.policy_output.aux["mot_history_anchor_frames"] >= 1
    history_debug = second.policy_output.aux["mot_packed_history_debug"]
    assert history_debug["past_clean_latent_frames"] >= 2
    assert history_debug["past_clean_action_frames"] >= 2
    assert history_debug["shared_history_frames"] >= 1
    assert history_debug["current_observed_latent_frames"] == 2
    assert history_debug["current_clean_condition_frames"] == 2


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
