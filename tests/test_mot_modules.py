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
    JointTimestepCoupling,
    MLPActionDecoderConfig,
    MoTGeneralistTrainingMode,
    MoTActionExpertInitMode,
    MoTConditionMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelSequenceContract,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.models.common.attention_profiles import build_chunked_temporal_exact_attention_profile
from open_wam.models.common.rollout_startup import build_strict_action_context_mask
from open_wam.models.policy_variants import PolicyInferContext, PolicyInferState, PolicyTrainBatch, RolloutCursor
from open_wam.models.policy_variants.mot.contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTRuntimeState,
)
from open_wam.models.policy_variants.mot.modules import (
    MoTActionExpert,
    init_action_expert_from_video_core,
)
from open_wam.models.policy_variants.mot.packed_block import MoTPackedBlock
from open_wam.models.policy_variants.mot.runtime import (
    build_chunk_causal_video_mask,
    build_mot_inference_action_attention_mask,
    build_mot_attention_mask,
    build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask,
    resolve_mot_condition_latents,
    trim_mot_action_cache_prefix,
)
from open_wam.models.policy_variants.mot.runtime_routing import (
    resolve_mot_rollout_cache_window_frames,
    resolve_mot_rollout_history_frames,
)
from open_wam.models.policy_variants.mot.variant import (
    MoTPolicyVariant,
    _rewind_runtime_action_cache_to_frame,
    _slice_current_noisy_action_flow,
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
    assert pre.cross_attention_mask is not None
    assert pre.cross_attention_mask.shape == (2, 6, 5)
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


def test_build_mot_inference_action_mask_honors_strict_chunk_origin() -> None:
    origin_zero = build_mot_inference_action_attention_mask(
        video_seq_len=5,
        past_action_seq_len=0,
        current_action_seq_len=4,
        video_tokens_per_frame=1,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        window_size_frames=8,
        device=torch.device("cpu"),
        video_frame_start=0,
        current_action_frame_start=1,
        chunk_origin_frame=0,
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    )
    strict_origin = build_mot_inference_action_attention_mask(
        video_seq_len=5,
        past_action_seq_len=0,
        current_action_seq_len=4,
        video_tokens_per_frame=1,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        window_size_frames=8,
        device=torch.device("cpu"),
        video_frame_start=0,
        current_action_frame_start=1,
        chunk_origin_frame=1,
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    )

    first_current_action_query = 5
    frame0_video_key = 0
    assert not origin_zero[first_current_action_query, frame0_video_key]
    assert strict_origin[first_current_action_query, frame0_video_key]


def test_build_mot_inference_action_mask_keeps_strict_first_target_chunk_together() -> None:
    mask = build_mot_inference_action_attention_mask(
        video_seq_len=5,
        past_action_seq_len=0,
        current_action_seq_len=4,
        video_tokens_per_frame=1,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        window_size_frames=8,
        device=torch.device("cpu"),
        video_frame_start=0,
        current_action_frame_start=1,
        chunk_origin_frame=1,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )

    first_current_action_query = 5
    frame4_video_key = 4
    assert mask[first_current_action_query, frame4_video_key]


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


def test_mot_packed_coupling_action_context_mask_hides_startup_action_tokens() -> None:
    action_context_mask = torch.ones(1, 20, 1)
    action_context_mask[:, :4] = 0.0
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=5,
        video_tokens_per_frame=1,
        num_action_frames=5,
        action_tokens_per_frame=4,
        chunk_size_frames=4,
        attention_window_size=8,
        device=torch.device("cpu"),
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        chunk_origin_frame=1,
        action_context_mask=action_context_mask,
    )

    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_token_count = 5
    action_token_count = 20
    action_noisy_start = latent_token_count * 2
    action_clean_start = action_noisy_start + action_token_count
    query_action_frame1 = action_noisy_start + 4
    kv_video_clean_frame0 = latent_token_count
    kv_action_noisy_frame0 = action_noisy_start
    kv_action_clean_frame0 = action_clean_start

    assert bool(mask[query_action_frame1, kv_video_clean_frame0].item()) is True
    assert bool(mask[query_action_frame1, kv_action_noisy_frame0].item()) is False
    assert bool(mask[query_action_frame1, kv_action_clean_frame0].item()) is False
    assert bool(mask[kv_action_noisy_frame0].any().item()) is True
    assert bool(mask[:, kv_action_noisy_frame0].any().item()) is False
    assert bool(mask[:, kv_action_clean_frame0].any().item()) is False
    assert profile.metadata["invalid_action_context_tokens"] == 4


def test_mot_joint_strict_startup_action_prefix_is_not_kv_context() -> None:
    action_tokens_per_frame = 4
    action_horizon = 16
    prefix_tokens = action_tokens_per_frame
    current_action_sequence_tokens = prefix_tokens + action_horizon
    video_tokens_per_frame = 2
    current_video_sequence_frames = 5
    action_context_mask = build_strict_action_context_mask(
        batch_size=1,
        history_action_tokens=0,
        current_action_sequence_tokens=current_action_sequence_tokens,
        invalid_current_prefix_tokens=prefix_tokens,
        device=torch.device("cpu"),
    )
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=current_video_sequence_frames,
        video_tokens_per_frame=video_tokens_per_frame,
        num_action_frames=current_action_sequence_tokens // action_tokens_per_frame,
        action_tokens_per_frame=action_tokens_per_frame,
        chunk_size_frames=4,
        attention_window_size=30,
        device=torch.device("cpu"),
        current_block_coupling=CurrentBlockCoupling.JOINT,
        action_context_mask=action_context_mask,
        history_stream_visibility="video_only",
        prefix_condition_frames=1,
    )

    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_tokens = current_video_sequence_frames * video_tokens_per_frame
    action_tokens = current_action_sequence_tokens
    action_noisy_start = 2 * latent_tokens
    action_clean_start = action_noisy_start + action_tokens
    invalid_noisy = slice(action_noisy_start, action_noisy_start + prefix_tokens)
    invalid_clean = slice(action_clean_start, action_clean_start + prefix_tokens)
    real_noisy = slice(action_noisy_start + prefix_tokens, action_noisy_start + action_tokens)
    real_clean = slice(action_clean_start + prefix_tokens, action_clean_start + action_tokens)

    invalid_kv = torch.zeros(mask.shape[1], dtype=torch.bool)
    invalid_kv[invalid_noisy] = True
    invalid_kv[invalid_clean] = True
    real_queries = torch.zeros(mask.shape[0], dtype=torch.bool)
    real_queries[: 2 * latent_tokens] = True
    real_queries[real_noisy] = True
    real_queries[real_clean] = True

    assert bool(mask[real_queries][:, invalid_kv].any().item()) is False
    assert bool(mask[invalid_kv].any().item()) is True
    assert profile.metadata["invalid_action_context_tokens"] == prefix_tokens


def test_mot_packed_coupling_profile_threads_history_stream_visibility() -> None:
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=4,
        video_tokens_per_frame=1,
        num_action_frames=4,
        action_tokens_per_frame=1,
        chunk_size_frames=2,
        attention_window_size=8,
        device=torch.device("cpu"),
        current_block_coupling=CurrentBlockCoupling.JOINT,
        history_stream_visibility="video_only",
    )

    assert profile.self_attention_mask is not None
    assert profile.metadata["history_stream_visibility"] == "video_only"
    mask = profile.self_attention_mask
    latent_tokens = 4
    action_tokens = 4
    video_noisy_chunk1 = 2
    action_noisy_chunk1 = latent_tokens * 2 + action_tokens + 2
    action_clean_history = latent_tokens * 2 + action_tokens
    assert bool(mask[video_noisy_chunk1, action_clean_history].item()) is False
    assert bool(mask[action_noisy_chunk1, action_clean_history].item()) is False


def test_mot_packed_coupling_profile_threads_prefix_condition_frames() -> None:
    profile = build_mot_packed_coupling_attention_profile(
        num_video_frames=5,
        video_tokens_per_frame=1,
        num_action_frames=4,
        action_tokens_per_frame=1,
        chunk_size_frames=2,
        attention_window_size=8,
        device=torch.device("cpu"),
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        history_stream_visibility="video_only",
        prefix_condition_frames=1,
    )

    assert profile.self_attention_mask is not None
    assert profile.metadata["prefix_condition_frames"] == 1
    mask = profile.self_attention_mask
    latent_tokens = 5
    action_tokens = 4
    video_noisy_target0 = 1
    video_clean_prefix = latent_tokens
    video_clean_target0 = latent_tokens + 1
    action_noisy_target0 = latent_tokens * 2
    action_clean_target0 = latent_tokens * 2 + action_tokens

    assert bool(mask[video_noisy_target0, video_clean_prefix].item()) is True
    assert bool(mask[video_noisy_target0, action_clean_target0].item()) is False
    assert bool(mask[action_noisy_target0, video_clean_prefix].item()) is True
    assert bool(mask[action_noisy_target0, video_clean_target0].item()) is True
    assert bool(mask[action_noisy_target0, action_clean_target0].item()) is False


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


def test_mot_prefers_condition_latents_by_default() -> None:
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
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=2,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.zeros(1, 48, 4, 8, 8)
    condition_latents = torch.full_like(video_latents, 3.0)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        extra={"condition_latents": condition_latents},
    )

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=torch.randn(1, 5, 16),
    )

    assert output.policy_output.aux["video_condition_source"] == "condition_latents"
    assert torch.isfinite(output.decoder_output.loss)


def test_mot_condition_latents_can_be_disabled() -> None:
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
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=2,
            use_condition_latents=False,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.zeros(1, 48, 4, 8, 8)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        extra={"condition_latents": torch.full_like(video_latents, 3.0)},
    )

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=torch.randn(1, 5, 16),
    )

    assert output.policy_output.aux["video_condition_source"] == "video_latents"
    assert torch.isfinite(output.decoder_output.loss)


def test_mot_deprecated_text_token_proprio_context_uses_shared_batch_context_for_train() -> None:
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
            condition_mode=MoTConditionMode.FIRST_FRAME,
            video_prefix_frames=1,
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    encoder = pipeline.visual_tower.core.proprio_context_encoder
    assert encoder is not None
    with torch.no_grad():
        encoder.proj.weight.fill_(0.25)
        encoder.proj.bias.fill_(0.5)
    state = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]]])
    proprio_context_state = torch.tensor(
        [[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]]]
    )
    proprio_context_state_mask = torch.ones_like(proprio_context_state)
    proprio_context_state_mask[:, 2, 2:] = 0
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.zeros(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(video_latents, text_context=text_context)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        state=state,
        extra={
            "proprio_context_state": proprio_context_state,
            "proprio_context_state_mask": proprio_context_state_mask,
        },
    )

    prepared = pipeline.policy_variant.prepare_train_inputs(visual_outputs, batch)
    resolved = pipeline.policy_variant._resolve_text_context_with_proprio(
        pipeline.visual_tower,
        prepared.variant_inputs["text_context"],
        prepared.variant_inputs["proprio_state"],
        batch_size=1,
        device=video_latents.device,
        dtype=video_latents.dtype,
        materialize_if_missing=True,
    )

    masked_proprio = proprio_context_state * proprio_context_state_mask
    expected = encoder(masked_proprio.reshape(3, 4)).reshape(1, 3, 16).to(dtype=video_latents.dtype)
    assert torch.allclose(prepared.variant_inputs["proprio_state"], masked_proprio)
    assert resolved is not None
    assert resolved.shape == (1, 8, 16)
    assert torch.allclose(resolved[:, :5, :], text_context)
    assert torch.allclose(resolved[:, 5:, :], expected)


def test_mot_per_chunk_additive_does_not_build_text_proprio_mask() -> None:
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.randn(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(video_latents, text_context=text_context)
    proprio_context_frames = torch.randn(1, 4, 4)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        extra={
            "proprio_context_frames": proprio_context_frames,
            "proprio_context_frames_mask": torch.ones_like(proprio_context_frames),
        },
    )

    prepared = pipeline.policy_variant.prepare_train_inputs(visual_outputs, batch)
    resolved = pipeline.policy_variant._resolve_text_context_with_proprio(
        pipeline.visual_tower,
        prepared.variant_inputs["text_context"],
        prepared.variant_inputs["proprio_state"],
        batch_size=1,
        device=video_latents.device,
        dtype=video_latents.dtype,
        materialize_if_missing=True,
    )
    mask = pipeline.policy_variant._build_proprio_cross_attention_mask(
        resolved_text_context=resolved,
        proprio_state=prepared.variant_inputs["proprio_state"],
        query_frames_per_copy=4,
        tokens_per_frame=1,
        chunk_size_frames=2,
    )

    assert prepared.variant_inputs["proprio_state"] is None
    assert torch.equal(prepared.variant_inputs["hidden_proprio_state"], proprio_context_frames)
    assert resolved is not None
    assert resolved.shape == text_context.shape
    assert mask is None


def test_mot_deprecated_text_token_proprio_mask_exposes_matching_chunk_token_only() -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=6,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=6, state_dim=4, state_horizon=1),
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
            condition_mode=MoTConditionMode.FIRST_FRAME,
            video_prefix_frames=1,
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=6),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    resolved_text = torch.zeros(1, 8, 16)
    proprio_context_state = torch.zeros(1, 3, 4)

    mask = pipeline.policy_variant._build_proprio_cross_attention_mask(
        resolved_text_context=resolved_text,
        proprio_state=proprio_context_state,
        query_frames_per_copy=6,
        tokens_per_frame=1,
        chunk_size_frames=2,
    )

    assert mask is not None
    assert mask.shape == (1, 6, 8)
    assert torch.equal(mask[0, :, :5], torch.ones(6, 5, dtype=torch.bool))
    assert torch.equal(
        mask[0, :, 5:],
        torch.tensor(
            [
                [True, False, False],
                [True, False, False],
                [False, True, False],
                [False, True, False],
                [False, False, True],
                [False, False, True],
            ],
            dtype=torch.bool,
        ),
    )


def test_mot_chunk_origin_aligns_one_frame_context_with_first_target_chunk() -> None:
    video_mask = build_chunk_causal_video_mask(
        video_seq_len=5,
        video_tokens_per_frame=1,
        action_chunk_size_frames=4,
        device=torch.device("cpu"),
        chunk_origin_frame=1,
    )

    # Context frame 0 is chunk -1, so it must not see target frame 1. Target
    # frames 1 and 4 remain in the same generated chunk.
    assert bool(video_mask[0, 1].item()) is False
    assert bool(video_mask[4, 1].item()) is True

    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=5,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=5, state_dim=4, state_horizon=1),
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
            condition_mode=MoTConditionMode.FIRST_FRAME,
            video_prefix_frames=1,
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=5),
        training=TrainingConfig(chunk_size=4, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=4),
    )
    pipeline = build_variant_pipeline_from_config(config)
    resolved_text = torch.zeros(1, 3, 16)
    proprio_context_state = torch.zeros(1, 2, 4)

    cross_mask = pipeline.policy_variant._build_proprio_cross_attention_mask(
        resolved_text_context=resolved_text,
        proprio_state=proprio_context_state,
        query_frames_per_copy=5,
        tokens_per_frame=1,
        chunk_size_frames=4,
        chunk_origin_frame=1,
    )

    assert cross_mask is not None
    assert cross_mask[0, 0, :3].tolist() == [True, False, False]
    assert cross_mask[0, 1, :3].tolist() == [True, True, False]
    assert cross_mask[0, 4, :3].tolist() == [True, True, False]


def test_mot_packed_block_accepts_query_dependent_cross_attention_masks() -> None:
    video_core = SharedVideoTransformerCore(
        SharedVideoTransformerConfig(
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
        action_dim=4,
        state_dim=4,
    )
    action_expert = MoTActionExpert(
        hidden_size=32,
        action_dim=4,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
    )
    packed_block = MoTPackedBlock(video_core.blocks[0], action_expert.blocks[0])
    batch_size = 2
    video_tokens = 3
    action_tokens = 2
    context_tokens = 5

    video_out, action_out = packed_block(
        torch.randn(batch_size, video_tokens, 32),
        torch.randn(batch_size, action_tokens, 32),
        video_timestep_proj=torch.randn(batch_size, video_tokens, 6, 32),
        video_rotary_emb=None,
        action_temb=torch.randn(batch_size, action_tokens, 6, 32),
        action_rotary_emb=None,
        video_attention_mask=None,
        action_attention_mask=None,
        video_text_hidden_states=torch.randn(batch_size, context_tokens, 32),
        action_text_hidden_states=torch.randn(batch_size, context_tokens, 32),
        video_cross_attention_mask=torch.ones(batch_size, video_tokens, context_tokens, dtype=torch.bool),
        action_cross_attention_mask=torch.ones(batch_size, action_tokens, context_tokens, dtype=torch.bool),
    )

    assert video_out.shape == (batch_size, video_tokens, 32)
    assert action_out.shape == (batch_size, action_tokens, 32)
    assert torch.isfinite(video_out).all()
    assert torch.isfinite(action_out).all()


def test_mot_prepare_infer_state_appends_deprecated_proprio_context_token() -> None:
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
            condition_mode=MoTConditionMode.FIRST_FRAME,
            video_prefix_frames=1,
            teacher_forcing_video_noise_prob=0.0,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=0.0),
        inference=InferenceConfig(frame_chunk_size=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    encoder = pipeline.visual_tower.core.proprio_context_encoder
    assert encoder is not None
    with torch.no_grad():
        encoder.proj.weight.fill_(0.1)
        encoder.proj.bias.fill_(0.2)
    state = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0]]])
    video_latents = torch.randn(1, 48, 4, 8, 8)
    text_context = torch.zeros(1, 5, 16)
    visual_outputs = pipeline.prepare_visual_outputs_from_latents(video_latents, text_context=text_context)

    infer_state = pipeline.policy_variant.prepare_infer_state(
        visual_tower=pipeline.visual_tower,
        visual_outputs=visual_outputs,
        context=PolicyInferContext(state=state),
    )

    runtime_state = infer_state.variant_state
    assert isinstance(runtime_state, MoTRuntimeState)
    assert runtime_state.text_context is not None
    expected = encoder(state[:, -1, :]).to(dtype=runtime_state.text_context.dtype)
    assert torch.allclose(runtime_state.proprio_state, state[:, -1, :])
    assert runtime_state.text_context.shape == (1, 6, 16)
    assert torch.allclose(runtime_state.text_context[:, -1, :], expected)
    assert torch.allclose(
        runtime_state.text_context[:, :5, :],
        torch.zeros_like(runtime_state.text_context[:, :5, :]),
    )


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
    if current_block_coupling in {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }:
        assert pipeline.policy_variant._legacy_inference_blocks_restored is True
    else:
        assert pipeline.policy_variant._legacy_inference_blocks_restored is False


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    ],
)
def test_mot_legacy_split_cache_infer_threads_per_chunk_action_proprio(
    current_block_coupling: CurrentBlockCoupling,
    monkeypatch: pytest.MonkeyPatch,
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
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    captured_hidden_contexts: list[torch.Tensor | None] = []
    original_pre_dit = pipeline.policy_variant.action_expert.pre_dit

    def capture_pre_dit(*args, **kwargs):
        hidden_context = kwargs.get("hidden_context")
        captured_hidden_contexts.append(None if hidden_context is None else hidden_context.detach().clone())
        return original_pre_dit(*args, **kwargs)

    monkeypatch.setattr(pipeline.policy_variant.action_expert, "pre_dit", capture_pre_dit)

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.randn(1, 5, 16),
    )

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert pipeline.policy_variant._legacy_inference_blocks_restored is True
    assert len(captured_hidden_contexts) == 3
    for hidden_context in captured_hidden_contexts:
        assert hidden_context is not None
        assert hidden_context.shape == (1, 4, 32)


def test_mot_generalist_packed_infer_couples_action_to_video_sigma_schedule(
    monkeypatch: pytest.MonkeyPatch,
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            mot_generalist_training_mode_probs={MoTGeneralistTrainingMode.JOINT: 1.0},
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(
            chunk_size=2,
            window_size=8,
            video_sigma_shift=5.0,
            action_sigma_shift=1.0,
            action_loss_weight=1.0,
            latent_loss_weight=1.0,
        ),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    captured_action_timesteps: list[torch.Tensor] = []
    original_pre_dit = pipeline.policy_variant.action_expert.pre_dit

    def capture_pre_dit(*args, **kwargs):
        captured_action_timesteps.append(kwargs["timestep"].detach().clone())
        return original_pre_dit(*args, **kwargs)

    monkeypatch.setattr(pipeline.policy_variant.action_expert, "pre_dit", capture_pre_dit)
    infer_state = PolicyInferState(step_index=1)
    infer_state.cursor.current_start_frame = 2
    video_latents = torch.randn(1, 48, 2, 8, 8)
    text_context = torch.randn(1, 5, 16)

    output = pipeline.forward_infer_step_from_latents(
        video_latents,
        context=PolicyInferContext(),
        infer_state=infer_state,
        text_context=text_context,
    )

    assert output.policy_output.aux["mot_packed_history_debug"]["coupled_action_video_sigmas"] is True
    assert (
        output.policy_output.aux["mot_packed_history_debug"]["joint_timestep_coupling"]
        == JointTimestepCoupling.MATCH_SIGMA.value
    )
    assert len(captured_action_timesteps) == 2
    first_step_noisy_action_t = captured_action_timesteps[0][0, :4]
    second_step_noisy_action_t = captured_action_timesteps[1][0, :4]
    assert torch.allclose(first_step_noisy_action_t, torch.full_like(first_step_noisy_action_t, 1000.0))
    assert torch.allclose(second_step_noisy_action_t, torch.full_like(second_step_noisy_action_t, 833.0))
    assert not torch.allclose(second_step_noisy_action_t, torch.full_like(second_step_noisy_action_t, 500.0))


def test_slice_current_noisy_action_flow_skips_packed_history_tokens() -> None:
    packed_action_flow = torch.arange(2 * 20 * 3, dtype=torch.float32).reshape(2, 20, 3)

    current = _slice_current_noisy_action_flow(
        packed_action_flow,
        history_action_tokens=6,
        action_horizon=8,
    )

    assert current.shape == (2, 8, 3)
    assert torch.equal(current, packed_action_flow[:, 6:14])
    assert current.is_contiguous()


@pytest.mark.parametrize(
    "current_block_coupling",
    [
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
def test_mot_packed_infer_modes_keep_two_chunk_history(
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


def test_mot_packed_infer_chunk0_uses_one_frame_startup_bootstrap() -> None:
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
            current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
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
    assert first.policy_output.aux["generation_frame_start"] == 1
    assert first.policy_output.aux["mot_action_cond_tokens"] == 0
    assert first.policy_output.aux["mot_invalid_startup_action_tokens"] == 2
    assert first.policy_output.aux["mot_action_context_invalid_tokens"] == 2
    assert first.decoder_output.action_pred.shape == (1, 4, 4)

    packed_state = first.policy_output.next_state.variant_state
    assert isinstance(packed_state, MoTRuntimeState)
    assert packed_state.past_clean_latents.shape[2] == 3
    assert packed_state.past_clean_actions.shape[1] == 4
    assert first.policy_output.next_state.cursor.current_start_frame == 3
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
    assert history_debug["past_clean_latent_frames"] == 3
    assert history_debug["past_clean_action_frames"] == 2
    assert history_debug["shared_history_frames"] == 2
    assert history_debug["current_observed_latent_frames"] == 2
    assert history_debug["current_clean_condition_frames"] == 2


def test_mot_action_then_video_action_only_rollout_skips_predicted_video() -> None:
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
            current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(extra={"mot_action_only_rollout": True}),
        text_context=torch.randn(1, 5, 16),
    )

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert output.policy_output.aux["mot_action_only_rollout"] is True
    assert output.policy_output.aux["predicted_latents"].shape[2] == 0
    assert output.policy_output.aux["mot_infer_artifacts"].predicted_latents.shape[2] == 0
    packed_state = output.policy_output.next_state.variant_state
    assert isinstance(packed_state, MoTRuntimeState)
    assert packed_state.pending_predicted_video_frames == 0
    assert packed_state.past_clean_latents is not None
    assert packed_state.past_clean_latents.shape[2] == 1
    assert packed_state.past_clean_actions is not None
    assert packed_state.past_clean_actions.shape[1] == 4


def test_mot_action_then_video_action_only_rollout_preserves_hidden_proprio_alignment() -> None:
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
            current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
            video_prefix_frames=1,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    first = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(
            state=torch.ones(1, 1, 4),
            extra={"mot_action_only_rollout": True},
        ),
        text_context=torch.randn(1, 5, 16),
    )
    first_state = first.policy_output.next_state.variant_state
    assert isinstance(first_state, MoTRuntimeState)
    assert first_state.past_hidden_proprio_states is not None
    assert first_state.past_clean_latents is not None
    assert first_state.past_hidden_proprio_states.shape[1] == first_state.past_clean_latents.shape[2]

    warmed_state = first_state
    warmed_state.past_clean_latents = torch.cat(
        [
            warmed_state.past_clean_latents,
            torch.randn(1, 48, 4, 8, 8),
        ],
        dim=2,
    )
    warmed_state.past_clean_actions = torch.cat(
        [
            warmed_state.past_clean_actions,
            torch.randn(1, 4, 4),
        ],
        dim=1,
    )
    warmed_state.past_hidden_proprio_states = torch.cat(
        [
            warmed_state.past_hidden_proprio_states,
            torch.full((1, 4, 4), 2.0),
        ],
        dim=1,
    )
    warmed_infer_state = first.policy_output.next_state
    warmed_infer_state.variant_state = warmed_state
    warmed_infer_state.step_index = 2
    warmed_infer_state.cursor.current_start_frame = 5

    second = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 4, 8, 8),
        context=PolicyInferContext(
            state=torch.full((1, 1, 4), 3.0),
            extra={"mot_action_only_rollout": True},
        ),
        infer_state=warmed_infer_state,
        text_context=torch.randn(1, 5, 16),
    )

    second_state = second.policy_output.next_state.variant_state
    assert isinstance(second_state, MoTRuntimeState)
    assert second_state.past_clean_latents is not None
    assert second_state.past_hidden_proprio_states is not None
    assert second_state.past_hidden_proprio_states.shape[1] == second_state.past_clean_latents.shape[2]
    assert second.policy_output.aux["predicted_latents"].shape[2] == 0


def test_mot_action_only_rollout_rejects_video_then_action() -> None:
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

    with pytest.raises(ValueError, match="action_then_video.*decoupled_same_step"):
        pipeline.forward_infer_step_from_latents(
            torch.randn(1, 48, 1, 8, 8),
            context=PolicyInferContext(extra={"mot_action_only_rollout": True}),
            text_context=torch.randn(1, 5, 16),
        )


def test_mot_decoupled_action_only_rollout_skips_split_cache_video_denoise() -> None:
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
            current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(extra={"mot_action_only_rollout": True}),
        text_context=torch.randn(1, 5, 16),
    )

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert output.policy_output.aux["mot_action_only_rollout"] is True
    assert output.policy_output.aux["predicted_latents"].shape[2] == 0
    assert output.policy_output.aux["mot_cache_debug"]["mot_action_only_rollout"] is True
    assert output.policy_output.aux["mot_cache_debug"]["video_commit_before_action"] is False
    assert output.policy_output.aux["mot_infer_artifacts"].predicted_latents.shape[2] == 0


def test_mot_rollout_history_window_matches_fixed128_context_contract() -> None:
    assert resolve_mot_rollout_history_frames(window_size=30, frame_chunk_size=4) == 60
    assert resolve_mot_rollout_cache_window_frames(window_size=30, frame_chunk_size=4) == 64
    assert resolve_mot_rollout_history_frames(window_size=31, frame_chunk_size=4) == 60
    assert resolve_mot_rollout_cache_window_frames(window_size=31, frame_chunk_size=4) == 64
    assert resolve_mot_rollout_history_frames(window_size=8, frame_chunk_size=2) == 8
    assert resolve_mot_rollout_cache_window_frames(window_size=8, frame_chunk_size=2) == 10


def test_mot_packed_infer_uses_rollout_history_contract_for_cached_context() -> None:
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_prefix_frames=1,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    action_tokens_per_frame = config.data.action_schema.action_horizon // config.inference.frame_chunk_size
    runtime_state = MoTRuntimeState(
        past_clean_latents=torch.randn(1, 48, 12, 8, 8),
        past_clean_actions=torch.randn(1, 12 * action_tokens_per_frame, 4),
    )
    infer_state = PolicyInferState(
        step_index=6,
        cursor=RolloutCursor(current_start_frame=12, chunk_size=2),
        variant_state=runtime_state,
    )

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 2, 8, 8),
        context=PolicyInferContext(),
        infer_state=infer_state,
        text_context=torch.randn(1, 5, 16),
    )

    history_debug = output.policy_output.aux["mot_packed_history_debug"]
    assert history_debug["history_window_frames"] == 10
    assert history_debug["max_history_frames"] == 8
    assert history_debug["shared_history_frames"] == 8
    next_state = output.policy_output.next_state.variant_state
    assert isinstance(next_state, MoTRuntimeState)
    assert next_state.past_clean_latents is not None
    assert next_state.past_clean_actions is not None
    assert next_state.past_clean_latents.shape[2] == 10
    assert next_state.past_clean_actions.shape[1] == 10 * action_tokens_per_frame


def test_mot_legacy_prefix_prepends_current_state_to_hidden_proprio() -> None:
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=4, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=4),
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.zeros(1, 48, 4, 8, 8)
    condition_latents = torch.ones_like(video_latents)
    target_states = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]).expand(-1, -1, 4).contiguous()
    batch = PolicyTrainBatch(actions=torch.zeros(1, 4, 4), state=torch.full((1, 1, 4), 9.0))

    model_video_latents, hidden_state, prefix_frames, source = pipeline.policy_variant._prepend_legacy_prefix_video_latents(
        video_latents=video_latents,
        condition_latents=condition_latents,
        hidden_proprio_state=target_states,
        batch=batch,
    )

    assert model_video_latents.shape[2] == 5
    assert prefix_frames == 1
    assert source == "condition_latents_prefix"
    assert hidden_state is not None
    assert hidden_state.shape == (1, 5, 4)
    assert torch.equal(hidden_state[:, 0, :], torch.full((1, 4), 9.0))
    assert torch.equal(hidden_state[:, 1:, :], target_states)


def test_mot_legacy_prefix_action_hidden_proprio_uses_causal_chunk_boundaries() -> None:
    states = torch.tensor([[[9.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]])

    resolved = MoTPolicyVariant._legacy_prefix_action_hidden_proprio_state(
        states,
        prefix_condition_frames=1,
        target_num_frames=8,
        chunk_size_frames=4,
    )

    assert resolved is not None
    assert resolved.squeeze(-1).tolist() == [[9.0, 9.0, 9.0, 9.0, 4.0, 4.0, 4.0, 4.0]]


def test_mot_legacy_prefix_requires_frame_level_hidden_proprio() -> None:
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=4, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=4),
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 4, 4),
        state=torch.zeros(1, 1, 4),
        extra={"proprio_context_state": torch.zeros(1, 1, 4)},
    )

    with pytest.raises(ValueError, match="requires frame-level `proprio_context_frames`"):
        pipeline.policy_variant._resolve_train_hidden_proprio_context(batch)


def test_mot_packed_strict_old_infer_skips_video_hidden_proprio(
    monkeypatch: pytest.MonkeyPatch,
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_prefix_frames=1,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
            parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    captured_video_inputs: list[torch.Tensor | None] = []
    captured_action_inputs: list[torch.Tensor | None] = []
    original_video_hidden = pipeline.policy_variant._video_hidden_context_for_tokens
    original_action_hidden = pipeline.policy_variant._action_hidden_context_for_tokens

    def capture_video_hidden(*args, **kwargs):
        hidden_proprio_state = args[1] if len(args) > 1 else None
        captured_video_inputs.append(None if hidden_proprio_state is None else hidden_proprio_state.detach().clone())
        return original_video_hidden(*args, **kwargs)

    def capture_action_hidden(*args, **kwargs):
        hidden_proprio_state = args[1] if len(args) > 1 else None
        captured_action_inputs.append(None if hidden_proprio_state is None else hidden_proprio_state.detach().clone())
        return original_action_hidden(*args, **kwargs)

    monkeypatch.setattr(pipeline.policy_variant, "_video_hidden_context_for_tokens", capture_video_hidden)
    monkeypatch.setattr(pipeline.policy_variant, "_action_hidden_context_for_tokens", capture_action_hidden)

    output = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.randn(1, 5, 16),
    )

    assert output.decoder_output.action_pred.shape == (1, 4, 4)
    assert captured_video_inputs == []
    assert captured_action_inputs
    assert all(hidden_state is not None for hidden_state in captured_action_inputs)


def test_mot_packed_infer_tracks_per_chunk_hidden_proprio_history() -> None:
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
            current_block_coupling=CurrentBlockCoupling.JOINT,
            video_prefix_frames=1,
            num_action_layers=1,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=4),
        training=TrainingConfig(chunk_size=2, window_size=8, action_loss_weight=1.0, latent_loss_weight=1.0),
        inference=InferenceConfig(frame_chunk_size=2, video_num_inference_steps=2, action_num_inference_steps=2),
    )
    pipeline = build_variant_pipeline_from_config(config)
    first = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 1, 8, 8),
        context=PolicyInferContext(state=torch.ones(1, 1, 4)),
        text_context=torch.randn(1, 5, 16),
    )
    first_state = first.policy_output.next_state.variant_state
    assert isinstance(first_state, MoTRuntimeState)
    assert first_state.past_hidden_proprio_states is not None
    assert first_state.past_hidden_proprio_states.shape == (1, 3, 4)

    second = pipeline.forward_infer_step_from_latents(
        torch.randn(1, 48, 2, 8, 8),
        context=PolicyInferContext(state=torch.full((1, 1, 4), 2.0)),
        infer_state=first.policy_output.next_state,
        text_context=torch.randn(1, 5, 16),
    )
    second_state = second.policy_output.next_state.variant_state
    assert isinstance(second_state, MoTRuntimeState)
    assert second_state.past_hidden_proprio_states is not None
    assert second_state.past_hidden_proprio_states.shape[1] == second_state.past_clean_latents.shape[2]


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
