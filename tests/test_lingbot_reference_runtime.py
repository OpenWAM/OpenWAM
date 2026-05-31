from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_wam.configs import (
    InferenceConfig,
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelContextConditionLatentSource,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    ProprioContextMode,
    TrainingConfig,
)
from open_wam.models.action_decoders.lingbot_parallel_decoder import LingbotParallelActionDecoder
from open_wam.models.common import (
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    SlotPoolLayerState,
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler as SharedFlowMatchScheduler
from open_wam.models.policy_variants.contracts import PolicyTrainBatch, PolicyTrainOutput
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    ExactCacheInterfaceSpec,
    FlowMatchScheduler,
    _write_exact_cache_chunk,
    initialize_reference_cache,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
    repeat_input_for_cfg,
    run_parallel_current_frame_action_chunk_inference_rollout,
    run_parallel_action_conditioned_action_override_inference_rollout,
    run_parallel_action_conditioned_inference_rollout,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_parallel_fastwam_first_frame_train,
    run_reference_single_stream_forward,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime as reference_runtime_module
from open_wam.models.policy_variants.parallel_stream.variant import ParallelStreamPolicyVariant
from open_wam.models.video_backbone.contracts import ChunkMetadata, ConditioningState, TokenGridMetadata
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig, SharedVideoTransformerConfig
from open_wam.models.visual_tower.contracts import VisualFrontendOutput, VisualStageOutputs
from open_wam.models.visual_tower import replica_core as replica_core_module
from open_wam.models.visual_tower.replica_core import (
    SharedVideoTransformerCore,
    _retained_slot_pool_indices_for_current_write,
)
from open_wam.models.visual_tower.sequence_adapters import prepare_exact_dual_stream_train_sequence
from open_wam.models.visual_tower.tower import VisualTower
from scripts.deprecated.run_libero_exact_visualization import (
    _binarize_raw_gripper_actions,
    _build_warmup_raw_actions,
    _extract_libero_eef_axisangle_gripper_state,
    _select_executed_raw_actions,
)


def test_repeat_input_for_cfg_preserves_hidden_context() -> None:
    input_dict = {
        "noisy_latents": torch.randn(2, 3, 1, 2, 2),
        "text_emb": torch.randn(2, 4, 8),
        "grid_id": torch.zeros(2, 4, 1),
        "timesteps": torch.zeros(2, 1),
        "hidden_context": torch.randn(2, 4, 8),
    }
    negative_text_emb = torch.randn(2, 4, 8)

    repeated = repeat_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)

    assert repeated["hidden_context"].shape == (4, 4, 8)
    torch.testing.assert_close(repeated["hidden_context"][:2], input_dict["hidden_context"])
    torch.testing.assert_close(repeated["hidden_context"][2:], input_dict["hidden_context"])


def test_m1_reference_runtime_uses_shared_flow_match_scheduler() -> None:
    assert FlowMatchScheduler is SharedFlowMatchScheduler


class _FakeReferenceTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (1, 2, 2)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.cache_batch_sizes: dict[str, int] = {}
        self.cache_layouts: dict[str, tuple[int, int]] = {}
        self.cache_attn_windows: dict[str, int] = {}
        self.cleared_pred_cache_names: list[str] = []
        self.last_text_emb: torch.Tensor | None = None
        self.last_noisy_latents: torch.Tensor | None = None

    def clear_cache(self, cache_name: str) -> None:
        self.cache_batch_sizes.pop(cache_name, None)

    def clear_pred_cache(self, cache_name: str) -> None:
        self.cleared_pred_cache_names.append(cache_name)

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "lingbot_slot_pool",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        del device, dtype, backend_name, prefix_visibility_mode
        self.cache_batch_sizes[cache_name] = batch_size
        self.cache_layouts[cache_name] = (latent_token_per_chunk, action_token_per_chunk)
        self.cache_attn_windows[cache_name] = int(attn_window)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        batch_size = input_dict["noisy_latents"].shape[0]
        self.last_text_emb = input_dict["text_emb"].detach().clone()
        self.last_noisy_latents = input_dict["noisy_latents"].detach().clone()
        if update_cache and cache_name in self.cache_batch_sizes:
            assert batch_size == self.cache_batch_sizes[cache_name]
        latents = input_dict["noisy_latents"]
        if action_mode:
            return latents.squeeze(-1).permute(0, 2, 3, 1).reshape(batch_size, -1, latents.shape[1])
        patch_t, patch_h, patch_w = self.patch_size
        return (
            latents.view(
                batch_size,
                latents.shape[1],
                latents.shape[2] // patch_t,
                patch_t,
                latents.shape[3] // patch_h,
                patch_h,
                latents.shape[4] // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch_size, -1, latents.shape[1] * patch_t * patch_h * patch_w)
        )


class _GradTrackingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.grad_enabled_during_forward: bool | None = None

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        del update_cache, cache_name, action_mode
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return input_dict["noisy_latents"] * self.weight


def test_exact_visualization_partial_execution_selects_executed_tail() -> None:
    raw_actions = torch.arange(4 * 4 * 7, dtype=torch.float32).reshape(4, 4, 7)

    executed = _select_executed_raw_actions(
        raw_actions,
        start_frame_group=0,
        execute_action_steps=8,
        action_per_frame=4,
    )

    assert torch.equal(executed, raw_actions[:2])


def test_exact_visualization_warmup_rejects_skipped_first_chunk_prefix() -> None:
    raw_actions = torch.arange(4 * 4 * 7, dtype=torch.float32).reshape(4, 4, 7)
    executed = _select_executed_raw_actions(
        raw_actions,
        start_frame_group=1,
        execute_action_steps=None,
        action_per_frame=4,
    )

    with pytest.raises(ValueError, match="deprecated"):
        _build_warmup_raw_actions(
            raw_actions=raw_actions,
            executed_raw_actions=executed,
            start_frame_group=1,
            first_chunk=True,
            exact_startup_bootstrap_padding=False,
            partial_execution_enabled=False,
            binarize_gripper=False,
        )


def test_exact_visualization_gripper_binarization_applies_to_last_channel_only() -> None:
    raw_actions = torch.tensor(
        [
            [[0.1, -0.2, 0.0], [0.3, 0.4, -0.1]],
            [[0.5, 0.6, 2.0], [0.7, 0.8, -3.0]],
        ],
        dtype=torch.float32,
    )

    binarized = _binarize_raw_gripper_actions(raw_actions)

    assert torch.equal(binarized[..., :2], raw_actions[..., :2])
    assert torch.equal(binarized[..., -1], torch.tensor([[1.0, -1.0], [1.0, -1.0]]))


def test_libero_proprio_state_matches_eef_axisangle_gripper_2d() -> None:
    state = _extract_libero_eef_axisangle_gripper_state(
        {
            "robot0_eef_pos": [1.0, 2.0, 3.0],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [0.4, 0.5],
        }
    )

    assert state.dtype == torch.empty((), dtype=torch.float32).numpy().dtype
    torch.testing.assert_close(
        torch.from_numpy(state),
        torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.4, 0.5], dtype=torch.float32),
    )


def test_deprecated_text_token_proprio_encoder_is_default_off_and_zero_init() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    text_emb = torch.randn(2, 5, 16)
    assert core.proprio_context_encoder is None
    assert core.append_proprio_context_tokens(text_emb, torch.randn(2, 8)) is text_emb  # deprecated helper

    core.configure_proprio_context_encoder(enabled=True, state_dim=8)
    assert core.proprio_context_encoder is not None
    assert "proprio_context_encoder.proj.weight" in core.state_dict()
    appended = core.append_proprio_context_tokens(text_emb, torch.randn(2, 8))  # deprecated helper

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)
    assert torch.equal(appended[:, 5:], torch.zeros_like(appended[:, 5:]))


def test_generalist_mode_context_encoder_is_default_off_and_small_random_init() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    text_emb = torch.randn(2, 5, 16)
    assert core.generalist_mode_context_encoder is None
    assert core.append_generalist_mode_context_token(text_emb, "joint") is text_emb

    core.configure_generalist_mode_context_encoder(enabled=True)
    assert core.generalist_mode_context_encoder is not None
    assert "generalist_mode_context_encoder.embedding.weight" in core.state_dict()
    appended = core.append_generalist_mode_context_token(
        text_emb,
        ["joint", "video_conditioned_action"],
    )

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)
    assert not torch.equal(appended[:, 5:], torch.zeros_like(appended[:, 5:]))
    assert float(appended[:, 5:].detach().abs().max()) < 0.2
    assert not torch.equal(appended[0, 5], appended[1, 5])


def test_generalist_mode_context_rejects_out_of_range_tensor_indices() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)

    with pytest.raises(ValueError, match="Generalist mode tensor indices"):
        core.append_generalist_mode_context_token(
            torch.randn(1, 5, 16),
            torch.tensor([3]),
        )


def test_generalist_mode_context_injection_preserves_cfg_negative_branch() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        generalist_mode_text_token=True,
    )
    text_emb = torch.randn(1, 4, 16)
    negative_text_emb = torch.randn(1, 4, 16)

    appended, appended_negative = reference_runtime_module._inject_generalist_mode_text_context(
        core,
        policy_config=policy_config,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
    )

    assert appended.shape == (1, 5, 16)
    assert appended_negative is not None
    assert appended_negative.shape == (1, 5, 16)
    assert torch.equal(appended[:, :4], text_emb)
    assert torch.equal(appended_negative[:, :4], negative_text_emb)
    assert not torch.equal(appended[:, 4:], torch.zeros_like(appended[:, 4:]))
    assert not torch.equal(appended_negative[:, 4:], torch.zeros_like(appended_negative[:, 4:]))
    torch.testing.assert_close(appended[:, 4:], appended_negative[:, 4:])
    assert (
        reference_runtime_module._generalist_mode_for_action_conditioning("forced_action_joint_fdm")
        == JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    assert (
        reference_runtime_module._generalist_mode_for_action_conditioning("vanilla_joint_rollout")
        == JointDenoiseTrainingMode.JOINT
    )


@pytest.mark.parametrize(
    ("rollout_mode", "expected_mode"),
    [
        ("joint", JointDenoiseTrainingMode.JOINT),
        ("vanilla_joint_rollout", JointDenoiseTrainingMode.JOINT),
        ("clean_action_feedback", JointDenoiseTrainingMode.JOINT),
        ("fdm", JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO),
        ("forced_action_joint_fdm", JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO),
        ("idm", JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION),
        ("video_conditioned_action", JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION),
    ],
)
def test_generalist_mode_context_maps_rollout_modes(
    rollout_mode: str,
    expected_mode: JointDenoiseTrainingMode,
) -> None:
    assert reference_runtime_module._generalist_mode_for_action_conditioning(rollout_mode) == expected_mode


def test_generalist_mode_context_rejects_unknown_rollout_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported joint-denoise rollout mode"):
        reference_runtime_module._generalist_mode_for_action_conditioning("unknown_rollout_mode")


def test_generalist_conditional_local_window_covers_full_previous_video_action_chunk() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 8, 1, 1),
        action_shape=(1, 1, 8, 1, 1),
        padded_length=0,
        chunk_size=4,
        window_size=3,
        patch_size=(1, 1, 1),
        text_token_count=0,
        chunk_origin_frame=0,
        device=torch.device("cpu"),
        build_dense_masks=True,
        current_block_coupling=CurrentBlockCoupling.JOINT,
    )
    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_tokens = 8
    action_tokens = 8
    current_video_noisy_frame4 = 4
    current_video_clean_frame4 = latent_tokens + 4
    current_action_noisy_frame4 = 2 * latent_tokens + 4
    previous_video_clean_frame0 = latent_tokens + 0
    previous_action_clean_frame0 = 2 * latent_tokens + action_tokens + 0
    current_action_clean_frame4 = 2 * latent_tokens + action_tokens + 4

    assert mask[current_action_noisy_frame4, previous_video_clean_frame0]
    assert mask[current_action_noisy_frame4, previous_action_clean_frame0]
    assert not mask[current_video_noisy_frame4, current_video_clean_frame4]
    assert not mask[current_action_noisy_frame4, current_action_clean_frame4]

    too_narrow_profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 8, 1, 1),
        action_shape=(1, 1, 8, 1, 1),
        padded_length=0,
        chunk_size=4,
        window_size=2,
        patch_size=(1, 1, 1),
        text_token_count=0,
        chunk_origin_frame=0,
        device=torch.device("cpu"),
        build_dense_masks=True,
        current_block_coupling=CurrentBlockCoupling.JOINT,
    )
    assert too_narrow_profile.self_attention_mask is not None
    assert not too_narrow_profile.self_attention_mask[current_action_noisy_frame4, previous_video_clean_frame0]


def test_generalist_conditional_rollout_modes_use_local_history_window() -> None:
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        video_condition_on_action=True,
        attn_window=30,
    )

    assert (
        reference_runtime_module._window_size_for_generalist_conditioning(
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            fallback_window_size=policy_config.attn_window,
        )
        == 3
    )
    assert (
        reference_runtime_module._window_size_for_generalist_conditioning(
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
            fallback_window_size=policy_config.attn_window,
        )
        == 3
    )
    assert (
        reference_runtime_module._window_size_for_generalist_conditioning(
            JointDenoiseTrainingMode.JOINT,
            fallback_window_size=policy_config.attn_window,
        )
        == 30
    )


def test_generalist_mode_context_requires_configured_encoder() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        generalist_mode_text_token=True,
    )

    with pytest.raises(ValueError, match="append exactly one token"):
        reference_runtime_module._inject_generalist_mode_text_context(
            core,
            policy_config=policy_config,
            text_emb=torch.randn(1, 4, 16),
            negative_text_emb=None,
            mode=JointDenoiseTrainingMode.JOINT,
        )


def test_action_conditioned_rollout_wrapper_forwards_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_impl(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(reference_runtime_module, "_run_parallel_action_conditioned_inference_rollout_impl", fake_impl)

    result = run_parallel_action_conditioned_inference_rollout(
        transformer=object(),
        backbone_config=object(),
        policy_config=object(),
        training_config=object(),
        inference_config=object(),
        action_dim=7,
        condition_latents=None,
        text_emb=None,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        action_conditioning_mode="idm",
    )

    assert result is sentinel
    assert captured["action_conditioning_mode"] == "idm"


def test_visual_tower_configures_deprecated_text_token_proprio_encoder_before_runtime_load() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    tower = VisualTower(backbone_config, action_dim=4, state_dim=8, proprio_context_state_dim=8)

    assert isinstance(tower.core, SharedVideoTransformerCore)
    assert tower.core.proprio_context_encoder is not None
    assert "proprio_context_encoder.proj.weight" in tower.core.state_dict()


def test_parallel_variant_attach_configures_generalist_mode_encoder() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        generalist_mode_text_token=True,
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    tower = VisualTower(backbone_config, action_dim=4, state_dim=8)

    variant.attach_visual_tower(tower)

    assert tower.core.generalist_mode_context_encoder is not None
    assert tower.core.proprio_context_encoder is not None
    assert "generalist_mode_context_encoder.embedding.weight" in tower.core.state_dict()


def test_visual_tower_loads_exported_generalist_mode_encoder_when_preconfigured(
    tmp_path: Path,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        pretrained_model_name_or_path=str(tmp_path),
        load_reference_core_weights=True,
    )
    probe_core = SharedVideoTransformerCore(backbone_config, action_dim=4, state_dim=8)
    probe_core.configure_generalist_mode_context_encoder(enabled=True)
    exported_state = {
        # Marks the safetensors file as an Open-WAM exported runtime backbone.
        "time_conditioner.time_proj.weight": probe_core.state_dict()[
            "time_conditioner.time_proj.weight"
        ].clone(),
        "generalist_mode_context_encoder.embedding.weight": torch.arange(
            3 * 16,
            dtype=torch.float32,
        ).reshape(3, 16),
    }
    save_file(exported_state, transformer_dir / "diffusion_pytorch_model.safetensors")

    tower = VisualTower(
        backbone_config,
        action_dim=4,
        state_dim=8,
        generalist_mode_context_enabled=True,
    )

    assert tower.core.generalist_mode_context_encoder is not None
    assert torch.equal(
        tower.core.generalist_mode_context_encoder.embedding.weight.detach().cpu(),
        exported_state["generalist_mode_context_encoder.embedding.weight"],
    )
    assert "generalist_mode_context_encoder.embedding.weight" in (
        tower.reference_core_load_report.loaded_keys
    )


def test_deprecated_proprio_context_appending_preserves_existing_text_tokens() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_proprio_context_encoder(enabled=True, state_dim=8)

    text_emb = torch.ones(2, 5, 16)
    appended = core.append_proprio_context_tokens(text_emb, torch.randn(2, 8))  # deprecated helper

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)


def test_deprecated_proprio_context_appending_runs_exact_single_stream_forward() -> None:
    torch.manual_seed(0)
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
            patch_size_t=1,
            patch_size_h=2,
            patch_size_w=2,
        ),
        action_dim=4,
        state_dim=8,
    ).eval()
    text_emb = torch.randn(1, 6, 16)
    input_dict = {
        "noisy_latents": torch.randn(1, 48, 1, 2, 2),
        "text_emb": text_emb,
        "grid_id": torch.zeros(1, 4, 1),
        "timesteps": torch.zeros(1, 1),
    }

    core.configure_proprio_context_encoder(enabled=True, state_dim=8)
    appended_input = dict(input_dict)
    appended_input["text_emb"] = core.append_proprio_context_tokens(  # deprecated helper
        text_emb,
        torch.randn(1, 8),
    )
    with_proprio = run_reference_single_stream_forward(
        core,
        input_dict=appended_input,
        update_cache=0,
        cache_name="parity",
        action_mode=False,
        guidance_scale=1.0,
        negative_text_emb=None,
    )

    assert with_proprio.shape == (1, 4, 48)
    assert torch.isfinite(with_proprio).all()


def test_deprecated_text_token_proprio_context_changes_exact_rollout_after_cache_warmup() -> None:
    torch.manual_seed(0)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
        max_text_tokens=4,
    )
    core = SharedVideoTransformerCore(backbone_config, action_dim=4, state_dim=8).eval()
    core.configure_proprio_context_encoder(enabled=True, state_dim=8)
    assert core.proprio_context_encoder is not None
    with torch.no_grad():
        core.proprio_context_encoder.proj.weight.fill_(0.25)
        core.proprio_context_encoder.proj.bias.zero_()

    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        frame_chunk_size=1,
        action_per_frame=1,
        attn_window=2,
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    training_config = TrainingConfig(chunk_size=1, window_size=2)
    inference_config = InferenceConfig(
        frame_chunk_size=1,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )
    observed_video_latents = torch.randn(1, 48, 1, 2, 2)
    observed_action_latents = torch.randn(1, 4, 1, 1, 1)
    text_emb = torch.zeros(1, 4, 8)
    warmup_proprio = torch.full((1, 8), 3.0)

    def run_with_current_proprio(current_proprio: torch.Tensor) -> torch.Tensor:
        warm_cache = run_parallel_exact_cache_warmup(
            transformer=core,
            backbone_config=backbone_config,
            policy_config=policy_config,
            inference_config=inference_config,
            observed_video_latents=observed_video_latents,
            observed_action_latents=observed_action_latents,
            text_emb=text_emb,
            negative_text_emb=None,
            action_channel_mask=None,
            infer_cache={},
            proprio_state=warmup_proprio,
        )
        torch.manual_seed(123)
        rollout = run_parallel_exact_inference_rollout(
            transformer=core,
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            inference_config=inference_config,
            action_dim=4,
            condition_latents=None,
            text_emb=text_emb,
            negative_text_emb=None,
            action_channel_mask=None,
            infer_cache=warm_cache,
            proprio_state=current_proprio,
        )
        return rollout.action_pred.detach().clone()

    positive = run_with_current_proprio(torch.full((1, 8), 10.0))
    negative = run_with_current_proprio(torch.full((1, 8), -10.0))

    # Guards the exact M1 cache boundary: warmup populated self-attn cache with
    # proprio=3, so this only differs if rollout cross-attn sees current proprio.
    assert not torch.allclose(positive, negative)


def test_deprecated_parallel_stream_text_token_proprio_adds_state_to_train_artifacts() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    video_latents = torch.randn(1, 48, 2, 2, 2)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 2, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=2,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=8,
            ),
            chunk=ChunkMetadata(chunk_start_frame=0, chunk_num_frames=2, frame_stride=1, chunk_type="test"),
            conditioning=ConditioningState(
                supported=True,
                text_context=torch.zeros(1, 512, 16),
            ),
        )
    )
    proprio_context_state = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    proprio_context_state_mask = torch.ones_like(proprio_context_state)
    proprio_context_state_mask[:, 1, 3:] = 0
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        state=torch.full((1, 2, 8), -1.0),
        extra={
            "proprio_context_state": proprio_context_state,
            "proprio_context_state_mask": proprio_context_state_mask,
        },
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    artifacts = prepared.variant_inputs["lingbot_train_artifacts"]

    torch.testing.assert_close(
        artifacts.input_dict["proprio_state"],
        proprio_context_state * proprio_context_state_mask,
    )


def test_fastwam_first_frame_per_chunk_proprio_context_uses_first_window_state() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=2, video_num_train_timesteps=10, action_num_train_timesteps=10),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    video_latents = torch.randn(1, 48, 2, 2, 2)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 2, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=2,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=8,
            ),
            chunk=ChunkMetadata(chunk_start_frame=0, chunk_num_frames=2, frame_stride=1, chunk_type="test"),
            conditioning=ConditioningState(
                supported=True,
                text_context=torch.zeros(1, 512, 16),
            ),
        )
    )
    proprio_context_state = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        state=torch.arange(16, dtype=torch.float32).reshape(1, 2, 8),
        extra={"proprio_context_state": proprio_context_state},
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    artifacts = prepared.variant_inputs["lingbot_train_artifacts"]

    torch.testing.assert_close(artifacts.input_dict["per_chunk_proprio_state"], proprio_context_state)
    assert artifacts.input_dict["per_chunk_proprio_state_granularity"] == "chunk"
    assert "proprio_state" not in artifacts.input_dict


def test_exact_runtime_forces_cfg_batch_when_cache_is_shared() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=2.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    observed_video_latents = torch.randn(2, 48, 2, 24, 20)
    observed_action_latents = torch.randn(2, 4, 2, 2, 1)
    text_emb = torch.randn(2, 512, 16)

    warm_cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=observed_video_latents,
        observed_action_latents=observed_action_latents,
        text_emb=text_emb,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
    )

    assert warm_cache["cache_initialized"] is True
    assert warm_cache["use_cfg"] is True
    assert warm_cache["debug_last_warmup"]["cache_write_mode"] == "single_stream_staged"
    assert transformer.cache_batch_sizes[warm_cache["cache_name"]] == 4

    rollout = run_parallel_exact_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=None,
        text_emb=text_emb,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache=warm_cache,
    )

    assert rollout.action_pred.shape == (2, 4, 4)
    assert rollout.predicted_latents.shape == (2, 48, 2, 24, 20)


def test_staged_action_condition_only_zeros_absolute_frame_zero(monkeypatch) -> None:
    captured_action_inputs: list[torch.Tensor] = []

    def fake_single_stream_forward(
        transformer,
        *,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        guidance_scale,
        negative_text_emb,
        combine_cfg=True,
        force_cfg_batch=False,
    ):
        del (
            update_cache,
            cache_name,
            guidance_scale,
            negative_text_emb,
            combine_cfg,
            force_cfg_batch,
        )
        latents = input_dict["noisy_latents"]
        if action_mode:
            captured_action_inputs.append(latents.detach().clone())
            return torch.zeros(
                latents.shape[0],
                latents.shape[2] * latents.shape[3],
                latents.shape[1],
                device=latents.device,
                dtype=latents.dtype,
            )
        patch_t, patch_h, patch_w = transformer.patch_size
        return torch.zeros(
            latents.shape[0],
            (latents.shape[2] // patch_t) * (latents.shape[3] // patch_h) * (latents.shape[4] // patch_w),
            latents.shape[1] * patch_t * patch_h * patch_w,
            device=latents.device,
            dtype=latents.dtype,
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=False,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )

    def run_with_frame_start(frame_start: int) -> torch.Tensor:
        captured_action_inputs.clear()
        torch.manual_seed(123)
        run_parallel_exact_inference_rollout(
            transformer=_FakeReferenceTransformer(),
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            inference_config=inference_config,
            action_dim=4,
            condition_latents=None,
            text_emb=torch.zeros(1, 8, 16),
            negative_text_emb=None,
            action_channel_mask=None,
            infer_cache={
                "batch_size": 1,
                "latent_height": 4,
                "latent_width": 4,
                "frame_start": frame_start,
                "step_index": 0,
            },
        )
        assert captured_action_inputs
        return captured_action_inputs[0]

    absolute_zero_input = run_with_frame_start(0)
    bootstrap_first_chunk_input = run_with_frame_start(1)

    assert torch.count_nonzero(absolute_zero_input[:, :, 0]) == 0
    assert torch.count_nonzero(bootstrap_first_chunk_input[:, :, 0]) > 0


def test_joint_like_first_chunk_anchors_observed_video_frame(monkeypatch) -> None:
    captured_inputs: list[dict[str, torch.Tensor]] = []

    def fake_joint_forward(
        transformer,
        *,
        input_dict,
        video_guidance_scale,
        action_guidance_scale,
        negative_text_emb,
        update_cache=0,
        cache_name="open_wam_exact",
    ):
        del video_guidance_scale, action_guidance_scale, negative_text_emb, update_cache, cache_name
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        captured_inputs.append(
            {
                "video_noisy": latent_dict["noisy_latents"].detach().clone(),
                "video_timesteps": latent_dict["timesteps"].detach().clone(),
            }
        )
        video = latent_dict["noisy_latents"]
        actions = action_dict["noisy_latents"]
        patch_t, patch_h, patch_w = transformer.patch_size
        video_tokens = (video.shape[2] // patch_t) * (video.shape[3] // patch_h) * (video.shape[4] // patch_w)
        action_tokens = actions.shape[2] * actions.shape[3]
        return (
            torch.zeros(
                video.shape[0],
                video_tokens,
                video.shape[1] * patch_t * patch_h * patch_w,
                device=video.device,
                dtype=video.dtype,
            ),
            torch.zeros(
                actions.shape[0],
                action_tokens,
                actions.shape[1],
                device=actions.device,
                dtype=actions.dtype,
            ),
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_joint_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_attention_scope="block_local",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=False,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    condition_latents = torch.randn(1, 48, 2, 4, 4)

    rollout = run_parallel_action_conditioned_inference_rollout(
        transformer=_FakeReferenceTransformer(),
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=condition_latents,
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={"frame_start": 0, "step_index": 0},
    )

    assert captured_inputs
    expected_anchor = condition_latents[:, :, 0].to(dtype=captured_inputs[0]["video_noisy"].dtype).float()
    assert torch.allclose(
        captured_inputs[0]["video_noisy"][:, :, 0].float(),
        expected_anchor,
    )
    assert torch.all(captured_inputs[0]["video_timesteps"][:, 0] == 0)
    assert torch.count_nonzero(captured_inputs[0]["video_timesteps"][:, 1]) > 0
    assert torch.allclose(rollout.predicted_latents[:, :, 0].float(), expected_anchor)
    assert rollout.debug["initial_observed_video_anchor"] is True


def test_staged_cache_write_respects_action_then_video_order(monkeypatch) -> None:
    calls: list[tuple[bool, int, dict[str, int]]] = []
    layer_state = SimpleNamespace(metadata={})

    class _FakeSlotPoolTransformer:
        def _resolve_exact_cache_state(self, cache_name: str):
            assert cache_name == "cache"
            return SimpleNamespace(
                backend_name="slot_pool_exact",
                backend_payload=SimpleNamespace(layer_states=(layer_state,)),
            )

    def fake_single_stream_forward(
        transformer,
        *,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        guidance_scale,
        negative_text_emb,
        combine_cfg=True,
        force_cfg_batch=False,
    ):
        del (
            transformer,
            update_cache,
            cache_name,
            guidance_scale,
            negative_text_emb,
            combine_cfg,
            force_cfg_batch,
        )
        calls.append(
            (
                bool(action_mode),
                int(input_dict["noisy_latents"].shape[2]),
                dict(layer_state.metadata),
            )
        )
        return torch.empty(1, 0, 0)

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    text_emb = torch.zeros(1, 4, 16)
    video_latents = torch.zeros(1, backbone_config.latent_channels, 4, 4, 4)
    action_latents = torch.zeros(1, 4, 4, 2, 1)

    _write_exact_cache_chunk(
        transformer=_FakeSlotPoolTransformer(),
        cache_spec=ExactCacheInterfaceSpec(write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED),
        cache_name="cache",
        frame_start=0,
        backbone_config=backbone_config,
        video_latents=video_latents,
        action_latents=action_latents,
        text_emb=text_emb,
        negative_text_emb=None,
        use_cfg=False,
        action_channel_mask=None,
        update_cache=2,
        chunk_size=2,
        window_size=8,
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        preserve_video_pretrain_history=True,
    )

    assert [(action_mode, frame_count) for action_mode, frame_count, _metadata in calls] == [
        (True, 2),
        (False, 2),
        (True, 2),
        (False, 2),
    ]
    assert calls[1][2][SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS] == 4
    assert calls[3][2][SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS] == 4
    assert SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS not in layer_state.metadata


def test_staged_rollout_applies_hidden_proprio_to_video_and_action(monkeypatch) -> None:
    calls: list[tuple[bool, bool, tuple[int, ...] | None]] = []

    class _HiddenContextFakeTransformer(_FakeReferenceTransformer):
        def encode_proprio_hidden_context(
            self,
            proprio_state: torch.Tensor,
            *,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.ones(
                int(proprio_state.shape[0]),
                int(proprio_state.shape[1]),
                32,
                device=device,
                dtype=dtype,
            )

    def fake_single_stream_forward(
        transformer,
        *,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        guidance_scale,
        negative_text_emb,
        combine_cfg=True,
        force_cfg_batch=False,
    ):
        del transformer, update_cache, cache_name, guidance_scale, negative_text_emb, combine_cfg, force_cfg_batch
        hidden_context = input_dict.get("hidden_context")
        calls.append(
            (
                bool(action_mode),
                hidden_context is not None,
                None if hidden_context is None else tuple(hidden_context.shape),
            )
        )
        latents = input_dict["noisy_latents"]
        batch_size = int(latents.shape[0])
        if action_mode:
            return latents.squeeze(-1).permute(0, 2, 3, 1).reshape(batch_size, -1, latents.shape[1])
        patch_t, patch_h, patch_w = (1, 2, 2)
        return (
            latents.view(
                batch_size,
                latents.shape[1],
                latents.shape[2] // patch_t,
                patch_t,
                latents.shape[3] // patch_h,
                patch_h,
                latents.shape[4] // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch_size, -1, latents.shape[1] * patch_t * patch_h * patch_w)
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
    )

    run_parallel_exact_inference_rollout(
        transformer=_HiddenContextFakeTransformer(),
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=TrainingConfig(chunk_size=2, window_size=8),
        inference_config=InferenceConfig(
            frame_chunk_size=2,
            use_cache=False,
            guidance_scale=1.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=1,
            action_num_inference_steps=1,
        ),
        action_dim=4,
        condition_latents=torch.zeros(1, 48, 2, 4, 4),
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={"frame_start": 0, "step_index": 0},
        hidden_proprio_state=torch.ones(1, 8),
    )

    assert any(not action_mode and has_hidden for action_mode, has_hidden, _shape in calls)
    assert any(action_mode and has_hidden for action_mode, has_hidden, _shape in calls)
    assert (False, True, (1, 8, 32)) in calls
    assert (True, True, (1, 4, 32)) in calls


def test_action_then_video_skip_video_prediction_runs_action_only(monkeypatch) -> None:
    calls: list[tuple[bool, int]] = []

    def fake_single_stream_forward(
        transformer,
        *,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        guidance_scale,
        negative_text_emb,
        combine_cfg=True,
        force_cfg_batch=False,
    ):
        del transformer, cache_name, guidance_scale, negative_text_emb, combine_cfg, force_cfg_batch
        calls.append((bool(action_mode), int(update_cache)))
        latents = input_dict["noisy_latents"]
        batch_size = int(latents.shape[0])
        if action_mode:
            return latents.squeeze(-1).permute(0, 2, 3, 1).reshape(batch_size, -1, latents.shape[1])
        patch_t, patch_h, patch_w = (1, 2, 2)
        return (
            latents.view(
                batch_size,
                latents.shape[1],
                latents.shape[2] // patch_t,
                patch_t,
                latents.shape[3] // patch_h,
                patch_h,
                latents.shape[4] // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch_size, -1, latents.shape[1] * patch_t * patch_h * patch_w)
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    artifacts = run_parallel_exact_inference_rollout(
        transformer=_FakeReferenceTransformer(),
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=TrainingConfig(chunk_size=2, window_size=8),
        inference_config=InferenceConfig(
            frame_chunk_size=2,
            use_cache=True,
            guidance_scale=1.0,
            action_guidance_scale=1.0,
            video_num_inference_steps=1,
            action_num_inference_steps=1,
        ),
        action_dim=4,
        condition_latents=torch.zeros(1, 48, 2, 4, 4),
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={"frame_start": 0, "step_index": 0},
        skip_video_prediction=True,
    )

    assert calls
    assert calls[0] == (False, 2)
    assert all(action_mode for action_mode, _ in calls[1:])
    assert calls[0][1] == 2
    assert all(update_cache == 0 for _, update_cache in calls[1:])
    assert artifacts.predicted_latents.shape[2] == 0
    assert artifacts.debug["generation_frame_start"] == 1
    assert artifacts.debug["initial_observed_context_committed"] is True
    assert artifacts.debug["skip_video_prediction"] is True
    assert artifacts.debug["cache_commit_strategy"] == "action_then_video_action_only_no_predicted_cache"


def test_staged_cache_write_scopes_action_then_video_tail_for_unequal_history(monkeypatch) -> None:
    calls: list[tuple[bool, int, dict[str, int]]] = []
    layer_state = SimpleNamespace(metadata={})

    class _FakeSlotPoolTransformer:
        def _resolve_exact_cache_state(self, cache_name: str):
            assert cache_name == "cache"
            return SimpleNamespace(
                backend_name="slot_pool_exact",
                backend_payload=SimpleNamespace(layer_states=(layer_state,)),
            )

    def fake_single_stream_forward(
        transformer,
        *,
        input_dict,
        update_cache,
        cache_name,
        action_mode,
        guidance_scale,
        negative_text_emb,
        combine_cfg=True,
        force_cfg_batch=False,
    ):
        del (
            transformer,
            update_cache,
            cache_name,
            guidance_scale,
            negative_text_emb,
            combine_cfg,
            force_cfg_batch,
        )
        calls.append(
            (
                bool(action_mode),
                int(input_dict["noisy_latents"].shape[2]),
                dict(layer_state.metadata),
            )
        )
        return torch.empty(1, 0, 0)

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )

    _write_exact_cache_chunk(
        transformer=_FakeSlotPoolTransformer(),
        cache_spec=ExactCacheInterfaceSpec(write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED),
        cache_name="cache",
        frame_start=0,
        backbone_config=backbone_config,
        video_latents=torch.zeros(1, backbone_config.latent_channels, 2, 4, 4),
        action_latents=torch.zeros(1, 4, 4, 2, 1),
        text_emb=torch.zeros(1, 4, 16),
        negative_text_emb=None,
        use_cfg=False,
        action_channel_mask=None,
        update_cache=2,
        chunk_size=2,
        window_size=8,
        current_block_coupling=CurrentBlockCoupling.ACTION_THEN_VIDEO,
        preserve_video_pretrain_history=True,
    )

    assert [(action_mode, frame_count) for action_mode, frame_count, _metadata in calls] == [
        (True, 2),
        (False, 2),
        (True, 2),
    ]
    assert SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS not in calls[0][2]
    assert calls[1][2][SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS] == 4
    assert SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS not in calls[2][2]
    assert SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS not in layer_state.metadata


def test_staged_cache_write_uses_decoupled_clean_cache_path(monkeypatch) -> None:
    single_stream_calls: list[tuple[bool, int]] = []
    clean_cache_calls: list[tuple[int, int, CurrentBlockCoupling]] = []
    clean_cache_hidden_shapes: list[tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = []

    def fake_single_stream_forward(*args, input_dict, action_mode, **kwargs):
        del args, kwargs
        single_stream_calls.append((bool(action_mode), int(input_dict["noisy_latents"].shape[2])))
        return torch.empty(1, 0, 0)

    def fake_joint_clean_cache(**kwargs):
        clean_cache_calls.append(
            (
                int(kwargs["frame_start"]),
                int(kwargs["latents"].shape[2]),
                CurrentBlockCoupling(kwargs["current_block_coupling"]),
            )
        )
        video_hidden_context = kwargs.get("video_hidden_context")
        action_hidden_context = kwargs.get("action_hidden_context")
        clean_cache_hidden_shapes.append(
            (
                None if video_hidden_context is None else tuple(video_hidden_context.shape),
                None if action_hidden_context is None else tuple(action_hidden_context.shape),
            )
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "run_reference_single_stream_forward",
        fake_single_stream_forward,
    )
    monkeypatch.setattr(
        reference_runtime_module,
        "_write_joint_clean_tokens_to_exact_cache",
        fake_joint_clean_cache,
    )

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    text_emb = torch.zeros(1, 4, 16)

    _write_exact_cache_chunk(
        transformer=object(),
        cache_spec=ExactCacheInterfaceSpec(write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED),
        cache_name="cache",
        frame_start=10,
        backbone_config=backbone_config,
        video_latents=torch.zeros(1, backbone_config.latent_channels, 2, 4, 4),
        action_latents=torch.zeros(1, 4, 4, 2, 1),
        text_emb=text_emb,
        negative_text_emb=None,
        use_cfg=False,
        action_channel_mask=None,
        update_cache=2,
        chunk_size=2,
        window_size=8,
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        preserve_video_pretrain_history=True,
        video_hidden_context=torch.ones(1, 8, 32),
        action_hidden_context=torch.ones(1, 4, 32),
    )

    assert single_stream_calls == [(True, 2)]
    assert clean_cache_calls == [
        (10, 2, CurrentBlockCoupling.DECOUPLED_SAME_STEP),
    ]
    assert clean_cache_hidden_shapes == [((1, 8, 32), (1, 4, 32))]


def test_decoupled_clean_cache_cfg_keeps_text_context_separate() -> None:
    class _RecordingBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attention_masks: list[torch.Tensor] = []

        def forward(
            self,
            hidden_states,
            *,
            encoder_hidden_states,
            temb,
            rotary_emb,
            attention_profile=None,
            **kwargs,
        ):
            del encoder_hidden_states, temb, rotary_emb, kwargs
            assert hidden_states.shape[0] == 2
            assert attention_profile is not None
            assert attention_profile.cross_attention_mask is not None
            self.cross_attention_masks.append(attention_profile.cross_attention_mask.detach().cpu())
            return hidden_states, None, None

    class _FakeJointCacheTransformer(nn.Module):
        def __init__(self, block: _RecordingBlock) -> None:
            super().__init__()
            self.patch_size = (1, 1, 1)
            self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.blocks = nn.ModuleList([block])

        def _input_embed(self, tensor: torch.Tensor, input_type: str) -> torch.Tensor:
            del input_type
            token_count = int(tensor.shape[2]) * int(tensor.shape[3]) * int(tensor.shape[4])
            return torch.zeros(
                int(tensor.shape[0]),
                token_count,
                8,
                device=tensor.device,
                dtype=self.weight.dtype,
            )

        def _exact_text_hidden_states(self, text_emb: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
            return text_emb.to(dtype=dtype)

        def rope(self, grid_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                int(grid_ids.shape[0]),
                int(grid_ids.shape[2]),
                1,
                device=grid_ids.device,
                dtype=self.weight.dtype,
            )

        def _time_embed(
            self,
            timesteps: torch.Tensor,
            height: int,
            width: int,
            *,
            dtype: torch.dtype,
            action_mode: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del height, width, action_mode
            token_count = int(timesteps.shape[1])
            projected = torch.zeros(
                int(timesteps.shape[0]),
                token_count,
                6,
                8,
                device=timesteps.device,
                dtype=dtype,
            )
            return projected, projected

        def _resolve_exact_cache_state(self, cache_name: str):
            del cache_name
            return None

    block = _RecordingBlock()
    transformer = _FakeJointCacheTransformer(block)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )

    reference_runtime_module._write_joint_clean_tokens_to_exact_cache(
        transformer=transformer,
        cache_name="cache",
        frame_start=0,
        latents=torch.zeros(1, backbone_config.latent_channels, 1, 1, 1),
        actions=torch.zeros(1, 4, 1, 1, 1),
        text_emb=torch.ones(1, 3, 8),
        negative_text_emb=torch.zeros(1, 3, 8),
        use_cfg=True,
        action_channel_mask=None,
        update_cache=2,
        backbone_config=backbone_config,
        chunk_size=1,
        window_size=4,
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        preserve_video_pretrain_history=True,
    )

    assert len(block.cross_attention_masks) == 1
    expected = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
        ]
    )
    assert torch.equal(block.cross_attention_masks[0], expected)


def test_decoupled_clean_cache_cfg_preserves_slot_pool_batch_rows() -> None:
    torch.manual_seed(0)
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    transformer = SharedVideoTransformerCore(backbone_config, action_dim=4).to(dtype=torch.bfloat16)
    initialize_reference_cache(
        transformer,
        cache_name="cache",
        attn_window=4,
        batch_size=1,
        frame_chunk_size=1,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        action_per_frame=1,
        use_cfg=True,
        cache_backend_name="slot_pool_exact",
        prefix_visibility_mode="preserve_video_pretrain_history",
    )

    _write_exact_cache_chunk(
        transformer=transformer,
        cache_spec=ExactCacheInterfaceSpec(write_mode=ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED),
        cache_name="cache",
        frame_start=0,
        backbone_config=backbone_config,
        video_latents=torch.randn(1, backbone_config.latent_channels, 1, 1, 1, dtype=torch.bfloat16),
        action_latents=torch.randn(1, 4, 1, 1, 1, dtype=torch.bfloat16),
        text_emb=torch.randn(1, 3, 16, dtype=torch.bfloat16),
        negative_text_emb=torch.zeros(1, 3, 16, dtype=torch.bfloat16),
        use_cfg=True,
        action_channel_mask=None,
        update_cache=2,
        chunk_size=1,
        window_size=4,
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        preserve_video_pretrain_history=True,
    )

    cache_state = transformer._resolve_exact_cache_state("cache")
    assert cache_state is not None
    layer_state = cache_state.backend_payload.layer_states[1]
    assert layer_state.key is not None
    assert layer_state.slot_mask is not None
    valid = layer_state.slot_mask.nonzero(as_tuple=False).squeeze(-1)
    key = layer_state.key[:, valid]
    assert key.shape[0] == 2
    assert (key[0] - key[1]).abs().max().item() > 0.0


def test_exact_cache_warmup_preserves_explicit_negative_frame_start_on_init() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    inference_config = InferenceConfig(frame_chunk_size=4, use_cache=True)
    observed_video_latents = torch.randn(1, 48, 4, 8, 8)
    observed_action_latents = torch.randn(1, 4, 4, 4, 1)
    text_emb = torch.randn(1, 512, 16)

    warm_cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=observed_video_latents,
        observed_action_latents=observed_action_latents,
        text_emb=text_emb,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        frame_start_override=-3,
    )

    assert warm_cache["frame_start"] == 1
    assert warm_cache["debug_last_warmup"]["frame_start_override"] == -3
    assert warm_cache["debug_last_warmup"]["frame_start_after"] == 1


def test_parallel_stream_variant_selects_exact_cache_write_contract() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(frame_chunk_size=2)

    canonical = ParallelStreamPolicyVariant(
        ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
        ),
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    action_conditioned = ParallelStreamPolicyVariant(
        ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            video_condition_on_action=True,
        ),
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )

    video_noisy_to_action = ParallelStreamPolicyVariant(
        ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            current_block_coupling=CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            video_condition_on_action=True,
        ),
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    action_noisy_to_video = ParallelStreamPolicyVariant(
        ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            current_block_coupling=CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=8,
            video_condition_on_action=True,
        ),
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )

    assert canonical.exact_cache_write_mode() == ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED
    assert action_conditioned.exact_cache_write_mode() == ParallelExactCacheWriteMode.JOINT_PACKED
    assert video_noisy_to_action.exact_cache_write_mode() == ParallelExactCacheWriteMode.JOINT_PACKED
    assert action_noisy_to_video.exact_cache_write_mode() == ParallelExactCacheWriteMode.JOINT_PACKED


def test_action_conditioned_reference_profile_validates_inference_step_counts() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )

    with pytest.raises(ValueError, match="action_num_inference_steps"):
        ParallelStreamPolicyVariant(
            ParallelStreamPolicyConfig(
                hidden_size=32,
                runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
                current_block_coupling=CurrentBlockCoupling.JOINT,
                reference_profile="libero_joint",
                frame_chunk_size=4,
                action_per_frame=4,
                attn_window=30,
                video_condition_on_action=True,
            ),
            backbone_config=backbone_config,
            training_config=TrainingConfig(chunk_size=4, window_size=30),
            inference_config=InferenceConfig(
                frame_chunk_size=4,
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            ),
            action_dim=30,
            action_horizon=16,
            num_frames=4,
        )


def test_exact_cache_warmup_allows_shorter_video_history_than_action_history() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=4,
        action_per_frame=2,
        attn_window=8,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=4,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    observed_video_latents = torch.randn(1, 48, 2, 24, 20)
    observed_action_latents = torch.randn(1, 4, 4, 2, 1)
    text_emb = torch.randn(1, 512, 16)

    warm_cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=observed_video_latents,
        observed_action_latents=observed_action_latents,
        text_emb=text_emb,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
    )

    assert warm_cache["cache_initialized"] is True
    assert warm_cache["frame_start"] == 2
    assert transformer.cache_attn_windows[warm_cache["cache_name"]] == 8
    assert transformer.cache_layouts[warm_cache["cache_name"]] == (4 * 24 * 20 // 4, 4 * 2)


def test_exact_runtime_uses_provided_negative_text_embeddings_for_cfg() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    observed_video_latents = torch.randn(1, 48, 2, 8, 8)
    observed_action_latents = torch.randn(1, 4, 2, 2, 1)
    text_emb = torch.randn(1, 512, 16)
    negative_text_emb = torch.full_like(text_emb, 3.0)

    warm_cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=observed_video_latents,
        observed_action_latents=observed_action_latents,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=None,
        infer_cache={},
    )

    assert transformer.last_text_emb is not None
    assert torch.equal(transformer.last_text_emb[0], text_emb[0].to(dtype=transformer.last_text_emb.dtype))
    assert torch.equal(transformer.last_text_emb[1], negative_text_emb[0].to(dtype=transformer.last_text_emb.dtype))

    rollout = run_parallel_exact_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=None,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        action_channel_mask=None,
        infer_cache=warm_cache,
    )

    assert rollout.debug["use_cfg"] is True
    assert rollout.action_pred.shape == (1, 4, 4)
    assert rollout.predicted_latents.shape == (1, 48, 2, 8, 8)
    assert transformer.last_text_emb is not None
    assert torch.equal(transformer.last_text_emb[0], text_emb[0].to(dtype=transformer.last_text_emb.dtype))
    assert torch.equal(transformer.last_text_emb[1], negative_text_emb[0].to(dtype=transformer.last_text_emb.dtype))


def test_exact_train_artifacts_default_to_flex_attention_profile() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        train_attn_mode=None,
        infer_attn_mode=None,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    video_latents = torch.randn(1, 48, 2, 8, 8)
    actions = torch.randn(1, 4, 4)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    text_emb = torch.randn(1, 512, 16)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
    )

    assert artifacts.input_dict["attention_profile_name"] == "chunked_temporal_exact"


def test_generalist_action_conditioned_override_drops_text_and_masks_action_loss() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        train_attn_mode=None,
        infer_attn_mode=None,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    video_latents = torch.randn(1, 48, 2, 8, 8)
    actions = torch.randn(1, 4, 4)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    text_emb = torch.randn(1, 512, 16)

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
        generalist_training_mode_override=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        generalist_drop_text_conditioning=True,
        generalist_training_source="counterfactual_dynamics",
    )

    assert artifacts.input_dict["joint_denoise_training_mode"] == "action_conditioned_video"
    assert artifacts.input_dict["joint_denoise_training_mode_override"] == "action_conditioned_video"
    assert artifacts.input_dict["joint_denoise_text_dropped"] is True
    assert artifacts.input_dict["generalist_training_source"] == "counterfactual_dynamics"
    assert torch.equal(artifacts.input_dict["latent_dict"]["text_emb"], torch.zeros_like(text_emb))
    assert torch.equal(artifacts.input_dict["action_dict"]["text_emb"], torch.zeros_like(text_emb))
    assert artifacts.input_dict["action_dict"]["loss_mask"].sum().item() == 0
    assert artifacts.input_dict["latent_dict"]["loss_mask"].sum().item() > 0


def test_exact_runtime_applies_action_channel_mask_to_action_stream() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )
    observed_video_latents = torch.randn(1, 48, 2, 8, 8)
    observed_action_latents = torch.ones(1, 4, 2, 2, 1)
    text_emb = torch.randn(1, 512, 16)
    action_channel_mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1, 1)

    run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=observed_video_latents,
        observed_action_latents=observed_action_latents,
        text_emb=text_emb,
        negative_text_emb=None,
        action_channel_mask=action_channel_mask,
        infer_cache={},
    )

    assert transformer.last_noisy_latents is not None
    expected = observed_action_latents * action_channel_mask
    assert torch.equal(transformer.last_noisy_latents, expected)


def test_reference_single_stream_forward_runs_in_inference_mode() -> None:
    transformer = _GradTrackingTransformer()
    input_dict = {
        "noisy_latents": torch.randn(1, 1, 1, 1, 1),
        "text_emb": torch.zeros(1, 226, 16),
        "grid_id": torch.zeros(1, 4, 1),
        "timesteps": torch.zeros(1, 1),
    }

    output = run_reference_single_stream_forward(
        transformer,
        input_dict=input_dict,
        update_cache=0,
        cache_name="test",
        action_mode=False,
        guidance_scale=1.0,
        negative_text_emb=None,
    )

    assert transformer.grad_enabled_during_forward is False
    assert output.requires_grad is False


def test_parallel_exact_train_artifacts_accept_contextual_overrides() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 8, 8, 16)
    actions = torch.randn(1, 32, 30)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        loss_frame_start=4,
        loss_frame_end=8,
        frame_shift=7,
    )

    assert artifacts.input_dict["chunk_size"] == 2
    assert artifacts.input_dict["window_size"] == 4
    assert artifacts.input_dict["loss_frame_start"] == 4
    assert artifacts.input_dict["loss_frame_end"] == 8
    assert artifacts.input_dict["frame_shift"] == 7
    assert artifacts.input_dict["latent_dict"]["loss_mask"][:, :, :4].sum().item() == 0
    assert torch.all(artifacts.input_dict["latent_dict"]["loss_mask"][:, :, 4:8] == 1)
    assert artifacts.input_dict["action_dict"]["loss_mask"][:, :, :4].sum().item() == 0
    assert torch.all(artifacts.input_dict["action_dict"]["loss_mask"][:, :, 4:8] == 1)
    assert float(artifacts.input_dict["latent_dict"]["grid_id"][0, 0, 0].item()) == 7.0
    assert torch.isclose(
        artifacts.input_dict["action_dict"]["grid_id"][0, 0, 0],
        torch.tensor(7.2),
    )


def test_current_frame_action_chunk_train_artifacts_use_anchor_frame_only() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 8, 8, 16)
    actions = torch.randn(1, 32, 30)
    action_mask = torch.ones_like(actions)
    text_emb = torch.randn(1, 512, 16)

    artifacts = prepare_parallel_current_frame_action_chunk_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
        frame_shift=17,
    )
    input_dict = artifacts.input_dict
    expected_condition = video_latents[:, :, :1].repeat(1, 1, 4, 1, 1)

    assert input_dict["attention_profile_name"] == "none"
    assert input_dict["current_frame_action_chunk"] is True
    assert input_dict["current_frame_condition_source"] == "video_latents"
    assert input_dict["chunk_size"] == 4
    assert input_dict["window_size"] == 4
    assert torch.equal(input_dict["latent_dict"]["noisy_latents"], expected_condition)
    assert torch.equal(input_dict["latent_dict"]["latent"], expected_condition)
    assert torch.all(input_dict["latent_dict"]["timesteps"] == 0)
    assert torch.all(input_dict["latent_dict"]["targets"] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 0)
    assert input_dict["action_dict"]["targets"].shape == (1, 30, 4, 4, 1)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert torch.all(input_dict["action_dict"]["latent"] == 0)


def test_exact_dual_stream_adapter_rejects_invalid_action_context_without_profile() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    artifacts = prepare_parallel_current_frame_action_chunk_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 48, 8, 8, 16),
        actions=torch.randn(1, 32, 30),
        action_mask=torch.ones(1, 32, 30),
        text_emb=torch.randn(1, 512, 16),
        frame_shift=17,
    )
    input_dict = dict(artifacts.input_dict)
    action_dict = dict(input_dict["action_dict"])
    action_mask = action_dict["actions_mask"].clone()
    action_mask[:, :, 0, 0, 0] = 0
    action_dict["actions_mask"] = action_mask
    input_dict["action_dict"] = action_dict

    def _input_embed(tensor: torch.Tensor, input_type: str) -> torch.Tensor:
        del input_type
        return torch.zeros(
            int(tensor.shape[0]),
            int(tensor.shape[2]) * int(tensor.shape[3]) * int(tensor.shape[4]),
            8,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def _text_hidden(text_emb: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(text_emb.shape[0]),
            int(text_emb.shape[1]),
            8,
            dtype=text_emb.dtype,
            device=text_emb.device,
        )

    def _time_embed(
        timesteps: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        action_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del height, width, action_mode
        projected = torch.zeros(
            int(timesteps.shape[0]),
            int(timesteps.shape[1]),
            6,
            8,
            dtype=dtype,
            device=timesteps.device,
        )
        return projected, projected

    def _rope(grid_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(grid_ids.shape[0]),
            int(grid_ids.shape[2]),
            1,
            dtype=torch.float32,
            device=grid_ids.device,
        )

    with pytest.raises(ValueError, match="invalid action tokens"):
        prepare_exact_dual_stream_train_sequence(
            input_dict,
            config=backbone_config,
            patch_size=(
                backbone_config.patch_size_t,
                backbone_config.patch_size_h,
                backbone_config.patch_size_w,
            ),
            model_dtype=torch.float32,
            input_embed=_input_embed,
            exact_text_hidden_states=_text_hidden,
            time_embed=_time_embed,
            rope=_rope,
        )


def test_parallel_exact_train_artifacts_prefer_full_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        noisy_video_condition_prob=0.0,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 2, 8, 16)
    condition_latents = torch.full_like(video_latents, 4.0)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 4, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    assert artifacts.input_dict["video_condition_source"] == "condition_latents"
    torch.testing.assert_close(artifacts.input_dict["latent_dict"]["latent"], condition_latents, rtol=0, atol=0)


def test_parallel_exact_train_artifacts_can_use_single_frame_context_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        noisy_video_condition_prob=0.0,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 3, 8, 16)
    video_latents[:, :, 1:] = 2.0
    condition_latents = torch.full_like(video_latents, 7.0)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 6, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        loss_frame_start=1,
        loss_frame_end=3,
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    assert artifacts.input_dict["video_condition_source"] == "context_condition_latents"
    torch.testing.assert_close(latent_dict["latent"][:, :, :1], condition_latents[:, :, :1], rtol=0, atol=0)
    torch.testing.assert_close(latent_dict["latent"][:, :, 1:], video_latents[:, :, 1:], rtol=0, atol=0)
    assert torch.equal(latent_dict["cond_timesteps"][:, :1], torch.zeros_like(latent_dict["cond_timesteps"][:, :1]))


def test_parallel_exact_train_artifacts_require_single_frame_context_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=4)

    with pytest.raises(ValueError, match="single_frame_condition_latent.*requires `condition_latents`"):
        prepare_parallel_exact_train_artifacts(
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            video_latents=torch.zeros(1, 48, 3, 8, 16),
            condition_latents=None,
            actions=torch.randn(1, 6, 30),
            action_mask=None,
            text_emb=torch.randn(1, 512, 16),
            chunk_size_override=2,
            window_size_override=4,
            loss_frame_start=1,
            loss_frame_end=3,
        )


def test_parallel_prefix_condition_train_artifacts_match_legacy_prefix_semantics() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        noisy_video_condition_prob=1.0,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=4)
    video_latents = torch.randn(1, 3, 4, 2, 2)
    condition_latents = torch.full_like(video_latents, 9.0)
    actions = torch.randn(1, 8, 5)

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        frame_shift=5,
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    assert latent_dict["noisy_latents"].shape[2] == 5
    assert action_dict["noisy_latents"].shape[2] == 4
    torch.testing.assert_close(latent_dict["noisy_latents"][:, :, :1], condition_latents[:, :, :1])
    torch.testing.assert_close(latent_dict["latent"][:, :, :1], condition_latents[:, :, :1])
    assert latent_dict["cond_timesteps"][:, :1].sum().item() == 0
    assert latent_dict["cond_timesteps"][:, 1:].sum().item() > 0
    assert not torch.allclose(latent_dict["latent"][:, :, 1:], video_latents)
    assert latent_dict["loss_mask"][:, :, :1].sum().item() == 0
    assert latent_dict["loss_mask"][:, :, 1:].sum().item() > 0
    assert action_dict["loss_mask"].sum().item() == action_dict["loss_mask"].numel()
    assert artifacts.input_dict["prefix_condition_frames"] == 1
    assert artifacts.input_dict["latent_loss_frame_start"] == 1
    assert artifacts.input_dict["action_loss_frame_start"] == 0
    assert artifacts.input_dict["frame_shift"] == 5


def test_parallel_prefix_condition_train_artifacts_honor_shared_video_schedule() -> None:
    torch.manual_seed(17)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=500,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    video_target_timesteps = input_dict["latent_dict"]["timesteps"][0, 1:]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_target_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_target_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert input_dict["joint_timestep_coupling"] == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE.value
    assert input_dict["coupled_action_video_timesteps"] is True
    assert input_dict["latent_dict"]["timesteps"][0, 0].item() == 0
    torch.testing.assert_close(action_timesteps, video_target_timesteps)
    torch.testing.assert_close(action_sigmas, video_target_sigmas)


def test_parallel_prefix_condition_train_artifacts_honor_match_sigma_coupling() -> None:
    torch.manual_seed(19)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    video_target_timesteps = input_dict["latent_dict"]["timesteps"][0, 1:]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_target_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_target_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert input_dict["joint_timestep_coupling"] == JointTimestepCoupling.MATCH_SIGMA.value
    assert input_dict["coupled_action_video_timesteps"] is True
    assert input_dict["latent_dict"]["timesteps"][0, 0].item() == 0
    torch.testing.assert_close(action_sigmas, video_target_sigmas, atol=2e-3, rtol=0.0)


def test_parallel_prefix_condition_generalist_joint_is_pure_joint_metadata() -> None:
    torch.manual_seed(23)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        video_condition_on_action=True,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        joint_denoise_training_mode_probs={JointDenoiseTrainingMode.JOINT: 1.0},
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    assert input_dict["prefix_condition_frames"] == 1
    assert input_dict["joint_denoise_training_mode"] == JointDenoiseTrainingMode.JOINT.value
    assert input_dict["joint_denoise_training_mode_probs"] == {
        JointDenoiseTrainingMode.JOINT.value: 1.0,
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO.value: 0.0,
        JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION.value: 0.0,
    }
    assert input_dict["video_condition_source"] == "condition_latents_prefix"
    assert input_dict["joint_denoise_shared_sigmas"].shape == (4,)


def test_parallel_prefix_condition_generalist_rejects_conditional_modes() -> None:
    with pytest.raises(ValueError, match="pure `joint`"):
        ParallelStreamPolicyConfig(
            hidden_size=32,
            runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
            current_block_coupling=CurrentBlockCoupling.JOINT,
            frame_chunk_size=2,
            action_per_frame=2,
            attn_window=4,
            parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
            video_condition_on_action=True,
            joint_denoise_training_mode_probs={
                JointDenoiseTrainingMode.JOINT: 0.5,
                JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO: 0.5,
            },
        )


def test_legacy_prefix_variant_preserves_chunk_level_proprio_state() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=4),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=8,
        num_frames=4,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 4, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=4,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=16,
            ),
            chunk=ChunkMetadata(chunk_start_frame=0, chunk_num_frames=4, frame_stride=1, chunk_type="test"),
            conditioning=ConditioningState(
                supported=True,
                text_context=torch.zeros(1, 512, 16),
            ),
        )
    )
    prefix_state = torch.full((1, 8), 9.0)
    chunk_state = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 8, 4),
        state=prefix_state,
        extra={
            "condition_latents": torch.full_like(video_latents, 3.0),
            "proprio_context_state": chunk_state,
            "metadata": ({"sampled_chunk_size": 2, "sampled_window_size": 4},),
        },
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    input_dict = prepared.variant_inputs["lingbot_train_artifacts"].input_dict

    assert input_dict["per_chunk_proprio_state_granularity"] == "chunk"
    assert input_dict["per_chunk_proprio_state"].shape == (1, 3, 8)
    torch.testing.assert_close(input_dict["per_chunk_proprio_state"][:, :1], prefix_state[:, None])
    torch.testing.assert_close(input_dict["per_chunk_proprio_state"][:, 1:], chunk_state)


def test_current_frame_action_chunk_train_artifacts_prefer_explicit_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
        require_condition_latents=True,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 8, 8, 16)
    condition_latents = torch.full((1, 48, 1, 8, 16), 5.0)

    artifacts = prepare_parallel_current_frame_action_chunk_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 32, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    expected_condition = condition_latents.repeat(1, 1, 4, 1, 1)

    assert input_dict["current_frame_condition_source"] == "condition_latents"
    torch.testing.assert_close(input_dict["latent_dict"]["noisy_latents"], expected_condition, rtol=0, atol=0)
    torch.testing.assert_close(input_dict["latent_dict"]["latent"], expected_condition, rtol=0, atol=0)


def test_current_frame_action_chunk_inference_rollout_is_action_only_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_inputs: list[dict] = []

    def fake_action_conditioned_forward(
        transformer: torch.nn.Module,
        *,
        input_dict: dict,
        video_guidance_scale: float,
        action_guidance_scale: float,
        negative_text_emb: torch.Tensor | None,
        update_cache: int = 0,
        cache_name: str = "open_wam_exact",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del transformer, action_guidance_scale, negative_text_emb, cache_name
        captured_inputs.append(input_dict)
        assert video_guidance_scale == 1.0
        assert update_cache == 0
        video_latents = input_dict["latent_dict"]["noisy_latents"]
        action_latents = input_dict["action_dict"]["noisy_latents"]
        video_tokens = int(video_latents.shape[2]) * int(video_latents.shape[3] // 2) * int(video_latents.shape[4] // 2)
        action_tokens = int(action_latents.shape[2]) * int(action_latents.shape[3])
        video_pred = torch.zeros(video_latents.shape[0], video_tokens, 192, device=video_latents.device, dtype=video_latents.dtype)
        action_pred = torch.zeros(
            action_latents.shape[0],
            action_tokens,
            action_latents.shape[1],
            device=action_latents.device,
            dtype=action_latents.dtype,
        )
        return video_pred, action_pred

    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_action_conditioned_forward,
    )

    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=4,
        use_cache=False,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )
    condition_latents = torch.randn(1, 48, 4, 8, 16)

    rollout = run_parallel_current_frame_action_chunk_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=30,
        condition_latents=condition_latents,
        text_emb=torch.randn(1, 512, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={"cache_name": "stale_cache", "cache_initialized": True, "frame_start": 12},
        hidden_proprio_state=torch.arange(8, dtype=torch.float32).reshape(1, 8),
    )

    assert transformer.cleared_pred_cache_names == ["stale_cache"]
    assert rollout.predicted_latents.shape == (1, 48, 0, 8, 16)
    assert rollout.action_pred.shape == (1, 16, 30)
    assert rollout.next_cache["cache_initialized"] is False
    assert rollout.next_cache["frame_start"] == 16
    assert captured_inputs
    expected_condition = condition_latents[:, :, :1].to(dtype=torch.bfloat16).repeat(1, 1, 4, 1, 1)
    torch.testing.assert_close(
        captured_inputs[0]["latent_dict"]["noisy_latents"],
        expected_condition,
        rtol=0,
        atol=0,
    )
    assert captured_inputs[0]["attention_profile_name"] == "none"
    assert captured_inputs[0]["current_frame_action_chunk"] is True
    torch.testing.assert_close(
        captured_inputs[0]["per_chunk_proprio_state"],
        torch.arange(8, dtype=torch.float32).reshape(1, 1, 8).to(dtype=torch.bfloat16),
    )
    assert captured_inputs[0]["per_chunk_proprio_state_granularity"] == "chunk"


def test_current_frame_action_chunk_rejects_video_guidance_scale() -> None:
    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
        frame_chunk_size=4,
        action_per_frame=4,
    )

    with pytest.raises(ValueError, match="does not support video CFG"):
        run_parallel_current_frame_action_chunk_inference_rollout(
            transformer=transformer,
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=TrainingConfig(),
            inference_config=InferenceConfig(frame_chunk_size=4, guidance_scale=5.0),
            action_dim=30,
            condition_latents=torch.randn(1, 48, 4, 8, 16),
            text_emb=torch.randn(1, 512, 16),
            negative_text_emb=None,
            action_channel_mask=None,
            infer_cache={},
        )


def test_fastwam_first_frame_train_artifacts_keep_anchor_clean_and_video_loss_future_only() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 4, 8, 16)
    actions = torch.randn(1, 16, 30)

    artifacts = prepare_parallel_fastwam_first_frame_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict

    assert input_dict["attention_profile_name"] == "fastwam_first_frame"
    assert input_dict["fastwam_first_frame"] is True
    assert input_dict["fastwam_condition_source"] == "video_latents"
    torch.testing.assert_close(
        input_dict["latent_dict"]["noisy_latents"][:, :, :1],
        video_latents[:, :, :1],
        rtol=0,
        atol=0,
    )
    assert torch.all(input_dict["latent_dict"]["targets"][:, :, :1] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"][:, :, :1] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"][:, :, 1:] == 1)
    assert input_dict["action_dict"]["targets"].shape == (1, 30, 4, 4, 1)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert torch.all(input_dict["action_dict"]["latent"] == 0)


def test_fastwam_first_frame_train_artifacts_prefer_explicit_condition_latents() -> None:
    torch.manual_seed(7)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 4, 8, 16)
    condition_latents = torch.full((1, 48, 1, 8, 16), 3.0)

    artifacts = prepare_parallel_fastwam_first_frame_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 16, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict

    assert input_dict["fastwam_condition_source"] == "condition_latents"
    torch.testing.assert_close(
        input_dict["latent_dict"]["noisy_latents"][:, :, :1],
        condition_latents,
        rtol=0,
        atol=0,
    )
    assert torch.all(input_dict["latent_dict"]["loss_mask"][:, :, :1] == 0)


def test_fastwam_first_frame_train_artifacts_can_require_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
        require_condition_latents=True,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )

    with pytest.raises(ValueError, match="require_condition_latents=true"):
        prepare_parallel_fastwam_first_frame_train_artifacts(
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            video_latents=torch.zeros(1, 48, 4, 8, 16),
            actions=torch.randn(1, 16, 30),
            action_mask=None,
            text_emb=torch.randn(1, 512, 16),
        )


def test_fastwam_first_frame_attention_mask_prevents_action_video_round_trip_leak() -> None:
    profile = reference_runtime_module._build_fastwam_first_frame_attention_profile(
        batch_size=1,
        video_seq_len=8,
        action_seq_len=16,
        video_tokens_per_frame=2,
        padded_length=0,
        text_token_count=4,
        device=torch.device("cpu"),
    )
    mask = profile.self_attention_mask
    assert mask is not None

    first_video = slice(0, 2)
    future_video = slice(2, 8)
    action = slice(8, 24)

    assert not mask[first_video, future_video].any()
    assert not mask[:8, action].any()
    assert not mask[action, future_video].any()
    assert mask[action, first_video].all()
    assert mask[action, action].all()
    assert mask[future_video, :8].all()


def test_fastwam_first_frame_forward_keeps_video_prediction_action_invariant() -> None:
    torch.manual_seed(123)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
        max_text_tokens=8,
    )
    transformer = SharedVideoTransformerCore(backbone_config, action_dim=4, state_dim=4).eval()
    transformer.configure_proprio_hidden_context_encoder(enabled=True, state_dim=4)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        frame_chunk_size=4,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 4, 4, 4)
    text_emb = torch.randn(1, 8, 16)
    actions_a = torch.zeros(1, 8, 4)
    actions_b = torch.full((1, 8, 4), 100.0)

    torch.manual_seed(999)
    artifacts_a = prepare_parallel_fastwam_first_frame_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions_a,
        action_mask=None,
        text_emb=text_emb,
    )
    torch.manual_seed(999)
    artifacts_b = prepare_parallel_fastwam_first_frame_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions_b,
        action_mask=None,
        text_emb=text_emb,
    )
    for artifacts in (artifacts_a, artifacts_b):
        artifacts.input_dict["per_chunk_proprio_state"] = torch.ones(1, 1, 4)
        artifacts.input_dict["per_chunk_proprio_state_granularity"] = "chunk"

    with torch.no_grad():
        latent_pred_a, action_pred_a = run_parallel_fastwam_first_frame_train(transformer, artifacts_a.input_dict)
        latent_pred_b, action_pred_b = run_parallel_fastwam_first_frame_train(transformer, artifacts_b.input_dict)

    torch.testing.assert_close(latent_pred_a, latent_pred_b, rtol=0, atol=0)
    assert not torch.equal(action_pred_a, action_pred_b)


def test_parallel_exact_train_artifacts_split_video_and_action_loss_masks() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 8, 8, 16)
    actions = torch.randn(1, 32, 30)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        latent_loss_frame_start=0,
        latent_loss_frame_end=5,
        action_loss_frame_start=0,
        action_loss_frame_end=8,
    )

    assert artifacts.input_dict["loss_frame_start"] == 0
    assert artifacts.input_dict["loss_frame_end"] == 8
    assert artifacts.input_dict["latent_loss_frame_start"] == 0
    assert artifacts.input_dict["latent_loss_frame_end"] == 5
    assert artifacts.input_dict["action_loss_frame_start"] == 0
    assert artifacts.input_dict["action_loss_frame_end"] == 8
    assert torch.all(artifacts.input_dict["latent_dict"]["loss_mask"][:, :, :5] == 1)
    assert artifacts.input_dict["latent_dict"]["loss_mask"][:, :, 5:].sum().item() == 0
    assert torch.all(artifacts.input_dict["action_dict"]["loss_mask"] == 1)


def test_lingbot_parallel_decoder_ignores_history_frames_outside_loss_mask() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 3, 2, 1, 1)
    actions = torch.randn(1, 4, 5)
    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        loss_frame_start=1,
        loss_frame_end=2,
    )

    target_action_pred = artifacts.input_dict["action_dict"]["targets"].squeeze(-1).permute(0, 2, 3, 1).reshape(1, 4, 5)
    corrupted_action_pred = target_action_pred.clone()
    corrupted_action_pred[:, :2] += 100.0

    target_latent_pred = (
        artifacts.input_dict["latent_dict"]["targets"].permute(0, 2, 3, 4, 1).reshape(1, 2, 3)
    )
    corrupted_latent_pred = target_latent_pred.clone()
    corrupted_latent_pred[:, :1] += 100.0

    decoder = LingbotParallelActionDecoder(hidden_size=32, action_dim=5, action_horizon=4)
    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=corrupted_action_pred,
            metrics={},
            aux={
                "latent_pred": corrupted_latent_pred,
                "lingbot_train_artifacts": artifacts,
                "loss_weights": {"latent": 0.0, "action": 1.0},
                "patch_size": (1, 1, 1),
            },
        ),
        PolicyTrainBatch(actions=actions),
    )

    assert torch.isclose(output.loss, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(output.metrics["action_mse"], torch.tensor(0.0), atol=1e-5)


def test_lingbot_parallel_decoder_accepts_prefix_video_action_frame_mismatch() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=4,
        parallel_sequence_contract=ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    condition_latents = torch.full_like(video_latents, 9.0)
    actions = torch.randn(1, 8, 5)
    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    assert artifacts.input_dict["latent_dict"]["timesteps"].shape == (1, 5)
    assert artifacts.input_dict["action_dict"]["timesteps"].shape == (1, 4)
    target_action_pred = (
        artifacts.input_dict["action_dict"]["targets"].squeeze(-1).permute(0, 2, 3, 1).reshape(1, 8, 5)
    )
    target_latent_pred = (
        artifacts.input_dict["latent_dict"]["targets"].permute(0, 2, 3, 4, 1).reshape(1, 20, 3)
    )

    decoder = LingbotParallelActionDecoder(hidden_size=32, action_dim=5, action_horizon=8)
    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=target_action_pred,
            metrics={},
            aux={
                "latent_pred": target_latent_pred,
                "lingbot_train_artifacts": artifacts,
                "loss_weights": {"latent": 1.0, "action": 1.0},
                "patch_size": (1, 1, 1),
            },
        ),
        PolicyTrainBatch(actions=actions),
    )

    assert torch.isfinite(output.loss)
    assert torch.isclose(output.loss, torch.tensor(0.0), atol=1e-5)


def test_parallel_action_conditioned_train_artifacts_accept_contextual_overrides() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        history_stream_visibility="video_only",
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 48, 6, 8, 8),
        actions=torch.randn(1, 24, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=5,
        loss_frame_start=4,
        loss_frame_end=6,
        frame_shift=9,
        chunk_origin_frame=4,
    )

    assert artifacts.input_dict["chunk_size"] == 2
    assert artifacts.input_dict["window_size"] == 5
    assert artifacts.input_dict["loss_frame_start"] == 4
    assert artifacts.input_dict["loss_frame_end"] == 6
    assert artifacts.input_dict["frame_shift"] == 9
    assert artifacts.input_dict["chunk_origin_frame"] == 4
    assert artifacts.input_dict["attention_profile_name"] == "chunked_temporal_exact_joint"


def test_parallel_action_conditioned_inference_uses_policy_attention_geometry(monkeypatch) -> None:
    captured: list[tuple[int, int, str | None]] = []

    def fake_action_conditioned_forward(transformer, *, input_dict, **kwargs):
        del transformer, kwargs
        visibility = input_dict.get("history_stream_visibility")
        captured.append(
            (
                int(input_dict["chunk_size"]),
                int(input_dict["window_size"]),
                None if visibility is None else str(getattr(visibility, "value", visibility)),
            )
        )
        latent_noisy = input_dict["latent_dict"]["noisy_latents"]
        action_noisy = input_dict["action_dict"]["noisy_latents"]
        batch_size = latent_noisy.shape[0]
        video_tokens = (
            latent_noisy.shape[2]
            // 1
            * latent_noisy.shape[3]
            // 2
            * latent_noisy.shape[4]
            // 2
        )
        video_channels = latent_noisy.shape[1] * 1 * 2 * 2
        action_tokens = action_noisy.shape[2] * action_noisy.shape[3]
        return (
            torch.zeros(
                batch_size,
                video_tokens,
                video_channels,
                device=latent_noisy.device,
                dtype=latent_noisy.dtype,
            ),
            torch.zeros(
                batch_size,
                action_tokens,
                action_noisy.shape[1],
                device=action_noisy.device,
                dtype=action_noisy.dtype,
            ),
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_action_conditioned_forward,
    )
    monkeypatch.setattr(reference_runtime_module, "_summarize_slot_pool_cache_state", lambda *_args, **_kwargs: None)


    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=30,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        history_stream_visibility="video_only",
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=4,
        use_cache=False,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=2,
        action_num_inference_steps=2,
    )

    run_parallel_action_conditioned_inference_rollout(
        transformer=_FakeReferenceTransformer(),
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=30,
        condition_latents=torch.randn(1, 48, 4, 8, 8),
        text_emb=torch.randn(1, 512, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
    )

    assert captured
    assert set(captured) == {(4, 30, "video_only")}


def test_action_conditioned_override_after_warmup_uses_local_startup_window(monkeypatch) -> None:
    captured_forwards: list[int] = []
    captured_writes: list[dict[str, object]] = []

    def fake_write_exact_cache_chunk(**kwargs):
        captured_writes.append(dict(kwargs))

    def fake_action_conditioned_forward(transformer, *, input_dict, **kwargs):
        del kwargs
        captured_forwards.append(int(input_dict["window_size"]))
        latent_noisy = input_dict["latent_dict"]["noisy_latents"]
        action_noisy = input_dict["action_dict"]["noisy_latents"]
        batch_size = int(latent_noisy.shape[0])
        video_tokens = (
            int(latent_noisy.shape[2])
            // transformer.patch_size[0]
            * int(latent_noisy.shape[3])
            // transformer.patch_size[1]
            * int(latent_noisy.shape[4])
            // transformer.patch_size[2]
        )
        video_channels = (
            int(latent_noisy.shape[1])
            * transformer.patch_size[0]
            * transformer.patch_size[1]
            * transformer.patch_size[2]
        )
        action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
        return (
            torch.zeros(batch_size, video_tokens, video_channels, device=latent_noisy.device, dtype=latent_noisy.dtype),
            torch.zeros(batch_size, action_tokens, action_noisy.shape[1], device=action_noisy.device, dtype=action_noisy.dtype),
        )

    monkeypatch.setattr(reference_runtime_module, "_write_exact_cache_chunk", fake_write_exact_cache_chunk)
    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_action_conditioned_forward,
    )
    monkeypatch.setattr(reference_runtime_module, "_summarize_slot_pool_cache_state", lambda *_args, **_kwargs: None)

    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=30,
        video_condition_on_action=True,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )

    cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=torch.zeros(1, 48, 1, 4, 4),
        observed_action_latents=torch.zeros(1, 4, 1, 2, 1),
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        cache_write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        action_conditioning_mode="forced_action_joint_fdm",
    )

    output = run_parallel_action_conditioned_action_override_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=torch.zeros(1, 48, 2, 4, 4),
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache=cache,
        advance_frame_start=True,
        forced_action_latents=torch.zeros(1, 4, 2, 2, 1),
        commit_action_latents=torch.zeros(1, 4, 2, 2, 1),
        action_conditioning_mode="forced_action_joint_fdm",
    )

    assert cache["debug_last_warmup"]["rollout_window_size"] == 3
    assert transformer.cache_attn_windows[cache["cache_name"]] == 3
    assert output.debug["rollout_window_size"] == 3
    assert output.debug["forced_clean_action_conditioning"] is True
    assert captured_forwards == [3]
    assert captured_writes
    assert all(int(write["window_size"]) == 3 for write in captured_writes)


def test_conditional_exact_cache_warmup_retains_only_one_history_chunk() -> None:
    torch.manual_seed(0)
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    transformer = SharedVideoTransformerCore(backbone_config, action_dim=2).to(dtype=torch.bfloat16)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=16,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=1,
        attn_window=30,
        video_condition_on_action=True,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )

    warm_cache = run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=torch.randn(1, backbone_config.latent_channels, 6, 1, 1, dtype=torch.bfloat16),
        observed_action_latents=torch.randn(1, 2, 6, 1, 1, dtype=torch.bfloat16),
        text_emb=torch.randn(1, 3, 8, dtype=torch.bfloat16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        cache_write_mode=ParallelExactCacheWriteMode.JOINT_PACKED,
        action_conditioning_mode="forced_action_joint_fdm",
    )

    cache_state = transformer._resolve_exact_cache_state(warm_cache["cache_name"])
    assert cache_state is not None
    assert cache_state.payload["attn_window"] == 3
    layer_state = cache_state.backend_payload.layer_states[0]
    assert layer_state.slot_mask is not None
    assert int(layer_state.slot_mask.sum().item()) == 4


def test_slot_pool_deferred_eviction_keeps_prefix_visible_during_update_attention() -> None:
    layer_state = SlotPoolLayerState(
        slot_ids=torch.arange(4, dtype=torch.long),
        slot_mask=torch.ones(4, dtype=torch.bool),
    )
    valid = torch.arange(4, dtype=torch.long)

    retained_without_defer = _retained_slot_pool_indices_for_current_write(
        layer_state,
        valid=valid,
        current_token_count=4,
        update_mode=1,
    )
    assert retained_without_defer.numel() == 0

    layer_state.metadata[SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION] = True
    retained_with_defer = _retained_slot_pool_indices_for_current_write(
        layer_state,
        valid=valid,
        current_token_count=4,
        update_mode=1,
    )
    assert torch.equal(retained_with_defer, valid)


def test_conditional_clean_cache_commit_attends_previous_chunk_before_evicting(monkeypatch) -> None:
    torch.manual_seed(0)
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    transformer = SharedVideoTransformerCore(backbone_config, action_dim=2).to(dtype=torch.bfloat16)
    initialize_reference_cache(
        transformer,
        cache_name="conditional_commit_cache",
        attn_window=3,
        batch_size=1,
        frame_chunk_size=2,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        action_per_frame=1,
        use_cfg=False,
        cache_backend_name="slot_pool_exact",
        prefix_visibility_mode="full_history",
    )

    observed_prefix_key_lengths: list[int] = []

    def fake_apply_attention_backend(*, query, key, value, attention_mask=None, block_mask=None, kernel_options=None):
        del value, attention_mask, block_mask, kernel_options
        if int(query.shape[2]) == 4 and int(key.shape[2]) > int(query.shape[2]):
            observed_prefix_key_lengths.append(int(key.shape[2]))
        return torch.zeros_like(query)

    monkeypatch.setattr(replica_core_module, "apply_attention_backend", fake_apply_attention_backend)

    text_emb = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    video_chunk = torch.randn(1, backbone_config.latent_channels, 2, 1, 1, dtype=torch.bfloat16)
    action_chunk = torch.randn(1, 2, 2, 1, 1, dtype=torch.bfloat16)
    cache_spec = ExactCacheInterfaceSpec(write_mode=ParallelExactCacheWriteMode.JOINT_PACKED)
    for frame_start, allow_prefix in ((0, False), (2, True)):
        _write_exact_cache_chunk(
            transformer=transformer,
            cache_spec=cache_spec,
            cache_name="conditional_commit_cache",
            frame_start=frame_start,
            backbone_config=backbone_config,
            video_latents=video_chunk,
            action_latents=action_chunk,
            text_emb=text_emb,
            negative_text_emb=None,
            use_cfg=False,
            action_channel_mask=None,
            update_cache=1,
            chunk_size=2,
            window_size=3,
            current_block_coupling=CurrentBlockCoupling.JOINT,
            preserve_video_pretrain_history=False,
            allow_cache_prefix_during_update_write=allow_prefix,
        )

    assert observed_prefix_key_lengths == [8]
    cache_state = transformer._resolve_exact_cache_state("conditional_commit_cache")
    assert cache_state is not None
    layer_state = cache_state.backend_payload.layer_states[0]
    assert layer_state.slot_mask is not None
    assert int(layer_state.slot_mask.sum().item()) == 4
    assert SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION not in layer_state.metadata


def test_conditional_rollout_rejects_reused_full_window_cache() -> None:
    torch.manual_seed(0)
    backbone_config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    transformer = SharedVideoTransformerCore(backbone_config, action_dim=2).to(dtype=torch.bfloat16)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=16,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=1,
        attn_window=30,
        video_condition_on_action=True,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )
    initialize_reference_cache(
        transformer,
        cache_name="stale_full_window_cache",
        attn_window=30,
        batch_size=1,
        frame_chunk_size=2,
        latent_height=1,
        latent_width=1,
        device=torch.device("cpu"),
        action_per_frame=1,
        use_cfg=False,
        cache_backend_name="slot_pool_exact",
        prefix_visibility_mode="full_history",
    )

    with pytest.raises(ValueError, match="existing=30, requested=3"):
        run_parallel_action_conditioned_action_override_inference_rollout(
            transformer=transformer,
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            inference_config=inference_config,
            action_dim=2,
            condition_latents=torch.randn(1, backbone_config.latent_channels, 2, 1, 1, dtype=torch.bfloat16),
            text_emb=torch.randn(1, 3, 8, dtype=torch.bfloat16),
            negative_text_emb=None,
            action_channel_mask=None,
            infer_cache={
                "cache_name": "stale_full_window_cache",
                "cache_initialized": True,
                "batch_size": 1,
                "latent_height": 1,
                "latent_width": 1,
                "use_cfg": False,
            },
            advance_frame_start=True,
            forced_action_latents=torch.randn(1, 2, 2, 1, 1, dtype=torch.bfloat16),
            commit_action_latents=torch.randn(1, 2, 2, 1, 1, dtype=torch.bfloat16),
            action_conditioning_mode="forced_action_joint_fdm",
        )


def test_video_conditioned_action_returns_prediction_but_commits_clean_action_history(monkeypatch) -> None:
    captured_writes: list[dict[str, object]] = []

    def fake_write_exact_cache_chunk(**kwargs):
        captured_writes.append(dict(kwargs))

    def fake_action_conditioned_forward(transformer, *, input_dict, **kwargs):
        del kwargs
        latent_noisy = input_dict["latent_dict"]["noisy_latents"]
        action_noisy = input_dict["action_dict"]["noisy_latents"]
        batch_size = int(latent_noisy.shape[0])
        video_tokens = (
            int(latent_noisy.shape[2])
            // transformer.patch_size[0]
            * int(latent_noisy.shape[3])
            // transformer.patch_size[1]
            * int(latent_noisy.shape[4])
            // transformer.patch_size[2]
        )
        video_channels = (
            int(latent_noisy.shape[1])
            * transformer.patch_size[0]
            * transformer.patch_size[1]
            * transformer.patch_size[2]
        )
        action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
        return (
            torch.zeros(batch_size, video_tokens, video_channels, device=latent_noisy.device, dtype=latent_noisy.dtype),
            torch.zeros(batch_size, action_tokens, action_noisy.shape[1], device=action_noisy.device, dtype=action_noisy.dtype),
        )

    monkeypatch.setattr(reference_runtime_module, "_write_exact_cache_chunk", fake_write_exact_cache_chunk)
    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_action_conditioned_forward,
    )
    monkeypatch.setattr(reference_runtime_module, "_summarize_slot_pool_cache_state", lambda *_args, **_kwargs: None)

    transformer = _FakeReferenceTransformer()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=30,
        video_condition_on_action=True,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )
    clean_commit = torch.full((1, 4, 2, 2, 1), 123.0)

    output = run_parallel_action_conditioned_action_override_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=torch.zeros(1, 48, 2, 4, 4),
        text_emb=torch.zeros(1, 8, 16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        advance_frame_start=True,
        forced_action_latents=None,
        commit_action_latents=clean_commit,
        action_conditioning_mode="video_conditioned_action",
    )

    assert captured_writes
    cached_action_latents = captured_writes[-1]["action_latents"]
    torch.testing.assert_close(cached_action_latents, clean_commit.to(dtype=cached_action_latents.dtype))
    assert output.debug["returned_action_source"] == "predicted"
    assert output.debug["cache_action_source"] == "commit_override"
    assert not torch.allclose(output.action_pred, torch.full_like(output.action_pred, 123.0))


def test_parallel_action_conditioned_train_artifacts_can_force_clean_video_condition() -> None:
    torch.manual_seed(0)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        frame_chunk_size=4,
        action_per_frame=4,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        noisy_video_condition_prob=1.0,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 6, 8, 8)
    actions = torch.randn(1, 24, 30)

    augmented = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    forced_clean = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        force_clean_video_condition=True,
    )

    assert torch.count_nonzero(augmented.input_dict["latent_dict"]["cond_timesteps"]) > 0
    assert torch.count_nonzero(forced_clean.input_dict["latent_dict"]["cond_timesteps"]) == 0
    assert torch.allclose(forced_clean.input_dict["latent_dict"]["latent"], video_latents)
    assert forced_clean.input_dict["force_clean_video_condition"] is True


def test_joint_inference_masks_inactive_action_channels(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_action_conditioned_forward(transformer, *, input_dict, **kwargs):
        del kwargs
        action_noisy = input_dict["action_dict"]["noisy_latents"]
        captured["action_noisy"] = action_noisy.detach().clone()
        captured["actions_mask"] = input_dict["action_dict"]["actions_mask"].detach().clone()
        video_noisy = input_dict["latent_dict"]["noisy_latents"]
        expected_video_tokens = (
            int(video_noisy.shape[2]) // transformer.patch_size[0]
        ) * (
            int(video_noisy.shape[3]) // transformer.patch_size[1]
        ) * (
            int(video_noisy.shape[4]) // transformer.patch_size[2]
        )
        expected_action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
        return (
            torch.zeros(
                video_noisy.shape[0],
                expected_video_tokens,
                video_noisy.shape[1] * transformer.patch_size[0] * transformer.patch_size[1] * transformer.patch_size[2],
                device=video_noisy.device,
                dtype=video_noisy.dtype,
            ),
            torch.ones(
                action_noisy.shape[0],
                expected_action_tokens,
                action_noisy.shape[1],
                device=action_noisy.device,
                dtype=action_noisy.dtype,
            ),
        )

    monkeypatch.setattr(
        reference_runtime_module,
        "_run_parallel_action_conditioned_forward",
        fake_action_conditioned_forward,
    )
    monkeypatch.setattr(reference_runtime_module, "_summarize_slot_pool_cache_state", lambda *_args, **_kwargs: None)

    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=False,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )
    action_channel_mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1, 1)

    rollout = run_parallel_action_conditioned_inference_rollout(
        transformer=_FakeReferenceTransformer(),
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        inference_config=inference_config,
        action_dim=4,
        condition_latents=torch.randn(1, 48, 2, 4, 4),
        text_emb=torch.randn(1, 8, 16),
        negative_text_emb=torch.randn(1, 8, 16),
        action_channel_mask=action_channel_mask,
        infer_cache={},
    )

    assert torch.count_nonzero(captured["action_noisy"][:, [1, 3]]) == 0
    assert torch.count_nonzero(captured["actions_mask"][:, [1, 3]]) == 0
    assert torch.count_nonzero(rollout.action_pred[:, :, [1, 3]]) == 0


def test_standard_joint_training_couples_video_and_action_noise_clarity() -> None:
    torch.manual_seed(11)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    actions = torch.randn(1, 8, 5)

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        input_dict["latent_dict"]["timesteps"][0]
    )
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(
        input_dict["action_dict"]["timesteps"][0]
    )

    assert input_dict["coupled_action_video_timesteps"] is True
    assert torch.allclose(video_sigmas, action_sigmas, atol=2e-3, rtol=0.0)


def test_standard_joint_training_can_share_video_scheduler_clock() -> None:
    torch.manual_seed(13)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=500,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_timesteps = input_dict["latent_dict"]["timesteps"][0]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)
    video_weights = artifacts.latent_scheduler.training_weight(video_timesteps.flatten())
    action_weights = artifacts.action_scheduler.training_weight(action_timesteps.flatten())

    assert input_dict["joint_timestep_coupling"] == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE.value
    assert input_dict["coupled_action_video_timesteps"] is True
    torch.testing.assert_close(action_timesteps, video_timesteps)
    torch.testing.assert_close(action_sigmas, video_sigmas)
    torch.testing.assert_close(action_weights, video_weights)


def test_standard_joint_training_can_match_scheduler_index_without_matching_sigma() -> None:
    torch.manual_seed(12)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_timesteps = input_dict["latent_dict"]["timesteps"][0]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_ids = torch.argmin(
        (artifacts.latent_scheduler.timesteps[:, None] - video_timesteps[None]).abs(),
        dim=0,
    )
    action_ids = torch.argmin(
        (artifacts.action_scheduler.timesteps[:, None] - action_timesteps[None]).abs(),
        dim=0,
    )
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert input_dict["joint_timestep_coupling"] == JointTimestepCoupling.MATCH_INDEX.value
    assert input_dict["coupled_action_video_timesteps"] is False
    assert torch.equal(video_ids, action_ids)
    assert not torch.allclose(video_sigmas, action_sigmas, atol=2e-3, rtol=0.0)


def test_staged_video_then_action_keeps_independent_noise_schedule() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT,
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
    )

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )

    assert artifacts.input_dict["coupled_action_video_timesteps"] is False
    assert artifacts.input_dict["joint_timestep_coupling"] == JointTimestepCoupling.INDEPENDENT.value


def test_shared_video_schedule_inference_uses_video_timestep_directly() -> None:
    video_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    video_scheduler.set_timesteps(20)

    step_index = 1
    shared_sigma = video_scheduler.sigmas[step_index]
    shared_sigma_next = video_scheduler.next_sigma(step_index)
    shared_action_timestep = video_scheduler.timesteps[step_index]
    model_output = torch.ones(1, 1, 1, 1, 1)
    sample = torch.zeros_like(model_output)

    shared_action_step = video_scheduler.step_with_sigmas(
        model_output,
        sigma=shared_sigma,
        sigma_next=shared_sigma_next,
        sample=sample,
    )

    torch.testing.assert_close(video_scheduler.sigma_for_timesteps(shared_action_timestep), shared_sigma)
    torch.testing.assert_close(shared_action_step, model_output * (shared_sigma_next - shared_sigma))


def test_coupled_inference_steps_action_on_shared_video_sigma_schedule() -> None:
    video_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    action_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=500,
    )
    video_scheduler.set_timesteps(20)
    action_scheduler.set_timesteps(20)

    step_index = 1
    shared_sigma = video_scheduler.sigmas[step_index]
    shared_sigma_next = video_scheduler.next_sigma(step_index)
    model_output = torch.ones(1, 1, 1, 1, 1)
    sample = torch.zeros_like(model_output)

    action_lookup_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=500,
    )
    action_lookup_scheduler.set_timesteps(500)

    coupled_action_timestep = action_lookup_scheduler.timestep_matching_sigma(shared_sigma)
    coupled_action_step = action_scheduler.step_with_sigmas(
        model_output,
        sigma=shared_sigma,
        sigma_next=shared_sigma_next,
        sample=sample,
    )
    independent_action_step = action_scheduler.step(
        model_output,
        action_scheduler.timesteps[step_index],
        sample,
    )

    assert torch.allclose(
        action_lookup_scheduler.sigma_for_timesteps(coupled_action_timestep),
        shared_sigma,
        atol=2e-3,
        rtol=0.0,
    )
    assert not torch.allclose(coupled_action_timestep, video_scheduler.timesteps[step_index])
    assert torch.allclose(coupled_action_step, model_output * (shared_sigma_next - shared_sigma))
    assert not torch.allclose(coupled_action_step, independent_action_step)


def _generalist_policy_config(
    mode: JointDenoiseTrainingMode,
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    generalist_mode_text_token: bool = False,
) -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode=ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        variant_profile=ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        attn_window=8,
        video_condition_on_action=True,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=joint_timestep_coupling,
        joint_denoise_training_mode_probs={mode: 1.0},
        generalist_mode_text_token=generalist_mode_text_token,
    )


def _small_generalist_artifacts(
    mode: JointDenoiseTrainingMode,
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    drop_text_conditioning: bool | None = None,
):
    torch.manual_seed(7)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = _generalist_policy_config(mode, joint_timestep_coupling=joint_timestep_coupling)
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=20,
        action_num_train_timesteps=20,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    actions = torch.randn(1, 8, 5)
    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        generalist_drop_text_conditioning=drop_text_conditioning,
    )
    action_latents = actions.reshape(1, 4, 2, 5).permute(0, 3, 1, 2).unsqueeze(-1)
    return artifacts, video_latents, action_latents


def test_parallel_variant_appends_generalist_mode_token_before_deprecated_text_token_proprio() -> None:
    artifacts, _, _ = _small_generalist_artifacts(JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO)
    original_text = artifacts.input_dict["latent_dict"]["text_emb"]
    policy_config = replace(
        _generalist_policy_config(
            JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
            generalist_mode_text_token=True,
        ),
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
            patch_size_t=1,
            patch_size_h=1,
            patch_size_w=1,
        ),
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=5,
        action_horizon=8,
        num_frames=4,
    )
    core = SharedVideoTransformerCore(
        variant.backbone_config,
        action_dim=5,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)
    core.configure_proprio_context_encoder(enabled=True, state_dim=8)

    mode_count = variant._append_generalist_mode_text_token(core, artifacts)
    base_text_token_count = int(artifacts.input_dict["latent_dict"]["text_emb"].shape[1])
    appended_text = core.append_proprio_context_tokens(  # deprecated helper
        artifacts.input_dict["latent_dict"]["text_emb"],
        torch.randn(1, 8),
    )

    assert mode_count == 1
    assert base_text_token_count == int(original_text.shape[1]) + 1
    assert appended_text.shape[1] == int(original_text.shape[1]) + 2
    assert artifacts.input_dict["generalist_mode_text_token"] == "action_conditioned_video"
    assert artifacts.input_dict["generalist_mode_text_token_count"] == 1
    assert torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"],
        artifacts.input_dict["action_dict"]["text_emb"],
    )
    assert torch.equal(artifacts.input_dict["latent_dict"]["text_emb"][:, :-1], original_text)
    assert not torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"][:, -1:],
        torch.zeros_like(artifacts.input_dict["latent_dict"]["text_emb"][:, -1:]),
    )


def test_generalist_joint_denoising_action_conditioned_video_uses_clean_action_slot() -> None:
    artifacts, video_latents, action_latents = _small_generalist_artifacts(
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
    )
    input_dict = artifacts.input_dict

    assert input_dict["variant_profile"] == "generalist_joint_denoising"
    assert input_dict["joint_denoise_training_mode"] == "action_conditioned_video"
    assert torch.equal(input_dict["action_dict"]["noisy_latents"], action_latents)
    assert torch.all(input_dict["action_dict"]["timesteps"] == 0)
    assert torch.all(input_dict["action_dict"]["targets"] == 0)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 1)
    assert torch.equal(input_dict["latent_dict"]["latent"], video_latents)
    assert torch.equal(input_dict["action_dict"]["latent"], action_latents)
    assert input_dict["window_size"] == 3
    assert input_dict["generalist_conditional_history_chunks"] == 1
    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    latent_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(input_dict["latent_dict"]["timesteps"][0])
    assert torch.allclose(latent_sigmas, shared_sigmas, atol=2e-3, rtol=0.0)


def test_generalist_joint_denoising_video_conditioned_action_uses_clean_video_slot() -> None:
    artifacts, video_latents, _ = _small_generalist_artifacts(
        JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
    )
    input_dict = artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "video_conditioned_action"
    assert torch.equal(input_dict["latent_dict"]["noisy_latents"], video_latents)
    assert torch.all(input_dict["latent_dict"]["timesteps"] == 0)
    assert torch.all(input_dict["latent_dict"]["targets"] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 0)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert torch.equal(input_dict["latent_dict"]["latent"], video_latents)
    assert torch.count_nonzero(input_dict["action_dict"]["latent"]) > 0
    assert input_dict["window_size"] == 3
    assert input_dict["generalist_conditional_history_chunks"] == 1
    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    expected_action_timesteps = artifacts.action_scheduler.timestep_matching_sigma(shared_sigmas)
    assert torch.equal(input_dict["action_dict"]["timesteps"][0], expected_action_timesteps)


def test_generalist_joint_denoising_joint_mode_matches_standard_m1_joint_artifacts() -> None:
    artifacts, video_latents, action_latents = _small_generalist_artifacts(JointDenoiseTrainingMode.JOINT)
    input_dict = artifacts.input_dict

    torch.manual_seed(7)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    standard_policy = replace(
        _generalist_policy_config(JointDenoiseTrainingMode.JOINT),
        variant_profile=ParallelStreamVariantProfile.STANDARD,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=20,
        action_num_train_timesteps=20,
    )
    standard_video_latents = torch.randn(1, 3, 4, 2, 2)
    standard_actions = torch.randn(1, 8, 5)
    standard_artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=standard_policy,
        training_config=training_config,
        video_latents=standard_video_latents,
        actions=standard_actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    standard_input = standard_artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "joint"
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 1)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert not torch.equal(input_dict["latent_dict"]["noisy_latents"], video_latents)
    assert not torch.equal(input_dict["action_dict"]["noisy_latents"], action_latents)
    assert torch.equal(video_latents, standard_video_latents)
    assert torch.equal(input_dict["latent_dict"]["noisy_latents"], standard_input["latent_dict"]["noisy_latents"])
    assert torch.equal(input_dict["latent_dict"]["latent"], standard_input["latent_dict"]["latent"])
    assert torch.equal(input_dict["latent_dict"]["targets"], standard_input["latent_dict"]["targets"])
    assert torch.equal(input_dict["latent_dict"]["timesteps"], standard_input["latent_dict"]["timesteps"])
    assert torch.equal(input_dict["action_dict"]["noisy_latents"], standard_input["action_dict"]["noisy_latents"])
    assert torch.equal(input_dict["action_dict"]["latent"], standard_input["action_dict"]["latent"])
    assert torch.equal(input_dict["action_dict"]["targets"], standard_input["action_dict"]["targets"])
    assert torch.equal(input_dict["action_dict"]["timesteps"], standard_input["action_dict"]["timesteps"])
    assert "generalist_conditional_history_chunks" not in input_dict

    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    assert shared_sigmas.shape == (4,)
    assert torch.all(shared_sigmas >= 0)
    assert torch.all(shared_sigmas <= 1)
    latent_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(input_dict["latent_dict"]["timesteps"][0])
    expected_action_timesteps = artifacts.action_scheduler.timestep_matching_sigma(shared_sigmas)
    assert torch.allclose(latent_sigmas, shared_sigmas, atol=2e-3, rtol=0.0)
    assert torch.equal(input_dict["action_dict"]["timesteps"][0], expected_action_timesteps)


def test_generalist_joint_denoising_conditional_modes_drop_text_by_default() -> None:
    joint_artifacts, _, _ = _small_generalist_artifacts(JointDenoiseTrainingMode.JOINT)
    assert joint_artifacts.input_dict["joint_denoise_text_dropped"] is False
    assert torch.count_nonzero(joint_artifacts.input_dict["latent_dict"]["text_emb"]) > 0

    for mode in (
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
    ):
        artifacts, _, _ = _small_generalist_artifacts(mode)
        input_dict = artifacts.input_dict

        assert input_dict["joint_denoise_text_dropped"] is True
        assert torch.equal(
            input_dict["latent_dict"]["text_emb"],
            torch.zeros_like(input_dict["latent_dict"]["text_emb"]),
        )
        assert torch.equal(
            input_dict["action_dict"]["text_emb"],
            torch.zeros_like(input_dict["action_dict"]["text_emb"]),
        )


def test_generalist_joint_denoising_conditional_modes_drop_text_even_with_false_override() -> None:
    artifacts, _, _ = _small_generalist_artifacts(
        JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        drop_text_conditioning=False,
    )

    assert artifacts.input_dict["joint_denoise_text_dropped"] is True
    assert torch.count_nonzero(artifacts.input_dict["latent_dict"]["text_emb"]) == 0
    assert torch.count_nonzero(artifacts.input_dict["action_dict"]["text_emb"]) == 0


def test_lingbot_parallel_decoder_logs_generalist_mode_sums_and_counts() -> None:
    artifacts, _, _ = _small_generalist_artifacts(JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO)
    action_targets = artifacts.input_dict["action_dict"]["targets"].squeeze(-1).permute(0, 2, 3, 1).reshape(1, 8, 5)
    latent_targets = artifacts.input_dict["latent_dict"]["targets"].permute(0, 2, 3, 4, 1).reshape(1, 16, 3)
    decoder = LingbotParallelActionDecoder(hidden_size=32, action_dim=5, action_horizon=8)

    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=action_targets,
            metrics={},
            aux={
                "latent_pred": latent_targets,
                "lingbot_train_artifacts": artifacts,
                "loss_weights": {"latent": 1.0, "action": 1.0},
                "patch_size": (1, 1, 1),
            },
        ),
        PolicyTrainBatch(actions=torch.zeros(1, 8, 5)),
    )

    assert output.metrics["joint_denoise/action_conditioned_video/count"].item() == 1.0
    assert output.metrics["joint_denoise/joint/count"].item() == 0.0
    assert "joint_denoise/action_conditioned_video/action_flow_loss_sum" in output.metrics
    assert "joint_denoise/action_conditioned_video/action_mse_sum" in output.metrics
    assert torch.equal(
        output.metrics["joint_denoise/action_conditioned_video/action_mse_sum"],
        output.metrics["joint_denoise/action_conditioned_video/action_flow_loss_sum"],
    )
    assert output.metrics["joint_denoise/action_loss_active"].item() == 0.0
    assert output.metrics["joint_denoise/latent_loss_active"].item() == 1.0


def test_exact_cache_warmup_passes_per_chunk_hidden_context(monkeypatch) -> None:
    captured: dict[str, tuple[int, ...] | None] = {}

    def fake_write_exact_cache_chunk(**kwargs):
        video_hidden_context = kwargs.get("video_hidden_context")
        action_hidden_context = kwargs.get("action_hidden_context")
        captured["video"] = None if video_hidden_context is None else tuple(video_hidden_context.shape)
        captured["action"] = None if action_hidden_context is None else tuple(action_hidden_context.shape)

    monkeypatch.setattr(reference_runtime_module, "_write_exact_cache_chunk", fake_write_exact_cache_chunk)

    transformer = _FakeReferenceTransformer()

    def encode_proprio_hidden_context(frame_state, *, device, dtype):
        return torch.ones(frame_state.shape[0], frame_state.shape[1], 32, device=device, dtype=dtype)

    transformer.encode_proprio_hidden_context = encode_proprio_hidden_context  # type: ignore[attr-defined]
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact",
        current_block_coupling=CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        frame_chunk_size=2,
        action_per_frame=3,
        attn_window=4,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )

    run_parallel_exact_cache_warmup(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        inference_config=inference_config,
        observed_video_latents=torch.zeros(1, 48, 2, 4, 4, dtype=torch.bfloat16),
        observed_action_latents=torch.zeros(1, 4, 2, 3, 1, dtype=torch.bfloat16),
        text_emb=torch.zeros(1, 4, 16, dtype=torch.bfloat16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        hidden_proprio_state=torch.ones(1, 8, dtype=torch.bfloat16),
    )

    assert captured == {"video": (1, 8, 32), "action": (1, 6, 32)}


def test_action_conditioned_rollout_cache_commit_passes_per_chunk_hidden_context(monkeypatch) -> None:
    write_calls: list[dict[str, tuple[int, ...] | None]] = []

    def fake_write_exact_cache_chunk(**kwargs):
        video_hidden_context = kwargs.get("video_hidden_context")
        action_hidden_context = kwargs.get("action_hidden_context")
        write_calls.append(
            {
                "video": None if video_hidden_context is None else tuple(video_hidden_context.shape),
                "action": None if action_hidden_context is None else tuple(action_hidden_context.shape),
            }
        )

    def fake_action_conditioned_forward(transformer, *, input_dict, video_guidance_scale, action_guidance_scale, negative_text_emb, update_cache, cache_name):
        del transformer, video_guidance_scale, action_guidance_scale, negative_text_emb, update_cache, cache_name
        latents = input_dict["latent_dict"]["noisy_latents"]
        actions = input_dict["action_dict"]["noisy_latents"]
        batch_size, channels, frames, height, width = latents.shape
        _, action_dim, action_frames, action_per_frame, action_width = actions.shape
        video_tokens = frames * (height // 2) * (width // 2)
        action_tokens = action_frames * action_per_frame * action_width
        return (
            torch.zeros(batch_size, video_tokens, channels * 4, device=latents.device, dtype=latents.dtype),
            torch.zeros(batch_size, action_tokens, action_dim, device=actions.device, dtype=actions.dtype),
        )

    monkeypatch.setattr(reference_runtime_module, "_write_exact_cache_chunk", fake_write_exact_cache_chunk)
    monkeypatch.setattr(reference_runtime_module, "_run_parallel_action_conditioned_forward", fake_action_conditioned_forward)

    transformer = _FakeReferenceTransformer()

    def encode_proprio_hidden_context(frame_state, *, device, dtype):
        return torch.ones(frame_state.shape[0], frame_state.shape[1], 32, device=device, dtype=dtype)

    transformer.encode_proprio_hidden_context = encode_proprio_hidden_context  # type: ignore[attr-defined]
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=3,
        attn_window=4,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        video_condition_on_action=True,
    )
    inference_config = InferenceConfig(
        frame_chunk_size=2,
        use_cache=True,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=1,
        action_num_inference_steps=1,
    )

    run_parallel_action_conditioned_inference_rollout(
        transformer=transformer,
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=TrainingConfig(chunk_size=2, window_size=4),
        inference_config=inference_config,
        action_dim=4,
        condition_latents=torch.zeros(1, 48, 2, 4, 4, dtype=torch.bfloat16),
        text_emb=torch.zeros(1, 4, 16, dtype=torch.bfloat16),
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        hidden_proprio_state=torch.ones(1, 8, dtype=torch.bfloat16),
    )

    assert write_calls[0] == {"video": (1, 4, 32), "action": None}
    assert write_calls[-1] == {"video": (1, 8, 32), "action": (1, 6, 32)}



def test_per_chunk_proprio_context_applies_to_clean_video_branch() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        def encode_proprio_hidden_context(self, frame_state, *, device, dtype):
            return torch.ones(frame_state.shape[0], frame_state.shape[1], 4, device=device, dtype=dtype)

    hidden_states = torch.zeros(1, 8, 4)
    output = reference_runtime_module._apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=hidden_states,
        split_list=[2, 2, 2, 2],
        input_dict={
            "chunk_size": 2,
            "latent_dict": {"noisy_latents": torch.zeros(1, 1, 2, 1, 1)},
            "action_dict": {"noisy_latents": torch.zeros(1, 1, 2, 1, 1)},
            "per_chunk_proprio_state": torch.zeros(1, 1, 3),
        },
    )

    torch.testing.assert_close(output[:, 0:2, :], torch.ones(1, 2, 4))
    torch.testing.assert_close(output[:, 2:4, :], torch.ones(1, 2, 4))
    torch.testing.assert_close(output[:, 4:6, :], torch.ones(1, 2, 4))
    torch.testing.assert_close(output[:, 6:8, :], torch.ones(1, 2, 4))


def test_per_chunk_proprio_context_applies_to_pre_target_prefix() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        def encode_proprio_hidden_context(self, frame_state, *, device, dtype):
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    hidden_states = torch.zeros(1, 16, 4)
    output = reference_runtime_module._apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=hidden_states,
        split_list=[4, 4, 4, 4],
        input_dict={
            "chunk_size": 2,
            "chunk_origin_frame": 1,
            "per_chunk_proprio_state_granularity": "frame",
            "latent_dict": {"noisy_latents": torch.zeros(1, 1, 4, 1, 1)},
            "action_dict": {"noisy_latents": torch.zeros(1, 1, 4, 1, 1)},
            "per_chunk_proprio_state": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        },
    )

    expected_context = torch.tensor(
        [[[[1.0] * 4, [1.0] * 4, [1.0] * 4, [3.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    torch.testing.assert_close(output[:, 0:4, :], expected_context)
    torch.testing.assert_close(output[:, 4:8, :], expected_context)
    torch.testing.assert_close(output[:, 8:12, :], expected_context)
    torch.testing.assert_close(output[:, 12:16, :], expected_context)


def test_legacy_prefix_per_chunk_proprio_context_skips_video_branch() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        def encode_proprio_hidden_context(self, frame_state, *, device, dtype):
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    hidden_states = torch.zeros(1, 18, 4)
    output = reference_runtime_module._apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=hidden_states,
        split_list=[5, 5, 4, 4],
        input_dict={
            "chunk_size": 2,
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
            "per_chunk_proprio_state_granularity": "frame",
            "latent_dict": {"noisy_latents": torch.zeros(1, 1, 5, 1, 1)},
            "action_dict": {"noisy_latents": torch.zeros(1, 1, 4, 1, 1)},
            "per_chunk_proprio_state": torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]]),
        },
    )

    expected_action_context = torch.tensor(
        [[[[1.0] * 4, [1.0] * 4, [3.0] * 4, [3.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    torch.testing.assert_close(output[:, 0:5, :], torch.zeros(1, 5, 4))
    torch.testing.assert_close(output[:, 5:10, :], torch.zeros(1, 5, 4))
    torch.testing.assert_close(output[:, 10:14, :], expected_action_context)
    torch.testing.assert_close(output[:, 14:18, :], expected_action_context)


def test_legacy_prefix_per_chunk_proprio_context_accepts_chunk_level_state() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        def encode_proprio_hidden_context(self, frame_state, *, device, dtype):
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    hidden_states = torch.zeros(1, 18, 4)
    output = reference_runtime_module._apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=hidden_states,
        split_list=[5, 5, 4, 4],
        input_dict={
            "chunk_size": 2,
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
            "per_chunk_proprio_state_granularity": "chunk",
            "latent_dict": {"noisy_latents": torch.zeros(1, 1, 5, 1, 1)},
            "action_dict": {"noisy_latents": torch.zeros(1, 1, 4, 1, 1)},
            "per_chunk_proprio_state": torch.tensor([[[1.0], [2.0], [4.0]]]),
        },
    )

    expected_action_context = torch.tensor(
        [[[[2.0] * 4, [2.0] * 4, [4.0] * 4, [4.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 4, 4)
    torch.testing.assert_close(output[:, 0:5, :], torch.zeros(1, 5, 4))
    torch.testing.assert_close(output[:, 5:10, :], torch.zeros(1, 5, 4))
    torch.testing.assert_close(output[:, 10:14, :], expected_action_context)
    torch.testing.assert_close(output[:, 14:18, :], expected_action_context)


def test_per_chunk_proprio_context_treats_state_as_chunk_level_at_chunk_size_one() -> None:
    class _ContextTransformer:
        patch_size = (1, 1, 1)

        def encode_proprio_hidden_context(self, frame_state, *, device, dtype):
            return frame_state.to(device=device, dtype=dtype).expand(-1, -1, 4)

    hidden_states = torch.zeros(1, 12, 4)
    output = reference_runtime_module._apply_parallel_chunk_proprio_context(
        _ContextTransformer(),
        hidden_states=hidden_states,
        split_list=[3, 3, 3, 3],
        input_dict={
            "chunk_size": 1,
            "latent_dict": {"noisy_latents": torch.zeros(1, 1, 3, 1, 1)},
            "action_dict": {"noisy_latents": torch.zeros(1, 1, 3, 1, 1)},
            "per_chunk_proprio_state": torch.tensor([[[10.0], [20.0], [30.0]]]),
        },
    )

    expected_context = torch.tensor(
        [[[[10.0] * 4, [20.0] * 4, [30.0] * 4]]],
        dtype=output.dtype,
    ).reshape(1, 3, 4)
    torch.testing.assert_close(output[:, 0:3, :], expected_context)
    torch.testing.assert_close(output[:, 3:6, :], expected_context)
    torch.testing.assert_close(output[:, 6:9, :], expected_context)
    torch.testing.assert_close(output[:, 9:12, :], expected_context)


def test_action_override_rollout_routes_per_chunk_proprio_state_to_hidden_context(monkeypatch) -> None:
    captured: dict[str, torch.Tensor | None] = {}

    def fake_impl(**kwargs):
        captured["proprio_state"] = kwargs.get("proprio_state")
        captured["hidden_proprio_state"] = kwargs.get("hidden_proprio_state")
        action_dim = int(kwargs["action_dim"])
        return reference_runtime_module.LingbotParallelInferArtifacts(
            action_pred=torch.empty(1, 0, action_dim),
            predicted_latents=torch.empty(1, 48, 0, 4, 4),
            next_cache={},
            debug={},
        )

    monkeypatch.setattr(reference_runtime_module, "_run_parallel_action_conditioned_inference_rollout_impl", fake_impl)
    proprio_state = torch.ones(1, 8)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=3,
        attn_window=4,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        video_condition_on_action=True,
    )

    reference_runtime_module.run_parallel_action_conditioned_action_override_inference_rollout(
        transformer=object(),
        backbone_config=LingbotCompatibleVideoBackboneConfig(hidden_size=32),
        policy_config=policy_config,
        training_config=TrainingConfig(chunk_size=2, window_size=4),
        inference_config=InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        condition_latents=None,
        text_emb=None,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        advance_frame_start=True,
        proprio_state=proprio_state,
    )

    assert captured["proprio_state"] is None
    assert captured["hidden_proprio_state"] is proprio_state


def test_action_override_rollout_selects_anchor_from_3d_per_chunk_proprio_state(monkeypatch) -> None:
    captured: dict[str, torch.Tensor | None] = {}

    def fake_impl(**kwargs):
        captured["hidden_proprio_state"] = kwargs.get("hidden_proprio_state")
        action_dim = int(kwargs["action_dim"])
        return reference_runtime_module.LingbotParallelInferArtifacts(
            action_pred=torch.empty(1, 0, action_dim),
            predicted_latents=torch.empty(1, 48, 0, 4, 4),
            next_cache={},
            debug={},
        )

    monkeypatch.setattr(reference_runtime_module, "_run_parallel_action_conditioned_inference_rollout_impl", fake_impl)
    proprio_state = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        runtime_mode="lingbot_exact_action_conditioned",
        current_block_coupling=CurrentBlockCoupling.JOINT,
        frame_chunk_size=2,
        action_per_frame=3,
        attn_window=4,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        video_condition_on_action=True,
    )

    reference_runtime_module.run_parallel_action_conditioned_action_override_inference_rollout(
        transformer=object(),
        backbone_config=LingbotCompatibleVideoBackboneConfig(hidden_size=32),
        policy_config=policy_config,
        training_config=TrainingConfig(chunk_size=2, window_size=4),
        inference_config=InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        condition_latents=None,
        text_emb=None,
        negative_text_emb=None,
        action_channel_mask=None,
        infer_cache={},
        advance_frame_start=True,
        proprio_state=proprio_state,
    )

    assert isinstance(captured["hidden_proprio_state"], torch.Tensor)
    torch.testing.assert_close(captured["hidden_proprio_state"], proprio_state[:, -1, :])
