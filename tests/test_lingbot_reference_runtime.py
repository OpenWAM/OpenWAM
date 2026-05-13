from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from open_wam.configs import (
    InferenceConfig,
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    TrainingConfig,
)
from open_wam.models.action_decoders.lingbot_parallel_decoder import LingbotParallelActionDecoder
from open_wam.models.common import SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS
from open_wam.models.common.flow_matching import FlowMatchScheduler as SharedFlowMatchScheduler
from open_wam.models.policy_variants.contracts import PolicyTrainBatch, PolicyTrainOutput
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    ExactCacheInterfaceSpec,
    FlowMatchScheduler,
    _write_exact_cache_chunk,
    initialize_reference_cache,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    run_parallel_action_conditioned_inference_rollout,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_reference_single_stream_forward,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime as reference_runtime_module
from open_wam.models.policy_variants.parallel_stream.variant import ParallelStreamPolicyVariant
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig, SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def test_m1_reference_runtime_uses_shared_flow_match_scheduler() -> None:
    assert FlowMatchScheduler is SharedFlowMatchScheduler


class _FakeReferenceTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (1, 2, 2)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.cache_batch_sizes: dict[str, int] = {}
        self.cache_layouts: dict[str, tuple[int, int]] = {}
        self.last_text_emb: torch.Tensor | None = None
        self.last_noisy_latents: torch.Tensor | None = None

    def clear_cache(self, cache_name: str) -> None:
        self.cache_batch_sizes.pop(cache_name, None)

    def clear_pred_cache(self, cache_name: str) -> None:
        del cache_name

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
        del attn_window, device, dtype, backend_name, prefix_visibility_mode
        self.cache_batch_sizes[cache_name] = batch_size
        self.cache_layouts[cache_name] = (latent_token_per_chunk, action_token_per_chunk)

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
        couple_action_to_video_timesteps=True,
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
    )

    assert single_stream_calls == [(True, 2)]
    assert clean_cache_calls == [
        (10, 2, CurrentBlockCoupling.DECOUPLED_SAME_STEP),
    ]


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
    )

    assert artifacts.input_dict["chunk_size"] == 2
    assert artifacts.input_dict["window_size"] == 5
    assert artifacts.input_dict["loss_frame_start"] == 4
    assert artifacts.input_dict["loss_frame_end"] == 6
    assert artifacts.input_dict["frame_shift"] == 9
    assert artifacts.input_dict["attention_profile_name"] == "chunked_temporal_exact_joint"


def test_parallel_action_conditioned_inference_uses_policy_attention_geometry(monkeypatch) -> None:
    captured: list[tuple[int, int]] = []

    def fake_action_conditioned_forward(transformer, *, input_dict, **kwargs):
        del transformer, kwargs
        captured.append((int(input_dict["chunk_size"]), int(input_dict["window_size"])))
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
    assert set(captured) == {(4, 30)}


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
        couple_action_to_video_timesteps=True,
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
        couple_action_to_video_timesteps=True,
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
        couple_action_to_video_timesteps=True,
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
    couple_action_to_video_timesteps: bool = True,
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
        couple_action_to_video_timesteps=couple_action_to_video_timesteps,
        joint_denoise_training_mode_probs={mode: 1.0},
    )


def _small_generalist_artifacts(mode: JointDenoiseTrainingMode):
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
    policy_config = _generalist_policy_config(mode)
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
    )
    action_latents = actions.reshape(1, 4, 2, 5).permute(0, 3, 1, 2).unsqueeze(-1)
    return artifacts, video_latents, action_latents


def test_generalist_joint_denoising_action_conditioned_video_uses_clean_action_slot() -> None:
    artifacts, _, action_latents = _small_generalist_artifacts(
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
    assert torch.all(input_dict["latent_dict"]["latent"] == 0)
    assert torch.all(input_dict["action_dict"]["latent"] == 0)


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
    assert torch.all(input_dict["latent_dict"]["latent"] == 0)
    assert torch.all(input_dict["action_dict"]["latent"] == 0)


def test_generalist_joint_denoising_joint_mode_couples_noise_clarity() -> None:
    artifacts, video_latents, action_latents = _small_generalist_artifacts(JointDenoiseTrainingMode.JOINT)
    input_dict = artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "joint"
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 1)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert not torch.equal(input_dict["latent_dict"]["noisy_latents"], video_latents)
    assert not torch.equal(input_dict["action_dict"]["noisy_latents"], action_latents)

    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    assert shared_sigmas.shape == (4,)
    assert torch.all(shared_sigmas >= 0)
    assert torch.all(shared_sigmas <= 1)


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
