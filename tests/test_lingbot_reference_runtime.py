from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import (
    InferenceConfig,
    CurrentBlockCoupling,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    TrainingConfig,
)
from open_wam.models.action_decoders.lingbot_parallel_decoder import LingbotParallelActionDecoder
from open_wam.models.policy_variants.contracts import PolicyTrainBatch, PolicyTrainOutput
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_reference_single_stream_forward,
)
from open_wam.models.policy_variants.parallel_stream.variant import ParallelStreamPolicyVariant
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig


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
