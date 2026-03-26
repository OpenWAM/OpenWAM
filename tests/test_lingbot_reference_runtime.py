from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, ParallelStreamPolicyConfig, TrainingConfig
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_reference_single_stream_forward,
)
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
    ) -> None:
        del attn_window, device, dtype
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

    assert transformer.last_text_emb is not None
    assert torch.equal(transformer.last_text_emb[0], text_emb[0].to(dtype=transformer.last_text_emb.dtype))
    assert torch.equal(transformer.last_text_emb[1], negative_text_emb[0].to(dtype=transformer.last_text_emb.dtype))
    assert rollout.action_pred.shape == (1, 4, 4)


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
