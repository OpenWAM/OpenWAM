from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, ParallelStreamPolicyConfig, TrainingConfig
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
)
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig


class _FakeReferenceTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (1, 2, 2)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.cache_batch_sizes: dict[str, int] = {}

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
        del attn_window, latent_token_per_chunk, action_token_per_chunk, device, dtype
        self.cache_batch_sizes[cache_name] = batch_size

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        batch_size = input_dict["noisy_latents"].shape[0]
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
        infer_cache=warm_cache,
    )

    assert rollout.action_pred.shape == (2, 4, 4)
    assert rollout.predicted_latents.shape == (2, 48, 2, 24, 20)
