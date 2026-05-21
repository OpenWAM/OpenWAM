from __future__ import annotations

import torch

from open_wam.configs import CausalVideoPredictionPolicyConfig, InferenceConfig, TrainingConfig
from open_wam.models.video_backbone.contracts import ChunkMetadata, ConditioningState, TokenGridMetadata
from open_wam.models.policy_variants.causal_video_prediction import CausalVideoPredictionPolicyVariant
from open_wam.models.visual_tower import VisualFrontendOutput, VisualStageOutputs


def test_causal_video_prediction_maps_raw_wan_windows_to_latent_layouts() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )

    layouts = variant._resolve_layouts(
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 6,
                "valid_video_frames": 8,
                "padded_video_frames": 16,
            },
        ),
        available_frames=5,
        frame_mapping={"kind": "wan_temporal_downsample", "raw_frames": 16, "latent_frames": 4},
    )

    assert len(layouts) == 1
    assert layouts[0].observed_frames == 1
    assert layouts[0].future_frames == 1
    assert layouts[0].total_frames == 2


def test_causal_video_prediction_keeps_latent_layouts_in_identity_mapping() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )

    layouts = variant._resolve_layouts(
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 6,
                "valid_video_frames": 8,
            },
        ),
        available_frames=8,
        frame_mapping={"kind": "identity", "raw_frames": 8, "latent_frames": 8},
    )

    assert len(layouts) == 1
    assert layouts[0].observed_frames == 2
    assert layouts[0].future_frames == 6
    assert layouts[0].total_frames == 8


class _CaptureVideoFlowTower:
    def __init__(self) -> None:
        self.attention_mask: torch.Tensor | None = None

    def predict_video_flow(self, **kwargs):
        self.attention_mask = kwargs.get("attention_mask")
        return torch.zeros_like(kwargs["noisy_latents"])


def test_causal_video_prediction_masks_padded_tokens_during_train_rollout() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(video_num_train_timesteps=8),
        inference_config=InferenceConfig(),
    )
    video_latents = torch.randn(2, 48, 6, 2, 2)
    token_grid = TokenGridMetadata(
        num_frames=6,
        latent_height=2,
        latent_width=2,
        patch_size=(1, 2, 2),
        patches_per_frame_h=1,
        patches_per_frame_w=1,
        tokens_per_frame=1,
        sequence_length=6,
    )
    frontend = VisualFrontendOutput(
        canonical_video=torch.zeros(2, 3, 6, 32, 32),
        video_latents=video_latents,
        video_tokens=torch.zeros(2, 6, 4),
        input_source="video_latents",
        token_grid=token_grid,
        chunk=ChunkMetadata(
            chunk_start_frame=0,
            chunk_num_frames=6,
            frame_stride=1,
            chunk_type="dense_video_chunk",
        ),
        conditioning=ConditioningState(
            supported=True,
            text_context=torch.zeros(2, 4, 16),
            metadata={},
        ),
    )
    tower = _CaptureVideoFlowTower()

    rollout = variant._build_train_rollout(
        visual_tower=tower,  # type: ignore[arg-type]
        visual_outputs=VisualStageOutputs(frontend=frontend),
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 2,
                "valid_video_frames": 4,
                "padded_video_frames": 6,
            },
            {
                "observed_prefix_frames": 1,
                "future_suffix_frames": 5,
                "valid_video_frames": 6,
            },
        ),
    )

    assert tower.attention_mask is not None
    assert tower.attention_mask.shape == (2, 6, 6)
    assert torch.all(tower.attention_mask[0, :, :4])
    assert not torch.any(tower.attention_mask[0, :, 4:])
    assert torch.all(tower.attention_mask[1])
    future_loss_mask = rollout["future_loss_mask"]
    assert torch.all(future_loss_mask[0, :, 2:4])
    assert not torch.any(future_loss_mask[0, :, 4:])
