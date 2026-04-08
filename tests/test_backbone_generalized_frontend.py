from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import torch

from open_wam.data import build_synthetic_batch
from open_wam.data.raw_video import ViewPlacement
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.video_backbone import LingbotCompatibleVideoBackbone
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata
from open_wam.models.visual_tower.contracts import VisualFrontendOutput
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backbone_encode_video_forwards_multiview_placements() -> None:
    backbone = LingbotCompatibleVideoBackbone(LingbotCompatibleVideoBackboneConfig())
    canonical_video = torch.zeros(1, 3, 2, 128, 256)
    placements = (
        ViewPlacement(source_name="image", canonical_name="image", top=0, left=0, height=128, width=128),
        ViewPlacement(source_name="wrist_image", canonical_name="wrist_image", top=0, left=128, height=128, width=128),
    )
    recorded: dict[str, object] = {}
    sentinel = torch.full((1, 48, 2, 8, 16), 3.0)

    def fake_encode_video(self, video, *, placements=None, reset_reference_cache=True):
        recorded["video_shape"] = tuple(video.shape)
        recorded["placements"] = placements
        recorded["reset_reference_cache"] = reset_reference_cache
        return sentinel

    backbone.tower.frontend.encode_video = MethodType(fake_encode_video, backbone.tower.frontend)  # type: ignore[method-assign]

    encoded = backbone.encode_video(canonical_video, placements=placements, reset_reference_cache=False)

    assert torch.equal(encoded, sentinel)
    assert recorded["video_shape"] == tuple(canonical_video.shape)
    assert recorded["placements"] == placements
    assert recorded["reset_reference_cache"] is False


def test_backbone_forward_preserves_frontend_context_inputs() -> None:
    backbone = LingbotCompatibleVideoBackbone(LingbotCompatibleVideoBackboneConfig(hidden_size=32))
    canonical_video = torch.zeros(1, 3, 2, 128, 256)
    placements = (
        ViewPlacement(source_name="image", canonical_name="image", top=0, left=0, height=128, width=128),
        ViewPlacement(source_name="wrist_image", canonical_name="wrist_image", top=0, left=128, height=128, width=128),
    )
    task_text = ("stack the cups",)
    text_context = torch.randn(1, 8, 32)
    negative_text_context = torch.randn(1, 8, 32)
    recorded: dict[str, object] = {}
    frontend_output = VisualFrontendOutput(
        canonical_video=canonical_video,
        video_latents=torch.randn(1, 48, 2, 8, 16),
        video_tokens=torch.randn(1, 16, 32),
        input_source="canonical_rgb",
        token_grid=TokenGridMetadata(
            num_frames=2,
            latent_height=8,
            latent_width=16,
            patch_size=(1, 2, 2),
            patches_per_frame_h=4,
            patches_per_frame_w=8,
            tokens_per_frame=32,
            sequence_length=16,
        ),
        chunk=ChunkMetadata(chunk_start_frame=0, chunk_num_frames=2, frame_stride=1, chunk_type="dense_video_chunk"),
        conditioning=ConditioningState(
            supported=True,
            text_context=text_context,
            negative_text_context=negative_text_context,
            first_frame_context=torch.randn(1, 48, 1, 8, 16),
        ),
    )

    def fake_run_frontend(
        self,
        video,
        *,
        placements=None,
        task_text=None,
        text_context=None,
        negative_text_context=None,
        preserve_stream_cache=False,
    ):
        recorded["video_shape"] = tuple(video.shape)
        recorded["placements"] = placements
        recorded["task_text"] = task_text
        recorded["text_context"] = text_context
        recorded["negative_text_context"] = negative_text_context
        recorded["preserve_stream_cache"] = preserve_stream_cache
        return frontend_output

    backbone.tower.run_frontend = MethodType(fake_run_frontend, backbone.tower)  # type: ignore[method-assign]
    backbone.tower.run_default_core = MethodType(
        lambda self, frontend_output: SimpleNamespace(
            tokens=frontend_output.video_tokens,
            cache_state=CacheState(supported=False, current_start_frame=0, cached_frames=0, chunk_size=0),
        ),
        backbone.tower,
    )  # type: ignore[method-assign]

    output = backbone(
        canonical_video,
        placements=placements,
        task_text=task_text,
        text_context=text_context,
        negative_text_context=negative_text_context,
        preserve_stream_cache=True,
    )

    assert recorded["video_shape"] == tuple(canonical_video.shape)
    assert recorded["placements"] == placements
    assert recorded["task_text"] == task_text
    assert recorded["text_context"] is text_context
    assert recorded["negative_text_context"] is negative_text_context
    assert recorded["preserve_stream_cache"] is True
    assert torch.equal(output.video_latents, frontend_output.video_latents)
    assert output.conditioning.text_context is text_context


def test_all_policy_variants_consume_shared_frontend_video_latents() -> None:
    config_names = (
        "post_latent_robotwin.yaml",
        "post_decoded_robotwin.yaml",
        "register_attached_robotwin_smoke.yaml",
        "parallel_stream_robotwin_smoke.yaml",
    )

    for config_name in config_names:
        config = load_experiment_config(REPO_ROOT / "configs/experiments" / config_name)
        if config.backbone.implementation != "lingbot_replica":
            config = replace(config, backbone=replace(config.backbone, implementation="lingbot_replica"))
        pipeline = build_variant_pipeline_from_config(config)
        batch = build_synthetic_batch(config.data, batch_size=2)
        train_batch = PolicyTrainBatch(
            actions=batch.actions,
            action_mask=batch.action_mask,
            state=batch.state,
            extra={"task_text": batch.task_text},
        )
        frontend = pipeline.visual_tower.frontend
        original_encode_video = frontend.encode_video
        call_count = 0

        def wrapped_encode_video(self, canonical_video, *, placements=None, reset_reference_cache=True):
            nonlocal call_count
            call_count += 1
            latents = original_encode_video(
                canonical_video,
                placements=placements,
                reset_reference_cache=reset_reference_cache,
            )
            return latents + 7.0

        frontend.encode_video = MethodType(wrapped_encode_video, frontend)  # type: ignore[method-assign]
        canonical_batch = pipeline.canonicalize(batch.views)
        expected_latents = original_encode_video(
            canonical_batch.video,
            placements=canonical_batch.placements,
            reset_reference_cache=True,
        ) + 7.0

        train_output = pipeline.forward_train(batch.views, train_batch)
        infer_output = pipeline.forward_infer_step(
            batch.views,
            PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
        )

        assert torch.equal(train_output.visual_outputs.frontend.video_latents, expected_latents), config_name
        assert torch.equal(infer_output.visual_outputs.frontend.video_latents, expected_latents), config_name
        assert call_count == 2, config_name
