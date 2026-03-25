from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from open_wam.data import build_synthetic_batch
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_train_batch(config_path: Path):
    config = load_experiment_config(config_path)
    config = replace(config, backbone=replace(config.backbone, implementation="lingbot_replica"))
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={"task_text": batch.task_text},
    )
    return config, pipeline, batch, train_batch


def test_post_latent_and_post_decoded_share_video_conditioner_path() -> None:
    for config_name in ("post_latent_robotwin.yaml", "post_decoded_robotwin.yaml"):
        _, pipeline, batch, train_batch = _build_train_batch(REPO_ROOT / "configs/experiments" / config_name)
        train_output = pipeline.forward_train(batch.views, train_batch)
        infer_output = pipeline.forward_infer_step(
            batch.views,
            PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
        )

        assert train_output.visual_outputs.core is not None
        assert train_output.visual_outputs.core.aux["weight_source"] == "local_init"
        assert train_output.visual_outputs.core.aux["used_action_conditioner"] is False
        assert infer_output.visual_outputs.core is not None
        assert infer_output.visual_outputs.core.aux["used_action_conditioner"] is False


def test_register_and_parallel_variants_use_action_conditioner_path() -> None:
    for config_name in ("register_attached_robotwin.yaml", "parallel_stream_robotwin.yaml"):
        _, pipeline, batch, train_batch = _build_train_batch(REPO_ROOT / "configs/experiments" / config_name)
        train_output = pipeline.forward_train(batch.views, train_batch)
        infer_output = pipeline.forward_infer_step(
            batch.views,
            PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
        )

        assert train_output.policy_output.aux["core_aux"]["weight_source"] == "local_init"
        assert train_output.policy_output.aux["core_aux"]["used_action_conditioner"] is True
        assert infer_output.policy_output.aux["core_aux"]["used_action_conditioner"] is True
