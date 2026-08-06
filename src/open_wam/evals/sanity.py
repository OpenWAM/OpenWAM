from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from open_wam.configs import (
    BatchAdapterName,
    DataConfig,
    ExperimentConfig,
    load_experiment_config,
    resolve_experiment_config_reference,
    serialize_experiment_config,
)
from open_wam.data import (
    LatentWAMBatch,
    LatentWAMSample,
    WAMBatch,
    WAMSample,
    build_canonical_video_preprocessor,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    move_latent_wam_batch_to_device,
    move_wam_batch_to_device,
    validate_action_mapping_preflight,
)
from open_wam.extensions import load_extension_modules
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import (
    VariantPipeline,
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
)
from open_wam.runtime import build_result_envelope
from open_wam.runtime.provenance import collect_runtime_provenance
from open_wam.runtime.results import write_result_json
from open_wam.utils import seed_everywhere
from open_wam.utils.libero_paradigm import require_current_libero_policy_paradigm


def run_sanity_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one parsed load/train/eval/rollout-style sanity command."""

    if args.max_batches != 1:
        raise SystemExit(
            "open-wam-sanity inspects exactly one batch, so --max-batches must be 1. "
            "Use open-wam-eval for multi-batch metrics."
        )
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive when provided.")
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive.")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("--require-gpu was set, but CUDA is not available.")

    seed_everywhere(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested CUDA device {device}, but CUDA is not available.")

    load_extension_modules(args.extension)
    config_path = resolve_experiment_config_reference(args.config).resolve()
    config = load_experiment_config(config_path)
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source="open-wam-sanity",
        allow_deprecated=bool(args.allow_deprecated_libero_config),
    )
    mapping_report = validate_action_mapping_preflight(
        config.data.action_mapping,
        action_schema_dim=config.data.action_schema.action_dim,
    )
    pipeline = build_variant_pipeline_from_config(config).to(device)
    pipeline.eval()

    uses_latents = _uses_latent_batches(config)
    train_dataset, val_dataset = _build_datasets(config.data, latent=uses_latents)
    dataset = train_dataset if args.split == "train" else val_dataset
    collate_fn = (
        collate_latent_wam_samples if uses_latents else collate_wam_samples
    )
    default_batch_size = (
        config.data.train_batch_size
        if args.split == "train"
        else config.data.val_batch_size
    )
    batch_size = args.batch_size or default_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    batch = _move_batch(batch, device)
    load_report = _build_load_report(batch, config)
    train_report = _run_train_forward(pipeline, batch)
    infer_report = _run_batch_infer(pipeline, batch)
    rollout_report = _run_rollout_style_infer(
        pipeline,
        dataset,
        device=device,
        steps=min(args.rollout_steps, len(dataset)),
        latent=uses_latents,
    )

    legacy_summary = {
        "config": str(config_path),
        "dataset_type": config.data.dataset_type,
        "dataset_name": config.data.dataset_name,
        "split": args.split,
        "device": str(device),
        "mapping": mapping_report,
        "load": load_report,
        "train_forward": train_report,
        "batch_infer": infer_report,
        "rollout_style_infer": rollout_report,
    }
    summary = build_result_envelope(
        command="open-wam-sanity",
        config=str(config_path),
        metrics={
            "train_loss": train_report["loss"],
            "rollout_steps": rollout_report["steps"],
        },
        checkpoint=None,
        benchmark=str(config.data.dataset_name),
        device=str(device),
        seed=int(args.seed),
        provenance=collect_runtime_provenance(
            config_path=config_path,
            resolved_config=serialize_experiment_config(config),
            dataset_root=config.data.local_root,
            mode=args.provenance_mode,
        ),
        extra=legacy_summary,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        write_result_json(args.output_json, summary)
    return summary


def _uses_latent_batches(config: ExperimentConfig) -> bool:
    return config.trainer.batch_adapter == BatchAdapterName.LATENTS


def _build_datasets(
    data_config: DataConfig,
    *,
    latent: bool,
) -> tuple[
    Dataset[WAMSample] | Dataset[LatentWAMSample],
    Dataset[WAMSample] | Dataset[LatentWAMSample],
]:
    if latent:
        return build_train_val_latent_datasets(data_config)
    return build_train_val_datasets(data_config)


def _move_batch(batch: WAMBatch | LatentWAMBatch, device: torch.device) -> WAMBatch | LatentWAMBatch:
    if isinstance(batch, LatentWAMBatch):
        return move_latent_wam_batch_to_device(batch, device)
    return move_wam_batch_to_device(batch, device)


def _policy_train_batch(batch: WAMBatch | LatentWAMBatch) -> PolicyTrainBatch:
    return PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={
            "task_text": batch.task_text,
            "metadata": batch.metadata,
            "state_mask": batch.state_mask,
        },
    )


def _policy_infer_context(batch: WAMBatch | LatentWAMBatch) -> PolicyInferContext:
    return PolicyInferContext(
        state=batch.state,
        extra={
            "task_text": batch.task_text,
            "metadata": batch.metadata,
        },
    )


def _build_load_report(
    batch: WAMBatch | LatentWAMBatch,
    config: ExperimentConfig,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "actions_shape": list(batch.actions.shape),
        "action_mask_sum": None if batch.action_mask is None else float(batch.action_mask.float().sum().item()),
        "state_shape": None if batch.state is None else list(batch.state.shape),
        "task_text_count": 0 if batch.task_text is None else len(batch.task_text),
    }
    if isinstance(batch, LatentWAMBatch):
        report["video_latents_shape"] = list(batch.video_latents.shape)
        expected_latent_shape = [
            batch.video_latents.shape[0],
            48,
            config.data.num_frames,
            config.data.canonical_height // config.backbone.latent_stride,
            config.data.canonical_width // config.backbone.latent_stride,
        ]
        report["expected_video_latents_shape"] = expected_latent_shape
        _assert_shape("video_latents", batch.video_latents.shape, expected_latent_shape)
        return report

    canonical = build_canonical_video_preprocessor(config.data)(batch.views)
    expected_video_shape = [
        batch.actions.shape[0],
        3,
        config.data.num_frames,
        config.data.canonical_height,
        config.data.canonical_width,
    ]
    report["view_shapes"] = {name: list(value.shape) for name, value in batch.views.items()}
    report["canonical_video_shape"] = list(canonical.video.shape)
    report["expected_canonical_video_shape"] = expected_video_shape
    _assert_shape("canonical_video", canonical.video.shape, expected_video_shape)
    expected_latent_shape = [
        batch.actions.shape[0],
        48,
        config.data.num_frames,
        config.data.canonical_height // config.backbone.latent_stride,
        config.data.canonical_width // config.backbone.latent_stride,
    ]
    report["expected_video_latents_shape"] = expected_latent_shape
    return report


def _run_train_forward(
    pipeline: VariantPipeline,
    batch: WAMBatch | LatentWAMBatch,
) -> dict[str, Any]:
    with torch.no_grad():
        start = time.perf_counter()
        if isinstance(batch, LatentWAMBatch):
            output = pipeline.forward_train_from_latents(
                batch.video_latents,
                _policy_train_batch(batch),
                canonical_video=batch.canonical_video,
                text_context=batch.text_context,
                negative_text_context=batch.negative_text_context,
            )
        else:
            output = pipeline.forward_train(batch.views, _policy_train_batch(batch))
        elapsed = time.perf_counter() - start
    return {
        "loss": float(output.decoder_output.loss.detach().float().item()),
        "metrics": {
            key: float(value.detach().float().item())
            for key, value in output.decoder_output.metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
        },
        "elapsed_s": elapsed,
    }


def _run_batch_infer(
    pipeline: VariantPipeline,
    batch: WAMBatch | LatentWAMBatch,
) -> dict[str, Any]:
    with torch.no_grad():
        start = time.perf_counter()
        if isinstance(batch, LatentWAMBatch):
            output = pipeline.forward_infer_step_from_latents(
                batch.video_latents,
                _policy_infer_context(batch),
                canonical_video=batch.canonical_video,
                text_context=batch.text_context,
                negative_text_context=batch.negative_text_context,
            )
        else:
            output = pipeline.forward_infer_step(batch.views, _policy_infer_context(batch))
        elapsed = time.perf_counter() - start
    action_pred = output.decoder_output.action_pred
    mse = None
    if action_pred.shape == batch.actions.shape:
        mse = _masked_mse(action_pred, batch.actions, batch.action_mask)
    return {
        "action_pred_shape": list(action_pred.shape),
        "target_action_shape": list(batch.actions.shape),
        "masked_action_mse": mse,
        "elapsed_s": elapsed,
        "steps_per_second": 1.0 / elapsed if elapsed > 0 else None,
    }


def _run_rollout_style_infer(
    pipeline: VariantPipeline,
    dataset: Dataset[WAMSample] | Dataset[LatentWAMSample],
    *,
    device: torch.device,
    steps: int,
    latent: bool,
) -> dict[str, Any]:
    runner = VariantRolloutRunner(pipeline)
    session = None
    previous_action = None
    elapsed_values: list[float] = []
    action_shapes: list[list[int]] = []
    with torch.no_grad():
        for index in range(steps):
            sample = dataset[index]
            if latent:
                batch = move_latent_wam_batch_to_device(
                    collate_latent_wam_samples([sample]),
                    device,
                )
            else:
                batch = move_wam_batch_to_device(collate_wam_samples([sample]), device)
            if session is None:
                session = runner.reset(
                    task_text=batch.task_text,
                    text_context=batch.text_context if isinstance(batch, LatentWAMBatch) else None,
                    negative_text_context=batch.negative_text_context if isinstance(batch, LatentWAMBatch) else None,
                )
            context = PolicyInferContext(
                state=batch.state,
                previous_action=previous_action,
                extra={
                    "task_text": batch.task_text,
                    "metadata": batch.metadata,
                },
            )
            start = time.perf_counter()
            if isinstance(batch, LatentWAMBatch):
                step_output = runner.infer_step(
                    session=session,
                    context=context,
                    video_latents=batch.video_latents,
                    canonical_video=batch.canonical_video,
                )
            else:
                step_output = runner.infer_step(
                    session=session,
                    context=context,
                    views=batch.views,
                )
            elapsed_values.append(time.perf_counter() - start)
            session = step_output.session
            action_pred = step_output.infer_output.decoder_output.action_pred.detach()
            previous_action = action_pred
            action_shapes.append(list(action_pred.shape))
    total_elapsed = sum(elapsed_values)
    return {
        "steps": steps,
        "total_elapsed_s": total_elapsed,
        "mean_step_s": total_elapsed / max(steps, 1),
        "mean_step_hz": steps / total_elapsed if total_elapsed > 0 else None,
        "action_pred_shapes": action_shapes,
    }


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    squared_error = (prediction.float() - target.float()).pow(2)
    if mask is not None:
        squared_error = squared_error * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(squared_error.numel()), device=squared_error.device)
    return float((squared_error.sum() / denom).item())


def _assert_shape(name: str, actual: torch.Size | tuple[int, ...], expected: list[int]) -> None:
    actual_list = list(actual)
    if actual_list != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {actual_list}.")
