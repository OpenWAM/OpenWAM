from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (  # noqa: E402
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
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run quantified load/train/eval/rollout-style sanity checks for one Open-WAM benchmark config."
        )
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_batches <= 0:
        raise SystemExit("--max-batches must be positive.")
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive.")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("--require-gpu was set, but CUDA is not available.")

    seed_everywhere(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested CUDA device {device}, but CUDA is not available.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    mapping_report = validate_action_mapping_preflight(
        config.data.action_mapping,
        action_schema_dim=config.data.action_schema.action_dim,
    )
    pipeline = build_variant_pipeline_from_config(config).to(device)
    pipeline.eval()

    train_dataset, val_dataset = _build_datasets(config.data)
    dataset = train_dataset if args.split == "train" else val_dataset
    collate_fn = collate_latent_wam_samples if _is_latent_dataset(config.data.dataset_type) else collate_wam_samples
    batch_size = args.batch_size or (config.data.train_batch_size if args.split == "train" else config.data.val_batch_size)
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
        latent=_is_latent_dataset(config.data.dataset_type),
    )

    summary = {
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
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def _is_latent_dataset(dataset_type: str) -> bool:
    return dataset_type == "lerobot_v2_latent_local"


def _build_datasets(data_config):
    if _is_latent_dataset(data_config.dataset_type):
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


def _build_load_report(batch: WAMBatch | LatentWAMBatch, config) -> dict[str, Any]:
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


def _run_train_forward(pipeline, batch: WAMBatch | LatentWAMBatch) -> dict[str, Any]:
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


def _run_batch_infer(pipeline, batch: WAMBatch | LatentWAMBatch) -> dict[str, Any]:
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
    pipeline,
    dataset,
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
            previous_action = step_output.infer_output.decoder_output.action_pred[:, :1].detach()
            action_shapes.append(list(step_output.infer_output.decoder_output.action_pred.shape))
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


if __name__ == "__main__":
    main()
