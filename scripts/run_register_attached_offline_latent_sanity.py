from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import DataSplit, ReferenceCoreInitMode  # noqa: E402
from open_wam.data import (  # noqa: E402
    LatentWAMBatch,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from open_wam.models.policy_variants import PolicyTrainBatch  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline latent sanity check for register-attached checkpoints.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/register_attached_libero_latent_local.yaml",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mode", type=str, default="infer", choices=("infer", "train"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="outputs/register_attached_offline_latent_sanity")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)

    checkpoint_step_dir = _resolve_checkpoint_step_dir(Path(args.checkpoint))
    object.__setattr__(config.backbone, "transformer_subdir", str((checkpoint_step_dir / "transformer").resolve()))
    object.__setattr__(config.backbone, "reference_core_init_mode", ReferenceCoreInitMode.FULL)

    device = _resolve_device(args.device)
    decode_device = _resolve_device(args.decode_device, fallback=device)
    seed_everywhere(args.seed)

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
    dataset = train_dataset if args.split == "train" else val_dataset
    if len(dataset) == 0:
        raise RuntimeError(f"Selected split '{args.split}' is empty.")
    sample_index = int(np.clip(args.sample_index, 0, len(dataset) - 1))
    batch = collate_latent_wam_samples([dataset[sample_index]])
    batch = move_latent_wam_batch_to_device(batch, device)

    pipeline = build_variant_pipeline_from_config(config).to(device)
    pipeline.eval()

    with torch.inference_mode():
        run_result = _run_pipeline_pass(
            pipeline=pipeline,
            batch=batch,
            mode=args.mode,
        )

    target_video_latents = run_result["target_video_latents"]
    predicted_video_latents = run_result["predicted_video_latents"]
    target_actions = batch.actions.detach().cpu()
    predicted_actions = run_result["predicted_actions"]
    action_mask = None if batch.action_mask is None else batch.action_mask.detach().cpu()

    observed_prefix_frames = int(config.inference.joint_observed_video_prefix_frames)
    observed_prefix_latents = target_video_latents[:, :, :observed_prefix_frames]
    target_future_latents = target_video_latents[:, :, observed_prefix_frames:]
    predicted_future_latents, aligned_target_future_latents = _resolve_aligned_future_latents(
        target_video_latents=target_video_latents,
        predicted_video_latents=predicted_video_latents,
        observed_prefix_frames=observed_prefix_frames,
    )
    first_future_block_predicted_latents, first_future_block_target_latents = _resolve_first_future_block_latents(
        target_future_latents=target_future_latents,
        predicted_future_latents=predicted_future_latents,
    )

    output_dir = Path(args.output_dir) / args.split / f"sample_{sample_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_prefix_video = _decode_latent_video(pipeline, observed_prefix_latents, decode_device=decode_device)
    target_full_video = _decode_latent_video(pipeline, target_video_latents, decode_device=decode_device)
    target_future_video = _decode_latent_video(pipeline, target_future_latents, decode_device=decode_device)
    predicted_full_video = (
        None
        if predicted_video_latents is None
        else _decode_latent_video(pipeline, predicted_video_latents, decode_device=decode_device)
    )
    predicted_future_video = (
        None
        if predicted_future_latents is None or predicted_future_latents.shape[2] == 0
        else _decode_latent_video(pipeline, predicted_future_latents, decode_device=decode_device)
    )

    observed_prefix_path = _write_video(output_dir / "observed_prefix.mp4", observed_prefix_video)
    target_full_path = _write_video(output_dir / "target_full.mp4", target_full_video)
    target_future_path = _write_video(output_dir / "target_future.mp4", target_future_video)
    predicted_full_path = _write_video(output_dir / "predicted_full.mp4", predicted_full_video)
    predicted_future_path = _write_video(output_dir / "predicted_future.mp4", predicted_future_video)
    comparison_path = _write_comparison_video(
        output_dir / "target_vs_predicted_future.mp4",
        left_video=target_future_video,
        right_video=predicted_future_video,
        left_label="target_future",
        right_label="predicted_future",
    )

    summary = {
        "checkpoint_step_dir": str(checkpoint_step_dir),
        "split": args.split,
        "sample_index": sample_index,
        "mode": args.mode,
        "task_text": None if batch.task_text is None else batch.task_text[0],
        "video_latent_shape": list(target_video_latents.shape),
        "predicted_video_latent_shape": None if predicted_video_latents is None else list(predicted_video_latents.shape),
        "action_shape": list(target_actions.shape),
        "predicted_action_shape": list(predicted_actions.shape),
        "target_video_latent_stats": _tensor_stats(target_video_latents),
        "predicted_video_latent_stats": _tensor_stats(predicted_video_latents),
        "target_action_stats": _tensor_stats(target_actions),
        "predicted_action_stats": _tensor_stats(predicted_actions),
        "video_flow_pred_stats": _tensor_stats(run_result.get("video_flow_pred")),
        "observed_prefix_frames": observed_prefix_frames,
        "video_latent_mse": (
            None
            if predicted_video_latents is None or tuple(predicted_video_latents.shape) != tuple(target_video_latents.shape)
            else _video_latent_mse(predicted_video_latents, target_video_latents)
        ),
        "future_video_latent_mse": (
            None
            if predicted_future_latents is None or aligned_target_future_latents is None
            else _video_latent_mse(predicted_future_latents, aligned_target_future_latents)
        ),
        "first_future_block_video_latent_mse": (
            None
            if first_future_block_predicted_latents is None or first_future_block_target_latents is None
            else _video_latent_mse(first_future_block_predicted_latents, first_future_block_target_latents)
        ),
        "first_future_block_target_stats": _tensor_stats(first_future_block_target_latents),
        "first_future_block_predicted_stats": _tensor_stats(first_future_block_predicted_latents),
        "action_mse": _masked_action_mse(predicted_actions, target_actions, action_mask),
        "metadata": batch.metadata[0],
        "videos": {
            "observed_prefix": None if observed_prefix_path is None else str(observed_prefix_path),
            "target_full": None if target_full_path is None else str(target_full_path),
            "target_future": None if target_future_path is None else str(target_future_path),
            "predicted_full": None if predicted_full_path is None else str(predicted_full_path),
            "predicted_future": None if predicted_future_path is None else str(predicted_future_path),
            "target_vs_predicted_future": None if comparison_path is None else str(comparison_path),
        },
        "joint_inference_step_debug": run_result.get("joint_inference_step_debug"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _run_pipeline_pass(
    *,
    pipeline,
    batch: LatentWAMBatch,
    mode: str,
) -> dict[str, torch.Tensor | None]:
    if mode == "train":
        output = pipeline(
            video_latents=batch.video_latents,
            batch=_build_policy_train_batch(batch),
            canonical_video=batch.canonical_video,
            text_context=batch.text_context,
            negative_text_context=batch.negative_text_context,
        )
        target_video_latents = output.visual_outputs.frontend.video_latents.detach().cpu()
        predicted_video_latents = _select_predicted_video_latents(
            target_video_latents=target_video_latents,
            decoder_aux=output.decoder_output.aux,
            policy_aux=output.policy_output.aux,
        )
        predicted_actions = output.decoder_output.action_pred.detach().cpu()
        train_artifacts = output.policy_output.aux.get("joint_train_decoder_artifacts", {})
        future_video_flow_pred = train_artifacts.get("aux", {}).get("future_video_flow_pred")
        if isinstance(future_video_flow_pred, torch.Tensor):
            future_video_flow_pred = future_video_flow_pred.detach().cpu()
        return {
            "target_video_latents": target_video_latents,
            "predicted_video_latents": predicted_video_latents,
            "predicted_actions": predicted_actions,
            "joint_inference_step_debug": None,
            "video_flow_pred": future_video_flow_pred,
        }

    runner = VariantRolloutRunner(pipeline)
    session = runner.reset(
        task_text=batch.task_text,
        text_context=batch.text_context,
        negative_text_context=batch.negative_text_context,
    )
    step_output = runner.infer_step(
        session=session,
        context=PolicyInferContext(
            state=batch.state,
            extra={"task_text": batch.task_text},
        ),
        video_latents=batch.video_latents,
        canonical_video=batch.canonical_video,
    )
    infer_output = step_output.infer_output
    target_video_latents = infer_output.visual_outputs.frontend.video_latents.detach().cpu()
    predicted_video_latents = _select_predicted_video_latents(
        target_video_latents=target_video_latents,
        decoder_aux=infer_output.decoder_output.aux,
        policy_aux=infer_output.policy_output.aux,
    )
    predicted_actions = infer_output.decoder_output.action_pred.detach().cpu()
    step_debug = infer_output.policy_output.aux.get("core_aux", {}).get("joint_inference_step_debug")
    return {
        "target_video_latents": target_video_latents,
        "predicted_video_latents": predicted_video_latents,
        "predicted_actions": predicted_actions,
        "joint_inference_step_debug": step_debug,
        "video_flow_pred": None,
    }


def _build_policy_train_batch(batch: LatentWAMBatch) -> PolicyTrainBatch:
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


def _resolve_checkpoint_step_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.name.startswith("checkpoint_step_"):
        return candidate
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]
    raise FileNotFoundError(f"Could not resolve checkpoint_step_* directory from {path}.")


def _resolve_device(device: str | None, *, fallback: torch.device | None = None) -> torch.device:
    if device is None:
        if fallback is not None:
            return fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _select_predicted_video_latents(
    *,
    target_video_latents: torch.Tensor,
    decoder_aux: dict[str, object],
    policy_aux: dict[str, object],
) -> torch.Tensor | None:
    for source in (decoder_aux, policy_aux):
        for key in ("predicted_latents", "predicted_video_latents"):
            candidate = source.get(key)
            if not isinstance(candidate, torch.Tensor):
                continue
            if (
                candidate.ndim == target_video_latents.ndim
                and candidate.shape[0] == target_video_latents.shape[0]
                and candidate.shape[1] == target_video_latents.shape[1]
                and candidate.shape[3:] == target_video_latents.shape[3:]
            ):
                return candidate.detach().cpu()
    return None


def _resolve_aligned_future_latents(
    *,
    target_video_latents: torch.Tensor,
    predicted_video_latents: torch.Tensor | None,
    observed_prefix_frames: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if predicted_video_latents is None:
        return None, None
    target_future_latents = target_video_latents[:, :, observed_prefix_frames:]
    if predicted_video_latents.shape[2] > observed_prefix_frames:
        predicted_future_latents = predicted_video_latents[:, :, observed_prefix_frames:]
    else:
        predicted_future_latents = predicted_video_latents
    if predicted_future_latents.shape[2] == 0:
        return None, None
    aligned_target_future = target_future_latents[:, :, : predicted_future_latents.shape[2]]
    if aligned_target_future.shape[2] != predicted_future_latents.shape[2]:
        return predicted_future_latents, None
    return predicted_future_latents, aligned_target_future


def _resolve_first_future_block_latents(
    *,
    target_future_latents: torch.Tensor,
    predicted_future_latents: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if predicted_future_latents is None or predicted_future_latents.shape[2] == 0:
        return None, None
    block_frames = min(int(predicted_future_latents.shape[2]), int(target_future_latents.shape[2]))
    if block_frames <= 0:
        return None, None
    return (
        predicted_future_latents[:, :, :block_frames],
        target_future_latents[:, :, :block_frames],
    )


def _decode_latent_video(
    pipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
) -> np.ndarray | None:
    if latents.shape[2] == 0:
        return None
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    try:
        vae.to(device=decode_device, dtype=torch.bfloat16)
        decode_dtype = next(vae.parameters()).dtype
        decode_latents = latents.to(device=decode_device, dtype=decode_dtype)
        decode_latents = _denormalize_reference_latents(decode_latents, vae)
        decoded = vae.decode(decode_latents, return_dict=False)[0]
        return video_processor.postprocess_video(decoded, output_type="np")[0]
    finally:
        vae.to(device=original_device, dtype=original_dtype)


def _denormalize_reference_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        mean = torch.tensor(latents_mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return (latents.float() * std + mean).to(dtype=latents.dtype)
    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is not None:
        return latents / float(scaling_factor)
    return latents


def _write_video(path: Path, video: np.ndarray | None) -> Path | None:
    if video is None:
        return None
    frames = [_to_uint8(frame) for frame in video]
    imageio.mimsave(path, frames, fps=5)
    return path


def _write_comparison_video(
    path: Path,
    *,
    left_video: np.ndarray | None,
    right_video: np.ndarray | None,
    left_label: str,
    right_label: str,
) -> Path | None:
    if left_video is None and right_video is None:
        return None
    left_frames = [] if left_video is None else [_to_uint8(frame) for frame in left_video]
    right_frames = [] if right_video is None else [_to_uint8(frame) for frame in right_video]
    frame_count = max(len(left_frames), len(right_frames))
    if frame_count == 0:
        return None
    sample_frame = left_frames[0] if left_frames else right_frames[0]
    blank = np.zeros_like(sample_frame)
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        left = left_frames[index] if index < len(left_frames) else blank
        right = right_frames[index] if index < len(right_frames) else blank
        combined = np.concatenate([left, right], axis=1)
        frames.append(np.array(_add_title_bar(combined, f"{left_label} | {right_label} | frame {index}")))
    imageio.mimsave(path, frames, fps=5)
    return path


def _add_title_bar(frame: np.ndarray, title: str) -> np.ndarray:
    pil = Image.fromarray(np.asarray(frame))
    title_height = 36
    canvas = Image.new("RGB", (pil.width, pil.height + title_height), color=(0, 0, 0))
    canvas.paste(pil, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _masked_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> float:
    squared_error = (predicted.float() - target.float()).pow(2)
    if action_mask is not None:
        squared_error = squared_error * action_mask.float()
        denom = action_mask.float().sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(squared_error.numel()), device=squared_error.device)
    return float((squared_error.sum() / denom).item())


def _video_latent_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float((predicted.float() - target.float()).pow(2).mean().item())


def _tensor_stats(
    tensor: torch.Tensor | None,
) -> dict[str, float | list[int]] | None:
    if tensor is None:
        return None
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "abs_mean": float(value.abs().mean().item()),
        "max_abs": float(value.abs().max().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


if __name__ == "__main__":
    main()
