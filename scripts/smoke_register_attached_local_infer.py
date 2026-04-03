from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import build_train_val_latent_datasets, collate_latent_wam_samples  # noqa: E402
from open_wam.evals.evaluate import _load_pipeline_checkpoint  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils.config_loader import load_experiment_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Experiment YAML path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file or checkpoint directory.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Latent-local dataset sample index to inspect.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Runtime device.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Which latent-local split to read from.",
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=None,
        help="Optional override for inference.video_num_inference_steps.",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help="Optional override for inference.action_num_inference_steps.",
    )
    parser.add_argument(
        "--joint-steps",
        type=int,
        default=None,
        help="Optional override for inference.joint_num_inference_steps.",
    )
    return parser.parse_args()


def _resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    step_dirs = sorted(
        (candidate for candidate in path.glob("checkpoint_step_*") if candidate.is_dir()),
        key=lambda candidate: int(candidate.name.rsplit("_", 1)[-1]),
    )
    for step_dir in reversed(step_dirs):
        full_state = step_dir / "full_training_state.pt"
        model_state = step_dir / "model_state.pt"
        if full_state.exists():
            return full_state
        if model_state.exists():
            return model_state
    full_state = path / "full_training_state.pt"
    model_state = path / "model_state.pt"
    if full_state.exists():
        return full_state
    if model_state.exists():
        return model_state
    raise FileNotFoundError(f"Could not resolve checkpoint file from {path}.")


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    print("stage=config")
    config = load_experiment_config(args.cfg)
    if args.video_steps is not None:
        object.__setattr__(config.inference, "video_num_inference_steps", int(args.video_steps))
    if args.action_steps is not None:
        object.__setattr__(config.inference, "action_num_inference_steps", int(args.action_steps))
    if args.joint_steps is not None:
        object.__setattr__(config.inference, "joint_num_inference_steps", int(args.joint_steps))
    print("stage=dataset")
    train_ds, val_ds = build_train_val_latent_datasets(config.data)
    dataset = train_ds if args.split == "train" else val_ds
    sample = dataset[args.sample_idx]
    batch = collate_latent_wam_samples([sample])

    print("stage=build_pipeline")
    pipeline = build_variant_pipeline_from_config(config).to(device)
    checkpoint_path = _resolve_checkpoint_file(Path(args.checkpoint))
    print("stage=load_checkpoint", checkpoint_path)
    _load_pipeline_checkpoint(pipeline, checkpoint_path, device=device)
    pipeline.eval()

    print("stage=infer")
    infer_output = pipeline.forward_infer_step_from_latents(
        batch.video_latents.to(device),
        context=PolicyInferContext(
            state=batch.state.to(device) if batch.state is not None else None,
            extra={"task_text": batch.task_text},
        ),
        canonical_video=batch.canonical_video.to(device) if batch.canonical_video is not None else None,
        text_context=batch.text_context.to(device) if batch.text_context is not None else None,
        negative_text_context=(
            batch.negative_text_context.to(device) if batch.negative_text_context is not None else None
        ),
    )

    action_pred = infer_output.decoder_output.action_pred.detach().cpu()
    predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
    if isinstance(predicted_latents, torch.Tensor):
        predicted_latents = predicted_latents.detach().cpu()

    print("checkpoint", str(checkpoint_path))
    print("sample_idx", args.sample_idx)
    print("task_text", batch.task_text[0] if batch.task_text is not None else None)
    print("metadata", batch.metadata[0])
    print("input.video_latents", tuple(batch.video_latents.shape))
    print("input.state", tuple(batch.state.shape) if batch.state is not None else None)
    print("infer.action_pred", tuple(action_pred.shape))
    print("infer.action_preview", action_pred[0, : min(3, action_pred.shape[1]), : min(7, action_pred.shape[2])])
    if predicted_latents is not None:
        print("infer.predicted_latents", tuple(predicted_latents.shape))


if __name__ == "__main__":
    main()
