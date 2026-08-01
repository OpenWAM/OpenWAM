from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import (
    DataConfig,
    DataSplit,
    EvalMode,
    EvalPredictionSource,
    ExperimentConfig,
    ReferenceCoreInitMode,
    TrainerAccelerator,
    load_experiment_config,
)
from open_wam.data import (
    LatentWAMBatch,
    LatentWAMSample,
    WAMBatch,
    WAMSample,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    move_latent_wam_batch_to_device,
    move_wam_batch_to_device,
)
from open_wam.evals.evaluation_contracts import (
    EvaluationRequest,
    EvaluationSummary,
    _coerce_optional_positive_int,
    _read_yaml,
    _resolve_relative_path,
    resolve_evaluation_request,
)
from open_wam.evals.evaluation_metrics import (
    _align_eval_action_tensors,
    _align_local_future_video_prediction,
    _masked_action_mse,
    _select_eval_action_prediction,
    _select_eval_video_prediction,
    _select_rollout_previous_action,
    _video_latent_mse,
)
from open_wam.evals.evaluation_windows import (
    _align_rollout_window_tensor,
    _group_dataset_indices_by_episode,
    _resolve_observation_frame_indices,
)
from open_wam.extensions import load_extension_modules
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config
from open_wam.runtime.checkpoints import load_pipeline_checkpoint, resolve_checkpoint_file
from open_wam.utils import seed_everywhere


__all__ = ["EvaluationRequest", "EvaluationSummary", "resolve_evaluation_request", "run_evaluation"]

# Preserve established private imports from this command module while the
# implementations live with their reusable metric and window contracts.
_EVALUATION_COMPATIBILITY_EXPORTS = (
    _align_eval_action_tensors,
    _align_local_future_video_prediction,
    _align_rollout_window_tensor,
    _coerce_optional_positive_int,
    _group_dataset_indices_by_episode,
    _masked_action_mse,
    _read_yaml,
    _resolve_observation_frame_indices,
    _resolve_relative_path,
    _select_eval_action_prediction,
    _select_eval_video_prediction,
    _select_rollout_previous_action,
    _video_latent_mse,
)


def _apply_checkpoint_runtime_override(
    experiment_config: ExperimentConfig,
    checkpoint_path: Path,
) -> Path | None:
    checkpoint_file = resolve_checkpoint_file(checkpoint_path)
    transformer_dir = checkpoint_file.parent / "transformer"
    if not _is_usable_transformer_dir(transformer_dir):
        return checkpoint_file
    object.__setattr__(experiment_config.backbone, "transformer_subdir", str(transformer_dir.resolve()))
    object.__setattr__(experiment_config.backbone, "reference_core_init_mode", ReferenceCoreInitMode.FULL)
    return checkpoint_file


def _is_usable_transformer_dir(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _resolve_device(device: str, experiment_config: ExperimentConfig) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if experiment_config.trainer.accelerator == TrainerAccelerator.CPU:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_eval_dataloader(
    data_config: DataConfig,
    *,
    split: DataSplit,
    batch_size_override: int | None,
) -> DataLoader[WAMBatch | LatentWAMBatch]:
    uses_latents = _uses_latent_dataset(data_config)
    if uses_latents:
        train_dataset, val_dataset = build_train_val_latent_datasets(data_config)
    else:
        train_dataset, val_dataset = build_train_val_datasets(data_config)
    dataset: Dataset[WAMSample]
    if split == DataSplit.TRAIN:
        dataset = train_dataset
        batch_size = batch_size_override or data_config.train_batch_size
    elif split == DataSplit.VAL:
        dataset = val_dataset
        batch_size = batch_size_override or data_config.val_batch_size
    else:
        raise ValueError(f"Unsupported eval split '{split}'. Expected 'train' or 'val'.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=collate_latent_wam_samples if uses_latents else collate_wam_samples,
    )


def _select_eval_dataset(
    data_config: DataConfig,
    *,
    split: DataSplit,
) -> Dataset[WAMSample] | Dataset[LatentWAMSample]:
    if _uses_latent_dataset(data_config):
        train_dataset, val_dataset = build_train_val_latent_datasets(data_config)
    else:
        train_dataset, val_dataset = build_train_val_datasets(data_config)
    if split == DataSplit.TRAIN:
        return train_dataset
    if split == DataSplit.VAL:
        return val_dataset
    raise ValueError(f"Unsupported eval split '{split}'. Expected 'train' or 'val'.")


def _uses_latent_dataset(data_config: DataConfig) -> bool:
    return str(data_config.dataset_type) == "lerobot_v2_latent_local"


def _mot_requires_observation_conditioned_session_reset(experiment_config: ExperimentConfig) -> bool:
    return str(experiment_config.policy_variant.name) == "mot"


def run_evaluation(
    request: EvaluationRequest,
) -> EvaluationSummary:
    """Run the generic evaluation pipeline on the requested split."""

    experiment_config = load_experiment_config(request.experiment_config_path)
    resolved_checkpoint_path: Path | None = None
    if request.checkpoint_path is not None:
        resolved_checkpoint_path = _apply_checkpoint_runtime_override(experiment_config, request.checkpoint_path)
    seed_everywhere(request.seed)
    device = _resolve_device(request.device, experiment_config)
    pipeline = build_variant_pipeline_from_config(experiment_config)
    if resolved_checkpoint_path is not None:
        # Load checkpoints on CPU first to avoid doubling GPU memory during
        # deserialization for large full-model eval checkpoints.
        checkpoint_report = load_pipeline_checkpoint(
            pipeline,
            resolved_checkpoint_path,
            map_location=torch.device("cpu"),
        )
        if checkpoint_report.missing_keys:
            print(f"eval.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
        if checkpoint_report.unexpected_keys:
            print(f"eval.checkpoint_unexpected_keys {len(checkpoint_report.unexpected_keys)}")
    pipeline = pipeline.to(device)
    pipeline.eval()

    action_mse_values: list[float] = []
    trajectory_mse_values: list[float] = []
    video_mse_values: list[float] = []
    trajectory_video_mse_values: list[float] = []
    action_prediction_shape: tuple[int, ...] | None = None
    target_action_shape: tuple[int, ...] | None = None
    action_prediction_source = EvalPredictionSource.UNAVAILABLE
    video_prediction_shape: tuple[int, ...] | None = None
    target_video_shape: tuple[int, ...] | None = None
    video_prediction_source = EvalPredictionSource.UNAVAILABLE
    num_batches = 0
    num_trajectories = 0

    with torch.no_grad():
        if request.mode == EvalMode.BATCH:
            dataloader = _build_eval_dataloader(
                experiment_config.data,
                split=request.split,
                batch_size_override=request.batch_size,
            )
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= request.max_batches:
                    break
                if isinstance(batch, LatentWAMBatch):
                    batch = move_latent_wam_batch_to_device(batch, device)
                else:
                    batch = move_wam_batch_to_device(batch, device)
                infer_context = PolicyInferContext(
                    state=batch.state,
                    extra={
                        "task_text": batch.task_text,
                        "metadata": batch.metadata,
                    },
                )
                # `forward_infer_step` already runs the full denoising loop for
                # the active variant. Batch mode simply evaluates that one-step
                # inference path independently on each sampled window.
                if isinstance(batch, LatentWAMBatch):
                    output = pipeline.forward_infer_step_from_latents(
                        batch.video_latents,
                        infer_context,
                        canonical_video=batch.canonical_video,
                        text_context=batch.text_context,
                        negative_text_context=batch.negative_text_context,
                    )
                else:
                    output = pipeline.forward_infer_step(batch.views, infer_context)
                action_prediction_source, action_prediction = _select_eval_action_prediction(
                    target_actions=batch.actions,
                    decoder_action_pred=output.decoder_output.action_pred,
                    policy_aux=output.policy_output.aux,
                )
                (
                    action_prediction_source,
                    action_prediction,
                    aligned_target_actions,
                    aligned_action_mask,
                ) = _align_eval_action_tensors(
                    source=action_prediction_source,
                    prediction=action_prediction,
                    target_actions=batch.actions,
                    action_mask=batch.action_mask,
                )
                video_prediction_source, video_prediction, aligned_target_video_latents = _select_eval_video_prediction(
                    target_video_latents=output.visual_outputs.frontend.video_latents,
                    decoder_aux=output.decoder_output.aux,
                    policy_aux=output.policy_output.aux,
                    sequence_context=output.policy_output.decoder_sequence_context,
                )
                action_prediction_shape = tuple(action_prediction.shape)
                target_action_shape = tuple(aligned_target_actions.shape)
                target_video_shape = tuple(aligned_target_video_latents.shape)
                if video_prediction is not None:
                    video_prediction_shape = tuple(video_prediction.shape)
                if action_prediction.shape == aligned_target_actions.shape:
                    action_mse_values.append(
                        _masked_action_mse(
                            action_prediction,
                            aligned_target_actions,
                            aligned_action_mask,
                        )
                    )
                if video_prediction is not None and video_prediction.shape == aligned_target_video_latents.shape:
                    video_mse_values.append(
                        _video_latent_mse(
                            video_prediction,
                            aligned_target_video_latents,
                        )
                    )
                num_batches += 1
        elif request.mode in {EvalMode.TRAJECTORY, EvalMode.TRAJECTORY_OPEN_LOOP}:
            dataset = _select_eval_dataset(experiment_config.data, split=request.split)
            episode_groups = _group_dataset_indices_by_episode(dataset)
            if request.max_trajectories is not None:
                episode_groups = episode_groups[: request.max_trajectories]

            for trajectory_index, dataset_indices in enumerate(episode_groups):
                requested_steps = request.max_steps_per_trajectory
                planned_steps = (
                    min(len(dataset_indices), requested_steps)
                    if requested_steps is not None
                    else len(dataset_indices)
                )
                print(
                    "eval.trajectory_start",
                    {
                        "trajectory_index": trajectory_index,
                        "num_dataset_steps": len(dataset_indices),
                        "planned_steps": planned_steps,
                        "mode": str(request.mode),
                    },
                    flush=True,
                )
                rollout_runner = VariantRolloutRunner(pipeline)
                session = None
                previous_action = None
                step_mse_values: list[float] = []
                step_video_mse_values: list[float] = []
                rollout_latents: torch.Tensor | None = None
                rollout_canonical_video: torch.Tensor | None = None
                rollout_frame_indices: tuple[int, ...] | None = None
                for step_index, dataset_index in enumerate(dataset_indices):
                    if request.max_steps_per_trajectory is not None and step_index >= request.max_steps_per_trajectory:
                        break
                    print(
                        "eval.trajectory_step",
                        {
                            "trajectory_index": trajectory_index,
                            "step_index": step_index,
                            "dataset_index": dataset_index,
                        },
                        flush=True,
                    )
                    sample = dataset[dataset_index]
                    if isinstance(sample, LatentWAMSample):
                        batch = move_latent_wam_batch_to_device(collate_latent_wam_samples([sample]), device)
                    else:
                        batch = move_wam_batch_to_device(collate_wam_samples([sample]), device)
                    if session is None:
                        session = rollout_runner.reset(
                            task_text=batch.task_text,
                            text_context=(
                                batch.text_context if isinstance(batch, LatentWAMBatch) else None
                            ),
                            negative_text_context=(
                                batch.negative_text_context if isinstance(batch, LatentWAMBatch) else None
                            ),
                        )
                    elif _mot_requires_observation_conditioned_session_reset(experiment_config):
                        # MoT's video-prefill cache is built from the current
                        # observation window. Trajectory eval advances windows,
                        # so reuse text conditioning but rebuild MoT cache.
                        session = rollout_runner.reset(
                            task_text=batch.task_text,
                            text_context=(
                                batch.text_context if isinstance(batch, LatentWAMBatch) else session.text_context
                            ),
                            negative_text_context=(
                                batch.negative_text_context
                                if isinstance(batch, LatentWAMBatch)
                                else session.negative_text_context
                            ),
                        )
                    infer_context = PolicyInferContext(
                        state=batch.state,
                        previous_action=previous_action,
                        extra={
                            "task_text": batch.task_text,
                            "metadata": batch.metadata,
                        },
                    )
                    if request.mode == EvalMode.TRAJECTORY_OPEN_LOOP:
                        if isinstance(batch, LatentWAMBatch):
                            target_visual_outputs = pipeline.prepare_visual_outputs_from_latents(
                                batch.video_latents,
                                task_text=batch.task_text,
                                text_context=batch.text_context,
                                negative_text_context=batch.negative_text_context,
                                canonical_video=batch.canonical_video,
                            )
                        else:
                            target_visual_outputs = pipeline.prepare_visual_outputs(
                                batch.views,
                                task_text=batch.task_text,
                            )
                        current_frame_indices = _resolve_observation_frame_indices(
                            batch.metadata[0],
                            num_frames=target_visual_outputs.frontend.video_latents.shape[2],
                        )
                        if rollout_latents is not None:
                            # Trajectory-open-loop steps advance the dataset
                            # observation window. Reuse predicted latents only
                            # for overlapping frame ids, and seed newly entered
                            # frames from the current clean window so the
                            # rollout stays temporally aligned.
                            aligned_rollout_latents = _align_rollout_window_tensor(
                                rollout_latents,
                                previous_frame_indices=rollout_frame_indices,
                                current_frame_indices=current_frame_indices,
                                current_target_tensor=target_visual_outputs.frontend.video_latents,
                                frame_dim=2,
                            )
                            aligned_canonical_video = _align_rollout_window_tensor(
                                rollout_canonical_video,
                                previous_frame_indices=rollout_frame_indices,
                                current_frame_indices=current_frame_indices,
                                current_target_tensor=target_visual_outputs.frontend.canonical_video,
                                frame_dim=2,
                            )
                            step_output = rollout_runner.infer_step(
                                session=session,
                                context=infer_context,
                                video_latents=aligned_rollout_latents,
                                canonical_video=aligned_canonical_video,
                            )
                            output = step_output.infer_output
                            session = step_output.session
                        else:
                            if isinstance(batch, LatentWAMBatch):
                                step_output = rollout_runner.infer_step(
                                    session=session,
                                    context=infer_context,
                                    video_latents=batch.video_latents,
                                    canonical_video=batch.canonical_video,
                                )
                            else:
                                step_output = rollout_runner.infer_step(
                                    session=session,
                                    context=infer_context,
                                    views=batch.views,
                                )
                            output = step_output.infer_output
                            session = step_output.session
                        target_video_latents = target_visual_outputs.frontend.video_latents
                    else:
                        if isinstance(batch, LatentWAMBatch):
                            step_output = rollout_runner.infer_step(
                                session=session,
                                context=infer_context,
                                video_latents=batch.video_latents,
                                canonical_video=batch.canonical_video,
                            )
                        else:
                            step_output = rollout_runner.infer_step(
                                session=session,
                                context=infer_context,
                                views=batch.views,
                            )
                        output = step_output.infer_output
                        session = step_output.session
                        target_video_latents = output.visual_outputs.frontend.video_latents
                    action_prediction_source, action_prediction = _select_eval_action_prediction(
                        target_actions=batch.actions,
                        decoder_action_pred=output.decoder_output.action_pred,
                        policy_aux=output.policy_output.aux,
                    )
                    (
                        action_prediction_source,
                        action_prediction,
                        aligned_target_actions,
                        aligned_action_mask,
                    ) = _align_eval_action_tensors(
                        source=action_prediction_source,
                        prediction=action_prediction,
                        target_actions=batch.actions,
                        action_mask=batch.action_mask,
                    )
                    video_prediction_source, video_prediction, aligned_target_video_latents = _select_eval_video_prediction(
                        target_video_latents=target_video_latents,
                        decoder_aux=output.decoder_output.aux,
                        policy_aux=output.policy_output.aux,
                        sequence_context=output.policy_output.decoder_sequence_context,
                    )
                    action_prediction_shape = tuple(action_prediction.shape)
                    target_action_shape = tuple(aligned_target_actions.shape)
                    target_video_shape = tuple(aligned_target_video_latents.shape)
                    if video_prediction is not None:
                        video_prediction_shape = tuple(video_prediction.shape)
                    if action_prediction.shape == aligned_target_actions.shape:
                        step_mse = _masked_action_mse(
                            action_prediction,
                            aligned_target_actions,
                            aligned_action_mask,
                        )
                        action_mse_values.append(step_mse)
                        step_mse_values.append(step_mse)
                    if video_prediction is not None and video_prediction.shape == aligned_target_video_latents.shape:
                        step_video_mse = _video_latent_mse(video_prediction, aligned_target_video_latents)
                        video_mse_values.append(step_video_mse)
                        step_video_mse_values.append(step_video_mse)
                    previous_action = _select_rollout_previous_action(
                        decoder_action_pred=output.decoder_output.action_pred,
                        policy_aux=output.policy_output.aux,
                    ).detach()
                    if video_prediction is not None:
                        rollout_latents = video_prediction.detach()
                    if request.mode == EvalMode.TRAJECTORY_OPEN_LOOP:
                        rollout_frame_indices = current_frame_indices
                    rollout_canonical_video = output.visual_outputs.frontend.canonical_video
                    num_batches += 1
                print(
                    "eval.trajectory_done",
                    {
                        "trajectory_index": trajectory_index,
                        "num_step_mse": len(step_mse_values),
                        "num_step_video_mse": len(step_video_mse_values),
                        "mean_step_action_mse": (
                            sum(step_mse_values) / len(step_mse_values)
                            if step_mse_values
                            else None
                        ),
                        "mean_step_video_mse": (
                            sum(step_video_mse_values) / len(step_video_mse_values)
                            if step_video_mse_values
                            else None
                        ),
                    },
                    flush=True,
                )
                if step_mse_values:
                    trajectory_mse_values.append(sum(step_mse_values) / len(step_mse_values))
                    if step_video_mse_values:
                        trajectory_video_mse_values.append(sum(step_video_mse_values) / len(step_video_mse_values))
                    num_trajectories += 1
        else:
            raise ValueError(
                f"Unsupported eval mode '{request.mode}'. Expected 'batch', 'trajectory', or 'trajectory_open_loop'."
            )

    if num_batches == 0:
        raise ValueError(
            f"Evaluation mode '{request.mode}' on split '{request.split}' for "
            f"{request.experiment_config_path} produced zero evaluation steps."
        )

    return EvaluationSummary(
        experiment_name=experiment_config.name,
        mode=request.mode,
        split=request.split,
        num_batches=num_batches,
        num_trajectories=num_trajectories,
        device=str(device),
        video_num_inference_steps=int(experiment_config.inference.video_num_inference_steps),
        action_num_inference_steps=int(experiment_config.inference.action_num_inference_steps),
        joint_num_inference_steps=(
            None
            if experiment_config.inference.joint_num_inference_steps is None
            else int(experiment_config.inference.joint_num_inference_steps)
        ),
        guidance_scale=float(experiment_config.inference.guidance_scale),
        action_guidance_scale=float(experiment_config.inference.action_guidance_scale),
        action_prediction_source=action_prediction_source,
        action_prediction_shape=action_prediction_shape or tuple(),
        target_action_shape=target_action_shape or tuple(),
        video_prediction_source=video_prediction_source,
        video_prediction_shape=video_prediction_shape or tuple(),
        target_video_shape=target_video_shape or tuple(),
        mean_action_mse=(sum(action_mse_values) / len(action_mse_values)) if action_mse_values else None,
        mean_trajectory_action_mse=(
            sum(trajectory_mse_values) / len(trajectory_mse_values) if trajectory_mse_values else None
        ),
        mean_video_latent_mse=(sum(video_mse_values) / len(video_mse_values)) if video_mse_values else None,
        mean_trajectory_video_latent_mse=(
            sum(trajectory_video_mse_values) / len(trajectory_video_mse_values)
            if trajectory_video_mse_values
            else None
        ),
        checkpoint_path=str(resolved_checkpoint_path) if resolved_checkpoint_path is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "--config", dest="config", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--extension", action="append", default=[])
    args = parser.parse_args()

    load_extension_modules(args.extension)
    request = resolve_evaluation_request(
        args.config,
        mode_override=args.mode,
        split_override=args.split,
        max_batches_override=args.max_batches,
        max_trajectories_override=args.max_trajectories,
        max_steps_per_trajectory_override=args.max_steps_per_trajectory,
        batch_size_override=args.batch_size,
        checkpoint_override=args.checkpoint,
        device_override=args.device,
        seed_override=args.seed,
    )
    summary = run_evaluation(request)
    print("eval.experiment_name", summary.experiment_name)
    print("eval.mode", summary.mode)
    print("eval.split", summary.split)
    print("eval.num_batches", summary.num_batches)
    print("eval.num_trajectories", summary.num_trajectories)
    print("eval.device", summary.device)
    print("eval.video_num_inference_steps", summary.video_num_inference_steps)
    print("eval.action_num_inference_steps", summary.action_num_inference_steps)
    print("eval.joint_num_inference_steps", summary.joint_num_inference_steps)
    print("eval.guidance_scale", summary.guidance_scale)
    print("eval.action_guidance_scale", summary.action_guidance_scale)
    print("eval.action_prediction_source", summary.action_prediction_source)
    print("eval.action_prediction_shape", summary.action_prediction_shape)
    print("eval.target_action_shape", summary.target_action_shape)
    print("eval.video_prediction_source", summary.video_prediction_source)
    print("eval.video_prediction_shape", summary.video_prediction_shape)
    print("eval.target_video_shape", summary.target_video_shape)
    print("eval.mean_action_mse", summary.mean_action_mse)
    print("eval.mean_trajectory_action_mse", summary.mean_trajectory_action_mse)
    print("eval.mean_video_latent_mse", summary.mean_video_latent_mse)
    print("eval.mean_trajectory_video_latent_mse", summary.mean_trajectory_video_latent_mse)
    print("eval.checkpoint_path", summary.checkpoint_path)


if __name__ == "__main__":
    main()
