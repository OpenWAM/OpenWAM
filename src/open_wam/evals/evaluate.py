from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import DataConfig, DataSplit, EvalMode, EvalPredictionSource, ExperimentConfig, TrainerAccelerator
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
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.models.policy_variants.contracts import DecoderSequenceContext
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config
from open_wam.utils.local_paths import read_yaml_with_local_paths
from open_wam.utils import load_experiment_config, seed_everywhere


@dataclass(frozen=True)
class EvaluationRequest:
    """Resolved evaluation request after applying YAML defaults and CLI overrides."""

    experiment_config_path: Path
    mode: EvalMode
    split: DataSplit
    max_batches: int
    max_trajectories: int | None
    max_steps_per_trajectory: int | None
    batch_size: int | None
    checkpoint_path: Path | None
    device: str
    seed: int


@dataclass(frozen=True)
class EvaluationSummary:
    """Minimal structured result for CLI output and tests."""

    experiment_name: str
    mode: EvalMode
    split: DataSplit
    num_batches: int
    num_trajectories: int
    device: str
    video_num_inference_steps: int
    action_num_inference_steps: int
    joint_num_inference_steps: int | None
    guidance_scale: float
    action_guidance_scale: float
    action_prediction_source: EvalPredictionSource
    action_prediction_shape: tuple[int, ...]
    target_action_shape: tuple[int, ...]
    video_prediction_source: EvalPredictionSource
    video_prediction_shape: tuple[int, ...]
    target_video_shape: tuple[int, ...]
    mean_action_mse: float | None
    mean_trajectory_action_mse: float | None
    mean_video_latent_mse: float | None
    mean_trajectory_video_latent_mse: float | None
    checkpoint_path: str | None


def _read_yaml(path: Path) -> dict[str, Any]:
    return read_yaml_with_local_paths(path)


def _resolve_relative_path(base_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    local_candidate = (base_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    raise FileNotFoundError(
        f"Could not resolve relative path '{value}' from base '{base_path}'. "
        f"Checked: {local_candidate} and {cwd_candidate}."
    )


def _coerce_optional_positive_int(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name} {value!r} in {config_path}; expected a positive integer or null."
        ) from exc
    if coerced <= 0:
        raise ValueError(
            f"Invalid {field_name} {coerced!r} in {config_path}; expected a positive integer > 0."
        )
    return coerced


def resolve_evaluation_request(
    config_path: str | Path,
    *,
    mode_override: EvalMode | str | None = None,
    split_override: DataSplit | str | None = None,
    max_batches_override: int | None = None,
    max_trajectories_override: int | None = None,
    max_steps_per_trajectory_override: int | None = None,
    batch_size_override: int | None = None,
    checkpoint_override: str | None = None,
    device_override: str | None = None,
    seed_override: int | None = None,
) -> EvaluationRequest:
    """Resolve either an experiment YAML or an eval-wrapper YAML.

    Eval wrappers are lightweight YAMLs under `configs/evals/` with an
    `experiment_config` field plus optional eval defaults such as split, device,
    checkpoint path, and batch count.
    """

    config_path = Path(config_path).resolve()
    raw = _read_yaml(config_path)
    experiment_config_path = (
        _resolve_relative_path(config_path, raw.get("experiment_config"))
        if "experiment_config" in raw
        else config_path
    )
    if experiment_config_path is None:
        raise ValueError(f"Eval config {config_path} is missing `experiment_config`.")
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else _coerce_optional_positive_int(raw.get("batch_size"), field_name="batch_size", config_path=config_path)
    )
    max_trajectories = (
        max_trajectories_override
        if max_trajectories_override is not None
        else _coerce_optional_positive_int(
            raw.get("max_trajectories"),
            field_name="max_trajectories",
            config_path=config_path,
        )
    )
    max_steps_per_trajectory = (
        max_steps_per_trajectory_override
        if max_steps_per_trajectory_override is not None
        else _coerce_optional_positive_int(
            raw.get("max_steps_per_trajectory"),
            field_name="max_steps_per_trajectory",
            config_path=config_path,
        )
    )

    return EvaluationRequest(
        experiment_config_path=experiment_config_path,
        mode=EvalMode(mode_override or raw.get("mode", "batch")),
        split=DataSplit(split_override or raw.get("split", "val")),
        max_batches=max_batches_override if max_batches_override is not None else int(raw.get("max_batches", 1)),
        max_trajectories=max_trajectories,
        max_steps_per_trajectory=max_steps_per_trajectory,
        batch_size=batch_size,
        checkpoint_path=_resolve_relative_path(config_path, checkpoint_override or raw.get("checkpoint_path")),
        device=device_override or raw.get("device", "auto"),
        seed=seed_override if seed_override is not None else int(raw.get("seed", 0)),
    )


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


def _normalize_checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a raw state_dict or a Lightning checkpoint with `state_dict`.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = key[len("pipeline.") :] if key.startswith("pipeline.") else key
        normalized[normalized_key] = value
    return normalized


def _load_pipeline_checkpoint(
    pipeline: torch.nn.Module,
    checkpoint_path: Path,
    *,
    map_location: torch.device,
) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _normalize_checkpoint_state_dict(checkpoint)
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"eval.checkpoint_missing_keys {len(missing)}")
    if unexpected:
        print(f"eval.checkpoint_unexpected_keys {len(unexpected)}")


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
    squared_error = (predicted.float() - target.float()).pow(2)
    return float(squared_error.mean().item())


def _select_eval_action_prediction(
    *,
    target_actions: torch.Tensor,
    decoder_action_pred: torch.Tensor,
    policy_aux: dict[str, Any],
) -> tuple[EvalPredictionSource, torch.Tensor]:
    if decoder_action_pred.shape == target_actions.shape:
        return EvalPredictionSource.DECODER_ACTION_PRED, decoder_action_pred
    raw_chunk_action_pred = policy_aux.get("raw_chunk_action_pred")
    if (
        isinstance(raw_chunk_action_pred, torch.Tensor)
        and raw_chunk_action_pred.ndim == target_actions.ndim
        and raw_chunk_action_pred.shape[0] == target_actions.shape[0]
        and raw_chunk_action_pred.shape[-1] == target_actions.shape[-1]
        and target_actions.shape[1] >= raw_chunk_action_pred.shape[1]
    ):
        return EvalPredictionSource.RAW_CHUNK_ACTION_PRED, raw_chunk_action_pred
    return EvalPredictionSource.DECODER_ACTION_PRED_UNMATCHED, decoder_action_pred


def _align_eval_action_tensors(
    *,
    source: EvalPredictionSource,
    prediction: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> tuple[EvalPredictionSource, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if prediction.shape == target_actions.shape:
        return source, prediction, target_actions, action_mask
    if (
        source == EvalPredictionSource.RAW_CHUNK_ACTION_PRED
        and prediction.ndim == target_actions.ndim
        and prediction.shape[0] == target_actions.shape[0]
        and prediction.shape[-1] == target_actions.shape[-1]
        and target_actions.shape[1] >= prediction.shape[1]
    ):
        target_start = int(target_actions.shape[1] - prediction.shape[1])
        aligned_target = target_actions[:, target_start:]
        aligned_mask = None if action_mask is None else action_mask[:, target_start:]
        return EvalPredictionSource.RAW_CHUNK_ACTION_PRED_TAIL_ALIGNED, prediction, aligned_target, aligned_mask
    return source, prediction, target_actions, action_mask


def _select_eval_video_prediction(
    *,
    target_video_latents: torch.Tensor,
    decoder_aux: dict[str, Any],
    policy_aux: dict[str, Any],
    sequence_context: DecoderSequenceContext | None = None,
) -> tuple[EvalPredictionSource, torch.Tensor | None, torch.Tensor]:
    for source_name in ("predicted_latents", "predicted_video_latents"):
        candidate = decoder_aux.get(source_name)
        if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
            return (
                EvalPredictionSource.DECODER_PREDICTED_LATENTS
                if source_name == "predicted_latents"
                else EvalPredictionSource.DECODER_PREDICTED_VIDEO_LATENTS,
                candidate,
                target_video_latents,
            )
        aligned_target = _align_local_future_video_prediction(
            candidate,
            target_video_latents=target_video_latents,
            sequence_context=sequence_context,
        )
        if aligned_target is not None:
            return EvalPredictionSource.DECODER_PREDICTED_LOCAL_FUTURE_LATENTS, candidate, aligned_target
    for source_name in ("predicted_latents", "predicted_video_latents"):
        candidate = policy_aux.get(source_name)
        if isinstance(candidate, torch.Tensor) and candidate.shape == target_video_latents.shape:
            return (
                EvalPredictionSource.POLICY_PREDICTED_LATENTS
                if source_name == "predicted_latents"
                else EvalPredictionSource.POLICY_PREDICTED_VIDEO_LATENTS,
                candidate,
                target_video_latents,
            )
        aligned_target = _align_local_future_video_prediction(
            candidate,
            target_video_latents=target_video_latents,
            sequence_context=sequence_context,
        )
        if aligned_target is not None:
            return EvalPredictionSource.POLICY_PREDICTED_LOCAL_FUTURE_LATENTS, candidate, aligned_target
    return EvalPredictionSource.UNAVAILABLE, None, target_video_latents


def _align_local_future_video_prediction(
    candidate: Any,
    *,
    target_video_latents: torch.Tensor,
    sequence_context: DecoderSequenceContext | None,
) -> torch.Tensor | None:
    if not isinstance(candidate, torch.Tensor):
        return None
    if candidate.ndim != target_video_latents.ndim or candidate.ndim != 5:
        return None
    if candidate.shape[0:2] != target_video_latents.shape[0:2] or candidate.shape[3:] != target_video_latents.shape[3:]:
        return None
    if sequence_context is None or sequence_context.video_condition_window is None:
        return None
    window = sequence_context.video_condition_window
    metadata = window.metadata
    if metadata.get("source_family") != "generated_future_video_tokens":
        return None
    observed_frames = int(metadata.get("observed_prefix_frames", window.observed_frame_count))
    observed_start = int(metadata.get("observed_prefix_start_index", 0))
    target_start = observed_start + observed_frames
    target_end = target_start + int(candidate.shape[2])
    if target_end > int(target_video_latents.shape[2]):
        return None
    return target_video_latents[:, :, target_start:target_end]


def _group_dataset_indices_by_episode(dataset: Dataset[WAMSample] | Dataset[LatentWAMSample]) -> list[list[int]]:
    """Group one split's windowed samples into episode-ordered trajectories.

    Trajectory-mode evaluation needs windows ordered by episode and observation
    start so one infer state can be carried across the rollout. The LeRobot and
    LIBERO offline datasets already expose a lightweight `sample_index` with
    exactly that metadata; we use it when available to avoid decoding RGB just
    to discover ordering.
    """

    sample_index = getattr(dataset, "sample_index", None)
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    if sample_index is not None:
        for dataset_index, window in enumerate(sample_index):
            episode_index = getattr(window, "episode_index", None)
            observation_start = getattr(window, "observation_start", None)
            if observation_start is None:
                observation_frame_indices = getattr(window, "observation_frame_indices", None)
                if isinstance(observation_frame_indices, (list, tuple)) and observation_frame_indices:
                    observation_start = observation_frame_indices[0]
            if episode_index is None or observation_start is None:
                raise ValueError(
                    "Trajectory evaluation requires dataset sample_index entries with "
                    "`episode_index` and `observation_start`."
                )
            dataset_identity = (
                getattr(window, "repo_id", None)
                or getattr(window, "member_id", None)
                or getattr(window, "dataset_id", None)
                or getattr(window, "repo_root", None)
                or getattr(window, "local_root", None)
                or "__default__"
            )
            grouped.setdefault((str(dataset_identity), int(episode_index)), []).append((int(observation_start), dataset_index))
        return [
            [dataset_index for _, dataset_index in sorted(entries)]
            for _, entries in sorted(grouped.items(), key=lambda item: item[0])
        ]

    # Fallback for simple datasets that only expose episode metadata via the
    # public sample contract. This is slower because it materializes samples,
    # but keeps trajectory eval usable for small custom datasets.
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        episode_index = sample.metadata.get("episode_index")
        observation_start = sample.metadata.get("observation_start")
        if observation_start is None:
            observation_start = sample.metadata.get("window_start_frame")
        if observation_start is None:
            observation_start = sample.metadata.get("sample_start_frame")
        if episode_index is None or observation_start is None:
            raise ValueError(
                "Trajectory evaluation requires either a dataset.sample_index with "
                "`episode_index`/`observation_start`, or per-sample metadata with "
                "those fields."
            )
        dataset_identity = (
            sample.metadata.get("repo_id")
            or sample.metadata.get("member_id")
            or sample.metadata.get("dataset_id")
            or sample.metadata.get("repo_root")
            or sample.metadata.get("local_root")
            or "__default__"
        )
        grouped.setdefault((str(dataset_identity), int(episode_index)), []).append((int(observation_start), dataset_index))
    return [
        [dataset_index for _, dataset_index in sorted(entries)]
        for _, entries in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _resolve_observation_frame_indices(
    metadata: dict[str, Any],
    *,
    num_frames: int,
) -> tuple[int, ...]:
    """Resolve per-window frame ids for trajectory-open-loop alignment.

    Open-loop evaluation carries predicted video latents across advancing dataset
    windows. Those windows often overlap, so the latent tensor for the next
    step must be shifted into the current frame-index basis before reuse.
    """

    raw_indices = metadata.get("observation_frame_indices")
    if isinstance(raw_indices, (list, tuple)):
        if len(raw_indices) != num_frames:
            raise ValueError(
                "Expected `observation_frame_indices` to match the current video "
                f"window length {num_frames}, got {len(raw_indices)}."
            )
        return tuple(int(value) for value in raw_indices)

    observed_frame_ids = metadata.get("observed_frame_ids")
    if isinstance(observed_frame_ids, (list, tuple)):
        resolved_ids = [int(value) for value in observed_frame_ids]
        if len(resolved_ids) == num_frames:
            return tuple(resolved_ids)
        if len(resolved_ids) > num_frames:
            bucket_boundaries = [
                (latent_index * len(resolved_ids)) // num_frames
                for latent_index in range(num_frames + 1)
            ]
            bucket_boundaries[-1] = len(resolved_ids)
            aligned_ids: list[int] = []
            for latent_index in range(num_frames):
                start = bucket_boundaries[latent_index]
                end = bucket_boundaries[latent_index + 1]
                if start >= len(resolved_ids):
                    aligned_ids.append(resolved_ids[-1])
                    continue
                bucket = resolved_ids[start:end]
                aligned_ids.append(bucket[-1] if bucket else resolved_ids[start])
            return tuple(aligned_ids)
        raise ValueError(
            "Expected `observed_frame_ids` to contain at least as many entries as the "
            f"current video window length {num_frames}, got {len(resolved_ids)}."
        )

    observation_start = metadata.get("observation_start")
    if observation_start is None:
        observation_start = metadata.get("window_start_frame")
    if observation_start is None:
        observation_start = metadata.get("sample_start_frame")
    if observation_start is None:
        raise ValueError(
            "Trajectory-open-loop evaluation requires per-sample metadata with "
            "`observation_frame_indices`, `observed_frame_ids`, or "
            "`observation_start`/`window_start_frame`/`sample_start_frame`."
        )
    return tuple(int(observation_start) + offset for offset in range(num_frames))


def _align_rollout_window_tensor(
    previous_tensor: torch.Tensor | None,
    *,
    previous_frame_indices: tuple[int, ...] | None,
    current_frame_indices: tuple[int, ...],
    current_target_tensor: torch.Tensor,
    frame_dim: int,
) -> torch.Tensor:
    """Shift a predicted rollout window into the current observation basis.

    Overlapping frame ids reuse the previous step's predicted tensor. Any newly
    entered frames are seeded from the current clean window so evaluation stays
    temporally aligned even when the dataset advances the observation window by
    one or more frames each step.
    """

    aligned = current_target_tensor.clone()
    if previous_tensor is None or previous_frame_indices is None:
        return aligned

    previous_lookup = {frame_index: index for index, frame_index in enumerate(previous_frame_indices)}
    max_previous_frames = previous_tensor.shape[frame_dim]
    for current_index, frame_index in enumerate(current_frame_indices):
        previous_index = previous_lookup.get(frame_index)
        if previous_index is None or previous_index >= max_previous_frames:
            continue
        aligned.select(frame_dim, current_index).copy_(previous_tensor.select(frame_dim, previous_index))
    return aligned


def _mot_requires_observation_conditioned_session_reset(experiment_config: ExperimentConfig) -> bool:
    return str(experiment_config.policy_variant.name) == "mot"


def run_evaluation(
    request: EvaluationRequest,
) -> EvaluationSummary:
    """Run the generic evaluation pipeline on the requested split."""

    experiment_config = load_experiment_config(request.experiment_config_path)
    seed_everywhere(request.seed)
    device = _resolve_device(request.device, experiment_config)
    pipeline = build_variant_pipeline_from_config(experiment_config)
    if request.checkpoint_path is not None:
        # Load checkpoints on CPU first to avoid doubling GPU memory during
        # deserialization for large full-model eval checkpoints.
        _load_pipeline_checkpoint(
            pipeline,
            request.checkpoint_path,
            map_location=torch.device("cpu"),
        )
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
                    previous_action = action_prediction.detach()
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
        checkpoint_path=str(request.checkpoint_path) if request.checkpoint_path is not None else None,
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
    args = parser.parse_args()

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
