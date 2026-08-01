"""Episode grouping and temporal window alignment for generic evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from open_wam.configs import LatentTemporalLayout
from open_wam.data import LatentWAMSample, WAMSample
from open_wam.data.latent_temporal import observed_frame_ids_for_latent_segment


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
            layout = metadata.get("latent_temporal_layout", LatentTemporalLayout.WAN_CAUSAL_STRIDE4)
            return tuple(
                observed_frame_ids_for_latent_segment(
                    raw_frame_ids=resolved_ids,
                    source_latent_frames=num_frames,
                    latent_start=0,
                    segment_length=num_frames,
                    layout=layout,
                )
            )
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


__all__: list[str] = []
