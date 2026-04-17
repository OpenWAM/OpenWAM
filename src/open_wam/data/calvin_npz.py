from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from open_wam.configs import ActionTargetRepresentation, CalvinDataConfig, DataConfig

from .action_mapping import apply_action_mapping, resolve_action_source_dim
from .contracts import WAMSample


@dataclass(frozen=True)
class CalvinEpisodeRecord:
    """One CALVIN episode represented by timestep npz files."""

    episode_index: int
    timestep_paths: tuple[Path, ...]

    @property
    def length(self) -> int:
        return len(self.timestep_paths)


@dataclass(frozen=True)
class CalvinWindow:
    """One model window over a CALVIN episode."""

    episode_index: int
    observation_start: int


class CalvinNPZWindowDataset(Dataset[WAMSample]):
    """Windowed native CALVIN reader for `episode_*.npz` timestep files."""

    def __init__(
        self,
        data_config: CalvinDataConfig,
        episodes: tuple[CalvinEpisodeRecord, ...],
        *,
        split_name: str,
    ) -> None:
        self.data_config = data_config
        self.episodes = tuple(episodes)
        self.split_name = split_name
        self.episodes_by_index = {episode.episode_index: episode for episode in self.episodes}
        self.sample_index = self._build_sample_index()
        self._language_spans = _load_calvin_language_spans(data_config.local_root)
        if not self.sample_index:
            raise ValueError(
                "No valid CALVIN windows were constructed. "
                f"Check num_frames={data_config.num_frames}, "
                f"action_horizon={data_config.action_schema.action_horizon}, "
                f"and selected episodes={len(episodes)}."
            )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        episode = self.episodes_by_index[window.episode_index]
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        state_horizon = self.data_config.action_schema.state_horizon

        observation_indices = [
            window.observation_start + offset * frame_stride
            for offset in range(num_frames)
        ]
        anchor_frame_index = window.observation_start + (num_frames - 1) * frame_stride
        action_indices = list(range(anchor_frame_index, anchor_frame_index + action_horizon))
        state_start = max(0, anchor_frame_index - state_horizon + 1)
        state_indices = list(range(state_start, anchor_frame_index + 1))

        observation_steps = [self._load_timestep(episode.timestep_paths[index]) for index in observation_indices]
        action_steps = [self._load_timestep(episode.timestep_paths[index]) for index in action_indices]
        state_steps = [self._load_timestep(episode.timestep_paths[index]) for index in state_indices]

        views = {
            camera_name: torch.stack(
                [
                    _as_uint8_rgb(step[camera_name], key=camera_name)
                    for step in observation_steps
                ],
                dim=0,
            )
            for camera_name in self.data_config.camera_names
        }
        actions, action_mask, action_metadata = self._build_action_targets(action_steps)
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            steps=state_steps,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=state_horizon,
            left_pad=True,
        )
        absolute_anchor = int(_episode_file_index(episode.timestep_paths[anchor_frame_index]))
        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=self._resolve_language(absolute_anchor),
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "dataset_name": self.data_config.dataset_name,
                "local_root": self.data_config.local_root,
                "split": self.split_name,
                "episode_index": episode.episode_index,
                "observation_start": window.observation_start,
                "anchor_frame_index": anchor_frame_index,
                "absolute_anchor_frame_index": absolute_anchor,
                "observation_frame_indices": observation_indices,
                "action_frame_indices": action_indices,
                "state_source_key": state_source_key,
                "action_representation": str(self.data_config.action_target.representation),
                **action_metadata,
            },
        )

    def _build_action_targets(
        self,
        action_steps: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        if action_target.representation != ActionTargetRepresentation.RAW:
            raise ValueError(
                "CALVIN native adapter currently supports only raw action targets, "
                f"got {action_target.representation}."
            )
        target_dim = self.data_config.action_schema.action_dim
        source_dim = resolve_action_source_dim(self.data_config.action_mapping, fallback_dim=target_dim)
        source_actions, source_mask = self._extract_sequence(
            steps=action_steps,
            key=action_target.source_key,
            target_dim=source_dim,
            target_length=self.data_config.action_schema.action_horizon,
        )
        mapped = apply_action_mapping(
            source_actions,
            source_mask,
            self.data_config.action_mapping,
            target_dim=target_dim,
        )
        return mapped.actions, mapped.action_mask, mapped.metadata

    def _extract_sequence(
        self,
        *,
        steps: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not steps:
            raise ValueError(f"Cannot extract CALVIN sequence for key '{key}' from an empty slice.")
        sequence = torch.stack(
            [torch.as_tensor(step[key], dtype=torch.float32).flatten() for step in steps],
            dim=0,
        )
        if sequence.shape[-1] > target_dim:
            raise ValueError(f"Raw CALVIN `{key}` dim {sequence.shape[-1]} exceeds target dim {target_dim}.")
        output = torch.zeros(target_length, target_dim, dtype=torch.float32)
        mask = torch.zeros(target_length, target_dim, dtype=torch.float32)
        clipped = sequence[:target_length]
        start_index = target_length - len(clipped) if left_pad else 0
        for offset, values in enumerate(clipped):
            output[start_index + offset, : values.shape[-1]] = values
            mask[start_index + offset, : values.shape[-1]] = 1.0
        return output, mask

    def _build_sample_index(self) -> list[CalvinWindow]:
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        sample_stride = self.data_config.sample_stride
        required_span = (num_frames - 1) * frame_stride + action_horizon
        windows: list[CalvinWindow] = []
        for episode in self.episodes:
            max_start = episode.length - required_span
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, sample_stride):
                windows.append(CalvinWindow(episode_index=episode.episode_index, observation_start=start))
        return windows

    def _load_timestep(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}

    def _resolve_language(self, absolute_anchor_frame_index: int) -> str | None:
        for start, end, text in self._language_spans:
            if start <= absolute_anchor_frame_index <= end:
                return text
        return "calvin task"


def build_calvin_npz_train_val_datasets(
    data_config: DataConfig,
) -> tuple[CalvinNPZWindowDataset, CalvinNPZWindowDataset]:
    """Build train/val CALVIN window datasets from a local native root."""

    if not isinstance(data_config, CalvinDataConfig):
        raise TypeError("CALVIN dataset builder requires CalvinDataConfig.")
    episodes = discover_calvin_npz_episodes(data_config.local_root)
    episode_indices = list(range(len(episodes)))
    rng = random.Random(data_config.split_seed)
    rng.shuffle(episode_indices)
    train_count = int(len(episode_indices) * data_config.train_fraction)
    train_count = min(max(train_count, 1), len(episode_indices))
    train_indices = episode_indices[:train_count]
    val_indices = episode_indices[train_count:] or train_indices[:1]
    if data_config.max_train_episodes is not None:
        train_indices = train_indices[: data_config.max_train_episodes]
    if data_config.max_val_episodes is not None:
        val_indices = val_indices[: data_config.max_val_episodes]
    return (
        CalvinNPZWindowDataset(
            data_config=data_config,
            episodes=tuple(episodes[index] for index in train_indices),
            split_name="train",
        ),
        CalvinNPZWindowDataset(
            data_config=data_config,
            episodes=tuple(episodes[index] for index in val_indices),
            split_name="val",
        ),
    )


def discover_calvin_npz_episodes(local_root: str | None) -> tuple[CalvinEpisodeRecord, ...]:
    """Discover CALVIN timestep npz files and optional official episode spans."""

    if local_root is None:
        raise ValueError("CALVIN native datasets require `data.local_root`.")
    root = Path(local_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"CALVIN local_root does not exist: {root}")
    sequence_root = _resolve_sequence_root(root)
    timestep_paths = tuple(sorted(sequence_root.glob("episode_*.npz"), key=_episode_file_index))
    if not timestep_paths:
        raise FileNotFoundError(f"No CALVIN `episode_*.npz` files found under {sequence_root}.")

    span_path = _find_first_existing_path(
        sequence_root / "ep_start_end_ids.npy",
        root / "ep_start_end_ids.npy",
    )
    if span_path is None:
        return (CalvinEpisodeRecord(episode_index=0, timestep_paths=timestep_paths),)

    spans = np.load(span_path)
    episodes: list[CalvinEpisodeRecord] = []
    by_index = {_episode_file_index(path): path for path in timestep_paths}
    for episode_index, raw_span in enumerate(spans):
        start, end = int(raw_span[0]), int(raw_span[1])
        paths = tuple(by_index[index] for index in range(start, end + 1) if index in by_index)
        if paths:
            episodes.append(CalvinEpisodeRecord(episode_index=episode_index, timestep_paths=paths))
    if not episodes:
        raise ValueError(f"CALVIN episode span file {span_path} did not match any timestep files.")
    return tuple(episodes)


def _resolve_sequence_root(root: Path) -> Path:
    for candidate in (root / "training", root / "validation", root):
        if any(candidate.glob("episode_*.npz")):
            return candidate
    return root


def _load_calvin_language_spans(local_root: str | None) -> tuple[tuple[int, int, str], ...]:
    if local_root is None:
        return ()
    root = Path(local_root).expanduser()
    annotation_path = _find_first_existing_path(
        root / "lang_annotations" / "auto_lang_ann.npy",
        root / "training" / "lang_annotations" / "auto_lang_ann.npy",
        root / "validation" / "lang_annotations" / "auto_lang_ann.npy",
    )
    if annotation_path is None:
        return ()
    raw = np.load(annotation_path, allow_pickle=True)
    payload = raw.item() if hasattr(raw, "item") else raw
    if not isinstance(payload, dict):
        return ()
    language = payload.get("language", {})
    info = payload.get("info", {})
    annotations = language.get("ann", ()) if isinstance(language, dict) else ()
    indices = info.get("indx", ()) if isinstance(info, dict) else ()
    spans: list[tuple[int, int, str]] = []
    for raw_index, raw_text in zip(indices, annotations, strict=False):
        if len(raw_index) < 2:
            continue
        spans.append((int(raw_index[0]), int(raw_index[1]), str(raw_text)))
    return tuple(spans)


def _as_uint8_rgb(value: Any, *, key: str) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected CALVIN `{key}` image with shape [H, W, 3], got {array.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array))


def _episode_file_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Could not parse CALVIN episode file index from {path.name}.") from exc


def _find_first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
