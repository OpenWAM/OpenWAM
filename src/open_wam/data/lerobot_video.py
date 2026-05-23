from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from open_wam.configs import ActionTargetRepresentation, DataConfig

from .action_mapping import apply_action_mapping, resolve_action_source_dim
from .contracts import WAMSample
from .lerobot_v2 import EpisodeWindow, LeRobotEpisodeRecord, _resolve_row_key
from .replay_status import load_replay_status_records, split_episode_indices_by_replay_status


@dataclass(frozen=True)
class LeRobotVideoMetadata:
    """Local LeRobot-v2 metadata for repos with external mp4 video features."""

    repo_root: Path
    codebase_version: str
    fps: int
    chunk_size: int
    total_episodes: int
    data_path_template: str
    video_path_template: str
    features: dict[str, dict[str, Any]]
    episodes: tuple[LeRobotEpisodeRecord, ...]
    tasks_by_index: dict[int, str]


class LeRobotV2VideoWindowDataset(Dataset[WAMSample]):
    """Windowed local LeRobot-v2 reader for external-video datasets."""

    def __init__(
        self,
        data_config: DataConfig,
        episodes: list[int],
    ) -> None:
        if data_config.local_root is None:
            raise ValueError("`lerobot_v2_video` currently requires `data.local_root`.")
        self.data_config = data_config
        self.metadata = load_lerobot_v2_video_metadata(Path(data_config.local_root).expanduser())
        self.episodes = tuple(episodes)
        self.episode_records = {episode.episode_index: episode for episode in self.metadata.episodes}
        self.sample_index = self._build_sample_index()
        self._episode_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._video_frame_cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        if not self.sample_index:
            raise ValueError(
                "No valid LeRobot-v2 video windows were constructed. "
                f"Check num_frames={data_config.num_frames}, "
                f"action_horizon={data_config.action_schema.action_horizon}, "
                f"and selected episodes={len(episodes)}."
            )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        rows = self._load_episode_rows(window.episode_index)
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        state_horizon = self.data_config.action_schema.state_horizon
        observation_rows = [
            rows[window.observation_start + offset * frame_stride]
            for offset in range(num_frames)
        ]
        anchor_frame_index = window.observation_start + (num_frames - 1) * frame_stride
        action_rows = rows[anchor_frame_index : anchor_frame_index + action_horizon]
        state_start = max(0, anchor_frame_index - state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]

        views = {
            camera_name: self._decode_video_sequence(
                observation_rows,
                key=camera_name,
                episode_index=window.episode_index,
            )
            for camera_name in self.data_config.camera_names
        }
        actions, action_mask, action_metadata = self._build_action_targets(action_rows)
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=state_horizon,
            left_pad=True,
        )
        task_index = int(observation_rows[-1].get("task_index", 0))
        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=self.metadata.tasks_by_index.get(task_index),
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "local_root": str(self.metadata.repo_root),
                "episode_index": window.episode_index,
                "task_index": task_index,
                "observation_start": window.observation_start,
                "anchor_frame_index": anchor_frame_index,
                "observation_frame_indices": [int(row["frame_index"]) for row in observation_rows],
                "action_frame_indices": [int(row["frame_index"]) for row in action_rows],
                "state_source_key": state_source_key,
                "action_representation": str(self.data_config.action_target.representation),
                **action_metadata,
            },
        )

    def _build_action_targets(
        self,
        action_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        if action_target.representation != ActionTargetRepresentation.RAW:
            raise ValueError(
                "`lerobot_v2_video` currently supports raw action targets. "
                f"Use an explicit transform before enabling {action_target.representation}."
            )
        target_dim = self.data_config.action_schema.action_dim
        source_dim = resolve_action_source_dim(self.data_config.action_mapping, fallback_dim=target_dim)
        source_actions, source_mask = self._extract_sequence(
            rows=action_rows,
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

    def _decode_video_sequence(
        self,
        rows: list[dict[str, Any]],
        *,
        key: str,
        episode_index: int,
    ) -> torch.Tensor:
        frames = self._load_video_frames(key=key, episode_index=episode_index)
        frame_indices = torch.tensor([int(row["frame_index"]) for row in rows], dtype=torch.long)
        if frame_indices.numel() and int(frame_indices.max().item()) >= int(frames.shape[0]):
            raise IndexError(
                f"LeRobot video sequence for episode={episode_index}, key='{key}' requested frame "
                f"{int(frame_indices.max().item())}, but decoded video has {frames.shape[0]} frames."
            )
        return frames.index_select(0, frame_indices)

    def _load_video_frames(self, *, key: str, episode_index: int) -> torch.Tensor:
        cache_key = (int(episode_index), key)
        if cache_key in self._video_frame_cache:
            self._video_frame_cache.move_to_end(cache_key)
            return self._video_frame_cache[cache_key]
        path = self._video_path(key=key, episode_index=episode_index)
        reader = imageio.get_reader(path)
        try:
            frames = []
            for frame in reader:
                tensor = torch.as_tensor(frame, dtype=torch.uint8)
                if tensor.ndim != 3 or tensor.shape[-1] < 3:
                    raise ValueError(f"Expected RGB video frame from {path}, got {tuple(tensor.shape)}.")
                frames.append(tensor[..., :3].contiguous())
            if not frames:
                raise ValueError(f"LeRobot video file has no decodable frames: {path}")
            decoded = torch.stack(frames, dim=0)
        finally:
            reader.close()
        self._video_frame_cache[cache_key] = decoded
        max_entries = max(1, int(self.data_config.episode_cache_size) * max(1, len(self.data_config.camera_names)))
        while len(self._video_frame_cache) > max_entries:
            self._video_frame_cache.popitem(last=False)
        return decoded

    def _video_path(self, *, key: str, episode_index: int) -> Path:
        relative_path = self.metadata.video_path_template.format(
            episode_chunk=episode_index // self.metadata.chunk_size,
            episode_index=episode_index,
            video_key=key,
        )
        path = self.metadata.repo_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing LeRobot video file for key '{key}': {path}")
        return path

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]
        path = self.metadata.repo_root / self.metadata.data_path_template.format(
            episode_chunk=episode_index // self.metadata.chunk_size,
            episode_index=episode_index,
        )
        rows = pq.read_table(path).to_pylist()
        self._episode_cache[episode_index] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows

    def _extract_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError(f"Cannot extract sequence for key '{key}' from an empty row slice.")
        sequence = torch.stack(
            [torch.tensor(row[_resolve_row_key(row, key)], dtype=torch.float32).flatten() for row in rows],
            dim=0,
        )
        if sequence.shape[-1] > target_dim:
            raise ValueError(f"Raw `{key}` dim {sequence.shape[-1]} exceeds configured target dim {target_dim}.")
        output = torch.zeros(target_length, target_dim, dtype=torch.float32)
        mask = torch.zeros(target_length, target_dim, dtype=torch.float32)
        clipped = sequence[:target_length]
        start_index = target_length - len(clipped) if left_pad else 0
        for offset, values in enumerate(clipped):
            output[start_index + offset, : values.shape[-1]] = values
            mask[start_index + offset, : values.shape[-1]] = 1.0
        return output, mask

    def _build_sample_index(self) -> list[EpisodeWindow]:
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        sample_stride = self.data_config.sample_stride
        required_span = (num_frames - 1) * frame_stride + action_horizon
        windows: list[EpisodeWindow] = []
        for episode_index in self.episodes:
            record = self.episode_records[episode_index]
            max_start = record.length - required_span
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, sample_stride):
                windows.append(EpisodeWindow(episode_index=episode_index, observation_start=start))
        return windows


def build_lerobot_v2_video_train_val_datasets(
    data_config: DataConfig,
) -> tuple[LeRobotV2VideoWindowDataset, LeRobotV2VideoWindowDataset]:
    if data_config.local_root is None:
        raise ValueError("`lerobot_v2_video` requires `data.local_root`.")
    metadata = load_lerobot_v2_video_metadata(Path(data_config.local_root).expanduser())
    replay_status_records, replay_status_path = load_replay_status_records(
        metadata.repo_root,
        replay_status_path=data_config.replay_status_path,
        require=data_config.require_replay_status,
    )
    split = split_episode_indices_by_replay_status(
        [episode.episode_index for episode in metadata.episodes],
        replay_status_records=replay_status_records,
        replay_status_path=replay_status_path,
        replay_status_policy=data_config.replay_status_policy,
        require_replay_status=data_config.require_replay_status,
        val_replay_status_policy=data_config.val_replay_status_policy,
        val_require_replay_status=data_config.val_require_replay_status,
        train_fraction=data_config.train_fraction,
        split_seed=data_config.split_seed,
        max_train_episodes=data_config.max_train_episodes,
        max_val_episodes=data_config.max_val_episodes,
    )
    return (
        LeRobotV2VideoWindowDataset(data_config=data_config, episodes=split.train_episodes),
        LeRobotV2VideoWindowDataset(data_config=data_config, episodes=split.val_episodes),
    )


def load_lerobot_v2_video_metadata(repo_root: Path) -> LeRobotVideoMetadata:
    if not repo_root.exists():
        raise FileNotFoundError(f"LeRobot video local_root does not exist: {repo_root}")
    info_path = repo_root / "meta" / "info.json"
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    tasks_path = repo_root / "meta" / "tasks.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = _load_lerobot_video_episodes(episodes_path)
    tasks_by_index = _load_lerobot_video_tasks(tasks_path)
    return LeRobotVideoMetadata(
        repo_root=repo_root,
        codebase_version=str(info.get("codebase_version", "")),
        fps=int(info.get("fps", 0)),
        chunk_size=int(info.get("chunks_size", info.get("chunk_size", 1000))),
        total_episodes=int(info.get("total_episodes", len(episodes))),
        data_path_template=str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")),
        video_path_template=str(
            info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        ),
        features=dict(info.get("features", {})),
        episodes=episodes,
        tasks_by_index=tasks_by_index,
    )


def _load_lerobot_video_episodes(path: Path) -> tuple[LeRobotEpisodeRecord, ...]:
    records: list[LeRobotEpisodeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                LeRobotEpisodeRecord(
                    episode_index=int(payload["episode_index"]),
                    length=int(payload.get("length", payload.get("num_frames", 0))),
                    tasks=tuple(str(task) for task in payload.get("tasks", ())),
                )
            )
    return tuple(records)


def _load_lerobot_video_tasks(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    tasks: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task_index = int(payload.get("task_index", payload.get("index", len(tasks))))
            task = payload.get("task", payload.get("text", payload.get("instruction", "")))
            tasks[task_index] = str(task)
    return tasks
