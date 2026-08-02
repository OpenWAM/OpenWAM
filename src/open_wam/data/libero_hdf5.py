from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset

from open_wam.configs import ActionTargetReferenceSource, ActionTargetRepresentation, DataConfig

from .action_normalization import normalize_action_targets
from .action_target_builders import build_relative_pose_targets, expected_pose_target_dim
from .contracts import WAMSample
from .replay_status import load_replay_status_records, split_episode_indices_by_replay_status
from .sequence_packing import pack_temporal_sequence


_LIBERO_LOCAL_VIEW_KEY_BY_NAME = {
    "image": "agentview_rgb",
    "wrist_image": "eye_in_hand_rgb",
}


@dataclass(frozen=True)
class LiberoOfflineEpisodeRecord:
    """One demo inside the local LIBERO HDF5 benchmark files."""

    episode_index: int
    file_path: str
    demo_key: str
    task_text: str
    length: int


@dataclass(frozen=True)
class LiberoOfflineMetadata:
    """Minimal index metadata for a local LIBERO HDF5 directory."""

    local_root: str
    episodes: tuple[LiberoOfflineEpisodeRecord, ...]


@dataclass(frozen=True)
class EpisodeWindow:
    """One training window over one local LIBERO demo."""

    episode_index: int
    observation_start: int


class LiberoOfflineWindowDataset(Dataset[WAMSample]):
    """Windowed reader for local LIBERO HDF5 demos.

    This path is for original offline benchmark files such as
    `<dataset-root>/libero_10/*.hdf5`. It preserves the same public view/action
    contract as the LeRobot-backed LIBERO adapter:

    - `image` <- `obs/agentview_rgb`
    - `wrist_image` <- `obs/eye_in_hand_rgb`
    - raw action <- `actions`
    - pose state <- `[ee_pos, ee_ori, gripper_states]`
    """

    def __init__(
        self,
        data_config: DataConfig,
        episodes: list[int],
    ) -> None:
        if data_config.local_root is None:
            raise ValueError("Local LIBERO HDF5 datasets require `data.local_root` in the experiment config.")

        self.data_config = data_config
        self.metadata = load_libero_offline_metadata(data_config.local_root)
        self.episodes = tuple(episodes)
        self.episode_records = {episode.episode_index: episode for episode in self.metadata.episodes}
        self.sample_index = self._build_sample_index()
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        if not self.sample_index:
            raise ValueError(
                "No valid local LIBERO HDF5 windows were constructed. "
                f"Check num_frames={data_config.num_frames}, "
                f"action_horizon={data_config.action_schema.action_horizon}, "
                f"and selected episodes={len(episodes)}."
            )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        episode = self._load_episode(window.episode_index)
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        state_horizon = self.data_config.action_schema.state_horizon

        observation_indices = torch.tensor(
            [
                window.observation_start + offset * frame_stride
                for offset in range(num_frames)
            ],
            dtype=torch.long,
        )
        anchor_frame_index = int(observation_indices[-1].item())
        action_rows = episode["actions"][anchor_frame_index : anchor_frame_index + action_horizon]
        target_state_rows = episode["state"][anchor_frame_index : anchor_frame_index + action_horizon]
        state_start = max(0, anchor_frame_index - state_horizon + 1)
        state_rows = episode["state"][state_start : anchor_frame_index + 1]

        views = {
            view_name: episode["views"][view_name].index_select(0, observation_indices)
            for view_name in self.data_config.camera_names
        }
        actions, action_mask, action_target_metadata = self._build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        state, state_mask = pack_temporal_sequence(
            sequence=state_rows,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=state_horizon,
            left_pad=True,
            sequence_name="state",
        )

        record = self.episode_records[window.episode_index]
        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=record.task_text,
            metadata={
                "dataset_source": "libero_hdf5",
                "local_root": self.metadata.local_root,
                "source_file": record.file_path,
                "demo_key": record.demo_key,
                "episode_index": window.episode_index,
                "observation_start": window.observation_start,
                "anchor_frame_index": anchor_frame_index,
                "observation_frame_indices": observation_indices.tolist(),
                "action_frame_indices": list(range(anchor_frame_index, anchor_frame_index + len(action_rows))),
                "target_state_frame_indices": list(range(anchor_frame_index, anchor_frame_index + len(target_state_rows))),
                "action_representation": self.data_config.action_target.representation,
                **action_target_metadata,
            },
        )

    def _build_action_targets(
        self,
        *,
        action_rows: torch.Tensor,
        target_state_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        target_dim = self.data_config.action_schema.action_dim
        target_length = self.data_config.action_schema.action_horizon

        if action_target.representation == ActionTargetRepresentation.RAW:
            actions, action_mask = pack_temporal_sequence(
                sequence=action_rows,
                target_dim=target_dim,
                target_length=target_length,
                sequence_name=action_target.source_key,
            )
            actions = normalize_action_targets(
                actions,
                normalization=action_target.normalization,
            )
            return actions, action_mask, {"action_target_normalization_mode": str(action_target.normalization.mode)}

        if action_target.representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
            if action_target.reference_source != ActionTargetReferenceSource.ANCHOR_STATE:
                raise ValueError(
                    "Local LIBERO HDF5 reference-relative EEF targets currently support only "
                    f"`reference_source=anchor_state`, got {action_target.reference_source}."
                )
            relative_targets, relative_mask, metadata = build_relative_pose_targets(
                target_state_rows.to(dtype=torch.float32),
                state_encoding=action_target.state_encoding,
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                raw_action_sequence=action_rows.to(dtype=torch.float32),
                gripper_action_index=action_target.gripper_action_index,
            )
            expected_dim = expected_pose_target_dim(
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            if target_dim != expected_dim:
                raise ValueError(
                    "Configured action_dim does not match the derived pose-target dimension: "
                    f"action_dim={target_dim}, expected={expected_dim} for "
                    f"[rotation_representation={action_target.rotation_representation}, "
                    f"gripper_representation={action_target.gripper_representation}]."
                )
            metadata.update(
                {
                    "reference_source": action_target.reference_source,
                    "pose_source_key": action_target.pose_source_key,
                    "gripper_source_key": action_target.source_key,
                }
            )
            actions, action_mask = pack_temporal_sequence(
                sequence=relative_targets,
                target_dim=target_dim,
                target_length=target_length,
                sequence_name="relative_pose_targets",
            )
            if relative_mask.shape != actions.shape:
                raise ValueError(
                    "Relative target mask shape must match the padded action target tensor shape, "
                    f"got mask={tuple(relative_mask.shape)} and actions={tuple(actions.shape)}."
                )
            return actions, relative_mask.to(dtype=torch.float32), metadata

        raise ValueError(f"Unsupported action target representation: {action_target.representation}")

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

    def _load_episode(self, episode_index: int) -> dict[str, Any]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        record = self.episode_records[episode_index]
        with h5py.File(record.file_path, "r") as handle:
            demo_group = handle["data"][record.demo_key]
            obs_group = demo_group["obs"]
            state = torch.cat(
                [
                    torch.from_numpy(obs_group["ee_pos"][...]).to(dtype=torch.float32),
                    torch.from_numpy(obs_group["ee_ori"][...]).to(dtype=torch.float32),
                    torch.from_numpy(obs_group["gripper_states"][...]).to(dtype=torch.float32),
                ],
                dim=-1,
            )
            episode = {
                "views": {
                    view_name: torch.from_numpy(obs_group[source_key][...]).to(dtype=torch.uint8)
                    for view_name, source_key in _resolve_local_view_keys(self.data_config.camera_names).items()
                },
                "actions": torch.from_numpy(demo_group["actions"][...]).to(dtype=torch.float32),
                "state": state,
            }

        self._episode_cache[episode_index] = episode
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode


def build_libero_offline_train_val_episode_split(data_config: DataConfig) -> tuple[list[int], list[int]]:
    """Deterministically split a local LIBERO HDF5 directory into train/val episodes."""

    if data_config.local_root is None:
        raise ValueError("Local LIBERO HDF5 datasets require `data.local_root` in the experiment config.")

    metadata = load_libero_offline_metadata(data_config.local_root)
    replay_status_records, replay_status_path = load_replay_status_records(
        data_config.local_root,
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
    return split.train_episodes, split.val_episodes


@lru_cache(maxsize=8)
def load_libero_offline_metadata(local_root: str) -> LiberoOfflineMetadata:
    """Scan one local LIBERO HDF5 directory into a stable demo index."""

    root = Path(local_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Local LIBERO dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Local LIBERO dataset root must be a directory, got: {root}")

    files = sorted(root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No `.hdf5` LIBERO demos were found under {root}")

    episodes: list[LiberoOfflineEpisodeRecord] = []
    for file_path in files:
        task_text = _infer_task_text_from_file(file_path)
        with h5py.File(file_path, "r") as handle:
            for demo_key in _sorted_demo_keys(handle["data"].keys()):
                demo_group = handle["data"][demo_key]
                length = int(demo_group["actions"].shape[0])
                episodes.append(
                    LiberoOfflineEpisodeRecord(
                        episode_index=len(episodes),
                        file_path=str(file_path),
                        demo_key=demo_key,
                        task_text=task_text,
                        length=length,
                    )
                )

    return LiberoOfflineMetadata(local_root=str(root), episodes=tuple(episodes))


def _resolve_local_view_keys(camera_names: tuple[str, ...]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for view_name in camera_names:
        try:
            resolved[view_name] = _LIBERO_LOCAL_VIEW_KEY_BY_NAME[view_name]
        except KeyError as exc:
            supported = ", ".join(sorted(_LIBERO_LOCAL_VIEW_KEY_BY_NAME))
            raise ValueError(
                f"Unsupported local LIBERO view '{view_name}'. Supported canonical view names: {supported}."
            ) from exc
    return resolved


def _sorted_demo_keys(demo_keys: Any) -> list[str]:
    return sorted(
        (str(key) for key in demo_keys),
        key=lambda value: int(value.split("_")[-1]) if value.startswith("demo_") else value,
    )


def _infer_task_text_from_file(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    match = re.match(r"^[A-Z_]+_SCENE\d+_(.+)$", stem)
    if match is not None:
        stem = match.group(1)
    return stem.replace("_", " ")
