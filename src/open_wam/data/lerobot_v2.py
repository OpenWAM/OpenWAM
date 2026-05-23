from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from open_wam.configs import ActionTargetReferenceSource, ActionTargetRepresentation, DataConfig, GripperRepresentation

from .action_transforms import (
    build_absolute_joint_position_targets,
    build_relative_pose_targets,
    expected_joint_position_target_dim,
    expected_pose_target_dim,
    normalize_action_targets,
)
from .action_mapping import (
    action_mapping_is_active,
    apply_action_mapping,
    resolve_action_source_dim,
)
from .contracts import WAMSample
from .replay_status import load_replay_status_records, split_episode_indices_by_replay_status


def _resolve_row_key(row: dict[str, Any], key: str) -> str:
    if key in row:
        return key
    if key.endswith("s") and key[:-1] in row:
        return key[:-1]
    singular_candidate = f"{key}s"
    if singular_candidate in row:
        return singular_candidate
    raise KeyError(key)


@dataclass(frozen=True)
class LeRobotEpisodeRecord:
    """Episode metadata loaded from `meta/episodes.jsonl`."""

    episode_index: int
    length: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class LeRobotV2Metadata:
    """Minimal metadata needed to index a LeRobot-v2 dataset repo."""

    repo_id: str
    codebase_version: str
    fps: int
    chunk_size: int
    total_episodes: int
    data_path_template: str
    features: dict[str, dict[str, Any]]
    episodes: tuple[LeRobotEpisodeRecord, ...]
    tasks_by_index: dict[int, str]


@dataclass(frozen=True)
class EpisodeWindow:
    """One training window over an episode.

    `observation_start` indexes the first video frame in the observation chunk.
    The action horizon starts at the anchor frame `observation_start + num_frames - 1`.
    """

    episode_index: int
    observation_start: int


class LeRobotV2WindowDataset(Dataset[WAMSample]):
    """Windowed reader for LeRobot-v2 episode-parquet datasets.

    This adapter reads directly from the repo's `meta/*.json*` and
    `data/chunk-*/episode_*.parquet` files. That is intentional: the installed
    `lerobot` package in this environment rejects `physical-intelligence/libero`
    due to a codebase-version compatibility guard, while the dataset repo itself
    already contains a stable self-describing schema.
    """

    def __init__(
        self,
        data_config: DataConfig,
        episodes: list[int],
    ) -> None:
        if data_config.repo_id is None:
            raise ValueError("LeRobot-v2 datasets require `data.repo_id` in the experiment config.")

        self.data_config = data_config
        self.metadata = load_lerobot_v2_metadata(
            repo_id=data_config.repo_id,
            cache_dir=data_config.cache_dir,
        )
        self.episodes = tuple(episodes)
        self.episode_records = {episode.episode_index: episode for episode in self.metadata.episodes}
        self.sample_index = self._build_sample_index()
        self._episode_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

        if not self.sample_index:
            raise ValueError(
                "No valid LeRobot-v2 windows were constructed. "
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

        # Observation rows are sparse-sampled from the episode according to the
        # configured frame stride. These are the only frames seen by the shared
        # video backbone for this sample.
        observation_rows = [
            rows[window.observation_start + offset * frame_stride]
            for offset in range(num_frames)
        ]
        anchor_frame_index = window.observation_start + (num_frames - 1) * frame_stride

        # Action supervision starts at the anchor frame, i.e. the last observed
        # frame. This makes the data contract agnostic to head placement:
        # action heads may consume video tokens in different ways, but they all
        # receive targets aligned to the same policy anchor.
        action_rows = rows[anchor_frame_index : anchor_frame_index + action_horizon]
        target_state_rows = rows[anchor_frame_index : anchor_frame_index + action_horizon]

        # State history is anchored on the last observed frame. This keeps the
        # sample semantics stable across head variants: the shared backbone owns
        # the observation chunk, while heads see state aligned to the current
        # policy anchor rather than to every intermediate frame.
        state_start = max(0, anchor_frame_index - state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]

        views = {
            view_name: self._decode_image_sequence(observation_rows, view_name)
            for view_name in self.data_config.camera_names
        }
        actions, action_mask, action_target_metadata = self._build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=state_horizon,
            left_pad=True,
        )

        episode = self.episode_records[window.episode_index]
        task_index = int(observation_rows[-1]["task_index"])
        task_text = self.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode.tasks:
            task_text = episode.tasks[0]

        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=task_text,
            metadata={
                "repo_id": self.metadata.repo_id,
                "episode_index": window.episode_index,
                "task_index": task_index,
                "observation_start": window.observation_start,
                "anchor_frame_index": anchor_frame_index,
                "observation_frame_indices": [int(row["frame_index"]) for row in observation_rows],
                "action_frame_indices": [int(row["frame_index"]) for row in action_rows],
                "target_state_frame_indices": [int(row["frame_index"]) for row in target_state_rows],
                "state_source_key": state_source_key,
                "action_representation": self.data_config.action_target.representation,
                **action_target_metadata,
            },
        )

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build one action target tensor under the configured representation.

        The output always follows the common WAM contract `[H_action, D_action]`
        even when the supervision source is dataset state rather than the raw
        dataset action tensor.
        """

        action_target = self.data_config.action_target
        action_mapping = self.data_config.action_mapping
        target_dim = self.data_config.action_schema.action_dim
        target_length = self.data_config.action_schema.action_horizon

        if action_target.representation == ActionTargetRepresentation.RAW:
            source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
            actions, action_mask = self._extract_sequence(
                rows=action_rows,
                key=action_target.source_key,
                target_dim=source_dim,
                target_length=target_length,
            )
            actions = normalize_action_targets(
                actions,
                normalization=action_target.normalization,
            )
            mapped = apply_action_mapping(
                actions,
                action_mask,
                action_mapping,
                target_dim=target_dim,
            )
            metadata = dict(mapped.metadata)
            metadata["action_target_normalization_mode"] = str(action_target.normalization.mode)
            return mapped.actions, mapped.action_mask, metadata

        if action_target.representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
            if action_target.reference_source != ActionTargetReferenceSource.ANCHOR_STATE:
                raise ValueError(
                    "LeRobot-v2 reference-relative EEF targets currently support only "
                    f"`reference_source=anchor_state`, got {action_target.reference_source}."
                )
            pose_source = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.pose_source_key)], dtype=torch.float32)
                    for row in target_state_rows
                ],
                dim=0,
            )
            raw_action_sequence = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in action_rows
                ],
                dim=0,
            )
            relative_targets, relative_mask, metadata = build_relative_pose_targets(
                pose_source,
                state_encoding=action_target.state_encoding,
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                raw_action_sequence=raw_action_sequence,
                gripper_action_index=action_target.gripper_action_index,
            )
            expected_dim = expected_pose_target_dim(
                rotation_representation=action_target.rotation_representation,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            target_or_source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
            if target_or_source_dim != expected_dim:
                raise ValueError(
                    "Configured action_dim does not match the derived pose-target dimension: "
                    f"configured_dim={target_or_source_dim}, expected={expected_dim} for "
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
            actions, action_mask = self._pack_sequence(
                sequence=relative_targets,
                target_dim=target_or_source_dim,
                target_length=target_length,
            )
            if relative_mask.shape[-1] != relative_targets.shape[-1]:
                raise ValueError("Relative target mask shape must match the relative target tensor shape.")
            action_mask[:, : relative_mask.shape[-1]] = relative_mask
            mapped = apply_action_mapping(
                actions,
                action_mask,
                action_mapping,
                target_dim=target_dim,
            )
            metadata.update(mapped.metadata)
            metadata["action_mapping_applied"] = action_mapping_is_active(action_mapping)
            return mapped.actions, mapped.action_mask, metadata

        if action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION:
            joint_position_source = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.joint_position_source_key)], dtype=torch.float32)
                    for row in target_state_rows
                ],
                dim=0,
            )
            raw_action_sequence = torch.stack(
                [
                    torch.tensor(row[_resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in action_rows
                ],
                dim=0,
            )
            gripper_position_sequence = None
            if (
                action_target.include_gripper
                and action_target.gripper_representation != GripperRepresentation.ACTION_COMMAND
            ):
                gripper_position_sequence = torch.stack(
                    [
                        torch.tensor(
                            row[_resolve_row_key(row, action_target.gripper_position_source_key)],
                            dtype=torch.float32,
                        )
                        for row in target_state_rows
                    ],
                    dim=0,
                )
            joint_targets, joint_mask, metadata = build_absolute_joint_position_targets(
                joint_position_source,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                gripper_position_sequence=gripper_position_sequence,
                raw_action_sequence=raw_action_sequence,
                gripper_action_index=action_target.gripper_action_index,
                normalization=action_target.joint_position_normalization,
            )
            expected_dim = expected_joint_position_target_dim(
                joint_dim=joint_position_source.shape[-1],
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            target_or_source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
            if target_or_source_dim != expected_dim:
                raise ValueError(
                    "Configured action_dim does not match the derived absolute-joint target dimension: "
                    f"configured_dim={target_or_source_dim}, expected={expected_dim}."
                )
            metadata.update(
                {
                    "joint_position_source_key": action_target.joint_position_source_key,
                    "gripper_source_key": action_target.source_key,
                }
            )
            actions, action_mask = self._pack_sequence(
                sequence=joint_targets,
                target_dim=target_or_source_dim,
                target_length=target_length,
                sequence_name="absolute_joint_position_targets",
            )
            if joint_mask.shape[-1] != joint_targets.shape[-1]:
                raise ValueError("Absolute-joint target mask shape must match the target tensor shape.")
            action_mask[:, : joint_mask.shape[-1]] = joint_mask
            mapped = apply_action_mapping(
                actions,
                action_mask,
                action_mapping,
                target_dim=target_dim,
            )
            metadata.update(mapped.metadata)
            metadata["action_mapping_applied"] = action_mapping_is_active(action_mapping)
            return mapped.actions, mapped.action_mask, metadata

        raise ValueError(f"Unsupported action target representation: {action_target.representation}")

    def _build_sample_index(self) -> list[EpisodeWindow]:
        num_frames = self.data_config.num_frames
        frame_stride = self.data_config.frame_stride
        action_horizon = self.data_config.action_schema.action_horizon
        sample_stride = self.data_config.sample_stride

        # A valid window needs:
        # - `num_frames` observations sampled with `frame_stride`
        # - `action_horizon` actions starting at the last observed frame
        #
        # So the last required frame index is:
        #   start + (num_frames - 1) * frame_stride + action_horizon - 1
        # and this must stay inside the episode.
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

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        # Episode-level caching keeps repeated window sampling cheap without
        # forcing the entire dataset into memory. The cache size is config-driven
        # because different collaborators may prefer different memory / network
        # tradeoffs depending on the dataset and machine.
        path = hf_hub_download(
            repo_id=self.metadata.repo_id,
            filename=self._episode_file_path(episode_index),
            repo_type="dataset",
            cache_dir=self.data_config.cache_dir,
        )
        rows = pq.read_table(path).to_pylist()
        self._episode_cache[episode_index] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows

    def _episode_file_path(self, episode_index: int) -> str:
        return self.metadata.data_path_template.format(
            episode_chunk=episode_index // self.metadata.chunk_size,
            episode_index=episode_index,
        )

    def _decode_image_sequence(
        self,
        rows: list[dict[str, Any]],
        key: str,
    ) -> torch.Tensor:
        frames = [self._decode_image(row[key]["bytes"]) for row in rows]
        return torch.stack(frames, dim=0)

    def _decode_image(self, image_bytes: bytes) -> torch.Tensor:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            # Keep uint8 `[H, W, 3]` here. The canonicalizer is responsible for
            # turning raw RGB into the backbone-ready `[B, 3, T, H, W]` float path.
            tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return tensor.reshape(rgb.height, rgb.width, 3)

    def _extract_sequence(
        self,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError(f"Cannot extract sequence for key '{key}' from an empty row slice.")

        sequence = torch.stack(
            [torch.tensor(row[_resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return self._pack_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
        )

    def _pack_sequence(
        self,
        *,
        sequence: torch.Tensor,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
        sequence_name: str = "sequence",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 2:
            raise ValueError(
                f"Expected {sequence_name} tensor with shape [T, D], got {tuple(sequence.shape)}."
            )

        raw_dim = sequence.shape[-1]
        if raw_dim > target_dim:
            raise ValueError(f"Raw {sequence_name} dim {raw_dim} exceeds configured target dim {target_dim}.")

        output = torch.zeros(target_length, target_dim, dtype=torch.float32)
        mask = torch.zeros(target_length, target_dim, dtype=torch.float32)

        # Left-padding is used for state history so short prefixes near the
        # start of an episode still align to the most recent timestep. Actions
        # keep left_pad=False because they are future-facing targets.
        if left_pad:
            start_index = target_length - len(sequence)
        else:
            start_index = 0

        for index, values in enumerate(sequence):
            output[start_index + index, : raw_dim] = values
            mask[start_index + index, : raw_dim] = 1.0

        return output, mask


def build_lerobot_train_val_episode_split(data_config: DataConfig) -> tuple[list[int], list[int]]:
    """Deterministically split a LeRobot-v2 repo into train/val episodes.

    The repository-level split is often just `train`, so validation is a local
    concern for this framework rather than something delegated to the dataset.
    """

    if data_config.repo_id is None:
        raise ValueError("LeRobot-v2 datasets require `data.repo_id` in the experiment config.")

    metadata = load_lerobot_v2_metadata(repo_id=data_config.repo_id, cache_dir=data_config.cache_dir)
    replay_status_records, replay_status_path = load_replay_status_records(
        None,
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


def load_lerobot_v2_metadata(
    repo_id: str,
    cache_dir: str | None = None,
) -> LeRobotV2Metadata:
    """Load the self-describing metadata files shipped with a LeRobot-v2 repo."""

    info = _read_json_from_hub(repo_id, "meta/info.json", cache_dir=cache_dir)
    episodes = _read_jsonl_from_hub(repo_id, "meta/episodes.jsonl", cache_dir=cache_dir)
    tasks = _read_jsonl_from_hub(repo_id, "meta/tasks.jsonl", cache_dir=cache_dir)

    return LeRobotV2Metadata(
        repo_id=repo_id,
        codebase_version=str(info["codebase_version"]),
        fps=int(info["fps"]),
        chunk_size=int(info["chunks_size"]),
        total_episodes=int(info["total_episodes"]),
        data_path_template=str(info["data_path"]),
        features={name: dict(feature) for name, feature in info["features"].items()},
        episodes=tuple(
            LeRobotEpisodeRecord(
                episode_index=int(record["episode_index"]),
                length=int(record["length"]),
                tasks=tuple(record.get("tasks", [])),
            )
            for record in episodes
        ),
        tasks_by_index={
            int(record["task_index"]): str(record["task"])
            for record in tasks
        },
    )


def _read_json_from_hub(repo_id: str, filename: str, cache_dir: str | None = None) -> dict[str, Any]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_from_hub(repo_id: str, filename: str, cache_dir: str | None = None) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
