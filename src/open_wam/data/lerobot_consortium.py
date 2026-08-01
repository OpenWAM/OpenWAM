from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from open_wam.configs import (
    ConsortiumMissingChannelPolicy,
    DataConfig,
    LeRobotConsortiumDataConfig,
)
from open_wam.configs.enums import serialize_enum_values

from .contracts import WAMSample
from .distributed_sampling import UnpaddedEpochOrderDistributedSampler
from .lerobot_consortium_catalog import (
    ConsortiumCatalog as ConsortiumCatalog,
    ConsortiumEpisodeRecord as ConsortiumEpisodeRecord,
    ConsortiumMemberContract as ConsortiumMemberContract,
    ConsortiumVisualChannelContract as ConsortiumVisualChannelContract,
    build_lerobot_consortium_catalog as build_lerobot_consortium_catalog,
    validate_lerobot_consortium_index_snapshot
    as validate_lerobot_consortium_index_snapshot,
)
from .lerobot_consortium_planning import (
    ConsortiumChannelSelection as ConsortiumChannelSelection,
    ConsortiumEpisodeKey as ConsortiumEpisodeKey,
    ConsortiumResolvedSplit as ConsortiumResolvedSplit,
    ConsortiumWindowRecord as ConsortiumWindowRecord,
    _resolve_member_id,
    build_lerobot_consortium_window_index
    as build_lerobot_consortium_window_index,
    resolve_lerobot_consortium_train_val_split
    as resolve_lerobot_consortium_train_val_split,
)
from .lerobot_consortium_sampling import ConsortiumEpochOrderPlan
from .lerobot_consortium_storage import (
    CloudConsortiumCache as CloudConsortiumCache,
    ConsortiumSourceResolver as ConsortiumSourceResolver,
    ConsortiumSourceSpec as ConsortiumSourceSpec,
    LocalConsortiumCache as LocalConsortiumCache,
    NoopConsortiumCache as NoopConsortiumCache,
    discover_local_lerobot_consortium_members
    as discover_local_lerobot_consortium_members,
)
from .row_action_targets import build_row_action_targets, resolve_row_key
from .sequence_packing import pack_temporal_sequence


_CONSORTIUM_CATALOG_COMPATIBILITY_EXPORTS = (
    ConsortiumCatalog,
    ConsortiumEpisodeRecord,
    ConsortiumMemberContract,
    ConsortiumVisualChannelContract,
    build_lerobot_consortium_catalog,
    validate_lerobot_consortium_index_snapshot,
)

_CONSORTIUM_PLANNING_COMPATIBILITY_EXPORTS = (
    ConsortiumChannelSelection,
    ConsortiumEpisodeKey,
    ConsortiumResolvedSplit,
    ConsortiumWindowRecord,
    build_lerobot_consortium_window_index,
    resolve_lerobot_consortium_train_val_split,
)

_CONSORTIUM_STORAGE_COMPATIBILITY_EXPORTS = (
    CloudConsortiumCache,
    ConsortiumSourceResolver,
    ConsortiumSourceSpec,
    LocalConsortiumCache,
    NoopConsortiumCache,
    discover_local_lerobot_consortium_members,
)


class ConsortiumTrainSampler(UnpaddedEpochOrderDistributedSampler):
    """Deterministic train sampler for consortium datasets."""

    def __init__(
        self,
        dataset: "LeRobotConsortiumWindowDataset",
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=None,
        )


class LeRobotConsortiumWindowDataset(Dataset[WAMSample]):
    """Windowed reader for a configurable consortium of LeRobot-format datasets."""

    def __init__(
        self,
        *,
        data_config: LeRobotConsortiumDataConfig,
        catalog: ConsortiumCatalog,
        window_index: tuple[ConsortiumWindowRecord, ...],
        split_name: str,
        split_audit_payload: dict[str, Any],
    ) -> None:
        self.data_config = data_config
        self.catalog = catalog
        self.sample_index = list(window_index)
        self.split_name = split_name
        self.split_audit_payload = split_audit_payload
        self._resolver = ConsortiumSourceResolver(data_config)
        self._member_lookup = {member.member_id: member for member in catalog.members}
        self._episode_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._member_sample_indices: dict[str, tuple[int, ...]] = defaultdict(tuple)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.sample_index):
            grouped_indices[record.member_id].append(index)
        self._member_sample_indices = {
            member_id: tuple(indices)
            for member_id, indices in grouped_indices.items()
        }
        self._epoch_order_plan = ConsortiumEpochOrderPlan.from_member_indices(
            member_indices=self._member_sample_indices,
            member_weights={
                _resolve_member_id(
                    explicit_member_id=member.member_id,
                    repo_id=member.repo_id,
                    local_root=member.local_root,
                ): (
                    member.sampling_weight
                    if member.sampling_weight is not None
                    else 1.0
                )
                for member in self.data_config.consortium_members
                if member.enabled
            },
            random_mode=self.data_config.random_mode,
            weight_mode=self.data_config.weight_mode,
            sampling_seed=self.data_config.sampling_seed,
        )
        self.audit_payload = self._build_audit_payload()

    def __len__(self) -> int:
        return len(self.sample_index)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> ConsortiumTrainSampler:
        return ConsortiumTrainSampler(self, world_size=world_size, rank=rank)

    @property
    def epoch_order_plan(self) -> ConsortiumEpochOrderPlan:
        """Return the immutable source-balancing plan for this split."""

        return self._epoch_order_plan

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        return list(self._epoch_order_plan.build_epoch_index_order(epoch=epoch))

    def write_audit_artifacts(self, output_dir: str | Path) -> None:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{self.split_name}_audit.json").write_text(
            json.dumps(self.audit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _build_audit_payload(self) -> dict[str, Any]:
        member_lookup = {member.member_id: member for member in self.catalog.members}
        channel_mappings: dict[str, list[dict[str, Any]]] = {}
        for record in self.sample_index:
            if record.member_id in channel_mappings:
                continue
            channel_mappings[record.member_id] = [asdict(item) for item in record.channel_selections]
        return {
            "dataset_type": self.data_config.dataset_type,
            "split": self.split_name,
            "config": serialize_enum_values(self.data_config),
            "members": [
                {
                    "member_id": member.member_id,
                    "repo_id": member.repo_id,
                    "local_root": member.local_root,
                    "source_group": member.source_group,
                    "observation_fps": member.observation_fps,
                    "action_fps": member.action_fps,
                    "visual_channels": [asdict(channel) for channel in member.visual_channels],
                    "window_count": len(self._member_sample_indices.get(member.member_id, ())),
                    "resolved_channel_mappings": channel_mappings.get(member.member_id, []),
                }
                for member in member_lookup.values()
            ],
            "split_resolution": self.split_audit_payload,
            "sampling": {
                "random_mode": self.data_config.random_mode,
                "weight_mode": self.data_config.weight_mode,
                "sampling_seed": self.data_config.sampling_seed,
                "epoch0_order": self.build_epoch_index_order(epoch=0),
            },
        }

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        member = self._member_lookup[window.member_id]
        rows = self._load_episode_rows(member, episode_index=window.episode_index)

        observation_rows = [
            rows[window.observation_start + offset * self.data_config.frame_stride]
            for offset in range(self.data_config.num_frames)
        ]
        anchor_frame_index = window.observation_start + (self.data_config.num_frames - 1) * self.data_config.frame_stride
        action_rows = rows[anchor_frame_index : anchor_frame_index + self.data_config.action_schema.action_horizon]
        target_state_rows = rows[anchor_frame_index : anchor_frame_index + self.data_config.action_schema.action_horizon]
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]

        views = {
            selection.target_slot: self._build_view_sequence(
                observation_rows=observation_rows,
                selection=selection,
            )
            for selection in window.channel_selections
        }
        actions, action_mask, action_metadata = self._build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
        )

        task_index = int(observation_rows[-1].get("task_index", 0))
        task_text = member.tasks_by_index.get(task_index)
        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=task_text,
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "member_id": member.member_id,
                "repo_id": member.repo_id,
                "local_root": member.local_root,
                "source_group": member.source_group,
                "episode_index": window.episode_index,
                "observation_start": window.observation_start,
                "source_camera_name": window.source_camera_name,
                "anchor_frame_index": anchor_frame_index,
                "observation_frame_indices": [int(row["frame_index"]) for row in observation_rows],
                "action_frame_indices": [int(row["frame_index"]) for row in action_rows],
                "target_state_frame_indices": [int(row["frame_index"]) for row in target_state_rows],
                "observation_fps": member.observation_fps,
                "action_fps": member.action_fps,
                "view_packing_mode": self.data_config.view_packing_mode,
                "frame_packing_order": self.data_config.frame_packing_order,
                "resolved_channel_slots": {
                    selection.target_slot: selection.source_name
                    for selection in window.channel_selections
                },
                "state_source_key": state_source_key,
                "action_representation": self.data_config.action_target.representation,
                **action_metadata,
            },
        )

    def _load_episode_rows(self, member: ConsortiumMemberContract, *, episode_index: int) -> list[dict[str, Any]]:
        cache_key = (member.member_id, episode_index)
        if cache_key in self._episode_cache:
            self._episode_cache.move_to_end(cache_key)
            return self._episode_cache[cache_key]
        source = ConsortiumSourceSpec(member_id=member.member_id, repo_id=member.repo_id, local_root=member.local_root)
        relative_path = member.data_path_template.format(
            episode_chunk=episode_index // member.chunk_size,
            episode_index=episode_index,
        )
        rows = self._resolver.read_parquet_rows(source=source, relative_path=relative_path)
        self._episode_cache[cache_key] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows

    def _build_view_sequence(
        self,
        *,
        observation_rows: list[dict[str, Any]],
        selection: ConsortiumChannelSelection,
    ) -> torch.Tensor:
        if selection.source_name is None:
            if self.data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
                raise KeyError(f"Missing required consortium slot '{selection.target_slot}'.")
            placement = next(
                (view for view in self.data_config.view_layout if view.source_name == selection.target_slot),
                None,
            )
            if placement is None:
                raise ValueError(
                    f"Missing view layout configuration for consortium slot '{selection.target_slot}'."
                )
            return torch.zeros(
                self.data_config.num_frames,
                placement.height,
                placement.width,
                3,
                dtype=torch.uint8,
            )
        frames = [self._decode_image_value(row[selection.source_name]) for row in observation_rows]
        return torch.stack(frames, dim=0)

    def _decode_image_value(self, raw_value: Any) -> torch.Tensor:
        image_bytes: bytes | None = None
        if isinstance(raw_value, dict) and "bytes" in raw_value:
            image_bytes = raw_value["bytes"]
        elif isinstance(raw_value, (bytes, bytearray)):
            image_bytes = bytes(raw_value)
        if image_bytes is None:
            raise ValueError("Expected LeRobot consortium image value to provide inline image bytes.")
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return tensor.reshape(rgb.height, rgb.width, 3)

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return build_row_action_targets(
            data_config=self.data_config,
            action_rows=action_rows,
            target_state_rows=target_state_rows,
            extract_sequence=self._extract_sequence,
            reference_source_subject="Consortium relative pose targets",
            relative_sequence_name="relative_pose_targets",
            include_pose_dimension_context=False,
        )

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
            [torch.tensor(row[resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return pack_temporal_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
        )


def build_lerobot_consortium_train_val_datasets(
    data_config: DataConfig,
    *,
    catalog: ConsortiumCatalog | None = None,
    split: ConsortiumResolvedSplit | None = None,
) -> tuple[LeRobotConsortiumWindowDataset, LeRobotConsortiumWindowDataset]:
    if not isinstance(data_config, LeRobotConsortiumDataConfig):
        raise TypeError("Consortium dataset builder requires LeRobotConsortiumDataConfig.")
    resolved_catalog = catalog or build_lerobot_consortium_catalog(data_config)
    resolved_split = split or resolve_lerobot_consortium_train_val_split(data_config, resolved_catalog)
    train_index = build_lerobot_consortium_window_index(data_config, resolved_catalog, resolved_split.train_episodes)
    val_index = build_lerobot_consortium_window_index(data_config, resolved_catalog, resolved_split.val_episodes)
    return (
        LeRobotConsortiumWindowDataset(
            data_config=data_config,
            catalog=resolved_catalog,
            window_index=train_index,
            split_name="train",
            split_audit_payload=resolved_split.audit_payload,
        ),
        LeRobotConsortiumWindowDataset(
            data_config=data_config,
            catalog=resolved_catalog,
            window_index=val_index,
            split_name="val",
            split_audit_payload=resolved_split.audit_payload,
        ),
    )
