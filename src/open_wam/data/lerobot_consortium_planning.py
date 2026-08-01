"""Deterministic split, channel, and window planning for LeRobot consortia."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Iterable

from open_wam.configs import (
    ConsortiumChannelSelectionMode,
    ConsortiumFramePackingOrder,
    ConsortiumMissingChannelPolicy,
    ConsortiumSplitMode,
    ConsortiumViewPackingMode,
    LeRobotConsortiumDataConfig,
)

if TYPE_CHECKING:
    from .lerobot_consortium_catalog import (
        ConsortiumCatalog,
        ConsortiumMemberContract,
    )


__all__ = [
    "ConsortiumChannelSelection",
    "ConsortiumEpisodeKey",
    "ConsortiumResolvedSplit",
    "ConsortiumWindowRecord",
    "build_lerobot_consortium_window_index",
    "resolve_lerobot_consortium_train_val_split",
]


@dataclass(frozen=True)
class ConsortiumEpisodeKey:
    member_id: str
    repo_id: str | None
    episode_index: int


@dataclass(frozen=True)
class ConsortiumChannelSelection:
    target_slot: str
    source_name: str | None


@dataclass(frozen=True)
class ConsortiumWindowRecord:
    member_id: str
    repo_id: str | None
    episode_index: int
    observation_start: int
    source_camera_name: str | None
    channel_selections: tuple[ConsortiumChannelSelection, ...]


@dataclass(frozen=True)
class ConsortiumResolvedSplit:
    train_episodes: tuple[ConsortiumEpisodeKey, ...]
    val_episodes: tuple[ConsortiumEpisodeKey, ...]
    audit_payload: dict[str, Any]


def _resolve_member_id(
    *,
    explicit_member_id: str | None,
    repo_id: str | None,
    local_root: str | None,
) -> str:
    if explicit_member_id:
        return explicit_member_id
    if repo_id:
        return repo_id
    if local_root:
        return Path(local_root).expanduser().resolve().name
    raise ValueError(
        "Cannot resolve consortium member id without explicit id, repo_id, or local_root."
    )


def _resolve_channel_selections(
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
) -> tuple[ConsortiumChannelSelection, ...]:
    available_names = [
        channel.source_name for channel in member_contract.visual_channels
    ]
    member_cfg = next(
        (
            member
            for member in data_config.consortium_members
            if _resolve_member_id(
                explicit_member_id=member.member_id,
                repo_id=member.repo_id,
                local_root=member.local_root,
            )
            == member_contract.member_id
        ),
        None,
    )

    if member_cfg is not None and member_cfg.channel_mappings:
        mapping_items = member_cfg.channel_mappings
    else:
        mapping_items = data_config.channel_mappings

    if data_config.view_packing_mode == ConsortiumViewPackingMode.MULTICAM_AS_FRAMES:
        return _resolve_frame_packed_channel_selections(
            data_config=data_config,
            member_contract=member_contract,
            available_names=available_names,
            member_cfg=member_cfg,
            mapping_items=mapping_items,
        )
    return _resolve_slot_packed_channel_selections(
        data_config=data_config,
        member_contract=member_contract,
        available_names=available_names,
        member_cfg=member_cfg,
        mapping_items=mapping_items,
    )


def _resolve_slot_packed_channel_selections(
    *,
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
    available_names: list[str],
    member_cfg: Any,
    mapping_items: tuple[Any, ...],
) -> tuple[ConsortiumChannelSelection, ...]:
    selections: dict[str, str | None] = {
        slot: None for slot in data_config.camera_names
    }

    if (
        data_config.channel_selection_mode
        == ConsortiumChannelSelectionMode.EXPLICIT_MAPPING
    ):
        for mapping in mapping_items:
            if mapping.target_slot not in selections:
                raise ValueError(
                    f"Consortium channel mapping for member '{member_contract.member_id}' targets unknown slot "
                    f"'{mapping.target_slot}'. Known slots: {data_config.camera_names}."
                )
            if mapping.source_name in available_names:
                selections[mapping.target_slot] = mapping.source_name
    elif (
        data_config.channel_selection_mode
        == ConsortiumChannelSelectionMode.REQUIRED_SUBSET
    ):
        required_channels = (
            member_cfg.include_channels
            if member_cfg and member_cfg.include_channels
            else data_config.required_channels
        )
        if not required_channels:
            raise ValueError(
                "`required_subset` channel selection requires non-empty `required_channels`."
            )
        for source_name in required_channels:
            if source_name not in available_names:
                if (
                    data_config.missing_channel_policy
                    == ConsortiumMissingChannelPolicy.ERROR
                ):
                    raise ValueError(
                        f"Member '{member_contract.member_id}' is missing required visual channel '{source_name}'."
                    )
                continue
            if source_name in selections:
                selections[source_name] = source_name
                continue
            raise ValueError(
                "Required consortium channels must either match configured camera_names or use explicit_mapping mode."
            )
    else:
        # All-available mode maps discovered source streams onto the declared
        # canonical slots in order. Missing slots are handled by the policy below.
        for target_slot, source_name in zip(
            data_config.camera_names, available_names
        ):
            selections[target_slot] = source_name

    resolved = tuple(
        ConsortiumChannelSelection(
            target_slot=target_slot,
            source_name=source_name,
        )
        for target_slot, source_name in selections.items()
    )
    if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
        missing = [item.target_slot for item in resolved if item.source_name is None]
        if missing:
            raise ValueError(
                f"Member '{member_contract.member_id}' is missing required consortium slots {missing}. "
                f"Available channels: {available_names}."
            )
    return resolved


def _dedupe_preserve_order(
    values: Iterable[str | None],
) -> tuple[str | None, ...]:
    seen: set[str | None] = set()
    resolved: list[str | None] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        resolved.append(value)
    return tuple(resolved)


def _resolve_frame_packed_channel_selections(
    *,
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
    available_names: list[str],
    member_cfg: Any,
    mapping_items: tuple[Any, ...],
) -> tuple[ConsortiumChannelSelection, ...]:
    target_slot = data_config.camera_names[0]
    source_names: tuple[str | None, ...]

    if data_config.frame_packing_order != ConsortiumFramePackingOrder.CAMERA_MAJOR:
        raise ValueError(
            f"Unsupported consortium frame packing order: {data_config.frame_packing_order}."
        )

    if (
        data_config.channel_selection_mode
        == ConsortiumChannelSelectionMode.EXPLICIT_MAPPING
    ):
        resolved_sources: list[str | None] = []
        for mapping in mapping_items:
            if mapping.target_slot != target_slot:
                raise ValueError(
                    "`view_packing_mode=multicam_as_frames` requires all explicit mappings to target "
                    f"the single configured slot '{target_slot}', got '{mapping.target_slot}'."
                )
            if mapping.source_name in available_names:
                resolved_sources.append(mapping.source_name)
        source_names = _dedupe_preserve_order(resolved_sources)
    elif (
        data_config.channel_selection_mode
        == ConsortiumChannelSelectionMode.REQUIRED_SUBSET
    ):
        required_channels = (
            member_cfg.include_channels
            if member_cfg and member_cfg.include_channels
            else data_config.required_channels
        )
        if not required_channels:
            raise ValueError(
                "`required_subset` channel selection requires non-empty `required_channels`."
            )
        resolved_sources = []
        for source_name in required_channels:
            if source_name not in available_names:
                if (
                    data_config.missing_channel_policy
                    == ConsortiumMissingChannelPolicy.ERROR
                ):
                    raise ValueError(
                        f"Member '{member_contract.member_id}' is missing required visual channel '{source_name}'."
                    )
                continue
            resolved_sources.append(source_name)
        source_names = _dedupe_preserve_order(resolved_sources)
    else:
        source_names = _dedupe_preserve_order(available_names)

    if not source_names:
        if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
            raise ValueError(
                f"Member '{member_contract.member_id}' exposes no usable channels for single-slot frame packing. "
                f"Available channels: {available_names}."
            )
        source_names = (None,)

    return tuple(
        ConsortiumChannelSelection(
            target_slot=target_slot,
            source_name=source_name,
        )
        for source_name in source_names
    )


def _episode_membership_from_manifest(
    manifest_rows: tuple,
    *,
    allowed_member_ids: set[str],
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in manifest_rows:
        if item.member_id not in allowed_member_ids:
            continue
        for episode_index in item.episode_indices:
            keys.add((item.member_id, int(episode_index)))
    return keys


def resolve_lerobot_consortium_train_val_split(
    data_config: LeRobotConsortiumDataConfig,
    catalog: ConsortiumCatalog,
) -> ConsortiumResolvedSplit:
    member_lookup = {member.member_id: member for member in catalog.members}
    allowed_member_ids = set(member_lookup)
    all_episode_keys = [
        ConsortiumEpisodeKey(
            member_id=member.member_id,
            repo_id=member.repo_id,
            episode_index=episode.episode_index,
        )
        for member in catalog.members
        for episode in member.episodes
    ]

    if data_config.split_mode == ConsortiumSplitMode.EXPLICIT_MANIFEST:
        train_membership = _episode_membership_from_manifest(
            data_config.explicit_train_episodes,
            allowed_member_ids=allowed_member_ids,
        )
        val_membership = _episode_membership_from_manifest(
            data_config.explicit_val_episodes,
            allowed_member_ids=allowed_member_ids,
        )
        train_keys = [
            key
            for key in all_episode_keys
            if (key.member_id, key.episode_index) in train_membership
        ]
        val_keys = [
            key
            for key in all_episode_keys
            if (key.member_id, key.episode_index) in val_membership
        ]
    elif data_config.split_mode == ConsortiumSplitMode.HASH_BY_EPISODE:
        train_keys = []
        val_keys = []
        for key in all_episode_keys:
            token = (
                f"{data_config.split_seed}:{key.member_id}:{key.episode_index}"
            ).encode("utf-8")
            score = int(hashlib.sha256(token).hexdigest()[:16], 16) / float(
                0xFFFFFFFFFFFFFFFF
            )
            if score < data_config.train_fraction:
                train_keys.append(key)
            else:
                val_keys.append(key)
    else:
        shuffled = list(all_episode_keys)
        rng = random.Random(data_config.split_seed)
        rng.shuffle(shuffled)
        train_count = int(len(shuffled) * data_config.train_fraction)
        train_count = (
            min(max(train_count, 1), len(shuffled)) if shuffled else 0
        )
        train_keys = shuffled[:train_count]
        val_keys = shuffled[train_count:]

    train_keys = sorted(
        train_keys,
        key=lambda item: (item.member_id, item.episode_index),
    )
    val_keys = sorted(
        val_keys,
        key=lambda item: (item.member_id, item.episode_index),
    )

    if data_config.max_train_episodes is not None:
        train_keys = train_keys[: data_config.max_train_episodes]
    if data_config.max_val_episodes is not None:
        val_keys = val_keys[: data_config.max_val_episodes]
    if not val_keys and train_keys:
        val_keys = train_keys[:1]

    audit_payload = {
        "split_mode": data_config.split_mode,
        "split_seed": data_config.split_seed,
        "train_fraction": data_config.train_fraction,
        "train_episode_keys": [
            {
                "member_id": key.member_id,
                "repo_id": key.repo_id,
                "episode_index": key.episode_index,
            }
            for key in train_keys
        ],
        "val_episode_keys": [
            {
                "member_id": key.member_id,
                "repo_id": key.repo_id,
                "episode_index": key.episode_index,
            }
            for key in val_keys
        ],
    }
    return ConsortiumResolvedSplit(
        train_episodes=tuple(train_keys),
        val_episodes=tuple(val_keys),
        audit_payload=audit_payload,
    )


def build_lerobot_consortium_window_index(
    data_config: LeRobotConsortiumDataConfig,
    catalog: ConsortiumCatalog,
    episode_keys: Iterable[ConsortiumEpisodeKey],
) -> tuple[ConsortiumWindowRecord, ...]:
    member_lookup = {member.member_id: member for member in catalog.members}
    window_records: list[ConsortiumWindowRecord] = []
    required_span = (
        (data_config.num_frames - 1) * data_config.frame_stride
        + data_config.action_schema.action_horizon
    )
    for episode_key in episode_keys:
        member = member_lookup[episode_key.member_id]
        episode_record = next(
            episode
            for episode in member.episodes
            if episode.episode_index == episode_key.episode_index
        )
        max_start = episode_record.length - required_span
        if max_start < 0:
            continue
        channel_selections = _resolve_channel_selections(data_config, member)
        if (
            data_config.view_packing_mode
            == ConsortiumViewPackingMode.MULTICAM_AS_FRAMES
        ):
            for selection in channel_selections:
                for start in range(
                    0, max_start + 1, data_config.sample_stride
                ):
                    window_records.append(
                        ConsortiumWindowRecord(
                            member_id=member.member_id,
                            repo_id=member.repo_id,
                            episode_index=episode_key.episode_index,
                            observation_start=start,
                            source_camera_name=selection.source_name,
                            channel_selections=(selection,),
                        )
                    )
        else:
            for start in range(0, max_start + 1, data_config.sample_stride):
                window_records.append(
                    ConsortiumWindowRecord(
                        member_id=member.member_id,
                        repo_id=member.repo_id,
                        episode_index=episode_key.episode_index,
                        observation_start=start,
                        source_camera_name=None,
                        channel_selections=channel_selections,
                    )
                )
    return tuple(window_records)
