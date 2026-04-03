from __future__ import annotations

from collections import Counter
from typing import Any

from open_wam.configs import LeRobotConsortiumDataConfig

from .lerobot_consortium import (
    ConsortiumMemberContract,
    ConsortiumResolvedSplit,
    LeRobotConsortiumWindowDataset,
    build_lerobot_consortium_catalog,
    build_lerobot_consortium_train_val_datasets,
    resolve_lerobot_consortium_train_val_split,
)


def _member_summary(member: ConsortiumMemberContract) -> dict[str, Any]:
    return {
        "member_id": member.member_id,
        "repo_id": member.repo_id,
        "local_root": member.local_root,
        "source_group": member.source_group,
        "observation_fps": member.observation_fps,
        "action_fps": member.action_fps,
        "action_dim": member.action_dim,
        "state_dim": member.state_dim,
        "episode_count": member.total_episodes,
        "total_frames": member.total_frames,
        "visual_channels": [
            {
                "source_name": channel.source_name,
                "dtype": channel.dtype,
                "height": channel.height,
                "width": channel.width,
                "channels": channel.channels,
                "channel_order": channel.channel_order,
            }
            for channel in member.visual_channels
        ],
    }


def _split_member_counts(split: ConsortiumResolvedSplit, *, split_name: str) -> dict[str, int]:
    episodes = split.train_episodes if split_name == "train" else split.val_episodes
    return dict(Counter(episode.member_id for episode in episodes))


def _sample_preview(
    dataset: LeRobotConsortiumWindowDataset,
    *,
    count: int,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for dataset_index in range(min(count, len(dataset))):
        sample = dataset[dataset_index]
        previews.append(
            {
                "dataset_index": dataset_index,
                "member_id": sample.metadata.get("member_id"),
                "repo_id": sample.metadata.get("repo_id"),
                "episode_index": sample.metadata.get("episode_index"),
                "observation_start": sample.metadata.get("observation_start"),
                "source_camera_name": sample.metadata.get("source_camera_name"),
                "observation_frame_indices": sample.metadata.get("observation_frame_indices"),
                "action_frame_indices": sample.metadata.get("action_frame_indices"),
                "resolved_channel_slots": sample.metadata.get("resolved_channel_slots"),
                "task_text": sample.task_text,
                "view_shapes": {
                    view_name: list(view_tensor.shape)
                    for view_name, view_tensor in sample.views.items()
                },
                "actions_shape": list(sample.actions.shape),
                "state_shape": list(sample.state.shape),
            }
        )
    return previews


def build_lerobot_consortium_report(
    data_config: LeRobotConsortiumDataConfig,
    *,
    preview_count: int = 3,
    sampler_preview_count: int = 16,
) -> dict[str, Any]:
    catalog = build_lerobot_consortium_catalog(data_config)
    split = resolve_lerobot_consortium_train_val_split(data_config, catalog)
    train_dataset, val_dataset = build_lerobot_consortium_train_val_datasets(
        data_config,
        catalog=catalog,
        split=split,
    )

    train_epoch0 = train_dataset.build_epoch_index_order(epoch=0)
    return {
        "dataset_type": data_config.dataset_type,
        "dataset_name": data_config.dataset_name,
        "catalog": {
            "member_count": len(catalog.members),
            "members": [_member_summary(member) for member in catalog.members],
        },
        "split_resolution": split.audit_payload,
        "splits": {
            "train": {
                "episode_count": len(split.train_episodes),
                "episode_count_by_member": _split_member_counts(split, split_name="train"),
                "window_count": len(train_dataset),
                "sampler_epoch0_preview": train_epoch0[:sampler_preview_count],
                "sampler_epoch0_member_counts": dict(
                    Counter(train_dataset.sample_index[index].member_id for index in train_epoch0)
                ),
                "sample_previews": _sample_preview(train_dataset, count=preview_count),
            },
            "val": {
                "episode_count": len(split.val_episodes),
                "episode_count_by_member": _split_member_counts(split, split_name="val"),
                "window_count": len(val_dataset),
                "sample_previews": _sample_preview(val_dataset, count=preview_count),
            },
        },
    }


def format_lerobot_consortium_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("LeRobot Consortium Report")
    lines.append(
        f"dataset: {report['dataset_name']} ({report['dataset_type']})"
    )
    lines.append(f"members: {report['catalog']['member_count']}")
    lines.append("")
    lines.append("Members")
    for member in report["catalog"]["members"]:
        visual_channels = ", ".join(channel["source_name"] for channel in member["visual_channels"])
        repo_or_root = member["repo_id"] or member["local_root"] or "<unknown>"
        lines.append(
            f"- {member['member_id']}: {repo_or_root} | "
            f"episodes={member['episode_count']} | "
            f"obs_fps={member['observation_fps']} | "
            f"action_fps={member['action_fps']} | "
            f"channels=[{visual_channels}]"
        )
    lines.append("")
    lines.append("Splits")
    for split_name in ("train", "val"):
        split = report["splits"][split_name]
        lines.append(
            f"- {split_name}: episodes={split['episode_count']} "
            f"windows={split['window_count']} "
            f"by_member={split['episode_count_by_member']}"
        )
        if split_name == "train":
            lines.append(
                f"  epoch0_preview={split['sampler_epoch0_preview']} "
                f"member_counts={split['sampler_epoch0_member_counts']}"
            )
        for preview in split["sample_previews"]:
            lines.append(
                f"  sample[{preview['dataset_index']}]: "
                f"member={preview['member_id']} episode={preview['episode_index']} "
                f"camera={preview['source_camera_name']} "
                f"obs_start={preview['observation_start']} "
                f"obs_frames={preview['observation_frame_indices']} "
                f"action_frames={preview['action_frame_indices']}"
            )
    return "\n".join(lines)
