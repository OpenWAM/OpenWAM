from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ConsortiumMemberConfig,
    LeRobotConsortiumDataConfig,
    ViewLayoutConfig,
)
from open_wam.data import build_lerobot_consortium_train_val_datasets


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _rgb_bytes(height: int, width: int, rgb: tuple[int, int, int]) -> bytes:
    tensor = torch.zeros(height, width, 3, dtype=torch.uint8)
    tensor[..., 0] = rgb[0]
    tensor[..., 1] = rgb[1]
    tensor[..., 2] = rgb[2]
    image = Image.fromarray(tensor.numpy(), mode="RGB")
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_local_lerobot_repo(
    repo_root: Path,
    *,
    channel_specs: dict[str, tuple[int, int, tuple[int, int, int]]],
    fps: int,
    action_dim: int,
    state_dim: int,
    episode_lengths: tuple[int, ...] = (6,),
) -> None:
    _write_json(
        repo_root / "meta" / "info.json",
        {
            "codebase_version": "v2.0",
            "fps": fps,
            "chunks_size": 1000,
            "total_episodes": len(episode_lengths),
            "total_frames": int(sum(episode_lengths)),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                **{
                    channel_name: {"dtype": "image", "shape": [height, width, 3]}
                    for channel_name, (height, width, _) in channel_specs.items()
                },
                "actions": {"dtype": "float32", "shape": [action_dim]},
                "state": {"dtype": "float32", "shape": [state_dim]},
            },
        },
    )
    _write_jsonl(
        repo_root / "meta" / "episodes.jsonl",
        [
            {"episode_index": episode_index, "length": length, "tasks": [f"task {episode_index}"]}
            for episode_index, length in enumerate(episode_lengths)
        ],
    )
    _write_jsonl(
        repo_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "task 0"}],
    )
    for episode_index, length in enumerate(episode_lengths):
        rows: list[dict[str, object]] = []
        for frame_index in range(length):
            row: dict[str, object] = {
                "frame_index": frame_index,
                "task_index": 0,
                "actions": [float(frame_index)] * action_dim,
                "state": [float(frame_index)] * state_dim,
            }
            for channel_name, (height, width, rgb) in channel_specs.items():
                row[channel_name] = _rgb_bytes(height, width, rgb)
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        parquet_path = repo_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)


def _build_slots_config(repo_root: Path) -> LeRobotConsortiumDataConfig:
    return LeRobotConsortiumDataConfig(
        consortium_members=(ConsortiumMemberConfig(member_id="repo", local_root=str(repo_root)),),
        camera_names=("observation.images.slot0", "observation.images.slot1"),
        latent_camera_names=("observation.images.slot0", "observation.images.slot1"),
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=32,
                width=32,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=32,
                left=0,
                height=16,
                width=32,
            ),
        ),
        canonical_height=48,
        canonical_width=32,
        num_frames=2,
        frame_stride=1,
        sample_stride=1,
        train_fraction=1.0,
        action_schema=ActionSchemaConfig(action_dim=6, action_horizon=2, state_dim=8, state_horizon=1),
        action_target=ActionTargetConfig(representation="raw", source_key="actions", pose_source_key="state"),
        view_packing_mode="multicam_as_slots",
        channel_selection_mode="all_available",
    )


def _build_frames_config(repo_root: Path) -> LeRobotConsortiumDataConfig:
    return LeRobotConsortiumDataConfig(
        consortium_members=(ConsortiumMemberConfig(member_id="repo", local_root=str(repo_root)),),
        camera_names=("observation.images.slot0",),
        latent_camera_names=("observation.images.slot0",),
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=32,
                width=32,
            ),
        ),
        canonical_height=32,
        canonical_width=32,
        num_frames=2,
        frame_stride=1,
        sample_stride=1,
        train_fraction=1.0,
        action_schema=ActionSchemaConfig(action_dim=6, action_horizon=2, state_dim=8, state_horizon=1),
        action_target=ActionTargetConfig(representation="raw", source_key="actions", pose_source_key="state"),
        view_packing_mode="multicam_as_frames",
        frame_packing_order="camera_major",
        channel_selection_mode="all_available",
    )


def _print_slots_summary(repo_root: Path) -> None:
    train_dataset, _ = build_lerobot_consortium_train_val_datasets(_build_slots_config(repo_root))
    sample = train_dataset[0]
    print("slots.mode", sample.metadata["view_packing_mode"])
    print("slots.dataset_len", len(train_dataset))
    print("slots.views", sorted(sample.views))
    print("slots.slot0.shape", tuple(sample.views["observation.images.slot0"].shape))
    print("slots.slot1.shape", tuple(sample.views["observation.images.slot1"].shape))
    print("slots.resolved_channels", sample.metadata["resolved_channel_slots"])


def _print_frames_summary(repo_root: Path) -> None:
    train_dataset, _ = build_lerobot_consortium_train_val_datasets(_build_frames_config(repo_root))
    cameras = sorted({window.source_camera_name for window in train_dataset.sample_index})
    print("frames.mode", train_dataset[0].metadata["view_packing_mode"])
    print("frames.dataset_len", len(train_dataset))
    print("frames.cameras", cameras)
    for camera_name in cameras:
        dataset_index = next(
            index for index, window in enumerate(train_dataset.sample_index) if window.source_camera_name == camera_name
        )
        sample = train_dataset[dataset_index]
        print("frames.camera", camera_name)
        print("frames.source_camera_name", sample.metadata["source_camera_name"])
        print("frames.views", sorted(sample.views))
        print("frames.slot0.shape", tuple(sample.views["observation.images.slot0"].shape))
        print("frames.resolved_channels", sample.metadata["resolved_channel_slots"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="open_wam_consortium_modes_") as temp_dir:
        repo_root = Path(temp_dir) / "repo"
        _build_local_lerobot_repo(
            repo_root,
            channel_specs={
                "cam_high": (12, 12, (255, 0, 0)),
                "cam_wrist": (8, 8, (0, 255, 0)),
            },
            fps=30,
            action_dim=4,
            state_dim=6,
            episode_lengths=(6,),
        )
        _print_slots_summary(repo_root)
        _print_frames_summary(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
