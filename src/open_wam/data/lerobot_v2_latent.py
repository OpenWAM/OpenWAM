from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from einops import rearrange
from torch.utils.data import Dataset

from open_wam.configs import ActionTargetReferenceSource, ActionTargetRepresentation, DataConfig

from .action_transforms import build_relative_pose_targets, expected_pose_target_dim
from .latent_contracts import LatentWAMSample
from .lerobot_v2 import LeRobotEpisodeRecord, LeRobotV2Metadata, _resolve_row_key


_LATENT_FILE_PATTERN = re.compile(r"episode_(?P<episode>\d{6})_(?P<start>\d+)_(?P<end>\d+)\.pth$")


@dataclass(frozen=True)
class LocalEpisodeWindow:
    """One latent window over one local episode file."""

    repo_root: Path
    episode_index: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class LocalRepoBundle:
    """Metadata and episode lookup for one discovered local repo."""

    root: Path
    metadata: LeRobotV2Metadata
    episodes_by_index: dict[int, LeRobotEpisodeRecord]


class LocalLeRobotLatentWindowDataset(Dataset[LatentWAMSample]):
    """Latent-first local-repo dataset for LingBot-style post-training exports."""

    def __init__(self, data_config: DataConfig, windows: list[LocalEpisodeWindow]) -> None:
        if data_config.local_root is None:
            raise ValueError("Local latent datasets require `data.local_root` in the experiment config.")
        self.data_config = data_config
        self.windows = list(windows)
        self.empty_text_embedding = self._load_empty_text_embedding()
        self._repo_bundles = {
            str(bundle.root): bundle
            for bundle in discover_local_lerobot_repo_bundles(data_config.local_root)
        }
        self._episode_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()

        if not self.windows:
            raise ValueError(
                "No valid latent windows were constructed. "
                f"Check local_root={data_config.local_root!r} and latent_camera_names={data_config.latent_camera_names!r}."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.windows[index]
        repo_bundle = self._repo_bundles[str(window.repo_root)]
        rows = self._load_episode_rows(window.repo_root, window.episode_index, repo_bundle.metadata)
        latent_payloads = self._load_window_latents(window, repo_bundle.metadata)
        video_latents, latent_layout_metadata = self._assemble_canonical_latents(latent_payloads)

        primary_payload = latent_payloads[self.data_config.latent_camera_names[0]]
        observed_frame_ids = [int(value) for value in list(primary_payload.get("frame_ids", []))]
        if not observed_frame_ids:
            observed_frame_ids = list(range(window.start_frame, window.end_frame))
        anchor_frame_index = observed_frame_ids[-1]

        actions, action_mask, action_target_metadata = self._build_lingbot_window_action_targets(
            rows=rows,
            window=window,
            observed_frame_ids=observed_frame_ids,
            latent_num_frames=int(video_latents.shape[1]),
        )
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
        )

        text_context = primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = self.empty_text_embedding.clone() if self.empty_text_embedding is not None else None

        episode_record = repo_bundle.episodes_by_index.get(window.episode_index)
        task_index = int(rows[min(anchor_frame_index, len(rows) - 1)].get("task_index", 0)) if rows else 0
        task_text = repo_bundle.metadata.tasks_by_index.get(task_index)
        if task_text is None and episode_record is not None and episode_record.tasks:
            task_text = episode_record.tasks[0]

        return LatentWAMSample(
            video_latents=video_latents,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            metadata={
                "repo_root": str(window.repo_root),
                "episode_index": window.episode_index,
                "window_start_frame": window.start_frame,
                "window_end_frame": window.end_frame,
                "anchor_frame_index": anchor_frame_index,
                "observed_frame_ids": observed_frame_ids,
                "task_index": task_index,
                "latent_layout": latent_layout_metadata,
                "state_source_key": state_source_key,
                "action_representation": self.data_config.action_target.representation,
                **action_target_metadata,
            },
        )

    def _load_empty_text_embedding(self) -> torch.Tensor | None:
        configured_path = self.data_config.empty_text_embedding_path
        candidate_path = (
            Path(configured_path)
            if configured_path is not None
            else Path(self.data_config.local_root) / "empty_emb.pt"
        )
        if not candidate_path.exists():
            if configured_path is not None:
                raise FileNotFoundError(
                    "Configured `data.empty_text_embedding_path` does not exist: "
                    f"{candidate_path}"
                )
            return None
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                "Expected `empty_text_embedding_path` to point at a tensor checkpoint, "
                f"got {type(payload)!r} from {candidate_path}"
            )
        if payload.ndim == 3 and payload.shape[0] == 1:
            payload = payload.squeeze(0)
        return payload.to(dtype=torch.float32).contiguous()

    def _load_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> dict[str, dict[str, Any]]:
        latent_root = resolve_latent_root(window.repo_root, self.data_config)
        chunk_dir = latent_root / f"chunk-{window.episode_index // metadata.chunk_size:03d}"
        payloads: dict[str, dict[str, Any]] = {}
        for camera_name in self.data_config.latent_camera_names:
            latent_path = chunk_dir / camera_name / latent_filename(
                episode_index=window.episode_index,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
            )
            payload = torch.load(latent_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected latent payload mapping at {latent_path}, got {type(payload).__name__}.")
            payloads[camera_name] = payload
        return payloads

    def _assemble_canonical_latents(
        self,
        latent_payloads: dict[str, dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, dict[str, int]]]:
        canonical_latents = None
        metadata: dict[str, dict[str, int]] = {}
        for view_layout, camera_name in zip(self.data_config.view_layout, self.data_config.latent_camera_names, strict=True):
            payload = latent_payloads[camera_name]
            view_latents = reshape_latent_payload(payload)
            latent_height = int(view_latents.shape[1])
            latent_width = int(view_latents.shape[2])
            stride_h = max(1, view_layout.height // latent_height)
            stride_w = max(1, view_layout.width // latent_width)
            top = view_layout.top // stride_h
            left = view_layout.left // stride_w
            full_height = self.data_config.canonical_height // stride_h
            full_width = self.data_config.canonical_width // stride_w

            if canonical_latents is None:
                frames = int(view_latents.shape[0])
                channels = int(view_latents.shape[-1])
                canonical_latents = torch.zeros(
                    frames,
                    full_height,
                    full_width,
                    channels,
                    dtype=view_latents.dtype,
                )
            canonical_latents[:, top : top + latent_height, left : left + latent_width, :] = view_latents
            metadata[camera_name] = {
                "latent_height": latent_height,
                "latent_width": latent_width,
                "top": top,
                "left": left,
            }

        if canonical_latents is None:
            raise ValueError("Expected at least one latent camera payload.")
        return canonical_latents.permute(3, 0, 1, 2).contiguous().to(dtype=torch.float32), metadata

    def _build_lingbot_window_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        window: LocalEpisodeWindow,
        observed_frame_ids: list[int],
        latent_num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        if action_target.representation != ActionTargetRepresentation.RAW:
            raise ValueError(
                "Long-window local latent datasets currently support only `action_target.representation=raw` "
                "for LingBot-compatible exact training."
            )
        if latent_num_frames <= 0:
            raise ValueError("Expected at least one latent frame in the local latent window.")
        if not observed_frame_ids:
            raise ValueError("Expected non-empty frame_ids metadata for the local latent window.")

        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = frame_stride * int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
        required_action_num = latent_num_frames * prefix_actions

        action_start_offset = max(0, int(observed_frame_ids[0] - window.start_frame))
        raw_window_rows = rows[window.start_frame : window.end_frame]
        raw_actions = torch.stack(
            [
                torch.tensor(row[_resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                for row in raw_window_rows
            ],
            dim=0,
        )
        raw_actions = raw_actions[action_start_offset:]
        action_dim = raw_actions.shape[-1] if raw_actions.numel() > 0 else self.data_config.action_schema.action_dim
        if raw_actions.shape[-1] != self.data_config.action_schema.action_dim:
            raise ValueError(
                "Configured action_dim does not match raw local latent supervision: "
                f"configured={self.data_config.action_schema.action_dim}, raw={raw_actions.shape[-1]}."
            )

        padded_actions = torch.cat(
            [
                torch.zeros(prefix_actions, action_dim, dtype=torch.float32),
                raw_actions,
            ],
            dim=0,
        )
        if padded_actions.shape[0] < required_action_num:
            padded_actions = torch.cat(
                [
                    padded_actions,
                    torch.zeros(required_action_num - padded_actions.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
        actions = padded_actions[:required_action_num].contiguous()

        action_mask = torch.ones_like(actions, dtype=torch.float32)
        if raw_actions.shape[0] + prefix_actions < required_action_num:
            action_mask[raw_actions.shape[0] + prefix_actions :] = 0.0
        return actions, action_mask, {
            "lingbot_window_action_alignment": {
                "latent_num_frames": latent_num_frames,
                "raw_frame_count": len(observed_frame_ids),
                "frame_stride": frame_stride,
                "prefix_actions": prefix_actions,
                "required_action_num": required_action_num,
                "action_start_offset": action_start_offset,
            }
        }

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        target_dim = self.data_config.action_schema.action_dim
        target_length = self.data_config.action_schema.action_horizon

        if action_target.representation == ActionTargetRepresentation.RAW:
            actions, action_mask = self._extract_sequence(
                rows=action_rows,
                key=action_target.source_key,
                target_dim=target_dim,
                target_length=target_length,
            )
            return actions, action_mask, {}

        if action_target.representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
            if action_target.reference_source != ActionTargetReferenceSource.ANCHOR_STATE:
                raise ValueError(
                    "Local latent LeRobot datasets currently support only "
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
            actions, action_mask = self._pack_sequence(
                sequence=relative_targets,
                target_dim=target_dim,
                target_length=target_length,
            )
            if relative_mask.shape[-1] != relative_targets.shape[-1]:
                raise ValueError("Relative target mask shape must match the relative target tensor shape.")
            action_mask[:, : relative_mask.shape[-1]] = relative_mask
            return actions, action_mask, metadata

        raise ValueError(f"Unsupported action target representation: {action_target.representation}")

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
            return (
                torch.zeros(target_length, target_dim, dtype=torch.float32),
                torch.zeros(target_length, target_dim, dtype=torch.float32),
            )
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
        clipped = sequence[:target_length]
        start_index = target_length - len(clipped) if left_pad else 0
        for index, values in enumerate(clipped):
            output[start_index + index, : raw_dim] = values
            mask[start_index + index, : raw_dim] = 1.0
        return output, mask

    def _load_episode_rows(
        self,
        repo_root: Path,
        episode_index: int,
        metadata: LeRobotV2Metadata,
    ) -> list[dict[str, Any]]:
        cache_key = (str(repo_root), episode_index)
        if cache_key in self._episode_cache:
            self._episode_cache.move_to_end(cache_key)
            return self._episode_cache[cache_key]

        path = repo_root / metadata.data_path_template.format(
            episode_chunk=episode_index // metadata.chunk_size,
            episode_index=episode_index,
        )
        rows = pq.read_table(path).to_pylist()
        self._episode_cache[cache_key] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows


def build_local_lerobot_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    bundles = discover_local_lerobot_repo_bundles(data_config.local_root or "")
    train_windows: list[LocalEpisodeWindow] = []
    val_windows: list[LocalEpisodeWindow] = []
    for bundle in bundles:
        repo_windows = scan_local_latent_windows(bundle.root, data_config)
        repo_episodes = [episode.episode_index for episode in bundle.metadata.episodes]
        train_episodes, repo_val_episodes = split_local_episode_indices(
            episode_indices=repo_episodes,
            train_fraction=data_config.train_fraction,
            split_seed=data_config.split_seed,
            max_train_episodes=data_config.max_train_episodes,
            max_val_episodes=data_config.max_val_episodes,
        )
        train_episode_set = set(train_episodes)
        val_episode_set = set(repo_val_episodes)
        repo_train_windows = [window for window in repo_windows if window.episode_index in train_episode_set]
        repo_val_windows = [window for window in repo_windows if window.episode_index in val_episode_set]
        if not repo_val_windows and repo_train_windows:
            repo_val_windows = repo_train_windows[:1]
        train_windows.extend(repo_train_windows)
        val_windows.extend(repo_val_windows)

    return (
        LocalLeRobotLatentWindowDataset(data_config=data_config, windows=train_windows),
        LocalLeRobotLatentWindowDataset(data_config=data_config, windows=val_windows),
    )


def discover_local_lerobot_repo_bundles(local_root: str | Path) -> list[LocalRepoBundle]:
    """Discover one or more local LeRobot-style repo roots."""

    root = Path(local_root).expanduser()
    repo_roots: list[Path] = []
    if (root / "meta" / "info.json").exists():
        repo_roots.append(root)
    else:
        repo_roots.extend(sorted(path.parent.parent for path in root.rglob("meta/info.json")))
    if not repo_roots:
        raise FileNotFoundError(f"No local LeRobot repo roots were discovered under {root}.")

    bundles: list[LocalRepoBundle] = []
    for repo_root in repo_roots:
        metadata = load_lerobot_v2_local_metadata(repo_root)
        bundles.append(
            LocalRepoBundle(
                root=repo_root,
                metadata=metadata,
                episodes_by_index={episode.episode_index: episode for episode in metadata.episodes},
            )
        )
    return bundles


def scan_local_latent_windows(repo_root: Path, data_config: DataConfig) -> list[LocalEpisodeWindow]:
    """Scan a local latent export tree into reusable latent windows."""

    primary_camera = data_config.latent_camera_names[0]
    windows: list[LocalEpisodeWindow] = []
    for camera_dir in sorted((path for path in resolve_latent_root(repo_root, data_config).glob(f"chunk-*/{primary_camera}") if path.is_dir())):
        for latent_file in sorted(camera_dir.glob("episode_*.pth")):
            match = _LATENT_FILE_PATTERN.match(latent_file.name)
            if match is None:
                continue
            windows.append(
                LocalEpisodeWindow(
                    repo_root=repo_root,
                    episode_index=int(match.group("episode")),
                    start_frame=int(match.group("start")),
                    end_frame=int(match.group("end")),
                )
            )
    return windows


def split_local_episode_indices(
    *,
    episode_indices: list[int],
    train_fraction: float,
    split_seed: int,
    max_train_episodes: int | None,
    max_val_episodes: int | None,
) -> tuple[list[int], list[int]]:
    shuffled = list(episode_indices)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_fraction)
    train_count = min(max(train_count, 1), len(shuffled))
    train_episodes = shuffled[:train_count]
    val_episodes = shuffled[train_count:]
    if max_train_episodes is not None:
        train_episodes = train_episodes[:max_train_episodes]
    if max_val_episodes is not None:
        val_episodes = val_episodes[:max_val_episodes]
    if not val_episodes and train_episodes:
        val_episodes = train_episodes[:1]
    return train_episodes, val_episodes


def load_lerobot_v2_local_metadata(repo_root: Path) -> LeRobotV2Metadata:
    """Load the self-describing metadata files from one local LeRobot-style repo."""

    info = read_json_local(repo_root / "meta" / "info.json")
    episodes = read_jsonl_local(repo_root / "meta" / "episodes.jsonl")
    tasks = read_jsonl_local(repo_root / "meta" / "tasks.jsonl")
    return LeRobotV2Metadata(
        repo_id=str(repo_root),
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
        tasks_by_index={int(record["task_index"]): str(record["task"]) for record in tasks},
    )


def resolve_latent_root(repo_root: Path, data_config: DataConfig) -> Path:
    if data_config.latent_root is None:
        return repo_root / data_config.latent_subdir
    configured = Path(data_config.latent_root).expanduser()
    if configured.is_absolute():
        return configured
    return (repo_root / configured).resolve()


def latent_filename(*, episode_index: int, start_frame: int, end_frame: int) -> str:
    return f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"


def read_json_local(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_local(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reshape_latent_payload(payload: dict[str, Any]) -> torch.Tensor:
    latent = payload["latent"]
    if not isinstance(latent, torch.Tensor):
        latent = torch.tensor(latent)
    latent_num_frames = int(payload["latent_num_frames"])
    latent_height = int(payload["latent_height"])
    latent_width = int(payload["latent_width"])
    if latent.ndim == 2:
        return rearrange(
            latent,
            "(f h w) c -> f h w c",
            f=latent_num_frames,
            h=latent_height,
            w=latent_width,
        )
    if latent.ndim == 4:
        if tuple(latent.shape[:3]) != (latent_num_frames, latent_height, latent_width):
            raise ValueError(
                "Latent payload shape does not match metadata. "
                f"shape={tuple(latent.shape)}, expected=({latent_num_frames}, {latent_height}, {latent_width}, C)."
            )
        return latent
    raise ValueError(
        "Unsupported latent payload shape. "
        f"Expected flattened `[F*H*W, C]` or `[F, H, W, C]`, got {tuple(latent.shape)}."
    )
