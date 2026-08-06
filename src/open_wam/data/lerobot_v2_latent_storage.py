"""LeRobot-v2 local latent repository discovery and payload I/O."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from einops import rearrange
import pyarrow.parquet as pq
import torch

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import DataConfig
from open_wam.utils.latent_filenames import match_latent_window_filename

from .latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
)
from .lerobot_v2 import LeRobotEpisodeRecord, LeRobotV2Metadata


__all__ = [
    "LocalEpisodeWindow",
    "LocalLatentRepository",
    "LocalRepoBundle",
    "assemble_canonical_latents",
    "condition_latent_offset_mismatches",
    "discover_local_lerobot_repo_bundles",
    "latent_filename",
    "load_empty_text_embedding",
    "load_lerobot_v2_local_metadata",
    "read_json_local",
    "read_jsonl_local",
    "reshape_latent_payload",
    "resolve_latent_root",
    "scan_local_latent_windows",
    "split_local_episode_indices",
]


@dataclass(frozen=True)
class LocalEpisodeWindow:
    """One latent window over one local episode file."""

    repo_root: Path
    episode_index: int
    start_frame: int
    end_frame: int
    observed_frame_ids: tuple[int, ...] = ()
    latent_frame_count: int | None = None

    @property
    def observation_start(self) -> int:
        if self.observed_frame_ids:
            return int(self.observed_frame_ids[0])
        return int(self.start_frame)

    @property
    def observation_frame_indices(self) -> tuple[int, ...]:
        if self.observed_frame_ids:
            return tuple(int(value) for value in self.observed_frame_ids)
        return tuple(range(int(self.start_frame), int(self.end_frame)))

    @property
    def latent_num_frames(self) -> int:
        if self.latent_frame_count is not None:
            return int(self.latent_frame_count)
        return len(self.observation_frame_indices)


@dataclass(frozen=True)
class LocalRepoBundle:
    """Metadata and episode lookup for one discovered local repo."""

    root: Path
    metadata: LeRobotV2Metadata
    episodes_by_index: dict[int, LeRobotEpisodeRecord]


CanonicalLatentWindowPayload = tuple[
    torch.Tensor,
    dict[str, dict[str, int]],
    dict[str, Any],
    torch.Tensor | None,
    dict[str, dict[str, int]],
]


class LocalLatentRepository:
    """Cached repository access for one local latent dataset configuration."""

    def __init__(self, data_config: DataConfig) -> None:
        if data_config.local_root is None:
            raise ValueError(
                "Local latent repositories require `data.local_root` in the experiment config."
            )
        self.data_config = data_config
        repo_roots = [data_config.local_root]
        if data_config.val_local_root and data_config.val_local_root not in repo_roots:
            repo_roots.append(data_config.val_local_root)
        self.repo_bundles = {
            str(bundle.root): bundle
            for repo_root in repo_roots
            for bundle in discover_local_lerobot_repo_bundles(repo_root)
        }
        self.episode_cache: OrderedDict[
            tuple[str, int],
            list[dict[str, Any]],
        ] = OrderedDict()
        self.latent_view_cache: OrderedDict[
            tuple[str, int, int, int],
            CanonicalLatentWindowPayload,
        ] = OrderedDict()

    def load_episode_rows(
        self,
        repo_root: Path,
        episode_index: int,
        metadata: LeRobotV2Metadata,
    ) -> list[dict[str, Any]]:
        cache_key = (str(repo_root), episode_index)
        if cache_key in self.episode_cache:
            self.episode_cache.move_to_end(cache_key)
            return self.episode_cache[cache_key]

        path = repo_root / metadata.data_path_template.format(
            episode_chunk=episode_index // metadata.chunk_size,
            episode_index=episode_index,
        )
        rows = pq.read_table(path).to_pylist()
        self.episode_cache[cache_key] = rows
        while len(self.episode_cache) > self.data_config.episode_cache_size:
            self.episode_cache.popitem(last=False)
        return rows

    def load_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> dict[str, dict[str, Any]]:
        latent_root = resolve_latent_root(window.repo_root, self.data_config)
        chunk_dir = (
            latent_root
            / f"chunk-{window.episode_index // metadata.chunk_size:03d}"
        )
        payloads: dict[str, dict[str, Any]] = {}
        for camera_name in self.data_config.latent_camera_names:
            latent_path = chunk_dir / camera_name / latent_filename(
                episode_index=window.episode_index,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
            )
            payload = load_tensor_artifact(latent_path)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected latent payload mapping at {latent_path}, "
                    f"got {type(payload).__name__}."
                )
            payloads[camera_name] = payload
        return payloads

    def load_canonical_window_latents(
        self,
        window: LocalEpisodeWindow,
        metadata: LeRobotV2Metadata,
    ) -> CanonicalLatentWindowPayload:
        cache_key = (
            str(window.repo_root),
            window.episode_index,
            window.start_frame,
            window.end_frame,
        )
        if cache_key in self.latent_view_cache:
            self.latent_view_cache.move_to_end(cache_key)
            return self.latent_view_cache[cache_key]

        latent_payloads = self.load_window_latents(window, metadata)
        video_latents, latent_layout_metadata = assemble_canonical_latents(
            self.data_config,
            latent_payloads,
        )
        assert video_latents is not None
        condition_latents, condition_layout_metadata = assemble_canonical_latents(
            self.data_config,
            latent_payloads,
            payload_key="condition_latent",
            require_payload_key=False,
        )
        if condition_latents is not None:
            expected_offset = int(
                self.data_config.sample_construction.condition_source_frame_offset
            )
            mismatches = condition_latent_offset_mismatches(
                self.data_config,
                latent_payloads,
                expected_offset=expected_offset,
            )
            if mismatches:
                if expected_offset == 0:
                    condition_latents = None
                    condition_layout_metadata = {}
                else:
                    preview = "; ".join(mismatches[:4])
                    raise ValueError(
                        "Latent payload condition_source_frame_offset/policy does not match "
                        f"`sample_construction.condition_source_frame_offset={expected_offset}`. "
                        "Re-run scripts/augment_lerobot_latents_with_single_frame_condition.py "
                        f"with --source-frame-offset {expected_offset} --overwrite. "
                        f"Mismatches: {preview}"
                    )
        primary_payload = dict(
            latent_payloads[self.data_config.latent_camera_names[0]]
        )
        payload = (
            video_latents,
            latent_layout_metadata,
            primary_payload,
            condition_latents,
            condition_layout_metadata,
        )
        self.latent_view_cache[cache_key] = payload
        while len(self.latent_view_cache) > max(
            1,
            int(self.data_config.episode_cache_size),
        ):
            self.latent_view_cache.popitem(last=False)
        return payload


def load_empty_text_embedding(data_config: DataConfig) -> torch.Tensor | None:
    """Load the configured or repository-local empty-text embedding."""

    configured_path = data_config.empty_text_embedding_path
    if configured_path is not None:
        configured_candidate = Path(configured_path)
        if not configured_candidate.exists():
            raise FileNotFoundError(
                "Configured `data.empty_text_embedding_path` does not exist: "
                f"{configured_candidate}"
            )
        candidate_path = configured_candidate
    else:
        if data_config.local_root is None:
            raise ValueError(
                "Local latent repositories require `data.local_root` to resolve `empty_emb.pt`."
            )
        candidate_path = Path(data_config.local_root) / "empty_emb.pt"
        if not candidate_path.exists():
            return None
    payload = load_tensor_artifact(candidate_path)
    if not isinstance(payload, torch.Tensor):
        raise TypeError(
            "Expected `empty_text_embedding_path` to point at a tensor checkpoint, "
            f"got {type(payload)!r} from {candidate_path}"
        )
    if payload.ndim == 3 and payload.shape[0] == 1:
        payload = payload.squeeze(0)
    return payload.to(dtype=torch.float32).contiguous()


def assemble_canonical_latents(
    data_config: DataConfig,
    latent_payloads: dict[str, dict[str, Any]],
    *,
    payload_key: str = "latent",
    require_payload_key: bool = True,
) -> tuple[torch.Tensor | None, dict[str, dict[str, int]]]:
    """Place per-camera payloads into the configured canonical latent canvas."""

    canonical_latents = None
    metadata: dict[str, dict[str, int]] = {}
    for view_layout, camera_name in zip(
        data_config.view_layout,
        data_config.latent_camera_names,
        strict=True,
    ):
        payload = latent_payloads[camera_name]
        if payload_key not in payload:
            if require_payload_key:
                raise KeyError(
                    f"Expected key {payload_key!r} in latent payload for camera "
                    f"{camera_name!r}."
                )
            return None, {}
        view_latents = reshape_latent_payload(payload, payload_key=payload_key)
        latent_height = int(view_latents.shape[1])
        latent_width = int(view_latents.shape[2])
        stride_h = max(1, view_layout.height // latent_height)
        stride_w = max(1, view_layout.width // latent_width)
        top = view_layout.top // stride_h
        left = view_layout.left // stride_w
        full_height = data_config.canonical_height // stride_h
        full_width = data_config.canonical_width // stride_w

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
        canonical_latents[
            :,
            top : top + latent_height,
            left : left + latent_width,
            :,
        ] = view_latents
        metadata[camera_name] = {
            "latent_height": latent_height,
            "latent_width": latent_width,
            "top": top,
            "left": left,
        }

    if canonical_latents is None:
        raise ValueError("Expected at least one latent camera payload.")
    return (
        canonical_latents.permute(3, 0, 1, 2)
        .contiguous()
        .to(dtype=torch.float32),
        metadata,
    )


def condition_latent_offset_mismatches(
    data_config: DataConfig,
    latent_payloads: dict[str, dict[str, Any]],
    *,
    expected_offset: int,
) -> list[str]:
    """Report payloads that violate the configured condition-frame contract."""

    mismatches: list[str] = []
    for camera_name in data_config.latent_camera_names:
        payload = latent_payloads[camera_name]
        if "condition_latent" not in payload:
            continue
        payload_offset = payload.get("condition_source_frame_offset")
        if payload_offset is None:
            if int(expected_offset) == 0:
                # Legacy optional condition payloads predate explicit source
                # metadata; only the unshifted contract can consume them.
                continue
            mismatches.append(
                f"{camera_name}: missing condition_source_frame_offset"
            )
            continue
        if int(payload_offset) != int(expected_offset):
            mismatches.append(
                f"{camera_name}: payload={int(payload_offset)} "
                f"expected={int(expected_offset)}"
            )
            continue
        payload_policy = payload.get("condition_source_frame_policy")
        if payload_policy is None:
            if int(expected_offset) == 0:
                continue
            mismatches.append(
                f"{camera_name}: missing condition_source_frame_policy"
            )
            continue
        if (
            payload_policy
            != CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET
        ):
            mismatches.append(
                f"{camera_name}: condition_source_frame_policy={payload_policy!r} "
                f"expected {CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET!r}"
            )
    return mismatches


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
    latent_root = resolve_latent_root(repo_root, data_config)
    for camera_dir in sorted((path for path in latent_root.glob(f"chunk-*/{primary_camera}") if path.is_dir())):
        chunk_dir = camera_dir.parent
        for latent_file in sorted(camera_dir.glob("episode_*.pth")):
            match = match_latent_window_filename(latent_file.name)
            if match is None:
                continue
            if any(
                not (chunk_dir / camera_name / latent_file.name).is_file()
                for camera_name in data_config.latent_camera_names[1:]
            ):
                continue
            payload = load_tensor_artifact(latent_file)
            observed_frame_ids: tuple[int, ...] = ()
            latent_frame_count: int | None = None
            if isinstance(payload, dict):
                raw_latent_num_frames = payload.get("latent_num_frames")
                if raw_latent_num_frames is not None:
                    latent_frame_count = int(raw_latent_num_frames)
                raw_frame_ids = payload.get("frame_ids")
                if isinstance(raw_frame_ids, torch.Tensor):
                    observed_frame_ids = tuple(int(value) for value in raw_frame_ids.flatten().tolist())
                elif isinstance(raw_frame_ids, (list, tuple)):
                    observed_frame_ids = tuple(int(value) for value in raw_frame_ids)
            windows.append(
                LocalEpisodeWindow(
                    repo_root=repo_root,
                    episode_index=int(match.group("episode")),
                    start_frame=int(match.group("start")),
                    end_frame=int(match.group("end")),
                    observed_frame_ids=observed_frame_ids,
                    latent_frame_count=latent_frame_count,
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


def reshape_latent_payload(payload: dict[str, Any], *, payload_key: str = "latent") -> torch.Tensor:
    latent = payload[payload_key]
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
