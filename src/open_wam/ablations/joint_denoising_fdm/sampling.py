from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import random
from typing import Any

from torch.utils.data import Dataset

from open_wam.configs import DataConfig, ReplayStatusPolicy
from open_wam.data import LatentWAMSample
from open_wam.data.lerobot_v2_latent import (
    FullSegmentLocalLeRobotLatentDataset,
    LocalEpisodeWindow,
    discover_local_lerobot_repo_bundles,
    resolve_latent_root,
)
from open_wam.data.replay_status import filter_episode_indices_by_replay_status, load_replay_status_records
from open_wam.utils.latent_filenames import match_latent_window_filename

from .types import FdmStartPolicy, FdmWindowSelection


def build_fdm_eval_dataset(
    data_config: DataConfig,
    *,
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL,
) -> Dataset[LatentWAMSample]:
    """Build a full-segment latent dataset using a fast filename-only scan."""

    eval_data = replace(
        data_config,
        train_fraction=1.0,
        max_train_episodes=None,
        max_val_episodes=None,
        replay_status_policy=replay_status_policy,
        require_replay_status=False if replay_status_policy == ReplayStatusPolicy.INCLUDE_ALL else data_config.require_replay_status,
        train_batch_size=1,
        val_batch_size=1,
        num_workers=0,
    )
    windows: list[LocalEpisodeWindow] = []
    for bundle in discover_local_lerobot_repo_bundles(eval_data.local_root or ""):
        repo_windows = _fast_scan_local_latent_windows(bundle.root, eval_data)
        repo_episodes = [episode.episode_index for episode in bundle.metadata.episodes]
        replay_status_records, replay_status_path = load_replay_status_records(
            bundle.root,
            replay_status_path=eval_data.replay_status_path,
            require=eval_data.require_replay_status,
        )
        selected_episodes, _ = filter_episode_indices_by_replay_status(
            repo_episodes,
            replay_status_records=replay_status_records,
            policy=eval_data.replay_status_policy,
            require_labeled=bool(replay_status_records) or bool(eval_data.require_replay_status),
            source_path=replay_status_path,
        )
        selected_episode_set = set(selected_episodes)
        windows.extend(window for window in repo_windows if window.episode_index in selected_episode_set)
    return FullSegmentLocalLeRobotLatentDataset(data_config=eval_data, windows=windows)


def select_early_middle_windows(
    dataset: Dataset[LatentWAMSample],
    *,
    horizon_frames: int,
    frame_chunk_size: int,
    trajectories_per_task: int,
    seed: int,
    action_per_frame: int = 1,
    min_context_frames: int = 4,
    context_window_frames: int | None = 16,
    early_fraction: float = 0.30,
    middle_fraction: float = 0.55,
    start_policy: FdmStartPolicy = FdmStartPolicy.EARLY_MIDDLE,
) -> list[FdmWindowSelection]:
    """Select deterministic early-middle windows, grouped by task text."""

    if frame_chunk_size <= 0:
        raise ValueError(f"frame_chunk_size must be positive, got {frame_chunk_size}.")
    generated_frames = require_chunk_aligned_horizon(
        horizon_frames=horizon_frames,
        frame_chunk_size=frame_chunk_size,
    )
    if trajectories_per_task <= 0:
        raise ValueError(f"trajectories_per_task must be positive, got {trajectories_per_task}.")
    windows = getattr(dataset, "windows", None)
    if windows is None:
        raise TypeError("FDM LIBERO sampling expects a local latent dataset with a `windows` attribute.")

    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for dataset_index, window in enumerate(windows):
        total_video_frames = _window_frame_count(window, action_per_frame=action_per_frame)
        if total_video_frames < min_context_frames + generated_frames + 1:
            continue
        task_key = _window_task_key(dataset, window)
        grouped[task_key].append((dataset_index, window))

    selections: list[FdmWindowSelection] = []
    rng = random.Random(seed)
    task_keys = sorted(grouped)
    for task_rank, task_key in enumerate(task_keys):
        candidates = sorted(
            grouped[task_key],
            key=lambda item: (
                int(getattr(item[1], "episode_index", 0)),
                int(getattr(item[1], "start_frame", 0)),
                int(getattr(item[1], "end_frame", 0)),
            ),
        )
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda item: _window_horizon_quality(
                item[1],
                action_per_frame=action_per_frame,
                generated_frames=generated_frames,
                min_context_frames=min_context_frames,
                early_fraction=early_fraction,
                middle_fraction=middle_fraction,
            ),
            reverse=True,
        )
        used_episodes: set[int] = set()
        picked: list[tuple[int, Any]] = []
        for candidate in candidates:
            episode_index = int(getattr(candidate[1], "episode_index", -1))
            if episode_index in used_episodes:
                continue
            picked.append(candidate)
            used_episodes.add(episode_index)
            if len(picked) >= trajectories_per_task:
                break
        if len(picked) < trajectories_per_task:
            for candidate in candidates:
                if candidate in picked:
                    continue
                picked.append(candidate)
                if len(picked) >= trajectories_per_task:
                    break

        for local_index, (dataset_index, window) in enumerate(picked[:trajectories_per_task]):
            total_video_frames = _window_frame_count(window, action_per_frame=action_per_frame)
            t0_frame = _select_t0_frame(
                start_policy=start_policy,
                total_video_frames=total_video_frames,
                generated_frames=generated_frames,
                min_context_frames=min_context_frames,
                local_index=local_index,
                count=max(trajectories_per_task, 1),
                early_fraction=early_fraction,
                middle_fraction=middle_fraction,
            )
            selections.append(
                FdmWindowSelection(
                    sample_index=len(selections),
                    dataset_index=int(dataset_index),
                    task_key=task_key,
                    task_rank=task_rank,
                    episode_index=int(getattr(window, "episode_index", -1)),
                    t0_frame=t0_frame,
                    horizon_frames=horizon_frames,
                    generated_frames=generated_frames,
                    context_start_frame=(
                        0
                        if context_window_frames is None
                        else max(0, t0_frame - max(min_context_frames, int(context_window_frames)))
                    ),
                    total_video_frames=total_video_frames,
                    repo_root=str(getattr(window, "repo_root", "")),
                )
            )

    return selections


def selection_to_manifest_row(selection: FdmWindowSelection) -> dict[str, Any]:
    return {
        "sample_index": selection.sample_index,
        "dataset_index": selection.dataset_index,
        "task_key": selection.task_key,
        "task_rank": selection.task_rank,
        "episode_index": selection.episode_index,
        "t0_frame": selection.t0_frame,
        "horizon_frames": selection.horizon_frames,
        "generated_frames": selection.generated_frames,
        "context_start_frame": selection.context_start_frame,
        "target_start_frame": selection.target_start_frame,
        "target_end_frame": selection.target_end_frame,
        "generation_end_frame": selection.generation_end_frame,
        "total_video_frames": selection.total_video_frames,
        "repo_root": selection.repo_root,
    }


def _fast_scan_local_latent_windows(repo_root: Path, data_config: DataConfig) -> list[LocalEpisodeWindow]:
    primary_camera = data_config.latent_camera_names[0]
    windows: list[LocalEpisodeWindow] = []
    for camera_dir in sorted(
        path for path in resolve_latent_root(repo_root, data_config).glob(f"chunk-*/{primary_camera}") if path.is_dir()
    ):
        for latent_file in sorted(camera_dir.glob("episode_*.pth")):
            match = match_latent_window_filename(latent_file.name)
            if match is None:
                continue
            start_frame = int(match.group("start"))
            end_frame = int(match.group("end"))
            windows.append(
                LocalEpisodeWindow(
                    repo_root=repo_root,
                    episode_index=int(match.group("episode")),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    observed_frame_ids=tuple(range(start_frame, end_frame)),
                )
            )
    return windows


def _window_frame_count(window: Any, *, action_per_frame: int = 1) -> int:
    raw_span = max(0, int(getattr(window, "end_frame", 0)) - int(getattr(window, "start_frame", 0)))
    if action_per_frame > 1:
        return max(0, raw_span // int(action_per_frame))
    observed_frame_ids = tuple(int(value) for value in getattr(window, "observed_frame_ids", ()) or ())
    if observed_frame_ids:
        return len(observed_frame_ids)
    return raw_span


def _window_task_key(dataset: Dataset[LatentWAMSample], window: Any) -> str:
    task_getter = getattr(dataset, "_window_task_text", None)
    if callable(task_getter):
        return str(task_getter(window))
    return f"task:{int(getattr(window, 'episode_index', 0))}"


def require_chunk_aligned_horizon(*, horizon_frames: int, frame_chunk_size: int) -> int:
    if horizon_frames <= 0:
        raise ValueError(f"horizon_frames must be positive, got {horizon_frames}.")
    if frame_chunk_size <= 0:
        raise ValueError(f"frame_chunk_size must be positive, got {frame_chunk_size}.")
    if int(horizon_frames) % int(frame_chunk_size) != 0:
        raise ValueError(
            "FDM evaluation requires --horizon-frames to be an exact multiple of "
            f"frame_chunk_size={frame_chunk_size}. Non-aligned horizons need an explicit "
            "last-chunk mask; silent action padding/truncation is intentionally rejected."
        )
    return int(horizon_frames)


def _window_horizon_quality(
    window: Any,
    *,
    action_per_frame: int,
    generated_frames: int,
    min_context_frames: int,
    early_fraction: float,
    middle_fraction: float,
) -> tuple[int, int, int, int]:
    """Prefer windows that keep long-horizon starts in the requested early-middle range."""

    total_video_frames = _window_frame_count(window, action_per_frame=action_per_frame)
    max_t0 = total_video_frames - int(generated_frames)
    ideal_early_t0 = max(int(min_context_frames), int(round(total_video_frames * early_fraction)))
    ideal_middle_t0 = max(int(min_context_frames), int(round(total_video_frames * middle_fraction)))
    return (
        int(max_t0 >= ideal_middle_t0),
        int(max_t0 >= ideal_early_t0),
        int(max_t0),
        int(total_video_frames),
    )


def _early_middle_t0(
    *,
    total_video_frames: int,
    generated_frames: int,
    min_context_frames: int,
    local_index: int,
    count: int,
    early_fraction: float,
    middle_fraction: float,
) -> int:
    max_t0 = int(total_video_frames) - int(generated_frames)
    if max_t0 < min_context_frames:
        raise ValueError(
            f"Cannot select t0: total_video_frames={total_video_frames}, generated_frames={generated_frames}, "
            f"min_context_frames={min_context_frames}."
        )
    fraction = early_fraction
    if count > 1:
        fraction = early_fraction + (middle_fraction - early_fraction) * (local_index / max(1, count - 1))
    proposed = int(round(total_video_frames * fraction))
    return min(max(min_context_frames, proposed), max_t0)


def _select_t0_frame(
    *,
    start_policy: FdmStartPolicy,
    total_video_frames: int,
    generated_frames: int,
    min_context_frames: int,
    local_index: int,
    count: int,
    early_fraction: float,
    middle_fraction: float,
) -> int:
    if start_policy == FdmStartPolicy.EARLY_MIDDLE:
        return _early_middle_t0(
            total_video_frames=total_video_frames,
            generated_frames=generated_frames,
            min_context_frames=min_context_frames,
            local_index=local_index,
            count=count,
            early_fraction=early_fraction,
            middle_fraction=middle_fraction,
        )
    if start_policy == FdmStartPolicy.LATEST_FIT:
        latest_t0 = int(total_video_frames) - int(generated_frames)
        if latest_t0 < int(min_context_frames):
            raise ValueError(
                f"Cannot select latest_fit t0: total_video_frames={total_video_frames}, "
                f"generated_frames={generated_frames}, min_context_frames={min_context_frames}."
            )
        return latest_t0
    raise ValueError(f"Unsupported FDM start policy: {start_policy}.")
