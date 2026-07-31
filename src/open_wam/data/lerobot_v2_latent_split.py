"""Train/validation window planning for local LeRobot latent repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_wam.configs import DataConfig, ReplayStatusPolicy

from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    discover_local_lerobot_repo_bundles,
    scan_local_latent_windows,
)
from .replay_status import (
    load_replay_status_records,
    split_episode_indices_by_replay_status,
)


__all__ = [
    "LocalLatentTrainValWindowPlan",
    "LocalLatentTrainValWindowPlanner",
]


_USE_CONFIG_REPLAY_STATUS_PATH = object()


@dataclass(frozen=True)
class LocalLatentTrainValWindowPlan:
    """Resolved local latent windows for both dataset splits."""

    train_windows: tuple[LocalEpisodeWindow, ...]
    val_windows: tuple[LocalEpisodeWindow, ...]


@dataclass(frozen=True)
class LocalLatentTrainValWindowPlanner:
    """Discover local repositories and apply episode/replay split policy."""

    data_config: DataConfig

    def plan(self) -> LocalLatentTrainValWindowPlan:
        if self.data_config.val_local_root:
            train_windows, val_windows = self._plan_explicit_roots()
        else:
            train_windows, val_windows = self._plan_shared_roots()
        return LocalLatentTrainValWindowPlan(
            train_windows=tuple(train_windows),
            val_windows=tuple(val_windows),
        )

    def _plan_explicit_roots(
        self,
    ) -> tuple[list[LocalEpisodeWindow], list[LocalEpisodeWindow]]:
        val_replay_status_path = self.data_config.val_replay_status_path
        if (
            val_replay_status_path is None
            and self.data_config.replay_status_path is not None
        ):
            train_status_path = Path(
                self.data_config.replay_status_path
            ).expanduser()
            val_replay_status_path = (
                None
                if train_status_path.is_absolute()
                else self.data_config.replay_status_path
            )
        train_windows = self._filtered_windows_for_roots(
            self.data_config.local_root or "",
            max_episodes=self.data_config.max_train_episodes,
        )
        val_windows = self._filtered_windows_for_roots(
            self.data_config.val_local_root or "",
            max_episodes=self.data_config.max_val_episodes,
            configured_replay_status_path=val_replay_status_path,
            replay_status_policy=(
                self.data_config.val_replay_status_policy
                or self.data_config.replay_status_policy
            ),
            require_replay_status=(
                self.data_config.require_replay_status
                if self.data_config.val_require_replay_status is None
                else self.data_config.val_require_replay_status
            ),
        )
        return train_windows, val_windows

    def _plan_shared_roots(
        self,
    ) -> tuple[list[LocalEpisodeWindow], list[LocalEpisodeWindow]]:
        train_windows: list[LocalEpisodeWindow] = []
        val_windows: list[LocalEpisodeWindow] = []
        bundles = discover_local_lerobot_repo_bundles(
            self.data_config.local_root or ""
        )
        for bundle in bundles:
            repo_windows = scan_local_latent_windows(
                bundle.root,
                self.data_config,
            )
            repo_episodes = [
                episode.episode_index for episode in bundle.metadata.episodes
            ]
            replay_status_records, replay_status_path = load_replay_status_records(
                bundle.root,
                replay_status_path=self.data_config.replay_status_path,
                require=self.data_config.require_replay_status,
            )
            split = split_episode_indices_by_replay_status(
                repo_episodes,
                replay_status_records=replay_status_records,
                replay_status_path=replay_status_path,
                replay_status_policy=self.data_config.replay_status_policy,
                require_replay_status=self.data_config.require_replay_status,
                val_replay_status_policy=self.data_config.val_replay_status_policy,
                val_require_replay_status=self.data_config.val_require_replay_status,
                train_fraction=self.data_config.train_fraction,
                split_seed=self.data_config.split_seed,
                max_train_episodes=self.data_config.max_train_episodes,
                max_val_episodes=self.data_config.max_val_episodes,
            )
            train_episode_set = set(split.train_episodes)
            val_episode_set = set(split.val_episodes)
            repo_train_windows = [
                window
                for window in repo_windows
                if window.episode_index in train_episode_set
            ]
            repo_val_windows = [
                window
                for window in repo_windows
                if window.episode_index in val_episode_set
            ]
            if (
                not split.used_explicit_val_policy
                and not repo_val_windows
                and repo_train_windows
            ):
                repo_val_windows = repo_train_windows[:1]
            train_windows.extend(repo_train_windows)
            val_windows.extend(repo_val_windows)
        return train_windows, val_windows

    def _filtered_windows_for_roots(
        self,
        local_root: str,
        *,
        max_episodes: int | None = None,
        configured_replay_status_path: str | None | object = (
            _USE_CONFIG_REPLAY_STATUS_PATH
        ),
        replay_status_policy: ReplayStatusPolicy | None = None,
        require_replay_status: bool | None = None,
    ) -> list[LocalEpisodeWindow]:
        windows: list[LocalEpisodeWindow] = []
        for bundle in discover_local_lerobot_repo_bundles(local_root):
            repo_windows = scan_local_latent_windows(
                bundle.root,
                self.data_config,
            )
            repo_episodes = [
                episode.episode_index for episode in bundle.metadata.episodes
            ]
            replay_status_records, replay_status_path = load_replay_status_records(
                bundle.root,
                replay_status_path=(
                    self.data_config.replay_status_path
                    if configured_replay_status_path
                    is _USE_CONFIG_REPLAY_STATUS_PATH
                    else configured_replay_status_path
                ),
                require=(
                    self.data_config.require_replay_status
                    if require_replay_status is None
                    else bool(require_replay_status)
                ),
            )
            split = split_episode_indices_by_replay_status(
                repo_episodes,
                replay_status_records=replay_status_records,
                replay_status_path=replay_status_path,
                replay_status_policy=(
                    replay_status_policy
                    or self.data_config.replay_status_policy
                ),
                require_replay_status=(
                    self.data_config.require_replay_status
                    if require_replay_status is None
                    else bool(require_replay_status)
                ),
                val_replay_status_policy=None,
                val_require_replay_status=None,
                train_fraction=1.0,
                split_seed=self.data_config.split_seed,
                max_train_episodes=max_episodes,
                max_val_episodes=None,
            )
            episode_set = set(split.train_episodes)
            windows.extend(
                window
                for window in repo_windows
                if window.episode_index in episode_set
            )
        return windows
