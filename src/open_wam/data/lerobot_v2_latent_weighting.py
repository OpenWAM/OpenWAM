"""Physical and virtual sample weighting for local LeRobot latents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from open_wam.configs import (
    DataConfig,
    LatentWindowProfile,
    SampleConstructionConfig,
    SampleWeightMode,
    WindowSamplingMode,
)

from .lerobot_v2_latent_storage import LocalEpisodeWindow, LocalRepoBundle


__all__ = ["LocalLatentWindowWeightPlan"]


def _build_local_latent_sample_weights(
    *,
    sample_config: SampleConstructionConfig,
    item_count: int,
    dataset_mean_valid_action_steps: float,
    dataset_mean_task_demo_count: float,
    valid_action_steps_for_index: Callable[[int], float],
    task_text_for_index: Callable[[int], str],
    task_demo_counts: Mapping[str, int],
    task_virtual_start_counts: Mapping[str, int] | None = None,
    dataset_mean_task_virtual_start_count: float = 1.0,
) -> tuple[float, ...]:
    """Build normalized weights for physical windows or virtual starts."""

    mode = sample_config.sample_weight_mode
    if mode == SampleWeightMode.UNIFORM:
        return tuple(1.0 for _ in range(item_count))
    reference_steps = max(1.0, float(dataset_mean_valid_action_steps))
    reference_task_count = max(1.0, float(dataset_mean_task_demo_count))
    weights: list[float] = []
    for index in range(item_count):
        weight = 1.0
        if mode in {
            SampleWeightMode.VALID_ACTION_STEPS,
            SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
        }:
            weight *= (
                max(
                    1.0,
                    float(valid_action_steps_for_index(index)),
                )
                / reference_steps
            )
        if mode in {
            SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
            SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
        }:
            task_count = max(
                1,
                task_demo_counts[task_text_for_index(index)],
            )
            weight *= reference_task_count / float(task_count)
        if (
            mode == SampleWeightMode.TASK_VIRTUAL_START_COUNT_POWER
            and task_virtual_start_counts is not None
        ):
            task_start_count = max(
                1.0,
                float(task_virtual_start_counts[task_text_for_index(index)]),
            )
            reference_start_count = max(
                1.0,
                float(dataset_mean_task_virtual_start_count),
            )
            weight *= (task_start_count / reference_start_count) ** (
                float(sample_config.sample_weight_length_power) - 1.0
            )
        if sample_config.sample_weight_min is not None:
            weight = max(float(sample_config.sample_weight_min), weight)
        if sample_config.sample_weight_max is not None:
            weight = min(float(sample_config.sample_weight_max), weight)
        weights.append(float(weight))
    if not any(weight > 0 for weight in weights):
        return tuple(1.0 for _ in range(item_count))
    return tuple(weights)


@dataclass(frozen=True)
class LocalLatentWindowWeightPlan:
    """Deterministic task statistics and weights for physical latent windows."""

    data_config: DataConfig
    windows: tuple[LocalEpisodeWindow, ...]
    window_valid_action_steps: tuple[int, ...]
    dataset_mean_valid_action_steps: float
    window_task_texts: tuple[str, ...]
    task_demo_counts: Counter[str]
    dataset_mean_task_demo_count: float
    sample_weights: tuple[float, ...]

    @classmethod
    def from_windows(
        cls,
        *,
        data_config: DataConfig,
        windows: Sequence[LocalEpisodeWindow],
        repo_bundles: Mapping[str, LocalRepoBundle],
    ) -> LocalLatentWindowWeightPlan:
        """Resolve all deterministic weighting state for physical windows."""

        window_tuple = tuple(windows)
        valid_action_steps: list[int] = []
        task_texts: list[str] = []
        for window in window_tuple:
            if (
                data_config.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
                and data_config.latent_window_profile
                == LatentWindowProfile.EXACT_CHUNKED_WINDOW
            ):
                prefix_actions = int(
                    data_config.action_schema.action_horizon
                    // max(1, data_config.num_frames)
                )
                window_span = max(0, window.end_frame - window.start_frame)
                valid_steps = prefix_actions + max(
                    len(window.observation_frame_indices),
                    window_span,
                )
            elif (
                data_config.sample_construction.mode
                == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
            ):
                valid_steps = 0
            else:
                valid_steps = int(data_config.action_schema.action_horizon)
            valid_action_steps.append(max(0, int(valid_steps)))

            repo_bundle = repo_bundles.get(str(window.repo_root))
            episode_record = (
                None
                if repo_bundle is None
                else repo_bundle.episodes_by_index.get(window.episode_index)
            )
            task_texts.append(
                str(episode_record.tasks[0])
                if episode_record is not None and episode_record.tasks
                else f"{window.repo_root}:episode:{window.episode_index}"
            )

        window_valid_action_steps = tuple(valid_action_steps)
        positive_estimates = [value for value in valid_action_steps if value > 0]
        if not positive_estimates:
            dataset_mean_valid_action_steps = float(
                max(1, data_config.action_schema.action_horizon)
            )
        else:
            dataset_mean_valid_action_steps = float(
                sum(positive_estimates) / len(positive_estimates)
            )
        window_task_texts = tuple(task_texts)
        demo_keys_by_task: dict[str, set[tuple[str, int]]] = {}
        for window, task_text in zip(
            window_tuple,
            window_task_texts,
            strict=True,
        ):
            demo_keys_by_task.setdefault(task_text, set()).add(
                (str(window.repo_root), int(window.episode_index))
            )
        task_demo_counts = Counter(
            {
                task_text: len(demo_keys)
                for task_text, demo_keys in demo_keys_by_task.items()
            }
        )
        dataset_mean_task_demo_count = (
            float(sum(task_demo_counts.values()) / len(task_demo_counts))
            if task_demo_counts
            else 1.0
        )
        sample_weights = _build_local_latent_sample_weights(
            sample_config=data_config.sample_construction,
            item_count=len(window_tuple),
            dataset_mean_valid_action_steps=dataset_mean_valid_action_steps,
            dataset_mean_task_demo_count=dataset_mean_task_demo_count,
            valid_action_steps_for_index=(
                lambda index: window_valid_action_steps[index]
            ),
            task_text_for_index=lambda index: window_task_texts[index],
            task_demo_counts=task_demo_counts,
        )
        return cls(
            data_config=data_config,
            windows=window_tuple,
            window_valid_action_steps=window_valid_action_steps,
            dataset_mean_valid_action_steps=dataset_mean_valid_action_steps,
            window_task_texts=window_task_texts,
            task_demo_counts=task_demo_counts,
            dataset_mean_task_demo_count=dataset_mean_task_demo_count,
            sample_weights=sample_weights,
        )

    def sample_weight_metadata(self, index: int) -> dict[str, Any]:
        """Describe weighting inputs for one physical window."""

        task_text = self.window_task_texts[index]
        return {
            "train_sample_weight": self.sample_weights[index],
            "train_sample_weight_mode": (
                self.data_config.sample_construction.sample_weight_mode
            ),
            "eligible_task_demo_count": self.task_demo_counts[task_text],
            "dataset_mean_eligible_task_demo_count": (
                self.dataset_mean_task_demo_count
            ),
        }

    def task_text_for_window_index(self, index: int) -> str:
        """Return the resolved task label for one physical window."""

        return self.window_task_texts[index]
