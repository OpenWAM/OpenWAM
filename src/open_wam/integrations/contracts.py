from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class BenchmarkActionSchema:
    """Benchmark-native action metadata documented outside simulator code."""

    native_action_dim: int
    model_action_dim: int
    action_mode: str
    gripper_mode: str | None = None


@dataclass(frozen=True)
class BenchmarkObservationSchema:
    """Benchmark observation metadata consumed by adapter cards and tests."""

    camera_names: tuple[str, ...]
    state_dim: int | None
    canonical_height: int
    canonical_width: int


class BenchmarkAdapterContract(Protocol):
    """Dependency-light simulator adapter contract for docs and fake tests."""

    benchmark_name: str
    action_schema: BenchmarkActionSchema
    observation_schema: BenchmarkObservationSchema

    def reset(self, *, task_id: int | None, episode_idx: int | None, seed: int | None) -> Any:
        """Reset the benchmark and return the first observation."""

    def task_text(self) -> str | None:
        """Return natural-language task text, if available."""

    def extract_views(self, observation: Any) -> Mapping[str, Any]:
        """Return RGB-like frames keyed by model camera names."""

    def extract_state(self, observation: Any) -> Any | None:
        """Return the benchmark state vector, if available."""

    def model_action_to_env_action(self, model_action: Any, *, data_config: Any) -> Any:
        """Convert one model-space action into the benchmark-native action."""

    def step(self, env_action: Any) -> Any:
        """Step the benchmark once and return a transition object."""

    def success(self, observation: Any, info: Mapping[str, Any]) -> bool:
        """Return whether the current rollout has succeeded."""

    def render_frame(self, observation: Any) -> Any | None:
        """Return an RGB visualization frame when supported."""

    def close(self) -> None:
        """Release benchmark resources."""
