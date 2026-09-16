from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Protocol, TypeAlias

from open_wam.runtime.control import ControlCommand, ControlTransition

if TYPE_CHECKING:
    import numpy as np

    _NumpyArray: TypeAlias = np.ndarray
else:
    _NumpyArray: TypeAlias = Any


@dataclass(frozen=True)
class EpisodeSpec:
    """Task/episode selection passed to simulator backends."""

    task_id: int | None = None
    episode_idx: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class SimulatorCapabilities:
    """Backend behavior that rollout schedulers must not infer implicitly."""

    action_step_semantics: str
    supports_render: bool = True
    supports_success: bool = True
    supports_expert_precheck: bool = False
    action_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimulatorObservation:
    """Canonical cameras and model-ready state, plus the untouched raw payload.

    state must use the configured data schema's encoding and exact feature width.
    Native qpos/EEF conversion belongs to the benchmark adapter, not the policy.
    It may be omitted only when the policy does not condition on proprio.
    """

    views: Mapping[str, _NumpyArray]
    state: _NumpyArray | None = None
    task_text: str | None = None
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulatorBackend(Protocol):
    """Normalized simulator boundary consumed by shared rollout engines."""

    benchmark_name: str
    capabilities: SimulatorCapabilities

    def reset(self, spec: EpisodeSpec) -> SimulatorObservation:
        """Reset the simulator and return the first policy-visible observation."""

    def task_text(self) -> str | None:
        """Return the current natural-language instruction, if available."""

    def materialize_control(
        self,
        source_action: _NumpyArray,
        *,
        data_config: Any,
    ) -> ControlCommand:
        """Convert a dataset-source command into executable controls."""

    def step(self, action: _NumpyArray) -> ControlTransition[SimulatorObservation]:
        """Execute one policy-visible control action."""

    def render_frame(
        self,
        observation: SimulatorObservation,
    ) -> _NumpyArray | None:
        """Return an RGB visualization frame, if available."""

    def close(self) -> None:
        """Release simulator resources."""


class ObservationAdapterBackend:
    """Normalize a benchmark's raw observations at the simulator boundary."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.benchmark_name = adapter.benchmark_name
        self.capabilities = adapter.capabilities

    def reset(self, spec: EpisodeSpec) -> SimulatorObservation:
        raw_observation = self.adapter.reset(
            task_id=spec.task_id,
            episode_idx=spec.episode_idx,
            seed=spec.seed,
        )
        return self._normalize_observation(raw_observation)

    def task_text(self) -> str | None:
        return self.adapter.task_text()

    def materialize_control(
        self,
        source_action: _NumpyArray,
        *,
        data_config: Any,
    ) -> ControlCommand:
        return self.adapter.materialize_control(source_action, data_config=data_config)

    def step(self, action: _NumpyArray) -> ControlTransition[SimulatorObservation]:
        transition = self.adapter.step(action)
        observation = self._normalize_observation(transition.observation)
        return replace(transition, observation=observation)

    def render_frame(
        self,
        observation: SimulatorObservation,
    ) -> _NumpyArray | None:
        return self.adapter.render_frame(observation.raw)

    def close(self) -> None:
        self.adapter.close()

    def _normalize_observation(self, raw_observation: Any) -> SimulatorObservation:
        return SimulatorObservation(
            views=self.adapter.extract_views(raw_observation),
            state=self.adapter.extract_state(raw_observation),
            task_text=self.adapter.task_text(),
            raw=raw_observation,
        )
