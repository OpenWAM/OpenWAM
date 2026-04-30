from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


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
    """Policy-visible simulator observation plus raw benchmark payload."""

    views: Mapping[str, np.ndarray]
    state: np.ndarray | None = None
    task_text: str | None = None
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorStepResult:
    """One policy-visible simulator control transition."""

    observation: SimulatorObservation
    reward: float | None = None
    done: bool = False
    success: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class SimulatorBackend(Protocol):
    """Normalized simulator boundary consumed by shared rollout engines."""

    benchmark_name: str
    capabilities: SimulatorCapabilities

    def reset(self, spec: EpisodeSpec) -> SimulatorObservation:
        """Reset the simulator and return the first policy-visible observation."""

    def task_text(self) -> str | None:
        """Return the current natural-language instruction, if available."""

    def action_from_model_action(self, model_action: np.ndarray, *, data_config: Any) -> np.ndarray:
        """Convert one model-facing action vector into the simulator action space."""

    def step(self, action: np.ndarray) -> SimulatorStepResult:
        """Execute one policy-visible control action."""

    def render_frame(self, observation: SimulatorObservation) -> np.ndarray | None:
        """Return an RGB visualization frame, if available."""

    def close(self) -> None:
        """Release simulator resources."""


class LegacyAdapterSimulatorBackend:
    """Compatibility wrapper for pre-normalized simulator adapters.

    Existing adapters expose raw observations plus ``extract_*`` methods. This
    wrapper turns them into the normalized backend contract so rollout code can
    depend on one interface while benchmark adapters migrate incrementally.
    """

    capabilities = SimulatorCapabilities(action_step_semantics="policy_control_step")

    def __init__(self, adapter: Any, *, capabilities: SimulatorCapabilities | None = None) -> None:
        self.adapter = adapter
        self.benchmark_name = str(getattr(adapter, "benchmark_name", "unknown"))
        if capabilities is not None:
            self.capabilities = capabilities
        elif hasattr(adapter, "capabilities"):
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

    def action_from_model_action(self, model_action: np.ndarray, *, data_config: Any) -> np.ndarray:
        return self.adapter.model_action_to_env_action(model_action, data_config=data_config)

    def step(self, action: np.ndarray) -> SimulatorStepResult:
        transition = self.adapter.step(action)
        observation = self._normalize_observation(transition.observation)
        info = dict(getattr(transition, "info", {}) or {})
        success = bool(self.adapter.success(transition.observation, info))
        return SimulatorStepResult(
            observation=observation,
            reward=getattr(transition, "reward", None),
            done=bool(getattr(transition, "done", False)),
            success=success,
            info=info,
        )

    def render_frame(self, observation: SimulatorObservation) -> np.ndarray | None:
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


def ensure_simulator_backend(value: Any) -> SimulatorBackend:
    """Return ``value`` if normalized, otherwise wrap a legacy adapter."""

    if hasattr(value, "action_from_model_action"):
        return value
    return LegacyAdapterSimulatorBackend(value)
