from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

import numpy as np
import torch

from open_wam.runtime.policy_planner import PolicyPlanner
from open_wam.configs import DataConfig
from open_wam.runtime.control import RolloutTermination, RolloutTerminationReason
from open_wam.runtime.planner_executor import PlannerTeardown

from .contracts import EpisodeSpec, SimulatorBackend


@dataclass(frozen=True)
class SimRolloutResult:
    """Structured result from one closed-loop simulator rollout."""

    benchmark: str
    task_text: str | None
    success: bool
    steps: int
    target_action_hz: float | None
    # Total includes startup and teardown; live covers only the control loop.
    wall_time_s: float
    live_wall_time_s: float
    mean_policy_step_s: float | None
    mean_env_step_s: float | None
    achieved_action_hz: float
    policy_action_shapes: tuple[tuple[int, ...], ...]
    action_records: tuple[dict[str, Any], ...]
    video_frames: tuple[np.ndarray, ...]
    termination: RolloutTermination
    planner_teardown: PlannerTeardown | None = None


class SimActionCommitMode(str, Enum):
    """How many predicted actions are committed before the next replan."""

    FIRST_FRAME = "first_frame"
    FULL_CHUNK = "full_chunk"


def normalize_quaternion_xyzw(values: np.ndarray, *, start: int) -> None:
    """Normalize an in-place xyzw quaternion slice when present."""

    quat = values[start : start + 4]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        values[start : start + 4] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return
    values[start : start + 4] = quat / norm


def run_closed_loop_sim_rollout(
    *,
    adapter: SimulatorBackend,
    rollout_runner: Any,
    data_config: DataConfig,
    device: torch.device,
    task_id: int | None,
    episode_idx: int | None,
    seed: int | None,
    max_steps: int,
    target_action_hz: float | None = None,
    action_commit_mode: SimActionCommitMode | str = SimActionCommitMode.FIRST_FRAME,
) -> SimRolloutResult:
    """Drive the shared policy lifecycle with a blocking simulator."""
    from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions
    from .policy_adapter import SimulatorPolicyAdapter

    backend = adapter
    initial = backend.reset(
        EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed)
    )
    task_text = initial.task_text or backend.task_text()
    commit_mode = _normalize_action_commit_mode(action_commit_mode)
    density = (
        rollout_runner.pipeline.policy_variant.rollout_contract.action_tokens_per_frame
    )
    policy_adapter = SimulatorPolicyAdapter(
        rollout_runner,
        data_config,
        device,
        lambda source: backend.materialize_control(source, data_config=data_config),
    )
    options = RolloutOptions(
        max_actions=max_steps,
        target_action_hz=target_action_hz,
        execute_prefix_actions=density
        if commit_mode is SimActionCommitMode.FIRST_FRAME
        else None,
    )
    frames = []

    def step(action):
        transition = backend.step(action)
        frame = backend.render_frame(transition.observation)
        if frame is not None:
            frames.append(_as_uint8_rgb(frame, key="render_frame"))
        return transition

    began = time.perf_counter()
    result = RolloutEngine(PolicyPlanner(rollout_runner, policy_adapter), policy_adapter, options).run(
        initial,
        rollout_runner.reset(task_text=(task_text,)),
        step=step,
    )
    elapsed = time.perf_counter() - began
    traces = [result.startup, *result.replans, *result.extensions]
    return SimRolloutResult(
        benchmark=backend.benchmark_name,
        task_text=task_text,
        success=result.success,
        steps=len(result.actions),
        target_action_hz=target_action_hz,
        wall_time_s=elapsed,
        live_wall_time_s=result.live_wall_time_s,
        mean_policy_step_s=_mean([trace.total_latency_s for trace in traces]),
        mean_env_step_s=_mean([row.env_step_s for row in result.actions]),
        achieved_action_hz=len(result.actions)
        / max(result.live_wall_time_s, 1e-12),
        policy_action_shapes=tuple(
            (1, *trace.policy_action_shape) for trace in traces
        ),
        action_records=tuple(row.to_record() for row in result.actions),
        video_frames=tuple(frames),
        termination=result.termination,
        planner_teardown=result.planner_teardown,
    )


def run_zero_control_smoke(
    *,
    adapter: SimulatorBackend,
    data_config: DataConfig,
    device: torch.device,
    task_id: int | None,
    episode_idx: int | None,
    seed: int | None,
    max_steps: int,
    target_action_hz: float | None = None,
    action_commit_mode: SimActionCommitMode | str = SimActionCommitMode.FIRST_FRAME,
) -> SimRolloutResult:
    """Environment wiring check, deliberately not a model or recurrent policy."""
    from open_wam.data.action_adapter import ConfiguredActionAdapter

    if max_steps <= 0 or (target_action_hz is not None and target_action_hz <= 0):
        raise ValueError("Control horizon and frequency must be positive.")
    _normalize_action_commit_mode(action_commit_mode)
    backend = adapter
    initial = backend.reset(
        EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed)
    )
    mapping = ConfiguredActionAdapter(
        data_config.action_mapping,
        data_config.action_target.normalization,
        data_config.action_schema.action_dim,
    )
    source = mapping.to_source(torch.zeros(1, mapping.model_dim))[0].numpy()
    actions, frames = [], []
    success = False
    began = time.perf_counter()
    for index in range(max_steps):
        if target_action_hz is not None:
            time.sleep(max(0.0, began + index / target_action_hz - time.perf_counter()))
        control = backend.materialize_control(source.copy(), data_config=data_config)
        start = time.perf_counter()
        transition = backend.step(control.action)
        actions.append(
            {
                "action_index": index,
                "action": control.action.tolist(),
                "env_step_s": time.perf_counter() - start,
                "reward": transition.reward,
            }
        )
        frame = backend.render_frame(transition.observation)
        if frame is not None:
            frames.append(_as_uint8_rgb(frame, key="render_frame"))
        success = transition.success
        if transition.done or success:
            break
    elapsed = time.perf_counter() - began
    return SimRolloutResult(
        benchmark=backend.benchmark_name,
        task_text=initial.task_text,
        success=success,
        steps=len(actions),
        target_action_hz=target_action_hz,
        wall_time_s=elapsed,
        live_wall_time_s=elapsed,
        mean_policy_step_s=None,
        mean_env_step_s=_mean([a["env_step_s"] for a in actions]),
        achieved_action_hz=len(actions) / max(elapsed, 1e-12),
        policy_action_shapes=(),
        action_records=tuple(actions),
        video_frames=tuple(frames),
        termination=RolloutTermination(
            RolloutTerminationReason.SUCCESS
            if success
            else RolloutTerminationReason.ENV_TERMINAL
            if transition.done
            else RolloutTerminationReason.MAX_ACTIONS,
            len(actions),
            transition=transition,
        ),
    )


def _normalize_action_commit_mode(
    value: SimActionCommitMode | str,
) -> SimActionCommitMode:
    if isinstance(value, SimActionCommitMode):
        return value
    try:
        return SimActionCommitMode(str(value))
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in SimActionCommitMode)
        raise ValueError(
            f"Unknown action commit mode {value!r}; expected one of: {choices}."
        ) from exc


def summarize_sim_rollout(
    result: SimRolloutResult, *, video_path: str | None = None
) -> dict[str, Any]:
    """Serialize one simulator rollout result without embedding video frames."""

    return {
        "benchmark": result.benchmark,
        "task_text": result.task_text,
        "success": result.success,
        "termination_reason": result.termination.reason.value,
        "planner_teardown_error": None
        if result.planner_teardown is None
        else result.planner_teardown.error,
        "planner_drain_time_s": None
        if result.planner_teardown is None
        else result.planner_teardown.drain_time_s,
        "steps": result.steps,
        "target_action_hz": result.target_action_hz,
        "wall_time_s": result.wall_time_s,
        "live_wall_time_s": result.live_wall_time_s,
        "achieved_action_hz": result.achieved_action_hz,
        "mean_policy_step_s": result.mean_policy_step_s,
        "mean_env_step_s": result.mean_env_step_s,
        "policy_action_shapes": [list(shape) for shape in result.policy_action_shapes],
        "video_path": video_path,
        "action_records": list(result.action_records),
    }


def _as_uint8_rgb(value: np.ndarray, *, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            f"Expected `{key}` RGB image with shape [H, W, 3], got {array.shape}."
        )
    array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
