from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

import numpy as np
import torch

from open_wam.configs import DataConfig
from open_wam.data.action_mapping import inverse_action_mapping

from .contracts import EpisodeSpec, SimulatorBackend, SimulatorObservation, ensure_simulator_backend


@dataclass(frozen=True)
class SimStepResult:
    """Legacy raw-observation transition returned by existing adapters."""

    observation: Any
    reward: float | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimRolloutResult:
    """Structured result from one closed-loop simulator rollout."""

    benchmark: str
    task_text: str | None
    success: bool
    steps: int
    target_action_hz: float | None
    wall_time_s: float
    mean_policy_step_s: float | None
    mean_env_step_s: float | None
    achieved_action_hz: float
    policy_action_shapes: tuple[tuple[int, ...], ...]
    action_records: tuple[dict[str, Any], ...]
    video_frames: tuple[np.ndarray, ...]


class SimActionCommitMode(str, Enum):
    """How many predicted actions are committed before the next replan."""

    FIRST_ACTION = "first_action"
    FULL_CHUNK = "full_chunk"


@dataclass
class SimPolicyInferContext:
    """Lightweight rollout context passed to policy runners.

    Full Open-WAM runners coerce this into their typed policy context. Keeping
    the simulator loop independent from model modules lets external simulator
    envs run wiring and zero-policy checks without installing diffusion deps.
    """

    state: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def source_action_from_model_action(
    model_action: np.ndarray,
    *,
    data_config: DataConfig,
) -> np.ndarray:
    """Convert one model-facing action vector back to the data-source action schema."""

    tensor = torch.as_tensor(model_action, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    source = inverse_action_mapping(tensor, data_config.action_mapping)
    array = source.detach().cpu().numpy().astype(np.float32)
    return array[0] if squeeze else array


def normalize_quaternion_xyzw(values: np.ndarray, *, start: int) -> None:
    """Normalize an in-place xyzw quaternion slice when present."""

    quat = values[start : start + 4]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        values[start : start + 4] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return
    values[start : start + 4] = quat / norm


def build_view_history_batch(
    history: dict[str, deque[np.ndarray]],
    *,
    camera_names: tuple[str, ...],
    num_frames: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build batched `[1, T, H, W, 3]` view tensors from a rolling history."""

    views: dict[str, torch.Tensor] = {}
    for camera_name in camera_names:
        frames = list(history[camera_name])
        if not frames:
            raise ValueError(f"Cannot build view history for empty camera '{camera_name}'.")
        while len(frames) < num_frames:
            frames.insert(0, frames[0])
        frames = frames[-num_frames:]
        array = np.stack([_as_uint8_rgb(frame, key=camera_name) for frame in frames], axis=0)
        views[camera_name] = torch.from_numpy(array).unsqueeze(0).to(device=device)
    return views


def build_state_history_tensor(
    history: deque[np.ndarray],
    *,
    state_dim: int,
    state_horizon: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build batched `[1, H_state, D_state]` state tensor from rolling history."""

    if state_dim <= 0:
        return None
    if not history:
        return torch.zeros(1, state_horizon, state_dim, dtype=torch.float32, device=device)
    states = [np.asarray(value, dtype=np.float32).reshape(-1) for value in history]
    while len(states) < state_horizon:
        states.insert(0, states[0])
    states = states[-state_horizon:]
    packed = np.zeros((state_horizon, state_dim), dtype=np.float32)
    for index, state in enumerate(states):
        dim = min(state_dim, state.shape[0])
        packed[index, :dim] = state[:dim]
    return torch.from_numpy(packed).unsqueeze(0).to(device=device)


def run_closed_loop_sim_rollout(
    *,
    adapter: Any,
    rollout_runner: Any,
    data_config: DataConfig,
    device: torch.device,
    task_id: int | None,
    episode_idx: int | None,
    seed: int | None,
    max_steps: int,
    target_action_hz: float | None = None,
    action_commit_mode: SimActionCommitMode | str = SimActionCommitMode.FIRST_ACTION,
) -> SimRolloutResult:
    """Run one synchronous closed-loop rollout against a simulator backend."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if target_action_hz is not None and target_action_hz <= 0:
        raise ValueError("target_action_hz must be positive when provided.")
    commit_mode = _normalize_action_commit_mode(action_commit_mode)
    backend: SimulatorBackend = ensure_simulator_backend(adapter)

    observation = backend.reset(EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed))
    task_text = observation.task_text or backend.task_text()
    session = rollout_runner.reset(task_text=(task_text,))
    camera_names = tuple(data_config.camera_names)
    view_history = {name: deque(maxlen=data_config.num_frames) for name in camera_names}
    state_history: deque[np.ndarray] = deque(maxlen=data_config.action_schema.state_horizon)
    previous_action: torch.Tensor | None = None
    action_records: list[dict[str, Any]] = []
    policy_action_shapes: list[tuple[int, ...]] = []
    video_frames: list[np.ndarray] = []
    policy_step_times: list[float] = []
    env_step_times: list[float] = []
    success = False

    def append_observation_to_history(current_observation: SimulatorObservation) -> None:
        for camera_name in camera_names:
            if camera_name not in current_observation.views:
                raise KeyError(
                    f"Simulator backend did not provide required camera '{camera_name}'. "
                    f"Available cameras: {sorted(current_observation.views)}"
                )
            view_history[camera_name].append(current_observation.views[camera_name])
        if current_observation.state is not None:
            state_history.append(np.asarray(current_observation.state, dtype=np.float32))

    append_observation_to_history(observation)
    rollout_start = time.perf_counter()
    next_deadline = rollout_start
    plan_index = 0
    while len(action_records) < max_steps:
        loop_start = time.perf_counter()
        views = build_view_history_batch(
            view_history,
            camera_names=camera_names,
            num_frames=data_config.num_frames,
            device=device,
        )
        state = build_state_history_tensor(
            state_history,
            state_dim=data_config.action_schema.state_dim,
            state_horizon=data_config.action_schema.state_horizon,
            device=device,
        )
        context = SimPolicyInferContext(
            state=state,
            previous_action=previous_action,
            extra={
                "task_text": (task_text,),
                "metadata": ({"sim_step": len(action_records), "plan_index": plan_index},),
            },
        )
        policy_start = time.perf_counter()
        with torch.no_grad():
            step_output = rollout_runner.infer_step(
                session=session,
                context=context,
                views=views,
            )
        policy_elapsed = time.perf_counter() - policy_start
        session = step_output.session
        action_pred = step_output.infer_output.decoder_output.action_pred.detach()
        policy_action_shapes.append(tuple(int(value) for value in action_pred.shape))
        policy_step_times.append(policy_elapsed)
        action_horizon = int(action_pred.shape[1])
        if action_horizon <= 0:
            raise ValueError(f"Policy returned an empty action horizon: {tuple(action_pred.shape)}.")
        commit_count = 1 if commit_mode is SimActionCommitMode.FIRST_ACTION else action_horizon
        commit_count = min(commit_count, max_steps - len(action_records))

        transition_done = False
        committed_actions = 0
        last_model_action: torch.Tensor | None = None
        for chunk_action_index in range(commit_count):
            step_index = len(action_records)
            model_action_tensor = action_pred[:, chunk_action_index : chunk_action_index + 1].detach()
            model_action = model_action_tensor[0, 0].float().cpu().numpy()
            env_action = backend.action_from_model_action(model_action, data_config=data_config)
            last_model_action = model_action_tensor[:, 0, :]

            env_start = time.perf_counter()
            transition = backend.step(env_action)
            env_elapsed = time.perf_counter() - env_start
            observation = transition.observation
            append_observation_to_history(observation)
            rendered = backend.render_frame(observation)
            if rendered is not None:
                video_frames.append(_as_uint8_rgb(rendered, key="render_frame"))
            success = bool(transition.success)

            env_step_times.append(env_elapsed)
            action_records.append(
                {
                    "step_index": step_index,
                    "plan_index": plan_index,
                    "chunk_action_index": chunk_action_index,
                    "action_commit_mode": commit_mode.value,
                    "policy_step_s": policy_elapsed if chunk_action_index == 0 else 0.0,
                    "env_step_s": env_elapsed,
                    "loop_step_s": time.perf_counter() - loop_start,
                    "model_action_dim": int(model_action.shape[-1]),
                    "env_action_dim": int(np.asarray(env_action).reshape(-1).shape[0]),
                    "success": bool(success),
                    "reused_policy_output": chunk_action_index > 0,
                    "sim_action_step_semantics": backend.capabilities.action_step_semantics,
                }
            )
            committed_actions += 1
            if success or transition.done:
                transition_done = True
                break
            if target_action_hz is not None:
                next_deadline += 1.0 / target_action_hz
                sleep_s = next_deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        if commit_mode is SimActionCommitMode.FULL_CHUNK and committed_actions == action_horizon:
            previous_action = action_pred.detach()
        elif last_model_action is not None:
            previous_action = last_model_action.detach()
        plan_index += 1
        if success or transition_done:
            break

    wall_time = time.perf_counter() - rollout_start
    return SimRolloutResult(
        benchmark=backend.benchmark_name,
        task_text=task_text,
        success=bool(success),
        steps=len(action_records),
        target_action_hz=target_action_hz,
        wall_time_s=wall_time,
        mean_policy_step_s=_mean(policy_step_times),
        mean_env_step_s=_mean(env_step_times),
        achieved_action_hz=(len(action_records) / wall_time) if wall_time > 0 else 0.0,
        policy_action_shapes=tuple(policy_action_shapes),
        action_records=tuple(action_records),
        video_frames=tuple(video_frames),
    )


def _normalize_action_commit_mode(value: SimActionCommitMode | str) -> SimActionCommitMode:
    if isinstance(value, SimActionCommitMode):
        return value
    try:
        return SimActionCommitMode(str(value))
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in SimActionCommitMode)
        raise ValueError(f"Unknown action commit mode {value!r}; expected one of: {choices}.") from exc


def summarize_sim_rollout(result: SimRolloutResult, *, video_path: str | None = None) -> dict[str, Any]:
    """Serialize one simulator rollout result without embedding video frames."""

    return {
        "benchmark": result.benchmark,
        "task_text": result.task_text,
        "success": result.success,
        "steps": result.steps,
        "target_action_hz": result.target_action_hz,
        "wall_time_s": result.wall_time_s,
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
        raise ValueError(f"Expected `{key}` RGB image with shape [H, W, 3], got {array.shape}.")
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
