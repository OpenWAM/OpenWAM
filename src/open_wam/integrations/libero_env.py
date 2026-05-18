from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml

from open_wam.configs import (
    ActionTargetRepresentation,
    DataConfig,
    GripperRepresentation,
    LiberoAbsoluteJointExecutionMode,
)
from open_wam.data import reconstruct_absolute_pose_targets
from open_wam.data.action_transforms import (
    PoseSequence,
    axis_angle_to_quaternion,
    collapse_gripper_state,
    denormalize_action_targets,
    denormalize_joint_positions,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
    rotation_matrix_to_quaternion,
)
from open_wam.data.action_mapping import inverse_action_mapping
from open_wam.simulators import EpisodeSpec, SimulatorCapabilities, SimulatorObservation, SimulatorStepResult


@dataclass(frozen=True)
class LiberoTaskSpec:
    """Resolved LIBERO benchmark task used to construct a simulator scene."""

    benchmark_name: str
    task_id: int
    task_name: str
    task_language: str
    problem_folder: str
    bddl_file_path: str
    init_states_path: str


@dataclass(frozen=True)
class LiberoControlConfig:
    """Closed-loop tracking gains for converting our public targets to OSC actions.

    `OSC_POSE` expects a 7D action `[dx, dy, dz, dax, day, daz, gripper]`.
    The first six channels are normalized and internally scaled by robosuite to
    +/- 0.05 m and +/- 0.5 rad respectively. The public WAM target, however, is
    a reference-relative absolute EEF target `[rel_xyz, rel_axis_angle, gripper]`
    expressed against a dataset-defined anchor pose.
    We therefore:
    1. reconstruct the desired absolute EEF target from the stored reference pose
    2. compute current world-frame pose error
    3. normalize that error into the controller's expected action range
    """

    max_pos_delta_m: float = 0.05
    max_rot_delta_rad: float = 0.5
    max_gripper_delta: float = 0.005
    control_substeps_per_target: int = 8
    env_control_hz: int = 20
    action_command_delay_steps: int = 1
    gripper_open_threshold: float = 0.060
    gripper_close_threshold: float = 0.030
    gripper_position_tolerance: float = 0.002


@dataclass(frozen=True)
class LiberoEnvConfig:
    """LIBERO simulator backend configuration.

    `action_mode=absolute_joint_position` constructs LIBERO with robosuite's
    `JOINT_POSITION` controller. Public model targets are interpreted as
    absolute Panda joint qpos plus either a scalar gripper command or measured
    gripper qpos targets, depending on the data action-target config.
    """

    benchmark_name: str = "libero_10"
    controller: str = "OSC_POSE"
    action_mode: str = "osc_pose_delta"
    env_backend: str = "offscreen"
    use_camera_obs: bool = True
    has_offscreen_renderer: bool = True
    camera_obs_keys: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image")
    render_camera_key: str = "agentview_image"
    camera_height: int = 128
    camera_width: int = 128
    horizon: int = 5000
    ignore_done: bool = True
    control_freq: int | None = None
    init_state_index: int | None = None
    joint_delta_limit_rad: float | tuple[float, ...] | None = None
    absolute_joint_execution_mode: LiberoAbsoluteJointExecutionMode | str = (
        LiberoAbsoluteJointExecutionMode.NORMALIZED_DELTA
    )
    absolute_joint_substeps_per_target: int = 1
    absolute_joint_gripper_substep_policy: str = "repeat"
    absolute_joint_kp: float | None = None
    absolute_joint_disable_interpolator: bool = False
    absolute_joint_delta_integration_scale: float | tuple[float, ...] | None = None
    integrated_eef_position_scale: float = 0.010576533139391671
    integrated_eef_rotation_scale: float = 0.1136411594890211

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "absolute_joint_execution_mode",
            LiberoAbsoluteJointExecutionMode(self.absolute_joint_execution_mode),
        )
        if self.action_mode == "absolute_joint_position" and self.controller != "JOINT_POSITION":
            raise ValueError("LIBERO absolute_joint_position mode requires controller='JOINT_POSITION'.")
        if self.action_mode == "integrated_eef6d_osc" and self.controller != "OSC_POSE":
            raise ValueError("LIBERO integrated_eef6d_osc mode requires controller='OSC_POSE'.")
        if self.action_mode == "integrated_eef6d_osc":
            if abs(float(self.integrated_eef_position_scale)) <= 1e-12:
                raise ValueError("integrated_eef_position_scale must be nonzero.")
            if abs(float(self.integrated_eef_rotation_scale)) <= 1e-12:
                raise ValueError("integrated_eef_rotation_scale must be nonzero.")
        if self.env_backend not in {"offscreen", "control"}:
            raise ValueError("LIBERO env_backend must be one of: offscreen, control.")
        if int(self.absolute_joint_substeps_per_target) < 1:
            raise ValueError("absolute_joint_substeps_per_target must be >= 1.")
        if (
            self.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA
            and int(self.absolute_joint_substeps_per_target) != 1
        ):
            raise ValueError("absolute_joint_execution_mode='integrated_delta' requires substeps_per_target=1.")
        if self.absolute_joint_gripper_substep_policy not in {"repeat", "first_only", "last_only"}:
            raise ValueError(
                "absolute_joint_gripper_substep_policy must be one of: repeat, first_only, last_only."
            )


@dataclass(frozen=True)
class LiberoTrackingResult:
    """Trajectory rollout and tracking metrics from a LIBERO env replay."""

    task_spec: LiberoTaskSpec
    init_state_index: int
    desired_pose: PoseSequence
    tracked_pose: PoseSequence
    position_error_per_target: torch.Tensor
    rotation_error_deg_per_target: torch.Tensor
    gripper_error_per_target: torch.Tensor
    camera_frames: dict[str, list[np.ndarray]]
    rendered_target_indices: list[int]


def ensure_local_libero_config(project_root: Path | None = None) -> Path:
    """Bootstrap LIBERO's config file without interactive prompts.

    The original LIBERO package prompts on import if `~/.libero/config.yaml`
    does not exist. Collaborative tooling should not depend on interactive setup,
    so Open-WAM writes a local config into `.cache/libero_config/` and points
    `LIBERO_CONFIG_PATH` there before importing the upstream package.
    """

    root = _project_root(project_root)
    libero_repo_root, libero_package_root = _resolve_libero_paths()
    config_dir = root / ".cache" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": str(libero_package_root.resolve()),
        "bddl_files": str((libero_package_root / "bddl_files").resolve()),
        "init_states": str((libero_package_root / "init_files").resolve()),
        "datasets": str((libero_repo_root / "libero" / "datasets").resolve()),
        "assets": str((libero_package_root / "assets").resolve()),
    }
    config_path = config_dir / "config.yaml"
    config_text = yaml.safe_dump(config, sort_keys=False)
    if not config_path.is_file() or config_path.read_text(encoding="utf-8") != config_text:
        tmp_path = config_path.with_name(f"{config_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(config_text, encoding="utf-8")
        tmp_path.replace(config_path)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_path


def resolve_libero_task(
    task_text: str,
    project_root: Path | None = None,
    *,
    benchmark_name: str | None = None,
) -> LiberoTaskSpec:
    """Resolve dataset task text to one upstream LIBERO benchmark task."""

    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    normalized_task_text = _normalize_task_text(task_text)
    matches: list[LiberoTaskSpec] = []
    benchmark_classes = benchmark.get_benchmark_dict()
    if benchmark_name is not None:
        try:
            benchmark_items = ((benchmark_name, benchmark_classes[benchmark_name]),)
        except KeyError as exc:
            available = ", ".join(sorted(benchmark_classes))
            raise ValueError(
                f"Unknown LIBERO benchmark {benchmark_name!r}; available benchmarks: {available}"
            ) from exc
    else:
        benchmark_items = tuple(benchmark_classes.items())

    for current_benchmark_name, benchmark_class in benchmark_items:
        try:
            benchmark_instance = benchmark_class()
        except Exception:
            # Upstream registers suites such as LIBERO_100 that are not fully
            # initialized in this checkout. Task resolution should ignore those
            # and keep searching the benchmark variants that are usable.
            continue
        for task_id in range(benchmark_instance.get_num_tasks()):
            task = benchmark_instance.get_task(task_id)
            if _normalize_task_text(task.language) != normalized_task_text:
                continue
            matches.append(
                LiberoTaskSpec(
                    benchmark_name=current_benchmark_name,
                    task_id=task_id,
                    task_name=task.name,
                    task_language=task.language,
                    problem_folder=task.problem_folder,
                    bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
                    init_states_path=os.path.join(
                        os.environ["LIBERO_CONFIG_PATH"],
                        "..",
                    ),  # overwritten below for clarity
                )
            )

    if not matches:
        raise ValueError(f"Could not resolve LIBERO task text: {task_text!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Task text {task_text!r} matched multiple LIBERO tasks; expected exactly one. "
            f"Matches: {[match.task_name for match in matches]}"
        )

    match = matches[0]
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return LiberoTaskSpec(
        benchmark_name=match.benchmark_name,
        task_id=match.task_id,
        task_name=match.task_name,
        task_language=match.task_language,
        problem_folder=match.problem_folder,
        bddl_file_path=match.bddl_file_path,
        init_states_path=str(Path(config["init_states"]) / match.problem_folder / f"{match.task_name}.pruned_init"),
    )


def resolve_libero_task_by_id(
    benchmark_name: str,
    task_id: int,
    project_root: Path | None = None,
) -> LiberoTaskSpec:
    """Resolve one LIBERO benchmark/task-id pair into a task spec."""

    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    benchmark_classes = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_classes:
        available = ", ".join(sorted(benchmark_classes))
        raise ValueError(f"Unknown LIBERO benchmark {benchmark_name!r}; available benchmarks: {available}")
    benchmark_instance = benchmark_classes[benchmark_name]()
    task_id = int(task_id)
    if task_id < 0 or task_id >= benchmark_instance.get_num_tasks():
        raise ValueError(
            f"task_id={task_id} is outside benchmark {benchmark_name!r} "
            f"with {benchmark_instance.get_num_tasks()} tasks."
        )
    task = benchmark_instance.get_task(task_id)
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return LiberoTaskSpec(
        benchmark_name=benchmark_name,
        task_id=task_id,
        task_name=task.name,
        task_language=task.language,
        problem_folder=task.problem_folder,
        bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
        init_states_path=str(Path(config["init_states"]) / task.problem_folder / f"{task.name}.pruned_init"),
    )


def load_libero_task_init_states(task_spec: LiberoTaskSpec, project_root: Path | None = None) -> Any:
    """Load benchmark init states with torch 2.6-compatible semantics."""

    ensure_local_libero_config(project_root)
    # Upstream uses `torch.load(path)` which defaults to `weights_only=True`
    # on torch 2.6+. The init-state files are not weight checkpoints.
    return torch.load(task_spec.init_states_path, weights_only=False)


def build_libero_offscreen_env(
    task_spec: LiberoTaskSpec,
    *,
    controller: str = "OSC_POSE",
    camera_height: int = 256,
    camera_width: int = 256,
    horizon: int = 5000,
    ignore_done: bool = True,
    control_freq: int | None = None,
    project_root: Path | None = None,
):
    """Construct one offscreen LIBERO environment for evaluation."""

    ensure_local_libero_config(project_root)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    env_kwargs: dict[str, Any] = {}
    if control_freq is not None:
        env_kwargs["control_freq"] = int(control_freq)

    return OffScreenRenderEnv(
        bddl_file_name=task_spec.bddl_file_path,
        controller=controller,
        camera_heights=camera_height,
        camera_widths=camera_width,
        horizon=horizon,
        ignore_done=ignore_done,
        **env_kwargs,
    )


def build_libero_control_env(
    task_spec: LiberoTaskSpec,
    *,
    controller: str = "OSC_POSE",
    camera_height: int = 256,
    camera_width: int = 256,
    horizon: int = 5000,
    ignore_done: bool = False,
    control_freq: int | None = None,
    use_camera_obs: bool = False,
    has_offscreen_renderer: bool = False,
    project_root: Path | None = None,
):
    """Construct LIBERO's ControlEnv with explicit render/camera knobs."""

    ensure_local_libero_config(project_root)
    from libero.libero.envs.env_wrapper import ControlEnv  # type: ignore

    env_kwargs: dict[str, Any] = {}
    if control_freq is not None:
        env_kwargs["control_freq"] = int(control_freq)

    return ControlEnv(
        bddl_file_name=task_spec.bddl_file_path,
        controller=controller,
        use_camera_obs=bool(use_camera_obs),
        has_offscreen_renderer=bool(has_offscreen_renderer),
        has_renderer=False,
        camera_heights=camera_height,
        camera_widths=camera_width,
        horizon=horizon,
        ignore_done=ignore_done,
        **env_kwargs,
    )


def infer_task_local_episode_rank(
    episode_records: list[Any] | tuple[Any, ...],
    *,
    episode_index: int,
    task_text: str,
) -> int:
    """Best-effort mapping from dataset episode to per-task demo index.

    The HF export does not expose the original demo id. The stable fallback is
    the count of prior episodes with the same task text. This is sufficient for
    reproducible env rollouts and often aligns with the original demo ordering.
    """

    normalized = _normalize_task_text(task_text)
    rank = 0
    for record in episode_records:
        record_task = record.tasks[0] if getattr(record, "tasks", None) else ""
        if int(record.episode_index) == episode_index:
            return rank
        if _normalize_task_text(record_task) == normalized:
            rank += 1
    raise ValueError(f"Episode index {episode_index} was not found in episode metadata.")


def extract_pose_from_obs(obs: dict[str, Any]) -> PoseSequence:
    """Parse LIBERO / robosuite observation dict into the common pose contract."""

    quaternion_xyzw = torch.tensor(obs["robot0_eef_quat"], dtype=torch.float32)
    return PoseSequence(
        position=torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32),
        quaternion=normalize_quaternion(quaternion_xyzw),
        gripper=torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32),
    )


def extract_joint_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda arm qpos from a LIBERO / robosuite observation."""

    if "robot0_joint_pos" not in obs:
        raise KeyError("LIBERO observation does not expose `robot0_joint_pos`.")
    joint_positions = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    if joint_positions.size == 0:
        raise ValueError("LIBERO `robot0_joint_pos` is empty.")
    return joint_positions


def extract_gripper_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda gripper qpos from a LIBERO / robosuite observation."""

    if "robot0_gripper_qpos" not in obs:
        raise KeyError("LIBERO observation does not expose `robot0_gripper_qpos`.")
    values = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LIBERO `robot0_gripper_qpos` is empty.")
    return values


def resolve_libero_joint_delta_limit(
    env: Any,
    *,
    fallback: float | tuple[float, ...] = 0.05,
    joint_dim: int = 7,
) -> np.ndarray:
    """Infer normalized JOINT_POSITION delta scaling from robosuite controller config."""

    fallback_limit = _joint_limit_array(fallback, joint_dim=joint_dim)
    robots = getattr(getattr(env, "env", env), "robots", None)
    if not robots:
        return fallback_limit
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        return fallback_limit
    output_max = getattr(controller, "output_max", None)
    output_min = getattr(controller, "output_min", None)
    if output_max is None:
        return fallback_limit
    max_values = np.asarray(output_max, dtype=np.float32).reshape(-1)
    if max_values.size < joint_dim:
        return fallback_limit
    if output_min is not None:
        min_values = np.asarray(output_min, dtype=np.float32).reshape(-1)
        if min_values.size >= joint_dim:
            max_values = np.maximum(np.abs(max_values[:joint_dim]), np.abs(min_values[:joint_dim]))
        else:
            max_values = np.abs(max_values[:joint_dim])
    else:
        max_values = np.abs(max_values[:joint_dim])
    if np.any(max_values <= 0.0):
        return fallback_limit
    return max_values.astype(np.float32)


def absolute_joint_position_to_libero_joint_delta_action(
    *,
    target_joint_positions: np.ndarray,
    current_joint_positions: np.ndarray,
    gripper_command: float = 0.0,
    joint_delta_limit_rad: float | tuple[float, ...] | np.ndarray = 0.05,
) -> np.ndarray:
    """Convert absolute joint qpos targets to normalized LIBERO `JOINT_POSITION` actions."""

    target = np.asarray(target_joint_positions, dtype=np.float32).reshape(-1)
    current = np.asarray(current_joint_positions, dtype=np.float32).reshape(-1)
    if target.shape != current.shape:
        raise ValueError(f"Target/current joint shapes must match, got {target.shape} and {current.shape}.")
    limits = _joint_limit_array(joint_delta_limit_rad, joint_dim=target.shape[0])
    arm_action = np.clip((target - current) / limits, -1.0, 1.0)
    gripper = np.asarray([float(np.clip(gripper_command, -1.0, 1.0))], dtype=np.float32)
    return np.concatenate([arm_action.astype(np.float32), gripper], axis=0)


def step_libero_absolute_joint_position_goal(
    env: Any,
    *,
    target_joint_positions: np.ndarray,
    gripper_command: float = 0.0,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """Step a LIBERO `JOINT_POSITION` env with an absolute joint-position goal.

    Robosuite's public `JOINT_POSITION` action is a normalized relative delta.
    For dataset replay validation we also need the stricter semantic of
    "track this absolute qpos target now". The upstream controller already has
    that hook via `set_goal(..., set_qpos=target)`, but the normal `env.step`
    path does not expose it. This helper keeps gripper actuation on the normal
    env action path and temporarily redirects only the arm goal update.
    """

    target = np.asarray(target_joint_positions, dtype=np.float32).reshape(-1)
    robot = _first_libero_robot(env)
    controller = getattr(robot, "controller", None)
    if controller is None or not hasattr(controller, "set_goal"):
        raise ValueError("LIBERO env does not expose a robosuite arm controller with `set_goal`.")
    control_dim = int(getattr(controller, "control_dim", target.shape[0]))
    if control_dim < target.shape[0]:
        raise ValueError(
            f"Controller control_dim={control_dim} is smaller than target joint dim={target.shape[0]}."
        )

    action_dim = int(getattr(robot, "action_dim", control_dim + 1))
    action = np.zeros(action_dim, dtype=np.float32)
    action[:control_dim] = 0.0
    if action_dim > control_dim:
        action[control_dim:] = float(np.clip(gripper_command, -1.0, 1.0))

    original_set_goal = controller.set_goal

    def _set_absolute_goal(action_arg: Any, *args: Any, **kwargs: Any) -> Any:
        del action_arg, args, kwargs
        return original_set_goal(np.zeros(control_dim, dtype=np.float32), set_qpos=target)

    controller.set_goal = _set_absolute_goal
    try:
        return env.step(action)
    finally:
        controller.set_goal = original_set_goal


def set_libero_joint_position_controller_gain(env: Any, *, kp: float) -> None:
    """Override JOINT_POSITION controller gains for deterministic absolute-goal tracking."""

    controller = getattr(_first_libero_robot(env), "controller", None)
    if controller is None:
        raise ValueError("LIBERO env robot does not expose a controller for gain override.")
    joint_dim = int(getattr(controller, "control_dim", 7))
    controller.kp = np.full(joint_dim, float(kp), dtype=np.float64)
    controller.kd = 2.0 * np.sqrt(controller.kp)


def disable_libero_joint_position_controller_interpolator(env: Any) -> None:
    """Disable robosuite's JOINT_POSITION interpolator for exact absolute-goal tracking."""

    controller = getattr(_first_libero_robot(env), "controller", None)
    if controller is None:
        raise ValueError("LIBERO env robot does not expose a controller for interpolator override.")
    controller.interpolator = None


def _raw_gripper_command_for_substep(
    command: float,
    *,
    substep_index: int,
    substeps: int,
    policy: str,
) -> float:
    if policy == "repeat":
        return float(command)
    if policy == "first_only":
        return float(command) if int(substep_index) == 0 else 0.0
    if policy == "last_only":
        return float(command) if int(substep_index) == int(substeps) - 1 else 0.0
    raise ValueError(f"Unknown gripper substep policy: {policy!r}.")


def _gripper_opening(gripper_positions: np.ndarray) -> float:
    values = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
    if values.size >= 2:
        return float(values[0] - values[1])
    return float(values[0])


def _gripper_qpos_tracking_command(
    *,
    current_gripper_positions: np.ndarray,
    target_gripper_positions: np.ndarray,
    tolerance: float = 0.001,
) -> float:
    target_values = np.asarray(target_gripper_positions, dtype=np.float32).reshape(-1)
    current_values = np.asarray(current_gripper_positions, dtype=np.float32).reshape(-1)
    if target_values.size == 1:
        current_value = float(current_values[0])
        target_value = float(target_values[0])
    else:
        current_value = _gripper_opening(current_values)
        target_value = _gripper_opening(target_values)
    if current_value > target_value + float(tolerance):
        return 1.0
    if current_value < target_value - float(tolerance):
        return -1.0
    return 0.0


def _first_libero_robot(env: Any) -> Any:
    robots = getattr(env, "robots", None)
    if robots is None:
        inner_env = getattr(env, "env", None)
        robots = getattr(inner_env, "robots", None)
    if not robots:
        raise ValueError("LIBERO env does not expose any robot handles.")
    return robots[0]


class LiberoBenchmarkAdapter:
    """Normalized LIBERO simulator backend.

    This adapter supports the legacy OSC delta action path and the new
    absolute-joint-position model target path. Absolute joint targets are
    either converted to the public normalized `JOINT_POSITION` delta action or
    executed through the adapter-owned absolute `set_qpos` controller hook.
    Dataset conversion and policy rollout should use the same configured mode.
    """

    def __init__(self, config: LiberoEnvConfig | None = None, *, project_root: Path | None = None) -> None:
        self.config = config or LiberoEnvConfig()
        self.project_root = project_root
        self.benchmark_name = self.config.benchmark_name
        self.capabilities = SimulatorCapabilities(
            action_step_semantics=(
                f"absolute_joint_{self.config.absolute_joint_execution_mode.value}"
                if self.config.action_mode == "absolute_joint_position"
                else "single_env_step"
            ),
            supports_render=True,
            supports_success=True,
            action_modes=(self.config.action_mode,),
        )
        self._env: Any | None = None
        self._task_spec: LiberoTaskSpec | None = None
        self._task_text: str | None = None
        self._last_obs: dict[str, Any] | None = None
        self._joint_delta_limit: np.ndarray | None = None
        self._absolute_joint_previous_target_qpos: np.ndarray | None = None
        self._integrated_eef_previous_target: PoseSequence | None = None
        self._integrated_eef_previous_position: np.ndarray | None = None
        self._integrated_eef_previous_rotation_matrix: np.ndarray | None = None
        self._pending_absolute_joint_gripper_representation: GripperRepresentation | None = None

    def reset(self, spec: EpisodeSpec) -> SimulatorObservation:
        task_id = 0 if spec.task_id is None else int(spec.task_id)
        task_spec = resolve_libero_task_by_id(self.config.benchmark_name, task_id, project_root=self.project_root)
        init_states = load_libero_task_init_states(task_spec, project_root=self.project_root)
        init_state_index = (
            self.config.init_state_index
            if self.config.init_state_index is not None
            else (0 if spec.episode_idx is None else int(spec.episode_idx))
        )
        init_state_index = int(np.clip(init_state_index, 0, len(init_states) - 1))

        self.close()
        if self.config.env_backend == "control":
            self._env = build_libero_control_env(
                task_spec,
                controller=self.config.controller,
                camera_height=self.config.camera_height,
                camera_width=self.config.camera_width,
                horizon=self.config.horizon,
                ignore_done=self.config.ignore_done,
                control_freq=self.config.control_freq,
                use_camera_obs=bool(self.config.use_camera_obs),
                has_offscreen_renderer=bool(self.config.has_offscreen_renderer),
                project_root=self.project_root,
            )
        else:
            self._env = build_libero_offscreen_env(
                task_spec,
                controller=self.config.controller,
                camera_height=self.config.camera_height,
                camera_width=self.config.camera_width,
                horizon=self.config.horizon,
                ignore_done=self.config.ignore_done,
                control_freq=self.config.control_freq,
                project_root=self.project_root,
            )
        if spec.seed is not None:
            reset_seed = int(spec.seed)
            random.seed(reset_seed)
            np.random.seed(reset_seed % (2**32 - 1))
            if hasattr(self._env, "seed"):
                self._env.seed(reset_seed)
        obs = self._env.reset()
        obs = self._env.set_init_state(init_states[init_state_index])
        if (
            self.config.action_mode == "absolute_joint_position"
            and (
                self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.DIRECT_GOAL
                or self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA
                or int(self.config.absolute_joint_substeps_per_target) > 1
            )
        ):
            if self.config.absolute_joint_kp is not None:
                set_libero_joint_position_controller_gain(self._env, kp=float(self.config.absolute_joint_kp))
            if self.config.absolute_joint_disable_interpolator:
                disable_libero_joint_position_controller_interpolator(self._env)
        self._task_spec = task_spec
        self._task_text = task_spec.task_language
        self._last_obs = obs
        self._joint_delta_limit = resolve_libero_joint_delta_limit(
            self._env,
            fallback=0.05 if self.config.joint_delta_limit_rad is None else self.config.joint_delta_limit_rad,
            joint_dim=extract_joint_positions_from_obs(obs).shape[0],
        )
        self._absolute_joint_previous_target_qpos = extract_joint_positions_from_obs(obs).astype(np.float32, copy=True)
        self._integrated_eef_previous_target = extract_pose_from_obs(obs)
        self._integrated_eef_previous_position = self._integrated_eef_previous_target.position.detach().cpu().numpy()
        self._integrated_eef_previous_rotation_matrix = _quaternion_xyzw_to_rotation_matrix_np(
            self._integrated_eef_previous_target.quaternion.detach().cpu().numpy()
        )
        return self._normalize_observation(obs, init_state_index=init_state_index)

    def task_text(self) -> str | None:
        return self._task_text

    def set_integrated_eef6d_previous_target_from_state(self, state: np.ndarray) -> None:
        """Set the previous pseudo-target anchor from `[xyz, axis_angle, ...]` state.

        Dataset replay uses this to match the exact initial observation that
        generated an integrated EEF6D target sequence. Online rollouts can omit
        it and default to the simulator's reset observation.
        """

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape[0] < 6:
            raise ValueError(f"Expected EEF state with at least 6 dims, got {state_array.shape[0]}.")
        self._integrated_eef_previous_position = state_array[0:3].astype(np.float32, copy=True)
        quaternion = axis_angle_to_quaternion(torch.as_tensor(state_array[3:6], dtype=torch.float32).unsqueeze(0))[0]
        self._integrated_eef_previous_rotation_matrix = _quaternion_xyzw_to_rotation_matrix_np(
            quaternion.detach().cpu().numpy()
        )
        self._integrated_eef_previous_target = PoseSequence(
            position=torch.as_tensor(self._integrated_eef_previous_position, dtype=torch.float32),
            quaternion=quaternion,
            gripper=None,
        )

    def action_from_model_action(self, model_action: np.ndarray, *, data_config: DataConfig) -> np.ndarray:
        source_action = _source_action_from_model_action(model_action, data_config=data_config)
        if data_config.action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION:
            if self._last_obs is None:
                raise RuntimeError("LIBERO adapter must be reset before converting absolute joint targets.")
            current_qpos = extract_joint_positions_from_obs(self._last_obs)
            joint_dim = current_qpos.shape[0]
            if source_action.shape[0] < joint_dim:
                raise ValueError(
                    f"Absolute-joint model action has dim {source_action.shape[0]}, "
                    f"but current LIBERO joint state has dim {joint_dim}."
                )
            normalized_target_qpos = torch.as_tensor(source_action[:joint_dim], dtype=torch.float32).unsqueeze(0)
            target_qpos = denormalize_joint_positions(
                normalized_target_qpos,
                normalization=data_config.action_target.joint_position_normalization,
            )[0].detach().cpu().numpy()
            gripper_representation = GripperRepresentation(data_config.action_target.gripper_representation)
            gripper_values = source_action[joint_dim:]
            if gripper_values.size == 0:
                gripper_command = 0.0
            elif gripper_representation == GripperRepresentation.ACTION_COMMAND:
                gripper_command = float(gripper_values[0])
            elif gripper_representation in {
                GripperRepresentation.FIRST_CHANNEL,
                GripperRepresentation.ALL_CHANNELS,
            }:
                gripper_command = _gripper_qpos_tracking_command(
                    current_gripper_positions=extract_gripper_positions_from_obs(self._last_obs),
                    target_gripper_positions=np.asarray(gripper_values, dtype=np.float32),
                )
            else:
                raise ValueError(
                    f"Unsupported absolute-joint gripper representation: "
                    f"{gripper_representation}"
                )
            limit = self._joint_delta_limit
            if limit is None:
                limit = _joint_limit_array(0.05, joint_dim=joint_dim)
            if (
                self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.DIRECT_GOAL
                or self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA
                or int(self.config.absolute_joint_substeps_per_target) > 1
            ):
                self._pending_absolute_joint_gripper_representation = gripper_representation
                if gripper_representation == GripperRepresentation.ACTION_COMMAND:
                    direct_goal_tail = np.asarray([gripper_command], dtype=np.float32)
                else:
                    direct_goal_tail = np.asarray(gripper_values, dtype=np.float32).reshape(-1)
                return np.concatenate(
                    [
                        np.asarray(target_qpos, dtype=np.float32),
                        direct_goal_tail,
                    ],
                    axis=0,
                )
            return absolute_joint_position_to_libero_joint_delta_action(
                target_joint_positions=target_qpos,
                current_joint_positions=current_qpos,
                gripper_command=gripper_command,
                joint_delta_limit_rad=limit,
            )

        if self.config.action_mode == "integrated_eef6d_osc":
            if self._integrated_eef_previous_position is None or self._integrated_eef_previous_rotation_matrix is None:
                if self._last_obs is None:
                    raise RuntimeError("LIBERO adapter must be reset before converting integrated EEF targets.")
                previous_pose = extract_pose_from_obs(self._last_obs)
                self._integrated_eef_previous_position = previous_pose.position.detach().cpu().numpy()
                self._integrated_eef_previous_rotation_matrix = _quaternion_xyzw_to_rotation_matrix_np(
                    previous_pose.quaternion.detach().cpu().numpy()
                )
            action, next_position, next_rotation = _integrated_eef6d_target_to_osc_action_from_arrays(
                previous_position=self._integrated_eef_previous_position,
                previous_rotation_matrix=self._integrated_eef_previous_rotation_matrix,
                target=source_action,
                position_scale=float(self.config.integrated_eef_position_scale),
                rotation_scale=float(self.config.integrated_eef_rotation_scale),
            )
            self._integrated_eef_previous_position = next_position
            self._integrated_eef_previous_rotation_matrix = next_rotation
            return action

        return source_action.astype(np.float32, copy=False)

    def step(self, action: np.ndarray) -> SimulatorStepResult:
        if self._env is None:
            raise RuntimeError("LIBERO adapter must be reset before step().")
        if (
            self.config.action_mode == "absolute_joint_position"
            and (
                self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.DIRECT_GOAL
                or self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA
                or int(self.config.absolute_joint_substeps_per_target) > 1
            )
        ):
            obs, reward, done, info = self._step_absolute_joint_target(action)
        else:
            obs, reward, done, info = self._env.step(np.asarray(action, dtype=np.float32))
        self._last_obs = obs
        success = bool(self._env.check_success()) if hasattr(self._env, "check_success") else False
        return SimulatorStepResult(
            observation=self._normalize_observation(obs),
            reward=float(reward) if reward is not None else None,
            done=bool(done),
            success=success,
            info=dict(info or {}),
        )

    def _step_absolute_joint_target(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float | None, bool, dict[str, Any]]:
        if self._env is None or self._last_obs is None:
            raise RuntimeError("LIBERO adapter must be reset before absolute-joint target stepping.")
        payload = np.asarray(action, dtype=np.float32).reshape(-1)
        joint_dim = extract_joint_positions_from_obs(self._last_obs).shape[0]
        if payload.shape[0] < joint_dim:
            raise ValueError(
                f"Absolute-joint direct-goal action has dim {payload.shape[0]}, "
                f"but LIBERO joint state has dim {joint_dim}."
            )
        target_qpos = payload[:joint_dim]
        gripper_payload = payload[joint_dim:]
        gripper_representation = self._pending_absolute_joint_gripper_representation
        if gripper_representation is None:
            gripper_representation = GripperRepresentation.ACTION_COMMAND
        reward: float | None = None
        done = False
        info: dict[str, Any] = {}
        obs: dict[str, Any] = self._last_obs
        substeps = int(self.config.absolute_joint_substeps_per_target)
        executed_substeps = 0
        for substep_index in range(substeps):
            if gripper_payload.size == 0:
                command = 0.0
            elif gripper_representation == GripperRepresentation.ACTION_COMMAND:
                command = _raw_gripper_command_for_substep(
                    float(gripper_payload[0]),
                    substep_index=substep_index,
                    substeps=substeps,
                    policy=self.config.absolute_joint_gripper_substep_policy,
                )
            else:
                command = _gripper_qpos_tracking_command(
                    current_gripper_positions=extract_gripper_positions_from_obs(obs),
                    target_gripper_positions=gripper_payload,
                )
            if self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.DIRECT_GOAL:
                obs, reward, done, step_info = step_libero_absolute_joint_position_goal(
                    self._env,
                    target_joint_positions=target_qpos,
                    gripper_command=command,
                )
            elif self.config.absolute_joint_execution_mode is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA:
                previous_target_qpos = self._absolute_joint_previous_target_qpos
                if previous_target_qpos is None:
                    previous_target_qpos = extract_joint_positions_from_obs(obs)
                scale = self._absolute_joint_delta_integration_scale(joint_dim=joint_dim)
                arm_action = np.clip(
                    (np.asarray(target_qpos, dtype=np.float32) - np.asarray(previous_target_qpos, dtype=np.float32))
                    / scale,
                    -1.0,
                    1.0,
                )
                env_action = np.concatenate(
                    [arm_action.astype(np.float32), np.asarray([float(np.clip(command, -1.0, 1.0))], dtype=np.float32)]
                )
                obs, reward, done, step_info = self._env.step(env_action)
                self._absolute_joint_previous_target_qpos = target_qpos.astype(np.float32, copy=True)
            else:
                current_qpos = extract_joint_positions_from_obs(obs)
                limit = self._joint_delta_limit
                if limit is None:
                    limit = _joint_limit_array(0.05, joint_dim=joint_dim)
                env_action = absolute_joint_position_to_libero_joint_delta_action(
                    target_joint_positions=target_qpos,
                    current_joint_positions=current_qpos,
                    gripper_command=command,
                    joint_delta_limit_rad=limit,
                )
                obs, reward, done, step_info = self._env.step(env_action)
            executed_substeps = substep_index + 1
            info = dict(step_info or {})
            if bool(done) or (hasattr(self._env, "check_success") and bool(self._env.check_success())):
                break
        current_qpos = extract_joint_positions_from_obs(obs)
        qpos_error = current_qpos - target_qpos
        info.update(
            {
                "absolute_joint_execution_mode": self.config.absolute_joint_execution_mode.value,
                "absolute_joint_env_substeps": int(executed_substeps),
                "absolute_joint_target_qpos": target_qpos.astype(np.float32).copy(),
                "absolute_joint_qpos_l2_error": float(np.linalg.norm(qpos_error)),
                "absolute_joint_qpos_linf_error": float(np.max(np.abs(qpos_error))),
            }
        )
        return obs, reward, bool(done), info

    def render_frame(self, observation: SimulatorObservation) -> np.ndarray | None:
        if self.config.render_camera_key in observation.views:
            return np.asarray(observation.views[self.config.render_camera_key], dtype=np.uint8)
        if observation.views:
            first_key = next(iter(observation.views))
            return np.asarray(observation.views[first_key], dtype=np.uint8)
        return None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._last_obs = None
        self._joint_delta_limit = None
        self._absolute_joint_previous_target_qpos = None
        self._integrated_eef_previous_target = None
        self._integrated_eef_previous_position = None
        self._integrated_eef_previous_rotation_matrix = None
        self._pending_absolute_joint_gripper_representation = None

    def _normalize_observation(self, obs: dict[str, Any], *, init_state_index: int | None = None) -> SimulatorObservation:
        views = {
            camera_key: np.asarray(obs[camera_key], dtype=np.uint8)
            for camera_key in self.config.camera_obs_keys
            if camera_key in obs
        }
        state = extract_joint_positions_from_obs(obs)
        metadata: dict[str, Any] = {}
        if init_state_index is not None:
            metadata["init_state_index"] = int(init_state_index)
        return SimulatorObservation(
            views=views,
            state=state,
            task_text=self._task_text,
            raw=obs,
            metadata=metadata,
        )

    def _absolute_joint_delta_integration_scale(self, *, joint_dim: int) -> np.ndarray:
        if self.config.absolute_joint_delta_integration_scale is not None:
            return _joint_scale_array(self.config.absolute_joint_delta_integration_scale, joint_dim=joint_dim)
        if self._joint_delta_limit is not None:
            return _joint_scale_array(self._joint_delta_limit, joint_dim=joint_dim)
        return _joint_scale_array(0.05, joint_dim=joint_dim)


def compute_osc_pose_action(
    *,
    current_pose: PoseSequence,
    desired_pose: PoseSequence,
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    """Convert one desired absolute pose into one normalized `OSC_POSE` action."""

    position_error = desired_pose.position - current_pose.position

    delta_quaternion = quaternion_multiply(
        desired_pose.quaternion.unsqueeze(0),
        quaternion_inverse(current_pose.quaternion).unsqueeze(0),
    )[0]
    delta_axis_angle = quaternion_to_axis_angle(normalize_quaternion(delta_quaternion.unsqueeze(0)))[0]

    position_command = torch.clamp(position_error / control_config.max_pos_delta_m, min=-1.0, max=1.0)
    rotation_command = torch.clamp(delta_axis_angle / control_config.max_rot_delta_rad, min=-1.0, max=1.0)

    if desired_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    elif gripper_representation == "action_command":
        # When the public target carries the raw LIBERO gripper command, replay
        # should pass that command through directly instead of re-interpreting
        # it as a finger-joint state target.
        gripper_command = desired_pose.gripper[0:1].clamp(min=-1.0, max=1.0).to(dtype=torch.float32)
    elif current_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    else:
        current_public = _project_gripper_state(
            current_pose.gripper,
            gripper_representation=gripper_representation,
        )

        if gripper_representation == "all_channels":
            # LIBERO exposes two finger joints in state. When the public target
            # keeps both channels, interpret them via the jaw opening.
            current_value = current_pose.gripper[0] - current_pose.gripper[1]
            desired_value = desired_pose.gripper[0] - desired_pose.gripper[1]
            open_threshold = control_config.gripper_open_threshold
            close_threshold = control_config.gripper_close_threshold
            tolerance = control_config.gripper_position_tolerance
        elif gripper_representation == "first_channel":
            # The default public representation keeps only the first finger
            # qpos. For Panda this is roughly half of the jaw opening.
            current_value = current_public[0]
            desired_value = desired_pose.gripper[0]
            open_threshold = control_config.gripper_open_threshold * 0.5
            close_threshold = control_config.gripper_close_threshold * 0.5
            tolerance = control_config.gripper_position_tolerance * 0.5
        else:
            raise ValueError(f"Unsupported gripper representation: {gripper_representation}")

        error_value = desired_value - current_value
        if desired_value >= open_threshold:
            gripper_command = torch.tensor([-1.0], dtype=torch.float32)
        elif desired_value <= close_threshold:
            gripper_command = torch.tensor([1.0], dtype=torch.float32)
        elif torch.abs(error_value) <= tolerance:
            gripper_command = torch.tensor([0.0], dtype=torch.float32)
        else:
            gripper_command = torch.clamp(
                -error_value / control_config.max_gripper_delta,
                min=-1.0,
                max=1.0,
            ).reshape(1)

    action = torch.cat([position_command, rotation_command, gripper_command], dim=0)
    return action.detach().cpu().numpy().astype(np.float32)


def integrated_eef6d_target_to_osc_action(
    *,
    previous_target: PoseSequence,
    target: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> tuple[np.ndarray, PoseSequence]:
    """Recover one LIBERO OSC action from a pseudo-absolute EEF-6D target.

    The target contract is `[absolute_xyz, continuous_rotation_6d, gripper]`.
    It is intentionally differenced against the previous pseudo-target, not the
    measured current pose, so dataset construction can be exactly invertible
    back to the source 7D OSC command.
    """

    action, target_position, target_rotation = _integrated_eef6d_target_to_osc_action_from_arrays(
        previous_position=previous_target.position.detach().cpu().numpy(),
        previous_rotation_matrix=_quaternion_xyzw_to_rotation_matrix_np(previous_target.quaternion.detach().cpu().numpy()),
        target=target,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
    )
    next_target = PoseSequence(
        position=torch.as_tensor(target_position, dtype=torch.float32),
        quaternion=rotation_matrix_to_quaternion(torch.as_tensor(target_rotation, dtype=torch.float32).unsqueeze(0))[0],
        gripper=torch.as_tensor([float(action[6])], dtype=torch.float32),
    )
    return action, next_target


def _integrated_eef6d_target_to_osc_action_from_arrays(
    *,
    previous_position: np.ndarray,
    previous_rotation_matrix: np.ndarray,
    target: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.asarray(target, dtype=np.float32).reshape(-1)
    if payload.shape[0] < 10:
        raise ValueError(f"Expected integrated EEF6D target with at least 10 dims, got {payload.shape[0]}.")
    if abs(float(position_scale)) <= 1e-12 or abs(float(rotation_scale)) <= 1e-12:
        raise ValueError("Integrated EEF position and rotation scales must be nonzero.")
    target_position = payload[0:3].astype(np.float32, copy=True)
    target_rotation = _continuous_6d_to_rotation_matrix_np(payload[3:9]).astype(np.float32, copy=False)
    delta_axis_angle = _relative_rotation_matrix_to_axis_angle_np(
        target_rotation,
        np.asarray(previous_rotation_matrix, dtype=np.float32),
    )
    position_command = np.clip(
        (target_position - np.asarray(previous_position, dtype=np.float32)) / float(position_scale),
        -1.0,
        1.0,
    )
    rotation_command = np.clip(delta_axis_angle / float(rotation_scale), -1.0, 1.0)
    action = np.concatenate(
        [
            position_command.astype(np.float32, copy=False),
            rotation_command.astype(np.float32, copy=False),
            np.asarray([float(np.clip(payload[9], -1.0, 1.0))], dtype=np.float32),
        ],
        axis=0,
    )
    return action.astype(np.float32, copy=False), target_position, target_rotation.astype(np.float32, copy=False)


def _quaternion_xyzw_to_rotation_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    x, y, z, w = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _continuous_6d_to_rotation_matrix_np(rotation_6d: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation_6d, dtype=np.float64).reshape(6)
    first = _normalize_np(rot[0:3])
    second_raw = rot[3:6] - float(np.dot(first, rot[3:6])) * first
    if float(np.linalg.norm(second_raw)) <= 1e-8:
        seed = np.asarray([0.0, 1.0, 0.0] if abs(float(first[0])) > 0.9 else [1.0, 0.0, 0.0], dtype=np.float64)
        second_raw = np.cross(first, seed)
    second = _normalize_np(second_raw)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1).astype(np.float32)


def _normalize_np(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _relative_rotation_matrix_to_axis_angle_np(target: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) @ np.asarray(previous, dtype=np.float64).T
    return _rotation_matrix_to_axis_angle_np(delta)


def _rotation_matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(mat))
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    vee = np.asarray(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ],
        dtype=np.float64,
    )
    if angle <= 1e-6:
        return (0.5 * vee).astype(np.float32)
    return (vee / max(2.0 * float(np.sin(angle)), 1e-12) * angle).astype(np.float32)


def track_relative_targets_in_libero_env(
    *,
    task_text: str,
    relative_pose_targets: torch.Tensor,
    rotation_representation: str,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    gripper_representation: str = "first_channel",
    init_state_index: int = 0,
    control_config: LiberoControlConfig | None = None,
    camera_obs_keys: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image"),
    camera_height: int = 256,
    camera_width: int = 256,
    project_root: Path | None = None,
) -> LiberoTrackingResult:
    """Replay one public WAM trajectory in the real LIBERO simulator.

    The public representation is reference-relative. Replay must therefore use
    the same reference pose that was used to build the public targets in the
    dataset adapter. For episode-mode LIBERO targets that is the first dataset
    frame; for sample-mode targets it is the sample's anchor state.
    """

    if control_config is None:
        control_config = LiberoControlConfig()

    task_spec = resolve_libero_task(task_text, project_root=project_root)
    init_states = load_libero_task_init_states(task_spec, project_root=project_root)
    init_state_index = int(np.clip(init_state_index, 0, len(init_states) - 1))

    env = build_libero_offscreen_env(
        task_spec,
        camera_height=camera_height,
        camera_width=camera_width,
        horizon=max(5000, int(relative_pose_targets.shape[0] * control_config.control_substeps_per_target + 32)),
        ignore_done=True,
        project_root=project_root,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(init_states[init_state_index])
        desired_pose = reconstruct_absolute_pose_targets(
            reference_position=reference_position,
            reference_quaternion=reference_quaternion,
            relative_pose_targets=relative_pose_targets,
            rotation_representation=rotation_representation,
        )
        aligned_gripper_targets = _align_replay_gripper_targets(
            desired_pose.gripper,
            gripper_representation=gripper_representation,
            delay_steps=control_config.action_command_delay_steps,
        )

        tracked_positions: list[torch.Tensor] = []
        tracked_quaternions: list[torch.Tensor] = []
        tracked_gripper: list[torch.Tensor] = []
        rendered_target_indices: list[int] = []
        camera_frames: dict[str, list[np.ndarray]] = {camera_key: [] for camera_key in camera_obs_keys}

        for target_index in range(relative_pose_targets.shape[0]):
            target_pose = PoseSequence(
                position=desired_pose.position[target_index],
                quaternion=desired_pose.quaternion[target_index],
                gripper=None if aligned_gripper_targets is None else aligned_gripper_targets[target_index],
            )
            for _ in range(control_config.control_substeps_per_target):
                current_pose = extract_pose_from_obs(obs)
                action = compute_osc_pose_action(
                    current_pose=current_pose,
                    desired_pose=target_pose,
                    control_config=control_config,
                    gripper_representation=gripper_representation,
                )
                obs, _, _, _ = env.step(action)
                rendered_target_indices.append(target_index)
                for camera_key in camera_obs_keys:
                    camera_frames[camera_key].append(np.array(obs[camera_key], copy=True))

            final_pose = extract_pose_from_obs(obs)
            tracked_positions.append(final_pose.position)
            tracked_quaternions.append(final_pose.quaternion)
            if final_pose.gripper is not None:
                if gripper_representation == "action_command":
                    tracked_gripper.append(torch.tensor([float(action[-1])], dtype=torch.float32))
                    continue
                tracked_gripper.append(
                    _project_gripper_state(
                        final_pose.gripper,
                        gripper_representation=gripper_representation,
                    )
                )

        tracked_pose = PoseSequence(
            position=torch.stack(tracked_positions, dim=0),
            quaternion=torch.stack(tracked_quaternions, dim=0),
            gripper=torch.stack(tracked_gripper, dim=0) if tracked_gripper else None,
        )

        position_error_per_target = torch.linalg.vector_norm(
            tracked_pose.position - desired_pose.position,
            dim=-1,
        )
        rotation_error_deg_per_target = quaternion_angular_error_degrees(
            tracked_pose.quaternion,
            desired_pose.quaternion,
        )
        if desired_pose.gripper is not None and tracked_pose.gripper is not None:
            gripper_error_per_target = torch.linalg.vector_norm(
                tracked_pose.gripper - aligned_gripper_targets,
                dim=-1,
            )
        else:
            gripper_error_per_target = torch.zeros_like(position_error_per_target)

        return LiberoTrackingResult(
            task_spec=task_spec,
            init_state_index=init_state_index,
            desired_pose=desired_pose,
            tracked_pose=tracked_pose,
            position_error_per_target=position_error_per_target,
            rotation_error_deg_per_target=rotation_error_deg_per_target,
            gripper_error_per_target=gripper_error_per_target,
            camera_frames=camera_frames,
            rendered_target_indices=rendered_target_indices,
        )
    finally:
        env.close()


def quaternion_angular_error_degrees(lhs_xyzw: torch.Tensor, rhs_xyzw: torch.Tensor) -> torch.Tensor:
    lhs = normalize_quaternion(lhs_xyzw)
    rhs = normalize_quaternion(rhs_xyzw)
    dot = (lhs * rhs).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))


def _source_action_from_model_action(
    model_action: np.ndarray,
    *,
    data_config: DataConfig,
) -> np.ndarray:
    tensor = torch.as_tensor(model_action, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    source = inverse_action_mapping(tensor, data_config.action_mapping)
    source = denormalize_action_targets(source, normalization=data_config.action_target.normalization)
    array = source.detach().cpu().numpy().astype(np.float32)
    return array[0] if squeeze else array


def _joint_limit_array(
    value: float | tuple[float, ...] | np.ndarray,
    *,
    joint_dim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.full(joint_dim, float(array[0]), dtype=np.float32)
    if array.size != joint_dim:
        raise ValueError(f"Expected {joint_dim} joint delta limits, got {array.size}.")
    if np.any(array <= 0.0):
        raise ValueError("Joint delta limits must be positive.")
    return array.astype(np.float32)


def _joint_scale_array(
    value: float | tuple[float, ...] | np.ndarray,
    *,
    joint_dim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.full(joint_dim, float(array[0]), dtype=np.float32)
    if array.size != joint_dim:
        raise ValueError(f"Expected {joint_dim} joint integration scales, got {array.size}.")
    if np.any(np.isclose(array, 0.0)):
        raise ValueError("Joint integration scales must be nonzero.")
    return array.astype(np.float32)


def _normalize_task_text(task_text: str) -> str:
    return " ".join(task_text.strip().lower().split())


def _project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_libero_paths() -> tuple[Path, Path]:
    """Resolve the installed LIBERO repo root and package root from Python imports.

    Upstream LIBERO uses an unusual nested package layout:
    `<repo>/libero/libero/__init__.py`.
    Some local installs therefore record distribution metadata without exposing
    an importable `libero` package. When that happens, fall back to a checkout
    path so the current uv environment can still import `libero.libero`.
    """

    env_repo_root = os.environ.get("LIBERO_REPO_ROOT")
    if env_repo_root:
        env_paths = _libero_paths_from_repo_root(Path(env_repo_root).expanduser())
        if env_paths is not None:
            return env_paths

    import_error: Exception | None = None
    try:
        libero_pkg = importlib.import_module("libero.libero")
    except EOFError as exc:
        # Upstream LIBERO can prompt on import when its config file has not
        # been bootstrapped yet, which raises EOFError in non-interactive
        # contexts. Fall back to a checkout path without importing so
        # `ensure_local_libero_config(...)` can write the config first.
        import_error = exc
        libero_pkg = None
    except ModuleNotFoundError as exc:
        if exc.name not in {"libero", "libero.libero"}:
            raise
        import_error = exc
        libero_pkg = None

    if libero_pkg is not None:
        package_root = Path(libero_pkg.__file__).resolve().parent
        repo_root = package_root.parents[1]
        return repo_root, package_root

    fallback_repo_roots: list[Path] = []

    project_root = _project_root(None)
    fallback_repo_roots.append(project_root.parent / "LIBERO")

    for repo_root in fallback_repo_roots:
        paths = _libero_paths_from_repo_root(repo_root)
        if paths is not None:
            return paths

    raise ImportError(
        "LIBERO could not be imported. Either install an importable LIBERO package into the uv environment "
        "or set LIBERO_REPO_ROOT to a checkout whose structure contains `libero/libero/__init__.py`."
    ) from import_error


def _libero_paths_from_repo_root(repo_root: Path) -> tuple[Path, Path] | None:
    package_root = repo_root / "libero" / "libero"
    if not (package_root / "__init__.py").exists():
        return None
    repo_root_resolved = repo_root.resolve()
    repo_root_str = str(repo_root_resolved)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root_resolved, package_root.resolve()


def _project_gripper_state(gripper_state: torch.Tensor, *, gripper_representation: str) -> torch.Tensor:
    """Expose one env gripper state in the same public representation as targets."""

    if gripper_state.ndim != 1:
        raise ValueError(f"Expected one gripper state vector, got shape {tuple(gripper_state.shape)}.")
    if gripper_representation == "action_command":
        raise ValueError(
            "action_command is a control-domain target and cannot be recovered from env gripper state alone."
        )
    return collapse_gripper_state(
        gripper_state.unsqueeze(0),
        gripper_representation=gripper_representation,
    )[0]


def _align_replay_gripper_targets(
    gripper_targets: torch.Tensor | None,
    *,
    gripper_representation: str,
    delay_steps: int,
) -> torch.Tensor | None:
    """Shift command-domain gripper targets to the state they actually produce.

    LIBERO's 1D action gripper command is causal: `action[t]` drives the
    transition from state `t` toward state `t+1`. For replay we compare against
    pose targets at state-aligned timesteps, so the command must be delayed by
    one target to avoid visibly closing / opening too early.
    """

    if gripper_targets is None:
        return None
    if gripper_representation != "action_command":
        return gripper_targets
    if delay_steps < 0:
        raise ValueError(f"Expected non-negative action_command_delay_steps, got {delay_steps}.")

    aligned = torch.zeros_like(gripper_targets)
    if delay_steps == 0:
        aligned.copy_(gripper_targets)
        return aligned
    if delay_steps >= gripper_targets.shape[0]:
        return aligned

    aligned[delay_steps:] = gripper_targets[:-delay_steps]
    return aligned
