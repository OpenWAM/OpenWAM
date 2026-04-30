from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator

import numpy as np
import yaml

from open_wam.configs import DataConfig
from open_wam.simulators import (
    SimStepResult,
    SimulatorCapabilities,
    normalize_quaternion_xyzw,
    source_action_from_model_action,
)


_CUROBO_IMPORT_STUB_SENTINEL = "_open_wam_curobo_import_stub"
_CUROBO_IMPORT_STUB_USERS = 0
_CUROBO_IMPORT_STUB_MODULES = (
    "curobo",
    "curobo.types",
    "curobo.types.math",
    "curobo.types.robot",
    "curobo.wrap",
    "curobo.wrap.reacher",
    "curobo.wrap.reacher.motion_gen",
    "curobo.util",
    "curobo.util.logger",
)


@dataclass(frozen=True)
class RobotwinEnvConfig:
    """Configuration needed to launch one RoboTwin task environment."""

    robotwin_root: str
    task_name: str
    task_config: str
    instruction: str | None = None
    seed_offset: int = 10000
    action_type: str = "ee"
    expert_precheck: bool = False
    instruction_type: str = "seen"


class RobotwinBenchmarkAdapter:
    """RoboTwin simulator adapter using the official task env API."""

    benchmark_name = "robotwin"
    capabilities = SimulatorCapabilities(
        action_step_semantics="blocking_high_level_target",
        supports_expert_precheck=True,
        action_modes=("ee", "qpos"),
    )

    def __init__(self, config: RobotwinEnvConfig) -> None:
        self.config = config
        self.root = Path(config.robotwin_root).expanduser().resolve()
        self._task_env = None
        self._args: dict[str, Any] | None = None
        self._task_text: str | None = config.instruction
        self._installed_curobo_stub = False
        self._ensure_import_path()

    def reset(self, *, task_id: int | None, episode_idx: int | None, seed: int | None) -> Any:
        if task_id is not None:
            # RoboTwin tasks are name/config driven. Keep task_id accepted for
            # CLI symmetry but do not pretend it maps to official task names.
            pass
        self.close()
        self._args = self._build_task_args()
        self._task_env = self._build_task_env(self.config.task_name)
        if self.config.action_type == "qpos":
            _install_qpos_planner_stub()
        elif self.config.action_type == "ee":
            _install_ee_skip_topp_planner_patch()
        now_ep_num = int(episode_idx or 0)
        resolved_seed = self._resolve_seed(seed=seed, episode_idx=episode_idx)
        generated_instruction = self._run_expert_precheck(now_ep_num=now_ep_num, seed=resolved_seed)
        with self._robotwin_cwd():
            self._task_env.setup_demo(now_ep_num=now_ep_num, seed=resolved_seed, is_test=True, **self._args)
            instruction = self.config.instruction if self.config.instruction is not None else generated_instruction
            if instruction is not None and hasattr(self._task_env, "set_instruction"):
                self._task_env.set_instruction(instruction=instruction)
            self._task_text = self._resolve_task_text()
            return self._task_env.get_obs()

    def task_text(self) -> str | None:
        return self._task_text

    def extract_views(self, observation: Any) -> dict[str, np.ndarray]:
        obs = observation.get("observation", observation)
        high = _extract_camera_rgb(obs, "head_camera")
        left = _extract_camera_rgb(obs, "left_camera")
        right = _extract_camera_rgb(obs, "right_camera")
        return {
            "cam_high": high,
            "cam_left_wrist": left,
            "cam_right_wrist": right,
            "observation.images.cam_high": high,
            "observation.images.cam_left_wrist": left,
            "observation.images.cam_right_wrist": right,
        }

    def extract_state(self, observation: Any) -> np.ndarray | None:
        if self.config.action_type == "ee":
            endpose_state = _extract_endpose_state(observation)
            if endpose_state is not None:
                return endpose_state
        joint_action = observation.get("joint_action")
        if isinstance(joint_action, dict) and "vector" in joint_action:
            return np.asarray(joint_action["vector"], dtype=np.float32)
        return _extract_endpose_state(observation)

    def model_action_to_env_action(self, model_action: np.ndarray, *, data_config: DataConfig) -> np.ndarray:
        source_action = np.asarray(
            source_action_from_model_action(model_action, data_config=data_config),
            dtype=np.float32,
        ).reshape(-1)
        if self.config.action_type == "qpos":
            return _robotwin_qpos_action(source_action)
        if source_action.shape[0] == 16:
            env_action = np.array(source_action, copy=True)
            normalize_quaternion_xyzw(env_action, start=3)
            normalize_quaternion_xyzw(env_action, start=11)
            return env_action
        if source_action.shape[0] == 14:
            return _dual_arm_euler14_to_quat16(source_action)
        raise ValueError(
            "RoboTwin env action adapter expects native 16D EEF action or 14D Euler EEF action, "
            f"got {source_action.shape[0]}D."
        )

    def step(self, env_action: np.ndarray) -> SimStepResult:
        if self._task_env is None:
            raise RuntimeError("RoboTwin adapter must be reset before stepping.")
        with self._robotwin_cwd():
            self._task_env.take_action(env_action, action_type=self.config.action_type)
            observation = self._task_env.get_obs()
        take_action_cnt = getattr(self._task_env, "take_action_cnt", None)
        step_lim = getattr(self._task_env, "step_lim", None)
        hit_step_limit = step_lim is not None and int(take_action_cnt or 0) >= int(step_lim)
        done = bool(getattr(self._task_env, "eval_success", False)) or hit_step_limit
        return SimStepResult(
            observation=observation,
            done=done,
            info={
                "eval_success": bool(getattr(self._task_env, "eval_success", False)),
                "take_action_cnt": None if take_action_cnt is None else int(take_action_cnt),
                "step_lim": None if step_lim is None else int(step_lim),
            },
        )

    def success(self, observation: Any, info: dict[str, Any]) -> bool:
        if bool(info.get("eval_success", False)):
            return True
        if self._task_env is not None and hasattr(self._task_env, "check_success"):
            try:
                with self._robotwin_cwd():
                    return bool(self._task_env.check_success())
            except Exception:
                return False
        return False

    def render_frame(self, observation: Any) -> np.ndarray | None:
        try:
            views = self.extract_views(observation)
        except Exception:
            return None
        frames = [views[key] for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )]
        heights = [frame.shape[0] for frame in frames]
        target_h = max(heights)
        resized = [_resize_nearest_to_height(frame, target_h) for frame in frames]
        return np.concatenate(resized, axis=1)

    def close(self) -> None:
        if self._task_env is not None and hasattr(self._task_env, "close_env"):
            try:
                with self._robotwin_cwd():
                    self._task_env.close_env()
            except Exception:
                pass
        self._task_env = None
        if self._installed_curobo_stub:
            _cleanup_curobo_import_stub()
            self._installed_curobo_stub = False

    def _ensure_import_path(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"RoboTwin root does not exist: {self.root}")
        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        curobo_src = self.root / "envs" / "curobo" / "src"
        if self.config.action_type != "qpos" and curobo_src.exists():
            curobo_src_str = str(curobo_src)
            if curobo_src_str not in sys.path:
                sys.path.insert(0, curobo_src_str)

    def _build_task_env(self, task_name: str):
        with self._robotwin_cwd():
            installed_stub = False
            if self.config.action_type == "qpos":
                installed_stub = _install_curobo_import_stub()
                self._installed_curobo_stub = self._installed_curobo_stub or installed_stub
            try:
                envs_module = importlib.import_module(f"envs.{task_name}")
                env_class = getattr(envs_module, task_name)
                return env_class()
            except Exception:
                if installed_stub:
                    _cleanup_curobo_import_stub()
                    self._installed_curobo_stub = False
                raise

    def _build_task_args(self) -> dict[str, Any]:
        with self._robotwin_cwd():
            args_path = self.root / "task_config" / f"{self.config.task_config}.yml"
            with args_path.open("r", encoding="utf-8") as handle:
                args = yaml.safe_load(handle) or {}
            args["task_name"] = self.config.task_name
            args["task_config"] = self.config.task_config
            args["eval_mode"] = True
            self._populate_embodiment_args(args)
            self._populate_camera_args(args)
            return args

    def _populate_embodiment_args(self, args: dict[str, Any]) -> None:
        try:
            from envs import CONFIGS_PATH  # type: ignore
        except Exception:
            return
        embodiment_type = args.get("embodiment")
        if not embodiment_type:
            return
        embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
        if not embodiment_config_path.exists():
            return
        with embodiment_config_path.open("r", encoding="utf-8") as handle:
            embodiment_types = yaml.safe_load(handle) or {}

        def embodiment_file(name: str) -> str:
            file_path = embodiment_types[name]["file_path"]
            if file_path is None:
                raise ValueError(f"RoboTwin embodiment {name!r} has no file_path.")
            path = Path(file_path)
            if not path.is_absolute():
                path = self.root / path
            return str(path)

        if len(embodiment_type) == 1:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = embodiment_file(embodiment_type[0])
            args["right_robot_file"] = embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("RoboTwin embodiment config should contain one or three entries.")
        args["left_embodiment_config"] = _read_yaml(Path(args["left_robot_file"]) / "config.yml")
        args["right_embodiment_config"] = _read_yaml(Path(args["right_robot_file"]) / "config.yml")

    def _populate_camera_args(self, args: dict[str, Any]) -> None:
        try:
            from envs import CONFIGS_PATH  # type: ignore
        except Exception:
            return
        camera_config_path = Path(CONFIGS_PATH) / "_camera_config.yml"
        if not camera_config_path.exists():
            return
        camera_args = _read_yaml(camera_config_path)
        camera_cfg = args.get("camera", {})
        head_type = camera_cfg.get("head_camera_type")
        if head_type in camera_args:
            args["head_camera_h"] = camera_args[head_type]["h"]
            args["head_camera_w"] = camera_args[head_type]["w"]

    def _resolve_seed(self, *, seed: int | None, episode_idx: int | None) -> int:
        base = int(seed or 0)
        return int(self.config.seed_offset * (1 + base) + int(episode_idx or 0))

    def _resolve_task_text(self) -> str | None:
        if self._task_env is None:
            return self.config.instruction
        if self.config.instruction is not None:
            return self.config.instruction
        if hasattr(self._task_env, "get_instruction"):
            try:
                with self._robotwin_cwd():
                    return str(self._task_env.get_instruction())
            except Exception:
                return None
        return None

    def _run_expert_precheck(self, *, now_ep_num: int, seed: int) -> str | None:
        """Run RoboTwin's expert validation path and return its generated prompt."""

        if not self.config.expert_precheck:
            return None
        if self._task_env is None or self._args is None:
            raise RuntimeError("RoboTwin expert precheck requires a constructed task env.")
        if self.config.action_type == "qpos":
            raise ValueError("RoboTwin expert precheck requires action_type='ee' so the official planner path is active.")

        precheck_args = dict(self._args)
        render_freq = precheck_args.get("render_freq")
        precheck_args["render_freq"] = 0
        with self._robotwin_cwd():
            self._task_env.setup_demo(now_ep_num=now_ep_num, seed=seed, is_test=True, **precheck_args)
            episode_info = self._task_env.play_once()
            plan_success = bool(getattr(self._task_env, "plan_success", False))
            task_success = bool(self._task_env.check_success()) if hasattr(self._task_env, "check_success") else True
            self._task_env.close_env()
        self._task_env = self._build_task_env(self.config.task_name)
        if render_freq is not None:
            self._args["render_freq"] = render_freq
        if not plan_success or not task_success:
            raise RuntimeError(f"RoboTwin expert precheck failed for seed={seed}.")
        return self._generate_instruction_from_episode_info(episode_info)

    def _generate_instruction_from_episode_info(self, episode_info: Any) -> str | None:
        if not isinstance(episode_info, dict) or not isinstance(episode_info.get("info"), dict):
            return None
        try:
            from description.utils.generate_episode_instructions import generate_episode_descriptions  # type: ignore
        except Exception:
            return None
        results = generate_episode_descriptions(self.config.task_name, [episode_info["info"]], 1)
        if not results:
            return None
        choices = results[0].get(self.config.instruction_type)
        if not choices:
            return None
        return str(choices[0])

    @contextmanager
    def _robotwin_cwd(self) -> Iterator[None]:
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            yield
        finally:
            os.chdir(cwd)


def _extract_camera_rgb(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    camera = observation.get(camera_name)
    if isinstance(camera, dict) and "rgb" in camera:
        return np.asarray(camera["rgb"])
    if camera_name in observation:
        return np.asarray(observation[camera_name])
    raise KeyError(f"RoboTwin observation does not expose camera '{camera_name}'.")


def _extract_endpose_state(observation: Any) -> np.ndarray | None:
    endpose = observation.get("endpose")
    if isinstance(endpose, dict):
        left = list(endpose.get("left_endpose", ())) + [endpose.get("left_gripper", 0.0)]
        right = list(endpose.get("right_endpose", ())) + [endpose.get("right_gripper", 0.0)]
        if len(left) == 8 and len(right) == 8:
            return np.asarray([*left, *right], dtype=np.float32)
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _robotwin_qpos_action(source_action: np.ndarray) -> np.ndarray:
    """Return a 14D joint-position action for RoboTwin qpos smoke rollouts."""

    if source_action.shape[0] == 14:
        return np.asarray(source_action, dtype=np.float32)
    if source_action.shape[0] == 16:
        return np.concatenate(
            [
                source_action[0:6],
                source_action[7:8],
                source_action[8:14],
                source_action[15:16],
            ],
            axis=0,
        ).astype(np.float32)
    raise ValueError(f"RoboTwin qpos action adapter expects 14D or 16D source actions, got {source_action.shape[0]}D.")


def _install_qpos_planner_stub() -> None:
    """Avoid CuRobo warmup for qpos-only RoboTwin simulator wiring runs."""

    class QposPlannerStub:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.motion_gen = self

        def TOPP(self, path: np.ndarray, *_: Any, **__: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            position = np.asarray(path, dtype=np.float32)
            velocity = np.zeros_like(position)
            acceleration = np.zeros_like(position)
            times = np.linspace(0.0, 1.0, max(position.shape[0], 1), dtype=np.float32)
            return times, position, velocity, acceleration, float(times[-1]) if times.size else 0.0

        def plan_path(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"status": "Fail"}

        def plan_batch(
            self,
            _curr_joint_pos: Any,
            target_gripper_pose_list: list[Any] | tuple[Any, ...],
            *_: Any,
            **__: Any,
        ) -> dict[str, Any]:
            return {"status": np.asarray(["Failure" for _ in target_gripper_pose_list], dtype=object)}

        def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
            num_step = 200
            result = np.linspace(float(now_val), float(target_val), num_step)
            return {"num_step": num_step, "per_step": (float(target_val) - float(now_val)) / num_step, "result": result}

        def update_point_cloud(self, *_: Any, **__: Any) -> None:
            return None

        def reset(self, *_: Any, **__: Any) -> None:
            return None

    for module_name in ("envs.robot.planner", "envs.robot.robot"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "CuroboPlanner", QposPlannerStub)
    robot_module = sys.modules.get("envs.robot.robot")
    robot_class = getattr(robot_module, "Robot", None) if robot_module is not None else None
    if robot_class is not None:

        def set_planner_stub(self: Any, scene: Any | None = None) -> None:
            del scene
            self.communication_flag = False
            self.left_planner = QposPlannerStub()
            self.right_planner = QposPlannerStub()
            self.left_mplib_planner = QposPlannerStub()
            self.right_mplib_planner = QposPlannerStub()

        robot_class.set_planner = set_planner_stub


def _install_ee_skip_topp_planner_patch() -> None:
    """Skip RoboTwin's qpos-only MPLib TOPP planners for EEF rollouts.

    RoboTwin constructs MPLib TOPP planners during task setup whenever
    ``need_topp`` is true, but the official EEF action path only calls the
    CuRobo ``plan_path`` planners.  Keeping TOPP disabled here avoids native
    SAPIEN/MPLib compatibility crashes while preserving the EEF planner path.
    """

    robot_module = sys.modules.get("envs.robot.robot")
    robot_class = getattr(robot_module, "Robot", None) if robot_module is not None else None
    if robot_class is None or getattr(robot_class, "_open_wam_skip_topp_patch", False):
        return
    original_set_planner = robot_class.set_planner

    def set_planner_without_topp(self: Any, scene: Any | None = None) -> None:
        original_need_topp = getattr(self, "need_topp", False)
        self.need_topp = False
        try:
            original_set_planner(self, scene=scene)
        finally:
            self.need_topp = original_need_topp

    robot_class.set_planner = set_planner_without_topp
    robot_class._open_wam_skip_topp_patch = True


def _install_curobo_import_stub() -> bool:
    """Provide just enough CuRobo symbols for RoboTwin qpos-only imports."""

    global _CUROBO_IMPORT_STUB_USERS

    existing_curobo = sys.modules.get("curobo")
    if existing_curobo is not None and getattr(existing_curobo, _CUROBO_IMPORT_STUB_SENTINEL, False):
        _CUROBO_IMPORT_STUB_USERS += 1
        return True
    if any(
        module_name in sys.modules
        and not getattr(sys.modules[module_name], _CUROBO_IMPORT_STUB_SENTINEL, False)
        for module_name in _CUROBO_IMPORT_STUB_MODULES
    ):
        return False
    try:
        if importlib.util.find_spec("curobo") is not None:
            return False
    except (ImportError, ValueError):
        pass

    curobo = types.ModuleType("curobo")
    curobo_types = types.ModuleType("curobo.types")
    curobo_math = types.ModuleType("curobo.types.math")
    curobo_robot = types.ModuleType("curobo.types.robot")
    curobo_wrap = types.ModuleType("curobo.wrap")
    curobo_reacher = types.ModuleType("curobo.wrap.reacher")
    curobo_motion_gen = types.ModuleType("curobo.wrap.reacher.motion_gen")
    curobo_util = types.ModuleType("curobo.util")
    curobo_logger = types.ModuleType("curobo.util.logger")

    class Pose:
        @classmethod
        def from_list(cls, *_: Any, **__: Any) -> "Pose":
            return cls()

    class JointState:
        @classmethod
        def from_position(cls, *_: Any, **__: Any) -> "JointState":
            return cls()

    class MotionGenConfig:
        @classmethod
        def load_from_robot_config(cls, *_: Any, **__: Any) -> "MotionGenConfig":
            return cls()

    class MotionGen:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.tensor_args = self

        def warmup(self, *_: Any, **__: Any) -> None:
            return None

        def to_device(self, value: Any) -> Any:
            return value

    class MotionGenPlanConfig:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.pose_cost_metric = None

    class PoseCostMetric:
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    def setup_logger(*_: Any, **__: Any) -> None:
        return None

    curobo_math.Pose = Pose
    curobo_robot.JointState = JointState
    curobo_motion_gen.MotionGen = MotionGen
    curobo_motion_gen.MotionGenConfig = MotionGenConfig
    curobo_motion_gen.MotionGenPlanConfig = MotionGenPlanConfig
    curobo_motion_gen.PoseCostMetric = PoseCostMetric
    curobo_logger.setup_logger = setup_logger
    curobo_util.logger = curobo_logger
    curobo.types = curobo_types
    curobo_types.math = curobo_math
    curobo_types.robot = curobo_robot
    curobo.wrap = curobo_wrap
    curobo_wrap.reacher = curobo_reacher
    curobo_reacher.motion_gen = curobo_motion_gen
    curobo.util = curobo_util

    modules = {
        "curobo": curobo,
        "curobo.types": curobo_types,
        "curobo.types.math": curobo_math,
        "curobo.types.robot": curobo_robot,
        "curobo.wrap": curobo_wrap,
        "curobo.wrap.reacher": curobo_reacher,
        "curobo.wrap.reacher.motion_gen": curobo_motion_gen,
        "curobo.util": curobo_util,
        "curobo.util.logger": curobo_logger,
    }
    for module_name, module in modules.items():
        setattr(module, _CUROBO_IMPORT_STUB_SENTINEL, True)
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        if module_name in {"curobo", "curobo.types", "curobo.wrap", "curobo.wrap.reacher", "curobo.util"}:
            module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[module_name] = module
    _CUROBO_IMPORT_STUB_USERS += 1
    return True


def _cleanup_curobo_import_stub() -> None:
    """Remove only the CuRobo modules injected by `_install_curobo_import_stub`."""

    global _CUROBO_IMPORT_STUB_USERS

    _CUROBO_IMPORT_STUB_USERS = max(0, _CUROBO_IMPORT_STUB_USERS - 1)
    if _CUROBO_IMPORT_STUB_USERS > 0:
        return
    for module_name in reversed(_CUROBO_IMPORT_STUB_MODULES):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, _CUROBO_IMPORT_STUB_SENTINEL, False):
            sys.modules.pop(module_name, None)


def _dual_arm_euler14_to_quat16(action: np.ndarray) -> np.ndarray:
    left_quat = _euler_xyz_to_quat_xyzw(action[3:6])
    right_quat = _euler_xyz_to_quat_xyzw(action[10:13])
    return np.concatenate(
        [
            action[0:3],
            left_quat,
            action[6:10],
            right_quat,
            action[13:14],
        ],
        axis=0,
    ).astype(np.float32)


def _euler_xyz_to_quat_xyzw(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in euler]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    quat = np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return quat


def _resize_nearest_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape[0] == target_h:
        return frame
    scale = target_h / frame.shape[0]
    target_w = max(1, int(round(frame.shape[1] * scale)))
    y_indices = np.clip((np.arange(target_h) / scale).astype(np.int64), 0, frame.shape[0] - 1)
    x_indices = np.clip((np.arange(target_w) / scale).astype(np.int64), 0, frame.shape[1] - 1)
    return frame[y_indices][:, x_indices]
