"""Open-WAM simulator adapter for LIBERO benchmark episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from open_wam.configs import (
    ActionTargetRepresentation,
    DataConfig,
    GripperRepresentation,
    LiberoAbsoluteJointExecutionMode,
)
from open_wam.data.action_normalization import (
    denormalize_action_targets,
    denormalize_joint_positions,
)
from open_wam.data.action_pose import (
    PoseSequence,
    axis_angle_to_quaternion,
)
from open_wam.data.action_mapping import inverse_action_mapping
from open_wam.integrations.libero_control import (
    LiberoControlConfig,
    absolute_joint_position_to_libero_joint_delta_action,
    compute_osc_pose_action,
    disable_libero_joint_position_controller_interpolator,
    extract_gripper_positions_from_obs,
    extract_joint_positions_from_obs,
    extract_pose_from_obs,
    gripper_command_for_substep as _raw_gripper_command_for_substep,
    gripper_qpos_tracking_command as _gripper_qpos_tracking_command,
    integrated_eef6d_target_to_osc_action,
    integrated_eef6d_target_to_osc_action_from_arrays as _integrated_eef6d_target_to_osc_action_from_arrays,
    quaternion_angular_error_degrees,
    quaternion_xyzw_to_rotation_matrix as _quaternion_xyzw_to_rotation_matrix_np,
    resolve_libero_joint_delta_limit,
    resolve_libero_joint_limit_array as _joint_limit_array,
    resolve_libero_joint_scale_array as _joint_scale_array,
    set_libero_joint_position_controller_gain,
    step_libero_absolute_joint_position_goal,
)
from open_wam.integrations.libero_runtime import (
    build_libero_control_env,
    build_libero_offscreen_env,
)
from open_wam.integrations.libero_tasks import (
    LiberoTaskSpec,
    ensure_local_libero_config,
    infer_task_local_episode_rank,
    load_libero_task_init_states,
    resolve_libero_task,
    resolve_libero_task_by_id,
)
from open_wam.integrations.libero_tracking import (
    LiberoTrackingResult,
    track_relative_targets_in_libero_env,
)
from open_wam.simulators import EpisodeSpec, SimulatorCapabilities, SimulatorObservation, SimulatorStepResult


_LIBERO_TASK_COMPATIBILITY_EXPORTS = (
    LiberoTaskSpec,
    ensure_local_libero_config,
    infer_task_local_episode_rank,
    load_libero_task_init_states,
    resolve_libero_task,
    resolve_libero_task_by_id,
)


_LIBERO_CONTROL_COMPATIBILITY_EXPORTS = (
    LiberoControlConfig,
    absolute_joint_position_to_libero_joint_delta_action,
    compute_osc_pose_action,
    disable_libero_joint_position_controller_interpolator,
    extract_gripper_positions_from_obs,
    extract_joint_positions_from_obs,
    extract_pose_from_obs,
    integrated_eef6d_target_to_osc_action,
    quaternion_angular_error_degrees,
    resolve_libero_joint_delta_limit,
    set_libero_joint_position_controller_gain,
    step_libero_absolute_joint_position_goal,
)

_LIBERO_RUNTIME_COMPATIBILITY_EXPORTS = (
    build_libero_control_env,
    build_libero_offscreen_env,
)

_LIBERO_TRACKING_COMPATIBILITY_EXPORTS = (
    LiberoTrackingResult,
    track_relative_targets_in_libero_env,
)


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
            return _joint_scale_array(
                self.config.absolute_joint_delta_integration_scale,
                joint_dim=joint_dim,
            )
        if self._joint_delta_limit is not None:
            return _joint_scale_array(self._joint_delta_limit, joint_dim=joint_dim)
        return _joint_scale_array(0.05, joint_dim=joint_dim)


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
