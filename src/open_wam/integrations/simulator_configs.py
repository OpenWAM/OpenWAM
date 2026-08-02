"""Dependency-light configuration records for optional simulator adapters."""

from __future__ import annotations

from dataclasses import dataclass

from open_wam.configs import LiberoAbsoluteJointExecutionMode


@dataclass(frozen=True)
class LiberoControlConfig:
    """Closed-loop gains for converting public targets to OSC actions.

    `OSC_POSE` expects `[dx, dy, dz, dax, day, daz, gripper]`. The first six
    channels are normalized and internally scaled by robosuite to +/- 0.05 m
    and +/- 0.5 rad. Public WAM targets are reference-relative absolute EEF
    targets, so replay reconstructs an absolute target, computes world-frame
    pose error, and normalizes that error for the controller.
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
        if (
            self.action_mode == "absolute_joint_position"
            and self.controller != "JOINT_POSITION"
        ):
            raise ValueError(
                "LIBERO absolute_joint_position mode requires controller='JOINT_POSITION'."
            )
        if self.action_mode == "integrated_eef6d_osc" and self.controller != "OSC_POSE":
            raise ValueError(
                "LIBERO integrated_eef6d_osc mode requires controller='OSC_POSE'."
            )
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
            self.absolute_joint_execution_mode
            is LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA
            and int(self.absolute_joint_substeps_per_target) != 1
        ):
            raise ValueError(
                "absolute_joint_execution_mode='integrated_delta' requires substeps_per_target=1."
            )
        if self.absolute_joint_gripper_substep_policy not in {
            "repeat",
            "first_only",
            "last_only",
        }:
            raise ValueError(
                "absolute_joint_gripper_substep_policy must be one of: repeat, first_only, last_only."
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


@dataclass(frozen=True)
class CalvinEnvConfig:
    """Configuration needed to launch one CALVIN play-table environment."""

    calvin_root: str | None = None
    dataset_root: str | None = None
    task_text: str | None = None
    show_gui: bool = False


__all__ = [
    "CalvinEnvConfig",
    "LiberoControlConfig",
    "LiberoEnvConfig",
    "RobotwinEnvConfig",
]
