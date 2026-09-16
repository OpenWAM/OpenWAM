"""LIBERO control-plan materialization, independent of model execution."""

from __future__ import annotations
import numpy as np
import torch
from open_wam.configs import ActionTargetRepresentation
from open_wam.configs.enums import DeadlineMissPolicy
from open_wam.data.action_pose import PoseSequence
from open_wam.integrations import libero_rollout
from open_wam.integrations.libero_osc_control import compute_osc_pose_action
from open_wam.runtime.realtime_contracts import PlannedControlStep
from open_wam.integrations.simulator_configs import LiberoControlConfig


def sequence_chunk_to_planned_steps(
    *,
    action_pred: np.ndarray,
    reference_obs: dict[str, np.ndarray],
    generation_action_start: int,
    execution_action_offset: int = 0,
    source: str,
    planner_step_index: int | None,
    ready_monotonic_s: float,
    action_target_representation: ActionTargetRepresentation | str,
    rotation_representation: str,
) -> list[PlannedControlStep]:
    """Project model-space sequence actions into typed LIBERO plan steps."""

    representation = ActionTargetRepresentation(action_target_representation)
    planned_steps: list[PlannedControlStep] = []
    execution_start = int(generation_action_start) - max(
        0,
        int(execution_action_offset),
    )
    if representation == ActionTargetRepresentation.RAW:
        for action_offset in range(action_pred.shape[0]):
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(execution_start + action_offset),
                    generation_action_start=int(generation_action_start),
                    source=str(source),
                    planner_step_index=planner_step_index,
                    ready_monotonic_s=ready_monotonic_s,
                    raw_action=np.asarray(
                        action_pred[action_offset],
                        dtype=np.float32,
                    ).copy(),
                    desired_position=None,
                    desired_quaternion=None,
                    desired_gripper=None,
                )
            )
        return planned_steps

    if representation != ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
        raise ValueError(
            "Unsupported action target representation for sequence rollout: "
            f"{representation!r}."
        )

    desired_pose_targets = libero_rollout.reconstruct_libero_pose_targets(
        action_pred,
        reference_observation=reference_obs,
        rotation_representation=rotation_representation,
    )
    for action_offset in range(action_pred.shape[0]):
        desired_gripper = None
        if desired_pose_targets.gripper is not None:
            desired_gripper = (
                desired_pose_targets.gripper[action_offset]
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        planned_steps.append(
            PlannedControlStep(
                absolute_action_index=int(execution_start + action_offset),
                generation_action_start=int(generation_action_start),
                source=str(source),
                planner_step_index=planner_step_index,
                ready_monotonic_s=ready_monotonic_s,
                raw_action=None,
                desired_position=(
                    desired_pose_targets.position[action_offset]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                ),
                desired_quaternion=(
                    desired_pose_targets.quaternion[action_offset]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                ),
                desired_gripper=desired_gripper,
            )
        )
    return planned_steps


def materialize_sequence_control_action(
    planned_step: PlannedControlStep,
    *,
    current_obs: dict[str, np.ndarray],
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    """Convert a typed raw or absolute-pose plan step to one LIBERO action."""

    if planned_step.raw_action is not None:
        return np.clip(
            np.asarray(planned_step.raw_action, dtype=np.float32),
            -1.0,
            1.0,
        )
    if planned_step.desired_position is None or planned_step.desired_quaternion is None:
        raise RuntimeError("Sequence rollout step is missing absolute pose targets.")
    desired_pose = PoseSequence(
        position=torch.from_numpy(
            np.asarray(planned_step.desired_position, dtype=np.float32)
        ),
        quaternion=torch.from_numpy(
            np.asarray(planned_step.desired_quaternion, dtype=np.float32)
        ),
        gripper=(
            None
            if planned_step.desired_gripper is None
            else torch.from_numpy(
                np.asarray(planned_step.desired_gripper, dtype=np.float32)
            )
        ),
    )
    return compute_osc_pose_action(
        current_pose=libero_rollout.pose_from_libero_observation(current_obs),
        desired_pose=desired_pose,
        control_config=control_config,
        gripper_representation=gripper_representation,
    ).astype(np.float32)


def build_fallback_frame_actions(
    *,
    action_dim: int,
    action_per_frame: int,
    policy: str,
    last_action: np.ndarray,
    preserve_absolute_tail_from: int | None = 6,
) -> np.ndarray:
    deadline_policy = DeadlineMissPolicy(policy)
    if deadline_policy is DeadlineMissPolicy.ZERO:
        return np.zeros((action_per_frame, action_dim), dtype=np.float32)
    if deadline_policy is DeadlineMissPolicy.HOLD_STATE:
        action = np.zeros((action_dim,), dtype=np.float32)
        last = np.asarray(last_action, dtype=np.float32)
        if last.shape != (action_dim,):
            raise ValueError(
                "Fallback last-action shape mismatch, "
                f"expected {(action_dim,)}, got {tuple(last.shape)}."
            )
        if preserve_absolute_tail_from is not None and action_dim > int(
            preserve_absolute_tail_from
        ):
            action[int(preserve_absolute_tail_from) :] = last[
                int(preserve_absolute_tail_from) :
            ]
        return np.repeat(action[None, :], action_per_frame, axis=0)
    repeated = np.repeat(
        np.asarray(last_action, dtype=np.float32)[None, :], action_per_frame, axis=0
    )
    if repeated.shape != (action_per_frame, action_dim):
        raise ValueError(
            "Fallback last-action shape mismatch, "
            f"expected {(action_per_frame, action_dim)}, got {tuple(repeated.shape)}."
        )
    return repeated
