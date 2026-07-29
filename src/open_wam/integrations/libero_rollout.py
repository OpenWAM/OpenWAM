from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from open_wam.configs import ActionTargetStateEncoding
from open_wam.data.action_transforms import normalize_quaternion, quaternion_to_axis_angle


def build_libero_state_history(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    state_horizon: int,
    state_encoding: ActionTargetStateEncoding | str,
) -> torch.Tensor:
    """Build policy proprio history from canonical LIBERO observations."""

    if state_horizon <= 0:
        raise ValueError(f"Expected positive state_horizon, got {state_horizon}.")
    if not observations:
        raise ValueError("Cannot build state inputs from an empty observation window.")
    resolved_encoding = ActionTargetStateEncoding(state_encoding)
    state_records = list(observations[-state_horizon:])
    if len(state_records) < state_horizon:
        state_records = [state_records[0]] * (state_horizon - len(state_records)) + state_records
    return torch.stack(
        [
            _build_libero_state_vector(
                observation,
                state_encoding=resolved_encoding,
            )
            for observation in state_records
        ],
        dim=0,
    )


def _build_libero_state_vector(
    observation: Mapping[str, np.ndarray],
    *,
    state_encoding: ActionTargetStateEncoding,
) -> torch.Tensor:
    position = torch.from_numpy(
        np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    )
    quaternion = normalize_quaternion(
        torch.from_numpy(
            np.asarray(observation["robot0_eef_quat"], dtype=np.float32)
        ).unsqueeze(0)
    )[0]
    gripper = torch.from_numpy(
        np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    )

    if state_encoding is ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D:
        axis_angle = quaternion_to_axis_angle(quaternion.unsqueeze(0))[0]
        return torch.cat([position, axis_angle, gripper], dim=0)
    if state_encoding is ActionTargetStateEncoding.EEF_POS_QUAT_GRIPPER_1D:
        return torch.cat([position, quaternion, gripper[:1]], dim=0)
    raise ValueError(
        f"Unsupported LIBERO rollout state encoding: {state_encoding.value}"
    )
