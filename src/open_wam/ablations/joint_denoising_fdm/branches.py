from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


ActionBranchFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class ActionBranchSpec:
    """Counterfactual action intervention used for FDM/IDM coverage."""

    name: str
    family: str
    strength: str
    description: str
    is_stochastic: bool = False
    is_ood: bool = False
    preserves_gripper: bool = True

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


def _copy(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions, dtype=np.float32).copy()


def _clip_control(actions: np.ndarray) -> np.ndarray:
    actions[:, :6] = np.clip(actions[:, :6], -1.0, 1.0)
    if actions.shape[1] > 6:
        actions[:, 6:] = np.clip(actions[:, 6:], -1.0, 1.0)
    return actions.astype(np.float32, copy=False)


def _gt(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    return _copy(actions)


def _noise_small(actions: np.ndarray, seed: int) -> np.ndarray:
    output = _copy(actions)
    rng = np.random.default_rng(int(seed))
    output[:, :6] += rng.normal(loc=0.0, scale=0.05, size=output[:, :6].shape).astype(np.float32)
    return _clip_control(output)


def _bias_x_pos(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    output[:, 0] += 0.12
    return _clip_control(output)


def _bias_y_neg(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    output[:, 1] -= 0.12
    return _clip_control(output)


def _stop_motion(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    output[:, :6] = 0.0
    return _clip_control(output)


def _reverse_translation(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    output[:, :3] *= -1.0
    return _clip_control(output)


def _swap_xy_clockwise(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    original_x = output[:, 0].copy()
    output[:, 0] = output[:, 1]
    output[:, 1] = -original_x
    return _clip_control(output)


def _strong_axis(axis: int, value: float) -> ActionBranchFn:
    def apply(actions: np.ndarray, seed: int) -> np.ndarray:
        del seed
        output = _copy(actions)
        output[:, axis] += np.float32(value)
        return _clip_control(output)

    return apply


def _saturate_axis(axis: int, value: float) -> ActionBranchFn:
    def apply(actions: np.ndarray, seed: int) -> np.ndarray:
        del seed
        output = _copy(actions)
        output[:, :6] = 0.0
        output[:, axis] = np.float32(value)
        return _clip_control(output)

    return apply


def _scale_demo(scale: float) -> ActionBranchFn:
    def apply(actions: np.ndarray, seed: int) -> np.ndarray:
        del seed
        output = _copy(actions)
        output[:, :6] *= np.float32(scale)
        return _clip_control(output)

    return apply


def _axis_pulse(axis: int, value: float, duty_cycle: float = 0.5) -> ActionBranchFn:
    def apply(actions: np.ndarray, seed: int) -> np.ndarray:
        del seed
        output = _copy(actions)
        output[:, :6] = 0.0
        pulse_len = max(1, int(round(output.shape[0] * float(duty_cycle))))
        output[:pulse_len, axis] = np.float32(value)
        return _clip_control(output)

    return apply


def _rotation_pulse(axis: int, value: float) -> ActionBranchFn:
    return _axis_pulse(axis, value, duty_cycle=0.5)


def _gripper_set(value: float) -> ActionBranchFn:
    def apply(actions: np.ndarray, seed: int) -> np.ndarray:
        del seed
        output = _copy(actions)
        if output.shape[1] <= 6:
            return output
        output[:, 6] = np.float32(value)
        return _clip_control(output)

    return apply


def _gripper_toggle(actions: np.ndarray, seed: int) -> np.ndarray:
    del seed
    output = _copy(actions)
    if output.shape[1] > 6:
        output[:, 6] = -np.sign(output[:, 6])
        output[output[:, 6] == 0.0, 6] = 1.0
    return _clip_control(output)


_BRANCH_IMPLS: dict[str, ActionBranchFn] = {
    "gt": _gt,
    "noise_small": _noise_small,
    "bias_x_pos": _bias_x_pos,
    "bias_y_neg": _bias_y_neg,
    "stop_motion": _stop_motion,
    "reverse_translation": _reverse_translation,
    "swap_xy_clockwise": _swap_xy_clockwise,
    "strong_y_neg": _strong_axis(1, -0.6),
    "strong_x_pos": _strong_axis(0, 0.6),
    "push_down": _strong_axis(2, -0.6),
    "saturate_x_pos": _saturate_axis(0, 1.0),
    "saturate_x_neg": _saturate_axis(0, -1.0),
    "saturate_y_pos": _saturate_axis(1, 1.0),
    "saturate_y_neg": _saturate_axis(1, -1.0),
    "saturate_z_up": _saturate_axis(2, 1.0),
    "saturate_z_down": _saturate_axis(2, -1.0),
    "scale_demo_0p25": _scale_demo(0.25),
    "scale_demo_0p5": _scale_demo(0.5),
    "scale_demo_1p5": _scale_demo(1.5),
    "axis_pulse_x_neg": _axis_pulse(0, -0.8),
    "axis_pulse_y_neg": _axis_pulse(1, -0.8),
    "axis_pulse_z_down": _axis_pulse(2, -0.8),
    "axis_pulse_z_up": _axis_pulse(2, 0.8),
    "rotation_yaw_pos": _rotation_pulse(5, 0.8),
    "rotation_yaw_neg": _rotation_pulse(5, -0.8),
    "gripper_open": _gripper_set(1.0),
    "gripper_close": _gripper_set(-1.0),
    "gripper_toggle": _gripper_toggle,
}


ACTION_BRANCH_SPECS: dict[str, ActionBranchSpec] = {
    "gt": ActionBranchSpec("gt", "demo", "none", "Recorded demonstration future."),
    "noise_small": ActionBranchSpec(
        "noise_small",
        "noise",
        "weak",
        "Small iid Gaussian perturbation on the six OSC channels.",
        is_stochastic=True,
    ),
    "bias_x_pos": ActionBranchSpec("bias_x_pos", "bias", "weak", "Add a weak positive x translation bias."),
    "bias_y_neg": ActionBranchSpec("bias_y_neg", "bias", "weak", "Add a weak negative y translation bias."),
    "stop_motion": ActionBranchSpec("stop_motion", "hold", "strong", "Zero the six OSC channels."),
    "reverse_translation": ActionBranchSpec(
        "reverse_translation",
        "transform",
        "medium",
        "Reverse xyz translation while preserving rotation and gripper.",
    ),
    "swap_xy_clockwise": ActionBranchSpec(
        "swap_xy_clockwise",
        "transform",
        "medium",
        "Rotate xy translation direction clockwise.",
    ),
    "strong_y_neg": ActionBranchSpec("strong_y_neg", "bias", "strong", "Add a strong negative y translation bias."),
    "strong_x_pos": ActionBranchSpec("strong_x_pos", "bias", "strong", "Add a strong positive x translation bias."),
    "push_down": ActionBranchSpec("push_down", "bias", "strong", "Add a strong negative z translation bias."),
    "saturate_x_pos": ActionBranchSpec("saturate_x_pos", "saturated_axis", "ood", "Saturated positive x command.", is_ood=True),
    "saturate_x_neg": ActionBranchSpec("saturate_x_neg", "saturated_axis", "ood", "Saturated negative x command.", is_ood=True),
    "saturate_y_pos": ActionBranchSpec("saturate_y_pos", "saturated_axis", "ood", "Saturated positive y command.", is_ood=True),
    "saturate_y_neg": ActionBranchSpec("saturate_y_neg", "saturated_axis", "ood", "Saturated negative y command.", is_ood=True),
    "saturate_z_up": ActionBranchSpec("saturate_z_up", "saturated_axis", "ood", "Saturated positive z command.", is_ood=True),
    "saturate_z_down": ActionBranchSpec("saturate_z_down", "saturated_axis", "ood", "Saturated negative z command.", is_ood=True),
    "scale_demo_0p25": ActionBranchSpec("scale_demo_0p25", "demo_scale", "weak", "Scale demo OSC channels by 0.25."),
    "scale_demo_0p5": ActionBranchSpec("scale_demo_0p5", "demo_scale", "weak", "Scale demo OSC channels by 0.5."),
    "scale_demo_1p5": ActionBranchSpec("scale_demo_1p5", "demo_scale", "medium", "Scale demo OSC channels by 1.5."),
    "axis_pulse_x_neg": ActionBranchSpec("axis_pulse_x_neg", "axis_pulse", "strong", "Half-horizon negative x pulse."),
    "axis_pulse_y_neg": ActionBranchSpec("axis_pulse_y_neg", "axis_pulse", "strong", "Half-horizon negative y pulse."),
    "axis_pulse_z_down": ActionBranchSpec("axis_pulse_z_down", "axis_pulse", "strong", "Half-horizon negative z pulse."),
    "axis_pulse_z_up": ActionBranchSpec("axis_pulse_z_up", "axis_pulse", "strong", "Half-horizon positive z pulse."),
    "rotation_yaw_pos": ActionBranchSpec("rotation_yaw_pos", "rotation_pulse", "strong", "Half-horizon positive yaw pulse."),
    "rotation_yaw_neg": ActionBranchSpec("rotation_yaw_neg", "rotation_pulse", "strong", "Half-horizon negative yaw pulse."),
    "gripper_open": ActionBranchSpec(
        "gripper_open",
        "gripper",
        "medium",
        "Set the gripper channel to +1.",
        preserves_gripper=False,
    ),
    "gripper_close": ActionBranchSpec(
        "gripper_close",
        "gripper",
        "medium",
        "Set the gripper channel to -1.",
        preserves_gripper=False,
    ),
    "gripper_toggle": ActionBranchSpec(
        "gripper_toggle",
        "gripper",
        "medium",
        "Flip the sign of the recorded gripper channel.",
        preserves_gripper=False,
    ),
}


BRANCH_PRESETS: dict[str, tuple[str, ...]] = {
    "diagnostic": (
        "gt",
        "stop_motion",
        "reverse_translation",
        "swap_xy_clockwise",
        "strong_y_neg",
    ),
    "training_10": (
        "gt",
        "stop_motion",
        "scale_demo_0p25",
        "scale_demo_0p5",
        "scale_demo_1p5",
        "reverse_translation",
        "axis_pulse_x_neg",
        "axis_pulse_y_neg",
        "axis_pulse_z_down",
        "gripper_toggle",
    ),
    "coverage_14": (
        "gt",
        "stop_motion",
        "scale_demo_0p25",
        "scale_demo_0p5",
        "scale_demo_1p5",
        "reverse_translation",
        "swap_xy_clockwise",
        "axis_pulse_x_neg",
        "axis_pulse_y_neg",
        "axis_pulse_z_down",
        "axis_pulse_z_up",
        "rotation_yaw_pos",
        "gripper_open",
        "gripper_close",
    ),
    "saturated_eval": (
        "stop_motion",
        "saturate_x_neg",
        "saturate_y_neg",
        "saturate_z_down",
    ),
}


def apply_action_branch(actions: np.ndarray, *, branch_name: str, seed: int) -> np.ndarray:
    try:
        impl = _BRANCH_IMPLS[branch_name]
    except KeyError as exc:
        raise ValueError(f"Unknown counterfactual action branch: {branch_name!r}") from exc
    return impl(actions, int(seed))


def branch_metadata(branch_name: str) -> dict[str, object]:
    try:
        return ACTION_BRANCH_SPECS[branch_name].to_metadata()
    except KeyError as exc:
        raise ValueError(f"Unknown counterfactual action branch: {branch_name!r}") from exc


def branch_seed_offset(branch_name: str) -> int:
    return sum(ord(char) for char in branch_name)


def expand_branch_names(value: str) -> tuple[str, ...]:
    names: list[str] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        preset = BRANCH_PRESETS.get(item)
        if preset is not None:
            names.extend(preset)
        else:
            names.append(item)
    unknown = [name for name in names if name not in ACTION_BRANCH_SPECS]
    if unknown:
        raise ValueError(f"Unknown counterfactual action branches: {unknown}")
    return tuple(dict.fromkeys(names))
