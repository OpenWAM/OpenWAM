"""LIBERO observation preparation for exact evaluation.

This optional module owns simulator-facing observation and runtime-input glue.
Legacy rendering helpers are identity aliases to the artifact owner; this
module does not define policy, cache, or sequence semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from open_wam.configs import ProprioContextMode
from open_wam.data.action_pose import quaternion_to_axis_angle
from open_wam.evals.libero_rollout_artifacts import (
    to_uint8 as _artifact_to_uint8,
)
from open_wam.evals.libero_rollout_artifacts import (
    with_title as _artifact_with_title,
)
from open_wam.integrations import (
    LIBERO_ROLLOUT_VIEW_KEYS,
    LiberoTaskSpec,
    resolve_libero_task_by_id,
)

LIBERO_OBS_KEYS = LIBERO_ROLLOUT_VIEW_KEYS
to_uint8 = _artifact_to_uint8
with_title = _artifact_with_title


def resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    """Resolve one benchmark task and return its model prompt."""

    task_spec = resolve_libero_task_by_id(benchmark_name, task_id)
    return task_spec, task_spec.task_language


def initialize_raw_observation(env: Any, init_state: Any) -> Mapping[str, Any]:
    """Reset a LIBERO environment and return its five-step startup observation."""

    env.reset()
    env.set_init_state(init_state)
    observation = None
    for _ in range(5):
        observation, _, _, _ = env.step([0.0] * 7)
    if observation is None:
        raise RuntimeError(
            "LIBERO env did not return an observation during initialization."
        )
    return observation


def extract_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract vertically corrected RGB views under canonical rollout keys."""

    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(observation["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1]
        ),
    }


def proprio_context_enabled(config: Any) -> bool:
    """Return whether the selected policy consumes exact-rollout proprio."""

    policy_config = getattr(config, "policy_variant", None)
    mode = getattr(policy_config, "proprio_context_mode", ProprioContextMode.NONE)
    return ProprioContextMode(mode) in {
        ProprioContextMode.TEXT_CONTEXT_TOKEN,
        ProprioContextMode.PER_CHUNK_ADDITIVE,
    }


def extract_proprio_context_tensor(
    observation: Mapping[str, Any],
    *,
    config: Any,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the exact parallel-stream proprio tensor when enabled."""

    if not proprio_context_enabled(config):
        return None
    state_encoding = getattr(
        getattr(config.data, "action_target", None), "state_encoding", None
    )
    if state_encoding != "eef_pos_axisangle_gripper_2d":
        raise ValueError(
            "LIBERO exact proprio context currently supports only "
            f"state_encoding='eef_pos_axisangle_gripper_2d', got {state_encoding!r}."
        )
    state = extract_eef_axisangle_gripper_state(observation)
    expected_dim = int(
        getattr(getattr(config.data, "action_schema", None), "state_dim", 0) or 0
    )
    if expected_dim > 0 and state.shape[0] != expected_dim:
        raise ValueError(
            "LIBERO proprio context state dim does not match data.action_schema.state_dim, "
            f"got {state.shape[0]} and expected {expected_dim}."
        )
    return torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)


def extract_eef_axisangle_gripper_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Convert raw LIBERO EEF state to the retained exact-rollout 8D layout."""

    eef_pos = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(
        observation["robot0_gripper_qpos"], dtype=np.float32
    ).reshape(-1)
    if eef_pos.shape[0] != 3:
        raise ValueError(
            f"Expected LIBERO robot0_eef_pos to have dim 3, got {eef_pos.shape[0]}."
        )
    if eef_quat.shape[0] != 4:
        raise ValueError(
            f"Expected LIBERO robot0_eef_quat to have dim 4, got {eef_quat.shape[0]}."
        )
    if gripper_qpos.shape[0] != 2:
        raise ValueError(
            f"Expected LIBERO robot0_gripper_qpos to have dim 2, got {gripper_qpos.shape[0]}."
        )
    axis_angle = (
        quaternion_to_axis_angle(
            torch.from_numpy(eef_quat).to(dtype=torch.float32).unsqueeze(0)
        )[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if axis_angle.shape[0] != 3:
        raise ValueError(
            f"Expected axis-angle proprio dim 3, got {axis_angle.shape[0]}."
        )
    return np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=0).astype(
        np.float32, copy=False
    )


def observations_to_views(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack canonical observation views into time-major tensors."""

    if not observations:
        raise ValueError(
            "Cannot build LIBERO views from an empty observation sequence."
        )
    return {
        key: torch.from_numpy(
            np.stack([observation[key] for observation in observations], axis=0)
        ).to(device=device)
        for key in LIBERO_OBS_KEYS
    }


def prepare_exact_runtime_inputs(
    runner: Any,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
    preserve_stream_cache: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Run canonicalization and the shared visual frontend for exact rollout."""

    canonical_batch = runner.pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend_output = runner.pipeline.visual_tower.run_frontend(
        canonical_video,
        placements=canonical_batch.placements,
        task_text=task_text,
        text_context=None
        if text_context is None
        else text_context.to(device=frontend_device),
        negative_text_context=(
            None
            if negative_text_context is None
            else negative_text_context.to(device=frontend_device)
        ),
        preserve_stream_cache=preserve_stream_cache,
    )
    return {
        "video_latents": frontend_output.video_latents.to(device=runtime_device),
        "text_context": (
            None
            if frontend_output.conditioning.text_context is None
            else frontend_output.conditioning.text_context.to(device=runtime_device)
        ),
        "negative_text_context": (
            None
            if frontend_output.conditioning.negative_text_context is None
            else frontend_output.conditioning.negative_text_context.to(
                device=runtime_device
            )
        ),
    }


def resolve_device(
    device_arg: str | None,
    *,
    fallback: torch.device | None = None,
) -> torch.device:
    """Resolve an explicit device, fallback, or the available default."""

    if device_arg is not None:
        return torch.device(device_arg)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
