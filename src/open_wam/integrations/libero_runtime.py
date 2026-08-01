"""Construction helpers for upstream LIBERO environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_wam.integrations.libero_tasks import (
    LiberoTaskSpec,
    ensure_local_libero_config,
)


__all__ = [
    "build_libero_control_env",
    "build_libero_offscreen_env",
]


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
) -> Any:
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
) -> Any:
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
