"""Process-level renderer selection for LIBERO simulator workloads."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from open_wam.configs.config_paths import SIMULATOR_CONFIG_ROOT
from open_wam.configs.enums import LiberoRendererBackend, LiberoRendererProfile


DEFAULT_LIBERO_RENDERER_CONFIG = (
    SIMULATOR_CONFIG_ROOT / "libero_renderer_profiles.yaml"
)
_RENDERER_MODULE_PREFIXES = (
    "OpenGL",
    "glfw",
    "mujoco",
    "mujoco_py",
    "robosuite",
)
_ACTIVE_RENDERER_BACKEND_ENV = "OPEN_WAM_LIBERO_RENDERER_BACKEND"


@dataclass(frozen=True)
class LiberoRendererConfig:
    """Resolved renderer settings for one process-wide LIBERO workload."""

    profile: LiberoRendererProfile
    backend: LiberoRendererBackend
    source_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "profile": self.profile.value,
            "backend": self.backend.value,
            "mujoco_gl": self.backend.value,
            "pyopengl_platform": self.backend.value,
            "source_path": str(self.source_path),
        }


def resolve_libero_renderer_config(
    profile: LiberoRendererProfile | str,
    *,
    config_path: str | Path = DEFAULT_LIBERO_RENDERER_CONFIG,
) -> LiberoRendererConfig:
    """Resolve the checked-in workload policy without importing LIBERO."""

    resolved_profile = LiberoRendererProfile(profile)
    resolved_path = Path(config_path).expanduser().resolve()
    backend_by_profile = _load_libero_renderer_profiles(resolved_path)
    return LiberoRendererConfig(
        profile=resolved_profile,
        backend=backend_by_profile[resolved_profile],
        source_path=resolved_path,
    )


@lru_cache(maxsize=8)
def _load_libero_renderer_profiles(
    resolved_path: Path,
) -> Mapping[LiberoRendererProfile, LiberoRendererBackend]:
    """Load and validate one immutable process-level renderer policy."""

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"LIBERO renderer config must be a mapping: {resolved_path}"
        )
    _reject_unknown_keys(
        raw,
        allowed={"schema_version", "profiles"},
        context=f"LIBERO renderer config {resolved_path}",
    )
    if raw.get("schema_version") != 1:
        raise ValueError(
            "LIBERO renderer config requires schema_version: 1: "
            f"{resolved_path}"
        )
    profiles = raw.get("profiles")
    if not isinstance(profiles, Mapping):
        raise TypeError(
            "LIBERO renderer config must contain a `profiles` mapping: "
            f"{resolved_path}"
        )
    _reject_unknown_keys(
        profiles,
        allowed={profile.value for profile in LiberoRendererProfile},
        context=f"LIBERO renderer profiles in {resolved_path}",
    )
    backend_by_profile: dict[LiberoRendererProfile, LiberoRendererBackend] = {}
    for profile in LiberoRendererProfile:
        profile_config = profiles.get(profile.value)
        if not isinstance(profile_config, Mapping):
            raise KeyError(
                f"Missing LIBERO renderer profile {profile.value!r} in "
                f"{resolved_path}"
            )
        _reject_unknown_keys(
            profile_config,
            allowed={"backend"},
            context=(
                f"LIBERO renderer profile {profile.value!r} in {resolved_path}"
            ),
        )
        try:
            backend_by_profile[profile] = LiberoRendererBackend(
                profile_config.get("backend")
            )
        except ValueError as exc:
            allowed = ", ".join(item.value for item in LiberoRendererBackend)
            raise ValueError(
                f"Invalid renderer backend for profile {profile.value!r} in "
                f"{resolved_path}; expected one of: {allowed}."
            ) from exc
    return backend_by_profile


def activate_libero_renderer(
    profile: LiberoRendererProfile | str,
    *,
    requested_backend: LiberoRendererBackend | str | None = None,
    requested_pyopengl_platform: str | None = None,
    config_path: str | Path = DEFAULT_LIBERO_RENDERER_CONFIG,
    environ: MutableMapping[str, str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
) -> LiberoRendererConfig:
    """Activate one renderer profile before upstream simulator imports.

    A profile is authoritative over inherited shell variables. A legacy CLI
    override may still be supplied, but only when it agrees with the profile.
    Switching a renderer after the OpenGL stack has loaded is rejected because
    changing environment variables at that point does not change the context.
    """

    config = resolve_libero_renderer_config(profile, config_path=config_path)
    if requested_backend is not None:
        requested = LiberoRendererBackend(requested_backend)
        if requested is not config.backend:
            raise ValueError(
                f"LIBERO renderer profile {config.profile.value!r} requires "
                f"{config.backend.value!r}, not {requested.value!r}."
            )
    if (
        requested_pyopengl_platform is not None
        and requested_pyopengl_platform != config.backend.value
    ):
        raise ValueError(
            f"LIBERO renderer profile {config.profile.value!r} requires "
            f"PYOPENGL_PLATFORM={config.backend.value!r}, not "
            f"{requested_pyopengl_platform!r}."
        )

    target_env = os.environ if environ is None else environ
    modules = sys.modules if loaded_modules is None else loaded_modules
    initialized = _initialized_renderer_modules(modules)
    if initialized:
        active_backend = target_env.get(_ACTIVE_RENDERER_BACKEND_ENV)
        conflicts = {
            key: target_env.get(key)
            for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
            if target_env.get(key) != config.backend.value
        }
        if active_backend != config.backend.value or conflicts:
            raise RuntimeError(
                "Cannot change the LIBERO renderer after OpenGL/MuJoCo "
                f"initialization. Loaded modules={initialized}, active_backend="
                f"{active_backend!r}, environment={conflicts}, requested="
                f"{config.to_dict()}. Start a fresh process."
            )

    target_env["MUJOCO_GL"] = config.backend.value
    target_env["PYOPENGL_PLATFORM"] = config.backend.value
    target_env[_ACTIVE_RENDERER_BACKEND_ENV] = config.backend.value
    target_env["OPEN_WAM_LIBERO_RENDERER_PROFILE"] = config.profile.value
    if config.backend is not LiberoRendererBackend.EGL:
        target_env.pop("MUJOCO_EGL_DEVICE_ID", None)
        target_env.pop("EGL_DEVICE_ID", None)
    return config


def _initialized_renderer_modules(modules: Mapping[str, Any]) -> tuple[str, ...]:
    initialized = {
        prefix
        for module_name in modules
        for prefix in _RENDERER_MODULE_PREFIXES
        if module_name == prefix or module_name.startswith(f"{prefix}.")
    }
    return tuple(sorted(initialized))


def _reject_unknown_keys(
    value: Mapping[Any, Any],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {unknown}.")


__all__ = [
    "DEFAULT_LIBERO_RENDERER_CONFIG",
    "LiberoRendererConfig",
    "activate_libero_renderer",
    "resolve_libero_renderer_config",
]
