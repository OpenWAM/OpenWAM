from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from open_wam.configs import (
    LiberoRendererBackend,
    LiberoRendererProfile,
)
from open_wam.evals.libero_dual_expert_rollout import (
    DualExpertLiberoEpisodeOptions,
    run_dual_expert_libero_episode,
)
from open_wam.evals.libero_dual_expert_runtime import (
    DualExpertActionRoute,
)
from open_wam.integrations.libero_rendering import (
    DEFAULT_LIBERO_RENDERER_CONFIG,
    activate_libero_renderer,
    resolve_libero_renderer_config,
)


@pytest.mark.parametrize(
    ("profile", "backend"),
    (
        (LiberoRendererProfile.ONLINE_ROLLOUT, LiberoRendererBackend.EGL),
        (
            LiberoRendererProfile.OFFLINE_ANALYSIS,
            LiberoRendererBackend.OSMESA,
        ),
        (
            LiberoRendererProfile.DATASET_GENERATION,
            LiberoRendererBackend.OSMESA,
        ),
    ),
)
def test_renderer_profiles_resolve_expected_backend(
    profile: LiberoRendererProfile,
    backend: LiberoRendererBackend,
) -> None:
    config = resolve_libero_renderer_config(profile)

    assert config.profile is profile
    assert config.backend is backend
    assert config.source_path == DEFAULT_LIBERO_RENDERER_CONFIG.resolve()


def test_renderer_config_rejects_unknown_schema(tmp_path) -> None:
    config_path = tmp_path / "renderer.yaml"
    config_path.write_text(
        "schema_version: 2\nprofiles: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version: 1"):
        resolve_libero_renderer_config(
            LiberoRendererProfile.ONLINE_ROLLOUT,
            config_path=config_path,
        )


@pytest.mark.parametrize(
    "document",
    (
        """schema_version: 1
profiles:
  online_rollout: {backend: egl}
  offline_analysis: {backend: osmesa}
  dataset_generation: {backend: osmesa}
extra: true
""",
        """schema_version: 1
profiles:
  online_rollout: {backend: egl}
  offline_analysis: {backend: osmesa}
  dataset_generation: {backend: osmesa}
  typo: {backend: egl}
""",
        """schema_version: 1
profiles:
  online_rollout: {backend: egl, typo: true}
  offline_analysis: {backend: osmesa}
  dataset_generation: {backend: osmesa}
""",
    ),
)
def test_renderer_config_rejects_unknown_keys(tmp_path, document: str) -> None:
    config_path = tmp_path / "renderer.yaml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys"):
        resolve_libero_renderer_config(
            LiberoRendererProfile.ONLINE_ROLLOUT,
            config_path=config_path,
        )


def test_activation_overrides_inherited_shell_renderer() -> None:
    environment = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "2",
        "EGL_DEVICE_ID": "2",
    }

    config = activate_libero_renderer(
        LiberoRendererProfile.OFFLINE_ANALYSIS,
        environ=environment,
        loaded_modules={},
    )

    assert config.backend is LiberoRendererBackend.OSMESA
    assert environment["MUJOCO_GL"] == "osmesa"
    assert environment["PYOPENGL_PLATFORM"] == "osmesa"
    assert environment["OPEN_WAM_LIBERO_RENDERER_PROFILE"] == (
        "offline_analysis"
    )
    assert environment["OPEN_WAM_LIBERO_RENDERER_BACKEND"] == "osmesa"
    assert "MUJOCO_EGL_DEVICE_ID" not in environment
    assert "EGL_DEVICE_ID" not in environment


def test_online_activation_overrides_inherited_osmesa() -> None:
    environment = {
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    }

    config = activate_libero_renderer(
        LiberoRendererProfile.ONLINE_ROLLOUT,
        environ=environment,
        loaded_modules={},
    )

    assert config.backend is LiberoRendererBackend.EGL
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"


def test_activation_rejects_profile_override_mismatch() -> None:
    with pytest.raises(ValueError, match="requires 'osmesa'"):
        activate_libero_renderer(
            LiberoRendererProfile.OFFLINE_ANALYSIS,
            requested_backend="egl",
            environ={},
            loaded_modules={},
        )


def test_activation_rejects_conflicting_late_switch() -> None:
    with pytest.raises(RuntimeError, match="Cannot change the LIBERO renderer"):
        activate_libero_renderer(
            LiberoRendererProfile.ONLINE_ROLLOUT,
            environ={
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
            },
            loaded_modules={"mujoco": object()},
        )


def test_activation_rejects_switch_after_environment_is_mutated() -> None:
    environment: dict[str, str] = {}
    activate_libero_renderer(
        LiberoRendererProfile.ONLINE_ROLLOUT,
        environ=environment,
        loaded_modules={},
    )
    environment["MUJOCO_GL"] = "osmesa"
    environment["PYOPENGL_PLATFORM"] = "osmesa"

    with pytest.raises(RuntimeError, match="active_backend='egl'"):
        activate_libero_renderer(
            LiberoRendererProfile.OFFLINE_ANALYSIS,
            environ=environment,
            loaded_modules={"mujoco": object()},
        )


def test_activation_rejects_loaded_stack_without_activation_marker() -> None:
    with pytest.raises(RuntimeError, match="active_backend=None"):
        activate_libero_renderer(
            LiberoRendererProfile.OFFLINE_ANALYSIS,
            environ={
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
            },
            loaded_modules={"mujoco": object()},
        )


def test_activation_is_idempotent_after_renderer_stack_loads() -> None:
    environment: dict[str, str] = {}
    activate_libero_renderer(
        LiberoRendererProfile.ONLINE_ROLLOUT,
        environ=environment,
        loaded_modules={},
    )

    config = activate_libero_renderer(
        LiberoRendererProfile.ONLINE_ROLLOUT,
        environ=environment,
        loaded_modules={"mujoco": object()},
    )

    assert config.backend is LiberoRendererBackend.EGL


def _episode_options(
    route: DualExpertActionRoute = DualExpertActionRoute.JOINT,
) -> DualExpertLiberoEpisodeOptions:
    return DualExpertLiberoEpisodeOptions(
        benchmark="libero_10",
        task_id=0,
        episode_idx=0,
        max_timestep=1,
        max_chunks=1,
        execute_action_steps=None,
        execute_frame_chunk_size=None,
        dual_expert_rollout_frame_chunk_size=None,
        dual_expert_inference_window_size=None,
        dual_expert_action_only_rollout=False,
        dual_expert_gjd_action_route=route.value,
        reset_policy_state_each_chunk=False,
        max_imagined_latent_frames=0,
        output_dir="outputs",
        suffix="test",
        video_fps=15.0,
        seed=0,
    )


@pytest.mark.parametrize("route", tuple(DualExpertActionRoute))
def test_dual_expert_online_routes_share_renderer_profile(
    route: DualExpertActionRoute,
) -> None:
    options = _episode_options(route)

    assert options.renderer_profile is LiberoRendererProfile.ONLINE_ROLLOUT


def test_dual_expert_episode_rejects_active_renderer_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.setenv("OPEN_WAM_LIBERO_RENDERER_BACKEND", "osmesa")
    monkeypatch.setitem(sys.modules, "mujoco", object())

    with pytest.raises(RuntimeError, match="active_backend='osmesa'"):
        run_dual_expert_libero_episode(
            _episode_options(),
            SimpleNamespace(),
            SimpleNamespace(),
            object(),
            include_episode_coordinates=False,
            close_env_after_rollout=False,
        )
