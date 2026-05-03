from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

from open_wam.data import reconstruct_absolute_pose_targets
from open_wam.data.action_transforms import (
    PoseSequence,
    collapse_gripper_state,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
)


@dataclass(frozen=True)
class LiberoTaskSpec:
    """Resolved LIBERO benchmark task used to construct a simulator scene."""

    benchmark_name: str
    task_id: int
    task_name: str
    task_language: str
    problem_folder: str
    bddl_file_path: str
    init_states_path: str


@dataclass(frozen=True)
class LiberoControlConfig:
    """Closed-loop tracking gains for converting our public targets to OSC actions.

    `OSC_POSE` expects a 7D action `[dx, dy, dz, dax, day, daz, gripper]`.
    The first six channels are normalized and internally scaled by robosuite to
    +/- 0.05 m and +/- 0.5 rad respectively. The public WAM target, however, is
    a reference-relative absolute EEF target `[rel_xyz, rel_axis_angle, gripper]`
    expressed against a dataset-defined anchor pose.
    We therefore:
    1. reconstruct the desired absolute EEF target from the stored reference pose
    2. compute current world-frame pose error
    3. normalize that error into the controller's expected action range
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
class LiberoTrackingResult:
    """Trajectory rollout and tracking metrics from a LIBERO env replay."""

    task_spec: LiberoTaskSpec
    init_state_index: int
    desired_pose: PoseSequence
    tracked_pose: PoseSequence
    position_error_per_target: torch.Tensor
    rotation_error_deg_per_target: torch.Tensor
    gripper_error_per_target: torch.Tensor
    camera_frames: dict[str, list[np.ndarray]]
    rendered_target_indices: list[int]


def ensure_local_libero_config(project_root: Path | None = None) -> Path:
    """Bootstrap LIBERO's config file without interactive prompts.

    The original LIBERO package prompts on import if `~/.libero/config.yaml`
    does not exist. Collaborative tooling should not depend on interactive setup,
    so Open-WAM writes a local config into `.cache/libero_config/` and points
    `LIBERO_CONFIG_PATH` there before importing the upstream package.
    """

    root = _project_root(project_root)
    libero_repo_root, libero_package_root = _resolve_libero_paths()
    config_dir = root / ".cache" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": str(libero_package_root.resolve()),
        "bddl_files": str((libero_package_root / "bddl_files").resolve()),
        "init_states": str((libero_package_root / "init_files").resolve()),
        "datasets": str((libero_repo_root / "libero" / "datasets").resolve()),
        "assets": str((libero_package_root / "assets").resolve()),
    }
    config_path = config_dir / "config.yaml"
    config_text = yaml.safe_dump(config, sort_keys=False)
    if not config_path.is_file() or config_path.read_text(encoding="utf-8") != config_text:
        tmp_path = config_path.with_name(f"{config_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(config_text, encoding="utf-8")
        tmp_path.replace(config_path)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_path


def resolve_libero_task(
    task_text: str,
    project_root: Path | None = None,
    *,
    benchmark_name: str | None = None,
) -> LiberoTaskSpec:
    """Resolve dataset task text to one upstream LIBERO benchmark task."""

    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    normalized_task_text = _normalize_task_text(task_text)
    matches: list[LiberoTaskSpec] = []
    benchmark_classes = benchmark.get_benchmark_dict()
    if benchmark_name is not None:
        try:
            benchmark_items = ((benchmark_name, benchmark_classes[benchmark_name]),)
        except KeyError as exc:
            available = ", ".join(sorted(benchmark_classes))
            raise ValueError(
                f"Unknown LIBERO benchmark {benchmark_name!r}; available benchmarks: {available}"
            ) from exc
    else:
        benchmark_items = tuple(benchmark_classes.items())

    for current_benchmark_name, benchmark_class in benchmark_items:
        try:
            benchmark_instance = benchmark_class()
        except Exception:
            # Upstream registers suites such as LIBERO_100 that are not fully
            # initialized in this checkout. Task resolution should ignore those
            # and keep searching the benchmark variants that are usable.
            continue
        for task_id in range(benchmark_instance.get_num_tasks()):
            task = benchmark_instance.get_task(task_id)
            if _normalize_task_text(task.language) != normalized_task_text:
                continue
            matches.append(
                LiberoTaskSpec(
                    benchmark_name=current_benchmark_name,
                    task_id=task_id,
                    task_name=task.name,
                    task_language=task.language,
                    problem_folder=task.problem_folder,
                    bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
                    init_states_path=os.path.join(
                        os.environ["LIBERO_CONFIG_PATH"],
                        "..",
                    ),  # overwritten below for clarity
                )
            )

    if not matches:
        raise ValueError(f"Could not resolve LIBERO task text: {task_text!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Task text {task_text!r} matched multiple LIBERO tasks; expected exactly one. "
            f"Matches: {[match.task_name for match in matches]}"
        )

    match = matches[0]
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return LiberoTaskSpec(
        benchmark_name=match.benchmark_name,
        task_id=match.task_id,
        task_name=match.task_name,
        task_language=match.task_language,
        problem_folder=match.problem_folder,
        bddl_file_path=match.bddl_file_path,
        init_states_path=str(Path(config["init_states"]) / match.problem_folder / f"{match.task_name}.pruned_init"),
    )


def load_libero_task_init_states(task_spec: LiberoTaskSpec, project_root: Path | None = None) -> Any:
    """Load benchmark init states with torch 2.6-compatible semantics."""

    ensure_local_libero_config(project_root)
    # Upstream uses `torch.load(path)` which defaults to `weights_only=True`
    # on torch 2.6+. The init-state files are not weight checkpoints.
    return torch.load(task_spec.init_states_path, weights_only=False)


def build_libero_offscreen_env(
    task_spec: LiberoTaskSpec,
    *,
    camera_height: int = 256,
    camera_width: int = 256,
    horizon: int = 5000,
    ignore_done: bool = True,
    project_root: Path | None = None,
):
    """Construct one offscreen LIBERO environment for evaluation."""

    ensure_local_libero_config(project_root)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    return OffScreenRenderEnv(
        bddl_file_name=task_spec.bddl_file_path,
        camera_heights=camera_height,
        camera_widths=camera_width,
        horizon=horizon,
        ignore_done=ignore_done,
    )


def infer_task_local_episode_rank(
    episode_records: list[Any] | tuple[Any, ...],
    *,
    episode_index: int,
    task_text: str,
) -> int:
    """Best-effort mapping from dataset episode to per-task demo index.

    The HF export does not expose the original demo id. The stable fallback is
    the count of prior episodes with the same task text. This is sufficient for
    reproducible env rollouts and often aligns with the original demo ordering.
    """

    normalized = _normalize_task_text(task_text)
    rank = 0
    for record in episode_records:
        record_task = record.tasks[0] if getattr(record, "tasks", None) else ""
        if int(record.episode_index) == episode_index:
            return rank
        if _normalize_task_text(record_task) == normalized:
            rank += 1
    raise ValueError(f"Episode index {episode_index} was not found in episode metadata.")


def extract_pose_from_obs(obs: dict[str, Any]) -> PoseSequence:
    """Parse LIBERO / robosuite observation dict into the common pose contract."""

    quaternion_xyzw = torch.tensor(obs["robot0_eef_quat"], dtype=torch.float32)
    return PoseSequence(
        position=torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32),
        quaternion=normalize_quaternion(quaternion_xyzw),
        gripper=torch.tensor(obs["robot0_gripper_qpos"], dtype=torch.float32),
    )


def compute_osc_pose_action(
    *,
    current_pose: PoseSequence,
    desired_pose: PoseSequence,
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    """Convert one desired absolute pose into one normalized `OSC_POSE` action."""

    position_error = desired_pose.position - current_pose.position

    delta_quaternion = quaternion_multiply(
        desired_pose.quaternion.unsqueeze(0),
        quaternion_inverse(current_pose.quaternion).unsqueeze(0),
    )[0]
    delta_axis_angle = quaternion_to_axis_angle(normalize_quaternion(delta_quaternion.unsqueeze(0)))[0]

    position_command = torch.clamp(position_error / control_config.max_pos_delta_m, min=-1.0, max=1.0)
    rotation_command = torch.clamp(delta_axis_angle / control_config.max_rot_delta_rad, min=-1.0, max=1.0)

    if desired_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    elif gripper_representation == "action_command":
        # When the public target carries the raw LIBERO gripper command, replay
        # should pass that command through directly instead of re-interpreting
        # it as a finger-joint state target.
        gripper_command = desired_pose.gripper[0:1].clamp(min=-1.0, max=1.0).to(dtype=torch.float32)
    elif current_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    else:
        current_public = _project_gripper_state(
            current_pose.gripper,
            gripper_representation=gripper_representation,
        )

        if gripper_representation == "all_channels":
            # LIBERO exposes two finger joints in state. When the public target
            # keeps both channels, interpret them via the jaw opening.
            current_value = current_pose.gripper[0] - current_pose.gripper[1]
            desired_value = desired_pose.gripper[0] - desired_pose.gripper[1]
            open_threshold = control_config.gripper_open_threshold
            close_threshold = control_config.gripper_close_threshold
            tolerance = control_config.gripper_position_tolerance
        elif gripper_representation == "first_channel":
            # The default public representation keeps only the first finger
            # qpos. For Panda this is roughly half of the jaw opening.
            current_value = current_public[0]
            desired_value = desired_pose.gripper[0]
            open_threshold = control_config.gripper_open_threshold * 0.5
            close_threshold = control_config.gripper_close_threshold * 0.5
            tolerance = control_config.gripper_position_tolerance * 0.5
        else:
            raise ValueError(f"Unsupported gripper representation: {gripper_representation}")

        error_value = desired_value - current_value
        if desired_value >= open_threshold:
            gripper_command = torch.tensor([-1.0], dtype=torch.float32)
        elif desired_value <= close_threshold:
            gripper_command = torch.tensor([1.0], dtype=torch.float32)
        elif torch.abs(error_value) <= tolerance:
            gripper_command = torch.tensor([0.0], dtype=torch.float32)
        else:
            gripper_command = torch.clamp(
                -error_value / control_config.max_gripper_delta,
                min=-1.0,
                max=1.0,
            ).reshape(1)

    action = torch.cat([position_command, rotation_command, gripper_command], dim=0)
    return action.detach().cpu().numpy().astype(np.float32)


def track_relative_targets_in_libero_env(
    *,
    task_text: str,
    relative_pose_targets: torch.Tensor,
    rotation_representation: str,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    gripper_representation: str = "first_channel",
    init_state_index: int = 0,
    control_config: LiberoControlConfig | None = None,
    camera_obs_keys: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image"),
    camera_height: int = 256,
    camera_width: int = 256,
    project_root: Path | None = None,
) -> LiberoTrackingResult:
    """Replay one public WAM trajectory in the real LIBERO simulator.

    The public representation is reference-relative. Replay must therefore use
    the same reference pose that was used to build the public targets in the
    dataset adapter. For episode-mode LIBERO targets that is the first dataset
    frame; for sample-mode targets it is the sample's anchor state.
    """

    if control_config is None:
        control_config = LiberoControlConfig()

    task_spec = resolve_libero_task(task_text, project_root=project_root)
    init_states = load_libero_task_init_states(task_spec, project_root=project_root)
    init_state_index = int(np.clip(init_state_index, 0, len(init_states) - 1))

    env = build_libero_offscreen_env(
        task_spec,
        camera_height=camera_height,
        camera_width=camera_width,
        horizon=max(5000, int(relative_pose_targets.shape[0] * control_config.control_substeps_per_target + 32)),
        ignore_done=True,
        project_root=project_root,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(init_states[init_state_index])
        desired_pose = reconstruct_absolute_pose_targets(
            reference_position=reference_position,
            reference_quaternion=reference_quaternion,
            relative_pose_targets=relative_pose_targets,
            rotation_representation=rotation_representation,
        )
        aligned_gripper_targets = _align_replay_gripper_targets(
            desired_pose.gripper,
            gripper_representation=gripper_representation,
            delay_steps=control_config.action_command_delay_steps,
        )

        tracked_positions: list[torch.Tensor] = []
        tracked_quaternions: list[torch.Tensor] = []
        tracked_gripper: list[torch.Tensor] = []
        rendered_target_indices: list[int] = []
        camera_frames: dict[str, list[np.ndarray]] = {camera_key: [] for camera_key in camera_obs_keys}

        for target_index in range(relative_pose_targets.shape[0]):
            target_pose = PoseSequence(
                position=desired_pose.position[target_index],
                quaternion=desired_pose.quaternion[target_index],
                gripper=None if aligned_gripper_targets is None else aligned_gripper_targets[target_index],
            )
            for _ in range(control_config.control_substeps_per_target):
                current_pose = extract_pose_from_obs(obs)
                action = compute_osc_pose_action(
                    current_pose=current_pose,
                    desired_pose=target_pose,
                    control_config=control_config,
                    gripper_representation=gripper_representation,
                )
                obs, _, _, _ = env.step(action)
                rendered_target_indices.append(target_index)
                for camera_key in camera_obs_keys:
                    camera_frames[camera_key].append(np.array(obs[camera_key], copy=True))

            final_pose = extract_pose_from_obs(obs)
            tracked_positions.append(final_pose.position)
            tracked_quaternions.append(final_pose.quaternion)
            if final_pose.gripper is not None:
                if gripper_representation == "action_command":
                    tracked_gripper.append(torch.tensor([float(action[-1])], dtype=torch.float32))
                    continue
                tracked_gripper.append(
                    _project_gripper_state(
                        final_pose.gripper,
                        gripper_representation=gripper_representation,
                    )
                )

        tracked_pose = PoseSequence(
            position=torch.stack(tracked_positions, dim=0),
            quaternion=torch.stack(tracked_quaternions, dim=0),
            gripper=torch.stack(tracked_gripper, dim=0) if tracked_gripper else None,
        )

        position_error_per_target = torch.linalg.vector_norm(
            tracked_pose.position - desired_pose.position,
            dim=-1,
        )
        rotation_error_deg_per_target = quaternion_angular_error_degrees(
            tracked_pose.quaternion,
            desired_pose.quaternion,
        )
        if desired_pose.gripper is not None and tracked_pose.gripper is not None:
            gripper_error_per_target = torch.linalg.vector_norm(
                tracked_pose.gripper - aligned_gripper_targets,
                dim=-1,
            )
        else:
            gripper_error_per_target = torch.zeros_like(position_error_per_target)

        return LiberoTrackingResult(
            task_spec=task_spec,
            init_state_index=init_state_index,
            desired_pose=desired_pose,
            tracked_pose=tracked_pose,
            position_error_per_target=position_error_per_target,
            rotation_error_deg_per_target=rotation_error_deg_per_target,
            gripper_error_per_target=gripper_error_per_target,
            camera_frames=camera_frames,
            rendered_target_indices=rendered_target_indices,
        )
    finally:
        env.close()


def quaternion_angular_error_degrees(lhs_xyzw: torch.Tensor, rhs_xyzw: torch.Tensor) -> torch.Tensor:
    lhs = normalize_quaternion(lhs_xyzw)
    rhs = normalize_quaternion(rhs_xyzw)
    dot = (lhs * rhs).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))


def _normalize_task_text(task_text: str) -> str:
    return " ".join(task_text.strip().lower().split())


def _project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_libero_paths() -> tuple[Path, Path]:
    """Resolve the installed LIBERO repo root and package root from Python imports.

    Upstream LIBERO uses an unusual nested package layout:
    `<repo>/libero/libero/__init__.py`.
    Some local installs therefore record distribution metadata without exposing
    an importable `libero` package. When that happens, fall back to a checkout
    path so the current uv environment can still import `libero.libero`.
    """

    env_repo_root = os.environ.get("LIBERO_REPO_ROOT")
    if env_repo_root:
        env_paths = _libero_paths_from_repo_root(Path(env_repo_root).expanduser())
        if env_paths is not None:
            return env_paths

    import_error: Exception | None = None
    try:
        libero_pkg = importlib.import_module("libero.libero")
    except EOFError as exc:
        # Upstream LIBERO can prompt on import when its config file has not
        # been bootstrapped yet, which raises EOFError in non-interactive
        # contexts. Fall back to a checkout path without importing so
        # `ensure_local_libero_config(...)` can write the config first.
        import_error = exc
        libero_pkg = None
    except ModuleNotFoundError as exc:
        if exc.name not in {"libero", "libero.libero"}:
            raise
        import_error = exc
        libero_pkg = None

    if libero_pkg is not None:
        package_root = Path(libero_pkg.__file__).resolve().parent
        repo_root = package_root.parents[1]
        return repo_root, package_root

    fallback_repo_roots: list[Path] = []

    project_root = _project_root(None)
    fallback_repo_roots.append(project_root.parent / "LIBERO")

    for repo_root in fallback_repo_roots:
        paths = _libero_paths_from_repo_root(repo_root)
        if paths is not None:
            return paths

    raise ImportError(
        "LIBERO could not be imported. Either install an importable LIBERO package into the uv environment "
        "or set LIBERO_REPO_ROOT to a checkout whose structure contains `libero/libero/__init__.py`."
    ) from import_error


def _libero_paths_from_repo_root(repo_root: Path) -> tuple[Path, Path] | None:
    package_root = repo_root / "libero" / "libero"
    if not (package_root / "__init__.py").exists():
        return None
    repo_root_resolved = repo_root.resolve()
    repo_root_str = str(repo_root_resolved)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root_resolved, package_root.resolve()


def _project_gripper_state(gripper_state: torch.Tensor, *, gripper_representation: str) -> torch.Tensor:
    """Expose one env gripper state in the same public representation as targets."""

    if gripper_state.ndim != 1:
        raise ValueError(f"Expected one gripper state vector, got shape {tuple(gripper_state.shape)}.")
    if gripper_representation == "action_command":
        raise ValueError(
            "action_command is a control-domain target and cannot be recovered from env gripper state alone."
        )
    return collapse_gripper_state(
        gripper_state.unsqueeze(0),
        gripper_representation=gripper_representation,
    )[0]


def _align_replay_gripper_targets(
    gripper_targets: torch.Tensor | None,
    *,
    gripper_representation: str,
    delay_steps: int,
) -> torch.Tensor | None:
    """Shift command-domain gripper targets to the state they actually produce.

    LIBERO's 1D action gripper command is causal: `action[t]` drives the
    transition from state `t` toward state `t+1`. For replay we compare against
    pose targets at state-aligned timesteps, so the command must be delayed by
    one target to avoid visibly closing / opening too early.
    """

    if gripper_targets is None:
        return None
    if gripper_representation != "action_command":
        return gripper_targets
    if delay_steps < 0:
        raise ValueError(f"Expected non-negative action_command_delay_steps, got {delay_steps}.")

    aligned = torch.zeros_like(gripper_targets)
    if delay_steps == 0:
        aligned.copy_(gripper_targets)
        return aligned
    if delay_steps >= gripper_targets.shape[0]:
        return aligned

    aligned[delay_steps:] = gripper_targets[:-delay_steps]
    return aligned
