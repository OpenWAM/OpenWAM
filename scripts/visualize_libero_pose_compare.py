from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (  # noqa: E402
    LeRobotV2WindowDataset,
    build_relative_pose_targets,
    build_lerobot_train_val_episode_split,
    reconstruct_absolute_pose_targets,
    state_sequence_to_pose_sequence,
)
from open_wam.configs import load_experiment_config  # noqa: E402
from open_wam.integrations.libero_osc_control import (  # noqa: E402
    quaternion_angular_error_degrees,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one LIBERO rollout as original absolute state, reconstructed "
            "from the public action representation, or both at once."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/contract_only_libero.yaml",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--trajectory",
        choices=("sample", "episode"),
        default="sample",
        help="Visualize either one sampled horizon or the full episode trajectory.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode to visualize when `--trajectory episode`. Defaults to the selected sample's episode.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.7)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print comparison stats without launching MuJoCo.")
    parser.add_argument(
        "--mode",
        choices=("original", "reconstructed", "compare"),
        default="compare",
        help="What rollout to animate in the MuJoCo scene.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional animation output path (.gif or .mp4). When set, render offscreen instead of launching a GLFW viewer.",
    )
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=None,
        help="Frame duration in seconds. Defaults to --sleep-seconds for animated steps.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.data.action_target.representation != "eef_pose_relative_to_reference":
        raise ValueError(
            "This visualizer expects `data.action_target.representation: eef_pose_relative_to_reference`."
        )

    train_episodes, _ = build_lerobot_train_val_episode_split(config.data)
    dataset = LeRobotV2WindowDataset(config.data, episodes=train_episodes)
    sample = dataset[args.sample_index]

    if args.trajectory == "sample":
        task_text = sample.task_text
        episode_index = int(sample.metadata["episode_index"])
        action_indices = sample.metadata["action_frame_indices"]
        target_indices = sample.metadata["target_state_frame_indices"]
        public_actions = sample.actions
        reference_position = torch.tensor(sample.metadata["reference_position"], dtype=torch.float32)
        reference_quaternion_xyzw = torch.tensor(sample.metadata["reference_quaternion_xyzw"], dtype=torch.float32)
        rotation_representation = str(sample.metadata["rotation_representation"])
        original_pose = _load_original_target_pose(dataset=dataset, sample=sample, config=config)
    else:
        episode_index = args.episode_index if args.episode_index is not None else int(sample.metadata["episode_index"])
        (
            task_text,
            action_indices,
            target_indices,
            public_actions,
            reference_position,
            reference_quaternion_xyzw,
            rotation_representation,
            original_pose,
        ) = _load_full_episode_rollout(
            dataset=dataset,
            episode_index=episode_index,
            config=config,
        )

    reconstructed_pose = reconstruct_absolute_pose_targets(
        reference_position=reference_position,
        reference_quaternion=reference_quaternion_xyzw,
        relative_pose_targets=public_actions,
        rotation_representation=rotation_representation,
    )

    position_errors = torch.linalg.vector_norm(original_pose.position - reconstructed_pose.position, dim=-1)
    angular_errors = quaternion_angular_error_degrees(
        original_pose.quaternion,
        reconstructed_pose.quaternion,
    )

    print("task_text:", task_text)
    print("mode:", args.mode)
    print("trajectory:", args.trajectory)
    print("sample_index:", args.sample_index)
    print("episode_index:", episode_index)
    print("num_steps:", len(original_pose.position))
    print("action_frame_indices:", action_indices[:10], "..." if len(action_indices) > 10 else "")
    print("target_state_frame_indices:", target_indices[:10], "..." if len(target_indices) > 10 else "")
    print("action_representation:", config.data.action_target.representation)
    print("rotation_representation:", rotation_representation)
    print("first_public_action:", public_actions[0].tolist())
    print("mean_position_error_m:", float(position_errors.mean()))
    print("max_position_error_m:", float(position_errors.max()))
    print("mean_rotation_error_deg:", float(angular_errors.mean()))
    print("max_rotation_error_deg:", float(angular_errors.max()))

    if args.dry_run:
        return

    scene_xml = _build_scene_xml(
        reference_position=reference_position,
        reference_quaternion_xyzw=reference_quaternion_xyzw,
        original_positions=original_pose.position,
        original_quaternions_xyzw=original_pose.quaternion,
        reconstructed_positions=reconstructed_pose.position,
        reconstructed_quaternions_xyzw=reconstructed_pose.quaternion,
        mode=args.mode,
    )
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)

    mocap_handles = _resolve_mocap_handles(model=model, mode=args.mode)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_duration = args.frame_duration if args.frame_duration is not None else args.sleep_seconds
        _render_animation_offscreen(
            model=model,
            data=data,
            mocap_handles=mocap_handles,
            reference_position=reference_position,
            reference_quaternion_xyzw=reference_quaternion_xyzw,
            original_positions=original_pose.position,
            original_quaternions_xyzw=original_pose.quaternion,
            reconstructed_positions=reconstructed_pose.position,
            reconstructed_quaternions_xyzw=reconstructed_pose.quaternion,
            mode=args.mode,
            output_path=output_path,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            frame_duration_seconds=frame_duration,
        )
        print("saved_animation:", str(output_path.resolve()))
        return

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No DISPLAY was found. On a headless/slurm node, pass --output outputs/libero_reference_pose_compare.mp4 "
            "to render offscreen instead of launching the GLFW viewer."
        )

    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _move_to_reference(
                data=data,
                mocap_handles=mocap_handles,
                reference_position=reference_position,
                reference_quaternion_xyzw=reference_quaternion_xyzw,
            )
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(args.sleep_seconds * 0.5, 0.1))

            for step_index in range(len(original_pose.position)):
                if not viewer.is_running():
                    break
                if "animated_original" in mocap_handles:
                    _set_mocap_pose(
                        data=data,
                        mocap_index=mocap_handles["animated_original"],
                        position=original_pose.position[step_index],
                        quaternion_xyzw=original_pose.quaternion[step_index],
                    )
                if "animated_reconstructed" in mocap_handles:
                    _set_mocap_pose(
                        data=data,
                        mocap_index=mocap_handles["animated_reconstructed"],
                        position=reconstructed_pose.position[step_index],
                        quaternion_xyzw=reconstructed_pose.quaternion[step_index],
                    )
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(args.sleep_seconds)

            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break


def _render_animation_offscreen(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_handles: dict[str, int],
    reference_position: torch.Tensor,
    reference_quaternion_xyzw: torch.Tensor,
    original_positions: torch.Tensor,
    original_quaternions_xyzw: torch.Tensor,
    reconstructed_positions: torch.Tensor,
    reconstructed_quaternions_xyzw: torch.Tensor,
    mode: str,
    output_path: Path,
    camera_height: int,
    camera_width: int,
    frame_duration_seconds: float,
) -> None:
    frames: list[np.ndarray] = []
    camera = _build_offscreen_camera(
        reference_position=reference_position,
        original_positions=original_positions,
        reconstructed_positions=reconstructed_positions,
    )

    try:
        with mujoco.Renderer(model, height=camera_height, width=camera_width) as renderer:
            _move_to_reference(
                data=data,
                mocap_handles=mocap_handles,
                reference_position=reference_position,
                reference_quaternion_xyzw=reference_quaternion_xyzw,
            )
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

            step_count = len(original_positions)
            for step_index in range(step_count):
                if "animated_original" in mocap_handles and mode in {"original", "compare"}:
                    _set_mocap_pose(
                        data=data,
                        mocap_index=mocap_handles["animated_original"],
                        position=original_positions[step_index],
                        quaternion_xyzw=original_quaternions_xyzw[step_index],
                    )
                if "animated_reconstructed" in mocap_handles and mode in {"reconstructed", "compare"}:
                    _set_mocap_pose(
                        data=data,
                        mocap_index=mocap_handles["animated_reconstructed"],
                        position=reconstructed_positions[step_index],
                        quaternion_xyzw=reconstructed_quaternions_xyzw[step_index],
                    )
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
    except mujoco.FatalError as exc:
        raise RuntimeError(
            "Offscreen rendering failed because MuJoCo could not create a headless OpenGL context. "
            "On slurm/GPU nodes, try `MUJOCO_GL=egl uv run python ... --output ...`. "
            "If EGL is unavailable, try `MUJOCO_GL=osmesa uv run python ... --output ...`."
        ) from exc

    _write_animation(output_path=output_path, frames=frames, frame_duration_seconds=frame_duration_seconds)


def _load_original_target_pose(
    *,
    dataset: LeRobotV2WindowDataset,
    sample,
    config,
):
    episode_index = int(sample.metadata["episode_index"])
    target_frame_indices = list(sample.metadata["target_state_frame_indices"])
    episode_rows = dataset._load_episode_rows(episode_index)  # noqa: SLF001
    target_rows = [episode_rows[frame_index] for frame_index in target_frame_indices]

    if [int(row["frame_index"]) for row in target_rows] != target_frame_indices:
        raise RuntimeError("Target frame indices no longer align to the underlying episode rows.")

    state_sequence = torch.stack(
        [torch.tensor(row[config.data.action_target.pose_source_key], dtype=torch.float32) for row in target_rows],
        dim=0,
    )
    return state_sequence_to_pose_sequence(
        state_sequence,
        state_encoding=config.data.action_target.state_encoding,
    )


def _load_full_episode_rollout(
    *,
    dataset: LeRobotV2WindowDataset,
    episode_index: int,
    config,
):
    rows = dataset._load_episode_rows(episode_index)  # noqa: SLF001
    if not rows:
        raise ValueError(f"Episode {episode_index} is empty.")

    state_sequence = torch.stack(
        [torch.tensor(row[config.data.action_target.pose_source_key], dtype=torch.float32) for row in rows],
        dim=0,
    )
    raw_action_sequence = torch.stack(
        [torch.tensor(row[config.data.action_target.source_key], dtype=torch.float32) for row in rows],
        dim=0,
    )
    original_pose = state_sequence_to_pose_sequence(
        state_sequence,
        state_encoding=config.data.action_target.state_encoding,
    )
    relative_targets, _, metadata = build_relative_pose_targets(
        state_sequence,
        state_encoding=config.data.action_target.state_encoding,
        rotation_representation=config.data.action_target.rotation_representation,
        include_gripper=config.data.action_target.include_gripper,
        gripper_representation=config.data.action_target.gripper_representation,
        raw_action_sequence=raw_action_sequence,
        gripper_action_index=config.data.action_target.gripper_action_index,
    )
    episode_record = dataset.episode_records[episode_index]
    task_index = int(rows[0]["task_index"])
    task_text = dataset.metadata.tasks_by_index.get(task_index)
    if task_text is None and episode_record.tasks:
        task_text = episode_record.tasks[0]

    reference_position = torch.tensor(metadata["reference_position"], dtype=torch.float32)
    reference_quaternion_xyzw = torch.tensor(metadata["reference_quaternion_xyzw"], dtype=torch.float32)
    frame_indices = [int(row["frame_index"]) for row in rows]
    return (
        task_text,
        frame_indices,
        frame_indices,
        relative_targets,
        reference_position,
        reference_quaternion_xyzw,
        str(metadata["rotation_representation"]),
        original_pose,
    )


def _build_scene_xml(
    *,
    reference_position: torch.Tensor,
    reference_quaternion_xyzw: torch.Tensor,
    original_positions: torch.Tensor,
    original_quaternions_xyzw: torch.Tensor,
    reconstructed_positions: torch.Tensor,
    reconstructed_quaternions_xyzw: torch.Tensor,
    mode: str,
) -> str:
    trace_bodies: list[str] = []

    if mode in {"original", "compare"}:
        trace_bodies.extend(
            _build_trace_geoms(
                prefix="original_trace",
                positions=original_positions,
                quaternions_xyzw=original_quaternions_xyzw,
                geom_type="sphere",
                size="0.014",
                rgba_builder=lambda index, total: _trace_rgba(index=index, total=total, family="original"),
            )
        )

    if mode in {"reconstructed", "compare"}:
        trace_bodies.extend(
            _build_trace_geoms(
                prefix="reconstructed_trace",
                positions=reconstructed_positions,
                quaternions_xyzw=reconstructed_quaternions_xyzw,
                geom_type="box",
                size="0.012 0.006 0.004",
                rgba_builder=lambda index, total: _trace_rgba(index=index, total=total, family="reconstructed"),
            )
        )

    animated_bodies: list[str] = []
    if mode in {"original", "compare"}:
        animated_bodies.append(
            f"""
    <body name="animated_original" mocap="true" pos="{_vec3(reference_position)}" quat="{_quat_wxyz(reference_quaternion_xyzw)}">
      <geom type="sphere" size="0.017" rgba="0.10 0.70 0.25 0.70" contype="0" conaffinity="0"/>
    </body>"""
        )
    if mode in {"reconstructed", "compare"}:
        animated_bodies.append(
            f"""
    <body name="animated_reconstructed" mocap="true" pos="{_vec3(reference_position)}" quat="{_quat_wxyz(reference_quaternion_xyzw)}">
      <geom type="box" size="0.012 0.006 0.004" rgba="0.95 0.45 0.10 0.90" contype="0" conaffinity="0"/>
    </body>"""
        )

    return f"""
<mujoco model="open_wam_libero_pose_compare">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.6" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.01" rgba="0.95 0.95 0.95 1"/>
    <camera name="overview" pos="0.0 -1.1 0.95" xyaxes="1 0 0 0 0.75 0.66"/>
    <body name="reference_pose" pos="{_vec3(reference_position)}" quat="{_quat_wxyz(reference_quaternion_xyzw)}">
      <geom type="box" size="0.020 0.010 0.006" rgba="0.15 0.40 0.95 1.0" contype="0" conaffinity="0"/>
    </body>
    {''.join(animated_bodies)}
    {''.join(trace_bodies)}
  </worldbody>
</mujoco>
""".strip()


def _build_offscreen_camera(
    *,
    reference_position: torch.Tensor,
    original_positions: torch.Tensor,
    reconstructed_positions: torch.Tensor,
) -> mujoco.MjvCamera:
    all_positions = torch.cat(
        [
            reference_position.unsqueeze(0),
            original_positions,
            reconstructed_positions,
        ],
        dim=0,
    )
    center = all_positions.mean(dim=0)
    radius = torch.linalg.vector_norm(all_positions - center.unsqueeze(0), dim=-1).max().item()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center.detach().cpu().numpy()
    camera.distance = max(radius * 3.2, 0.55)
    camera.azimuth = 145.0
    camera.elevation = -28.0
    return camera


def _build_trace_geoms(
    *,
    prefix: str,
    positions: torch.Tensor,
    quaternions_xyzw: torch.Tensor,
    geom_type: str,
    size: str,
    rgba_builder,
) -> list[str]:
    traces: list[str] = []
    for index, (position, quaternion_xyzw) in enumerate(zip(positions, quaternions_xyzw, strict=True)):
        traces.append(
            f"""
    <body name="{prefix}_{index}" pos="{_vec3(position)}" quat="{_quat_wxyz(quaternion_xyzw)}">
      <geom type="{geom_type}" size="{size}" rgba="{rgba_builder(index, len(positions))}" contype="0" conaffinity="0"/>
    </body>"""
        )
    return traces


def _resolve_mocap_handles(*, model: mujoco.MjModel, mode: str) -> dict[str, int]:
    handles: dict[str, int] = {}
    for body_name in ("animated_original", "animated_reconstructed"):
        if (body_name == "animated_original" and mode not in {"original", "compare"}) or (
            body_name == "animated_reconstructed" and mode not in {"reconstructed", "compare"}
        ):
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Could not find body `{body_name}` in the MuJoCo scene.")
        mocap_index = int(model.body_mocapid[body_id])
        if mocap_index < 0:
            raise RuntimeError(f"Body `{body_name}` is not backed by a mocap slot.")
        handles[body_name] = mocap_index
    return handles


def _move_to_reference(
    *,
    data: mujoco.MjData,
    mocap_handles: dict[str, int],
    reference_position: torch.Tensor,
    reference_quaternion_xyzw: torch.Tensor,
) -> None:
    for mocap_index in mocap_handles.values():
        _set_mocap_pose(
            data=data,
            mocap_index=mocap_index,
            position=reference_position,
            quaternion_xyzw=reference_quaternion_xyzw,
        )


def _set_mocap_pose(
    *,
    data: mujoco.MjData,
    mocap_index: int,
    position: torch.Tensor,
    quaternion_xyzw: torch.Tensor,
) -> None:
    data.mocap_pos[mocap_index] = position.detach().cpu().numpy()
    data.mocap_quat[mocap_index] = _quat_wxyz_array(quaternion_xyzw)


def _quat_wxyz(quaternion_xyzw: torch.Tensor) -> str:
    return " ".join(f"{value:.6f}" for value in _quat_wxyz_array(quaternion_xyzw))


def _quat_wxyz_array(quaternion_xyzw: torch.Tensor) -> list[float]:
    quat = quaternion_xyzw.detach().cpu().tolist()
    return [quat[3], quat[0], quat[1], quat[2]]


def _vec3(vector: torch.Tensor) -> str:
    return " ".join(f"{value:.6f}" for value in vector.detach().cpu().tolist())


def _trace_rgba(*, index: int, total: int, family: str) -> str:
    alpha = 0.25 + 0.55 * ((index + 1) / max(total, 1))
    if family == "original":
        return f"0.10 {0.55 + 0.30 * (index / max(total - 1, 1)):.3f} 0.25 {alpha:.3f}"
    return f"0.95 {0.55 + 0.25 * (index / max(total - 1, 1)):.3f} 0.10 {alpha:.3f}"


def _write_animation(*, output_path: Path, frames: list[np.ndarray], frame_duration_seconds: float) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output_path, frames, duration=frame_duration_seconds)
        return
    if suffix == ".mp4":
        fps = max(int(round(1.0 / max(frame_duration_seconds, 1e-3))), 1)
        imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
        return
    raise ValueError(f"Unsupported output format `{suffix}`. Use .gif or .mp4.")


if __name__ == "__main__":
    main()
