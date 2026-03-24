from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (  # noqa: E402
    LeRobotV2WindowDataset,
    build_lerobot_train_val_episode_split,
    reconstruct_absolute_pose_targets,
)
from open_wam.utils import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LIBERO reference-relative EEF pose targets in MuJoCo.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/contract_only_libero.yaml",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.7)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.data.action_target.representation != "eef_pose_relative_to_reference":
        raise ValueError(
            "This visualizer expects `data.action_target.representation: eef_pose_relative_to_reference`."
        )

    train_episodes, _ = build_lerobot_train_val_episode_split(config.data)
    dataset = LeRobotV2WindowDataset(config.data, episodes=train_episodes)
    sample = dataset[args.sample_index]

    reference_position = torch.tensor(sample.metadata["reference_position"], dtype=torch.float32)
    reference_quaternion_xyzw = torch.tensor(sample.metadata["reference_quaternion_xyzw"], dtype=torch.float32)
    absolute_pose = reconstruct_absolute_pose_targets(
        reference_position=reference_position,
        reference_quaternion=reference_quaternion_xyzw,
        relative_pose_targets=sample.actions,
    )

    print("task_text:", sample.task_text)
    print("action_representation:", sample.metadata["action_representation"])
    print("reference_position:", sample.metadata["reference_position"])
    print("reference_quaternion_xyzw:", sample.metadata["reference_quaternion_xyzw"])
    print("action_frame_indices:", sample.metadata["action_frame_indices"])
    print("target_state_frame_indices:", sample.metadata["target_state_frame_indices"])
    print("first_relative_pose:", sample.actions[0].tolist())

    model = mujoco.MjModel.from_xml_string(
        _build_scene_xml(
            reference_position=reference_position,
            reference_quaternion_xyzw=reference_quaternion_xyzw,
            target_positions=absolute_pose.position,
            target_quaternions_xyzw=absolute_pose.quaternion,
        )
    )
    data = mujoco.MjData(model)

    target_mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "animated_target")
    if target_mocap_id < 0:
        raise RuntimeError("Could not find animated target body in MuJoCo scene.")
    target_mocap_index = model.body_mocapid[target_mocap_id]
    if target_mocap_index < 0:
        raise RuntimeError("Animated target body is not backed by a mocap slot.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _set_mocap_pose(
                data=data,
                mocap_index=target_mocap_index,
                position=reference_position,
                quaternion_xyzw=reference_quaternion_xyzw,
            )
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(args.sleep_seconds * 0.5, 0.1))

            for step_index in range(len(absolute_pose.position)):
                if not viewer.is_running():
                    break
                _set_mocap_pose(
                    data=data,
                    mocap_index=target_mocap_index,
                    position=absolute_pose.position[step_index],
                    quaternion_xyzw=absolute_pose.quaternion[step_index],
                )
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(args.sleep_seconds)

            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break


def _build_scene_xml(
    *,
    reference_position: torch.Tensor,
    reference_quaternion_xyzw: torch.Tensor,
    target_positions: torch.Tensor,
    target_quaternions_xyzw: torch.Tensor,
) -> str:
    trace_bodies = []
    for index, (position, quaternion_xyzw) in enumerate(zip(target_positions, target_quaternions_xyzw, strict=True)):
        rgba = _trace_rgba(index=index, total=len(target_positions))
        trace_bodies.append(
            f"""
        <body name="trace_{index}" pos="{_vec3(position)}" quat="{_quat_wxyz(quaternion_xyzw)}">
          <geom type="box" size="0.015 0.008 0.005" rgba="{rgba}" contype="0" conaffinity="0"/>
        </body>"""
        )

    return f"""
<mujoco model="open_wam_libero_reference_pose">
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
      <geom type="box" size="0.02 0.01 0.006" rgba="0.15 0.4 0.95 1" contype="0" conaffinity="0"/>
    </body>
    <body name="animated_target" mocap="true" pos="{_vec3(reference_position)}" quat="{_quat_wxyz(reference_quaternion_xyzw)}">
      <geom type="box" size="0.018 0.009 0.005" rgba="0.95 0.45 0.1 1" contype="0" conaffinity="0"/>
    </body>
    {''.join(trace_bodies)}
  </worldbody>
</mujoco>
""".strip()


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


def _trace_rgba(*, index: int, total: int) -> str:
    # Earlier targets are lighter; later targets get more saturated so the
    # rollout direction is visible at a glance.
    alpha = 0.25 + 0.55 * ((index + 1) / max(total, 1))
    green = 0.55 + 0.25 * (index / max(total - 1, 1))
    return f"0.95 {green:.3f} 0.15 {alpha:.3f}"


if __name__ == "__main__":
    main()
