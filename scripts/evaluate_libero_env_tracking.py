from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (  # noqa: E402
    LeRobotV2WindowDataset,
    build_canonical_video_preprocessor,
    build_lerobot_train_val_episode_split,
    build_relative_pose_targets,
)
from open_wam.integrations import (  # noqa: E402
    LiberoControlConfig,
    infer_task_local_episode_rank,
    track_relative_targets_in_libero_env,
)
from open_wam.configs import load_experiment_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one OpenWAM public LIBERO trajectory in the real LIBERO "
            "environment and save a side-by-side video of original vs env replay."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_raw_smoke.yaml",
    )
    parser.add_argument(
        "--trajectory",
        choices=("sample", "episode"),
        default="sample",
        help="Use either one sampled horizon or the entire episode trajectory.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument(
        "--init-state-index",
        type=int,
        default=None,
        help="Benchmark init-state index. Defaults to the task-local episode rank.",
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--control-substeps-per-target",
        type=int,
        default=4,
        help=(
            "Env control steps to spend on each dataset target. `4` is the "
            "current default because it fixes the old over-held command "
            "behavior while keeping pose tracking tighter than the strict "
            "10FPS-to-20Hz match of `2`."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/libero_tracking_compare.gif",
        help="Animation output path (.gif or .mp4) for the side-by-side original vs env replay.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional cap on the number of target waypoints for quick debugging.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.data.action_target.representation != "eef_pose_relative_to_reference":
        raise ValueError(
            "This evaluator expects `data.action_target.representation: eef_pose_relative_to_reference`."
        )

    train_episodes, _ = build_lerobot_train_val_episode_split(config.data)
    dataset = LeRobotV2WindowDataset(config.data, episodes=train_episodes)
    sample = dataset[args.sample_index]
    control_substeps_per_target = args.control_substeps_per_target

    (
        task_text,
        episode_index,
        relative_targets,
        rotation_representation,
        reference_position,
        reference_quaternion,
        original_frames,
        task_local_episode_rank,
    ) = _load_public_trajectory(
        dataset=dataset,
        sample=sample,
        config=config,
        trajectory=args.trajectory,
        episode_index=args.episode_index,
    )

    if args.max_targets is not None:
        relative_targets = relative_targets[: args.max_targets]
        original_frames = original_frames[: args.max_targets]

    init_state_index = args.init_state_index if args.init_state_index is not None else task_local_episode_rank
    tracking = track_relative_targets_in_libero_env(
        task_text=task_text,
        relative_pose_targets=relative_targets,
        rotation_representation=rotation_representation,
        reference_position=reference_position,
        reference_quaternion=reference_quaternion,
        gripper_representation=config.data.action_target.gripper_representation,
        init_state_index=init_state_index,
        control_config=LiberoControlConfig(
            control_substeps_per_target=control_substeps_per_target,
        ),
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )

    video_frames = _build_comparison_video_frames(
        original_frames=original_frames,
        env_frames=tracking.camera_frames,
        rendered_target_indices=tracking.rendered_target_indices,
        position_error_per_target=tracking.position_error_per_target,
        rotation_error_deg_per_target=tracking.rotation_error_deg_per_target,
        gripper_error_per_target=tracking.gripper_error_per_target,
        task_text=task_text,
        trajectory=args.trajectory,
        episode_index=episode_index,
        init_state_index=tracking.init_state_index,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_animation(output_path=output_path, frames=video_frames, frame_duration_seconds=0.08)

    print("task_text:", task_text)
    print("trajectory:", args.trajectory)
    print("episode_index:", episode_index)
    print("task_local_episode_rank:", task_local_episode_rank)
    print("init_state_index:", tracking.init_state_index)
    print("num_targets:", int(relative_targets.shape[0]))
    print("control_substeps_per_target:", control_substeps_per_target)
    print("mean_position_error_m:", float(tracking.position_error_per_target.mean()))
    print("max_position_error_m:", float(tracking.position_error_per_target.max()))
    print("mean_rotation_error_deg:", float(tracking.rotation_error_deg_per_target.mean()))
    print("max_rotation_error_deg:", float(tracking.rotation_error_deg_per_target.max()))
    print("mean_gripper_error:", float(tracking.gripper_error_per_target.mean()))
    print("max_gripper_error:", float(tracking.gripper_error_per_target.max()))
    print("saved_animation:", str(output_path.resolve()))


def _load_public_trajectory(
    *,
    dataset: LeRobotV2WindowDataset,
    sample,
    config,
    trajectory: str,
    episode_index: int | None,
) -> tuple[str, int, torch.Tensor, str, torch.Tensor, torch.Tensor, list[Image.Image], int]:
    canonicalizer = build_canonical_video_preprocessor(config.data)

    if trajectory == "sample":
        task_text = sample.task_text
        resolved_episode_index = int(sample.metadata["episode_index"])
        rows = dataset._load_episode_rows(resolved_episode_index)  # noqa: SLF001
        target_rows = [rows[int(frame_index)] for frame_index in sample.metadata["target_state_frame_indices"]]
        public_targets = sample.actions
        reference_position = torch.tensor(sample.metadata["reference_position"], dtype=torch.float32)
        reference_quaternion = torch.tensor(sample.metadata["reference_quaternion_xyzw"], dtype=torch.float32)
        original_frames = [
            _canonicalize_row_views(
                canonicalizer=canonicalizer,
                image_row=row,
                image_key_map={
                    "image": "image",
                    "wrist_image": "wrist_image",
                },
            )
            for row in target_rows
        ]
    else:
        resolved_episode_index = episode_index if episode_index is not None else int(sample.metadata["episode_index"])
        rows = dataset._load_episode_rows(resolved_episode_index)  # noqa: SLF001
        state_sequence = torch.stack(
            [torch.tensor(row[config.data.action_target.pose_source_key], dtype=torch.float32) for row in rows],
            dim=0,
        )
        raw_action_sequence = torch.stack(
            [torch.tensor(row[config.data.action_target.source_key], dtype=torch.float32) for row in rows],
            dim=0,
        )
        public_targets, _, metadata = build_relative_pose_targets(
            state_sequence,
            state_encoding=config.data.action_target.state_encoding,
            rotation_representation=config.data.action_target.rotation_representation,
            include_gripper=config.data.action_target.include_gripper,
            gripper_representation=config.data.action_target.gripper_representation,
            raw_action_sequence=raw_action_sequence,
            gripper_action_index=config.data.action_target.gripper_action_index,
        )
        reference_position = torch.tensor(metadata["reference_position"], dtype=torch.float32)
        reference_quaternion = torch.tensor(metadata["reference_quaternion_xyzw"], dtype=torch.float32)
        original_frames = [
            _canonicalize_row_views(
                canonicalizer=canonicalizer,
                image_row=row,
                image_key_map={
                    "image": "image",
                    "wrist_image": "wrist_image",
                },
            )
            for row in rows
        ]
        task_index = int(rows[0]["task_index"])
        task_text = dataset.metadata.tasks_by_index.get(task_index)
        if task_text is None:
            task_text = dataset.episode_records[resolved_episode_index].tasks[0]

    task_local_episode_rank = infer_task_local_episode_rank(
        dataset.metadata.episodes,
        episode_index=resolved_episode_index,
        task_text=task_text,
    )
    return (
        task_text,
        resolved_episode_index,
        public_targets,
        config.data.action_target.rotation_representation,
        reference_position,
        reference_quaternion,
        original_frames,
        task_local_episode_rank,
    )


def _canonicalize_row_views(*, canonicalizer, image_row: dict, image_key_map: dict[str, str]) -> Image.Image:
    views = {}
    dataset_owner = None
    for public_name, row_key in image_key_map.items():
        if dataset_owner is None:
            dataset_owner = image_row[row_key]
        image_bytes = image_row[row_key]["bytes"]
        tensor = _decode_image_bytes(image_bytes)
        views[public_name] = tensor.unsqueeze(0)
    frame = canonicalizer(views).video[0, :, 0].permute(1, 2, 0).clamp(0.0, 1.0)
    return Image.fromarray((frame.detach().cpu().numpy() * 255.0).astype("uint8"))


def _decode_image_bytes(image_bytes: bytes) -> torch.Tensor:
    import io
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        return torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(rgb.height, rgb.width, 3)


def _build_comparison_video_frames(
    *,
    original_frames: list[Image.Image],
    env_frames: dict[str, list],
    rendered_target_indices: list[int],
    position_error_per_target: torch.Tensor,
    rotation_error_deg_per_target: torch.Tensor,
    gripper_error_per_target: torch.Tensor,
    task_text: str,
    trajectory: str,
    episode_index: int,
    init_state_index: int,
) -> list:
    video_frames = []
    env_canonical_frames = [
        _canonicalize_env_pair(agentview=agentview, wrist=wrist)
        for agentview, wrist in zip(
            env_frames["agentview_image"],
            env_frames["robot0_eye_in_hand_image"],
            strict=True,
        )
    ]

    for frame_index, (target_index, env_frame) in enumerate(zip(rendered_target_indices, env_canonical_frames, strict=True)):
        original = original_frames[target_index]
        composite = Image.new("RGB", (original.width + env_frame.width, original.height + 56), color=(245, 245, 245))
        composite.paste(original, (0, 0))
        composite.paste(env_frame, (original.width, 0))

        draw = ImageDraw.Draw(composite)
        footer_top = original.height + 8
        draw.text((10, footer_top), f"task: {task_text}", fill=(15, 15, 15))
        draw.text((10, footer_top + 18), f"{trajectory} episode={episode_index} init_state={init_state_index}", fill=(15, 15, 15))
        draw.text(
            (10, footer_top + 36),
            (
                f"target={target_index + 1}/{len(original_frames)} frame={frame_index + 1}/{len(env_canonical_frames)} "
                f"pos_err={float(position_error_per_target[target_index]):.4f}m "
                f"rot_err={float(rotation_error_deg_per_target[target_index]):.2f}deg "
                f"grip_err={float(gripper_error_per_target[target_index]):.4f}"
            ),
            fill=(15, 15, 15),
        )
        video_frames.append(composite)
    return video_frames


def _canonicalize_env_pair(*, agentview, wrist) -> Image.Image:
    # robosuite / LIBERO offscreen renders arrive vertically flipped relative
    # to the dataset images saved in the HF export, so flip them before
    # composing the side-by-side comparison.
    top = Image.fromarray(agentview).transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((320, 256))
    bottom = Image.fromarray(wrist).transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((320, 128))
    frame = Image.new("RGB", (320, 384))
    frame.paste(top, (0, 0))
    frame.paste(bottom, (0, 256))
    return frame


def _write_animation(*, output_path: Path, frames: list[Image.Image], frame_duration_seconds: float) -> None:
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
