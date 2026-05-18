from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
import sys
import time

from einops import rearrange
import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import run_libero_exact_visualization as exact_viz  # noqa: E402
from open_wam.configs import ActionTargetRepresentation, LiberoAbsoluteJointExecutionMode  # noqa: E402
from open_wam.data.action_transforms import expected_joint_position_target_dim  # noqa: E402
from open_wam.integrations import LiberoBenchmarkAdapter, LiberoEnvConfig  # noqa: E402
from open_wam.pipelines import LingbotExactRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.simulators import (  # noqa: E402
    EpisodeSpec,
    SimActionCommitMode,
    SimRolloutResult,
    SimulatorObservation,
    SimulatorStepResult,
    summarize_sim_rollout,
)
from open_wam.data.action_transforms import normalize_quaternion, quaternion_to_axis_angle  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402


LIBERO_RAW_VIEW_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def main() -> None:
    args = _parse_args()
    seed_everywhere(args.seed)
    config_path = _resolve_repo_path(args.config)
    config = load_experiment_config(config_path)
    checkpoint_path = _resolve_checkpoint_file(Path(args.checkpoint))
    _apply_checkpoint_backbone_override(config, checkpoint_path=checkpoint_path)
    rollout_action_mode = _resolve_rollout_action_mode(config)

    device = _resolve_device(args.device)
    pipeline = build_variant_pipeline_from_config(config)
    pipeline.eval()
    pipeline.visual_tower.ensure_runtime_backbone_device(
        action_dim=int(config.action_decoder.action_dim),
        device=device,
    )
    runner = LingbotExactRunner(pipeline)
    adapter = _MappedLiberoAbsJointAdapter(
        config=LiberoEnvConfig(
            benchmark_name=args.benchmark,
            controller=rollout_action_mode["controller"],
            action_mode=rollout_action_mode["action_mode"],
            camera_obs_keys=LIBERO_RAW_VIEW_KEYS,
            render_camera_key="agentview_image",
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            horizon=args.env_horizon,
            ignore_done=True,
            control_freq=(
                args.env_control_freq
                if args.env_control_freq is not None
                else int(rollout_action_mode["default_control_freq"])
            ),
            init_state_index=args.init_state_index,
            absolute_joint_execution_mode=LiberoAbsoluteJointExecutionMode(args.absolute_joint_execution_mode),
            absolute_joint_substeps_per_target=args.absolute_joint_substeps_per_target,
            absolute_joint_gripper_substep_policy=args.absolute_joint_gripper_substep_policy,
            absolute_joint_kp=args.absolute_joint_kp,
            absolute_joint_disable_interpolator=args.absolute_joint_disable_interpolator,
        ),
        model_camera_names=tuple(config.data.camera_names),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"libero_abs_joint_{args.suffix}.mp4"
    summary_path = output_dir / f"libero_abs_joint_{args.suffix}.json"
    try:
        result = _run_abs_joint_exact_rollout(
            adapter=adapter,
            runner=runner,
            data_config=config.data,
            config=config,
            runtime_device=device,
            frontend_device=device,
            decode_device=device,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            seed=args.seed,
            max_steps=args.max_actions,
            target_action_hz=args.target_action_hz,
            action_commit_mode=SimActionCommitMode(args.action_commit_mode),
            video_path=video_path,
            video_fps=args.video_fps,
        )
    finally:
        adapter.close()

    summary = summarize_sim_rollout(result, video_path=str(video_path) if video_path.is_file() else None)
    summary.update(
        {
            "config": str(config_path),
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_dir": str(checkpoint_path.parent),
            "transformer_dir": str(Path(str(config.backbone.transformer_subdir)).resolve()),
            "device": str(device),
            "action_target_representation": str(config.data.action_target.representation),
            "libero_controller": rollout_action_mode["controller"],
            "libero_action_mode": rollout_action_mode["action_mode"],
            "action_commit_mode": args.action_commit_mode,
            "absolute_joint_execution_mode": args.absolute_joint_execution_mode,
            "absolute_joint_substeps_per_target": int(args.absolute_joint_substeps_per_target),
            "absolute_joint_gripper_substep_policy": args.absolute_joint_gripper_substep_policy,
            "absolute_joint_kp": args.absolute_joint_kp,
            "absolute_joint_disable_interpolator": bool(args.absolute_joint_disable_interpolator),
            "env_control_freq": args.env_control_freq,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_abs_joint_exact_rollout(
    *,
    adapter: "_MappedLiberoAbsJointAdapter",
    runner: LingbotExactRunner,
    data_config: Any,
    config: Any,
    runtime_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    task_id: int,
    episode_idx: int,
    seed: int,
    max_steps: int,
    target_action_hz: float | None,
    action_commit_mode: SimActionCommitMode,
    video_path: Path,
    video_fps: float,
) -> SimRolloutResult:
    if action_commit_mode != SimActionCommitMode.FULL_CHUNK:
        raise ValueError(
            "The exact absolute-joint debug runner intentionally mirrors the established M1 visualizer "
            "and currently supports only action_commit_mode='full_chunk'."
        )

    start_wall = time.perf_counter()
    observation = adapter.reset(EpisodeSpec(task_id=task_id, episode_idx=episode_idx, seed=seed))
    task_text = observation.task_text or adapter.task_text()
    session = runner.reset(task_text=(task_text,))
    first_obs = adapter.real_obs_list[-1]
    latest_observation = observation

    predicted_latent_chunks: list[torch.Tensor] = []
    action_records: list[dict[str, Any]] = []
    policy_action_shapes: list[tuple[int, ...]] = []
    policy_step_times: list[float] = []
    env_step_times: list[float] = []
    first_chunk = True
    chunk_count = 0
    success = False
    done = False
    steps = 0

    while steps < int(max_steps) and not done and not success:
        seed_everywhere(int(seed) + chunk_count)
        policy_start = time.perf_counter()
        if first_chunk:
            first_chunk_inputs = exact_viz._prepare_exact_runtime_inputs(
                runner,
                views=exact_viz._obs_list_to_views([first_obs], config=config, device=frontend_device),
                task_text=(task_text,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            chunk = runner.infer_chunk(
                session=session,
                video_latents=first_chunk_inputs["video_latents"],
                text_context=first_chunk_inputs["text_context"],
                negative_text_context=first_chunk_inputs["negative_text_context"],
                proprio_state=_proprio_state_tensor(latest_observation, device=runtime_device),
            )
        else:
            chunk = runner.infer_chunk(
                session=session,
                proprio_state=_proprio_state_tensor(latest_observation, device=runtime_device),
            )
        policy_elapsed = time.perf_counter() - policy_start
        policy_step_times.append(policy_elapsed)

        raw_chunk_action_pred = chunk.raw_chunk_action_pred
        if raw_chunk_action_pred is None:
            raise RuntimeError("Exact abs-joint rollout expected raw target-space action predictions.")
        predicted_latent_chunks.append(chunk.predicted_latents.detach().cpu())
        policy_action_shapes.append(tuple(int(dim) for dim in raw_chunk_action_pred.shape))
        session = chunk.session

        raw_actions = rearrange(
            raw_chunk_action_pred[0],
            "(f a) c -> f a c",
            f=int(config.inference.frame_chunk_size),
            a=int(config.policy_variant.action_per_frame),
        )
        raw_actions_batched = raw_actions.unsqueeze(0)
        obs_stride = max(1, raw_actions.shape[1] // max(1, int(config.inference.frame_chunk_size)))
        key_frame_list: list[dict[str, np.ndarray]] = []

        for frame_group in range(raw_actions.shape[0]):
            for action_index in range(raw_actions.shape[1]):
                if steps >= int(max_steps):
                    break
                raw_action = raw_actions[frame_group, action_index].detach().to(dtype=torch.float32).cpu().numpy()
                env_action = adapter.action_from_model_action(raw_action, data_config=data_config)
                env_start = time.perf_counter()
                transition = adapter.step(env_action.astype(np.float32, copy=False))
                env_elapsed = time.perf_counter() - env_start
                env_step_times.append(env_elapsed)
                latest_observation = transition.observation
                success = bool(transition.success)
                done = bool(transition.done)
                action_records.append(
                    {
                        "step_index": int(steps),
                        "plan_index": int(chunk_count),
                        "chunk_action_index": int(frame_group * raw_actions.shape[1] + action_index),
                        "policy_step_s": float(policy_elapsed) if action_index == 0 and frame_group == 0 else 0.0,
                        "env_step_s": float(env_elapsed),
                        "model_action_dim": int(raw_action.shape[-1]),
                        "env_action_dim": int(env_action.shape[-1]),
                        "reused_policy_output": not (action_index == 0 and frame_group == 0),
                        "action_commit_mode": action_commit_mode.value,
                        "success": success,
                    }
                )
                steps += 1
                if (action_index + 1) % obs_stride == 0 and adapter.real_obs_list:
                    key_frame_list.append(adapter.real_obs_list[-1])
                if done or success:
                    break
            if steps >= int(max_steps) or done or success:
                break

        chunk_count += 1
        if steps >= int(max_steps) or done or success or not key_frame_list:
            break

        if first_chunk:
            new_visual_outputs = exact_viz._prepare_exact_runtime_inputs(
                runner,
                views=exact_viz._obs_list_to_views(key_frame_list, config=config, device=frontend_device),
                task_text=(task_text,),
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                preserve_stream_cache=True,
            )
            if chunk.visual_outputs is None:
                raise RuntimeError("First exact abs-joint chunk did not retain visual outputs for cache warmup.")
            combined_latents = torch.cat(
                [chunk.visual_outputs.frontend.video_latents, new_visual_outputs["video_latents"]],
                dim=2,
            )
            warmup = runner.warmup_cache(
                session=session,
                video_latents=combined_latents,
                text_context=new_visual_outputs["text_context"],
                negative_text_context=new_visual_outputs["negative_text_context"],
                action_history=raw_actions_batched,
                action_space="raw",
                proprio_state=_proprio_state_tensor(latest_observation, device=runtime_device),
            )
        else:
            warmup_inputs = exact_viz._prepare_exact_runtime_inputs(
                runner,
                views=exact_viz._obs_list_to_views(key_frame_list, config=config, device=frontend_device),
                task_text=(task_text,),
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
                frontend_device=frontend_device,
                runtime_device=runtime_device,
                preserve_stream_cache=True,
            )
            warmup = runner.warmup_cache(
                session=session,
                video_latents=warmup_inputs["video_latents"],
                text_context=warmup_inputs["text_context"],
                negative_text_context=warmup_inputs["negative_text_context"],
                action_history=raw_actions_batched,
                action_space="raw",
                proprio_state=_proprio_state_tensor(latest_observation, device=runtime_device),
            )
        session = warmup.session
        first_chunk = False

    video_frames = _build_exact_style_video_frames(
        runner=runner,
        predicted_latent_chunks=predicted_latent_chunks,
        adapter=adapter,
        decode_device=decode_device,
    )
    if video_frames:
        imageio.mimsave(video_path, video_frames, fps=float(video_fps), macro_block_size=1)

    wall_time_s = time.perf_counter() - start_wall
    return SimRolloutResult(
        benchmark=adapter.benchmark_name,
        task_text=task_text,
        success=bool(success),
        steps=int(steps),
        target_action_hz=target_action_hz,
        wall_time_s=float(wall_time_s),
        mean_policy_step_s=float(np.mean(policy_step_times)) if policy_step_times else None,
        mean_env_step_s=float(np.mean(env_step_times)) if env_step_times else None,
        achieved_action_hz=float(steps / wall_time_s) if wall_time_s > 0 else 0.0,
        policy_action_shapes=tuple(policy_action_shapes),
        action_records=tuple(action_records),
        video_frames=tuple(video_frames),
    )


class _MappedLiberoAbsJointAdapter:
    """Map LIBERO raw camera keys into the experiment's LeRobot camera names."""

    def __init__(self, *, config: LiberoEnvConfig, model_camera_names: tuple[str, ...]) -> None:
        if len(model_camera_names) != len(LIBERO_RAW_VIEW_KEYS):
            raise ValueError(
                "Abs-joint LIBERO debug rollout expects exactly two model camera names, "
                f"got {model_camera_names}."
            )
        self.inner = LiberoBenchmarkAdapter(config)
        self.model_camera_names = tuple(model_camera_names)
        self.benchmark_name = self.inner.benchmark_name
        self.capabilities = self.inner.capabilities
        self.real_obs_list: list[dict[str, np.ndarray]] = []

    def reset(self, spec: Any) -> SimulatorObservation:
        self.real_obs_list.clear()
        return self._map_observation(self.inner.reset(spec))

    def task_text(self) -> str | None:
        return self.inner.task_text()

    def action_from_model_action(self, model_action: np.ndarray, *, data_config: Any) -> np.ndarray:
        return self.inner.action_from_model_action(model_action, data_config=data_config)

    def step(self, action: np.ndarray) -> SimulatorStepResult:
        transition = self.inner.step(action)
        return replace(transition, observation=self._map_observation(transition.observation))

    def render_frame(self, observation: SimulatorObservation) -> np.ndarray | None:
        if self.model_camera_names[0] in observation.views:
            return np.asarray(observation.views[self.model_camera_names[0]], dtype=np.uint8)
        return self.inner.render_frame(observation)

    def close(self) -> None:
        self.inner.close()

    def _map_observation(self, observation: SimulatorObservation) -> SimulatorObservation:
        raw_views = observation.views
        mapped_views = {
            model_key: np.ascontiguousarray(raw_views[raw_key][::-1], dtype=np.uint8)
            for model_key, raw_key in zip(self.model_camera_names, LIBERO_RAW_VIEW_KEYS, strict=True)
            if raw_key in raw_views
        }
        if len(mapped_views) != len(self.model_camera_names):
            raise KeyError(
                f"LIBERO observation did not expose all required raw views {LIBERO_RAW_VIEW_KEYS}; "
                f"available={sorted(raw_views)}."
            )
        exact_obs = {
            exact_key: np.ascontiguousarray(raw_views[raw_key][::-1])
            for exact_key, raw_key in zip(exact_viz.LIBERO_OBS_KEYS, LIBERO_RAW_VIEW_KEYS, strict=True)
            if raw_key in raw_views
        }
        if len(exact_obs) == len(exact_viz.LIBERO_OBS_KEYS):
            self.real_obs_list.append(exact_obs)
        return SimulatorObservation(
            views=mapped_views,
            state=_eef_axis_angle_state(observation.raw),
            task_text=observation.task_text,
            raw=observation.raw,
            metadata=dict(observation.metadata),
        )


def _build_exact_style_video_frames(
    *,
    runner: LingbotExactRunner,
    predicted_latent_chunks: list[torch.Tensor],
    adapter: _MappedLiberoAbsJointAdapter,
    decode_device: torch.device,
) -> list[np.ndarray]:
    imagined_video = exact_viz._decode_imagined_video(
        runner,
        predicted_latent_chunks,
        decode_device=decode_device,
    )
    return exact_viz._build_comparison_video_frames(
        real_obs_list=adapter.real_obs_list,
        imagined_video=imagined_video,
    )


def _eef_axis_angle_state(obs: dict[str, Any]) -> np.ndarray:
    position = torch.as_tensor(obs["robot0_eef_pos"], dtype=torch.float32)
    quaternion = normalize_quaternion(torch.as_tensor(obs["robot0_eef_quat"], dtype=torch.float32).unsqueeze(0))[0]
    axis_angle = quaternion_to_axis_angle(quaternion.unsqueeze(0))[0]
    gripper = torch.as_tensor(obs["robot0_gripper_qpos"], dtype=torch.float32).reshape(-1)[:2]
    return torch.cat([position, axis_angle, gripper], dim=0).numpy().astype(np.float32)


def _proprio_state_tensor(observation: SimulatorObservation, *, device: torch.device) -> torch.Tensor:
    if observation.state is None:
        raise ValueError("Abs-action rollout with proprio context requires simulator observation.state.")
    return torch.as_tensor(observation.state, dtype=torch.float32, device=device).reshape(1, -1)


def _resolve_rollout_action_mode(config: Any) -> dict[str, Any]:
    representation = ActionTargetRepresentation(config.data.action_target.representation)
    if representation is ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION:
        expected_dim = expected_joint_position_target_dim(
            joint_dim=len(tuple(config.data.action_target.joint_position_normalization.lower)),
            include_gripper=bool(config.data.action_target.include_gripper),
            gripper_representation=config.data.action_target.gripper_representation,
        )
        if int(config.data.action_schema.action_dim) != int(expected_dim):
            raise ValueError(
                f"Expected {expected_dim}D abs-joint target actions, got {config.data.action_schema.action_dim}."
            )
        return {
            "controller": "JOINT_POSITION",
            "action_mode": "absolute_joint_position",
            "default_control_freq": 20,
        }

    source_key = str(getattr(config.data.action_target, "source_key", "") or "")
    rotation_representation = str(getattr(config.data.action_target, "rotation_representation", "") or "")
    if (
        representation is ActionTargetRepresentation.RAW
        and source_key == "integrated_eef6d_action"
        and int(config.data.action_schema.action_dim) == 10
        and rotation_representation == "continuous_6d"
    ):
        return {
            "controller": "OSC_POSE",
            "action_mode": "integrated_eef6d_osc",
            "default_control_freq": 20,
        }

    raise ValueError(
        "This debug runner supports absolute_joint_position targets or raw integrated_eef6d_action targets, "
        f"got representation={representation}, source_key={source_key!r}, "
        f"action_dim={config.data.action_schema.action_dim}, rotation_representation={rotation_representation!r}."
    )


def _apply_checkpoint_backbone_override(config: Any, *, checkpoint_path: Path) -> None:
    transformer_dir = checkpoint_path.parent / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Expected exported transformer directory next to checkpoint: {transformer_dir}")
    object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))


def _resolve_checkpoint_file(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return candidate
    for filename in ("model_state.pt", "full_training_state.pt"):
        checkpoint_file = candidate / filename
        if checkpoint_file.is_file():
            return checkpoint_file
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    for checkpoint_dir in reversed(checkpoint_dirs):
        for filename in ("model_state.pt", "full_training_state.pt"):
            checkpoint_file = checkpoint_dir / filename
            if checkpoint_file.is_file():
                return checkpoint_file
    raise FileNotFoundError(f"Could not resolve model_state.pt or full_training_state.pt from {path}.")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested CUDA device {device}, but CUDA is not available.")
    return device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LIBERO debug rollout for M1 target-action checkpoints.")
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--target-action-hz", type=float, default=None)
    parser.add_argument(
        "--action-commit-mode",
        choices=tuple(mode.value for mode in SimActionCommitMode),
        default=SimActionCommitMode.FULL_CHUNK.value,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-horizon", type=int, default=5000)
    parser.add_argument("--env-control-freq", type=int, default=None)
    parser.add_argument(
        "--absolute-joint-execution-mode",
        choices=tuple(mode.value for mode in LiberoAbsoluteJointExecutionMode),
        default=LiberoAbsoluteJointExecutionMode.DIRECT_GOAL.value,
        help=(
            "`direct_goal` uses the adapter-owned absolute set_qpos controller hook. "
            "`normalized_delta` preserves the old public JOINT_POSITION delta mapping."
        ),
    )
    parser.add_argument("--absolute-joint-substeps-per-target", type=int, default=2)
    parser.add_argument(
        "--absolute-joint-gripper-substep-policy",
        choices=("repeat", "first_only", "last_only"),
        default="repeat",
    )
    parser.add_argument("--absolute-joint-kp", type=float, default=500.0)
    parser.add_argument("--absolute-joint-disable-interpolator", action="store_true")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", default="outputs/libero_abs_joint_debug_rollouts")
    parser.add_argument("--suffix", default="rollout")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
