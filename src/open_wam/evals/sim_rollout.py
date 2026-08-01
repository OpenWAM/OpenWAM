from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import torch

from open_wam.simulators import (
    run_closed_loop_sim_rollout,
    summarize_sim_rollout,
)
from open_wam.runtime import build_result_envelope, resolve_repo_path
from open_wam.runtime.checkpoints import (
    load_pipeline_checkpoint,
    resolve_checkpoint_file,
    resolve_checkpoint_step_dir_from_transformer_dir,
)
from open_wam.configs import load_experiment_config, load_local_path_registry
from open_wam.utils import seed_everywhere


def run_simulator_rollout_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one parsed simulator rollout command and persist its artifacts."""

    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive.")
    if args.target_action_hz is not None and args.target_action_hz <= 0:
        raise SystemExit("--target-action-hz must be positive when provided.")
    if args.video_fps <= 0:
        raise SystemExit("--video-fps must be positive.")

    from open_wam.extensions import load_extension_modules

    load_extension_modules(args.extension)
    seed_everywhere(args.seed)
    config_path = resolve_repo_path(args.config)
    config = load_experiment_config(config_path)
    device = _resolve_device(args.device)

    checkpoint_path = _resolve_checkpoint_for_config(config=config, checkpoint_arg=args.checkpoint)
    if checkpoint_path is not None:
        _apply_checkpoint_backbone_override(config, checkpoint_path=checkpoint_path)

    adapter = _build_adapter(args)
    if args.zero_policy:
        rollout_runner = _ZeroActionRolloutRunner(
            action_dim=config.data.action_schema.action_dim,
            action_horizon=config.data.action_schema.action_horizon,
            device=device,
        )
    else:
        from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config

        pipeline = build_variant_pipeline_from_config(config).to(device)
        pipeline.eval()
        if checkpoint_path is not None:
            checkpoint_report = load_pipeline_checkpoint(pipeline, checkpoint_path)
            if checkpoint_report.missing_keys:
                print(f"sim.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
            if checkpoint_report.unexpected_keys:
                print(f"sim.checkpoint_unexpected_keys {len(checkpoint_report.unexpected_keys)}")
        rollout_runner = VariantRolloutRunner(pipeline)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.benchmark}_{args.suffix}.mp4"
    summary_path = output_dir / f"{args.benchmark}_{args.suffix}.json"
    try:
        result = run_closed_loop_sim_rollout(
            adapter=adapter,
            rollout_runner=rollout_runner,
            data_config=config.data,
            device=device,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            seed=args.seed,
            max_steps=args.max_steps,
            target_action_hz=args.target_action_hz,
            action_commit_mode=args.action_commit_mode,
        )
    finally:
        adapter.close()

    saved_video_path: str | None = None
    if result.video_frames:
        imageio.mimsave(video_path, list(result.video_frames), fps=float(args.video_fps), macro_block_size=1)
        saved_video_path = str(video_path)

    legacy_summary = summarize_sim_rollout(result, video_path=saved_video_path)
    legacy_summary.update(
        {
            "config": str(config_path),
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "zero_policy": bool(args.zero_policy),
            "device": str(device),
            "action_commit_mode": args.action_commit_mode,
        }
    )
    summary = build_result_envelope(
        command="open-wam-sim-rollout",
        config=str(config_path),
        metrics={
            "success": bool(result.success),
            "steps": int(result.steps),
            "achieved_action_hz": float(result.achieved_action_hz),
        },
        artifacts={"video_path": saved_video_path, "summary_path": str(summary_path)},
        checkpoint=None if checkpoint_path is None else str(checkpoint_path),
        benchmark=args.benchmark,
        device=str(device),
        seed=int(args.seed),
        extra=legacy_summary,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return summary


class _ZeroActionRolloutRunner:
    def __init__(self, *, action_dim: int, action_horizon: int, device: torch.device) -> None:
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.device = device

    def reset(self, **_: Any) -> Any:
        return SimpleNamespace(policy_state=None)

    def infer_step(self, *, session: Any, context: Any, views: Any | None = None, **_: Any) -> Any:
        del context, views
        action_pred = torch.zeros(
            1,
            self.action_horizon,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        return SimpleNamespace(
            session=session,
            infer_output=SimpleNamespace(
                decoder_output=SimpleNamespace(action_pred=action_pred),
            ),
        )


def _build_adapter(args: argparse.Namespace) -> Any:
    registry = load_local_path_registry()
    if args.benchmark == "robotwin":
        from open_wam.integrations.robotwin_env import RobotwinBenchmarkAdapter, RobotwinEnvConfig

        root = args.robotwin_root or registry.get("simulators.robotwin_root")
        if root is None:
            raise SystemExit(
                "RoboTwin rollout requires --robotwin-root or paths.simulators.robotwin_root in configs/local_paths.yaml."
            )
        if args.robotwin_task_name is None:
            raise SystemExit("RoboTwin rollout requires --robotwin-task-name.")
        task_config = args.robotwin_task_config or args.robotwin_task_name
        return RobotwinBenchmarkAdapter(
            RobotwinEnvConfig(
                robotwin_root=root,
                task_name=args.robotwin_task_name,
                task_config=task_config,
                instruction=args.instruction,
                action_type=args.robotwin_action_type,
                expert_precheck=bool(args.robotwin_expert_precheck),
                instruction_type=args.robotwin_instruction_type,
            )
        )
    from open_wam.integrations.calvin_env import CalvinBenchmarkAdapter, CalvinEnvConfig

    calvin_root = args.calvin_root or registry.get("simulators.calvin_root")
    calvin_dataset_root = args.calvin_dataset_root or registry.get("datasets.calvin_root")
    return CalvinBenchmarkAdapter(
        CalvinEnvConfig(
            calvin_root=calvin_root,
            dataset_root=calvin_dataset_root,
            task_text=args.calvin_task_text or args.instruction,
            show_gui=bool(args.show_gui),
        )
    )


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested CUDA device {device}, but CUDA is not available.")
    return device


def _resolve_checkpoint_for_config(*, config: Any, checkpoint_arg: str | None) -> Path | None:
    if checkpoint_arg is not None:
        return resolve_checkpoint_file(Path(checkpoint_arg))
    transformer_subdir = getattr(config.backbone, "transformer_subdir", None)
    if transformer_subdir is None:
        return None
    try:
        checkpoint_step_dir = resolve_checkpoint_step_dir_from_transformer_dir(
            Path(str(transformer_subdir))
        )
        return resolve_checkpoint_file(checkpoint_step_dir)
    except (FileNotFoundError, ValueError):
        return None


def _apply_checkpoint_backbone_override(config: Any, *, checkpoint_path: Path) -> None:
    transformer_dir = checkpoint_path.parent / "transformer"
    if transformer_dir.is_dir():
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))
