from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import torch

from open_wam.simulators import (
    SimulatorBackend,
    run_closed_loop_sim_rollout,
    summarize_sim_rollout,
)
from open_wam.simulators.builtins import (
    normalize_builtin_simulator_options,
    register_builtin_simulator_adapters,
)
from open_wam.runtime import build_result_envelope
from open_wam.runtime.provenance import collect_runtime_provenance
from open_wam.runtime.results import write_result_json
from open_wam.simulators.registry import (
    SimulatorFactoryContext,
    build_simulator_adapter,
)
from open_wam.runtime.checkpoints import (
    CheckpointCompatibilityPolicy,
    load_pipeline_checkpoint,
    resolve_checkpoint_file,
    resolve_checkpoint_step_dir_from_transformer_dir,
)
from open_wam.configs import (
    ExperimentConfig,
    load_experiment_config,
    load_local_path_registry,
    resolve_experiment_config_reference,
    serialize_experiment_config,
)
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

    register_builtin_simulator_adapters()
    load_extension_modules(args.extension)
    seed_everywhere(args.seed)
    config_path = resolve_experiment_config_reference(args.config).resolve()
    config = load_experiment_config(config_path)
    device = _resolve_device(args.device)

    checkpoint_path = _resolve_checkpoint_for_config(config=config, checkpoint_arg=args.checkpoint)
    if checkpoint_path is not None:
        config = _with_checkpoint_backbone_override(
            config,
            checkpoint_path=checkpoint_path,
        )

    adapter = _build_adapter(args)
    checkpoint_report = None
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
            checkpoint_report = load_pipeline_checkpoint(
                pipeline,
                checkpoint_path,
                compatibility=(
                    CheckpointCompatibilityPolicy.ALLOW_PARTIAL
                    if args.allow_partial_checkpoint
                    else CheckpointCompatibilityPolicy.STRICT
                ),
            )
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
            "checkpoint_compatibility": (
                CheckpointCompatibilityPolicy.ALLOW_PARTIAL.value
                if args.allow_partial_checkpoint
                else CheckpointCompatibilityPolicy.STRICT.value
            ),
            "checkpoint_missing_keys": (
                []
                if checkpoint_report is None
                else list(checkpoint_report.missing_keys)
            ),
            "checkpoint_unexpected_keys": (
                []
                if checkpoint_report is None
                else list(checkpoint_report.unexpected_keys)
            ),
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
        provenance=collect_runtime_provenance(
            config_path=config_path,
            resolved_config=serialize_experiment_config(config),
            checkpoint_path=checkpoint_path,
            dataset_root=config.data.local_root,
            mode=args.provenance_mode,
        ),
        extra=legacy_summary,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    write_result_json(summary_path, summary)
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


def _build_adapter(args: argparse.Namespace) -> SimulatorBackend:
    options = normalize_builtin_simulator_options(
        benchmark=args.benchmark,
        arguments=vars(args),
        explicit_options=_parse_simulator_options(args.sim_option),
    )
    context = SimulatorFactoryContext(
        benchmark=args.benchmark,
        options=options,
        local_paths=load_local_path_registry(),
    )
    try:
        return build_simulator_adapter(context)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _parse_simulator_options(values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw_value in values:
        key, separator, value = raw_value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise SystemExit(
                f"Invalid --sim-option {raw_value!r}; expected KEY=VALUE."
            )
        if key in options:
            raise SystemExit(f"Duplicate --sim-option key {key!r}.")
        options[key] = value
    return options


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


def _with_checkpoint_backbone_override(
    config: ExperimentConfig,
    *,
    checkpoint_path: Path,
) -> ExperimentConfig:
    transformer_dir = checkpoint_path.parent / "transformer"
    if not transformer_dir.is_dir():
        return config
    return replace(
        config,
        backbone=replace(
            config.backbone,
            transformer_subdir=str(transformer_dir.resolve()),
        ),
    )
