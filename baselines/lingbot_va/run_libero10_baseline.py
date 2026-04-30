from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    CheckpointSpec,
    RolloutSuiteConfig,
    load_suite_config,
    parse_int_selection,
    resolve_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run read-only LingBot-VA baselines on LIBERO-10.")
    parser.add_argument("--suite", type=str, default=None, help="Optional YAML suite file.")
    parser.add_argument("--source-repo", type=str, default=None, help="Path to a LingBot-VA source checkout.")
    parser.add_argument("--pretrained-root", type=str, default=None, help="LingBot-VA base root with vae/text/tokenizer.")
    parser.add_argument("--transformer-dir", type=str, default=None, help="Trained transformer directory.")
    parser.add_argument("--checkpoint-name", type=str, default="lingbot_va", help="Name used in outputs.")
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-ids", type=str, default="0:10")
    parser.add_argument("--episode-indices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/lingbot_va_baseline")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-render-video", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument("--enable-offload", action="store_true")
    offload.add_argument("--disable-offload", action="store_true")
    args = parser.parse_args()

    if args.suite:
        config = load_suite_config(args.suite)
    else:
        pretrained_root = resolve_path(args.pretrained_root)
        if pretrained_root is None:
            raise ValueError("--pretrained-root is required when --suite is not provided.")
        enable_offload = None
        if args.enable_offload:
            enable_offload = True
        elif args.disable_offload:
            enable_offload = False
        config = RolloutSuiteConfig(
            checkpoints=(
                CheckpointSpec(
                    name=args.checkpoint_name,
                    pretrained_root=pretrained_root,
                    transformer_dir=resolve_path(args.transformer_dir),
                    source_repo=resolve_path(args.source_repo),
                    enable_offload=enable_offload,
                ),
            ),
            benchmark=args.benchmark,
            task_ids=tuple(parse_int_selection(args.task_ids)),
            episode_indices=tuple(parse_int_selection(args.episode_indices)),
            seed=args.seed,
            max_timestep=args.max_timestep,
            max_chunks=args.max_chunks,
            video_fps=args.video_fps,
            output_dir=Path(args.output_dir),
            cuda_device=args.cuda_device,
            render_video=not args.no_render_video,
            continue_on_error=args.continue_on_error,
        )

    from .experiment import run_suite

    result = run_suite(config)
    print(f"summary: {result.summary_path}")
    print(f"markdown: {result.markdown_path}")


if __name__ == "__main__":
    main()
