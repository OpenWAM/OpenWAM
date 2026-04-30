from __future__ import annotations

import argparse
from pathlib import Path

from .config import CheckpointSpec, RolloutSuiteConfig, load_episode_manifest, load_suite_config, parse_int_selection, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vanilla LingBot-VA released LIBERO-LONG baseline on LIBERO-10.")
    parser.add_argument("--suite", type=str, default=None, help="Optional YAML suite file.")
    parser.add_argument("--source-repo", type=str, default=None, help="Path to a LingBot-VA source checkout.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=None,
        help="Released LingBot-VA full model root with vae/text_encoder/tokenizer/transformer.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="lingbot_va_posttrain_libero_long", help="Name used in outputs.")
    parser.add_argument("--hf-repo-id", type=str, default="robbyant/lingbot-va-posttrain-libero-long")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-ids", type=str, default="0:10")
    parser.add_argument("--episode-indices", type=str, default="0")
    parser.add_argument(
        "--episode-manifest",
        type=str,
        default=None,
        help="Optional YAML manifest with explicit benchmark/task/episode rows.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional one-time per-episode seed. Omit for upstream-like RNG behavior.")
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--output-dir", type=str, default="outputs/lingbot_va_baseline")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-render-video", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in results.jsonl.")
    offload = parser.add_mutually_exclusive_group()
    offload.add_argument("--enable-offload", action="store_true")
    offload.add_argument("--disable-offload", action="store_true")
    args = parser.parse_args()

    if args.suite:
        config = load_suite_config(args.suite)
    else:
        model_root = resolve_path(args.model_root)
        if model_root is None:
            raise ValueError("--model-root is required when --suite is not provided.")
        enable_offload = None
        if args.enable_offload:
            enable_offload = True
        elif args.disable_offload:
            enable_offload = False
        config = RolloutSuiteConfig(
            checkpoints=(
                CheckpointSpec(
                    name=args.checkpoint_name,
                    model_root=model_root,
                    source_repo=resolve_path(args.source_repo),
                    enable_offload=enable_offload,
                    hf_repo_id=args.hf_repo_id,
                    hf_revision=args.hf_revision,
                ),
            ),
            benchmark=args.benchmark,
            task_ids=tuple(parse_int_selection(args.task_ids)),
            episode_indices=tuple(parse_int_selection(args.episode_indices)),
            episode_specs=load_episode_manifest(args.episode_manifest, base_dir=Path.cwd())
            if args.episode_manifest
            else (),
            seed=args.seed,
            max_timestep=args.max_timestep,
            max_chunks=args.max_chunks,
            video_fps=args.video_fps,
            output_dir=Path(args.output_dir),
            cuda_device=args.cuda_device,
            render_video=not args.no_render_video,
            continue_on_error=args.continue_on_error,
            resume=args.resume,
        )

    from .experiment import run_suite

    result = run_suite(config)
    print(f"summary: {result.summary_path}")
    print(f"markdown: {result.markdown_path}")


if __name__ == "__main__":
    main()
