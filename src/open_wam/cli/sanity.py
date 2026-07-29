from __future__ import annotations

import argparse
import sys

from ._legacy_script import run_legacy_script


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quantified Open-WAM data/train/eval/rollout-style sanity checks."
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extension", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    resolved_argv = sys.argv[1:] if argv is None else argv
    build_arg_parser().parse_args(resolved_argv)
    if argv is None:
        run_legacy_script("run_benchmark_pipeline_sanity.py")
        return
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        run_legacy_script("run_benchmark_pipeline_sanity.py")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
