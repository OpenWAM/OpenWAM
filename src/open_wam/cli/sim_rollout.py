from __future__ import annotations

import argparse
import sys

from ._legacy_script import run_legacy_script


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a shared closed-loop Open-WAM simulator rollout for RoboTwin or CALVIN."
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--benchmark", choices=("robotwin", "calvin"), required=True)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--target-action-hz", type=float, default=None)
    parser.add_argument(
        "--action-commit-mode",
        choices=("first_action", "full_chunk"),
        default="first_action",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/sim_realtime")
    parser.add_argument("--suffix", type=str, default="rollout")
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--robotwin-root", type=str, default=None)
    parser.add_argument("--robotwin-task-name", type=str, default=None)
    parser.add_argument("--robotwin-task-config", type=str, default=None)
    parser.add_argument("--robotwin-action-type", type=str, default="ee")
    parser.add_argument("--robotwin-expert-precheck", action="store_true")
    parser.add_argument("--robotwin-instruction-type", type=str, default="seen")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--calvin-root", type=str, default=None)
    parser.add_argument("--calvin-dataset-root", type=str, default=None)
    parser.add_argument("--calvin-task-text", type=str, default=None)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--extension", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    resolved_argv = sys.argv[1:] if argv is None else argv
    build_arg_parser().parse_args(resolved_argv)
    if argv is None:
        run_legacy_script("run_sim_realtime_sandbox.py")
        return
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        run_legacy_script("run_sim_realtime_sandbox.py")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
