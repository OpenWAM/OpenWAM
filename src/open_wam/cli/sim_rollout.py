from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Open-WAM policy in a benchmark simulator through the shared "
            "closed-loop realtime adapter. Supports RoboTwin and CALVIN when the "
            "external simulator packages are installed locally."
        )
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
        help="Commit only the first predicted action per replan, or blockingly execute the full predicted chunk.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/sim_realtime")
    parser.add_argument("--suffix", type=str, default="rollout")
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--zero-policy",
        action="store_true",
        help="Step the simulator with zero model-space actions. Useful for env wiring dry runs.",
    )
    parser.add_argument("--robotwin-root", type=str, default=None)
    parser.add_argument("--robotwin-task-name", type=str, default=None)
    parser.add_argument("--robotwin-task-config", type=str, default=None)
    parser.add_argument("--robotwin-action-type", type=str, default="ee")
    parser.add_argument(
        "--robotwin-expert-precheck",
        action="store_true",
        help="Run RoboTwin's expert play_once/check_success path and generate the episode instruction before reset.",
    )
    parser.add_argument("--robotwin-instruction-type", type=str, default="seen")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--calvin-root", type=str, default=None)
    parser.add_argument("--calvin-dataset-root", type=str, default=None)
    parser.add_argument("--calvin-task-text", type=str, default=None)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--extension", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        from open_wam.evals.sim_rollout import run_simulator_rollout_command
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("open_wam"):
            raise
        missing = error.name or "an optional runtime module"
        raise SystemExit(
            "Simulator dependencies are not installed. Install with "
            "`pip install 'open-wam[sim]'` or `uv sync --extra sim`. "
            f"Missing module: {missing}."
        ) from error

    run_simulator_rollout_command(args)


if __name__ == "__main__":
    main()
