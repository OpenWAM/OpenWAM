from __future__ import annotations

import argparse
import sys

from open_wam.runtime.provenance import ProvenanceMode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Open-WAM experiment.")
    parser.add_argument("--cfg", "--config", dest="config", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Permit missing or unexpected model keys for migration diagnostics.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--provenance-mode",
        choices=tuple(mode.value for mode in ProvenanceMode),
        default=ProvenanceMode.STANDARD.value,
    )
    parser.add_argument("--extension", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    # Let argparse handle --help without importing Torch-backed eval code.
    build_arg_parser().parse_args(argv)
    try:
        from open_wam.evals.evaluate import main as evaluate_main
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Evaluation dependencies are not installed. Install with `pip install 'open-wam[eval]'` "
            "or `uv sync --extra eval`."
        ) from exc
    if argv is not None:
        old_argv = sys.argv
        sys.argv = [old_argv[0], *argv]
        try:
            evaluate_main()
        finally:
            sys.argv = old_argv
        return
    evaluate_main()


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    main()
