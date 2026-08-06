from __future__ import annotations

import argparse

from open_wam.runtime.provenance import ProvenanceMode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quantified Open-WAM data/train/eval/rollout-style sanity checks."
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help=(
            "Compatibility control; sanity reports inspect exactly one batch. "
            "Use open-wam-eval for multi-batch metrics."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Allow historical LIBERO policy configs that do not match the current strict fixed-128, "
            "one-frame, proprio-conditioned training/eval paradigm."
        ),
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--provenance-mode",
        choices=tuple(mode.value for mode in ProvenanceMode),
        default=ProvenanceMode.STANDARD.value,
    )
    parser.add_argument("--extension", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        from open_wam.evals.sanity import run_sanity_command
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("open_wam"):
            raise
        missing = error.name or "an optional runtime module"
        raise SystemExit(
            "Sanity runtime dependencies are not installed. Install with "
            "`pip install 'open-wam[train]'` or `uv sync --extra train`. "
            f"Missing module: {missing}."
        ) from error

    run_sanity_command(args)


if __name__ == "__main__":
    main()
