from __future__ import annotations

import argparse

from .train_arguments import build_train_arg_parser


def build_arg_parser() -> argparse.ArgumentParser:
    return build_train_arg_parser()


def main(argv: list[str] | None = None) -> None:
    # Let argparse handle --help without importing the Torch training stack.
    build_arg_parser().parse_known_args(argv)
    try:
        from open_wam.training.train import main as training_main
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Training dependencies are not installed. Install with `pip install 'openwam[train]'` "
            "or `uv sync --extra train`."
        ) from exc
    training_main(argv)


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    main()
