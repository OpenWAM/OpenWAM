from __future__ import annotations

import argparse
from pprint import pprint

from open_wam.utils import load_experiment_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and print one typed Open-WAM experiment config.")
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    pprint(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
