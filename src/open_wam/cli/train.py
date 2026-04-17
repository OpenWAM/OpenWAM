from __future__ import annotations

import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Open-WAM experiment.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--cfg", "--config", dest="config", type=str)
    config_group.add_argument("--config-name", dest="config_name", type=str)
    parser.add_argument("--save-root", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--latent-root", type=str)
    parser.add_argument("--transformer-subdir", type=str)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-mode", type=str)
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    # Let argparse handle --help without importing Lightning/Torch.
    build_arg_parser().parse_known_args(argv)
    try:
        from open_wam.training.train import main as training_main
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Training dependencies are not installed. Install with `pip install 'open-wam[train]'` "
            "or `uv sync --extra train`."
        ) from exc
    if argv is not None:
        old_argv = sys.argv
        sys.argv = [old_argv[0], *argv]
        try:
            training_main()
        finally:
            sys.argv = old_argv
        return
    training_main()


__all__ = ["build_arg_parser", "main"]
