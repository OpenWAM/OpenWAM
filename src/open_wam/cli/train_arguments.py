from __future__ import annotations

import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
    """Build the argument contract shared by every training entrypoint."""

    parser = argparse.ArgumentParser(description="Train an Open-WAM experiment.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--cfg", "--config", dest="config", type=str)
    config_group.add_argument("--config-name", dest="config_name", type=str)
    parser.add_argument(
        "--save-root",
        type=str,
        help="Full run output directory. This mirrors LingBot's `save_root` semantics.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory for checkpoints written by this training run.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-root",
        type=str,
        help=(
            "Deprecated ambiguous option; always errors. Use "
            "--initialize-weights-from or --resume-from."
        ),
    )
    checkpoint_group.add_argument("--initialize-weights-from", type=str)
    checkpoint_group.add_argument(
        "--resume-from",
        type=str,
        help=(
            "Full training-state checkpoint used to resume model, optimizer, "
            "scheduler, strategy, and step state."
        ),
    )
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--latent-root", type=str)
    parser.add_argument(
        "--runtime-backbone-path",
        dest="runtime_backbone_artifact_path",
        type=str,
        help="Detached runtime-backbone artifact used to initialize model weights.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        help=(
            "Compatibility launch expectation recorded in trainer.devices. This does "
            "not spawn workers; use --expected-world-size for new commands."
        ),
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        help="Require the external launcher to provide exactly this WORLD_SIZE.",
    )
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-mode", type=str)
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help=(
            "Load `package.module[:hook]` before parsing and constructing the "
            "experiment."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Repeatable `section.field=value` override.",
    )
    return parser


__all__ = ["build_train_arg_parser"]
