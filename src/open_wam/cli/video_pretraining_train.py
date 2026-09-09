"""Train a frozen pretraining snapshot with the ordinary OpenWAM TrainingRuntime."""

import argparse
import json
from pathlib import Path

from open_wam.data.preparation.snapshot import load_verified_snapshot


def validate_snapshot(snapshot, config):
    root = Path(snapshot).resolve()
    record = load_verified_snapshot(root)
    if any(s["labelled"] != s["clips"] for s in record["sources"].values()):
        raise ValueError(
            "Task-prompt pretraining snapshot contains missing native labels"
        )
    wanted = {
        str((root / name).resolve()) for name in record["manifests_sha256"]
    }
    actual = {
        str(Path(s.manifest_csv).resolve())
        for s in config.data.video_sources
        if s.enabled
    }
    if actual != wanted:
        raise ValueError("Training sources differ from the pinned snapshot")
    if (
        not config.backbone.load_text_conditioning
        and config.backbone.prompt_cache is None
    ):
        raise ValueError(
            "Configure backbone.prompt_cache or enable backbone.load_text_conditioning"
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--new-data-phase",
        action="store_true",
        help="Reset loader cursor after restoring a full checkpoint",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--require-multiview",
        action="store_true",
        help="Verify that this snapshot contains single-view and RGB multi view clips",
    )
    args, remaining = parser.parse_known_args()
    from open_wam.training import (
        TrainingRuntime,
        load_training_cli_config,
        parse_train_cli,
    )
    from open_wam.training.launch import (
        DistributedLaunchContext,
        validate_training_launch,
    )

    overrides = parse_train_cli(remaining)
    config = load_training_cli_config(overrides)
    record = validate_snapshot(args.snapshot, config)
    from open_wam.data.preparation.view_mixture import validate_view_mixture

    view_mixture = validate_view_mixture(
        args.snapshot, record, require_multiview=args.require_multiview
    )
    if args.new_data_phase and not config.trainer.resume_from:
        raise ValueError(
            "--new-data-phase requires --resume-from full_training_state.pt"
        )
    if (
        args.new_data_phase
        and Path(config.trainer.resume_from).name != "full_training_state.pt"
    ):
        raise ValueError(
            "A new data phase requires full model/optimizer/scheduler state"
        )
    if args.check_only:
        print(
            json.dumps(
                dict(
                    status="config_and_snapshot_verified",
                    snapshot_sha256=record["snapshot_sha256"],
                    view_mixture=view_mixture,
                )
            )
        )
        return
    context = DistributedLaunchContext.from_env()
    validate_training_launch(
        config.trainer,
        context,
        expected_world_size=overrides.expected_world_size or overrides.devices,
    )
    runtime = TrainingRuntime.from_config(config, launch_context=context)
    runtime.log_sink.log_event(name="pretraining_view_mixture", payload=view_mixture)
    if args.new_data_phase:
        runtime.train_state.seen_batches = 0
        runtime.train_state.epoch_index = 0
        runtime.train_state.next_batch_index = 0
        runtime.log_sink.log_event(
            name="pretraining_new_data_phase",
            payload={
                "snapshot_sha256": record["snapshot_sha256"],
                "optimizer_step": runtime.train_state.optimizer_step,
                "loader_cursor_reset": True,
            },
        )
    runtime.run()


if __name__ == "__main__":
    main()
