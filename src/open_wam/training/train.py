from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import DatasetArtifactPreflightError
from open_wam.training import TrainingRuntime, load_training_cli_config, parse_train_cli
from open_wam.training.launch import DistributedLaunchContext, validate_training_launch


def main(argv: list[str] | None = None) -> None:
    if os.getenv("OPEN_WAM_DETECT_ANOMALY", "0") == "1":
        import torch

        torch.autograd.set_detect_anomaly(True)
    try:
        overrides = parse_train_cli(argv)
        config = load_training_cli_config(overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        launch_context = DistributedLaunchContext.from_env()
        expected_world_size = (
            overrides.expected_world_size
            if overrides.expected_world_size is not None
            else overrides.devices
        )
        validate_training_launch(
            config.trainer,
            launch_context,
            expected_world_size=expected_world_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        runtime = TrainingRuntime.from_config(config, launch_context=launch_context)
    except DatasetArtifactPreflightError as exc:
        raise SystemExit(str(exc)) from exc
    runtime.run()


if __name__ == "__main__":
    main()
