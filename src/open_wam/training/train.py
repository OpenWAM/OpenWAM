from __future__ import annotations

import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.training import TrainingRuntime, load_training_cli_config, parse_train_cli


def main() -> None:
    if os.getenv("OPEN_WAM_DETECT_ANOMALY", "0") == "1":
        import torch

        torch.autograd.set_detect_anomaly(True)
    try:
        overrides = parse_train_cli()
        config = load_training_cli_config(overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime = TrainingRuntime.from_config(config)
    runtime.run()


if __name__ == "__main__":
    main()
