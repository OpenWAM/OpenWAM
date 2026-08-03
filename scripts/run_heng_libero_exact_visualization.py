"""Compatibility entry point for the renamed LingBot reference runner."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


def main() -> None:
    warnings.warn(
        "scripts/run_heng_libero_exact_visualization.py is deprecated; use "
        "scripts/run_lingbot_reference_visualization.py instead.",
        FutureWarning,
        stacklevel=2,
    )
    runpy.run_path(
        str(Path(__file__).with_name("run_lingbot_reference_visualization.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
