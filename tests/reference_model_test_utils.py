from __future__ import annotations

from pathlib import Path

import pytest


def reference_model_path_or_skip() -> str:
    candidate = Path(__file__).resolve().parents[1] / "previous_works" / "lingbot-va" / "wan_va" / "modules" / "model.py"
    if not candidate.exists():
        pytest.skip("LingBot reference model source file is unavailable in this checkout.")
    return str(candidate)
