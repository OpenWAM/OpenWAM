"""Path construction and trace serialization for LIBERO rollout artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from open_wam.evals.libero_rollout_artifact_contracts import (
    LiberoRealtimeArtifactIdentity,
    LiberoRolloutArtifactIdentity,
)


def build_libero_realtime_output_stem(
    *,
    root: Path,
    identity: LiberoRealtimeArtifactIdentity,
) -> Path:
    """Resolve one sanitized realtime artifact stem without a suffix."""

    safe_prompt = _safe_path_token(identity.prompt)
    safe_suffix = _safe_path_token(identity.suffix)
    return (
        root
        / identity.benchmark
        / f"{identity.task_id}_{safe_prompt}"
        / f"{identity.episode_idx}_{safe_suffix}"
    )


def build_libero_rollout_output_path(
    *,
    root: Path,
    identity: LiberoRolloutArtifactIdentity,
) -> Path:
    """Resolve the maintained per-task, per-episode artifact path."""

    safe_prompt = identity.prompt.replace(" ", "_")
    return (
        root
        / identity.benchmark
        / f"{identity.task_id}_{safe_prompt}"
        / (f"{identity.episode_idx}_{identity.success}_{identity.suffix}.mp4")
    )


def _safe_path_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "run"


def _write_jsonl_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")


def _write_action_trace(
    path: Path,
    actions: Sequence[np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for action_index, action in enumerate(actions):
            handle.write(
                json.dumps(
                    {
                        "action_index": int(action_index),
                        "action": np.asarray(
                            action,
                            dtype=np.float32,
                        ).tolist(),
                    }
                )
                + "\n"
            )
