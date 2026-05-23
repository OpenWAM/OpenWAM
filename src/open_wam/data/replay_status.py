from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


DEFAULT_REPLAY_STATUS_RELATIVE_PATH = Path("meta") / "replay_status.jsonl"
SUCCESS_REPLAY_STATUS = "success"
FAILURE_REPLAY_STATUSES = frozenset({"failure", "error"})
REPLAY_STATUS_POLICIES = frozenset({"include_all", "successful_only", "failure_only"})


@dataclass(frozen=True)
class ReplayStatusRecord:
    """Replay label for one dataset episode."""

    dataset_episode_index: int
    replay_status: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ReplayStatusFilterReport:
    """Summary of one replay-status episode filter operation."""

    source_path: str | None
    policy: str
    require_replay_status: bool
    total_episodes: int
    labeled_episodes: int
    kept_episodes: int
    filtered_episodes: int
    status_counts: dict[str, int]
    missing_status_file: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayStatusTrainValSplit:
    """Episode split after optional replay-status train/val filtering."""

    train_episodes: list[int]
    val_episodes: list[int]
    train_report: ReplayStatusFilterReport
    val_report: ReplayStatusFilterReport | None
    used_explicit_val_policy: bool


def resolve_replay_status_path(
    dataset_root: str | Path | None,
    replay_status_path: str | Path | None,
) -> Path | None:
    """Resolve the canonical replay-status path for one dataset root."""

    root = Path(dataset_root).expanduser() if dataset_root is not None else None
    if replay_status_path is None:
        return None if root is None else root / DEFAULT_REPLAY_STATUS_RELATIVE_PATH

    path = Path(replay_status_path).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path


def load_replay_status_records(
    dataset_root: str | Path | None,
    *,
    replay_status_path: str | Path | None = None,
    require: bool = False,
) -> tuple[dict[int, ReplayStatusRecord], Path | None]:
    """Load `<dataset_root>/meta/replay_status.jsonl` into an episode lookup.

    Missing files are allowed when `require=False`; this lets older datasets run
    unchanged while new labeled datasets are filtered automatically.
    """

    path = resolve_replay_status_path(dataset_root, replay_status_path)
    if path is None:
        if require:
            raise FileNotFoundError(
                "Replay-status filtering was required, but no replay_status_path was configured "
                "and no dataset root was available."
            )
        return {}, None
    if not path.is_file():
        if require:
            raise FileNotFoundError(f"Replay-status filtering was required, but the status file is missing: {path}")
        return {}, path

    records: dict[int, ReplayStatusRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in replay-status file {path}:{line_number}") from exc
            try:
                episode_index = _episode_index_from_payload(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid dataset episode index in replay-status file {path}:{line_number}: {exc}"
                ) from exc
            try:
                status = normalize_replay_status(_status_from_payload(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid replay status in replay-status file {path}:{line_number}: {exc}") from exc
            if episode_index in records:
                raise ValueError(
                    f"Duplicate replay status for dataset_episode_index={episode_index} in {path}:{line_number}"
                )
            records[episode_index] = ReplayStatusRecord(
                dataset_episode_index=episode_index,
                replay_status=status,
                raw=payload,
            )
    if require and not records:
        raise ValueError(f"Replay-status filtering was required, but the status file contains no records: {path}")
    return records, path


def filter_episode_indices_by_replay_status(
    episode_indices: Iterable[int],
    *,
    replay_status_records: Mapping[int, ReplayStatusRecord],
    policy: Any,
    require_labeled: bool,
    source_path: str | Path | None = None,
) -> tuple[list[int], ReplayStatusFilterReport]:
    """Apply a replay-status policy to one episode-index sequence."""

    normalized_policy = normalize_replay_status_policy(policy)
    selected = [int(index) for index in episode_indices]
    if normalized_policy == "include_all":
        return selected, build_replay_status_filter_report(
            selected,
            kept=selected,
            replay_status_records=replay_status_records,
            policy=normalized_policy,
            require_labeled=require_labeled,
            source_path=source_path,
        )

    if not replay_status_records:
        if require_labeled:
            raise ValueError("Replay-status filtering was required, but no replay-status records were loaded.")
        return selected, build_replay_status_filter_report(
            selected,
            kept=selected,
            replay_status_records=replay_status_records,
            policy=normalized_policy,
            require_labeled=require_labeled,
            source_path=source_path,
            missing_status_file=_replay_status_file_is_missing(source_path),
        )

    missing = [index for index in selected if index not in replay_status_records]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(
            "Replay-status file does not label every selected episode. "
            f"Missing dataset_episode_index values: {preview}{suffix}"
        )

    kept = [
        index
        for index in selected
        if replay_status_matches_policy(replay_status_records[index].replay_status, normalized_policy)
    ]
    return kept, build_replay_status_filter_report(
        selected,
        kept=kept,
        replay_status_records=replay_status_records,
        policy=normalized_policy,
        require_labeled=require_labeled,
        source_path=source_path,
    )


def split_episode_indices_by_replay_status(
    episode_indices: Iterable[int],
    *,
    replay_status_records: Mapping[int, ReplayStatusRecord],
    replay_status_path: str | Path | None,
    replay_status_policy: Any,
    require_replay_status: bool,
    val_replay_status_policy: Any | None,
    val_require_replay_status: bool | None,
    train_fraction: float,
    split_seed: int,
    max_train_episodes: int | None = None,
    max_val_episodes: int | None = None,
) -> ReplayStatusTrainValSplit:
    """Split episodes, optionally validating on replay-labeled unused trajectories.

    When `val_replay_status_policy` is unset, this preserves the historical
    behavior: apply the train replay-status policy first, then randomly split
    the remaining episodes by `train_fraction`.

    When `val_replay_status_policy` is set and labels are present or required,
    train and validation are selected independently from the original episode
    set and validation episodes used by training are removed. This lets configs
    train on successful replay rows while validating on unused failure/error
    rows without adding runtime-specific launch logic.
    """

    all_episodes = [int(index) for index in episode_indices]
    train_policy = normalize_replay_status_policy(replay_status_policy)
    train_require_labeled = bool(replay_status_records) or bool(require_replay_status)
    val_require_labeled = bool(replay_status_records) or bool(
        require_replay_status if val_require_replay_status is None else val_require_replay_status
    )
    use_explicit_val_policy = val_replay_status_policy is not None and (
        bool(replay_status_records) or val_require_labeled
    )

    train_candidates, train_report = filter_episode_indices_by_replay_status(
        all_episodes,
        replay_status_records=replay_status_records,
        policy=train_policy,
        require_labeled=train_require_labeled,
        source_path=replay_status_path,
    )
    rng = random.Random(split_seed)
    rng.shuffle(train_candidates)
    train_count = int(len(train_candidates) * train_fraction)
    train_count = min(max(train_count, 1), len(train_candidates)) if train_candidates else 0
    train_episodes = train_candidates[:train_count]
    if max_train_episodes is not None:
        train_episodes = train_episodes[:max_train_episodes]

    if not use_explicit_val_policy:
        val_episodes = train_candidates[train_count:]
        if max_val_episodes is not None:
            val_episodes = val_episodes[:max_val_episodes]
        if not val_episodes and train_episodes:
            val_episodes = train_episodes[:1]
        return ReplayStatusTrainValSplit(
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            train_report=train_report,
            val_report=None,
            used_explicit_val_policy=False,
        )

    val_policy = normalize_replay_status_policy(val_replay_status_policy)
    val_candidates, val_report = filter_episode_indices_by_replay_status(
        all_episodes,
        replay_status_records=replay_status_records,
        policy=val_policy,
        require_labeled=val_require_labeled,
        source_path=replay_status_path,
    )
    val_candidates = sorted(val_candidates)
    random.Random(split_seed + 1).shuffle(val_candidates)
    train_episode_set = set(train_episodes)
    val_episodes = [episode for episode in val_candidates if episode not in train_episode_set]
    if max_val_episodes is not None:
        val_episodes = val_episodes[:max_val_episodes]
    if not val_episodes:
        raise ValueError(
            "`data.val_replay_status_policy` selected no validation episodes after removing training episodes. "
            "Use a non-overlapping validation policy, fix the replay-status labels, lower `train_fraction`, or unset "
            "`val_replay_status_policy` to keep legacy train-fraction validation."
        )
    return ReplayStatusTrainValSplit(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        train_report=train_report,
        val_report=val_report,
        used_explicit_val_policy=True,
    )


def build_replay_status_filter_report(
    episode_indices: Iterable[int],
    *,
    kept: Iterable[int],
    replay_status_records: Mapping[int, ReplayStatusRecord],
    policy: str,
    require_labeled: bool,
    source_path: str | Path | None,
    missing_status_file: bool = False,
) -> ReplayStatusFilterReport:
    selected = [int(index) for index in episode_indices]
    kept_list = [int(index) for index in kept]
    status_counts = Counter(
        replay_status_records[index].replay_status
        for index in selected
        if index in replay_status_records
    )
    return ReplayStatusFilterReport(
        source_path=str(source_path) if source_path is not None else None,
        policy=policy,
        require_replay_status=bool(require_labeled),
        total_episodes=len(selected),
        labeled_episodes=sum(1 for index in selected if index in replay_status_records),
        kept_episodes=len(kept_list),
        filtered_episodes=len(selected) - len(kept_list),
        status_counts=dict(sorted(status_counts.items())),
        missing_status_file=missing_status_file,
    )


def replay_status_matches_policy(status: str, policy: str) -> bool:
    if policy == "include_all":
        return True
    if policy == "successful_only":
        return status == SUCCESS_REPLAY_STATUS
    if policy == "failure_only":
        return status in FAILURE_REPLAY_STATUSES
    raise ValueError(f"Unsupported replay status policy: {policy!r}")


def normalize_replay_status_policy(policy: Any) -> str:
    value = getattr(policy, "value", policy)
    normalized = str(value).strip().lower()
    if normalized not in REPLAY_STATUS_POLICIES:
        raise ValueError(
            f"Unsupported replay status policy {value!r}; expected one of: "
            f"{', '.join(sorted(REPLAY_STATUS_POLICIES))}."
        )
    return normalized


def normalize_replay_status(value: Any) -> str:
    if isinstance(value, bool):
        return SUCCESS_REPLAY_STATUS if value else "failure"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return SUCCESS_REPLAY_STATUS if bool(value) else "failure"
    normalized = str(value).strip().lower()
    if normalized in {"success", "successful", "pass", "passed", "true"}:
        return SUCCESS_REPLAY_STATUS
    if normalized in {"failure", "failed", "fail", "false"}:
        return "failure"
    if normalized in {"error", "errored", "crash", "crashed"}:
        return "error"
    raise ValueError(
        f"Unsupported replay status {value!r}; expected success/failure/error or a boolean success value."
    )


def _replay_status_file_is_missing(source_path: str | Path | None) -> bool:
    return source_path is None or not Path(source_path).is_file()


def _episode_index_from_payload(payload: Mapping[str, Any]) -> int:
    for key in ("dataset_episode_index", "episode_index"):
        if key in payload:
            return int(payload[key])
    raise ValueError("row is missing `dataset_episode_index` or `episode_index`.")


def _status_from_payload(payload: Mapping[str, Any]) -> Any:
    for key in ("replay_status", "status"):
        if key in payload:
            return payload[key]
    for key in ("replay_success", "success"):
        if key in payload:
            return payload[key]
    raise ValueError("row is missing `replay_status`, `status`, `replay_success`, or `success`.")
