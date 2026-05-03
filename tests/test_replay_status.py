from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_wam.data.replay_status import (
    filter_episode_indices_by_replay_status,
    load_replay_status_records,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_replay_status_records_uses_dataset_meta_default(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "meta" / "replay_status.jsonl",
        [
            {"dataset_episode_index": 7, "replay_status": "success"},
            {"episode_index": 8, "success": False},
        ],
    )

    records, path = load_replay_status_records(tmp_path)

    assert path == tmp_path / "meta" / "replay_status.jsonl"
    assert records[7].replay_status == "success"
    assert records[8].replay_status == "failure"


def test_filter_episode_indices_by_replay_status_keeps_successes(tmp_path: Path) -> None:
    records, _ = load_replay_status_records(
        None,
        replay_status_path=_fixture_replay_status_path(tmp_path),
        require=True,
    )

    kept, report = filter_episode_indices_by_replay_status(
        [0, 1, 2],
        replay_status_records=records,
        policy="successful_only",
        require_labeled=True,
    )

    assert kept == [0, 2]
    assert report.filtered_episodes == 1
    assert report.status_counts == {"failure": 1, "success": 2}


def test_filter_episode_indices_by_replay_status_requires_complete_labels(tmp_path: Path) -> None:
    records, _ = load_replay_status_records(
        None,
        replay_status_path=_fixture_replay_status_path(tmp_path),
        require=True,
    )

    with pytest.raises(ValueError, match="does not label every selected episode"):
        filter_episode_indices_by_replay_status(
            [0, 1, 2, 9],
            replay_status_records=records,
            policy="successful_only",
            require_labeled=True,
        )


def test_missing_replay_status_file_is_allowed_when_not_required(tmp_path: Path) -> None:
    records, path = load_replay_status_records(tmp_path, require=False)

    kept, report = filter_episode_indices_by_replay_status(
        [0, 1],
        replay_status_records=records,
        policy="successful_only",
        require_labeled=False,
        source_path=path,
    )

    assert kept == [0, 1]
    assert report.missing_status_file is True


def test_empty_replay_status_file_is_reported_as_present(tmp_path: Path) -> None:
    status_path = tmp_path / "meta" / "replay_status.jsonl"
    status_path.parent.mkdir(parents=True)
    status_path.write_text("", encoding="utf-8")
    records, path = load_replay_status_records(tmp_path, require=False)

    kept, report = filter_episode_indices_by_replay_status(
        [0, 1],
        replay_status_records=records,
        policy="successful_only",
        require_labeled=False,
        source_path=path,
    )

    assert path == status_path
    assert kept == [0, 1]
    assert report.missing_status_file is False
    assert report.labeled_episodes == 0


def test_malformed_replay_status_rows_include_file_and_line_context(tmp_path: Path) -> None:
    path = tmp_path / "replay_status.jsonl"
    _write_jsonl(path, [{"dataset_episode_index": "not-an-int", "replay_status": "success"}])

    with pytest.raises(ValueError, match=r"Invalid dataset episode index.*replay_status\.jsonl:1"):
        load_replay_status_records(None, replay_status_path=path, require=True)


def _fixture_replay_status_path(tmp_path: Path) -> Path:
    path = tmp_path / "replay_status.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_episode_index": 0, "replay_status": "success"},
            {"dataset_episode_index": 1, "replay_status": "failure"},
            {"dataset_episode_index": 2, "replay_status": "success"},
        ],
    )
    return path
