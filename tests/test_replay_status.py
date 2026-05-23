from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_wam.data.replay_status import (
    filter_episode_indices_by_replay_status,
    load_replay_status_records,
    split_episode_indices_by_replay_status,
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


def test_split_episode_indices_can_validate_on_unused_failures(tmp_path: Path) -> None:
    records, path = load_replay_status_records(
        None,
        replay_status_path=_fixture_replay_status_path(tmp_path),
        require=True,
    )

    split = split_episode_indices_by_replay_status(
        [0, 1, 2],
        replay_status_records=records,
        replay_status_path=path,
        replay_status_policy="successful_only",
        require_replay_status=True,
        val_replay_status_policy="failure_only",
        val_require_replay_status=None,
        train_fraction=1.0,
        split_seed=0,
    )

    assert set(split.train_episodes) == {0, 2}
    assert split.val_episodes == [1]
    assert split.used_explicit_val_policy is True
    assert split.val_report is not None
    assert split.val_report.kept_episodes == 1


def test_explicit_val_replay_status_policy_fails_when_no_validation_rows(tmp_path: Path) -> None:
    path = tmp_path / "replay_status.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_episode_index": 0, "replay_status": "success"},
            {"dataset_episode_index": 1, "replay_status": "success"},
        ],
    )
    records, _ = load_replay_status_records(None, replay_status_path=path, require=True)

    with pytest.raises(ValueError, match="selected no validation episodes"):
        split_episode_indices_by_replay_status(
            [0, 1],
            replay_status_records=records,
            replay_status_path=path,
            replay_status_policy="successful_only",
            require_replay_status=False,
            val_replay_status_policy="failure_only",
            val_require_replay_status=False,
            train_fraction=1.0,
            split_seed=0,
        )


def test_explicit_val_replay_status_policy_shuffles_before_cap(tmp_path: Path) -> None:
    records, path = load_replay_status_records(
        None,
        replay_status_path=_fixture_many_failure_replay_status_path(tmp_path),
        require=True,
    )

    split_a = split_episode_indices_by_replay_status(
        [0, 1, 2, 3, 4],
        replay_status_records=records,
        replay_status_path=path,
        replay_status_policy="successful_only",
        require_replay_status=True,
        val_replay_status_policy="failure_only",
        val_require_replay_status=True,
        train_fraction=1.0,
        split_seed=7,
        max_val_episodes=2,
    )
    split_b = split_episode_indices_by_replay_status(
        [4, 3, 2, 1, 0],
        replay_status_records=records,
        replay_status_path=path,
        replay_status_policy="successful_only",
        require_replay_status=True,
        val_replay_status_policy="failure_only",
        val_require_replay_status=True,
        train_fraction=1.0,
        split_seed=7,
        max_val_episodes=2,
    )

    assert split_a.val_episodes == split_b.val_episodes
    assert split_a.val_episodes != [1, 2]
    assert len(split_a.val_episodes) == 2


def test_split_episode_indices_preserves_legacy_fraction_when_no_val_policy(tmp_path: Path) -> None:
    records, path = load_replay_status_records(
        None,
        replay_status_path=_fixture_replay_status_path(tmp_path),
        require=True,
    )

    split = split_episode_indices_by_replay_status(
        [0, 1, 2],
        replay_status_records=records,
        replay_status_path=path,
        replay_status_policy="successful_only",
        require_replay_status=True,
        val_replay_status_policy=None,
        val_require_replay_status=None,
        train_fraction=0.5,
        split_seed=0,
    )

    assert len(split.train_episodes) == 1
    assert len(split.val_episodes) == 1
    assert set(split.train_episodes + split.val_episodes) == {0, 2}
    assert split.used_explicit_val_policy is False


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


def _fixture_many_failure_replay_status_path(tmp_path: Path) -> Path:
    path = tmp_path / "many_failure_replay_status.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_episode_index": 0, "replay_status": "success"},
            {"dataset_episode_index": 1, "replay_status": "failure"},
            {"dataset_episode_index": 2, "replay_status": "failure"},
            {"dataset_episode_index": 3, "replay_status": "failure"},
            {"dataset_episode_index": 4, "replay_status": "failure"},
        ],
    )
    return path
