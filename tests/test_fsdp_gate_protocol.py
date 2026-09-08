"""CPU-only regression coverage for the two-worker GPU gate's log protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_fsdp_gate_protocol_subject", Path(__file__).with_name("test_token_budget_fsdp.py")
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


pytestmark = pytest.mark.unit


def _record(rank=0, **extra):
    return {"rank": rank, "phase": "forward_end", "monotonic": 123.25, **extra}


def _framed(record):
    return gate._PHASE_PREFIX + json.dumps(record)


def test_merged_rank_records_are_both_decoded():
    records = [_record(1), _record(0)]
    assert gate._parse_phase_line("".join(map(_framed, records)) + "\n") == records


def test_phase_records_can_be_interspersed_with_ordinary_logs():
    records = [_record(0), _record(1, phase="backward_begin")]
    line = "[rank0] compile finished " + _framed(records[0])
    line += " [rank1] another ordinary message " + _framed(records[1]) + " trailing log\n"
    assert gate._parse_phase_line(line) == records
    assert gate._parse_phase_line("ordinary log without a phase record\n") == []


def test_protocol_marker_inside_json_message_is_not_a_new_record():
    embedded = gate._PHASE_PREFIX + '{"rank":99,"phase":"not a real event"}'
    records = [
        _record(0, phase="error", message=f'quoted "text", braces }} and {embedded}\n'),
        _record(1),
    ]
    assert gate._parse_phase_line("".join(map(_framed, records))) == records


@pytest.mark.parametrize(
    "payload",
    ["", "not JSON", '{"rank":0', '{"rank":0,"phase":}', '"plain string"',
     '[]', '{"phase":"forward_end"}', '{"rank":0,"phase":null}',
     '{"rank":0,"phase":""}'],
)
def test_malformed_phase_is_not_silently_ignored(payload):
    with pytest.raises(ValueError, match="Malformed FSDP phase"):
        gate._parse_phase_line(gate._PHASE_PREFIX + payload)


def test_valid_record_followed_by_malformed_record_still_fails():
    line = _framed(_record(0)) + " ordinary output " + gate._PHASE_PREFIX + '{"rank":1'
    with pytest.raises(ValueError, match="Malformed FSDP phase JSON"):
        gate._parse_phase_line(line)


def test_json_whitespace_after_marker_is_supported():
    record = _record()
    assert gate._parse_phase_line(gate._PHASE_PREFIX + " \t" + json.dumps(record)) == [record]


def test_phase_uses_exactly_one_write_including_newline(monkeypatch):
    writes = []

    def write(fd, payload):
        writes.append((fd, payload))
        return len(payload)

    monkeypatch.setattr(gate.os, "write", write)
    record = _record()
    gate._write_phase(record)
    assert len(writes) == 1
    fd, payload = writes[0]
    assert fd == 1 and isinstance(payload, bytes)
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    assert len(payload) <= gate._MAX_PHASE_BYTES == 4096
    assert gate._parse_phase_line(payload.decode("ascii")) == [record]


def test_record_at_exact_pipe_limit_is_not_split(monkeypatch):
    writes = []
    record = _record(message="")
    overhead = len(gate._encode_phase(record))
    record["message"] = "x" * (gate._MAX_PHASE_BYTES - overhead)

    def write(fd, payload):
        writes.append((fd, payload))
        return len(payload)

    monkeypatch.setattr(gate.os, "write", write)
    gate._write_phase(record)
    assert len(writes) == 1 and len(writes[0][1]) == gate._MAX_PHASE_BYTES
    assert gate._parse_phase_line(writes[0][1].decode("ascii")) == [record]


@pytest.mark.parametrize(
    "message", ["x" * 10000, "\u20ac\u20ac\U0001f642" * 3000, "\ud800" * 3000]
)
def test_oversized_error_message_is_valid_bounded_json(monkeypatch, message):
    writes = []

    def write(fd, payload):
        writes.append((fd, payload))
        return len(payload)

    monkeypatch.setattr(gate.os, "write", write)
    record = _record(phase="error", error_type="RuntimeError", message=message)
    original = dict(record)
    gate._write_phase(record)
    assert len(writes) == 1 and len(writes[0][1]) <= gate._MAX_PHASE_BYTES
    decoded, = gate._parse_phase_line(writes[0][1].decode("ascii"))
    assert decoded["message_truncated"] is True
    assert decoded["message"].endswith(" [truncated]")
    assert decoded["phase"] == "error" and decoded["error_type"] == "RuntimeError"
    assert decoded["rank"] == record["rank"]
    assert record == original


@pytest.mark.parametrize("extra", [{"shape": [1] * 5000}, {"program": "x" * 5000, "message": "x"}])
def test_oversized_non_message_metadata_fails_before_any_write(monkeypatch, extra):
    writes = []
    monkeypatch.setattr(gate.os, "write", lambda *args: writes.append(args))
    with pytest.raises(ValueError, match="atomic log limit"):
        gate._write_phase(_record(**extra))
    assert writes == []


@pytest.mark.parametrize("short_count", [0, 1])
def test_short_os_write_is_an_error_without_a_second_chunk(monkeypatch, short_count):
    writes = []

    def write(fd, payload):
        writes.append((fd, payload))
        return short_count

    monkeypatch.setattr(gate.os, "write", write)
    with pytest.raises(RuntimeError, match="Short atomic FSDP phase write"):
        gate._write_phase(_record())
    assert len(writes) == 1


def test_os_write_error_propagates_without_retry(monkeypatch):
    writes = []

    def write(fd, payload):
        writes.append((fd, payload))
        raise BrokenPipeError("closed test pipe")

    monkeypatch.setattr(gate.os, "write", write)
    with pytest.raises(BrokenPipeError, match="closed test pipe"):
        gate._write_phase(_record())
    assert len(writes) == 1
