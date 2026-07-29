from __future__ import annotations

import pytest

from open_wam.configs import (
    SampleOrderMode,
    coerce_bool,
    coerce_enum,
    coerce_enum_tuple,
    coerce_optional_enum,
    coerce_strict_chunk_size,
)


def test_config_coercion_helpers_are_public_and_typed() -> None:
    assert coerce_enum(SampleOrderMode, "replacement") is SampleOrderMode.REPLACEMENT
    assert coerce_enum(SampleOrderMode, SampleOrderMode.EPOCH_ORDER) is SampleOrderMode.EPOCH_ORDER
    assert coerce_optional_enum(SampleOrderMode, None) is None
    assert coerce_enum_tuple(
        SampleOrderMode,
        ["replacement", SampleOrderMode.EPOCH_ORDER],
    ) == (SampleOrderMode.REPLACEMENT, SampleOrderMode.EPOCH_ORDER)


@pytest.mark.parametrize(("value", "expected"), (("yes", True), ("OFF", False), (True, True)))
def test_coerce_bool_accepts_supported_yaml_and_cli_values(value, expected: bool) -> None:
    assert coerce_bool(value, field_name="feature.enabled") is expected


def test_coerce_bool_reports_the_field_for_invalid_values() -> None:
    with pytest.raises(ValueError, match="feature.enabled"):
        coerce_bool(2, field_name="feature.enabled")


@pytest.mark.parametrize(("value", "expected"), ((4, 4), ("4", 4), (4.0, 4)))
def test_coerce_strict_chunk_size_accepts_integral_values(value, expected: int) -> None:
    assert coerce_strict_chunk_size("training.chunk_size", value) == expected


@pytest.mark.parametrize("value", (True, 1.5, "many"))
def test_coerce_strict_chunk_size_rejects_non_integral_values(value) -> None:
    with pytest.raises(ValueError, match="training.chunk_size"):
        coerce_strict_chunk_size("training.chunk_size", value)
