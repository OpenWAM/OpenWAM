"""Temporal laws are derived once, independent of policy architecture."""

import pytest

from open_wam.runtime.rollout_temporal import ResolvedRolloutTemporalContract


@pytest.mark.parametrize("density", [1, 2, 4])
@pytest.mark.parametrize("startup", [1, 3])
def test_temporal_round_trip_and_partial_execution(density, startup):
    temporal = ResolvedRolloutTemporalContract(startup, density, density, 4, 30)
    for index in range(19):
        assert temporal.frame_for_control(index) == startup + index // density
        assert temporal.control_for_frame(temporal.frame_for_control(index)) == (
            temporal.complete_control_end(index)
        )
        assert temporal.control_offset(index) == index % density
    span = temporal.observed_span(2 * density, 5 * density)
    assert (span.start_frame, span.frame_count) == (startup + 2, 3)
    assert temporal.raw_window_frames(5) == 4 * density + 1
    assert temporal.prediction_controls == 4 * density
    temporal.validate_observations(
        span, observations=3 * density + 1, actions=3 * density, latents=4
    )
    with pytest.raises(ValueError, match="observed interval"):
        temporal.validate_observations(
            span, observations=3 * density, actions=3 * density, latents=4
        )
    if density > 1:
        with pytest.raises(ValueError, match="complete"):
            temporal.observed_span(0, density + 1)
    with pytest.raises(ValueError):
        temporal.control_for_frame(startup - 1)


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_temporal_contract_uses_the_assembled_frontend(architecture):
    from tests.test_simulator_policy_lifecycle import _pipeline

    pipeline, _ = _pipeline(architecture, density=4)
    temporal = ResolvedRolloutTemporalContract.from_pipeline(pipeline)
    assert temporal.raw_frames_per_frame == 4
    assert temporal.controls_per_frame == 4
    assert temporal.prediction_controls == 8
    assert temporal.attention_window_size == 4
