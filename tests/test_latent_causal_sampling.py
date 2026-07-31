from __future__ import annotations

import random

import pytest

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    DataSplit,
    LatentTemporalLayout,
    SampleConstructionConfig,
    WindowSamplingMode,
)
from open_wam.data import (
    LatentCausalPrefixSuffixCandidate,
    LatentCausalPrefixSuffixWindowPlanner,
)


RAW_FRAME_IDS = tuple(range(37))


def _planner(split: DataSplit) -> LatentCausalPrefixSuffixWindowPlanner:
    return LatentCausalPrefixSuffixWindowPlanner(
        sample_config=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=8,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(
                    observed_frames=1,
                    future_frames=3,
                ),
                CausalPrefixSuffixBucketConfig(
                    observed_frames=2,
                    future_frames=4,
                ),
            ),
        ),
        latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
        split=split,
        split_seed=17,
    )


def test_causal_planner_enumerates_bucket_major_candidates() -> None:
    candidates = _planner(DataSplit.TRAIN).build_candidates(
        raw_frame_ids=RAW_FRAME_IDS,
        source_latent_frames=10,
        row_count=37,
    )

    assert candidates == tuple(
        LatentCausalPrefixSuffixCandidate(latent_start=start, bucket_index=bucket)
        for bucket, starts in ((0, range(7)), (1, range(5)))
        for start in starts
    )


def test_causal_validation_plan_is_split_seeded_and_rng_neutral() -> None:
    planner = _planner(DataSplit.VAL)
    rng_state = random.getstate()
    try:
        random.seed(999)
        before = random.getstate()
        plan = planner.plan(
            raw_frame_ids=RAW_FRAME_IDS,
            source_latent_frames=10,
            row_count=37,
            sample_index=0,
        )
        after = random.getstate()
    finally:
        random.setstate(rng_state)

    assert plan is not None
    assert before == after
    assert plan.latent_start == 1
    assert plan.latent_end == 7
    assert plan.raw_start_position == 1
    assert plan.raw_end_position == 25
    assert plan.sample_start_frame == 1
    assert plan.sample_end_frame == 25
    assert plan.observed_frame_ids == (4, 8, 12, 16, 20, 24)
    assert plan.observed_prefix_frames == 2
    assert plan.future_suffix_frames == 4
    assert plan.valid_video_frames == 6
    assert plan.padded_video_frames == 8


def test_causal_train_selection_preserves_global_rng_advance() -> None:
    planner = _planner(DataSplit.TRAIN)
    candidates = planner.build_candidates(
        raw_frame_ids=RAW_FRAME_IDS,
        source_latent_frames=10,
        row_count=37,
    )
    rng_state = random.getstate()
    try:
        random.seed(23)
        draw_seed = random.randrange(1 << 30)
        expected_next_random = random.random()
        expected = candidates[
            random.Random(draw_seed + 4).randrange(len(candidates))
        ]

        random.seed(23)
        selected = planner.select_candidate(candidates, sample_index=4)
        actual_next_random = random.random()
    finally:
        random.setstate(rng_state)

    assert selected == expected
    assert actual_next_random == expected_next_random


def test_causal_planner_reports_invalid_configuration_and_geometry() -> None:
    empty = LatentCausalPrefixSuffixWindowPlanner(
        sample_config=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=8,
        ),
        latent_temporal_layout=LatentTemporalLayout.WAN_CAUSAL_STRIDE4,
        split=DataSplit.TRAIN,
        split_seed=17,
    )
    with pytest.raises(
        ValueError,
        match="requires non-empty `sample_construction.causal_prefix_suffix_buckets`",
    ):
        empty.build_candidates(
            raw_frame_ids=RAW_FRAME_IDS,
            source_latent_frames=10,
            row_count=37,
        )

    assert (
        _planner(DataSplit.TRAIN).plan(
            raw_frame_ids=RAW_FRAME_IDS,
            source_latent_frames=3,
            row_count=37,
            sample_index=0,
        )
        is None
    )
    with pytest.raises(ValueError, match="requires at least one eligible candidate"):
        _planner(DataSplit.TRAIN).select_candidate((), sample_index=0)
