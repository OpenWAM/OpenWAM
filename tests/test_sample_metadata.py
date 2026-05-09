from __future__ import annotations

import pytest

from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
)
from open_wam.data.sample_metadata import SampleConstructionMetadata, single_sample_metadata_mapping


def test_sample_construction_metadata_parses_geometry_and_generalist_fields() -> None:
    metadata = {
        "sampled_chunk_size": 4,
        "sampled_window_size": 8,
        "history_frames": 12,
        "frame_shift": 30,
        "loss_frame_start": 12,
        "loss_frame_end": 20,
        GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY: "action_conditioned_video",
        GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY: True,
        GENERALIST_TRAINING_SOURCE_METADATA_KEY: "counterfactual_dynamics",
    }

    parsed = SampleConstructionMetadata.from_batch_metadata((metadata,))

    assert parsed is not None
    assert parsed.sampled_chunk_size_for(16) == 4
    assert parsed.sampled_window_size == 8
    assert parsed.history_frames == 12
    assert parsed.frame_shift == 30
    assert parsed.frame_range_or_default(observed_num_frames=24) == (12, 20)
    assert parsed.generalist.mode_override == "action_conditioned_video"
    assert parsed.generalist.drop_text_conditioning is True
    assert parsed.generalist.source == "counterfactual_dynamics"


def test_sample_construction_metadata_falls_back_to_generic_loss_range() -> None:
    parsed = SampleConstructionMetadata.from_mapping(
        {
            "loss_frame_start": 5,
            "loss_frame_end": 9,
        }
    )

    assert parsed is not None
    assert parsed.optional_frame_range(
        observed_num_frames=12,
        start_key="latent_loss_frame_start",
        end_key="latent_loss_frame_end",
    ) == (5, 9)


def test_sample_construction_metadata_rejects_invalid_ranges() -> None:
    parsed = SampleConstructionMetadata.from_mapping(
        {
            "loss_frame_start": 3,
            "loss_frame_end": 20,
        }
    )

    assert parsed is not None
    with pytest.raises(ValueError, match="observed_num_frames=8"):
        parsed.frame_range_or_default(observed_num_frames=8)


def test_single_sample_metadata_mapping_rejects_multi_sample_batches() -> None:
    assert single_sample_metadata_mapping(({"a": 1}, {"a": 2})) is None
