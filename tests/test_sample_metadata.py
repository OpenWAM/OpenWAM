from __future__ import annotations

import pytest

from open_wam.contracts import (
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
    ConditionalDynamicsSequenceLayout,
    SampleConstructionMetadata,
    single_sample_metadata_mapping,
)
from open_wam.data.sample_metadata import (
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY
    as LegacyGeneralistTrainingDropTextMetadataKey,
    DYNAMICS_ROUTING_MODE_METADATA_KEY
    as LegacyGeneralistTrainingModeOverrideMetadataKey,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY
    as LegacyGeneralistTrainingSourceMetadataKey,
    SampleConstructionMetadata as LegacySampleConstructionMetadata,
    single_sample_metadata_mapping as legacy_single_sample_metadata_mapping,
)


def test_sample_metadata_legacy_imports_preserve_identity() -> None:
    assert LegacySampleConstructionMetadata is SampleConstructionMetadata
    assert legacy_single_sample_metadata_mapping is single_sample_metadata_mapping
    assert (
        LegacyGeneralistTrainingDropTextMetadataKey
        is DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY
    )
    assert (
        LegacyGeneralistTrainingModeOverrideMetadataKey
        is DYNAMICS_ROUTING_MODE_METADATA_KEY
    )
    assert (
        LegacyGeneralistTrainingSourceMetadataKey
        is DYNAMICS_ROUTING_SOURCE_METADATA_KEY
    )


def test_conditional_dynamics_sequence_layout_is_canonical() -> None:
    layout = ConditionalDynamicsSequenceLayout()

    assert layout.loss_frame_range(observed_num_frames=4) == (1, 4)
    with pytest.raises(TypeError):
        ConditionalDynamicsSequenceLayout(history_frames=2)


def test_sample_construction_metadata_parses_geometry_and_generalist_fields() -> None:
    metadata = {
        "sampled_chunk_size": 4,
        "sampled_window_size": 8,
        "history_frames": 12,
        "context_prefix_frames_in_sample": 1,
        "frame_shift": 30,
        "loss_frame_start": 12,
        "loss_frame_end": 20,
        DYNAMICS_ROUTING_MODE_METADATA_KEY: "action_conditioned_video",
        DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY: True,
        DYNAMICS_ROUTING_SOURCE_METADATA_KEY: "counterfactual_dynamics",
    }

    parsed = SampleConstructionMetadata.from_batch_metadata((metadata,))

    assert parsed is not None
    assert parsed.sampled_chunk_size_for(16) == 4
    assert parsed.sampled_window_size == 8
    assert parsed.history_frames == 12
    assert parsed.context_prefix_frames_in_sample == 1
    assert parsed.frame_shift == 30
    assert parsed.frame_range_or_default(observed_num_frames=24) == (12, 20)
    assert parsed.dynamics_routing.mode_override == "action_conditioned_video"
    assert parsed.dynamics_routing.drop_text_conditioning is True
    assert parsed.dynamics_routing.source == "counterfactual_dynamics"


def test_sample_construction_metadata_preserves_absent_drop_text_as_unspecified() -> None:
    parsed = SampleConstructionMetadata.from_mapping(
        {
            DYNAMICS_ROUTING_MODE_METADATA_KEY: "action_conditioned_video",
        }
    )

    assert parsed is not None
    assert parsed.dynamics_routing.drop_text_conditioning is None


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


def test_sample_construction_metadata_rejects_negative_context_prefix() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        SampleConstructionMetadata.from_mapping(
            {"context_prefix_frames_in_sample": -1}
        )


def test_single_sample_metadata_mapping_rejects_multi_sample_batches() -> None:
    assert single_sample_metadata_mapping(({"a": 1}, {"a": 2})) is None
