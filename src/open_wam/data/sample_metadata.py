"""Compatibility imports for sample metadata now owned by ``open_wam.contracts``."""

from open_wam.contracts.sample_metadata import (
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY
    as GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY
    as GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY
    as GENERALIST_TRAINING_SOURCE_METADATA_KEY,
    GeneralistTrainingSampleMetadata as GeneralistTrainingSampleMetadata,
    SampleConstructionMetadata as SampleConstructionMetadata,
    single_sample_metadata_mapping as single_sample_metadata_mapping,
)

__all__ = [
    "GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY",
    "GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY",
    "GENERALIST_TRAINING_SOURCE_METADATA_KEY",
    "GeneralistTrainingSampleMetadata",
    "SampleConstructionMetadata",
    "single_sample_metadata_mapping",
]
