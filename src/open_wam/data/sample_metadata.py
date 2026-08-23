"""Compatibility imports for sample metadata now owned by ``open_wam.contracts``."""

from open_wam.contracts.sample_metadata import (
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY
    as DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY
    as DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY
    as DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
    DynamicsRoutingSampleMetadata as DynamicsRoutingSampleMetadata,
    SampleConstructionMetadata as SampleConstructionMetadata,
    single_sample_metadata_mapping as single_sample_metadata_mapping,
)

__all__ = [
    "DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY",
    "DYNAMICS_ROUTING_MODE_METADATA_KEY",
    "DYNAMICS_ROUTING_SOURCE_METADATA_KEY",
    "DynamicsRoutingSampleMetadata",
    "SampleConstructionMetadata",
    "single_sample_metadata_mapping",
]
