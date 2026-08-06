"""Stable structured-result contracts."""

from open_wam.runtime.results import (
    OPEN_WAM_RESULT_SCHEMA_V1,
    build_result_envelope,
    write_result_json,
)
from open_wam.runtime.provenance import (
    OPEN_WAM_PROVENANCE_SCHEMA_V1,
    ProvenanceMode,
    collect_runtime_provenance,
)

__all__ = [
    "OPEN_WAM_PROVENANCE_SCHEMA_V1",
    "OPEN_WAM_RESULT_SCHEMA_V1",
    "ProvenanceMode",
    "build_result_envelope",
    "collect_runtime_provenance",
    "write_result_json",
]
