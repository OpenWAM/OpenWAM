from __future__ import annotations

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.inference_artifacts import (
    LingbotParallelInferArtifacts,
    ParallelInferArtifacts,
)
from open_wam.models.policy_variants.parallel_stream.staged_rollout import (
    run_parallel_staged_inference_rollout,
)


def test_generic_inference_artifact_preserves_legacy_type_identity() -> None:
    assert ParallelInferArtifacts is LingbotParallelInferArtifacts
    assert reference_runtime.ParallelInferArtifacts is ParallelInferArtifacts
    assert (
        reference_runtime.LingbotParallelInferArtifacts
        is LingbotParallelInferArtifacts
    )


def test_reference_runtime_staged_rollout_name_aliases_canonical_owner() -> None:
    assert (
        reference_runtime.run_parallel_exact_inference_rollout
        is run_parallel_staged_inference_rollout
    )
