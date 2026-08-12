from __future__ import annotations

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.training_artifacts import (
    LingbotParallelTrainArtifacts,
    ParallelTrainArtifacts,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
)


def test_generic_training_artifact_name_preserves_legacy_type_identity() -> None:
    assert ParallelTrainArtifacts is LingbotParallelTrainArtifacts


def test_reference_runtime_training_artifact_names_alias_canonical_owner() -> None:
    assert (
        reference_runtime.LingbotParallelTrainArtifacts
        is LingbotParallelTrainArtifacts
    )
    assert (
        reference_runtime.prepare_parallel_action_conditioned_train_artifacts
        is prepare_parallel_action_conditioned_train_artifacts
    )
    assert (
        reference_runtime.prepare_parallel_exact_train_artifacts
        is prepare_parallel_exact_train_artifacts
    )
    assert (
        reference_runtime.prepare_parallel_prefix_condition_exact_train_artifacts
        is prepare_parallel_prefix_condition_exact_train_artifacts
    )
