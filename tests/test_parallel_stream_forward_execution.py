from __future__ import annotations

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.forward_execution import (
    build_parallel_first_frame_attention_profile,
    run_parallel_action_conditioned_forward,
    run_parallel_action_conditioned_train,
    run_parallel_exact_dual_stream_forward,
    run_parallel_exact_train,
    run_parallel_first_frame_conditioned_forward,
    run_parallel_first_frame_conditioned_train,
)


def test_reference_runtime_forward_execution_names_alias_canonical_owner() -> None:
    assert (
        reference_runtime._build_fastwam_first_frame_attention_profile
        is build_parallel_first_frame_attention_profile
    )
    assert (
        reference_runtime._run_parallel_action_conditioned_forward
        is run_parallel_action_conditioned_forward
    )
    assert (
        reference_runtime._run_parallel_exact_joint_forward_manual
        is run_parallel_exact_dual_stream_forward
    )
    assert (
        reference_runtime._run_parallel_fastwam_first_frame_forward_manual
        is run_parallel_first_frame_conditioned_forward
    )
    assert (
        reference_runtime.run_parallel_fastwam_first_frame_train
        is run_parallel_first_frame_conditioned_train
    )
    assert (
        reference_runtime.run_parallel_action_conditioned_train
        is run_parallel_action_conditioned_train
    )
    assert reference_runtime.run_parallel_exact_train is run_parallel_exact_train
