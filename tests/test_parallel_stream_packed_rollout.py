from __future__ import annotations

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.packed_rollout import (
    _run_parallel_packed_inference_rollout_impl,
    run_parallel_packed_inference_rollout,
)


def test_reference_runtime_packed_rollout_names_alias_canonical_owner() -> None:
    assert (
        reference_runtime._run_parallel_action_conditioned_inference_rollout_impl
        is _run_parallel_packed_inference_rollout_impl
    )
    assert (
        reference_runtime.run_parallel_action_conditioned_inference_rollout
        is run_parallel_packed_inference_rollout
    )
