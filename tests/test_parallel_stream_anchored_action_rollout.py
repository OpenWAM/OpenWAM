from __future__ import annotations

from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.anchored_action_rollout import (
    run_parallel_current_frame_action_chunk_inference_rollout,
    run_parallel_fastwam_first_frame_inference_rollout,
)


def test_reference_runtime_anchored_rollouts_alias_canonical_owner() -> None:
    assert (
        reference_runtime.run_parallel_current_frame_action_chunk_inference_rollout
        is run_parallel_current_frame_action_chunk_inference_rollout
    )
    assert (
        reference_runtime.run_parallel_fastwam_first_frame_inference_rollout
        is run_parallel_fastwam_first_frame_inference_rollout
    )
