from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import VideoActionProgram
from open_wam.models.policy_variants import (
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferenceCapabilities,
    PolicyInferenceOutputRequest,
    PolicyInferOutput,
    PolicyInferState,
    PolicyOutputModality,
    PolicyRecurrentHistoryPolicy,
    PolicyVideoGenerationRequest,
)
from open_wam.models.policy_variants.output_semantics import (
    video_action_program_output_modalities,
)
from open_wam.pipelines import (
    build_video_conditioned_action_request,
    require_generated_video,
    resolve_policy_video_producer_plan,
)


def _policy(
    *,
    native: frozenset[PolicyOutputModality],
    selective: tuple[PolicyInferenceOutputRequest, ...] = (),
    history_policy: PolicyRecurrentHistoryPolicy = (
        PolicyRecurrentHistoryPolicy.NEXT_OBSERVATION
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        inference_capabilities=PolicyInferenceCapabilities(
            native_modalities=native,
            selective_requests=selective,
            recurrent_history_policy=history_policy,
        )
    )


@pytest.mark.parametrize(
    ("program", "expected"),
    (
        *(
            (program, frozenset(PolicyOutputModality))
            for program in VideoActionProgram
            if program
            not in {
                VideoActionProgram.FORWARD_DYNAMICS,
                VideoActionProgram.INVERSE_DYNAMICS,
            }
        ),
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            frozenset({PolicyOutputModality.VIDEO}),
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            frozenset({PolicyOutputModality.ACTION}),
        ),
    ),
)
def test_every_video_action_program_declares_output_modalities(
    program: VideoActionProgram,
    expected: frozenset[PolicyOutputModality],
) -> None:
    assert video_action_program_output_modalities(program) == expected


def test_native_video_only_producer_uses_its_normal_output() -> None:
    plan = resolve_policy_video_producer_plan(
        _policy(native=frozenset({PolicyOutputModality.VIDEO}))  # type: ignore[arg-type]
    )

    assert plan.output_request is None
    assert plan.uses_selective_output is False
    assert plan.to_report()["native_modalities"] == ["video"]


def test_multimodal_producer_prefers_selective_video_when_supported() -> None:
    video_only = PolicyInferenceOutputRequest.video_only()
    plan = resolve_policy_video_producer_plan(
        _policy(
            native=frozenset(PolicyOutputModality),
            selective=(video_only,),
        )  # type: ignore[arg-type]
    )

    assert plan.output_request == video_only
    assert plan.uses_selective_output is True


def test_coupled_multimodal_producer_may_publish_video_from_native_output() -> None:
    plan = resolve_policy_video_producer_plan(
        _policy(native=frozenset(PolicyOutputModality))  # type: ignore[arg-type]
    )

    assert plan.output_request is None
    assert plan.native_modalities == frozenset(PolicyOutputModality)


def test_action_only_policy_cannot_be_used_as_video_producer() -> None:
    with pytest.raises(ValueError, match="does not produce every required modality"):
        resolve_policy_video_producer_plan(
            _policy(native=frozenset({PolicyOutputModality.ACTION}))  # type: ignore[arg-type]
        )


def test_video_policy_without_recurrent_history_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="recurrent generated-video history"):
        resolve_policy_video_producer_plan(
            _policy(
                native=frozenset({PolicyOutputModality.VIDEO}),
                history_policy=PolicyRecurrentHistoryPolicy.UNSUPPORTED,
            )  # type: ignore[arg-type]
        )


def test_selective_output_capabilities_reject_redundant_requests() -> None:
    video_only = PolicyInferenceOutputRequest.video_only()
    with pytest.raises(ValueError, match="strict subset"):
        PolicyInferenceCapabilities(
            native_modalities=frozenset({PolicyOutputModality.VIDEO}),
            selective_requests=(video_only,),
        )
    with pytest.raises(ValueError, match="must be unique"):
        PolicyInferenceCapabilities(
            native_modalities=frozenset(PolicyOutputModality),
            selective_requests=(video_only, video_only),
        )


def test_generated_video_handoff_is_future_only_and_preserves_chunk_geometry() -> None:
    generated = PolicyGeneratedVideo(
        latents=torch.randn(1, 48, 4, 8, 16),
        frame_start=9,
    )
    pipeline_output = SimpleNamespace(
        policy_output=PolicyInferOutput(
            policy_features=torch.empty(1, 0, 1),
            next_state=PolicyInferState(),
            generated_video=generated,
        )
    )

    resolved = require_generated_video(pipeline_output)  # type: ignore[arg-type]
    request = build_video_conditioned_action_request(resolved)

    assert resolved is generated
    assert request.clean_video is generated.latents
    assert request.frame_chunk_size == 4


def test_policy_output_rejects_conflicting_typed_temporal_origins() -> None:
    with pytest.raises(ValueError, match="temporal origins differ"):
        PolicyInferOutput(
            policy_features=torch.empty(1, 0, 1),
            next_state=PolicyInferState(),
            generated_video=PolicyGeneratedVideo(
                latents=torch.randn(1, 48, 4, 8, 16),
                frame_start=1,
            ),
            generation_frame_start=2,
        )


def test_missing_typed_generated_video_is_rejected_instead_of_using_debug_aux() -> None:
    pipeline_output = SimpleNamespace(
        policy_output=PolicyInferOutput(
            policy_features=torch.empty(1, 0, 1),
            next_state=PolicyInferState(),
            aux={"predicted_video_latents": torch.randn(1, 48, 4, 8, 16)},
        )
    )

    with pytest.raises(RuntimeError, match="future-only PolicyGeneratedVideo"):
        require_generated_video(pipeline_output)  # type: ignore[arg-type]


def test_generated_video_must_honor_requested_chunk_geometry() -> None:
    pipeline_output = SimpleNamespace(
        policy_output=PolicyInferOutput(
            policy_features=torch.empty(1, 0, 1),
            next_state=PolicyInferState(),
            generated_video=PolicyGeneratedVideo(
                latents=torch.randn(1, 48, 3, 8, 16)
            ),
        )
    )

    with pytest.raises(RuntimeError, match="requested_frames=4, generated_frames=3"):
        require_generated_video(
            pipeline_output,  # type: ignore[arg-type]
            request=PolicyVideoGenerationRequest(frame_count=4),
        )


def test_generated_video_must_publish_origin_for_typed_composition() -> None:
    pipeline_output = SimpleNamespace(
        policy_output=PolicyInferOutput(
            policy_features=torch.empty(1, 0, 1),
            next_state=PolicyInferState(),
            generated_video=PolicyGeneratedVideo(
                latents=torch.randn(1, 48, 4, 8, 16)
            ),
        )
    )

    with pytest.raises(RuntimeError, match="temporal origin"):
        require_generated_video(
            pipeline_output,  # type: ignore[arg-type]
            request=PolicyVideoGenerationRequest(frame_count=4),
        )


def test_generated_video_extension_preserves_policy_output_positional_aux() -> None:
    legacy_aux = {"extension": "legacy-positional-constructor"}
    output = PolicyInferOutput(
        torch.empty(1, 0, 1),
        PolicyInferState(),
        None,
        legacy_aux,
    )

    assert output.aux is legacy_aux
    assert output.generated_video is None


def test_infer_context_extensions_preserve_positional_extra() -> None:
    legacy_extra = {"extension": "legacy-positional-constructor"}
    context = PolicyInferContext(None, None, None, legacy_extra)

    assert context.extra is legacy_extra
    assert context.output_request is None
    assert context.video_generation is None
