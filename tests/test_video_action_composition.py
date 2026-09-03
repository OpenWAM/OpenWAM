from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import VideoActionProgram
from open_wam.contracts import identify_video_latent_space
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyCompositionCapability,
    PolicyCompositionRngPolicy,
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
    build_video_conditioned_action_context,
    build_video_conditioned_action_request,
    require_generated_video,
    resolve_policy_video_action_consumer_plan,
    resolve_policy_video_producer_plan,
)


def _latent_identity(root: Path, *, marker: str):
    root.mkdir()
    (root / "config.json").write_text('{"model": "test"}', encoding="utf-8")
    (root / "weights.safetensors").write_bytes(marker.encode("ascii"))
    return identify_video_latent_space(
        root,
        encoder_family="test",
        encoding_contract="test",
    )


def _policy(
    *,
    native: frozenset[PolicyOutputModality],
    selective: tuple[PolicyInferenceOutputRequest, ...] = (),
    compositions: tuple[PolicyCompositionCapability, ...] = (),
    history_policy: PolicyRecurrentHistoryPolicy = (
        PolicyRecurrentHistoryPolicy.NEXT_OBSERVATION
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        inference_capabilities=PolicyInferenceCapabilities(
            native_modalities=native,
            selective_requests=selective,
            composition_capabilities=compositions,
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


def test_video_generation_request_validates_optional_attention_window() -> None:
    request = PolicyVideoGenerationRequest(
        frame_count=4,
        attention_window_size=30,
    )
    assert request.attention_window_size == 30
    with pytest.raises(ValueError, match="attention_window_size"):
        PolicyVideoGenerationRequest(frame_count=4, attention_window_size=0)


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


def test_video_action_consumer_resolves_declared_execution_semantics() -> None:
    capability = PolicyCompositionCapability.video_to_action(
        rng_policy=PolicyCompositionRngPolicy.CALLER_STREAM
    )
    policy = _policy(
        native=frozenset(PolicyOutputModality),
        compositions=(capability,),
    )

    plan = resolve_policy_video_action_consumer_plan(policy)  # type: ignore[arg-type]

    assert plan.capability is capability
    assert plan.capability.rng_policy is PolicyCompositionRngPolicy.CALLER_STREAM
    assert plan.resolve_step_seed(rollout_seed=11, step_index=3) is None


def test_isolated_consumer_requires_and_offsets_rollout_seed() -> None:
    plan = resolve_policy_video_action_consumer_plan(  # type: ignore[arg-type]
        _policy(
            native=frozenset(PolicyOutputModality),
            compositions=(PolicyCompositionCapability.video_to_action(),),
        )
    )

    assert plan.resolve_step_seed(rollout_seed=11, step_index=3) == 14
    with pytest.raises(ValueError, match="requires an explicit rollout seed"):
        plan.resolve_step_seed(rollout_seed=None, step_index=3)


def test_caller_stream_composition_bridges_cuda_rng_between_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    consumer_state = torch.tensor([4, 5, 6], dtype=torch.uint8)
    consumed_state = torch.tensor([7, 8, 9], dtype=torch.uint8)
    states = {
        torch.device("cuda:0"): producer_state,
        torch.device("cuda:1"): consumer_state,
    }
    calls: list[tuple[torch.Tensor, torch.device]] = []
    plan = resolve_policy_video_action_consumer_plan(  # type: ignore[arg-type]
        _policy(
            native=frozenset(PolicyOutputModality),
            compositions=(
                PolicyCompositionCapability.video_to_action(
                    rng_policy=PolicyCompositionRngPolicy.CALLER_STREAM
                ),
            ),
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: states[torch.device(device)],
    )

    def _set_rng_state(value, *, device):
        resolved = torch.device(device)
        states[resolved] = value
        calls.append((value, resolved))

    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        _set_rng_state,
    )

    with plan.rng_stream(
        producer_device=torch.device("cuda:0"),
        consumer_device=torch.device("cuda:1"),
    ):
        assert torch.equal(states[torch.device("cuda:1")], producer_state)
        states[torch.device("cuda:1")] = consumed_state

    assert calls == [
        (producer_state, torch.device("cuda:1")),
        (consumed_state, torch.device("cuda:0")),
    ]


def test_isolated_composition_does_not_bridge_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = resolve_policy_video_action_consumer_plan(  # type: ignore[arg-type]
        _policy(
            native=frozenset(PolicyOutputModality),
            compositions=(PolicyCompositionCapability.video_to_action(),),
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: pytest.fail(f"unexpected RNG read from {device}"),
    )

    with plan.rng_stream(
        producer_device=torch.device("cuda:0"),
        consumer_device=torch.device("cuda:1"),
    ):
        pass


def test_video_action_consumer_rejects_undeclared_composition() -> None:
    policy = _policy(
        native=frozenset(PolicyOutputModality),
        selective=(PolicyInferenceOutputRequest.video_only(),),
    )

    with pytest.raises(ValueError, match="does not declare generated-video"):
        resolve_policy_video_action_consumer_plan(policy)  # type: ignore[arg-type]


def test_video_conditioned_action_context_preserves_all_available_inputs() -> None:
    generated = PolicyGeneratedVideo(
        latents=torch.randn(1, 48, 4, 8, 16),
        frame_start=3,
    )
    source = PolicyInferContext(
        state=torch.ones(1, 1, 2),
        previous_action=torch.ones(1, 1, 3),
        dynamics=DynamicsRolloutRequest(),
        output_request=PolicyInferenceOutputRequest.video_only(),
        video_generation=PolicyVideoGenerationRequest(frame_count=2),
        extra={"task_text": ("task",)},
    )

    context = build_video_conditioned_action_context(source, generated)

    assert context.state is source.state
    assert context.previous_action is source.previous_action
    assert context.extra == source.extra
    assert context.dynamics is None
    assert context.video_generation is None
    assert context.output_request is None
    assert context.video_conditioned_action is not None
    assert context.video_conditioned_action.generated_video is generated


def test_video_conditioned_action_request_validates_latent_space_and_origin(
    tmp_path: Path,
) -> None:
    producer_identity = _latent_identity(tmp_path / "producer", marker="producer")
    consumer_identity = _latent_identity(tmp_path / "consumer", marker="consumer")
    request = build_video_conditioned_action_request(
        PolicyGeneratedVideo(
            latents=torch.randn(1, 48, 4, 8, 16),
            frame_start=3,
            latent_space_identity=producer_identity,
        )
    )

    with pytest.raises(ValueError, match="different latent spaces"):
        request.validate_consumer_latent_space(consumer_identity)
    with pytest.raises(RuntimeError, match="did not report its generated frame origin"):
        request.validate_output_frame_start(None)
    with pytest.raises(RuntimeError, match="different temporal origins"):
        request.validate_output_frame_start(4)

    request.validate_consumer_latent_space(producer_identity)
    request.validate_output_frame_start(3)


def test_video_conditioned_action_request_allows_unidentified_in_memory_latents() -> None:
    request = build_video_conditioned_action_request(
        PolicyGeneratedVideo(
            latents=torch.randn(1, 48, 4, 8, 16),
            frame_start=3,
        )
    )

    request.validate_consumer_latent_space(None)


def test_video_conditioned_action_request_requires_temporal_origin() -> None:
    with pytest.raises(ValueError, match="missing its temporal origin"):
        build_video_conditioned_action_request(
            PolicyGeneratedVideo(latents=torch.randn(1, 48, 4, 8, 16))
        )


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
    assert request.generated_video is generated


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
