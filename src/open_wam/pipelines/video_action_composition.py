"""Generic contracts for composing generated video with action inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from open_wam.contracts import require_compatible_video_latent_spaces
from open_wam.models.training_provenance import PolicyTrainingProvenance
from open_wam.configs.enums import DynamicsObjective
from open_wam.models.policy_variants import (
    PolicyCompositionCapability,
    PolicyCompositionRngPolicy,
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferenceOutputRequest,
    PolicyOutputModality,
    PolicyRecurrentHistoryPolicy,
    PolicyVariant,
    PolicyVideoConditionedActionRequest,
    PolicyVideoGenerationRequest,
)

from .variant_pipeline import VariantPipelineInferOutput
from .rollout import (
    VariantRolloutRunner,
    VariantRolloutSession,
    VariantRolloutStepOutput,
)
from open_wam.models.visual_tower import VisualStageOutputs
from open_wam.utils.seeding import (
    snapshot_rng_state,
    restore_rng_state,
    seed_everywhere,
)


@dataclass(frozen=True)
class PolicyVideoProducerPlan:
    """How one policy should expose video for a downstream stage."""

    output_request: PolicyInferenceOutputRequest | None
    native_modalities: frozenset[PolicyOutputModality]
    recurrent_history_policy: PolicyRecurrentHistoryPolicy
    required_training_objective: DynamicsObjective | None = None

    @property
    def uses_selective_output(self) -> bool:
        return self.output_request is not None

    def to_report(self) -> dict[str, object]:
        return {
            "required_training_objective": (
                None
                if self.required_training_objective is None
                else self.required_training_objective.value
            ),
            "native_modalities": sorted(
                modality.value for modality in self.native_modalities
            ),
            "requested_modalities": (
                None
                if self.output_request is None
                else sorted(
                    modality.value for modality in self.output_request.modalities
                )
            ),
            "uses_selective_output": self.uses_selective_output,
            "recurrent_history_policy": self.recurrent_history_policy.value,
        }


@dataclass(frozen=True)
class PolicyVideoActionConsumerPlan:
    """A policy that accepts transferred video and emits actions."""

    capability: PolicyCompositionCapability
    recurrent_history_policy: PolicyRecurrentHistoryPolicy

    def infer(
        self,
        runner: VariantRolloutRunner,
        *,
        session: VariantRolloutSession,
        context: PolicyInferContext,
        visual_outputs: VisualStageOutputs,
        generated_video: PolicyGeneratedVideo,
        producer_device: torch.device | str,
        rollout_seed: int | None,
        step_index: int,
    ) -> VariantRolloutStepOutput:
        """Consume an independent video product with policy-owned context and RNG.

        The caller supplies all available observations, proprio and text. The
        consumer's objective alone decides which conditioning it can attend to.
        """
        observed = visual_outputs.frontend.video_latents
        predicted = generated_video.latents
        if observed.ndim != 5 or (
            observed.shape[0],
            observed.shape[1],
            *observed.shape[-2:],
        ) != (
            predicted.shape[0],
            predicted.shape[1],
            *predicted.shape[-2:],
        ):
            raise ValueError(
                "Generated video does not match the consumer's latent geometry."
            )
        require_compatible_video_latent_spaces(
            generated_video.latent_space_identity,
            visual_outputs.frontend.latent_space_identity,
        )
        context = build_video_conditioned_action_context(
            context,
            replace(
                generated_video,
                latents=predicted.detach().to(observed),
            ),
        )
        seed = self.resolve_step_seed(rollout_seed=rollout_seed, step_index=step_index)
        snapshot = snapshot_rng_state() if seed is not None else None
        try:
            if seed is not None:
                seed_everywhere(seed)
            with self.rng_stream(
                producer_device=producer_device, consumer_device=observed.device
            ):
                return runner.infer_prepared_step(
                    session=session,
                    context=context,
                    visual_outputs=visual_outputs,
                )
        finally:
            if snapshot is not None:
                restore_rng_state(snapshot)

    def resolve_step_seed(
        self,
        *,
        rollout_seed: int | None,
        step_index: int,
    ) -> int | None:
        """Resolve the consumer seed required by its declared RNG policy."""

        if self.capability.rng_policy is PolicyCompositionRngPolicy.CALLER_STREAM:
            return None
        if rollout_seed is None:
            raise ValueError(
                "The action consumer requires an explicit rollout seed so each "
                "composed step receives an isolated deterministic random stream."
            )
        return int(rollout_seed) + int(step_index)

    @contextmanager
    def rng_stream(
        self,
        *,
        producer_device: torch.device | str,
        consumer_device: torch.device | str,
    ) -> Iterator[None]:
        """Preserve one logical random stream across a composed policy call."""

        if self.capability.rng_policy is not PolicyCompositionRngPolicy.CALLER_STREAM:
            yield
            return
        source = torch.device(producer_device)
        target = torch.device(consumer_device)
        if source.type != target.type:
            raise ValueError(
                "Chained composition RNG requires producer and consumer devices "
                f"of the same type; got producer={source}, consumer={target}."
            )
        if source.type != "cuda":
            yield
            return
        source = _resolve_cuda_device(source)
        target = _resolve_cuda_device(target)
        if source == target:
            yield
            return
        _copy_cuda_rng_state(source=source, target=target)
        try:
            yield
        finally:
            _copy_cuda_rng_state(source=target, target=source)

    def to_report(self) -> dict[str, object]:
        return {
            "required_training_objective": (
                None
                if self.capability.required_training_objective is None
                else self.capability.required_training_objective.value
            ),
            "input_modalities": sorted(
                modality.value for modality in self.capability.input_modalities
            ),
            "output_modalities": sorted(
                modality.value for modality in self.capability.output_modalities
            ),
            "rng_policy": self.capability.rng_policy.value,
            "recurrent_history_policy": self.recurrent_history_policy.value,
        }


def resolve_policy_video_action_consumer_plan(
    policy: PolicyVariant,
    *,
    training: PolicyTrainingProvenance = PolicyTrainingProvenance(frozenset()),
) -> PolicyVideoActionConsumerPlan:
    """Require a policy to support independent generated-video consumption."""

    capability = PolicyCompositionCapability.video_to_action()
    capabilities = policy.inference_capabilities
    declared_capability = capabilities.composition_for(capability)
    if declared_capability is None:
        raise ValueError(
            f"{type(policy).__name__} does not declare generated-video to action "
            "composition support."
        )
    training.require(declared_capability.required_training_objective)
    if (
        capabilities.recurrent_history_policy
        is PolicyRecurrentHistoryPolicy.UNSUPPORTED
    ):
        raise ValueError(
            f"{type(policy).__name__} does not declare safe recurrent history "
            "semantics for video-conditioned action composition."
        )
    return PolicyVideoActionConsumerPlan(
        capability=declared_capability,
        recurrent_history_policy=capabilities.recurrent_history_policy,
    )


def resolve_policy_video_producer_plan(
    policy: PolicyVariant,
    *,
    training: PolicyTrainingProvenance = PolicyTrainingProvenance(frozenset()),
) -> PolicyVideoProducerPlan:
    """Resolve video production from declared capabilities, never policy names."""

    capabilities = policy.inference_capabilities
    capabilities.require_future_inputs(frozenset())
    training.require(capabilities.required_training_objective)
    request = capabilities.request_for(frozenset({PolicyOutputModality.VIDEO}))
    history_policy = capabilities.recurrent_history_policy
    if history_policy is PolicyRecurrentHistoryPolicy.UNSUPPORTED:
        raise ValueError(
            f"{type(policy).__name__} emits video but does not declare safe "
            "recurrent generated-video history semantics. The policy must either "
            "consume the next real observation directly or implement explicit "
            "observed-history reconciliation before online composition."
        )
    return PolicyVideoProducerPlan(
        output_request=request,
        native_modalities=capabilities.native_modalities,
        recurrent_history_policy=history_policy,
        required_training_objective=capabilities.required_training_objective,
    )


def require_generated_video(
    output: VariantPipelineInferOutput,
    *,
    request: PolicyVideoGenerationRequest | None = None,
) -> PolicyGeneratedVideo:
    """Return the canonical future-only video product from a pipeline step."""

    generated_video = output.policy_output.generated_video
    if generated_video is None:
        raise RuntimeError(
            "The selected video producer did not publish a future-only "
            "PolicyGeneratedVideo artifact."
        )
    if request is not None and int(generated_video.latents.shape[2]) != int(
        request.frame_count
    ):
        raise RuntimeError(
            "The video producer did not honor the requested future chunk geometry: "
            f"requested_frames={int(request.frame_count)}, "
            f"generated_frames={int(generated_video.latents.shape[2])}."
        )
    if request is not None and generated_video.frame_start is None:
        raise RuntimeError(
            "The video producer did not publish the generated chunk's temporal "
            "origin, so composed action inference cannot verify frame alignment."
        )
    return generated_video


def build_video_conditioned_action_request(
    generated_video: PolicyGeneratedVideo,
) -> PolicyVideoConditionedActionRequest:
    """Build a consumer-neutral request from a generated video chunk."""

    return PolicyVideoConditionedActionRequest(
        generated_video=generated_video,
    )


def build_video_conditioned_action_context(
    context: PolicyInferContext,
    generated_video: PolicyGeneratedVideo,
) -> PolicyInferContext:
    """Attach transferred video while preserving all available consumer context."""

    return replace(
        context,
        dynamics=None,
        output_request=None,
        video_generation=None,
        video_conditioned_action=build_video_conditioned_action_request(
            generated_video
        ),
    )


def _resolve_cuda_device(device: torch.device) -> torch.device:
    if device.index is not None:
        return device
    return torch.device("cuda", torch.cuda.current_device())


def _copy_cuda_rng_state(*, source: torch.device, target: torch.device) -> None:
    torch.cuda.set_rng_state(torch.cuda.get_rng_state(source), device=target)


__all__ = [
    "PolicyVideoActionConsumerPlan",
    "PolicyVideoProducerPlan",
    "build_video_conditioned_action_context",
    "build_video_conditioned_action_request",
    "require_compatible_video_latent_spaces",
    "require_generated_video",
    "resolve_policy_video_action_consumer_plan",
    "resolve_policy_video_producer_plan",
]
