"""Generic contracts for composing generated video with action inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from open_wam.contracts import require_compatible_video_latent_spaces
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


@dataclass(frozen=True)
class PolicyVideoProducerPlan:
    """How one policy should expose video for a downstream stage."""

    output_request: PolicyInferenceOutputRequest | None
    native_modalities: frozenset[PolicyOutputModality]
    recurrent_history_policy: PolicyRecurrentHistoryPolicy

    @property
    def uses_selective_output(self) -> bool:
        return self.output_request is not None

    def to_report(self) -> dict[str, object]:
        return {
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

        if (
            self.capability.rng_policy
            is not PolicyCompositionRngPolicy.CALLER_STREAM
        ):
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
) -> PolicyVideoProducerPlan:
    """Resolve video production from declared capabilities, never policy names."""

    capabilities = policy.inference_capabilities
    request = capabilities.request_for(
        frozenset({PolicyOutputModality.VIDEO})
    )
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
    if (
        request is not None
        and int(generated_video.latents.shape[2]) != int(request.frame_count)
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
