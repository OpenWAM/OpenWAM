"""Generic contracts for composing generated video with action inference."""

from __future__ import annotations

from dataclasses import dataclass

from open_wam.configs import DynamicsObjective
from open_wam.contracts import VideoLatentSpaceIdentity
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyGeneratedVideo,
    PolicyInferenceOutputRequest,
    PolicyOutputModality,
    PolicyRecurrentHistoryPolicy,
    PolicyVariant,
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
) -> DynamicsRolloutRequest:
    """Build the unified second-stage request from a generated video chunk."""

    return DynamicsRolloutRequest(
        objective=DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        clean_video=generated_video.latents,
        frame_chunk_size=int(generated_video.latents.shape[2]),
    )


def require_compatible_video_latent_spaces(
    producer: VideoLatentSpaceIdentity | None,
    consumer: VideoLatentSpaceIdentity | None,
) -> dict[str, str]:
    """Require a content-identical latent coordinate space for composition."""

    if producer is None or consumer is None:
        missing = []
        if producer is None:
            missing.append("producer")
        if consumer is None:
            missing.append("consumer")
        raise ValueError(
            "Generated-video composition requires artifact-backed latent-space "
            f"identity for {', '.join(missing)}."
        )
    if producer != consumer:
        raise ValueError(
            "Generated-video producer and action consumer use different latent "
            "spaces: "
            f"producer={producer.artifact_sha256}, "
            f"consumer={consumer.artifact_sha256}."
        )
    return producer.to_mapping()


__all__ = [
    "PolicyVideoProducerPlan",
    "build_video_conditioned_action_request",
    "require_compatible_video_latent_spaces",
    "require_generated_video",
    "resolve_policy_video_producer_plan",
]
