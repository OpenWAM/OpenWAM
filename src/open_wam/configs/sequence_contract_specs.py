"""Canonical specifications for video/action sequence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from .enums import (
    ContextConditionLatentSource,
    HistoryStreamVisibility,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    VideoActionSequenceContract,
)


@dataclass(frozen=True, slots=True)
class _PolicyVariantSequenceSpec:
    proprio_context_mode: ProprioContextMode
    context_condition_latent_source: ContextConditionLatentSource
    history_stream_visibility: HistoryStreamVisibility
    use_condition_latents: bool
    require_condition_latents: bool


@dataclass(frozen=True, slots=True)
class _SampleConstructionSequenceSpec:
    condition_source_frame_offset: int
    start_padding_frames: int
    target_alignment: SampleTargetAlignment
    rollout_context_policy: RolloutContextPolicy | None = None


def _defined_field_values(
    value: _PolicyVariantSequenceSpec | _SampleConstructionSequenceSpec,
) -> dict[str, object]:
    return {
        field.name: field_value
        for field in fields(value)
        if (field_value := getattr(value, field.name)) is not None
    }


@dataclass(frozen=True, slots=True)
class VideoActionSequenceContractSpec:
    """Fields owned by one non-default video/action sequence contract."""

    policy_variant: _PolicyVariantSequenceSpec
    sample_construction: _SampleConstructionSequenceSpec
    requires_frame_aligned_proprio_context: bool
    additional_managed_override_keys: frozenset[str] = frozenset()

    def policy_variant_updates(self) -> dict[str, object]:
        """Return a writable copy of the contract-owned policy values."""

        return _defined_field_values(self.policy_variant)

    def sample_construction_updates(self) -> dict[str, object]:
        """Return a writable copy of the contract-owned sample values."""

        return _defined_field_values(self.sample_construction)

    def managed_override_keys(self) -> frozenset[str]:
        """Return CLI paths reserved while this contract is active."""

        return frozenset(
            {f"policy_variant.{key}" for key in self.policy_variant_updates()}
            | {
                f"data.sample_construction.{key}"
                for key in self.sample_construction_updates()
            }
            | self.additional_managed_override_keys
        )


_SINGLE_FRAME_PERCHUNK_PROPRIO_POLICY = _PolicyVariantSequenceSpec(
    proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
    context_condition_latent_source=(
        ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    ),
    history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
    use_condition_latents=True,
    require_condition_latents=True,
)


VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS: Mapping[
    VideoActionSequenceContract,
    VideoActionSequenceContractSpec,
] = MappingProxyType(
    {
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO: (
            VideoActionSequenceContractSpec(
                policy_variant=_SINGLE_FRAME_PERCHUNK_PROPRIO_POLICY,
                sample_construction=_SampleConstructionSequenceSpec(
                    condition_source_frame_offset=-1,
                    start_padding_frames=0,
                    target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                    rollout_context_policy=RolloutContextPolicy.ONE_FRAME,
                ),
                requires_frame_aligned_proprio_context=True,
            )
        ),
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO: (
            VideoActionSequenceContractSpec(
                policy_variant=_SINGLE_FRAME_PERCHUNK_PROPRIO_POLICY,
                sample_construction=_SampleConstructionSequenceSpec(
                    condition_source_frame_offset=-1,
                    start_padding_frames=0,
                    target_alignment=SampleTargetAlignment.LEGACY,
                ),
                requires_frame_aligned_proprio_context=True,
                # Preserve the existing CLI guard without making this field a
                # raw or typed legacy-prefix default.
                additional_managed_override_keys=frozenset(
                    {"data.sample_construction.rollout_context_policy"}
                ),
            )
        ),
    }
)


def _validate_spec_exhaustiveness() -> None:
    expected = frozenset(VideoActionSequenceContract) - {
        VideoActionSequenceContract.DEFAULT
    }
    actual = frozenset(VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS)
    if actual == expected:
        return
    missing = sorted(contract.value for contract in expected - actual)
    extra = sorted(contract.value for contract in actual - expected)
    raise RuntimeError(
        "Video/action sequence-contract specifications are not exhaustive: "
        f"missing={missing}, extra={extra}."
    )


_validate_spec_exhaustiveness()


def get_video_action_sequence_contract_spec(
    contract: VideoActionSequenceContract | str,
) -> VideoActionSequenceContractSpec | None:
    """Return the authoritative spec, or ``None`` for the default contract."""

    resolved_contract = VideoActionSequenceContract(contract)
    if resolved_contract is VideoActionSequenceContract.DEFAULT:
        return None
    return VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS[resolved_contract]


def video_action_sequence_contract_managed_override_keys(
    contract: VideoActionSequenceContract | str,
) -> frozenset[str]:
    """Return CLI override paths reserved by one active contract."""

    spec = get_video_action_sequence_contract_spec(contract)
    return frozenset() if spec is None else spec.managed_override_keys()


__all__ = [
    "VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS",
    "VideoActionSequenceContractSpec",
    "get_video_action_sequence_contract_spec",
    "video_action_sequence_contract_managed_override_keys",
]
