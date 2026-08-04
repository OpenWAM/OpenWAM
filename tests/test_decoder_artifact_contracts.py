from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from open_wam.models.action_decoders.base import require_decoder_artifact_payload
from open_wam.models.policy_variants.contracts import (
    DecoderArtifactEnvelope,
    PolicyTrainOutput,
)


@dataclass(frozen=True)
class _Payload:
    value: int


def _policy_output(
    *,
    envelope: DecoderArtifactEnvelope | None = None,
    aux: dict[str, object] | None = None,
) -> PolicyTrainOutput:
    return PolicyTrainOutput(
        policy_features=torch.zeros(1, 1, 1),
        metrics={},
        decoder_artifacts=envelope,
        aux={} if aux is None else aux,
    )


def test_decoder_artifact_envelope_returns_typed_payload() -> None:
    payload = _Payload(value=7)
    envelope = DecoderArtifactEnvelope(contract="test.decoder.v1", payload=payload)

    assert envelope.require(
        contract="test.decoder.v1",
        payload_type=_Payload,
    ) is payload


def test_decoder_artifact_envelope_rejects_wrong_contract() -> None:
    envelope = DecoderArtifactEnvelope(
        contract="test.decoder.v1",
        payload=_Payload(value=7),
    )

    with pytest.raises(ValueError, match="does not match"):
        envelope.require(contract="other.decoder.v1", payload_type=_Payload)


def test_decoder_artifact_envelope_rejects_wrong_payload_type() -> None:
    envelope = DecoderArtifactEnvelope(contract="test.decoder.v1", payload=object())

    with pytest.raises(TypeError, match="requires payload _Payload"):
        envelope.require(contract="test.decoder.v1", payload_type=_Payload)


def test_decoder_uses_typed_envelope_and_ignores_aux_metadata() -> None:
    typed_payload = _Payload(value=7)
    output = _policy_output(
        envelope=DecoderArtifactEnvelope(
            contract="test.decoder.v1",
            payload=typed_payload,
        ),
        aux={"legacy": _Payload(value=3)},
    )

    assert require_decoder_artifact_payload(
        output,
        contract="test.decoder.v1",
        payload_type=_Payload,
    ) is typed_payload


def test_decoder_rejects_aux_only_or_untyped_artifacts() -> None:
    output = _policy_output(aux={"legacy": object()})

    with pytest.raises(ValueError, match="requires artifact contract"):
        require_decoder_artifact_payload(
            output,
            contract="test.decoder.v1",
            payload_type=_Payload,
        )
