from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from .contracts import VisualCoreInput, VisualCoreOutput


class RuntimeSequenceFamily(StrEnum):
    """Sequence representation dispatched by the shared runtime executor."""

    DENSE = "dense_default"
    CHUNKED_DUAL_STREAM_TRAIN = "chunked_dual_stream_exact"
    CHUNKED_DUAL_STREAM_INFERENCE = "chunked_dual_stream_exact_inference"
    SINGLE_STREAM = "single_stream_exact"


@dataclass(frozen=True)
class RuntimeProgramSpec:
    """Semantic description of one runtime family over the shared backbone.

    A runtime program is the narrow contract between a policy variant and the
    shared transformer. Variants should decide *which* semantic program they
    want to run, while the shared backbone decides *how* that program is
    executed through sequence adapters, cache backends, attention kernels, and
    projection heads.
    """

    name: str
    sequence_family: RuntimeSequenceFamily
    attention_profile_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_family",
            RuntimeSequenceFamily(self.sequence_family),
        )
        if not self.name:
            raise ValueError("Runtime program names must be non-empty.")


@dataclass
class RuntimeStepInput:
    """Unified runtime-step request accepted by the shared backbone executor.

    Only one of the payload surfaces is normally used for a given program:

    - `core_input` for the generic dense shared-core path
    - `payload` for exact-runtime compatibility programs

    Keeping them on one request object gives dense and exact sequence families
    the same executor entrypoint while allowing distinct preparation contracts.
    """

    program: RuntimeProgramSpec
    core_input: VisualCoreInput | None = None
    payload: dict[str, Any] | None = None
    update_cache: int = 0
    cache_name: str = "open_wam_exact"
    action_mode: bool = False


@dataclass
class RuntimeStepOutput:
    """Unified runtime-step response returned by the shared backbone executor.

    `tokens` exposes raw hidden states when the caller wants to keep slicing or
    post-processing outside the core. Exact programs may additionally return
    named projections prepared by the shared backbone.
    """

    tokens: torch.Tensor | None = None
    core_output: VisualCoreOutput | None = None
    projected_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    cache_state: Any = None
    aux: dict[str, Any] = field(default_factory=dict)


def build_dense_runtime_program() -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="dense_default",
        sequence_family=RuntimeSequenceFamily.DENSE,
    )


def build_chunked_dual_stream_exact_train_program(
    *,
    attention_profile_name: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="chunked_dual_stream_exact_train",
        sequence_family=RuntimeSequenceFamily.CHUNKED_DUAL_STREAM_TRAIN,
        attention_profile_name=attention_profile_name,
    )


def build_chunked_dual_stream_exact_inference_program(
    *,
    attention_profile_name: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="chunked_dual_stream_exact_inference",
        sequence_family=RuntimeSequenceFamily.CHUNKED_DUAL_STREAM_INFERENCE,
        attention_profile_name=attention_profile_name,
    )


def build_single_stream_exact_runtime_program() -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="single_stream_exact",
        sequence_family=RuntimeSequenceFamily.SINGLE_STREAM,
    )
