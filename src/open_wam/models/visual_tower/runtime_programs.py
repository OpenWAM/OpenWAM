from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import (
    StructuredAttentionContext,
    StructuredBlockSemantics,
    StructuredFrequencyBundle,
    VisualCoreInput,
    VisualCoreOutput,
)


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
    sequence_family: str
    attention_profile_name: str | None = None
    cache_backend_name: str | None = None
    conditioning_mode: str = "default"
    teacher_forcing_layout: str = "none"
    stream_layout: str = "single"
    projection_mode: str = "core_output"
    runtime_family: str = "shared"
    input_adapter_family: str | None = None
    output_head_family: str | None = None
    structured_cache_kernel: str | None = None


@dataclass
class RuntimeStepInput:
    """Unified runtime-step request accepted by the shared backbone executor.

    Only one of the payload surfaces is normally used for a given program:

    - `core_input` for the generic packed/structured shared-core path
    - `payload` for exact-runtime compatibility programs

    Keeping them on one request object lets method 1 and method 2 share the
    same executor entrypoint even though their sequence preparation differs.
    """

    program: RuntimeProgramSpec
    core_input: VisualCoreInput | None = None
    payload: dict[str, Any] | None = None
    update_cache: int = 0
    cache_name: str = "open_wam_exact"
    action_mode: bool = False
    train_mode: bool = False
    structured_block_semantics: StructuredBlockSemantics | None = None
    structured_frequency_bundle: StructuredFrequencyBundle | None = None
    structured_attention_context: StructuredAttentionContext | None = None


@dataclass
class RuntimeStepOutput:
    """Unified runtime-step response returned by the shared backbone executor.

    `tokens` exposes raw hidden states when the caller wants to keep slicing or
    post-processing outside the core. `projected_outputs` is the shared path
    for backbone-owned stream heads, which method 2 now uses directly.
    """

    tokens: torch.Tensor | None = None
    core_output: VisualCoreOutput | None = None
    projected_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    named_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    cache_state: Any = None
    aux: dict[str, Any] = field(default_factory=dict)


def build_dense_runtime_program() -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="dense_default",
        sequence_family="dense_default",
        stream_layout="single",
        projection_mode="core_output",
    )


def build_register_sequence_runtime_program(
    *,
    input_adapter_family: str | None = None,
    output_head_family: str | None = None,
    structured_cache_kernel: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="register_sequence",
        sequence_family="register_sequence",
        teacher_forcing_layout="clean_prefix",
        stream_layout="video_action_state",
        projection_mode="structured_joint_flow",
        input_adapter_family=input_adapter_family,
        output_head_family=output_head_family,
        structured_cache_kernel=structured_cache_kernel,
    )


def build_chunked_dual_stream_exact_train_program(
    *,
    attention_profile_name: str | None = None,
    cache_backend_name: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="chunked_dual_stream_exact_train",
        sequence_family="chunked_dual_stream_exact",
        attention_profile_name=attention_profile_name,
        cache_backend_name=cache_backend_name,
        teacher_forcing_layout="chunked_dual_stream",
        stream_layout="video_then_action_dual",
        projection_mode="dual_stream_exact",
        runtime_family="exact",
    )


def build_chunked_dual_stream_exact_inference_program(
    *,
    attention_profile_name: str | None = None,
    cache_backend_name: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="chunked_dual_stream_exact_inference",
        sequence_family="chunked_dual_stream_exact_inference",
        attention_profile_name=attention_profile_name,
        cache_backend_name=cache_backend_name,
        teacher_forcing_layout="chunked_dual_stream",
        stream_layout="video_then_action_dual",
        projection_mode="dual_stream_exact",
        runtime_family="exact",
    )


def build_single_stream_exact_runtime_program(
    *,
    cache_backend_name: str | None = None,
) -> RuntimeProgramSpec:
    return RuntimeProgramSpec(
        name="single_stream_exact",
        sequence_family="single_stream_exact",
        cache_backend_name=cache_backend_name,
        stream_layout="single",
        projection_mode="single_stream_exact",
        runtime_family="exact",
    )
