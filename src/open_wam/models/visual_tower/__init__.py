"""Stage-aware visual tower shared across policy variants."""

from .contracts import (
    DecodedFeatureLayout,
    RegisterSequenceComponents,
    RegisterSequenceSemantics,
    StructuredAttentionContext,
    StructuredBlockSemantics,
    StructuredFrequencyBundle,
    VisualCoreInput,
    VisualCoreOutput,
    VisualDecodeOutput,
    VisualFrontendOutput,
    VisualSequenceMetadata,
    VisualStageOutputs,
)
from .runtime_programs import (
    RuntimeProgramSpec,
    RuntimeStepInput,
    RuntimeStepOutput,
    build_chunked_dual_stream_exact_train_program,
    build_dense_runtime_program,
    build_register_sequence_runtime_program,
    build_single_stream_exact_runtime_program,
)
from .reference_transformer import build_reference_transformer, preferred_reference_dtype
from .stream_adapters import PreparedStreamInput, SharedRuntimeStreamAdapters, StreamInputAdapterSpec
from .stream_heads import StreamOutputHeadSpec
from .tower import VisualTower

__all__ = [
    "build_reference_transformer",
    "build_chunked_dual_stream_exact_train_program",
    "build_dense_runtime_program",
    "build_register_sequence_runtime_program",
    "build_single_stream_exact_runtime_program",
    "DecodedFeatureLayout",
    "PreparedStreamInput",
    "preferred_reference_dtype",
    "RegisterSequenceComponents",
    "RegisterSequenceSemantics",
    "SharedRuntimeStreamAdapters",
    "StreamInputAdapterSpec",
    "StreamOutputHeadSpec",
    "StructuredAttentionContext",
    "StructuredBlockSemantics",
    "StructuredFrequencyBundle",
    "RuntimeProgramSpec",
    "RuntimeStepInput",
    "RuntimeStepOutput",
    "VisualCoreInput",
    "VisualCoreOutput",
    "VisualDecodeOutput",
    "VisualFrontendOutput",
    "VisualSequenceMetadata",
    "VisualStageOutputs",
    "VisualTower",
]
