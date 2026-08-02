"""Stage-aware visual tower shared across policy variants."""

from .cache_lifecycle import RuntimeCacheLifecycle
from .contracts import (
    DecodedFeatureLayout,
    VisualCoreInput,
    VisualCoreOutput,
    VisualDecodeOutput,
    VisualFrontendOutput,
    VisualIntermediateReadout,
    VisualReadoutRequest,
    VisualRuntimeStateSnapshot,
    VisualSequenceMetadata,
    VisualStageOutputs,
)
from .exact_runtime import (
    build_reference_mesh_id,
    clear_exact_prediction_cache,
    initialize_exact_runtime_cache,
    prepare_exact_single_stream_forward_input,
    prepare_exact_single_stream_input,
    repeat_exact_single_stream_input_for_cfg,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)
from .runtime_programs import (
    RuntimeProgramSpec,
    RuntimeStepInput,
    RuntimeStepOutput,
    build_chunked_dual_stream_exact_inference_program,
    build_chunked_dual_stream_exact_train_program,
    build_dense_runtime_program,
    build_single_stream_exact_runtime_program,
)
from .reference_transformer import build_reference_transformer, preferred_reference_dtype
from .runtime_parameter_ops import (
    feed_forward_with_materialized_params,
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    rms_norm_with_materialized_weight,
)
from .shared_transformer_embeddings import (
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    apply_rotary_emb,
)
from .shared_transformer_layout import select_chunk_slices, select_split_segments
from .shared_transformer_support import (
    SharedTransformerAttention,
    SharedTransformerBlock,
)
from .tower import VisualTower

__all__ = [
    "build_reference_transformer",
    "build_chunked_dual_stream_exact_inference_program",
    "build_chunked_dual_stream_exact_train_program",
    "build_dense_runtime_program",
    "build_reference_mesh_id",
    "build_single_stream_exact_runtime_program",
    "clear_exact_prediction_cache",
    "DecodedFeatureLayout",
    "preferred_reference_dtype",
    "SharedTransformerAttention",
    "SharedTransformerBlock",
    "SharedTransformerRotaryPositionalEmbedding",
    "SharedTransformerTimeEmbedding",
    "RuntimeProgramSpec",
    "RuntimeCacheLifecycle",
    "RuntimeStepInput",
    "RuntimeStepOutput",
    "apply_rotary_emb",
    "feed_forward_with_materialized_params",
    "layer_norm_with_materialized_params",
    "linear_with_materialized_params",
    "materialize_runtime_parameter",
    "initialize_exact_runtime_cache",
    "prepare_exact_single_stream_forward_input",
    "prepare_exact_single_stream_input",
    "repeat_exact_single_stream_input_for_cfg",
    "resolve_runtime_module_dtype",
    "rms_norm_with_materialized_weight",
    "select_chunk_slices",
    "select_split_segments",
    "run_exact_single_stream_forward",
    "VisualCoreInput",
    "VisualCoreOutput",
    "VisualDecodeOutput",
    "VisualFrontendOutput",
    "VisualIntermediateReadout",
    "VisualReadoutRequest",
    "VisualRuntimeStateSnapshot",
    "VisualSequenceMetadata",
    "VisualStageOutputs",
    "VisualTower",
]
