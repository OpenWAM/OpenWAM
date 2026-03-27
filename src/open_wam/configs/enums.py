from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, TypeAlias, TypeVar


EnumT = TypeVar("EnumT", bound=StrEnum)
FieldTransform: TypeAlias = Callable[[Any], Any]
EnumFieldMap: TypeAlias = Mapping[str, type[EnumT]]
TransformFieldMap: TypeAlias = Mapping[str, FieldTransform]


class ActionDecoderName(StrEnum):
    """Final action-decoder family selected by experiment config."""

    MLP = "mlp_decoder"
    REGISTER = "register_decoder"
    DECODED_FEATURE = "decoded_feature_decoder"
    LINGBOT_PARALLEL = "lingbot_parallel_decoder"


class DataSplit(StrEnum):
    """Dataset split used by train/eval loaders."""

    TRAIN = "train"
    VAL = "val"


# Action-target and supervision enums.
class ActionTargetRepresentation(StrEnum):
    """Public action-target family exposed by the data layer."""

    RAW = "raw"
    EEF_POSE_RELATIVE_TO_REFERENCE = "eef_pose_relative_to_reference"


class ActionTargetStateEncoding(StrEnum):
    """How proprio state should be unpacked into pose/gripper fields."""

    IDENTITY = "identity"
    EEF_POS_AXISANGLE_GRIPPER_2D = "eef_pos_axisangle_gripper_2d"
    EEF_POS_QUAT_GRIPPER_1D = "eef_pos_quat_gripper_1d"


class ActionTargetReferenceSource(StrEnum):
    """Reference pose source used by relative action-target construction."""

    ANCHOR_STATE = "anchor_state"


class RotationRepresentation(StrEnum):
    """Rotation parameterization used in pose targets."""

    QUAT = "quat"
    AXIS_ANGLE = "axis_angle"


class GripperRepresentation(StrEnum):
    """Public gripper target representation."""

    FIRST_CHANNEL = "first_channel"
    ALL_CHANNELS = "all_channels"
    ACTION_COMMAND = "action_command"


# Inference-time rollout and CFG enums.
class JointSampler(StrEnum):
    """Joint video/action sampler family for rollout-time denoising."""

    FLOW_MATCH = "flow_match"
    UNIPC = "unipc"


class CFGMode(StrEnum):
    """Per-stream classifier-free guidance behavior."""

    GUIDED = "guided"
    CONDITIONED = "conditioned"
    UNCONDITIONED = "unconditioned"


class JointCfgApplication(StrEnum):
    """Legacy shorthand for configuring joint rollout CFG behavior."""

    JOINT = "joint"
    VIDEO_ONLY = "video_only"


class CacheUpdateMode(StrEnum):
    """When cache state should be updated during rollout."""

    WARMUP_ONLY = "warmup_only"
    FINAL_STEP = "final_step"
    EVERY_STEP = "every_step"
    NONE = "none"


class CacheWarmupSource(StrEnum):
    """Where rollout cache warmup should source clean reference frames from."""

    REFERENCE_VIDEO = "reference_video"
    NONE = "none"


class WarmupAnchor(StrEnum):
    """How a warmup slice should be selected from the current reference window."""

    START = "start"
    END = "end"
    FULL = "full"


# Policy-variant architecture enums.
class PolicyVariantName(StrEnum):
    """Top-level policy family supported by the repo."""

    POST_LATENT = "post_latent"
    POST_DECODED = "post_decoded"
    REGISTER_ATTACHED = "register_attached"
    PARALLEL_STREAM = "parallel_stream"


class AttachSite(StrEnum):
    """Where policy logic conceptually attaches relative to the visual stack."""

    POST_FRONTEND_LATENTS = "post_frontend_latents"
    POST_VISUAL_CORE = "post_visual_core"
    POST_VISUAL_DECODE = "post_visual_decode"
    WITHIN_VISUAL_CORE = "within_visual_core"


class PoolingMode(StrEnum):
    """How frame/token features are pooled into policy features."""

    PER_FRAME_MEAN = "per_frame_mean"
    COMPAT_GLOBAL_MEAN = "compat_global_mean"


class TemporalProjection(StrEnum):
    """How feature sequences are aligned to the action horizon."""

    INTERPOLATE = "interpolate"


class DecodeFeatureMode(StrEnum):
    """How decoded visual features are surfaced to a decoder."""

    FRAME_TOKEN_SEQUENCE = "frame_token_sequence"


class RegisterLayout(StrEnum):
    """Ordering of action/state registers in register-attached variants."""

    ACTION_THEN_STATE = "action_then_state"


class RegisterMaskMode(StrEnum):
    """Masking profile for register-attached sequence packing."""

    DREAMZERO_BLOCKWISE = "dreamzero_blockwise"


class StreamEncoderType(StrEnum):
    """Adapter family used for action/state stream embeddings."""

    MLP = "mlp"


class StructuredBlockMode(StrEnum):
    """Structured block semantics understood by the shared visual core."""

    REGISTER_EXPLICIT = "register_explicit"


class StructuredTimeLayout(StrEnum):
    """Temporal ordering convention for structured register sequences."""

    VIDEO_ACTION_STATE = "video_action_state"


class StructuredFrequencyMode(StrEnum):
    """How frequency/position signals are allocated across structured streams."""

    STREAM_LOCAL = "stream_local"


class StructuredTeacherForcingLayout(StrEnum):
    """Teacher-forcing layout used by structured block runtimes."""

    CLEAN_PREFIX = "clean_prefix"


class StructuredAttentionKernel(StrEnum):
    """Attention-kernel family used by structured block execution."""

    BRANCHWISE_EXPLICIT = "branchwise_explicit"


class StructuredCacheKernel(StrEnum):
    """Cache-update kernel used by structured rollout execution."""

    BRANCHWISE_ROLLOUT_EXPLICIT = "branchwise_rollout_explicit"


class StreamInputAdapterFamily(StrEnum):
    """Shared-core input adapter family for structured runtime programs."""

    STRUCTURED_REGISTER_STREAMS = "structured_register_streams"


class StreamOutputHeadFamily(StrEnum):
    """Shared-core output head family for structured runtime programs."""

    STRUCTURED_JOINT_FLOW = "structured_joint_flow"


class ParallelRuntimeMode(StrEnum):
    """Execution mode for the method-1 parallel-stream variant."""

    LINGBOT_EXACT = "lingbot_exact"


class ParallelSequenceComponent(StrEnum):
    """Sequence components packed by the exact method-1 runtime."""

    VIDEO_NOISY = "video_noisy"
    VIDEO_CONDITION = "video_condition"
    ACTION_NOISY = "action_noisy"
    ACTION_CONDITION = "action_condition"


class ParallelMaskMode(StrEnum):
    """Mask profile used by the exact method-1 runtime."""

    LINGBOT_CHUNKED = "lingbot_chunked"


class ParallelCacheMode(StrEnum):
    """How much cache metadata/state the exact method-1 runtime stores locally."""

    METADATA_ONLY = "metadata_only"


class ActionNormMethod(StrEnum):
    """Raw-to-model action normalization strategy for exact method-1 paths."""

    NONE = "none"
    QUANTILES = "quantiles"


class ActionSpace(StrEnum):
    """Whether an action tensor is in raw dataset space or model space."""

    AUTO = "auto"
    MODEL = "model"
    RAW = "raw"


# Trainer/runtime enums.
class TrainerAccelerator(StrEnum):
    """Device family requested by the train/eval launcher."""

    CPU = "cpu"
    GPU = "gpu"


class TrainerPrecision(StrEnum):
    """Numerical precision mode used by the trainer/runtime strategy."""

    FP32 = "32-true"
    BF16 = "bf16-mixed"
    FP16 = "16-mixed"


class TrainerRuntimeName(StrEnum):
    """Top-level training engine used to run one experiment."""

    LIGHTNING = "lightning"
    COMPOSABLE = "composable"


class BatchAdapterName(StrEnum):
    """Input adapter used by the training runtime."""

    VIEWS = "views"
    LATENTS = "latents"


class LoopPolicyName(StrEnum):
    """Primary control structure used by the training runtime."""

    EPOCHS = "epochs"
    STEPS = "steps"


class StrategyName(StrEnum):
    """Distribution/wrapping backend used by the composable runtime."""

    LIGHTNING = "lightning"
    SINGLE_DEVICE = "single_device"
    DDP = "ddp"
    FSDP = "fsdp"


class CheckpointMode(StrEnum):
    """Checkpoint payload level written by the composable runtime."""

    FULL_TRAINING_STATE = "full_training_state"
    MODEL_ONLY = "model_only"


class WandBMode(StrEnum):
    """Weights & Biases connectivity mode."""

    DISABLED = "disabled"
    OFFLINE = "offline"
    ONLINE = "online"


class OptimizerName(StrEnum):
    """Optimizer family supported by the shared training config."""

    ADAMW = "adamw"


class SchedulerName(StrEnum):
    """Learning-rate schedule family supported by the shared training config."""

    CONSTANT = "constant"
    WARMUP_CONSTANT = "warmup_constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class TrainingObjective(StrEnum):
    """Supervision objective families that can be enabled or disabled."""

    ACTION = "action"
    LATENT = "latent"


class TrainingComponentSelector(StrEnum):
    """Named module groups that can be frozen or made trainable."""

    ALL = "all"
    VISUAL_TOWER = "visual_tower"
    VISUAL_TOWER_FRONTEND = "visual_tower.frontend"
    VISUAL_TOWER_CORE = "visual_tower.core"
    VISUAL_TOWER_RUNTIME_BACKBONE = "visual_tower.runtime_backbone"
    VISUAL_TOWER_DECODER = "visual_tower.decoder"
    POLICY_VARIANT = "policy_variant"
    ACTION_DECODER = "action_decoder"


# Backbone/evaluation enums.
class BackboneImplementation(StrEnum):
    """Visual-backbone implementation family."""

    SHARED_TRANSFORMER = "shared_transformer"
    DUMMY = "dummy"


class AttentionMode(StrEnum):
    """Attention backend used by the shared transformer."""

    TORCH = "torch"
    FLEX = "flex"


class ReferenceAssetsDevicePolicy(StrEnum):
    """Placement policy for VAE/text reference assets."""

    RUNTIME = "runtime"
    CPU_OFFLOAD = "cpu_offload"


class EvalMode(StrEnum):
    """Evaluation mode supported by the generic eval entrypoint."""

    BATCH = "batch"
    TRAJECTORY = "trajectory"
    TRAJECTORY_OPEN_LOOP = "trajectory_open_loop"


class EvalPredictionSource(StrEnum):
    """Which tensor source was used to score an eval prediction."""

    UNAVAILABLE = "unavailable"
    DECODER_ACTION_PRED = "decoder_action_pred"
    RAW_CHUNK_ACTION_PRED = "raw_chunk_action_pred"
    DECODER_ACTION_PRED_UNMATCHED = "decoder_action_pred_unmatched"
    DECODER_PREDICTED_LATENTS = "decoder_predicted_latents"
    DECODER_PREDICTED_VIDEO_LATENTS = "decoder_predicted_video_latents"
    POLICY_PREDICTED_LATENTS = "policy_predicted_latents"
    POLICY_PREDICTED_VIDEO_LATENTS = "policy_predicted_video_latents"


def coerce_enum_value(enum_cls: type[EnumT], value: EnumT | str) -> EnumT:
    """Convert a raw string or existing enum member into one enum member."""

    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def coerce_optional_enum_value(enum_cls: type[EnumT], value: EnumT | str | None) -> EnumT | None:
    """Optional version of `coerce_enum_value` for nullable config fields."""

    if value is None:
        return None
    return coerce_enum_value(enum_cls, value)


def coerce_enum_tuple(
    enum_cls: type[EnumT],
    values: tuple[EnumT | str, ...] | list[EnumT | str],
) -> tuple[EnumT, ...]:
    """Convert one sequence of raw strings/enum members into an enum tuple."""

    return tuple(coerce_enum_value(enum_cls, value) for value in values)


def set_frozen_fields(instance: Any, /, **updates: Any) -> None:
    """Apply field updates to a frozen dataclass instance."""

    for field_name, value in updates.items():
        object.__setattr__(instance, field_name, value)


def coerce_fields(
    instance: Any,
    *,
    enum_fields: EnumFieldMap[EnumT] | None = None,
    optional_enum_fields: EnumFieldMap[EnumT] | None = None,
    enum_tuple_fields: EnumFieldMap[EnumT] | None = None,
    transforms: TransformFieldMap | None = None,
) -> None:
    """Coerce selected frozen-dataclass fields in one compact declaration.

    This keeps enum normalization close to each config class while avoiding
    repeated `object.__setattr__` blocks in every `__post_init__`.
    """

    updates: dict[str, Any] = {}
    for field_name, enum_cls in (enum_fields or {}).items():
        updates[field_name] = coerce_enum_value(enum_cls, getattr(instance, field_name))
    for field_name, enum_cls in (optional_enum_fields or {}).items():
        updates[field_name] = coerce_optional_enum_value(enum_cls, getattr(instance, field_name))
    for field_name, enum_cls in (enum_tuple_fields or {}).items():
        updates[field_name] = coerce_enum_tuple(enum_cls, getattr(instance, field_name))
    for field_name, transform in (transforms or {}).items():
        updates[field_name] = transform(getattr(instance, field_name))
    set_frozen_fields(instance, **updates)


def serialize_enum_values(value: Any) -> Any:
    """Recursively convert enums/dataclasses into plain JSON/YAML-safe values."""

    if isinstance(value, StrEnum):
        return str(value)
    if is_dataclass(value):
        return serialize_enum_values(asdict(value))
    if isinstance(value, dict):
        return {key: serialize_enum_values(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [serialize_enum_values(item) for item in value]
    if isinstance(value, list):
        return [serialize_enum_values(item) for item in value]
    return value
