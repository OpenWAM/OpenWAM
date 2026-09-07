from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, TypeAlias, TypeVar

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised in RoboTwin's Python 3.10 env.

    class StrEnum(str, Enum):
        """Python 3.10 fallback matching the string behavior of stdlib StrEnum."""



EnumT = TypeVar("EnumT", bound=StrEnum)
FieldTransform: TypeAlias = Callable[[Any], Any]
EnumFieldMap: TypeAlias = Mapping[str, type[EnumT]]
TransformFieldMap: TypeAlias = Mapping[str, FieldTransform]


class ActionDecoderName(StrEnum):
    """Final action-decoder family selected by experiment config."""

    PARALLEL_STREAM = "parallel_stream_decoder"
    # Deprecated Python/value compatibility. Raw legacy values are normalized
    # to PARALLEL_STREAM at the config boundary.
    LINGBOT_PARALLEL = "lingbot_parallel_decoder"
    DUAL_EXPERT = "dual_expert_decoder"
    # Deprecated Python symbol alias. Raw `mot_decoder` config values are
    # normalized at the config boundary.
    MOT = "dual_expert_decoder"
    VIDEO_ONLY = "video_only_decoder"
    EXTENSION = "extension"


class DataSplit(StrEnum):
    """Dataset split used by train/eval loaders."""

    TRAIN = "train"
    VAL = "val"


class LegacyPicklePolicy(StrEnum):
    """Whether a dataset may deserialize a documented legacy pickle format."""

    SAFE_ONLY = "safe_only"
    TRUSTED_LEGACY = "trusted_legacy"


class AuxiliaryValidationSource(StrEnum):
    """Dataset source used by an auxiliary validation probe."""

    DATASET = "dataset"
    REAL_DEMO = "real_demo"
    COUNTERFACTUAL_DYNAMICS = "counterfactual_dynamics"
    COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE = "counterfactual_dynamics_if_available"


class ReplayStatusPolicy(StrEnum):
    """How dataset replay labels constrain episode selection."""

    INCLUDE_ALL = "include_all"
    SUCCESSFUL_ONLY = "successful_only"
    FAILURE_ONLY = "failure_only"


class ActionMappingMode(StrEnum):
    """How data-layer action targets are mapped into model-facing dimensions."""

    NONE = "none"
    PAD_AND_REORDER = "pad_and_reorder"
    SPARSE_CANVAS = "sparse_canvas"


class ActionMappingLossMaskMode(StrEnum):
    """How mapped action dimensions contribute to supervised losses."""

    SOURCE_MASK = "source_mask"
    ACTIVE_TARGET_INDICES = "active_target_indices"


class ActionMappingSamplerMaskMode(StrEnum):
    """How inactive mapped action dimensions should be treated by samplers."""

    NONE = "none"
    PIN_INACTIVE_CHANNELS = "pin_inactive_channels"


class ActionNormalizationMode(StrEnum):
    """Optional data-layer normalization applied around action mappings."""

    NONE = "none"
    QUANTILES = "quantiles"
    JOINT_LIMITS = "joint_limits"
    GAUSSIAN = "gaussian"


class WindowSamplingMode(StrEnum):
    """How one training sample is constructed from a latent source segment."""

    FULL_SEGMENT = "full_segment"
    UNIFORM_SEGMENT = "uniform_segment"
    HIERARCHICAL_FIXED_SEGMENT = "hierarchical_fixed_segment"
    CAUSAL_PREFIX_SUFFIX = "causal_prefix_suffix"


class SegmentContextPolicy(StrEnum):
    """How fixed-segment samplers reserve rollout context before supervised frames."""

    NONE = "none"
    FIXED = "fixed"
    ROLLOUT_HISTORY = "rollout_history"


class SampleTargetAlignment(StrEnum):
    """How fixed-segment samples align materialized context to supervised targets."""

    LEGACY = "legacy"
    # Materialize context before the first supervised target. Frame 0 is
    # observed context; frames 1..segment_frames are generated targets.
    NEXT_AFTER_CONTEXT = "next_after_context"


class RolloutContextPolicy(StrEnum):
    """How strict rollout-parity fixed segments choose pre-target context."""

    ONE_FRAME = "one_frame"
    ROLLOUT_HISTORY = "rollout_history"


class TailPaddingPolicy(StrEnum):
    """How fixed segment samplers fill positions beyond the real trajectory tail."""

    ZERO_ORDER_HOLD = "zero_order_hold"


class PaddedTargetPolicy(StrEnum):
    """How fixed segment samplers supervise synthetic padded positions."""

    MASK_LOSS = "mask_loss"


class AnchorPolicy(StrEnum):
    """How one local subwindow anchor is chosen within a valid source segment."""

    RANDOM_VALID = "random_valid"


class SampleStateAnchorMode(StrEnum):
    """Which observed raw frame anchors the state sequence in latent samples."""

    PROPRIO_CONTEXT_FRAME = "proprio_context_frame"
    ANCHOR_FRAME = "anchor_frame"
    SAMPLE_START_FRAME = "sample_start_frame"
    FIRST_OBSERVED_FRAME = "first_observed_frame"


class TemporalPositionMode(StrEnum):
    """How local windows are mapped onto the transformer temporal position axis."""

    GLOBAL_SHIFTED = "global_shifted"
    LOCAL_ZERO_BASED = "local_zero_based"


class LatentWindowProfile(StrEnum):
    """High-level latent-window contract for local latent datasets."""

    EXACT_CHUNKED_WINDOW = "exact_chunked_window"
    STANDARD_POLICY_WINDOW = "standard_policy_window"


class LatentTemporalLayout(StrEnum):
    """How raw video frames map onto encoded video latent indices."""

    # Wan/LingBot VAE encodes the first frame alone, then causal stride-4 groups.
    WAN_CAUSAL_STRIDE4 = "wan_causal_stride4"
    # Deprecated sentinel. Configs using this value are rejected with an explicit error.
    EQUAL_BUCKET_LEGACY = "equal_bucket_legacy"


class SampleWeightMode(StrEnum):
    """How local latent datasets weight train-sampler draws."""

    UNIFORM = "uniform"
    VALID_ACTION_STEPS = "valid_action_steps"
    INVERSE_TASK_DEMO_COUNT = "inverse_task_demo_count"
    VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT = "valid_action_steps_x_inverse_task_demo_count"
    TASK_VIRTUAL_START_COUNT_POWER = "task_virtual_start_count_power"


class SampleOrderMode(StrEnum):
    """How local latent train samplers order candidate examples."""

    EPOCH_ORDER = "epoch_order"
    REPLACEMENT = "replacement"


class ConsortiumChannelSelectionMode(StrEnum):
    """How a consortium dataset selects visual channels from each member repo."""

    ALL_AVAILABLE = "all_available"
    REQUIRED_SUBSET = "required_subset"
    EXPLICIT_MAPPING = "explicit_mapping"


class ConsortiumViewPackingMode(StrEnum):
    """How multi-camera observations are exposed to the model."""

    MULTICAM_AS_SLOTS = "multicam_as_slots"
    MULTICAM_AS_FRAMES = "multicam_as_frames"


class ConsortiumFramePackingOrder(StrEnum):
    """How cameras are enumerated when cameras are flattened into frames."""

    CAMERA_MAJOR = "camera_major"


class ConsortiumMissingChannelPolicy(StrEnum):
    """What to do when one configured canonical slot has no source channel."""

    ERROR = "error"
    ZERO_FILL = "zero_fill"


class MixedVideoMissingStreamPolicy(StrEnum):
    """What to do when a mixed-video episode lacks a configured output stream."""

    ERROR = "error"
    ZERO_FILL = "zero_fill"


class MixedVideoDecodeSizeMode(StrEnum):
    """How mixed-video RGB streams are resized before VAE encoding."""

    FIXED = "fixed"
    ASPECT_RATIO_BINS = "aspect_ratio_bins"


class MixedVideoFrameFitMode(StrEnum):
    """How mixed-video RGB frames are fit into the selected decode canvas."""

    CENTER_CROP = "center_crop"
    LETTERBOX_PAD = "letterbox_pad"


class MixedVideoLatentEncodingMode(StrEnum):
    """Which latent sidecar representation the mixed-video encoder writes."""

    CANONICAL = "canonical"
    PER_VIEW = "per_view"
    CANONICAL_AND_PER_VIEW = "canonical_and_per_view"


class MixedVideoEncodingSplit(StrEnum):
    """Episode split selected by the offline mixed-video encoder."""

    ALL = "all"
    TRAIN = "train"
    VAL = "val"


class MixedVideoSourceFormat(StrEnum):
    """Which media representations one mixed-video source can provide."""

    RGB = "rgb"
    LATENT = "latent"
    RGB_AND_LATENT = "rgb_and_latent"


class ConsortiumRandomMode(StrEnum):
    """How the train sampler randomizes consortium samples."""

    NONE = "none"
    WITHIN_DATASET = "within_dataset"
    TRAJECTORY_GLOBAL = "trajectory_global"


class MixedVideoRandomMode(StrEnum):
    """How the mixed-video train sampler randomizes source-balanced samples."""

    NONE = "none"
    WITHIN_SOURCE = "within_source"
    GLOBAL = "global"


class ConsortiumWeightMode(StrEnum):
    """How per-dataset weight overrides affect one training epoch."""

    PROPORTIONAL_TO_SIZE = "proportional_to_size"
    PROPORTIONAL_THEN_MANUAL_SCALE = "proportional_then_manual_scale"
    MANUAL_OVERRIDE = "manual_override"


class MixedVideoWeightMode(StrEnum):
    """How mixed-video source weights are converted into one training epoch."""

    PROPORTIONAL_TO_SIZE = "proportional_to_size"
    PROPORTIONAL_THEN_MANUAL_SCALE = "proportional_then_manual_scale"
    MANUAL_OVERRIDE = "manual_override"


class ConsortiumSplitMode(StrEnum):
    """How consortium member episodes are split into train and val."""

    HASH_BY_EPISODE = "hash_by_episode"
    SEEDED_SHUFFLE_BY_EPISODE = "seeded_shuffle_by_episode"
    EXPLICIT_MANIFEST = "explicit_manifest"


class ConsortiumCacheMode(StrEnum):
    """Runtime behavior of one optional consortium cache tier."""

    DISABLED = "disabled"
    WRITE_THROUGH = "write_through"
    READ_ONLY = "read_only"


class ConsortiumCloudCacheBackend(StrEnum):
    """Backend family for the optional consortium cloud cache."""

    FILESYSTEM = "filesystem"


# Action-target and supervision enums.
class ActionTargetRepresentation(StrEnum):
    """Public action-target family exposed by the data layer."""

    RAW = "raw"
    EEF_POSE_RELATIVE_TO_REFERENCE = "eef_pose_relative_to_reference"
    ABSOLUTE_JOINT_POSITION = "absolute_joint_position"


class LiberoAbsoluteJointExecutionMode(StrEnum):
    """How the LIBERO adapter executes absolute joint-position targets."""

    # Public robosuite JOINT_POSITION API: normalized relative joint delta.
    NORMALIZED_DELTA = "normalized_delta"
    # Model target is an integrated pseudo-qpos; finite differences recover the
    # normalized JOINT_POSITION command.
    INTEGRATED_DELTA = "integrated_delta"
    # Adapter-owned absolute qpos goal hook: controller.set_goal(..., set_qpos=target).
    DIRECT_GOAL = "direct_goal"


class LiberoRendererBackend(StrEnum):
    """Headless OpenGL backend used by a LIBERO simulator process."""

    EGL = "egl"
    OSMESA = "osmesa"


class LiberoRendererProfile(StrEnum):
    """Workload-level renderer contract for reproducible LIBERO execution."""

    ONLINE_ROLLOUT = "online_rollout"
    OFFLINE_ANALYSIS = "offline_analysis"
    DATASET_GENERATION = "dataset_generation"


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
    CONTINUOUS_6D = "continuous_6d"


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

    CAUSAL_VIDEO_PREDICTION = "causal_video_prediction"
    DUAL_EXPERT = "dual_expert"
    # Deprecated Python symbol alias. Raw `mot` config values are normalized
    # before enum coercion.
    MOT = "dual_expert"
    PARALLEL_STREAM = "parallel_stream"
    EXTENSION = "extension"


class CausalVideoProgram(StrEnum):
    """Sequence and supervision contract for video-only prediction."""

    PREFIX_SUFFIX = "prefix_suffix"
    CHUNKED_CONDITIONED_VIDEO = "chunked_conditioned_video"


class TextConditioningMode(StrEnum):
    """How a policy supplies semantic text context to the visual stack."""

    TASK_PROMPT = "task_prompt"
    DISABLED = "disabled"


class DualExpertActionExpertInitMode(StrEnum):
    """How the DualExpert action expert should initialize from the video expert."""

    RANDOM = "random"
    VIDEO_WEIGHT_COPY = "video_weight_copy"
    VIDEO_WEIGHT_INTERPOLATE = "video_weight_interpolate"


class DualExpertConditionMode(StrEnum):
    """Which video branch the DualExpert action expert conditions on."""

    FIRST_FRAME = "first_frame"
    FULL_VIDEO = "full_video"
    TEACHER_FORCING_COND_VIDEO = "teacher_forcing_cond_video"


class DualExpertPreset(StrEnum):
    """High-level FastWAM-style preset families for DualExpert policy defaults."""

    FASTWAM = "fastwam"
    FASTWAM_JOINT = "fastwam_joint"
    FASTWAM_IDM = "fastwam_idm"
    FASTWAM_NON_JOINT = "fastwam_non_joint"


# Deprecated Method-5/MoT type names. Keeping class identity preserves old
# imports and enum-bearing serialized objects without creating two semantics.
MoTActionExpertInitMode = DualExpertActionExpertInitMode
MoTConditionMode = DualExpertConditionMode
MoTPreset = DualExpertPreset


class AttachSite(StrEnum):
    """Where policy logic conceptually attaches relative to the visual stack."""

    POST_FRONTEND_LATENTS = "post_frontend_latents"
    POST_VISUAL_CORE = "post_visual_core"
    WITHIN_VISUAL_CORE = "within_visual_core"


class ParallelRuntimeMode(StrEnum):
    """Numerical execution backend for the parallel-stream architecture."""

    LINGBOT_EXACT = "lingbot_exact"
    LINGBOT_EXACT_ACTION_CONDITIONED = "lingbot_exact_action_conditioned"


class ParallelStreamVariantProfile(StrEnum):
    """Checkpoint-era Parallel Stream profile retained for metadata migration."""

    STANDARD = "standard"
    GENERALIST_JOINT_DENOISING = "generalist_joint_denoising"


class DynamicsObjective(StrEnum):
    """One video/action denoising objective selected for a routed sample.

    Under a fixed joint coupling, each training segment samples one regime.
    ``joint`` denoises both modalities. The conditional modes place the clean
    supplied modality in its noisy slot at timestep zero, mask that modality's
    loss, remove task text, and retain one local clean video-history anchor.
    """

    JOINT = "joint"
    ACTION_CONDITIONED_VIDEO = "action_conditioned_video"
    VIDEO_CONDITIONED_ACTION = "video_conditioned_action"

    @property
    def is_conditional(self) -> bool:
        """Whether the objective predicts one modality from the other."""

        return self in {
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        }


class DynamicsSource(StrEnum):
    """Dataset source used by one dynamics-routed training route."""

    REAL_DEMO = "real_demo"
    COUNTERFACTUAL_DYNAMICS = "counterfactual_dynamics"


class ParallelActionConditionSource(StrEnum):
    """Which action stream should be exposed to video denoising."""

    NOISY_ACTION = "noisy_action"
    CLEAN_ACTION = "clean_action"


class ParallelActionAttentionScope(StrEnum):
    """How broadly video tokens may attend to action tokens."""

    FULL = "full"
    BLOCK_LOCAL = "block_local"


class ContextConditionLatentSource(StrEnum):
    """Which latent source supplies clean pre-target video context frames."""

    VIDEO_LATENTS = "video_latents"
    SINGLE_FRAME_CONDITION_LATENT = "single_frame_condition_latent"


class BatchingMode(StrEnum):
    """Rank-local latent batching; bucket sorts nearby lengths then pads.

    Bucket mode preserves the selected sampler indices, including replacement
    multiplicity. It does not require identical lengths or discard rare lengths.
    Packed mode additionally removes padding inside the transformer runtime.
    """

    STRICT = "strict"
    BUCKET = "bucket"
    PADDED = "padded"
    PACKED = "packed"


class HistoryStreamVisibility(StrEnum):
    """Which clean video/action history streams a query may attend."""

    FULL = "full"
    # Video queries see only video history; action queries keep full history.
    VIDEO_QUERIES_VIDEO_ONLY = "video_queries_video_only"
    # Strict history filter: all queries see video history only.
    VIDEO_ONLY = "video_only"


class VideoActionSequenceContract(StrEnum):
    """Shared sequence semantics layered on top of video/action coupling modes."""

    DEFAULT = "default"
    # Rollout-parity contract: data samples include one pre-target context frame,
    # use a single-frame condition latent before the target segment, inject
    # proprio per chunk, and restrict clean history attention to video tokens.
    ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO = "rollout_parity_single_frame_perchunk_proprio"
    # Legacy exact-prefix contract used by the original parallel-stream proprio
    # modes: data samples contain target frames only, and runtime prepends one
    # clean condition latent before the target video stream.
    LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO = "legacy_prefix_single_frame_perchunk_proprio"


class CurrentBlockCoupling(StrEnum):
    """Same-chunk video/action visibility for video/action rollout variants."""

    VIDEO_THEN_ACTION = "video_then_action"
    JOINT = "joint"
    ACTION_THEN_VIDEO = "action_then_video"
    DECOUPLED_SAME_STEP = "decoupled_same_step"
    # Joint-like one-way modes: both streams are noisy, but same-block cross-stream visibility is directional.
    VIDEO_NOISY_TO_ACTION = "video_noisy_to_action"
    ACTION_NOISY_TO_VIDEO = "action_noisy_to_video"


class VideoActionProgram(StrEnum):
    """Architecture-independent video/action conditioning program."""

    VIDEO_THEN_ACTION = "video_then_action"
    ACTION_THEN_VIDEO = "action_then_video"
    JOINT = "joint"
    DECOUPLED_SAME_STEP = "decoupled_same_step"
    VIDEO_NOISY_TO_ACTION = "video_noisy_to_action"
    ACTION_NOISY_TO_VIDEO = "action_noisy_to_video"
    GENERALIST_JOINT_DENOISING = "generalist_joint_denoising"
    # Fixed conditional programs use the same tensor and attention semantics
    # as their single-mode GJD counterparts and cannot select another GJD mode.
    FORWARD_DYNAMICS = "forward_dynamics"
    INVERSE_DYNAMICS = "inverse_dynamics"


class JointTimestepCoupling(StrEnum):
    """How joint video/action denoising synchronizes modality noise clocks."""

    # Ablation: action and video share the same actual noise amount.
    MATCH_SIGMA = "match_sigma"
    # Ablation: action and video use the same scheduler grid index/progress.
    MATCH_INDEX = "match_index"
    # Ablation: action reuses the video scheduler timestep/sigma grid directly.
    SHARED_VIDEO_SCHEDULE = "shared_video_schedule"
    # Checkpoint-validated baseline: each modality uses its own scheduler clock.
    INDEPENDENT = "independent"


# Backward-compatible export for early parallel-stream configs/code paths.
ParallelCurrentBlockCoupling = CurrentBlockCoupling


class ProprioContextMode(StrEnum):
    """How policy variants inject proprio state into transformer conditioning."""

    NONE = "none"
    # Deprecated compatibility only. Current LIBERO proprio context uses
    # PER_CHUNK_ADDITIVE hidden-state conditioning, not text-space tokens.
    TEXT_CONTEXT_TOKEN = "text_context_token"
    PER_CHUNK_ADDITIVE = "per_chunk_additive"


# Deprecated symbol aliases. They intentionally preserve class identity so old
# imports and serialized config objects remain loadable without duplicating the
# public semantic types.
ParallelContextConditionLatentSource = ContextConditionLatentSource
ParallelHistoryStreamVisibility = HistoryStreamVisibility
ParallelSequenceContract = VideoActionSequenceContract


class ParallelSequenceComponent(StrEnum):
    """Sequence components packed by the exact parallel-stream runtime."""

    VIDEO_NOISY = "video_noisy"
    VIDEO_CONDITION = "video_condition"
    ACTION_NOISY = "action_noisy"
    ACTION_CONDITION = "action_condition"


class ParallelMaskMode(StrEnum):
    """Mask profile used by the exact parallel-stream runtime."""

    LINGBOT_CHUNKED = "lingbot_chunked"


class ParallelCacheMode(StrEnum):
    """How much cache metadata/state the exact parallel-stream runtime stores locally."""

    METADATA_ONLY = "metadata_only"


class ParallelExactCacheWriteMode(StrEnum):
    """How exact-runtime video/action chunks are committed to rollout cache."""

    SINGLE_STREAM_STAGED = "single_stream_staged"
    JOINT_PACKED = "joint_packed"


class FallbackHistoryPolicy(StrEnum):
    """How exact/joint realtime rollouts expose fallback-period history to replanning."""

    INCLUDE_FALLBACK_HISTORY = "include_fallback_history"
    FREEZE_UNTIL_CLEAN_CHUNK = "freeze_until_clean_chunk"


class FallbackHistoryDecision(StrEnum):
    """How one observed frame/action was handled by fallback quarantine."""

    INCLUDED = "included"
    FALLBACK = "fallback"
    WASHOUT = "washout"


class DeadlineMissPolicy(StrEnum):
    """Fallback action to execute when a realtime plan misses its deadline."""

    HOLD_STATE = "hold_state"
    HOLD_LAST = "hold_last"
    ZERO = "zero"


class RealtimePlannerMode(StrEnum):
    """Priority policy for observed-history replans and open-loop extension."""

    HISTORY_ONLY = "history_only"
    ASYNC_BUFFER = "async_buffer"
    ASYNC_MIX = "async_mix"
    ASYNC_HISTORY_FIRST = "async_history_first"


class RealtimeEmptyPlanPolicy(StrEnum):
    """Control-loop behavior when the next required plan is unavailable."""

    FALLBACK = "fallback"
    WAIT_FOR_REPLAN = "wait_for_replan"


class RealtimeSchedulerProfile(StrEnum):
    """Named bundle of realtime scheduling defaults."""

    MANUAL = "manual"
    BLOCKING_CONTROL = "blocking_control"
    FREEZE_UNTIL_CLEAN_CHUNK = "freeze_until_clean_chunk"
    ASYNC_HISTORY_FIRST = "async_history_first"


class RealtimePlannerJob(StrEnum):
    """Planner work selected for one available scheduling slot."""

    HISTORY_REPLAN = "history_replan"
    BUFFER_EXTENSION = "buffer_extension"


class RolloutArtifactProfile(StrEnum):
    """Amount of rollout video and diagnostic state persisted per episode."""

    LEAN = "lean"
    STANDARD = "standard"
    DEBUG = "debug"


class ActionNormMethod(StrEnum):
    """Raw-to-model action normalization strategy for exact parallel-stream paths."""

    PROFILE = "profile"
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


class SampleLossWeightMode(StrEnum):
    """How runtime training loss should be scaled from per-sample metadata."""

    NONE = "none"
    VALID_ACTION_STEPS = "valid_action_steps"
    SQRT_VALID_ACTION_STEPS = "sqrt_valid_action_steps"


class TrainingComponentSelector(StrEnum):
    """Named module groups that can be frozen or made trainable."""

    ALL = "all"
    VISUAL_TOWER = "visual_tower"
    VISUAL_TOWER_FRONTEND = "visual_tower.frontend"
    VISUAL_TOWER_CORE = "visual_tower.core"
    VISUAL_TOWER_RUNTIME_BACKBONE = "visual_tower.runtime_backbone"
    VISUAL_TOWER_PROPRIO_CONTEXT_ENCODER = "visual_tower.proprio_context_encoder"
    VISUAL_TOWER_GENERALIST_MODE_CONTEXT_ENCODER = "visual_tower.generalist_mode_context_encoder"
    VISUAL_TOWER_SHARED_VIDEO_BACKBONE = "visual_tower.shared_video_backbone"
    VISUAL_TOWER_SHARED_ACTION_RUNTIME = "visual_tower.shared_action_runtime"
    VISUAL_TOWER_SHARED_RUNTIME_ADAPTERS = "visual_tower.shared_runtime_adapters"
    POLICY_VARIANT = "policy_variant"
    POLICY_VARIANT_ACTION_EXPERT = "policy_variant.action_expert"
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


class ReferenceCoreInitMode(StrEnum):
    """How shared-core reference weights should initialize the replica backbone."""

    FULL = "full"
    VIDEO_ONLY = "video_only"
    RAW_WAN_VIDEO_ONLY = "raw_wan_video_only"
    RAW_WAN_VIDEO_ONLY_WITH_BASE_NORM2 = "raw_wan_video_only_with_base_norm2"


class ExportedRuntimeActionInitMode(StrEnum):
    """How exported-runtime loads should initialize action/runtime-specific modules."""

    LOAD_FROM_CHECKPOINT = "load_from_checkpoint"
    RANDOM = "random"


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
    RAW_CHUNK_ACTION_PRED_TAIL_ALIGNED = "raw_chunk_action_pred_tail_aligned"
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
        return {
            serialize_enum_values(key): serialize_enum_values(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [serialize_enum_values(item) for item in value]
    if isinstance(value, list):
        return [serialize_enum_values(item) for item in value]
    return value
