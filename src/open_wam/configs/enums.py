from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping, TypeAlias, TypeVar

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised in RoboTwin's Python 3.10 env.

    class StrEnum(str, Enum):
        """Python 3.10 fallback matching the string behavior of stdlib StrEnum."""

        pass


EnumT = TypeVar("EnumT", bound=StrEnum)
FieldTransform: TypeAlias = Callable[[Any], Any]
EnumFieldMap: TypeAlias = Mapping[str, type[EnumT]]
TransformFieldMap: TypeAlias = Mapping[str, FieldTransform]


class ActionDecoderName(StrEnum):
    """Final action-decoder family selected by experiment config."""

    MLP = "mlp_decoder"
    REGISTER = "register_decoder"
    DECODED_FEATURE = "decoded_feature_decoder"
    VIDEO_CONDITIONED = "video_conditioned_action_decoder"
    VPP = "vpp_decoder"
    LINGBOT_PARALLEL = "lingbot_parallel_decoder"
    MOT = "mot_decoder"
    VIDEO_ONLY = "video_only_decoder"


class ActionChunkAnchorMode(StrEnum):
    """How an action chunk is anchored relative to the local video window."""

    FUTURE_ONLY = "future_only"
    CURRENT_PLUS_FUTURE = "current_plus_future"


class ActionExpertInitMode(StrEnum):
    """How a reusable action expert should initialize from the shared video core."""

    RANDOM = "random"
    VIDEO_WEIGHT_COPY = "video_weight_copy"
    VIDEO_WEIGHT_INTERPOLATE = "video_weight_interpolate"


class VideoConditionInputSpace(StrEnum):
    """Which video-space family a method-4 action decoder should treat as input."""

    VIDEO_LATENT = "video_latent"
    RGB_VIDEO = "rgb_video"


class VideoConditionSource(StrEnum):
    """Source used to build method-4 local video-conditioning windows."""

    LOCAL_WINDOW = "local_window"
    GENERATED_FUTURE = "generated_future"


class VideoConditionTrainMode(StrEnum):
    """How a method-4 video-conditioned decoder is trained."""

    ROLLOUT_WINDOW_DIFFUSION = "rollout_window_diffusion"
    CURRENT_FRAME_REGRESSION = "current_frame_regression"


class DataSplit(StrEnum):
    """Dataset split used by train/eval loaders."""

    TRAIN = "train"
    VAL = "val"


class DatasetPreflightKind(StrEnum):
    """Filesystem preflight check used by launch dataset profiles."""

    LOCAL_LATENT = "local_latent"


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
    RANDOM_SUBWINDOW = "random_subwindow"
    CONTEXTUAL_SUBWINDOW = "contextual_subwindow"
    ALIGNED_SUBWINDOW = "aligned_subwindow"
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

    POST_LATENT = "post_latent"
    POST_DECODED = "post_decoded"
    VIDEO_SEQUENCE_POLICY = "video_sequence_policy"
    CAUSAL_VIDEO_PREDICTION = "causal_video_prediction"
    MOT = "mot"
    # Obsolete traditional Method 2. Kept for loading historical configs only;
    # pipeline construction raises an explicit error.
    REGISTER_ATTACHED = "register_attached"
    PARALLEL_STREAM = "parallel_stream"


class MoTRuntimeMode(StrEnum):
    """Execution mode for the MoT policy family.

    `VIDEO_PREFILL_ACTION_DENOISE` keeps the video branch clean during training
    and only denoises actions against a cached video prefix — useful as a
    stage-0 / action-only posttrain on top of a frozen video backbone.

    `NON_JOINT_TWO_STREAM` aligns with method-1 `lingbot_exact` (non-joint):
    both streams get a history-clean / current-noisy split and are denoised
    simultaneously, but the mask disallows same-chunk noisy-to-noisy cross-
    stream attention (video blocks are even, action blocks are odd, so
    `kv_block == q_block` only fires within the same stream). Combined with
    `video_can_attend_action=false` this gives the MoT analogue of method-1
    non-joint (minus the unavoidable "video sees earlier clean action" delta).

    `JOINT_DENOISE` aligns with method-1 `lingbot_exact_action_conditioned`
    (joint): both streams noisy, and the mask allows same-chunk noisy-to-noisy
    cross-stream attention (subject to `video_can_attend_action`).
    """

    VIDEO_PREFILL_ACTION_DENOISE = "video_prefill_action_denoise"
    NON_JOINT_TWO_STREAM = "non_joint_two_stream"
    JOINT_DENOISE = "joint_denoise"


class MoTActionExpertInitMode(StrEnum):
    """How the MoT action expert should initialize from the video expert."""

    RANDOM = "random"
    VIDEO_WEIGHT_COPY = "video_weight_copy"
    VIDEO_WEIGHT_INTERPOLATE = "video_weight_interpolate"


class MoTConditionMode(StrEnum):
    """Which video branch the MoT action expert conditions on."""

    FIRST_FRAME = "first_frame"
    FULL_VIDEO = "full_video"
    TEACHER_FORCING_COND_VIDEO = "teacher_forcing_cond_video"


class MoTPreset(StrEnum):
    """High-level FastWAM-style preset families for MoT policy defaults."""

    FASTWAM = "fastwam"
    FASTWAM_JOINT = "fastwam_joint"
    FASTWAM_IDM = "fastwam_idm"
    FASTWAM_NON_JOINT = "fastwam_non_joint"


class VisualReadoutSourceFamily(StrEnum):
    """Which shared visual representation a post-visual policy reads from."""

    FINAL_CORE_TOKENS = "final_core_tokens"
    CORE_LAYER_TOKENS = "core_layer_tokens"
    CORE_MULTI_LAYER_TOKENS = "core_multi_layer_tokens"
    GENERATED_FUTURE_TOKENS = "generated_future_tokens"
    DIFFUSION_FEATURE_TOKENS = "diffusion_feature_tokens"


class VisualReadoutFusionMode(StrEnum):
    """How multi-layer shared visual readouts are fused."""

    NONE = "none"
    MEAN = "mean"
    LEARNED_WEIGHTED_SUM = "learned_weighted_sum"
    CONCAT_PROJECT = "concat_project"


class GoalConditioningAdapterFamily(StrEnum):
    """Goal/language conditioning adapter family for sequence decoders."""

    PASSTHROUGH = "passthrough"
    MEAN_POOL = "mean_pool"


class StateSequenceAdapterFamily(StrEnum):
    """State/proprio adapter family for sequence decoders."""

    IDENTITY = "identity"
    LINEAR = "linear"


class TemporalCompressionAdapterFamily(StrEnum):
    """Temporal/token compression family for sequence decoders."""

    IDENTITY = "identity"
    FRAME_MEAN_POOL = "frame_mean_pool"
    TEMPORAL_LATENT_RESAMPLER_3D = "temporal_latent_resampler_3d"
    VIDEO_FORMER_3D = "video_former_3d"


class SequenceDenoiserFamily(StrEnum):
    """Sequence-denoiser architecture for sequence-native action decoders."""

    GENERIC_TRANSFORMER = "generic_transformer"
    FILM_DIFFUSION_TRANSFORMER = "film_diffusion_transformer"


class ActionGenerationBackendFamily(StrEnum):
    """Action-generation backend family for sequence-native decoders."""

    EDM_DIFFUSION = "edm_diffusion"


class DiffusionNoiseSchedule(StrEnum):
    """Noise schedule family for diffusion-based action decoders."""

    EXPONENTIAL = "exponential"
    KARRAS = "karras"


class DiffusionSampler(StrEnum):
    """Sampling solver family for diffusion-based action decoders."""

    DDIM = "ddim"
    EULER = "euler"


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


class VisualStateSource(StrEnum):
    """Which visual-state family a policy variant should consume."""

    CORE_TOKENS = "core_tokens"
    DENOISED_VIDEO_TOKENS = "denoised_video_tokens"


class VisualReadoutSourceFamily(StrEnum):
    """Which shared visual readout family a post-visual variant should consume."""

    FINAL_CORE_TOKENS = "final_core_tokens"
    CORE_LAYER_TOKENS = "core_layer_tokens"
    CORE_MULTI_LAYER_TOKENS = "core_multi_layer_tokens"
    DIFFUSION_FEATURE_TOKENS = "diffusion_feature_tokens"


class VisualReadoutFusionMode(StrEnum):
    """How multiple shared-core readouts should be fused."""

    NONE = "none"
    CONCAT_PROJECT = "concat_project"
    LEARNED_WEIGHTED_SUM = "learned_weighted_sum"
    MEAN = "mean"


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
    LINGBOT_EXACT_ACTION_CONDITIONED = "lingbot_exact_action_conditioned"
    # Current observation + text plus hidden-state proprio -> action chunk, without exact history cache.
    CURRENT_FRAME_ACTION_CHUNK = "current_frame_action_chunk"
    # FastWAM-style first-frame video/action training with action-only rollout.
    FASTWAM_FIRST_FRAME = "fastwam_first_frame"


class ParallelStreamVariantProfile(StrEnum):
    """Named Method-1 variant profile layered on the exact parallel runtime."""

    STANDARD = "standard"
    GENERALIST_JOINT_DENOISING = "generalist_joint_denoising"


class JointDenoiseTrainingMode(StrEnum):
    """Per-segment training mode for generalist joint video/action denoising."""

    JOINT = "joint"
    ACTION_CONDITIONED_VIDEO = "action_conditioned_video"
    VIDEO_CONDITIONED_ACTION = "video_conditioned_action"


class ParallelActionConditionSource(StrEnum):
    """Which action stream should be exposed to video denoising."""

    NOISY_ACTION = "noisy_action"
    CLEAN_ACTION = "clean_action"


class ParallelActionAttentionScope(StrEnum):
    """How broadly video tokens may attend to action tokens."""

    FULL = "full"
    BLOCK_LOCAL = "block_local"


class ParallelContextConditionLatentSource(StrEnum):
    """Which latent source supplies clean pre-target video context frames."""

    VIDEO_LATENTS = "video_latents"
    SINGLE_FRAME_CONDITION_LATENT = "single_frame_condition_latent"


class ParallelHistoryStreamVisibility(StrEnum):
    """Which clean history streams exact Method-1 queries may attend."""

    FULL = "full"
    # Backward-compatible behavior of `preserve_video_pretrain_history=true`:
    # video queries see only video history, action queries keep full history.
    VIDEO_QUERIES_VIDEO_ONLY = "video_queries_video_only"
    # Strict history filter: all queries see video history only.
    VIDEO_ONLY = "video_only"


class ParallelSequenceContract(StrEnum):
    """Shared sequence semantics layered on top of video/action coupling modes."""

    DEFAULT = "default"
    # Rollout-parity contract: data samples include one pre-target context frame,
    # use a single-frame condition latent before the target segment, inject
    # proprio per chunk, and restrict clean history attention to video tokens.
    ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO = "rollout_parity_single_frame_perchunk_proprio"
    # Legacy exact-prefix contract used by the original parallel-proprio M1
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


class JointTimestepCoupling(StrEnum):
    """How joint video/action denoising synchronizes modality noise clocks."""

    # Canonical joint denoising: action and video share the same actual noise amount.
    MATCH_SIGMA = "match_sigma"
    # Ablation: action and video use the same scheduler grid index/progress.
    MATCH_INDEX = "match_index"
    # Ablation: action reuses the video scheduler timestep/sigma grid directly.
    SHARED_VIDEO_SCHEDULE = "shared_video_schedule"
    # Legacy/control: action and video sample or step their clocks independently.
    INDEPENDENT = "independent"


# Backward-compatible export for early Method-1 configs/code paths.
ParallelCurrentBlockCoupling = CurrentBlockCoupling


class ProprioContextMode(StrEnum):
    """How policy variants inject proprio state into transformer conditioning."""

    NONE = "none"
    # Deprecated compatibility only. Current LIBERO proprio context uses
    # PER_CHUNK_ADDITIVE hidden-state conditioning, not text-space tokens.
    TEXT_CONTEXT_TOKEN = "text_context_token"
    PER_CHUNK_ADDITIVE = "per_chunk_additive"


class MoTGeneralistTrainingMode(StrEnum):
    """Per-segment training regime for the M5 generalist joint-denoise variant.

    Under a fixed JOINT coupling, each training segment samples one regime.
    ``joint`` denoises both modalities with the same packed clean-history
    condition slots used by plain M5 joint rollout. The conditional modes place
    the clean modality into its noisy slot, force its per-frame timesteps to 0,
    and mask its loss. Clean history slots remain real context; attention
    windowing and loss masks, not zeroed context slots, define the local
    conditional objective.
    """

    JOINT = "joint"
    ACTION_CONDITIONED_VIDEO = "action_conditioned_video"
    VIDEO_CONDITIONED_ACTION = "video_conditioned_action"


class GeneralistTrainingParadigm(StrEnum):
    """High-level data/objective mixture used by generalist video-action methods."""

    DEMO_ONLY = "demo_only"
    MIXED_DYNAMICS = "mixed_dynamics"


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


class ParallelExactCacheWriteMode(StrEnum):
    """How exact-runtime video/action chunks are committed to rollout cache."""

    SINGLE_STREAM_STAGED = "single_stream_staged"
    JOINT_PACKED = "joint_packed"


class FallbackHistoryPolicy(StrEnum):
    """How exact/joint realtime rollouts expose fallback-period history to replanning."""

    INCLUDE_FALLBACK_HISTORY = "include_fallback_history"
    FREEZE_UNTIL_CLEAN_CHUNK = "freeze_until_clean_chunk"


class DeadlineMissPolicy(StrEnum):
    """Fallback action to execute when a realtime plan misses its deadline."""

    HOLD_STATE = "hold_state"
    HOLD_LAST = "hold_last"
    ZERO = "zero"


class ActionNormMethod(StrEnum):
    """Raw-to-model action normalization strategy for exact method-1 paths."""

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
    VISUAL_TOWER_DECODER = "visual_tower.decoder"
    POLICY_VARIANT = "policy_variant"
    POLICY_VARIANT_ACTION_EXPERT = "policy_variant.action_expert"
    ACTION_DECODER = "action_decoder"
    ACTION_DECODER_ADAPTERS = "action_decoder.adapters"


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
    DECODER_PREDICTED_LOCAL_FUTURE_LATENTS = "decoder_predicted_local_future_latents"
    POLICY_PREDICTED_LATENTS = "policy_predicted_latents"
    POLICY_PREDICTED_VIDEO_LATENTS = "policy_predicted_video_latents"
    POLICY_PREDICTED_LOCAL_FUTURE_LATENTS = "policy_predicted_local_future_latents"


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
