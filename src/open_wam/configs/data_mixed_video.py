"""Configuration contracts for manifest-backed mixed-video training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .data_consortium import ConsortiumChannelMappingConfig
from .data_contracts import (
    ActionMappingConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    DataConfig,
    SampleConstructionConfig,
    ViewLayoutConfig,
)
from .enums import (
    DataSplit,
    LatentWindowProfile,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoMissingStreamPolicy,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoWeightMode,
    ReplayStatusPolicy,
    WindowSamplingMode,
    coerce_fields,
)

__all__ = [
    "MixedVideoDataConfig",
    "MixedVideoResizeBinConfig",
    "MixedVideoSourceConfig",
    "MixedVideoViewCombinationConfig",
    "default_mixed_video_resize_bins",
]


@dataclass(frozen=True)
class MixedVideoSourceConfig:
    """One manifest-backed video source in a mixed video-only pretraining run."""

    source_id: str
    manifest_csv: str
    repo_id: str | None = None
    local_root: str | None = None
    latent_root: str | None = None
    source_format: MixedVideoSourceFormat = MixedVideoSourceFormat.RGB
    latent_key: str = "video_latents"
    enabled: bool = True
    source_group: str | None = None
    include_streams: tuple[str, ...] = ()
    channel_mappings: tuple[ConsortiumChannelMappingConfig, ...] = ()
    sampling_weight: float | None = None

    def __post_init__(self) -> None:
        coerce_fields(self, enum_fields={"source_format": MixedVideoSourceFormat})
        if not self.source_id:
            raise ValueError("MixedVideoSourceConfig requires a non-empty `source_id`.")
        if not self.manifest_csv:
            raise ValueError("MixedVideoSourceConfig requires `manifest_csv`.")
        if not self.latent_key:
            raise ValueError("MixedVideoSourceConfig requires a non-empty `latent_key`.")
        if self.sampling_weight is not None:
            if not math.isfinite(float(self.sampling_weight)) or float(self.sampling_weight) <= 0.0:
                raise ValueError("`sampling_weight` must be finite and positive when set.")


@dataclass(frozen=True)
class MixedVideoResizeBinConfig:
    """One aspect-ratio bin used to standardize mixed-video VAE inputs."""

    name: str
    aspect_width: int
    aspect_height: int
    target_height: int
    target_width: int
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MixedVideoResizeBinConfig requires a non-empty `name`.")
        for field_name in ("aspect_width", "aspect_height", "target_height", "target_width"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"`{field_name}` must be positive for mixed-video resize bins.")
            object.__setattr__(self, field_name, value)
        if self.max_pixels is not None:
            max_pixels = int(self.max_pixels)
            if max_pixels <= 0:
                raise ValueError("`max_pixels` must be positive when set for mixed-video resize bins.")
            object.__setattr__(self, "max_pixels", max_pixels)

    @property
    def aspect_ratio(self) -> float:
        return float(self.aspect_width) / float(self.aspect_height)


@dataclass(frozen=True)
class MixedVideoViewCombinationConfig:
    """Ordered latent slots assembled into one training sample."""

    name: str
    slots: tuple[str, ...]
    sampling_weight: float = 1.0
    source_ids: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "slots", tuple(str(slot) for slot in self.slots))
        object.__setattr__(self, "sampling_weight", float(self.sampling_weight))
        object.__setattr__(self, "source_ids", tuple(str(source_id) for source_id in self.source_ids))
        object.__setattr__(self, "enabled", bool(self.enabled))
        if not self.name:
            raise ValueError("MixedVideoViewCombinationConfig requires a non-empty `name`.")
        if not 1 <= len(self.slots) <= 4:
            raise ValueError(
                "Mixed-video latent view combinations support 1 to 4 slots, "
                f"got {len(self.slots)} for {self.name!r}."
            )
        if len(set(self.slots)) != len(self.slots):
            raise ValueError(f"Mixed-video latent view combination {self.name!r} contains duplicate slots.")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise ValueError("Mixed-video latent view combination `sampling_weight` must be finite and positive.")


def default_mixed_video_resize_bins() -> tuple[MixedVideoResizeBinConfig, ...]:
    """Default VAE-friendly bins for common web/video aspect ratios."""

    return (
        MixedVideoResizeBinConfig(
            name="square_128",
            aspect_width=1,
            aspect_height=1,
            target_height=128,
            target_width=128,
            max_pixels=128 * 128,
        ),
        MixedVideoResizeBinConfig(
            name="square_256",
            aspect_width=1,
            aspect_height=1,
            target_height=256,
            target_width=256,
        ),
        MixedVideoResizeBinConfig(
            name="four_three_352x256",
            aspect_width=4,
            aspect_height=3,
            target_height=256,
            target_width=352,
        ),
        MixedVideoResizeBinConfig(
            name="sixteen_nine_352x192",
            aspect_width=16,
            aspect_height=9,
            target_height=192,
            target_width=352,
        ),
    )


@dataclass(frozen=True)
class MixedVideoDataConfig(DataConfig):
    """Manifest-first multi-source RGB video config for video-only pretraining.

    This mirrors the nmotions pipeline contract at the data boundary: manifests
    enumerate video streams, the adapter decodes every source into one common
    target size, and the rest of OpenWAM only sees the standard `views` batch.
    """

    dataset_name: str = "mixed_video"
    dataset_type: str = "mixed_video"
    repo_id: str | None = None
    local_root: str | None = None
    val_local_root: str | None = None
    empty_text_embedding_path: str | None = None
    latent_root: str | None = None
    latent_subdir: str = "latents"
    latent_window_profile: LatentWindowProfile = LatentWindowProfile.EXACT_CHUNKED_WINDOW
    split: DataSplit = DataSplit.TRAIN
    cache_dir: str | None = None
    camera_names: tuple[str, ...] = ("observation.images.slot0",)
    latent_camera_names: tuple[str, ...] = ("observation.images.slot0",)
    canonical_height: int = 128
    canonical_width: int = 128
    view_layout: tuple[ViewLayoutConfig, ...] = field(
        default_factory=lambda: (
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=128,
                width=128,
            ),
        )
    )
    num_frames: int = 16
    frame_stride: int = 1
    sample_stride: int = 1
    episode_cache_size: int = 2
    train_fraction: float = 0.98
    split_seed: int = 0
    max_train_episodes: int | None = None
    max_val_episodes: int | None = None
    replay_status_path: str | None = None
    val_replay_status_path: str | None = None
    replay_status_policy: ReplayStatusPolicy = ReplayStatusPolicy.INCLUDE_ALL
    require_replay_status: bool = False
    val_replay_status_policy: ReplayStatusPolicy | None = None
    val_require_replay_status: bool | None = None
    train_batch_size: int = 2
    val_batch_size: int = 2
    num_workers: int = 0
    action_schema: ActionSchemaConfig = field(
        default_factory=lambda: ActionSchemaConfig(
            action_dim=1,
            action_horizon=0,
            state_dim=1,
            state_horizon=0,
        )
    )
    action_target: ActionTargetConfig = field(default_factory=ActionTargetConfig)
    action_mapping: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    sample_construction: SampleConstructionConfig = field(
        default_factory=lambda: SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=16,
            action_horizon=0,
            state_horizon=0,
        )
    )
    video_sources: tuple[MixedVideoSourceConfig, ...] = ()
    latent_encoding_mode: MixedVideoLatentEncodingMode = MixedVideoLatentEncodingMode.CANONICAL
    latent_view_combinations: tuple[MixedVideoViewCombinationConfig, ...] = field(default_factory=tuple)
    decode_size_mode: MixedVideoDecodeSizeMode = MixedVideoDecodeSizeMode.FIXED
    decode_resize_bins: tuple[MixedVideoResizeBinConfig, ...] = field(default_factory=default_mixed_video_resize_bins)
    decode_height: int = 128
    decode_width: int = 128
    decode_fit_mode: MixedVideoFrameFitMode = MixedVideoFrameFitMode.LETTERBOX_PAD
    decode_center_crop: bool = False
    decode_allow_upscale: bool = True
    target_observation_fps: float | None = 15.0
    missing_observation_fps: float = 30.0
    missing_stream_policy: MixedVideoMissingStreamPolicy = MixedVideoMissingStreamPolicy.ZERO_FILL
    random_mode: MixedVideoRandomMode = MixedVideoRandomMode.WITHIN_SOURCE
    weight_mode: MixedVideoWeightMode = MixedVideoWeightMode.PROPORTIONAL_TO_SIZE
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        coerce_fields(
            self,
            enum_fields={
                "decode_size_mode": MixedVideoDecodeSizeMode,
                "decode_fit_mode": MixedVideoFrameFitMode,
                "latent_encoding_mode": MixedVideoLatentEncodingMode,
                "missing_stream_policy": MixedVideoMissingStreamPolicy,
                "random_mode": MixedVideoRandomMode,
                "weight_mode": MixedVideoWeightMode,
            },
        )
        object.__setattr__(
            self,
            "decode_resize_bins",
            tuple(
                bin_config
                if isinstance(bin_config, MixedVideoResizeBinConfig)
                else MixedVideoResizeBinConfig(**bin_config)
                for bin_config in self.decode_resize_bins
            ),
        )
        object.__setattr__(
            self,
            "latent_view_combinations",
            tuple(
                combination
                if isinstance(combination, MixedVideoViewCombinationConfig)
                else MixedVideoViewCombinationConfig(**combination)
                for combination in self.latent_view_combinations
            ),
        )
        combination_names = [combination.name for combination in self.latent_view_combinations if combination.enabled]
        if len(set(combination_names)) != len(combination_names):
            raise ValueError("Enabled mixed-video latent view combination names must be unique.")
        if self.decode_center_crop and self.decode_fit_mode != MixedVideoFrameFitMode.CENTER_CROP:
            raise ValueError(
                "`decode_center_crop=True` is a legacy alias for `decode_fit_mode=center_crop`; "
                "set `decode_fit_mode: center_crop` or remove `decode_center_crop`."
            )
        if self.decode_fit_mode == MixedVideoFrameFitMode.CENTER_CROP and not self.decode_allow_upscale:
            raise ValueError(
                "`decode_fit_mode=center_crop` requires `decode_allow_upscale=True` so decoded frames always match "
                "the configured target canvas. Use `decode_fit_mode=letterbox_pad` to preserve small inputs without "
                "upscaling."
            )
        if self.decode_height <= 0 or self.decode_width <= 0:
            raise ValueError("`decode_height` and `decode_width` must be positive.")
        if self.target_observation_fps is not None and self.target_observation_fps <= 0:
            raise ValueError("`target_observation_fps` must be positive or null to disable FPS normalization.")
        if self.missing_observation_fps <= 0:
            raise ValueError("`missing_observation_fps` must be positive.")
        if self.decode_size_mode == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS and not self.decode_resize_bins:
            raise ValueError("`decode_size_mode=aspect_ratio_bins` requires at least one `decode_resize_bins` entry.")
        if self.decode_size_mode == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS and (
            self.train_batch_size != 1 or self.val_batch_size != 1
        ):
            raise ValueError(
                "`decode_size_mode=aspect_ratio_bins` currently requires train_batch_size=1 and val_batch_size=1 "
                "because samples can have different decoded heights/widths."
            )
        if not self.video_sources:
            raise ValueError("`mixed_video` requires at least one `video_sources` entry.")
        source_ids = [source.source_id for source in self.video_sources if source.enabled]
        if not source_ids:
            raise ValueError("`mixed_video` requires at least one enabled video source.")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Enabled mixed-video `source_id` values must be unique.")
        if self.action_schema.action_horizon != 0 or self.action_schema.state_horizon != 0:
            raise ValueError("`mixed_video` is video-only; set action_horizon=0 and state_horizon=0.")
        if self.sample_construction.action_horizon != 0 or self.sample_construction.state_horizon != 0:
            raise ValueError("`mixed_video` sample_construction must use zero action/state horizons.")
        if self.sample_construction.num_frames != self.num_frames:
            raise ValueError("`mixed_video` requires data.num_frames and sample_construction.num_frames to match.")
        if self.sample_construction.frame_stride != self.frame_stride:
            raise ValueError("`mixed_video` requires data.frame_stride and sample_construction.frame_stride to match.")
