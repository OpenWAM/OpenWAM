from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .coercion import coerce_enum, coerce_optional_enum
from .enums import (
    CFGMode,
    CacheUpdateMode,
    CacheWarmupSource,
    JointCfgApplication,
    JointSampler,
    WarmupAnchor,
    coerce_fields,
)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-layer config shared by every policy architecture."""

    video_num_inference_steps: int = 25
    action_num_inference_steps: int = 50
    # Optional shared denoising count for variants that update video and action
    # in one joint loop.
    joint_num_inference_steps: int | None = None
    # `flow_match`: first-order LingBot-style flow step
    # `unipc`: DreamZero-style multistep sampler for joint video/action rollout
    joint_sampler: JointSampler = JointSampler.UNIPC
    # Shared per-stream CFG modes for joint rollout.
    # - `guided`: apply standard CFG combine on that stream
    # - `conditioned`: keep the conditioned prediction directly
    # - `unconditioned`: keep the unconditional prediction directly
    video_cfg_mode: CFGMode = CFGMode.GUIDED
    action_cfg_mode: CFGMode = CFGMode.CONDITIONED
    # Backward-compatible shorthand retained while configs migrate toward the
    # per-stream knobs above. Shared runtime code should prefer
    # `video_cfg_mode` / `action_cfg_mode`.
    joint_cfg_application: JointCfgApplication | None = None
    # Shared cache-update policy for cache-aware joint rollout paths.
    # - `warmup_only`: prefill cache from clean reference video, then freeze during denoising
    # - `final_step`: update cache only on the last denoising step
    # - `every_step`: update cache on every denoising step
    # - `none`: never write into cache
    joint_cache_update_mode: CacheUpdateMode = CacheUpdateMode.WARMUP_ONLY
    # Source used for the shared warmup pass when `joint_cache_update_mode`
    # requests cache prefill.
    # - `reference_video`: warm from clean current visual context
    # - `none`: skip warmup
    joint_cache_warmup_source: CacheWarmupSource = CacheWarmupSource.REFERENCE_VIDEO
    # Phase-aware warmup slice selection. This keeps warmup semantics generic:
    # the first rollout step and later rollout steps can choose different
    # anchors/counts without baking any benchmark-specific naming into common
    # runtime code.
    # Anchors:
    # - `start`: take frames from the start of the current clean reference
    # - `end`: take frames from the end of the current clean reference
    # - `full`: use the entire clean reference window
    joint_cache_initial_warmup_anchor: WarmupAnchor = WarmupAnchor.START
    joint_cache_initial_warmup_frames: int | None = 1
    joint_cache_rollout_warmup_anchor: WarmupAnchor = WarmupAnchor.END
    # `None` means "use the current rollout block/chunk size".
    joint_cache_rollout_warmup_frames: int | None = None
    # Number of observed video frames that should stay fixed when a joint
    # video/action rollout variant denoises a window from the current visual
    # observation. Joint inference uses this to keep the observed prefix
    # anchored while future frames are generated.
    joint_observed_video_prefix_frames: int = 1
    frame_chunk_size: int = 2
    # Logical temporal-block distance visible during recurrent inference.
    # VTA-compatible interleaved and causal-video programs use two block ids
    # per model chunk.
    attention_window_size: int = 30
    use_cache: bool = True
    guidance_scale: float = 1.0
    action_guidance_scale: float = 1.0
    video_exec_step: int = -1
    # DreamZero-style DiT execution schedule.
    # - When `joint_dynamic_cache_schedule` is false, the runtime uses the
    #   fixed 16-step mask selected by `joint_num_dit_steps`.
    # - When true, the runtime falls back to similarity-based prediction reuse.
    joint_dynamic_cache_schedule: bool = False
    joint_num_dit_steps: int | None = 8
    joint_dit_step_mask: tuple[bool, ...] | None = None
    # DreamZero-style DIT reuse: skip selected denoising steps when recent
    # video flow predictions are highly aligned, and reuse the latest flow
    # estimate instead of rerunning the transformer.
    joint_enable_prediction_reuse: bool = False
    joint_prediction_reuse_thresholds: tuple[float, ...] = (0.95, 0.93)
    joint_prediction_reuse_countdowns: tuple[int, ...] = (4, 2)

    def __post_init__(self) -> None:
        coerce_fields(
            self,
            enum_fields={
                "joint_sampler": JointSampler,
                "video_cfg_mode": CFGMode,
                "action_cfg_mode": CFGMode,
                "joint_cache_update_mode": CacheUpdateMode,
                "joint_cache_warmup_source": CacheWarmupSource,
                "joint_cache_initial_warmup_anchor": WarmupAnchor,
                "joint_cache_rollout_warmup_anchor": WarmupAnchor,
            },
            optional_enum_fields={
                "joint_cfg_application": JointCfgApplication,
            },
        )
        if int(self.attention_window_size) <= 0:
            raise ValueError(
                "Inference attention_window_size must be positive, "
                f"got {self.attention_window_size}."
            )


def parse_inference_config(raw_value: Mapping[str, Any] | None) -> InferenceConfig:
    """Parse shared denoising, CFG, and cache controls."""

    raw = raw_value or {}
    joint_cfg_application = coerce_optional_enum(
        JointCfgApplication,
        raw.get("joint_cfg_application"),
    )
    if joint_cfg_application == JointCfgApplication.VIDEO_ONLY:
        video_cfg_mode = CFGMode.GUIDED
        action_cfg_mode = CFGMode.CONDITIONED
    elif joint_cfg_application == JointCfgApplication.JOINT:
        video_cfg_mode = CFGMode.GUIDED
        action_cfg_mode = CFGMode.GUIDED
    else:
        video_cfg_mode = coerce_enum(
            CFGMode,
            raw.get("video_cfg_mode", "guided"),
        )
        action_cfg_mode = coerce_enum(
            CFGMode,
            raw.get("action_cfg_mode", "conditioned"),
        )

    joint_cache_warmup_source = raw.get("joint_cache_warmup_source")
    if joint_cache_warmup_source == "dreamzero_reference_block":
        resolved_warmup_source = CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source == "reference_video":
        resolved_warmup_source = CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_initial_warmup_anchor", "full"),
        )
        initial_warmup_frames = raw.get("joint_cache_initial_warmup_frames")
        rollout_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_rollout_warmup_anchor", "full"),
        )
        rollout_warmup_frames = raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source in {None, "none"}:
        resolved_warmup_source = (
            CacheWarmupSource.REFERENCE_VIDEO
            if joint_cache_warmup_source is None
            else CacheWarmupSource.NONE
        )
        initial_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = raw.get("joint_cache_rollout_warmup_frames")
    else:
        resolved_warmup_source = coerce_enum(
            CacheWarmupSource,
            joint_cache_warmup_source,
        )
        initial_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = coerce_enum(
            WarmupAnchor,
            raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = raw.get("joint_cache_rollout_warmup_frames")

    return InferenceConfig(
        video_num_inference_steps=raw.get("video_num_inference_steps", 25),
        action_num_inference_steps=raw.get("action_num_inference_steps", 50),
        joint_num_inference_steps=raw.get("joint_num_inference_steps"),
        joint_sampler=coerce_enum(
            JointSampler,
            raw.get("joint_sampler", "unipc"),
        ),
        video_cfg_mode=video_cfg_mode,
        action_cfg_mode=action_cfg_mode,
        joint_cfg_application=joint_cfg_application,
        joint_cache_update_mode=coerce_enum(
            CacheUpdateMode,
            raw.get("joint_cache_update_mode", "warmup_only"),
        ),
        joint_cache_warmup_source=resolved_warmup_source,
        joint_cache_initial_warmup_anchor=initial_warmup_anchor,
        joint_cache_initial_warmup_frames=initial_warmup_frames,
        joint_cache_rollout_warmup_anchor=rollout_warmup_anchor,
        joint_cache_rollout_warmup_frames=rollout_warmup_frames,
        joint_observed_video_prefix_frames=raw.get(
            "joint_observed_video_prefix_frames",
            1,
        ),
        frame_chunk_size=raw.get("frame_chunk_size", 2),
        attention_window_size=raw.get("attention_window_size", 30),
        use_cache=raw.get("use_cache", True),
        guidance_scale=raw.get("guidance_scale", 1.0),
        action_guidance_scale=raw.get("action_guidance_scale", 1.0),
        video_exec_step=raw.get("video_exec_step", -1),
        joint_dynamic_cache_schedule=raw.get(
            "joint_dynamic_cache_schedule",
            False,
        ),
        joint_num_dit_steps=raw.get("joint_num_dit_steps", 8),
        joint_dit_step_mask=(
            tuple(bool(value) for value in raw["joint_dit_step_mask"])
            if raw.get("joint_dit_step_mask") is not None
            else None
        ),
        joint_enable_prediction_reuse=raw.get(
            "joint_enable_prediction_reuse",
            False,
        ),
        joint_prediction_reuse_thresholds=tuple(
            float(value)
            for value in raw.get(
                "joint_prediction_reuse_thresholds",
                (0.95, 0.93),
            )
        ),
        joint_prediction_reuse_countdowns=tuple(
            int(value)
            for value in raw.get(
                "joint_prediction_reuse_countdowns",
                (4, 2),
            )
        ),
    )
