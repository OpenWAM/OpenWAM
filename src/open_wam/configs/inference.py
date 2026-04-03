from __future__ import annotations

from dataclasses import dataclass

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
    """Inference-layer config shared by all future action heads."""

    video_num_inference_steps: int = 25
    action_num_inference_steps: int = 50
    # Optional shared denoising count for variants such as DreamZero-like
    # register-attached models that update video and action in one joint loop.
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
    # observation. Method-2 style register-attached inference uses this to keep
    # the observed prefix anchored while future frames are generated.
    joint_observed_video_prefix_frames: int = 1
    frame_chunk_size: int = 2
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
