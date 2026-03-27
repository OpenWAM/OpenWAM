from __future__ import annotations

from dataclasses import dataclass


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
    joint_sampler: str = "unipc"
    # Shared per-stream CFG modes for joint rollout.
    # - `guided`: apply standard CFG combine on that stream
    # - `conditioned`: keep the conditioned prediction directly
    # - `unconditioned`: keep the unconditional prediction directly
    video_cfg_mode: str = "guided"
    action_cfg_mode: str = "conditioned"
    # Backward-compatible shorthand retained while configs migrate toward the
    # per-stream knobs above. Shared runtime code should prefer
    # `video_cfg_mode` / `action_cfg_mode`.
    joint_cfg_application: str | None = None
    # Shared cache-update policy for cache-aware joint rollout paths.
    # - `warmup_only`: prefill cache from clean reference video, then freeze during denoising
    # - `final_step`: update cache only on the last denoising step
    # - `every_step`: update cache on every denoising step
    # - `none`: never write into cache
    joint_cache_update_mode: str = "warmup_only"
    # Source used for the shared warmup pass when `joint_cache_update_mode`
    # requests cache prefill.
    # - `reference_video`: warm from clean current visual context
    # - `none`: skip warmup
    joint_cache_warmup_source: str = "reference_video"
    # Phase-aware warmup slice selection. This keeps warmup semantics generic:
    # the first rollout step and later rollout steps can choose different
    # anchors/counts without baking any benchmark-specific naming into common
    # runtime code.
    # Anchors:
    # - `start`: take frames from the start of the current clean reference
    # - `end`: take frames from the end of the current clean reference
    # - `full`: use the entire clean reference window
    joint_cache_initial_warmup_anchor: str = "start"
    joint_cache_initial_warmup_frames: int | None = 1
    joint_cache_rollout_warmup_anchor: str = "end"
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
