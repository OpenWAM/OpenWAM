"""Compatibility facade for historical MoT runtime imports.

Maintained code imports each symbol from its role-specific owner.
"""

from open_wam.models.common.flow_schedule import (
    expand_scalar_timestep as expand_mot_scalar_timestep,
    explicit_sigma_euler_step as step_mot_flow_with_sigmas,
    zero_terminal_next_sigma as mot_scheduler_next_sigma,
)
from open_wam.models.common.sharded_execution import (
    checkpoint_unshard_context as _checkpoint_summon_context,
    summon_full_parameters as _summon_full_params,
    unshard_runtime_parameters as _unshard_runtime_params,
)

from .attention_cached import (
    build_mot_inference_action_attention_mask as build_mot_inference_action_attention_mask,
)
from .attention_packed import (
    build_mot_packed_coupling_attention_mask as build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile as build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask as build_packed_action_attention_mask,
)
from .attention_unpacked import (
    build_chunk_causal_video_mask as build_chunk_causal_video_mask,
    build_mot_attention_mask as build_mot_attention_mask,
)
from .cache_execution import (
    forward_action_with_video_and_action_cache as forward_action_with_video_and_action_cache,
    forward_action_with_video_cache as forward_action_with_video_cache,
    prefill_video_kv_cache as prefill_video_kv_cache,
)
from .cache_state import (
    append_mot_action_cache as append_mot_action_cache,
    move_mot_action_cache as move_mot_action_cache,
    move_mot_video_cache as move_mot_video_cache,
    rewind_mot_runtime_action_cache_to_frame as rewind_mot_runtime_action_cache_to_frame,
    trim_mot_action_cache_prefix as trim_mot_action_cache_prefix,
    trim_mot_action_cache_tail as trim_mot_action_cache_tail,
    trim_mot_video_cache_tail as trim_mot_video_cache_tail,
)
from .conditioning import resolve_mot_condition_latents as resolve_mot_condition_latents
from .dual_stream_execution import (
    _video_token_grid_for_latents as _video_token_grid_for_latents,
    forward_joint_video_action_denoise as forward_joint_video_action_denoise,
    forward_mot_packed_coupling_denoise as forward_mot_packed_coupling_denoise,
)

_COMPATIBILITY_EXPORTS = (
    expand_mot_scalar_timestep,
    step_mot_flow_with_sigmas,
    mot_scheduler_next_sigma,
    _checkpoint_summon_context,
    _summon_full_params,
    _unshard_runtime_params,
    build_chunk_causal_video_mask,
    build_mot_attention_mask,
    build_mot_inference_action_attention_mask,
    build_mot_packed_coupling_attention_mask,
    build_mot_packed_coupling_attention_profile,
    build_packed_action_attention_mask,
    forward_action_with_video_and_action_cache,
    forward_action_with_video_cache,
    prefill_video_kv_cache,
    append_mot_action_cache,
    move_mot_action_cache,
    move_mot_video_cache,
    rewind_mot_runtime_action_cache_to_frame,
    trim_mot_action_cache_prefix,
    trim_mot_action_cache_tail,
    trim_mot_video_cache_tail,
    resolve_mot_condition_latents,
    _video_token_grid_for_latents,
    forward_joint_video_action_denoise,
    forward_mot_packed_coupling_denoise,
)
