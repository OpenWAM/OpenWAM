"""Compatibility facade for historical DualExpert runtime imports.

Maintained code imports each symbol from its role-specific owner.
"""

from open_wam.models.common.flow_schedule import (
    expand_scalar_timestep as expand_dual_expert_scalar_timestep,
)
from open_wam.models.common.flow_schedule import (
    explicit_sigma_euler_step as step_dual_expert_flow_with_sigmas,
)
from open_wam.models.common.flow_schedule import (
    zero_terminal_next_sigma as dual_expert_scheduler_next_sigma,
)
from open_wam.models.common.sharded_execution import (
    checkpoint_unshard_context as _checkpoint_summon_context,
)
from open_wam.models.common.sharded_execution import (
    summon_full_parameters as _summon_full_params,
)
from open_wam.models.common.sharded_execution import (
    unshard_runtime_parameters as _unshard_runtime_params,
)

from .attention_cached import (
    build_dual_expert_inference_action_attention_mask as build_dual_expert_inference_action_attention_mask,
)
from .attention_packed import (
    build_dual_expert_packed_coupling_attention_mask as build_dual_expert_packed_coupling_attention_mask,
)
from .attention_packed import (
    build_dual_expert_packed_coupling_attention_profile as build_dual_expert_packed_coupling_attention_profile,
)
from .attention_packed import (
    build_packed_action_attention_mask as build_packed_action_attention_mask,
)
from .attention_unpacked import (
    build_chunk_causal_video_mask as build_chunk_causal_video_mask,
)
from .attention_unpacked import (
    build_dual_expert_attention_mask as build_dual_expert_attention_mask,
)
from .cache_execution import (
    forward_action_with_video_and_action_cache as forward_action_with_video_and_action_cache,
)
from .cache_execution import (
    forward_action_with_video_cache as forward_action_with_video_cache,
)
from .cache_execution import (
    prefill_video_kv_cache as prefill_video_kv_cache,
)
from .cache_state import (
    append_dual_expert_action_cache as append_dual_expert_action_cache,
)
from .cache_state import (
    move_dual_expert_action_cache as move_dual_expert_action_cache,
)
from .cache_state import (
    move_dual_expert_video_cache as move_dual_expert_video_cache,
)
from .cache_state import (
    rewind_dual_expert_runtime_action_cache_to_frame as rewind_dual_expert_runtime_action_cache_to_frame,
)
from .cache_state import (
    trim_dual_expert_action_cache_prefix as trim_dual_expert_action_cache_prefix,
)
from .cache_state import (
    trim_dual_expert_action_cache_tail as trim_dual_expert_action_cache_tail,
)
from .cache_state import (
    trim_dual_expert_video_cache_tail as trim_dual_expert_video_cache_tail,
)
from .conditioning import (
    resolve_dual_expert_condition_latents as resolve_dual_expert_condition_latents,
)
from .dual_stream_execution import (
    _video_token_grid_for_latents as _video_token_grid_for_latents,
)
from .dual_stream_execution import (
    forward_dual_expert_packed_coupling_denoise as forward_dual_expert_packed_coupling_denoise,
)
from .dual_stream_execution import (
    forward_joint_video_action_denoise as forward_joint_video_action_denoise,
)

_COMPATIBILITY_EXPORTS = (
    expand_dual_expert_scalar_timestep,
    step_dual_expert_flow_with_sigmas,
    dual_expert_scheduler_next_sigma,
    _checkpoint_summon_context,
    _summon_full_params,
    _unshard_runtime_params,
    build_chunk_causal_video_mask,
    build_dual_expert_attention_mask,
    build_dual_expert_inference_action_attention_mask,
    build_dual_expert_packed_coupling_attention_mask,
    build_dual_expert_packed_coupling_attention_profile,
    build_packed_action_attention_mask,
    forward_action_with_video_and_action_cache,
    forward_action_with_video_cache,
    prefill_video_kv_cache,
    append_dual_expert_action_cache,
    move_dual_expert_action_cache,
    move_dual_expert_video_cache,
    rewind_dual_expert_runtime_action_cache_to_frame,
    trim_dual_expert_action_cache_prefix,
    trim_dual_expert_action_cache_tail,
    trim_dual_expert_video_cache_tail,
    resolve_dual_expert_condition_latents,
    _video_token_grid_for_latents,
    forward_joint_video_action_denoise,
    forward_dual_expert_packed_coupling_denoise,
)
