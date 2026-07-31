"""Compatibility exports for the historical parallel-stream runtime module.

Executable policy behavior lives in role-owned modules beside this facade.
New code should import those owners directly.
"""

from open_wam.models.common.flow_matching import (
    FlowMatchScheduler,
    sample_timestep_id,
)
from open_wam.models.common.video_geometry import (
    unpatchify_video_sequence as data_seq_to_patch,
)
from open_wam.models.visual_tower.exact_runtime import (
    build_reference_mesh_id as get_mesh_id,
    clear_exact_prediction_cache as _clear_exact_prediction_cache,
    initialize_exact_runtime_cache as initialize_reference_cache,
    prepare_exact_single_stream_forward_input as prepare_reference_forward_input,
    prepare_exact_single_stream_input as prepare_reference_single_stream_input,
    repeat_exact_single_stream_input_for_cfg as repeat_input_for_cfg,
    resolve_runtime_module_dtype as reference_runtime_dtype,
    run_exact_single_stream_forward as run_reference_single_stream_forward,
)

from .anchored_action_rollout import (
    run_parallel_current_frame_action_chunk_inference_rollout,
    run_parallel_fastwam_first_frame_inference_rollout,
)
from .cache_execution import (
    build_joint_clean_cache_attention_mask as _build_joint_clean_cache_attention_mask,
    build_joint_clean_cache_attention_profile as _build_joint_clean_cache_attention_profile,
    summarize_slot_pool_cache_state as _summarize_slot_pool_cache_state,
    write_exact_cache_chunk as _write_exact_cache_chunk,
    write_joint_clean_tokens_to_exact_cache as _write_joint_clean_tokens_to_exact_cache,
)
from .cache_lifecycle import (
    commit_initial_observed_video_context as _maybe_commit_initial_observed_video_context,
    run_parallel_exact_cache_warmup,
)
from .conditional_rollout import (
    generalist_conditioning_chunk_size as _chunk_size_for_generalist_conditioning,
    generalist_conditioning_history_stream_visibility as _history_stream_visibility_for_generalist_conditioning,
    generalist_conditioning_prefix_visibility_mode as _prefix_visibility_mode_for_generalist_conditioning,
    generalist_conditioning_window_size as _window_size_for_generalist_conditioning,
    is_conditional_joint_denoise_mode as _is_conditional_joint_denoise_mode,
    resolve_action_conditioning_mode as _generalist_mode_for_action_conditioning,
    select_conditional_warmup_history_suffix as _select_conditional_warmup_history_suffix,
    slice_conditioning_chunk as _slice_conditioning_chunk,
    uses_generalist_mode_text_token as _uses_generalist_mode_text_token,
)
from .exact_cache import (
    ExactCacheContext,
    ExactCacheInterfaceSpec,
    build_clean_video_action_cache_stream_ids as _stream_ids_for_clean_video_action_tokens,
    build_dual_stream_cache_stream_ids as _stream_ids_for_exact_dual_stream_split,
    build_exact_cache_spec as _build_exact_cache_spec,
    count_single_stream_action_tokens as _single_stream_action_token_count,
    ensure_exact_cache_initialized as _ensure_exact_cache_initialized,
    ensure_exact_text_embeddings as ensure_reference_text_embeddings,
    existing_exact_cache_attention_window as _existing_exact_cache_attn_window,
    restore_slot_pool_layer_metadata as _restore_slot_pool_layer_metadata,
    resolve_exact_cache_context as _resolve_exact_cache_context,
    set_slot_pool_layer_metadata as _set_slot_pool_layer_metadata,
    validate_existing_exact_cache_attention_window as _validate_existing_exact_cache_attn_window,
)
from .forward_execution import (
    build_parallel_first_frame_attention_profile as _build_fastwam_first_frame_attention_profile,
    run_parallel_action_conditioned_forward as _run_parallel_action_conditioned_forward,
    run_parallel_action_conditioned_train,
    run_parallel_exact_dual_stream_forward as _run_parallel_exact_joint_forward_manual,
    run_parallel_exact_train,
    run_parallel_first_frame_conditioned_forward as _run_parallel_fastwam_first_frame_forward_manual,
    run_parallel_first_frame_conditioned_train as run_parallel_fastwam_first_frame_train,
)
from .generalist_training import (
    apply_generalist_joint_denoise_training_mode as _apply_generalist_joint_denoise_training_mode,
    apply_generalist_legacy_prefix_joint_training_mode as _apply_generalist_legacy_prefix_joint_training_mode,
    sample_generalist_joint_denoise_training_mode as _sample_joint_denoise_training_mode,
)
from .inference_artifacts import (
    LingbotParallelInferArtifacts,
    ParallelInferArtifacts,
)
from .inference_conditioning import (
    append_generalist_mode_text_context as _inject_generalist_mode_text_context,
    repeat_parallel_exact_input_for_cfg as _repeat_joint_input_for_cfg,
)
from .latent_conditioning import (
    build_repeated_first_frame_condition as _build_clean_video_condition_from_anchor,
    resolve_full_window_condition_latents as _resolve_full_condition_latents,
    select_first_frame_condition_latents as _select_first_frame_condition_latents,
)
from .packed_rollout import (
    _run_parallel_packed_inference_rollout_impl as _run_parallel_action_conditioned_inference_rollout_impl,
    run_parallel_packed_action_override_rollout as run_parallel_action_conditioned_action_override_inference_rollout,
    run_parallel_packed_inference_rollout as run_parallel_action_conditioned_inference_rollout,
)
from .proprio_conditioning import (
    apply_parallel_chunk_proprio_context as _apply_parallel_chunk_proprio_context,
    build_single_stream_hidden_proprio_context as _single_stream_hidden_proprio_context,
    inject_deprecated_proprio_text_context as _inject_proprio_text_context,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
    prefix_visibility_mode_for_policy as _prefix_visibility_mode_for_policy,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract as _uses_legacy_prefix_per_chunk_proprio_contract,
)
from .staged_rollout import (
    run_parallel_staged_inference_rollout as run_parallel_exact_inference_rollout,
)
from .training_artifacts import (
    LingbotParallelTrainArtifacts,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
)
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
    sample_coupled_parallel_timestep_values as _sample_coupled_timestep_values,
    sample_index_matched_timestep_values as _sample_index_matched_timestep_values,
    sample_shared_video_schedule_timestep_values as _sample_shared_video_schedule_timestep_values,
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
)

_COMPATIBILITY_EXPORTS = (
    FlowMatchScheduler,
    sample_timestep_id,
    data_seq_to_patch,
    get_mesh_id,
    _clear_exact_prediction_cache,
    initialize_reference_cache,
    prepare_reference_forward_input,
    prepare_reference_single_stream_input,
    repeat_input_for_cfg,
    reference_runtime_dtype,
    run_reference_single_stream_forward,
    run_parallel_current_frame_action_chunk_inference_rollout,
    run_parallel_fastwam_first_frame_inference_rollout,
    _build_joint_clean_cache_attention_mask,
    _build_joint_clean_cache_attention_profile,
    _summarize_slot_pool_cache_state,
    _write_exact_cache_chunk,
    _write_joint_clean_tokens_to_exact_cache,
    _maybe_commit_initial_observed_video_context,
    run_parallel_exact_cache_warmup,
    _chunk_size_for_generalist_conditioning,
    _history_stream_visibility_for_generalist_conditioning,
    _prefix_visibility_mode_for_generalist_conditioning,
    _window_size_for_generalist_conditioning,
    _is_conditional_joint_denoise_mode,
    _generalist_mode_for_action_conditioning,
    _select_conditional_warmup_history_suffix,
    _slice_conditioning_chunk,
    _uses_generalist_mode_text_token,
    ExactCacheContext,
    ExactCacheInterfaceSpec,
    _stream_ids_for_clean_video_action_tokens,
    _stream_ids_for_exact_dual_stream_split,
    _build_exact_cache_spec,
    _single_stream_action_token_count,
    _ensure_exact_cache_initialized,
    ensure_reference_text_embeddings,
    _existing_exact_cache_attn_window,
    _restore_slot_pool_layer_metadata,
    _resolve_exact_cache_context,
    _set_slot_pool_layer_metadata,
    _validate_existing_exact_cache_attn_window,
    _build_fastwam_first_frame_attention_profile,
    _run_parallel_action_conditioned_forward,
    run_parallel_action_conditioned_train,
    _run_parallel_exact_joint_forward_manual,
    run_parallel_exact_train,
    _run_parallel_fastwam_first_frame_forward_manual,
    run_parallel_fastwam_first_frame_train,
    _apply_generalist_joint_denoise_training_mode,
    _apply_generalist_legacy_prefix_joint_training_mode,
    _sample_joint_denoise_training_mode,
    LingbotParallelInferArtifacts,
    ParallelInferArtifacts,
    _inject_generalist_mode_text_context,
    _repeat_joint_input_for_cfg,
    _build_clean_video_condition_from_anchor,
    _resolve_full_condition_latents,
    _select_first_frame_condition_latents,
    _run_parallel_action_conditioned_inference_rollout_impl,
    run_parallel_action_conditioned_action_override_inference_rollout,
    run_parallel_action_conditioned_inference_rollout,
    _apply_parallel_chunk_proprio_context,
    _single_stream_hidden_proprio_context,
    _inject_proprio_text_context,
    _attention_profile_name_for_current_block_coupling,
    _prefix_visibility_mode_for_policy,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    _uses_legacy_prefix_per_chunk_proprio_contract,
    run_parallel_exact_inference_rollout,
    LingbotParallelTrainArtifacts,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
    _add_noise,
    _sample_coupled_timestep_values,
    _sample_index_matched_timestep_values,
    _sample_shared_video_schedule_timestep_values,
    _share_video_scheduler_grid_with_action_scheduler,
)
