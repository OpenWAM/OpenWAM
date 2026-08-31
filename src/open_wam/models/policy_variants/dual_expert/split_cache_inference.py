"""Split-cache recurrent inference program for staged DualExpert policies."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    TrainingConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.models.common.dynamics_objectives import (
    is_conditional_dynamics_objective,
    resolve_dynamics_rollout_plan,
)
from open_wam.models.common.flow_inference import (
    build_action_flow_match_inference_scheduler,
)
from open_wam.models.common.flow_schedule import (
    FlowMatchScheduler,
)
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
from open_wam.models.visual_tower.exact_runtime import (
    clear_exact_prediction_cache,
    initialize_exact_runtime_cache,
    prepare_exact_single_stream_input,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)

from ..contracts import (
    DecoderArtifactEnvelope,
    PolicyGeneratedVideo,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyOutputModality,
)
from ..output_semantics import video_action_program_output_modalities
from .attention_cached import build_dual_expert_inference_action_attention_mask
from .cache_execution import forward_action_with_video_and_action_cache
from .cache_state import (
    append_dual_expert_action_cache,
    move_dual_expert_action_cache,
    move_dual_expert_video_cache,
    rewind_dual_expert_runtime_action_cache_to_frame,
    trim_dual_expert_action_cache_tail,
    trim_dual_expert_video_cache_tail,
)
from .conditioning import DualExpertConditioning
from .contracts import (
    DualExpertActionCache,
    DualExpertRuntimeState,
    DualExpertVideoCache,
    DualExpertVideoLayerCache,
)
from .coupling_semantics import resolve_dual_expert_current_block_coupling
from .decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertInferArtifacts,
)
from .modules import DualExpertActionExpert
from .rollout_geometry import (
    resolve_dual_expert_inference_output_request,
    resolve_dual_expert_inference_window_size,
    resolve_dual_expert_rollout_frame_chunk_size,
)
from .runtime_routes import (
    DUAL_EXPERT_SPLIT_CACHE_INFERENCE_COUPLINGS,
)
from .sequence_layout import build_action_grid_ids_for_sequence

# Default LingBot-reference slot-pool window used by both
# `initialize_exact_runtime_cache` and the parallel-stream-aligned video-cache trim.
_DUAL_EXPERT_SLOT_POOL_ATTN_WINDOW = 30


@dataclass(frozen=True)
class DualExpertSplitCacheInferenceProgram:
    """Execute staged video/action rollout with persistent split caches."""

    config: DualExpertPolicyConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    conditioning: DualExpertConditioning
    action_expert: DualExpertActionExpert
    action_dim: int
    action_horizon: int

    def run(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: DualExpertRuntimeState,
    ) -> PolicyInferOutput:
        # True parallel-stream-aligned NON_JOINT_TWO_STREAM rollout with persistent
        # KV cache on the shared video core. Each chunk:
        #   1) First chunk only -- bootstrap the named shared-transformer cache
        #      with observed env latents
        #      at frame_start=0, all frames clean (timestep=0), update_cache=2.
        #      Matches parallel-stream's `_write_exact_cache_chunk` bootstrap.
        #   2) Every chunk -- run video denoise with ONLY the
        #      `frame_chunk_size` noisy current-chunk latents as Q. Past
        #      context comes from cache. `update_cache=0` during denoise
        #      steps, `update_cache=1` on the last step writes the new
        #      chunk's clean K/V back into cache, matching parallel-stream exactly.
        #   3) Extract a DualExpertVideoCache view of the updated cache (taking the
        #      cond half when CFG is doubled) so the action expert can
        #      cross-attend the full rollout history.
        #   4) Run action denoise against that DualExpertVideoCache.
        video_latents = visual_outputs.frontend.video_latents
        batch_size = int(video_latents.shape[0])
        device = next(self.action_expert.parameters()).device
        dtype = next(self.action_expert.parameters()).dtype
        video_device = next(visual_tower.core.parameters()).device
        video_dtype = resolve_runtime_module_dtype(visual_tower.core)

        chunk_frames, action_horizon, action_tokens_per_frame = (
            resolve_dual_expert_rollout_frame_chunk_size(
                context,
                default_frame_chunk_size=int(self.inference_config.frame_chunk_size),
                base_action_horizon=int(self.action_horizon),
            )
        )
        runtime_state.chunk_advance_frames = int(chunk_frames)
        current_block_coupling = resolve_dual_expert_current_block_coupling(self.config)
        native_modalities = video_action_program_output_modalities(
            self.config.program
        )
        output_request = resolve_dual_expert_inference_output_request(
            context,
            current_block_coupling=current_block_coupling,
            native_modalities=native_modalities,
        )
        action_only_rollout = (
            output_request.modalities != native_modalities
            and output_request.modalities
            == frozenset({PolicyOutputModality.ACTION})
        )
        video_only_rollout = (
            output_request.modalities != native_modalities
            and output_request.modalities
            == frozenset({PolicyOutputModality.VIDEO})
        )
        prior_output_request = runtime_state.split_cache_output_request
        if video_only_rollout:
            if runtime_state.action_cache is not None:
                raise ValueError(
                    "DualExpert video-only inference cannot reuse a session that already "
                    "contains split-cache action history. Reset the policy session first."
                )
            if prior_output_request is None:
                runtime_state.split_cache_output_request = output_request
            elif prior_output_request != output_request:
                raise ValueError(
                    "DualExpert split-cache inference cannot switch away from video-only "
                    "execution inside one recurrent policy session."
                )
        elif prior_output_request is not None:
            raise ValueError(
                "DualExpert split-cache inference cannot switch from video-only execution "
                "to an action-producing route inside one recurrent policy session."
            )
        if current_block_coupling not in DUAL_EXPERT_SPLIT_CACHE_INFERENCE_COUPLINGS:
            raise NotImplementedError(
                "Dual Expert split-cache inference only supports video_then_action "
                "and decoupled_same_step; "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        dynamics_rollout_plan = resolve_dynamics_rollout_plan(
            program=self.config.program,
            request=context.dynamics,
        )
        generalist_rollout_mode = dynamics_rollout_plan.objective
        if is_conditional_dynamics_objective(generalist_rollout_mode):
            raise ValueError(
                "dual-expert conditional dynamics rollout requires packed joint "
                "coupling. "
                "Do not use split-cache action-only rollout as an IDM/FDM substitute."
            )
        generalist_rollout_semantics = dynamics_rollout_plan.semantics
        video_commit_before_action = (
            current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION
        )

        text_context_for_video = runtime_state.text_context
        if text_context_for_video is None:
            text_context_for_video = visual_outputs.frontend.conditioning.text_context
        if (
            generalist_rollout_semantics.drop_text_conditioning
            and text_context_for_video is not None
        ):
            text_context_for_video = torch.zeros_like(text_context_for_video)
        if text_context_for_video is None:
            text_context_for_video = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=video_device,
                dtype=video_dtype,
            )
        else:
            text_context_for_video = text_context_for_video.to(
                device=video_device, dtype=video_dtype
            )
        if (
            self.config.generalist_mode_text_token
            and runtime_state.generalist_mode_text_token_count <= 0
        ):
            text_context_for_video, token_count = (
                self.conditioning.append_generalist_mode_text_token(
                    visual_tower,
                    text_context_for_video,
                    generalist_rollout_mode,
                )
            )
            runtime_state.text_context = text_context_for_video
            runtime_state.generalist_mode_text_token_count = int(token_count)
        # Full parallel-stream alignment: cache at 2B with CFG throughout the
        # video path. Bootstrap uses `force_cfg_batch=True` so every
        # subsequent denoise step (with `guidance_scale>1` and
        # `negative_text_emb`) can do CFG batching consistently. The
        # action expert runs at batch=B, so when we extract the
        # DualExpertVideoCache for action we slice the cond half `[:B]`.
        negative_text_context = (
            visual_outputs.frontend.conditioning.negative_text_context
        )
        negative_text_context = self.conditioning.resolve_text_context(
            visual_tower,
            negative_text_context,
            runtime_state.proprio_state,
            batch_size=batch_size,
            device=video_device,
            dtype=video_dtype,
            materialize_if_missing=False,
        )
        if self.config.generalist_mode_text_token and negative_text_context is not None:
            negative_text_context, _ = (
                self.conditioning.append_generalist_mode_text_token(
                    visual_tower,
                    negative_text_context,
                    generalist_rollout_mode,
                )
            )
        use_cfg = negative_text_context is not None and bool(
            self.inference_config.use_cache
        )

        cache_name = "dual_expert_split_cache"
        latent_channels = int(visual_tower.config.latent_channels)
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        is_first_chunk = runtime_state.past_clean_latents is None
        skip_observation_update = bool(
            context.extra.get("dual_expert_skip_observation_update", False)
        )
        if skip_observation_update and is_first_chunk:
            raise ValueError(
                "DualExpert open-loop extension requires an initialized non-joint rollout cache."
            )
        condition_frame_start_override_raw = context.extra.get(
            "dual_expert_condition_frame_start"
        )
        if skip_observation_update and condition_frame_start_override_raw is not None:
            raise ValueError(
                "DualExpert condition-frame rewind is only valid for observation-conditioned replans."
            )
        inference_window_size = resolve_dual_expert_inference_window_size(
            context,
            default_window_size=_DUAL_EXPERT_SLOT_POOL_ATTN_WINDOW,
        )
        # parallel-stream-aligned per-chunk warmup. On chunk 0 we allocate the
        # slot-pool backend via `initialize_exact_runtime_cache` and write the
        # bootstrap obs latents at frame_start=0. On subsequent chunks the
        # driver passes a fresh window of real env observations (encoded
        # into `video_latents`). We:
        #   1) clear the prediction cache (last chunk's denoise last-step
        #      pred K/V),
        #   2) write the real-env observation as a NEW stable chunk at
        #      `frame_start = runtime_state.next_condition_frame_start`.
        # `observed_prefix.shape[2]` is allowed to vary between chunks --
        # chunk 0 may use a 1-frame bootstrap (matching parallel-stream) while
        # subsequent chunks pass `chunk_frames` real env-observation latents
        # that overwrite the previous chunk's pred slots. The Route-A
        # inference mask removes the old chunk_frames-aligned bootstrap
        # constraint by treating past KV as always-visible.
        current_obs_frame_start = int(runtime_state.next_condition_frame_start)
        if is_first_chunk:
            initialize_exact_runtime_cache(
                visual_tower.core,
                cache_name=cache_name,
                attn_window=inference_window_size,
                batch_size=batch_size,
                frame_chunk_size=chunk_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                device=video_device,
                action_per_frame=action_tokens_per_frame,
                use_cfg=use_cfg,
            )
            current_obs_frame_start = 0
        elif condition_frame_start_override_raw is not None:
            current_obs_frame_start = int(condition_frame_start_override_raw)
        if self.inference_config.use_cache and not skip_observation_update:
            clear_exact_prediction_cache(visual_tower.core, cache_name=cache_name)
        observed_prefix = video_latents.to(device=video_device, dtype=video_dtype)
        if not skip_observation_update:
            boot_video_input = prepare_exact_single_stream_input(
                latents=observed_prefix,
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=current_obs_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            run_exact_single_stream_forward(
                visual_tower.core,
                input_dict=boot_video_input,
                update_cache=2,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            runtime_state.past_clean_latents = observed_prefix.detach()
        # Observation-conditioned replans write real observations into the
        # next slots. Async open-loop extensions intentionally skip this
        # write so planning ahead does not leak too-early real frames into a
        # future chunk; they extend from the already generated cache instead.
        generation_frame_start = (
            current_obs_frame_start + chunk_frames
            if skip_observation_update
            else current_obs_frame_start + int(observed_prefix.shape[2])
        )
        if is_first_chunk:
            runtime_state.chunk_origin_frame = int(generation_frame_start) % int(
                chunk_frames
            )
        runtime_state.next_condition_frame_start = (
            generation_frame_start + chunk_frames
            if skip_observation_update
            else generation_frame_start
        )

        if action_only_rollout:
            predicted_latents = observed_prefix.new_empty(
                batch_size,
                latent_channels,
                0,
                latent_height,
                latent_width,
            )
        else:
            # Cache-aware video denoise on the current noisy chunk only.
            latents = torch.randn(
                batch_size,
                latent_channels,
                chunk_frames,
                latent_height,
                latent_width,
                device=video_device,
                dtype=video_dtype,
            )
            video_scheduler = FlowMatchScheduler(
                shift=self.training_config.video_sigma_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=self.training_config.video_num_train_timesteps,
            )
            video_scheduler.set_timesteps(
                self.inference_config.video_num_inference_steps
            )
            video_timesteps = F.pad(
                video_scheduler.timesteps.to(device=video_device),
                (0, 1),
                mode="constant",
                value=0,
            )
            for index, timestep in enumerate(video_timesteps):
                last_step = index == len(video_timesteps) - 1
                video_input = prepare_exact_single_stream_input(
                    latents=latents,
                    timestep=timestep,
                    text_emb=text_context_for_video,
                    frame_st_id=generation_frame_start,
                    backbone_config=visual_tower.config,
                    action_mode=False,
                )
                video_noise_pred = run_exact_single_stream_forward(
                    visual_tower.core,
                    input_dict=video_input,
                    update_cache=1
                    if (
                        last_step
                        and (video_commit_before_action or video_only_rollout)
                        and self.inference_config.use_cache
                    )
                    else 0,
                    cache_name=cache_name,
                    action_mode=False,
                    guidance_scale=self.inference_config.guidance_scale,
                    negative_text_emb=negative_text_context,
                    force_cfg_batch=use_cfg,
                )
                if not last_step:
                    video_noise_pred = unpatchify_video_sequence(
                        visual_tower.core.patch_size,
                        video_noise_pred,
                        chunk_frames,
                        latent_height,
                        latent_width,
                        batch_size=batch_size,
                    ).to(dtype=video_dtype)
                    latents = video_scheduler.step(video_noise_pred, timestep, latents)
            predicted_latents = latents

        if video_only_rollout:
            next_state = infer_state
            next_state.step_index += 1
            next_state.cursor.current_start_frame = int(
                infer_state.cursor.current_start_frame
                + max(1, runtime_state.chunk_advance_frames)
            )
            next_state.variant_state = runtime_state
            empty_actions = torch.empty(
                batch_size,
                0,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            return PolicyInferOutput(
                policy_features=empty_actions.new_zeros(
                    batch_size,
                    0,
                    self.action_expert.hidden_size,
                ),
                next_state=next_state,
                decoder_artifacts=DecoderArtifactEnvelope(
                    contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
                    payload=DualExpertInferArtifacts(
                        action_pred=empty_actions,
                        predicted_latents=predicted_latents.detach(),
                        condition_mode=str(self.config.condition_mode),
                        program=self.config.program.value,
                    ),
                ),
                generated_video=PolicyGeneratedVideo(
                    latents=predicted_latents.detach(),
                    frame_start=int(generation_frame_start),
                    latent_space_identity=(
                        visual_outputs.frontend.latent_space_identity
                    ),
                ),
                generation_frame_start=int(generation_frame_start),
                aux={
                    "variant": self.config.name,
                    "architecture": "dual_expert",
                    "condition_mode": str(self.config.condition_mode),
                    "current_block_coupling": current_block_coupling.value,
                    "generation_frame_start": int(generation_frame_start),
                    "dual_expert_action_only_rollout": False,
                    "dual_expert_video_only_rollout": True,
                    "action_pred_executable": False,
                    "dual_expert_output_modalities": sorted(
                        modality.value for modality in output_request.modalities
                    ),
                    "predicted_latents": predicted_latents.detach(),
                    "predicted_video_latents": predicted_latents.detach(),
                    "dual_expert_cache_debug": {
                        "action_cache_seq_len": 0,
                        "current_action_frame_start": int(generation_frame_start),
                        "current_start_frame": int(
                            next_state.cursor.current_start_frame
                        ),
                        "next_condition_frame_start": int(
                            runtime_state.next_condition_frame_start
                        ),
                        "chunk_advance_frames": int(runtime_state.chunk_advance_frames),
                        "chunk_origin_frame": int(runtime_state.chunk_origin_frame),
                        "skip_observation_update": bool(skip_observation_update),
                        "current_block_coupling": current_block_coupling.value,
                        "dual_expert_action_only_rollout": False,
                        "dual_expert_video_only_rollout": True,
                        "video_commit_before_action": bool(video_commit_before_action),
                        "inference_window_size": int(inference_window_size),
                        "rollout_frame_chunk_size": int(chunk_frames),
                        "rollout_action_horizon": int(action_horizon),
                    },
                },
            )

        # Don't advance `next_condition_frame_start` past the observation
        # write position. parallel-stream with `advance_frame_start=False` keeps
        # frame_start at the value warmup set it to, so that the NEXT
        # chunk's warmup writes its real observations at the same rotary
        # positions that the current chunk's pred entries just landed on
        # (teacher-forcing the pred positions with real obs). The pred
        # entries (chunk_frames tokens at rotary [gen_start..gen_start+4))
        # will be cleared + overwritten by the next chunk's stable obs
        # write via `clear_exact_prediction_cache` plus
        # `run_exact_single_stream_forward(update_cache=2)`. The TOTAL number
        # of clean video frames the
        # action expert sees this chunk is observation frames +
        # current-chunk pred frames.
        total_clean_video_frames = generation_frame_start + (
            0 if action_only_rollout else chunk_frames
        )
        action_visible_video_end_frame = (
            total_clean_video_frames
            if video_commit_before_action
            else generation_frame_start
        )

        def extract_dual_expert_video_cache_from_exact_cache() -> DualExpertVideoCache:
            # With CFG active the cache is doubled `[cond, uncond]` on the
            # batch dim; slice the cond half for the action expert (batch=B).
            cache_state = visual_tower.get_runtime_cache_state(cache_name)
            if cache_state is None:
                raise RuntimeError(
                    f"Dual Expert split-cache inference expected cache state at `{cache_name}` "
                    "but the shared transformer returned None."
                )
            extracted_layers: list[DualExpertVideoLayerCache] = []
            for entry in cache_state.self_attention_kv:
                if entry.key is None or entry.value is None:
                    raise RuntimeError(
                        "Dual Expert split-cache extraction found an empty layer entry."
                    )
                key = entry.key
                value = entry.value
                if key.shape[0] == 2 * batch_size:
                    key = key[:batch_size]
                    value = value[:batch_size]
                elif key.shape[0] != batch_size:
                    raise RuntimeError(
                        "Dual Expert split-cache batch dimension must match the current batch "
                        f"(or 2x for CFG), got cache_batch={key.shape[0]}, batch_size={batch_size}."
                    )
                extracted_layers.append(
                    DualExpertVideoLayerCache(key=key.detach(), value=value.detach())
                )
            return DualExpertVideoCache(
                layers=tuple(extracted_layers),
                video_seq_len=int(extracted_layers[0].key.shape[2]),
            )

        action_video_cache = extract_dual_expert_video_cache_from_exact_cache()

        def prepare_action_video_cache(
            cache: DualExpertVideoCache,
        ) -> DualExpertVideoCache:
            # parallel-stream alignment: parallel-stream's slot pool stores both video and
            # action so video occupies `(attn_window // 2) * latent_token_per_chunk`
            # tokens, which is integer-frame-aligned. dual-expert only writes video so the
            # slot pool fills with `(attn_window // 2) * (latent + action)` tokens
            # (= 67.5 frames here), leaving a partial leading frame after eviction.
            # Trim to parallel-stream's per-stream cap so the action expert sees the
            # same frame-aligned video lookback parallel-stream does.
            per_stream_video_lookback_frames = (inference_window_size // 2) * int(
                chunk_frames
            )
            max_video_tokens_for_action = (
                int(per_stream_video_lookback_frames)
                * int(runtime_state.video_tokens_per_frame)
                if runtime_state.video_tokens_per_frame
                else None
            )
            if (
                max_video_tokens_for_action is not None
                and max_video_tokens_for_action > 0
                and cache.video_seq_len > max_video_tokens_for_action
            ):
                cache = trim_dual_expert_video_cache_tail(
                    cache,
                    max_video_seq_len=max_video_tokens_for_action,
                )
            return move_dual_expert_video_cache(cache, device=device, dtype=dtype)

        action_video_cache = prepare_action_video_cache(action_video_cache)
        runtime_state.video_cache = action_video_cache
        cached_batch_size = int(action_video_cache.layers[0].key.shape[0])
        if cached_batch_size != batch_size:
            raise ValueError(
                "DualExpert cached-action inference requires the current observation batch to match the cached video batch, "
                f"got current_batch_size={batch_size}, cached_batch_size={cached_batch_size}."
            )
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            batch_size,
            action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        text_context = runtime_state.text_context
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        # parallel-stream-aligned action denoise with persistent action K/V
        # cache. Past action chunks' clean K/V live in
        # `runtime_state.action_cache`; fresh `action_horizon` tokens are
        # the only Q this forward recomputes every step. On the final
        # (padded timestep=0) step we capture the fresh per-layer K/V
        # and append to `runtime_state.action_cache`, mirroring parallel-stream's
        # `update_cache=1` at the last action step.
        action_cache_rewind_frame_start_raw = context.extra.get(
            "dual_expert_action_cache_rewind_frame_start"
        )
        if action_cache_rewind_frame_start_raw is None:
            action_cache_rewind_frame_start_raw = context.extra.get(
                "dual_expert_action_cache_prefix_frames"
            )
        if action_cache_rewind_frame_start_raw is not None:
            rewind_dual_expert_runtime_action_cache_to_frame(
                runtime_state,
                absolute_frame_start=int(action_cache_rewind_frame_start_raw),
                action_tokens_per_frame=action_tokens_per_frame,
            )
        past_action_cache = runtime_state.action_cache
        # Diagnostic: setting OPEN_WAM_DUAL_EXPERT_DISABLE_PAST_ACTION_CACHE=1 forces
        # the action expert to see only video + current noisy action per
        # chunk (no past action history). Useful for isolating whether
        # autoregressive drift in the past_action K/V chain is the source
        # of chunk-to-chunk instability.
        if os.environ.get("OPEN_WAM_DUAL_EXPERT_DISABLE_PAST_ACTION_CACHE", "0") == "1":
            past_action_cache = None
        past_action_seq_len = (
            int(past_action_cache.action_seq_len)
            if past_action_cache is not None
            else 0
        )
        if past_action_seq_len % action_tokens_per_frame != 0:
            raise ValueError(
                "Dual Expert split-cache action length must be a multiple of action_tokens_per_frame, "
                f"got past_action_seq_len={past_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        past_action_frames = past_action_seq_len // action_tokens_per_frame
        # parallel-stream byte-aligned mask: replicates
        # `build_chunked_temporal_exact_attention_profile` for the inference
        # `[video_cache; past_action_cache; current_action]` layout. Block
        # ids are video=chunk*2 / action=chunk*2+1, the within-window check
        # uses `training_config.window_size` (same value parallel-stream passes as
        # `input_dict["window_size"]` at inference), and clean/noise causal
        # rules match parallel-stream's chunked_temporal_exact profile.
        if (
            runtime_state.video_tokens_per_frame is None
            or runtime_state.video_tokens_per_frame <= 0
        ):
            raise RuntimeError(
                "DualExpert inference mask requires `runtime_state.video_tokens_per_frame` to be set, "
                f"got {runtime_state.video_tokens_per_frame!r}."
            )
        video_lookback_frames_for_mask = int(action_video_cache.video_seq_len) // int(
            runtime_state.video_tokens_per_frame
        )
        current_action_frame_start = int(generation_frame_start)
        video_frame_start = int(
            action_visible_video_end_frame - video_lookback_frames_for_mask
        )
        past_action_frame_start = (
            int(runtime_state.action_cache_start_frame)
            if past_action_cache is not None
            else int(current_action_frame_start)
        )
        if past_action_cache is not None:
            cached_action_end_frame = int(past_action_frame_start + past_action_frames)
            if cached_action_end_frame != current_action_frame_start:
                raise RuntimeError(
                    "DualExpert action cache frame span is not contiguous with the current chunk, "
                    f"cache_span=[{past_action_frame_start}, {cached_action_end_frame}), "
                    f"current_action_frame_start={current_action_frame_start}."
                )
        attention_mask = build_dual_expert_inference_action_attention_mask(
            video_seq_len=action_video_cache.video_seq_len,
            past_action_seq_len=past_action_seq_len,
            current_action_seq_len=action_horizon,
            video_tokens_per_frame=int(runtime_state.video_tokens_per_frame),
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=max(1, int(self.training_config.chunk_size)),
            window_size_frames=max(1, int(self.training_config.window_size)),
            device=device,
            video_can_attend_action=False,
            video_frame_start=video_frame_start,
            past_action_frame_start=past_action_frame_start,
            current_action_frame_start=current_action_frame_start,
            chunk_origin_frame=int(runtime_state.chunk_origin_frame),
            current_block_coupling=current_block_coupling,
            history_stream_visibility=self.config.history_stream_visibility,
        )
        # parallel-stream-aligned cache write: run the denoise loop without
        # capturing K/V, then issue a SEPARATE fresh forward at timestep=0
        # with the final denoised sample to capture cache-bound K/V. Mirrors
        # `_write_exact_cache_chunk(update_cache=1)` in
        # `run_parallel_action_conditioned_inference_rollout`, which calls a
        # fresh single-stream forward after all denoise steps complete
        # rather than reusing the loop's last-step K/V.
        fresh_action_kv: DualExpertActionCache | None = None
        for timestep in scheduler.timesteps.to(device=device):
            dense_timestep = torch.full(
                (batch_size, action_horizon),
                float(timestep),
                device=device,
                dtype=torch.float32,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=sample,
                timestep=dense_timestep,
                context=text_context.to(device=device, dtype=dtype),
                action_grid_ids=build_action_grid_ids_for_sequence(
                    batch_size=batch_size,
                    seq_len=action_horizon,
                    action_tokens_per_frame=action_tokens_per_frame,
                    device=device,
                    frame_shift=int(current_action_frame_start),
                ),
                hidden_context=self.conditioning.action_hidden_context_for_tokens(
                    visual_tower,
                    runtime_state.hidden_proprio_state,
                    action_tokens=sample,
                    action_tokens_per_frame=action_tokens_per_frame,
                    chunk_size_frames=chunk_frames,
                ),
            )
            action_hidden_states, _ = forward_action_with_video_and_action_cache(
                action_expert=self.action_expert,
                action_pre=action_pre,
                video_cache=action_video_cache,
                action_cache=past_action_cache,
                attention_mask=attention_mask,
            )
            flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
            sample = scheduler.step(flow_pred, timestep, sample)
        # Separate cache-write forward at timestep=0 with the final denoised
        # sample. This is the parallel-stream parity step.
        cache_write_timestep = torch.zeros(
            (batch_size, action_horizon),
            device=device,
            dtype=torch.float32,
        )
        cache_write_action_pre = self.action_expert.pre_dit(
            action_tokens=sample,
            timestep=cache_write_timestep,
            context=text_context.to(device=device, dtype=dtype),
            action_grid_ids=build_action_grid_ids_for_sequence(
                batch_size=batch_size,
                seq_len=action_horizon,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
                frame_shift=int(current_action_frame_start),
            ),
            hidden_context=self.conditioning.action_hidden_context_for_tokens(
                visual_tower,
                runtime_state.hidden_proprio_state,
                action_tokens=sample,
                action_tokens_per_frame=action_tokens_per_frame,
                chunk_size_frames=chunk_frames,
            ),
        )
        _, fresh_action_kv = forward_action_with_video_and_action_cache(
            action_expert=self.action_expert,
            action_pre=cache_write_action_pre,
            video_cache=action_video_cache,
            action_cache=past_action_cache,
            attention_mask=attention_mask,
        )
        if fresh_action_kv is None:
            raise RuntimeError(
                "Dual Expert split-cache write did not produce fresh K/V."
            )
        # Append fresh action K/V to the persistent action cache for next chunk.
        fresh_action_kv_moved = move_dual_expert_action_cache(
            fresh_action_kv, device=device, dtype=dtype
        )
        if past_action_cache is None:
            runtime_state.action_cache = fresh_action_kv_moved
            runtime_state.action_cache_start_frame = int(current_action_frame_start)
        else:
            runtime_state.action_cache = append_dual_expert_action_cache(
                past_action_cache, fresh_action_kv_moved
            )
        if (
            current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP
            and self.inference_config.use_cache
            and not action_only_rollout
        ):
            deferred_video_input = prepare_exact_single_stream_input(
                latents=predicted_latents.to(device=video_device, dtype=video_dtype),
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=generation_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            run_exact_single_stream_forward(
                visual_tower.core,
                input_dict=deferred_video_input,
                update_cache=1,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            action_video_cache = prepare_action_video_cache(
                extract_dual_expert_video_cache_from_exact_cache()
            )
            runtime_state.video_cache = action_video_cache
        # parallel-stream alignment: video and action share the same effective
        # lookback. parallel-stream stores both streams in the slot pool and
        # `attn_window` evicts them together; to mirror that with dual-expert's
        # split caches we trim the action cache to exactly the video cache's
        # current frame count. Asymmetric lookback (action shorter or longer
        # than video) is OOD: training always saw matched lengths, so the
        # action expert hallucinates when past action covers a different
        # frame span than past video.
        video_tokens_per_frame_for_trim = runtime_state.video_tokens_per_frame
        if (
            video_tokens_per_frame_for_trim is not None
            and video_tokens_per_frame_for_trim > 0
        ):
            video_lookback_frames = int(action_video_cache.video_seq_len) // int(
                video_tokens_per_frame_for_trim
            )
            max_action_seq_len = max(
                action_tokens_per_frame,
                video_lookback_frames * action_tokens_per_frame,
            )
            action_cache_before_trim = runtime_state.action_cache
            if action_cache_before_trim.action_seq_len > max_action_seq_len:
                dropped_action_tokens = int(
                    action_cache_before_trim.action_seq_len - max_action_seq_len
                )
                runtime_state.action_cache_start_frame += int(
                    dropped_action_tokens // action_tokens_per_frame
                )
            runtime_state.action_cache = trim_dual_expert_action_cache_tail(
                action_cache_before_trim,
                max_action_seq_len=max_action_seq_len,
            )
        next_state = infer_state
        next_state.step_index += 1
        next_state.cursor.current_start_frame = int(
            infer_state.cursor.current_start_frame
            + max(1, runtime_state.chunk_advance_frames)
        )
        next_state.variant_state = runtime_state
        decoder_payload = DualExpertInferArtifacts(
            action_pred=sample,
            predicted_latents=(
                predicted_latents.detach()
                if isinstance(predicted_latents, torch.Tensor)
                else None
            ),
            condition_mode=str(self.config.condition_mode),
            program=self.config.program.value,
        )
        return PolicyInferOutput(
            policy_features=sample.new_zeros(
                batch_size, 0, self.action_expert.hidden_size
            ),
            next_state=next_state,
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
            ),
            generated_video=(
                PolicyGeneratedVideo(
                    latents=predicted_latents.detach(),
                    frame_start=int(current_action_frame_start),
                    latent_space_identity=(
                        visual_outputs.frontend.latent_space_identity
                    ),
                )
                if isinstance(predicted_latents, torch.Tensor)
                and int(predicted_latents.shape[2]) > 0
                else None
            ),
            generation_frame_start=int(current_action_frame_start),
            aux={
                "variant": self.config.name,
                "architecture": "dual_expert",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "generation_frame_start": int(current_action_frame_start),
                "dual_expert_action_only_rollout": bool(action_only_rollout),
                "dual_expert_generalist_mode_text_token": (
                    generalist_rollout_mode.value
                    if int(runtime_state.generalist_mode_text_token_count) > 0
                    else None
                ),
                "dual_expert_generalist_mode_text_token_count": int(
                    runtime_state.generalist_mode_text_token_count
                ),
                "dual_expert_cache_debug": {
                    "video_cache_seq_len": int(runtime_state.video_cache.video_seq_len)
                    if runtime_state.video_cache is not None
                    else 0,
                    "action_video_cache_seq_len": int(action_video_cache.video_seq_len),
                    "action_cache_seq_len": int(
                        runtime_state.action_cache.action_seq_len
                    )
                    if runtime_state.action_cache is not None
                    else 0,
                    "action_cache_start_frame": int(
                        runtime_state.action_cache_start_frame
                    ),
                    "action_cache_frames_before_chunk": int(past_action_frames),
                    "total_clean_video_frames": int(total_clean_video_frames),
                    "is_first_chunk": bool(is_first_chunk),
                    "use_cfg": bool(use_cfg),
                    "current_start_frame": int(next_state.cursor.current_start_frame),
                    "next_condition_frame_start": int(
                        runtime_state.next_condition_frame_start
                    ),
                    "chunk_advance_frames": int(runtime_state.chunk_advance_frames),
                    "video_frame_start": int(video_frame_start),
                    "past_action_frame_start": int(past_action_frame_start),
                    "current_action_frame_start": int(current_action_frame_start),
                    "chunk_origin_frame": int(runtime_state.chunk_origin_frame),
                    "skip_observation_update": bool(skip_observation_update),
                    "condition_frame_start_override": (
                        None
                        if condition_frame_start_override_raw is None
                        else int(condition_frame_start_override_raw)
                    ),
                    "current_block_coupling": current_block_coupling.value,
                    "dual_expert_action_only_rollout": bool(action_only_rollout),
                    "video_commit_before_action": bool(video_commit_before_action),
                    "action_visible_video_end_frame": int(
                        action_visible_video_end_frame
                    ),
                    "inference_window_size": int(inference_window_size),
                    "rollout_frame_chunk_size": int(chunk_frames),
                    "rollout_action_horizon": int(action_horizon),
                    "action_cache_rewind_frame_start": (
                        None
                        if action_cache_rewind_frame_start_raw is None
                        else int(action_cache_rewind_frame_start_raw)
                    ),
                },
                **(
                    {
                        "predicted_latents": predicted_latents.detach(),
                        "predicted_video_latents": predicted_latents.detach(),
                    }
                    if isinstance(predicted_latents, torch.Tensor)
                    else {}
                ),
            },
        )
