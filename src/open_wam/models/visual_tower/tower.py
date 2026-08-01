from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_wam.configs import BackboneImplementation
from open_wam.contracts import ViewPlacement
from open_wam.models.common import (
    FlowMatchScheduler,
    RolloutCursor,
    unpatchify_video_sequence,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig, normalize_backbone_implementation
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata

from .cache_lifecycle import RuntimeCacheLifecycle, _MAX_CACHED_FRAMES_UNSET
from .contracts import (
    VisualCoreInput,
    VisualReadoutRequest,
    VisualRuntimeStateSnapshot,
    VisualStageOutputs,
)
from .core import PackedSequenceVisualCore
from .decoder import VisualFeatureDecoder
from .exact_runtime import (
    prepare_exact_single_stream_input,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)
from .frontend import SharedVideoFrontend
from .grid_ids import build_mesh_id, build_video_grid_ids
from .reference_core_weights import BackboneLoadReport
from .replica_core import SharedVideoTransformerCore
from .runtime_backbone import (
    ensure_runtime_module_device,
    initialize_runtime_backbone,
    log_runtime_backbone_missing_keys,
    reset_runtime_module_cache,
    validate_runtime_backbone_request,
)
from .runtime_programs import (
    RuntimeStepInput,
    RuntimeStepOutput,
    build_dense_runtime_program,
    build_single_stream_exact_runtime_program,
)

_ALLOWED_RUNTIME_MISSING_PREFIXES = (
    "proprio_context_encoder.",
    "proprio_hidden_context_encoder.",
    "generalist_mode_context_encoder.",
)


class VisualTower(nn.Module):
    """Stage-aware visual tower used by all policy variants."""

    def __init__(
        self,
        config: SharedVideoTransformerConfig | None = None,
        *,
        action_dim: int | None = None,
        state_dim: int | None = None,
        proprio_context_state_dim: int | None = None,
        proprio_hidden_context_state_dim: int | None = None,
        generalist_mode_context_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or SharedVideoTransformerConfig()
        self.action_dim = action_dim
        self.state_dim = state_dim
        implementation = normalize_backbone_implementation(self.config.implementation)
        self.frontend = SharedVideoFrontend(self.config)
        if implementation == BackboneImplementation.SHARED_TRANSFORMER:
            self.core = SharedVideoTransformerCore(self.config, action_dim=action_dim, state_dim=state_dim)
            if generalist_mode_context_enabled:
                configure_mode = getattr(self.core, "configure_generalist_mode_context_encoder", None)
                if not callable(configure_mode):
                    raise ValueError("Generalist mode text-token ablation requires a shared transformer core.")
                configure_mode(enabled=True)
            if proprio_context_state_dim is not None:
                configure_proprio = getattr(self.core, "configure_proprio_context_encoder", None)
                if not callable(configure_proprio):
                    raise ValueError("Proprio context mode requires a shared transformer core.")
                configure_proprio(enabled=True, state_dim=int(proprio_context_state_dim))
            if proprio_hidden_context_state_dim is not None:
                configure_proprio_hidden = getattr(self.core, "configure_proprio_hidden_context_encoder", None)
                if not callable(configure_proprio_hidden):
                    raise ValueError("Per-chunk proprio context mode requires a shared transformer core.")
                configure_proprio_hidden(enabled=True, state_dim=int(proprio_hidden_context_state_dim))
        elif implementation == BackboneImplementation.DUMMY:
            self.core = PackedSequenceVisualCore(self.config)
        else:
            raise ValueError(
                f"Unsupported backbone implementation '{self.config.implementation}'. "
                "Expected 'dummy' or 'shared_transformer'."
            )
        self.decoder = VisualFeatureDecoder(self.config.hidden_size)
        self.reference_core_load_report: BackboneLoadReport | None = None
        if self.config.load_reference_core_weights:
            if implementation != BackboneImplementation.SHARED_TRANSFORMER:
                raise ValueError("`backbone.load_reference_core_weights` requires `backbone.implementation = shared_transformer`.")
            if self.action_dim is None:
                raise ValueError("VisualTower requires `action_dim` to load reference weights into the shared core.")
            self._ensure_runtime_backbone_initialized()

    def run_frontend(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        negative_text_context=None,
        preserve_stream_cache: bool = False,
    ):
        self._ensure_frontend_runtime_device(canonical_video.device)
        return self.frontend(
            canonical_video,
            placements=placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )

    def run_frontend_from_latents(
        self,
        video_latents,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context=None,
        negative_text_context=None,
        canonical_video=None,
    ):
        self._ensure_frontend_runtime_device(video_latents.device)
        return self.frontend.from_video_latents(
            video_latents,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            canonical_video=canonical_video,
        )

    def reset_runtime_state(self) -> None:
        self.frontend.reset_runtime_state()

    def snapshot_runtime_state(
        self,
        *,
        cache_name: str | None = None,
    ) -> VisualRuntimeStateSnapshot | None:
        """Copy causal frontend and named backbone state for speculation."""

        frontend_state = self.frontend.snapshot_runtime_state()
        if cache_name is None:
            if frontend_state is None:
                return None
            return VisualRuntimeStateSnapshot(frontend_state=frontend_state)

        transformer = self.get_runtime_backbone(
            action_dim=self._runtime_state_action_dim()
        )
        snapshot_cache = getattr(transformer, "snapshot_runtime_cache_state", None)
        if not callable(snapshot_cache):
            raise TypeError(
                "The runtime backbone does not expose `snapshot_runtime_cache_state`."
            )
        cache_existed, cache_state = snapshot_cache(str(cache_name))
        return VisualRuntimeStateSnapshot(
            frontend_state=frontend_state,
            runtime_cache_name=str(cache_name),
            runtime_cache_existed=bool(cache_existed),
            runtime_cache_state=cache_state,
        )

    def restore_runtime_state(
        self,
        snapshot: VisualRuntimeStateSnapshot | None,
    ) -> None:
        """Restore a visual runtime snapshot after rejected speculation."""

        if snapshot is None:
            return
        self.frontend.restore_runtime_state(snapshot.frontend_state)
        if snapshot.runtime_cache_name is None:
            return
        transformer = self.get_runtime_backbone(
            action_dim=self._runtime_state_action_dim()
        )
        restore_cache = getattr(transformer, "restore_runtime_cache_state", None)
        if not callable(restore_cache):
            raise TypeError(
                "The runtime backbone does not expose `restore_runtime_cache_state`."
            )
        restore_cache(
            snapshot.runtime_cache_name,
            existed=bool(snapshot.runtime_cache_existed),
            cache_state=snapshot.runtime_cache_state,
        )

    def _runtime_state_action_dim(self) -> int:
        if self.action_dim is None:
            raise ValueError(
                "VisualTower runtime-state snapshots require a configured "
                "action_dim when a named backbone cache is requested."
            )
        return int(self.action_dim)

    def run_core(self, core_input: VisualCoreInput):
        core_output = self.core(core_input)
        core_output.aux.setdefault(
            "weight_source",
            "reference_initialized" if self.reference_core_load_report is not None else "local_init",
        )
        if self.reference_core_load_report is not None:
            core_output.aux.setdefault("reference_core_loaded_keys", len(self.reference_core_load_report.loaded_keys))
        return core_output

    def execute_runtime_step(self, step_input: RuntimeStepInput) -> RuntimeStepOutput:
        if hasattr(self.core, "execute_runtime_step"):
            step_output = self.core.execute_runtime_step(step_input)
        else:  # pragma: no cover - defensive fallback for alternate cores
            if step_input.core_input is None:
                raise ValueError("VisualTower runtime execution fallback requires `core_input`.")
            core_output = self.core(step_input.core_input)
            step_output = RuntimeStepOutput(
                tokens=core_output.tokens,
                core_output=core_output,
                cache_state=core_output.cache_state,
                aux=dict(core_output.aux),
            )
        resolved_weight_source = (
            "reference_initialized" if self.reference_core_load_report is not None else "local_init"
        )
        step_output.aux.setdefault("weight_source", resolved_weight_source)
        if step_output.core_output is not None:
            step_output.core_output.aux.setdefault("weight_source", step_output.aux["weight_source"])
        if self.reference_core_load_report is not None:
            loaded_key_count = len(self.reference_core_load_report.loaded_keys)
            step_output.aux.setdefault("reference_core_loaded_keys", loaded_key_count)
            if step_output.core_output is not None:
                step_output.core_output.aux.setdefault("reference_core_loaded_keys", loaded_key_count)
        return step_output

    def configure_runtime_devices(
        self,
        devices: tuple[torch.device, ...],
        *,
        prep_device: torch.device | None = None,
        output_device: torch.device | None = None,
    ) -> None:
        configure = getattr(self.core, "configure_runtime_block_devices", None)
        if callable(configure):
            configure(
                tuple(torch.device(device) for device in devices),
                prep_device=None if prep_device is None else torch.device(prep_device),
                output_device=None if output_device is None else torch.device(output_device),
            )

    def project_video_tokens_to_latents(
        self,
        *,
        hidden_states: torch.Tensor,
        token_grid,
    ) -> torch.Tensor:
        projector = getattr(self.core, "project_video_tokens_to_latents", None)
        if not callable(projector):
            raise ValueError("Current visual core does not support direct video-token latent projection.")
        return projector(
            hidden_states=hidden_states,
            token_grid=token_grid,
        )

    def predict_video_flow(
        self,
        *,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_context: torch.Tensor | None,
        frame_start: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the shared exact single-stream video path without variant-specific logic."""

        if noisy_latents.ndim != 5:
            raise ValueError(
                "Expected `noisy_latents` with shape [B, C, T, H, W], "
                f"got {tuple(noisy_latents.shape)}."
            )
        batch_size, _, num_frames, latent_height, latent_width = noisy_latents.shape
        if timesteps.shape != (batch_size, num_frames):
            raise ValueError(
                "Video-flow prediction expects `timesteps` with shape [B, T], "
                f"got {tuple(timesteps.shape)} for latents {tuple(noisy_latents.shape)}."
            )
        model_dtype = resolve_runtime_module_dtype(self.core)
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                self.config.max_text_tokens,
                self.config.text_dim,
                device=noisy_latents.device,
                dtype=model_dtype,
            )
        else:
            text_context = text_context.to(device=noisy_latents.device, dtype=model_dtype)
        grid_id = build_mesh_id(
            f=num_frames // self.config.patch_size_t,
            h=latent_height // self.config.patch_size_h,
            w=latent_width // self.config.patch_size_w,
            t=0.0,
            f_shift=float(frame_start),
            action=False,
            device=noisy_latents.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        step_output = self.execute_runtime_step(
            RuntimeStepInput(
                program=build_single_stream_exact_runtime_program(),
                payload={
                    "noisy_latents": noisy_latents.to(dtype=model_dtype),
                    "timesteps": timesteps.to(device=noisy_latents.device, dtype=torch.float32),
                    "grid_id": grid_id,
                    "text_emb": text_context,
                    "attention_mask": attention_mask,
                },
                action_mode=False,
            )
        )
        if step_output.tokens is None:
            raise ValueError("Exact single-stream runtime step did not return video flow tokens.")
        return unpatchify_video_sequence(
            self.core.patch_size,
            step_output.tokens,
            num_frames,
            latent_height,
            latent_width,
            batch_size=batch_size,
        ).to(dtype=noisy_latents.dtype)

    def prefill_exact_video_cache(
        self,
        *,
        observed_prefix: torch.Tensor,
        text_context: torch.Tensor | None,
        frame_start: int = 0,
        cache_name: str = "mot_video_prefill",
        attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        detach_cache: bool = True,
    ) -> CacheState:
        """Materialize a single-stream video self-attention cache via shared runtime execution."""

        if observed_prefix.ndim != 5:
            raise ValueError(
                "Expected `observed_prefix` with shape [B, C, T, H, W], "
                f"got {tuple(observed_prefix.shape)}."
            )
        batch_size, _, num_frames, latent_height, latent_width = observed_prefix.shape
        if num_frames <= 0:
            raise ValueError("Video cache prefill requires at least one observed frame.")
        model_dtype = resolve_runtime_module_dtype(self.core)
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                self.config.max_text_tokens,
                self.config.text_dim,
                device=observed_prefix.device,
                dtype=model_dtype,
            )
        else:
            text_context = text_context.to(device=observed_prefix.device, dtype=model_dtype)
        _, token_grid = self.frontend.tokenize_video_latents(observed_prefix)
        grid_id = build_video_grid_ids(
            token_grid,
            device=observed_prefix.device,
            frame_shift=float(frame_start),
        )[None].expand(batch_size, -1, -1)
        timesteps = torch.zeros(
            batch_size,
            num_frames,
            device=observed_prefix.device,
            dtype=torch.float32,
        )
        transformer = self.get_runtime_backbone(action_dim=int(self.action_dim))
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=True,
            current_start_frame=frame_start,
            cached_frames=num_frames,
            chunk_size=num_frames,
            capability="self_attn_only",
            backend_name="merged_prefix",
            backend_payload=None,
            payload={
                "cache_name": cache_name,
                "stage": "mot_video_prefill",
                "tokens_per_frame": int(token_grid.tokens_per_frame),
                "detach_self_attention_cache": bool(detach_cache),
            },
            self_attention_kv=tuple(),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(
                current_start_frame=frame_start,
                update_kv_cache=True,
            ),
        )
        step_output = self.execute_runtime_step(
            RuntimeStepInput(
                program=build_single_stream_exact_runtime_program(),
                payload={
                    "noisy_latents": observed_prefix.to(dtype=model_dtype),
                    "timesteps": timesteps,
                    "grid_id": grid_id,
                    "text_emb": text_context,
                    "attention_mask": attention_mask,
                    "cross_attention_mask": cross_attention_mask,
                },
                update_cache=0,
                cache_name=cache_name,
                action_mode=False,
            )
        )
        if step_output.cache_state is None:
            raise ValueError("Exact video cache prefill did not return a cache state.")
        return step_output.cache_state

    def run_packed_exact_video_forward(
        self,
        *,
        video_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_context: torch.Tensor | None,
        frame_start: int = 0,
        attention_mask: torch.Tensor | None = None,
        cache_name: str = "packed_exact_video_forward",
        packed_copies: int = 1,
        detach_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[AttentionCacheEntry, ...]]:
        """Run an exact video forward and expose its self-attention K/V."""

        if video_latents.ndim != 5:
            raise ValueError(
                "Expected `video_latents` with shape [B, C, T, H, W], "
                f"got {tuple(video_latents.shape)}."
            )
        copy_count = int(packed_copies)
        if copy_count <= 0:
            raise ValueError(f"`packed_copies` must be positive, got {packed_copies}.")

        batch_size, _, num_frames, latent_height, latent_width = video_latents.shape
        if num_frames <= 0:
            raise ValueError("Packed exact video forward requires at least one frame.")
        if num_frames % copy_count != 0:
            raise ValueError(
                "Packed exact video forward requires the frame count to be divisible "
                f"by `packed_copies`, got num_frames={num_frames}, packed_copies={packed_copies}."
            )
        if timesteps.shape != (batch_size, num_frames):
            raise ValueError(
                "Packed exact video forward expects `timesteps` with shape [B, T], "
                f"got {tuple(timesteps.shape)} for latents {tuple(video_latents.shape)}."
            )

        patch_t = int(self.config.patch_size_t)
        patch_h = int(self.config.patch_size_h)
        patch_w = int(self.config.patch_size_w)
        frames_per_copy = num_frames // copy_count
        if (
            num_frames % patch_t != 0
            or frames_per_copy % patch_t != 0
            or latent_height % patch_h != 0
            or latent_width % patch_w != 0
        ):
            raise ValueError(
                "Packed exact video latents must be divisible by patch size. "
                f"latents={tuple(video_latents.shape)}, patch={(patch_t, patch_h, patch_w)}."
            )

        model_dtype = resolve_runtime_module_dtype(self.core)
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                self.config.max_text_tokens,
                self.config.text_dim,
                device=video_latents.device,
                dtype=model_dtype,
            )
        else:
            text_context = text_context.to(device=video_latents.device, dtype=model_dtype)

        tokens_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        # Packed teacher-forced copies share physical frame positions; the
        # attention mask distinguishes copies by sequence segment.
        grid_per_copy = build_mesh_id(
            f=frames_per_copy // patch_t,
            h=latent_height // patch_h,
            w=latent_width // patch_w,
            t=0.0,
            f_shift=float(frame_start),
            action=False,
            device=video_latents.device,
        )
        grid_id = torch.cat([grid_per_copy] * copy_count, dim=1).unsqueeze(0).expand(batch_size, -1, -1)

        transformer = self.get_runtime_backbone(action_dim=int(self.action_dim))
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=True,
            current_start_frame=frame_start,
            cached_frames=num_frames,
            chunk_size=num_frames,
            capability="self_attn_only",
            backend_name="merged_prefix",
            backend_payload=None,
            payload={
                "cache_name": cache_name,
                "stage": "packed_exact_video_forward",
                "tokens_per_frame": int(tokens_per_frame),
                "packed_copies": copy_count,
                "detach_self_attention_cache": bool(detach_cache),
            },
            self_attention_kv=tuple(),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(
                current_start_frame=frame_start,
                update_kv_cache=True,
            ),
        )
        step_output = self.execute_runtime_step(
            RuntimeStepInput(
                program=build_single_stream_exact_runtime_program(),
                payload={
                    "noisy_latents": video_latents.to(dtype=model_dtype),
                    "timesteps": timesteps.to(device=video_latents.device, dtype=torch.float32),
                    "grid_id": grid_id,
                    "text_emb": text_context,
                    "attention_mask": attention_mask,
                },
                update_cache=0,
                cache_name=cache_name,
                action_mode=False,
            )
        )
        if step_output.tokens is None:
            raise ValueError("Packed exact video forward did not return video flow tokens.")
        if step_output.cache_state is None:
            raise ValueError("Packed exact video forward did not return a cache state.")
        flow_pred = unpatchify_video_sequence(
            self.core.patch_size,
            step_output.tokens,
            num_frames,
            latent_height,
            latent_width,
            batch_size=batch_size,
        ).to(dtype=video_latents.dtype)
        return flow_pred, tuple(step_output.cache_state.self_attention_kv)

    def generate_conditioned_future_latents(
        self,
        *,
        observed_prefix: torch.Tensor,
        future_template: torch.Tensor,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
        frame_start: int,
        num_inference_steps: int,
        num_train_timesteps: int,
        sigma_shift: float,
        guidance_scale: float,
        denoise_ratio: float = 1.0,
        cache_name: str = "visual_tower_future_video_denoise",
        sample_seed: int | None = None,
    ) -> torch.Tensor:
        """Generate future video latents conditioned on a clean observed prefix.

        The visual tower owns the shared visual execution path. Variants can ask
        for a future-video rollout state, but they should not own the denoising
        loop itself.
        """

        if observed_prefix.ndim != 5 or future_template.ndim != 5:
            raise ValueError(
                "Expected observed_prefix and future_template with shape [B, C, T, H, W], "
                f"got observed_prefix={tuple(observed_prefix.shape)}, future_template={tuple(future_template.shape)}."
            )
        if observed_prefix.shape[0] != future_template.shape[0] or observed_prefix.shape[1] != future_template.shape[1]:
            raise ValueError(
                "Observed prefix and future template must agree on batch/channel dimensions, "
                f"got observed_prefix={tuple(observed_prefix.shape)}, future_template={tuple(future_template.shape)}."
            )
        if future_template.shape[2] <= 0:
            raise ValueError("Expected at least one future frame to generate.")

        transformer = self.core
        model_dtype = resolve_runtime_module_dtype(transformer)
        batch_size, channels, future_num_frames, latent_height, latent_width = future_template.shape
        total_num_frames = observed_prefix.shape[2] + future_num_frames
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = torch.zeros(
                batch_size,
                self.config.max_text_tokens,
                self.config.text_dim,
                device=future_template.device,
                dtype=model_dtype,
            )
        else:
            resolved_text_context = resolved_text_context.to(device=future_template.device, dtype=model_dtype)

        generator = None
        if sample_seed is not None:
            generator = torch.Generator(device=future_template.device)
            generator.manual_seed(int(sample_seed))
        latents = torch.randn(
            batch_size,
            channels,
            total_num_frames,
            latent_height,
            latent_width,
            device=future_template.device,
            dtype=model_dtype,
            generator=generator,
        )
        observed_prefix = observed_prefix.to(dtype=model_dtype)
        latents[:, :, : observed_prefix.shape[2]] = observed_prefix

        scheduler = FlowMatchScheduler(
            shift=sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=num_train_timesteps,
        )
        scheduler.set_timesteps(num_inference_steps)
        total_updates = len(scheduler.timesteps)
        denoise_updates = max(1, min(total_updates, int(round(total_updates * float(denoise_ratio)))))
        timesteps = scheduler.timesteps[:denoise_updates].to(device=future_template.device)

        with torch.inference_mode():
            for timestep in timesteps:
                video_input = prepare_exact_single_stream_input(
                    latents=latents,
                    timestep=timestep,
                    text_emb=resolved_text_context,
                    frame_st_id=frame_start,
                    backbone_config=self.config,
                    action_mode=False,
                    cond=observed_prefix,
                )
                video_noise_pred = run_exact_single_stream_forward(
                    transformer,
                    input_dict=video_input,
                    update_cache=0,
                    cache_name=cache_name,
                    action_mode=False,
                    guidance_scale=guidance_scale,
                    negative_text_emb=negative_text_context,
                    force_cfg_batch=False,
                )
                video_noise_pred = unpatchify_video_sequence(
                    transformer.patch_size,
                    video_noise_pred,
                    total_num_frames,
                    latent_height,
                    latent_width,
                    batch_size=batch_size,
                ).to(dtype=model_dtype)
                latents = scheduler.step(video_noise_pred, timestep, latents)
                latents[:, :, : observed_prefix.shape[2]] = observed_prefix

        return latents[:, :, observed_prefix.shape[2] :].to(dtype=future_template.dtype)

    def cache_capability(self) -> str:
        if normalize_backbone_implementation(self.config.implementation) == BackboneImplementation.SHARED_TRANSFORMER:
            return "self_attn_plus_cross_attn"
        return "none"

    def _runtime_cache_lifecycle(self) -> RuntimeCacheLifecycle:
        return RuntimeCacheLifecycle(
            capability=self.cache_capability(),
            num_layers=len(getattr(self.core, "blocks", [])),
        )

    def init_runtime_cache_state(
        self,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        backend_name: str = "merged_prefix",
        backend_payload=None,
        backend_init_kwargs: dict[str, Any] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        return self._runtime_cache_lifecycle().init_state(
            cursor=cursor,
            stage=stage,
            payload=payload,
            backend_name=backend_name,
            backend_payload=backend_payload,
            backend_init_kwargs=backend_init_kwargs,
            cfg_mode=cfg_mode,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=update_cross_attention_cache,
            max_cached_frames=max_cached_frames,
            sink_frames=sink_frames,
            local_attn_window=local_attn_window,
        )

    def resolve_runtime_cache_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str,
        payload: dict[str, object] | None = None,
        backend_name: str = "merged_prefix",
        backend_payload=None,
        backend_init_kwargs: dict[str, Any] | None = None,
        cfg_mode: str = "none",
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool = False,
        max_cached_frames: int | None | object = _MAX_CACHED_FRAMES_UNSET,
        sink_frames: int = 0,
        local_attn_window: int | None = None,
    ) -> CacheState:
        """Resolve a runtime cache state for one rollout step.

        Stateless variants use this to obtain an explicit no-op cache object,
        while cache-aware variants can pass through an existing backbone-owned
        cache without reimplementing initialization guards.
        """

        return self._runtime_cache_lifecycle().resolve_state(
            cache_state,
            cursor=cursor,
            stage=stage,
            payload=payload,
            backend_name=backend_name,
            backend_payload=backend_payload,
            backend_init_kwargs=backend_init_kwargs,
            cfg_mode=cfg_mode,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=update_cross_attention_cache,
            max_cached_frames=max_cached_frames,
            sink_frames=sink_frames,
            local_attn_window=local_attn_window,
        )

    def build_runtime_cache_update_metadata(
        self,
        cache_state: CacheState,
        *,
        current_start_frame: int,
        update_kv_cache: bool = False,
        update_cross_attention_cache: bool | None = None,
        cfg_mode: str | None = None,
        cache_branch: str | None = None,
    ) -> CacheUpdateMetadata:
        """Build one cache-update instruction from the shared runtime state."""

        return self._runtime_cache_lifecycle().build_update_metadata(
            cache_state,
            current_start_frame=current_start_frame,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=update_cross_attention_cache,
            cfg_mode=cfg_mode,
            cache_branch=cache_branch,
        )

    def ensure_runtime_cache_branches(
        self,
        cache_state: CacheState,
        *,
        branch_names: tuple[str, ...],
    ) -> CacheState:
        """Ensure named cache branches exist on a shared runtime cache."""

        return self._runtime_cache_lifecycle().ensure_branches(
            cache_state,
            branch_names=branch_names,
        )

    def truncate_runtime_cache_state(
        self,
        cache_state: CacheState,
        *,
        tokens_per_frame: int | None = None,
    ) -> CacheState:
        """Apply the shared retention policy to a cache state.

        The first cache-aware rollout users mainly need a rolling-window policy.
        The helper also understands a simple sink-plus-local-window layout so
        future variants can reuse the same retention vocabulary.
        """

        return self._runtime_cache_lifecycle().truncate_state(
            cache_state,
            tokens_per_frame=tokens_per_frame,
        )

    def advance_runtime_cache_state(
        self,
        cache_state: CacheState,
        *,
        next_cursor: RolloutCursor,
        payload_updates: dict[str, object] | None = None,
        tokens_per_frame: int | None = None,
        cached_frames_increment: int | None = None,
    ) -> CacheState:
        """Advance one runtime cache state to the next rollout cursor."""

        return self._runtime_cache_lifecycle().advance_state(
            cache_state,
            next_cursor=next_cursor,
            payload_updates=payload_updates,
            tokens_per_frame=tokens_per_frame,
            cached_frames_increment=cached_frames_increment,
        )

    def clear_runtime_cache_state(
        self,
        cache_state: CacheState | None,
        *,
        cursor: RolloutCursor,
        stage: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> CacheState:
        """Clear cached tensors while preserving the shared cache policy."""

        return self._runtime_cache_lifecycle().clear_state(
            cache_state,
            cursor=cursor,
            stage=stage,
            payload=payload,
        )

    def run_default_core(
        self,
        frontend_output,
        *,
        readout_request: VisualReadoutRequest | None = None,
    ):
        batch_size, seq_len, _ = frontend_output.video_tokens.shape
        step_output = self.execute_runtime_step(
            RuntimeStepInput(
                program=build_dense_runtime_program(),
                core_input=VisualCoreInput(
                    tokens=frontend_output.video_tokens,
                    token_layout=frontend_output.token_grid,
                    grid_ids=build_video_grid_ids(
                        frontend_output.token_grid,
                        device=frontend_output.video_tokens.device,
                    ),
                    timestep_values=frontend_output.video_tokens.new_zeros(
                        (batch_size, seq_len),
                        dtype=frontend_output.video_tokens.dtype,
                    ),
                    stream_ids=frontend_output.video_tokens.new_zeros(
                        (batch_size, seq_len),
                        dtype=frontend_output.video_tokens.dtype,
                    ).long(),
                    text_context=frontend_output.conditioning.text_context,
                    conditioning=frontend_output.conditioning,
                    readout_request=readout_request,
                ),
            )
        )
        if step_output.core_output is None:
            raise ValueError("Default dense runtime execution did not return a `core_output`.")
        return step_output.core_output

    def run_decode(self, frontend_output, core_output):
        return self.decoder(frontend_output=frontend_output, core_output=core_output)

    def decode_tokens(
        self,
        frontend_output,
        *,
        tokens: torch.Tensor,
        token_layout,
    ):
        return self.decoder.forward_tokens(
            frontend_output=frontend_output,
            tokens=tokens,
            token_layout=token_layout,
        )

    def _ensure_runtime_backbone_initialized(self) -> None:
        self.reference_core_load_report = initialize_runtime_backbone(
            current_report=self.reference_core_load_report,
            core=self.core,
            config=self.config,
            action_dim=self.action_dim,
        )

    @staticmethod
    def _log_runtime_backbone_missing_keys(
        report: BackboneLoadReport | None,
        *,
        config: SharedVideoTransformerConfig,
    ) -> None:
        log_runtime_backbone_missing_keys(report, config=config)

    def get_runtime_backbone(self, *, action_dim: int) -> nn.Module:
        """Return the shared transformer backbone for runtime-driven variants.

        Variants with custom rollout semantics may need direct access to the
        shared backbone object rather than the generic `run_core(...)` entry
        point. This keeps that access generic and avoids method-1-specific
        naming at the tower boundary.
        """
        validate_runtime_backbone_request(
            config=self.config,
            configured_action_dim=self.action_dim,
            requested_action_dim=action_dim,
        )
        self._ensure_runtime_backbone_initialized()
        return self.core

    def ensure_runtime_backbone_device(self, *, action_dim: int, device) -> nn.Module:
        """Move the shared runtime backbone onto the requested device/dtype."""
        transformer = self.get_runtime_backbone(action_dim=action_dim)
        return ensure_runtime_module_device(transformer, device=device)

    def _ensure_frontend_runtime_device(self, device) -> None:
        device = torch.device(device)
        if any(parameter.device != device for parameter in self.frontend.parameters()):
            self.frontend.to(device=device)
            return
        if any(buffer.device != device for buffer in self.frontend.buffers()):
            self.frontend.to(device=device)

    def reset_runtime_backbone_cache(self, *, action_dim: int, cache_name: str = "open_wam_exact") -> None:
        """Clear shared-backbone runtime cache state for a named session."""
        transformer = self.get_runtime_backbone(action_dim=action_dim)
        reset_runtime_module_cache(transformer, cache_name=cache_name)

    def forward_default(
        self,
        canonical_video,
        *,
        placements: tuple[ViewPlacement, ...] | None = None,
        task_text: tuple[str | None, ...] | None = None,
        include_decode: bool = False,
    ) -> VisualStageOutputs:
        frontend_output = self.run_frontend(canonical_video, placements=placements, task_text=task_text)
        core_output = self.run_default_core(frontend_output)
        decode_output = self.run_decode(frontend_output, core_output) if include_decode else None
        return VisualStageOutputs(frontend=frontend_output, core=core_output, decode=decode_output)
