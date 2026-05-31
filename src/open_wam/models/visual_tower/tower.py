from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_wam.configs import BackboneImplementation, ExportedRuntimeActionInitMode
from open_wam.data.raw_video import ViewPlacement
from open_wam.models.common import (
    RolloutCursor,
    clear_cache_backend_payload,
    init_cache_backend_payload,
    resolve_cache_backend_spec,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, normalize_backbone_implementation
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata
from open_wam.models.video_backbone.contracts import CacheBranchState

from .contracts import VisualCoreInput, VisualReadoutRequest, VisualStageOutputs
from .core import PackedSequenceVisualCore
from .decoder import VisualFeatureDecoder
from .exported_runtime_backbone import (
    is_allowed_runtime_missing_key,
    is_open_wam_exported_runtime_backbone_dir,
    load_exported_runtime_backbone_into_replica_core,
    resolve_runtime_backbone_dir,
)
from .frontend import SharedVideoFrontend
from .grid_ids import build_mesh_id, build_video_grid_ids
from .reference_core_weights import BackboneLoadReport, load_reference_weights_into_replica_core
from .replica_core import SharedVideoTransformerCore
from .reference_transformer import preferred_reference_dtype
from .runtime_programs import (
    RuntimeStepInput,
    RuntimeStepOutput,
    build_dense_runtime_program,
    build_single_stream_exact_runtime_program,
)

_MAX_CACHED_FRAMES_UNSET = object()
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

    def prepare_runtime_stream_inputs(
        self,
        *,
        family: str,
        action_inputs: torch.Tensor | None,
        state_inputs: torch.Tensor | None,
        action_timesteps: torch.Tensor | None,
        state_timesteps: torch.Tensor | None,
        action_adapter_name: str = "mlp",
        state_adapter_name: str = "mlp",
        use_state_adapter: bool = True,
    ):
        prepare_stream_inputs = getattr(self.core, "prepare_runtime_stream_inputs", None)
        if not callable(prepare_stream_inputs):
            raise ValueError("Current visual core does not support shared runtime stream adapters.")
        return prepare_stream_inputs(
            family=family,
            action_inputs=action_inputs,
            state_inputs=state_inputs,
            action_timesteps=action_timesteps,
            state_timesteps=state_timesteps,
            action_adapter_name=action_adapter_name,
            state_adapter_name=state_adapter_name,
            use_state_adapter=use_state_adapter,
        )

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

    def project_runtime_stream_outputs(
        self,
        *,
        family: str,
        hidden_states: torch.Tensor,
        token_layout: object | None,
    ) -> dict[str, torch.Tensor]:
        project_stream_outputs = getattr(self.core, "project_runtime_stream_outputs", None)
        if not callable(project_stream_outputs):
            raise ValueError("Current visual core does not support shared runtime stream output heads.")
        return project_stream_outputs(
            family=family,
            hidden_states=hidden_states,
            token_layout=token_layout,
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

        from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
            data_seq_to_patch,
            reference_runtime_dtype,
        )

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
        model_dtype = reference_runtime_dtype(self.core)
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
        return data_seq_to_patch(
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

        from open_wam.models.policy_variants.parallel_stream.reference_runtime import reference_runtime_dtype

        if observed_prefix.ndim != 5:
            raise ValueError(
                "Expected `observed_prefix` with shape [B, C, T, H, W], "
                f"got {tuple(observed_prefix.shape)}."
            )
        batch_size, _, num_frames, latent_height, latent_width = observed_prefix.shape
        if num_frames <= 0:
            raise ValueError("Video cache prefill requires at least one observed frame.")
        model_dtype = reference_runtime_dtype(self.core)
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

        from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
            data_seq_to_patch,
            reference_runtime_dtype,
        )

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

        model_dtype = reference_runtime_dtype(self.core)
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
        flow_pred = data_seq_to_patch(
            self.core.patch_size,
            step_output.tokens,
            num_frames,
            latent_height,
            latent_width,
            batch_size=batch_size,
        ).to(dtype=video_latents.dtype)
        return flow_pred, tuple(step_output.cache_state.self_attention_kv)

    def run_mot_packed_video_forward(
        self,
        *,
        noisy_video_latents: torch.Tensor,
        clean_video_latents: torch.Tensor,
        noisy_timesteps: torch.Tensor,
        clean_timesteps: torch.Tensor | None,
        text_context: torch.Tensor | None,
        attention_mask: torch.Tensor,
        frame_start: int = 0,
        cache_name: str = "mot_packed_video_training",
        use_activation_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, tuple[AttentionCacheEntry, ...]]:
        """Compatibility entry point for Method-5 packed video training.

        Method-5 historically called this helper with separate noisy and clean
        video copies. The generic packed exact runtime now owns the actual
        execution; this wrapper preserves the older Method-5 contract while
        keeping the shared implementation in one place.
        """

        del use_activation_checkpointing  # Activation checkpointing is handled by the shared exact runtime.
        if noisy_video_latents.shape != clean_video_latents.shape:
            raise ValueError(
                "MoT packed video forward expects matching noisy/clean video shapes, "
                f"got noisy={tuple(noisy_video_latents.shape)}, clean={tuple(clean_video_latents.shape)}."
            )
        effective_clean_timesteps = (
            torch.zeros_like(noisy_timesteps) if clean_timesteps is None else clean_timesteps
        )
        if noisy_timesteps.shape != effective_clean_timesteps.shape:
            raise ValueError(
                "MoT packed video forward expects matching noisy/clean timestep shapes, "
                f"got noisy={tuple(noisy_timesteps.shape)}, clean={tuple(effective_clean_timesteps.shape)}."
            )
        return self.run_packed_exact_video_forward(
            video_latents=torch.cat([noisy_video_latents, clean_video_latents], dim=2),
            timesteps=torch.cat([noisy_timesteps, effective_clean_timesteps], dim=1),
            text_context=text_context,
            frame_start=frame_start,
            attention_mask=attention_mask,
            cache_name=cache_name,
            packed_copies=2,
            detach_cache=False,
        )

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

        from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
            FlowMatchScheduler,
            data_seq_to_patch,
            prepare_reference_single_stream_input,
            reference_runtime_dtype,
            run_reference_single_stream_forward,
        )

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
        model_dtype = reference_runtime_dtype(transformer)
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
                video_input = prepare_reference_single_stream_input(
                    latents=latents,
                    timestep=timestep,
                    text_emb=resolved_text_context,
                    frame_st_id=frame_start,
                    backbone_config=self.config,
                    action_mode=False,
                    cond=observed_prefix,
                )
                video_noise_pred = run_reference_single_stream_forward(
                    transformer,
                    input_dict=video_input,
                    update_cache=0,
                    cache_name=cache_name,
                    action_mode=False,
                    guidance_scale=guidance_scale,
                    negative_text_emb=negative_text_context,
                    force_cfg_batch=False,
                )
                video_noise_pred = data_seq_to_patch(
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
        backend_spec = resolve_cache_backend_spec(backend_name)
        capability = self.cache_capability()
        resolved_max_cached_frames = (
            cursor.chunk_size if max_cached_frames is _MAX_CACHED_FRAMES_UNSET else max_cached_frames
        )
        resolved_payload = {"stage": stage, "block_index": cursor.block_index}
        if payload is not None:
            resolved_payload.update(payload)
        resolved_backend_payload = backend_payload
        if resolved_backend_payload is None:
            resolved_backend_payload = init_cache_backend_payload(
                backend_spec.name,
                num_layers=len(getattr(self.core, "blocks", [])),
                **(backend_init_kwargs or {}),
                metadata={"stage": stage, "block_index": cursor.block_index},
            )
        return CacheState(
            supported=capability != "none",
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=capability,
            backend_name=backend_spec.name,
            backend_payload=resolved_backend_payload,
            payload=resolved_payload,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=update_kv_cache,
                update_cross_attention_cache=update_cross_attention_cache,
                cfg_mode=cfg_mode,
                max_cached_frames=resolved_max_cached_frames,
                sink_frames=sink_frames,
                local_attn_window=local_attn_window,
            ),
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

        if isinstance(cache_state, CacheState):
            return cache_state
        return self.init_runtime_cache_state(
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

        previous_metadata = cache_state.update_metadata
        return CacheUpdateMetadata(
            current_start_frame=current_start_frame,
            update_kv_cache=update_kv_cache,
            update_cross_attention_cache=(
                previous_metadata.update_cross_attention_cache
                if update_cross_attention_cache is None
                else update_cross_attention_cache
            ),
            cfg_mode=previous_metadata.cfg_mode if cfg_mode is None else cfg_mode,
            max_cached_frames=previous_metadata.max_cached_frames,
            sink_frames=previous_metadata.sink_frames,
            local_attn_window=previous_metadata.local_attn_window,
            cache_branch=previous_metadata.cache_branch if cache_branch is None else cache_branch,
        )

    def ensure_runtime_cache_branches(
        self,
        cache_state: CacheState,
        *,
        branch_names: tuple[str, ...],
    ) -> CacheState:
        """Ensure named cache branches exist on a shared runtime cache."""

        next_branch_states = dict(cache_state.branch_states)
        for branch_name in branch_names:
            if branch_name == "default" or branch_name in next_branch_states:
                continue
            next_branch_states[branch_name] = CacheBranchState(
                backend_name=cache_state.backend_name,
                backend_payload=clear_cache_backend_payload(cache_state.backend_payload),
                payload={**cache_state.payload, "cache_branch": branch_name},
                self_attention_kv=tuple(),
                cross_attention_kv=tuple(),
            )
        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=cache_state.self_attention_kv,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
            branch_states=next_branch_states,
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

        if not cache_state.supported:
            return cache_state
        if cache_state.backend_name != "merged_prefix":
            return cache_state

        resolved_tokens_per_frame = tokens_per_frame
        if resolved_tokens_per_frame is None:
            payload_tokens_per_frame = cache_state.payload.get("tokens_per_frame")
            if isinstance(payload_tokens_per_frame, int) and payload_tokens_per_frame > 0:
                resolved_tokens_per_frame = payload_tokens_per_frame
        if resolved_tokens_per_frame is None or resolved_tokens_per_frame <= 0:
            return cache_state

        metadata = cache_state.update_metadata
        max_cached_frames = metadata.max_cached_frames
        sink_frames = max(0, metadata.sink_frames)
        local_attn_window = metadata.local_attn_window
        if max_cached_frames is None and local_attn_window is None:
            return cache_state

        sink_tokens = sink_frames * resolved_tokens_per_frame
        local_window_tokens = (
            None
            if local_attn_window is None
            else max(0, local_attn_window) * resolved_tokens_per_frame
        )
        max_cached_tokens = (
            None
            if max_cached_frames is None
            else max(0, max_cached_frames) * resolved_tokens_per_frame
        )

        truncated_self_attention = tuple(
            self._truncate_attention_cache_entry(
                entry,
                max_cached_tokens=max_cached_tokens,
                sink_tokens=sink_tokens,
                local_window_tokens=local_window_tokens,
            )
            for entry in cache_state.self_attention_kv
        )
        truncated_cross_attention = tuple(cache_state.cross_attention_kv)
        truncated_branch_states = {
            branch_name: CacheBranchState(
                backend_name=branch_state.backend_name,
                backend_payload=branch_state.backend_payload,
                payload=dict(branch_state.payload),
                self_attention_kv=tuple(
                    self._truncate_attention_cache_entry(
                        entry,
                        max_cached_tokens=max_cached_tokens,
                        sink_tokens=sink_tokens,
                        local_window_tokens=local_window_tokens,
                    )
                    for entry in branch_state.self_attention_kv
                ),
                cross_attention_kv=tuple(branch_state.cross_attention_kv),
            )
            for branch_name, branch_state in cache_state.branch_states.items()
        }

        retained_frame_cap = cache_state.cached_frames
        if max_cached_frames is not None:
            retained_frame_cap = min(retained_frame_cap, max_cached_frames)
        if local_attn_window is not None:
            retained_frame_cap = min(retained_frame_cap, sink_frames + max(0, local_attn_window))

        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=retained_frame_cap,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=truncated_self_attention,
            cross_attention_kv=truncated_cross_attention,
            update_metadata=cache_state.update_metadata,
            branch_states=truncated_branch_states,
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

        increment = next_cursor.chunk_size if cached_frames_increment is None else cached_frames_increment
        next_payload = dict(cache_state.payload)
        next_payload["block_index"] = next_cursor.block_index
        if tokens_per_frame is not None:
            next_payload["tokens_per_frame"] = tokens_per_frame
        if payload_updates is not None:
            next_payload.update(payload_updates)

        next_cache_state = CacheState(
            supported=cache_state.supported,
            current_start_frame=next_cursor.current_start_frame,
            cached_frames=cache_state.cached_frames + increment,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_state.backend_payload,
            payload=next_payload,
            self_attention_kv=cache_state.self_attention_kv,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=CacheUpdateMetadata(
                current_start_frame=next_cursor.current_start_frame,
                update_kv_cache=cache_state.update_metadata.update_kv_cache,
                update_cross_attention_cache=cache_state.update_metadata.update_cross_attention_cache,
                cfg_mode=cache_state.update_metadata.cfg_mode,
                max_cached_frames=cache_state.update_metadata.max_cached_frames,
                sink_frames=cache_state.update_metadata.sink_frames,
                local_attn_window=cache_state.update_metadata.local_attn_window,
                cache_branch=cache_state.update_metadata.cache_branch,
            ),
            branch_states=dict(cache_state.branch_states),
        )
        return self.truncate_runtime_cache_state(
            next_cache_state,
            tokens_per_frame=tokens_per_frame,
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

        resolved_cache = self.resolve_runtime_cache_state(
            cache_state,
            cursor=cursor,
            stage=stage or "runtime_reset",
            payload=payload,
        )
        next_payload = dict(resolved_cache.payload)
        if payload is not None:
            next_payload.update(payload)
        if stage is not None:
            next_payload["stage"] = stage
        return CacheState(
            supported=resolved_cache.supported,
            current_start_frame=cursor.current_start_frame,
            cached_frames=0,
            chunk_size=cursor.chunk_size,
            capability=resolved_cache.capability,
            backend_name=resolved_cache.backend_name,
            backend_payload=clear_cache_backend_payload(resolved_cache.backend_payload),
            payload=next_payload,
            self_attention_kv=tuple(),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(
                current_start_frame=cursor.current_start_frame,
                update_kv_cache=False,
                update_cross_attention_cache=False,
                cfg_mode=resolved_cache.update_metadata.cfg_mode,
                max_cached_frames=resolved_cache.update_metadata.max_cached_frames,
                sink_frames=resolved_cache.update_metadata.sink_frames,
                local_attn_window=resolved_cache.update_metadata.local_attn_window,
                cache_branch=resolved_cache.update_metadata.cache_branch,
            ),
            branch_states={
                branch_name: CacheBranchState(
                    backend_name=branch_state.backend_name,
                    backend_payload=clear_cache_backend_payload(branch_state.backend_payload),
                    payload=dict(branch_state.payload),
                    self_attention_kv=tuple(),
                    cross_attention_kv=tuple(),
                )
                for branch_name, branch_state in resolved_cache.branch_states.items()
            },
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
        if self.reference_core_load_report is not None:
            return
        if self.config.pretrained_model_name_or_path is None:
            return
        runtime_backbone_dir = resolve_runtime_backbone_dir(self.config)
        is_exported_runtime_dir = is_open_wam_exported_runtime_backbone_dir(runtime_backbone_dir)
        print(
            "[runtime_backbone_load] "
            f"resolved_dir={runtime_backbone_dir} "
            f"is_exported_runtime_dir={is_exported_runtime_dir}",
            flush=True,
        )
        if is_exported_runtime_dir:
            self.reference_core_load_report = load_exported_runtime_backbone_into_replica_core(
                self.core,
                backbone_config=self.config,
            )
            print(
                "[runtime_backbone_load] "
                f"mode=exported_runtime loaded_keys={len(self.reference_core_load_report.loaded_keys)} "
                f"missing_keys={len(self.reference_core_load_report.missing_reference_keys)}",
                flush=True,
            )
            self._log_runtime_backbone_missing_keys(self.reference_core_load_report, config=self.config)
            return
        self.reference_core_load_report = load_reference_weights_into_replica_core(
            self.core,
            backbone_config=self.config,
            action_dim=self.action_dim,
        )
        print(
            "[runtime_backbone_load] "
            f"mode=reference loaded_keys={len(self.reference_core_load_report.loaded_keys)} "
            f"missing_keys={len(self.reference_core_load_report.missing_reference_keys)}",
            flush=True,
        )
        self._log_runtime_backbone_missing_keys(self.reference_core_load_report, config=self.config)

    @staticmethod
    def _log_runtime_backbone_missing_keys(
        report: BackboneLoadReport | None,
        *,
        config: SharedVideoTransformerConfig,
    ) -> None:
        if report is None or not report.missing_reference_keys:
            return
        allow_random_action = config.exported_runtime_action_init_mode == ExportedRuntimeActionInitMode.RANDOM
        allowed = tuple(
            key
            for key in report.missing_reference_keys
            if is_allowed_runtime_missing_key(key, allow_random_action=allow_random_action)
        )
        unexpected = tuple(
            key
            for key in report.missing_reference_keys
            if not is_allowed_runtime_missing_key(key, allow_random_action=allow_random_action)
        )
        if allowed:
            print(
                "[runtime_backbone_load] "
                f"allowed_missing_keys={list(allowed)}",
                flush=True,
            )
        if unexpected:
            preview = list(unexpected[:20])
            print(
                "[runtime_backbone_load] "
                f"unexpected_missing_keys_count={len(unexpected)} "
                f"unexpected_missing_keys_preview={preview}",
                flush=True,
            )

    def get_runtime_backbone(self, *, action_dim: int) -> nn.Module:
        """Return the shared transformer backbone for runtime-driven variants.

        Variants with custom rollout semantics may need direct access to the
        shared backbone object rather than the generic `run_core(...)` entry
        point. This keeps that access generic and avoids method-1-specific
        naming at the tower boundary.
        """
        if normalize_backbone_implementation(self.config.implementation) != "shared_transformer":
            raise ValueError("Runtime backbone access requires `backbone.implementation = shared_transformer`.")
        if self.action_dim is None:
            raise ValueError("VisualTower runtime backbone access requires a configured action_dim.")
        if int(action_dim) != int(self.action_dim):
            raise ValueError(
                "Shared video-transformer backbone was constructed for a different action_dim, "
                f"requested={action_dim}, tower_action_dim={self.action_dim}."
            )
        self._ensure_runtime_backbone_initialized()
        return self.core

    def ensure_runtime_backbone_device(self, *, action_dim: int, device) -> nn.Module:
        """Move the shared runtime backbone onto the requested device/dtype."""
        transformer = self.get_runtime_backbone(action_dim=action_dim)
        device = torch.device(device)
        target_dtype = preferred_reference_dtype(device)
        needs_move = False
        for parameter in transformer.parameters():
            if parameter.device != device:
                needs_move = True
                break
            if parameter.is_floating_point() and parameter.dtype != target_dtype:
                needs_move = True
                break
        if not needs_move:
            for buffer in transformer.buffers():
                if buffer.device != device:
                    needs_move = True
                    break
                if buffer.is_floating_point() and buffer.dtype != target_dtype:
                    needs_move = True
                    break
        if needs_move:
            transformer.to(device=device, dtype=target_dtype)
        return transformer

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
        try:
            transformer.clear_runtime_prediction_cache(cache_name)
        except KeyError:
            pass
        except AttributeError:
            try:
                transformer.clear_pred_cache(cache_name)
            except KeyError:
                pass
        try:
            transformer.clear_runtime_cache_state(cache_name)
        except KeyError:
            pass
        except AttributeError:
            try:
                transformer.clear_cache(cache_name)
            except KeyError:
                pass

    def get_exact_runtime_transformer(self, *, action_dim: int) -> nn.Module:
        return self.get_runtime_backbone(action_dim=action_dim)

    def ensure_exact_runtime_transformer_device(self, *, action_dim: int, device) -> nn.Module:
        return self.ensure_runtime_backbone_device(action_dim=action_dim, device=device)

    def reset_exact_runtime_cache(self, *, action_dim: int, cache_name: str = "open_wam_exact") -> None:
        self.reset_runtime_backbone_cache(action_dim=action_dim, cache_name=cache_name)

    def get_lingbot_reference_transformer(self, *, action_dim: int) -> nn.Module:
        return self.get_runtime_backbone(action_dim=action_dim)

    def ensure_lingbot_reference_transformer_device(self, *, action_dim: int, device) -> nn.Module:
        return self.ensure_runtime_backbone_device(action_dim=action_dim, device=device)

    def reset_lingbot_reference_runtime(self, *, action_dim: int, cache_name: str = "open_wam_exact") -> None:
        self.reset_runtime_backbone_cache(action_dim=action_dim, cache_name=cache_name)

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

    def _truncate_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        *,
        max_cached_tokens: int | None,
        sink_tokens: int,
        local_window_tokens: int | None,
    ) -> AttentionCacheEntry:
        if entry.key is None or entry.value is None:
            return entry
        sequence_length = entry.key.shape[2]
        if sequence_length == 0:
            return entry

        if max_cached_tokens is not None and sequence_length <= max_cached_tokens:
            return entry

        total_tokens = sequence_length
        target_local_tokens = local_window_tokens
        if max_cached_tokens is not None:
            if sink_tokens >= max_cached_tokens:
                keep_indices = torch.arange(min(max_cached_tokens, total_tokens), device=entry.key.device)
                return self._slice_attention_cache_entry(entry, keep_indices)
            tail_budget = max(0, max_cached_tokens - sink_tokens)
            if target_local_tokens is None:
                target_local_tokens = tail_budget
            else:
                target_local_tokens = min(target_local_tokens, tail_budget)

        if target_local_tokens is None:
            if max_cached_tokens is None:
                return entry
            keep_indices = torch.arange(total_tokens - max_cached_tokens, total_tokens, device=entry.key.device)
            return self._slice_attention_cache_entry(entry, keep_indices)

        sink_tokens = min(sink_tokens, total_tokens)
        remaining_tokens = max(0, total_tokens - sink_tokens)
        target_local_tokens = min(target_local_tokens, remaining_tokens)
        if sink_tokens + target_local_tokens >= total_tokens:
            return entry

        head_indices = (
            torch.arange(sink_tokens, device=entry.key.device)
            if sink_tokens > 0
            else torch.empty(0, dtype=torch.long, device=entry.key.device)
        )
        tail_indices = torch.arange(
            total_tokens - target_local_tokens,
            total_tokens,
            device=entry.key.device,
        )
        keep_indices = torch.cat((head_indices, tail_indices), dim=0)
        return self._slice_attention_cache_entry(entry, keep_indices)

    def _slice_attention_cache_entry(
        self,
        entry: AttentionCacheEntry,
        keep_indices: torch.Tensor,
    ) -> AttentionCacheEntry:
        key = entry.key.index_select(2, keep_indices)
        value = entry.value.index_select(2, keep_indices)
        next_metadata = dict(entry.metadata)
        next_metadata["sequence_length"] = int(key.shape[2])
        return AttentionCacheEntry(key=key, value=value, metadata=next_metadata)
