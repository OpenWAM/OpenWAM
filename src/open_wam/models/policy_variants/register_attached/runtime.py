from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import RegisterAttachedPolicyConfig
from open_wam.models.common import RegisterSequenceLayout, build_register_sequence_layout
from open_wam.models.video_backbone.contracts import CacheState, CacheUpdateMetadata, ConditioningState
from open_wam.models.visual_tower import (
    RegisterSequenceComponents,
    RegisterSequenceSemantics,
    StructuredBlockSemantics,
    StructuredFrequencyBundle,
    RuntimeStepInput,
    VisualCoreInput,
    VisualSequenceMetadata,
    VisualStageOutputs,
    VisualTower,
    build_register_sequence_runtime_program,
)


@dataclass(frozen=True)
class RegisterCoreRuntimeResult:
    """Packed-core result for the DreamZero-style register runtime."""

    video_hidden: torch.Tensor
    action_hidden: torch.Tensor
    projected_outputs: dict[str, torch.Tensor]
    layout: RegisterSequenceLayout
    cache_state: CacheState
    aux: dict[str, object]


@dataclass(frozen=True)
class RegisterRuntimeSpec:
    """Static method-2 runtime settings derived from config."""

    hidden_size: int
    action_horizon: int
    state_horizon: int
    num_frame_per_block: int
    num_action_per_block: int
    num_state_per_block: int
    variant_name: str
    structured_block_mode: str
    structured_time_layout: str
    structured_frequency_mode: str
    structured_teacher_forcing_layout: str
    structured_attention_kernel: str
    structured_cache_kernel: str
    stream_input_adapter_family: str
    stream_output_head_family: str
    use_state_encoder: bool
    action_encoder_type: str
    state_encoder_type: str


class RegisterAttachedRuntime:
    """Owns method-2 sequence assembly and core-call preparation.

    This keeps DreamZero-like train/infer packing semantics out of the policy
    variant so the next rewrite stage can move more of these semantics into the
    LingBot replica core without another large refactor.
    """

    def __init__(self, spec: RegisterRuntimeSpec) -> None:
        self.spec = spec

    @staticmethod
    def _move_tensor_dict(
        tensor_dict: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {name: tensor.to(device=device) for name, tensor in tensor_dict.items()}

    def build_layout(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        include_clean_video_prefix: bool,
        include_register_tokens: bool = True,
        require_matching_block_counts: bool = True,
    ) -> RegisterSequenceLayout:
        return build_register_sequence_layout(
            token_grid=visual_outputs.frontend.token_grid,
            action_horizon=self.spec.action_horizon,
            state_horizon=self.spec.state_horizon,
            num_frame_per_block=self.spec.num_frame_per_block,
            num_action_per_block=self.spec.num_action_per_block,
            num_state_per_block=self.spec.num_state_per_block,
            include_clean_video_prefix=include_clean_video_prefix,
            include_register_tokens=include_register_tokens,
            require_matching_block_counts=require_matching_block_counts,
        )

    def build_state_timestep_values(self, action_timesteps: torch.Tensor) -> torch.Tensor:
        if self.spec.state_horizon == 0:
            return action_timesteps.new_zeros(action_timesteps.shape[0], 0)
        if self.spec.state_horizon == self.spec.action_horizon:
            return action_timesteps
        if self.spec.action_horizon % self.spec.num_action_per_block != 0:
            raise ValueError(
                "Expected action horizon to be divisible by `num_action_per_block` "
                f"when building state timestep context, got {self.spec.action_horizon} "
                f"and {self.spec.num_action_per_block}."
            )
        block_count = self.spec.action_horizon // self.spec.num_action_per_block
        action_block_timesteps = action_timesteps.view(
            action_timesteps.shape[0],
            block_count,
            self.spec.num_action_per_block,
        )[:, :, 0]
        state_timesteps = action_block_timesteps.repeat_interleave(self.spec.num_state_per_block, dim=1)
        if state_timesteps.shape[1] != self.spec.state_horizon:
            raise ValueError(
                "State timestep expansion did not match the configured state horizon, "
                f"got {state_timesteps.shape[1]} and expected {self.spec.state_horizon}."
            )
        return state_timesteps

    def build_register_timestep_values(
        self,
        *,
        layout: RegisterSequenceLayout,
        visual_outputs: VisualStageOutputs,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = video_timesteps.shape[0]
        device = video_timesteps.device
        clean_video_values = torch.zeros(
            batch_size,
            layout.clean_video_sequence_length,
            device=device,
            dtype=torch.float32,
        )
        noisy_video_values = video_timesteps.repeat_interleave(
            visual_outputs.frontend.token_grid.tokens_per_frame,
            dim=1,
        )
        state_timesteps = self.build_state_timestep_values(action_timesteps)
        timestep_chunks = []
        if layout.has_clean_video_prefix:
            timestep_chunks.append(clean_video_values)
        timestep_chunks.extend([noisy_video_values, action_timesteps, state_timesteps])
        return torch.cat(timestep_chunks, dim=1)

    def build_noisy_frontend_outputs(
        self,
        visual_tower: VisualTower,
        *,
        visual_outputs: VisualStageOutputs,
        noisy_video_latents: torch.Tensor,
    ) -> VisualStageOutputs:
        noisy_frontend = visual_tower.run_frontend_from_latents(
            noisy_video_latents,
            task_text=None,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            canonical_video=visual_outputs.frontend.canonical_video,
        )
        return VisualStageOutputs(frontend=noisy_frontend)

    def run_core(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        clean_video_prefix_tokens: torch.Tensor | None,
        noisy_video_tokens: torch.Tensor,
        action_inputs: torch.Tensor,
        state_inputs: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        current_start_frame: int,
        cache_state: CacheState | None = None,
        cache_update_metadata: CacheUpdateMetadata | None = None,
        conditioning_override: ConditioningState | None = None,
        include_register_tokens: bool = True,
        cache_reference_token_span: tuple[int, int] | None = None,
        require_matching_block_counts: bool = True,
    ) -> RegisterCoreRuntimeResult:
        # Method 2 describes its structured sequence in terms of semantic
        # components here, then delegates actual materialization/execution to
        # the shared runtime program. The variant should stay at the level of
        # layout, cache policy, and scheduler behavior rather than owning token
        # encoders or flow heads directly.
        return_device = noisy_video_tokens.device
        layout = self.build_layout(
            visual_outputs,
            include_clean_video_prefix=clean_video_prefix_tokens is not None,
            include_register_tokens=include_register_tokens,
            require_matching_block_counts=require_matching_block_counts,
        )
        batch_size = noisy_video_tokens.shape[0]

        if include_register_tokens:
            state_timesteps = self.build_state_timestep_values(action_timesteps)
            prepared_streams = visual_tower.prepare_runtime_stream_inputs(
                family=self.spec.stream_input_adapter_family,
                action_inputs=action_inputs,
                state_inputs=state_inputs,
                action_timesteps=action_timesteps,
                state_timesteps=state_timesteps,
                action_adapter_name=self.spec.action_encoder_type,
                state_adapter_name=self.spec.state_encoder_type,
                use_state_adapter=self.spec.use_state_encoder,
            )
            action_hidden = prepared_streams["action_register"].tokens
            state_hidden = prepared_streams["state_register"].tokens
        else:
            action_hidden = noisy_video_tokens.new_zeros((batch_size, 0, self.spec.hidden_size))
            state_hidden = noisy_video_tokens.new_zeros((batch_size, 0, self.spec.hidden_size))
            state_timesteps = action_timesteps.new_zeros((batch_size, 0))

        pack_device = action_hidden.device if include_register_tokens else noisy_video_tokens.device
        if noisy_video_tokens.device != pack_device:
            noisy_video_tokens = noisy_video_tokens.to(device=pack_device)
        if clean_video_prefix_tokens is not None and clean_video_prefix_tokens.device != pack_device:
            clean_video_prefix_tokens = clean_video_prefix_tokens.to(device=pack_device)
        if action_hidden.device != pack_device:
            action_hidden = action_hidden.to(device=pack_device)
        if state_hidden.device != pack_device:
            state_hidden = state_hidden.to(device=pack_device)
        if video_timesteps.device != pack_device:
            video_timesteps = video_timesteps.to(device=pack_device)
        if action_timesteps.device != pack_device:
            action_timesteps = action_timesteps.to(device=pack_device)
        if state_timesteps.device != pack_device:
            state_timesteps = state_timesteps.to(device=pack_device)

        cache_reference_start, cache_reference_end = (
            cache_reference_token_span if cache_reference_token_span is not None else layout.noisy_video_span
        )
        step_output = visual_tower.execute_runtime_step(
            RuntimeStepInput(
                program=build_register_sequence_runtime_program(
                    input_adapter_family=self.spec.stream_input_adapter_family,
                    output_head_family=self.spec.stream_output_head_family,
                    structured_cache_kernel=self.spec.structured_cache_kernel,
                ),
                # `VisualCoreInput` here is intentionally high level: it carries
                # raw noisy/clean stream components plus structured semantics,
                # and the shared sequence adapter/core own the actual packing.
                core_input=VisualCoreInput(
                tokens=None,
                token_layout=layout,
                position_context=None,
                timestep_context=None,
                grid_ids=None,
                timestep_values=None,
                stream_ids=None,
                attention_mask=None,
                cache_state=cache_state,
                cache_update_metadata=cache_update_metadata,
                conditioning=conditioning_override or visual_outputs.frontend.conditioning,
                sequence_metadata=VisualSequenceMetadata(
                    teacher_forcing=clean_video_prefix_tokens is not None,
                    clean_prefix_tokens=layout.clean_video_sequence_length,
                    noisy_video_tokens=layout.noisy_video_sequence_length,
                    action_register_tokens=action_hidden.shape[1],
                    state_register_tokens=state_hidden.shape[1],
                    metadata={
                        "variant": self.spec.variant_name,
                        "num_image_blocks": layout.num_image_blocks,
                        "current_start_frame": current_start_frame,
                        # Cache whichever token span the caller marks as the
                        # clean/reference slice for this step. DreamZero-like
                        # warmup uses this to cache either the first observed
                        # frame or the latest clean reference block, not just a
                        # hard-coded prefix length.
                        "cacheable_video_tokens": max(cache_reference_end - cache_reference_start, 0),
                        "cache_reference_start": cache_reference_start,
                        "cache_reference_end": cache_reference_end,
                        "tokens_per_frame": layout.tokens_per_frame,
                    },
                ),
                register_components=RegisterSequenceComponents(
                    layout=layout,
                    token_grid=visual_outputs.frontend.token_grid,
                    clean_video_prefix_tokens=clean_video_prefix_tokens,
                    noisy_video_tokens=noisy_video_tokens,
                    action_register_tokens=action_hidden,
                    state_register_tokens=state_hidden,
                    current_start_frame=current_start_frame,
                    video_timesteps=video_timesteps,
                    action_timesteps=action_timesteps,
                    state_timesteps=state_timesteps,
                    semantics=RegisterSequenceSemantics(
                        sequence_family="register_sequence",
                        attention_style="blockwise_causal",
                        teacher_forcing_layout="clean_prefix",
                        timestep_layout="video_action_state",
                        video_sequence_tokens=layout.noisy_video_sequence_length,
                        action_register_tokens=action_hidden.shape[1],
                        state_register_tokens=state_hidden.shape[1],
                        current_start_frame=current_start_frame,
                        teacher_forcing=clean_video_prefix_tokens is not None,
                        structured_block_mode=self.spec.structured_block_mode,
                        structured_time_layout=self.spec.structured_time_layout,
                        structured_frequency_mode=self.spec.structured_frequency_mode,
                        structured_teacher_forcing_layout=self.spec.structured_teacher_forcing_layout,
                        structured_attention_kernel=self.spec.structured_attention_kernel,
                        structured_cache_kernel=self.spec.structured_cache_kernel,
                    ),
                ),
                ),
            )
        )
        if step_output.core_output is None:
            raise ValueError("Register sequence runtime execution did not return a `core_output`.")
        core_output = step_output.core_output
        noisy_video_start, noisy_video_end = layout.noisy_video_span
        output_device = return_device
        if layout.action_block_spans:
            action_start = layout.action_block_spans[0][0]
            action_end = layout.action_block_spans[-1][1]
            action_hidden = core_output.tokens[:, action_start:action_end, :].to(device=output_device)
        else:
            action_hidden = core_output.tokens.new_zeros((batch_size, 0, self.spec.hidden_size))
        return RegisterCoreRuntimeResult(
            video_hidden=core_output.tokens[:, noisy_video_start:noisy_video_end, :].to(device=output_device),
            action_hidden=action_hidden,
            projected_outputs=self._move_tensor_dict(
                dict(step_output.projected_outputs),
                device=output_device,
            ),
            layout=layout,
            cache_state=core_output.cache_state,
            aux={**core_output.aux, **step_output.aux},
        )
