from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, ParallelStreamPolicyConfig, TrainingConfig
from open_wam.models.policy_variants.common.layouts import expand_previous_action
from open_wam.models.visual_tower.grid_ids import build_action_grid_ids, build_video_grid_ids
from open_wam.models.visual_tower import VisualCoreInput, VisualStageOutputs, VisualTower

from ..base import PolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .cache import advance_parallel_cache, init_parallel_cache
from .masks import build_parallel_attention_mask
from .packing import ParallelPackedSequenceLayout, action_tokens_to_frame_major, build_parallel_layout
from .positions import build_parallel_position_context
from .timesteps import build_parallel_timestep_context


class ParallelStreamPolicyVariant(PolicyVariant):
    """LingBot-style parallel-stream policy variant."""

    def __init__(
        self,
        config: ParallelStreamPolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_frames = num_frames
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def _validate_action_layout(self, action_horizon: int) -> None:
        expected_horizon = self.num_frames * self.config.action_per_frame
        if action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream variant requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_horizon}, num_frames={self.num_frames}, "
                f"action_per_frame={self.config.action_per_frame}"
            )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        self._validate_action_layout(batch.actions.shape[1])
        return PolicyPreparedInputs(batch=batch)

    def _build_action_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        self._validate_action_layout(actions.shape[1])
        hidden = self.action_embedder(actions)
        action_tokens_to_frame_major(hidden, self.num_frames, self.config.action_per_frame)
        return hidden

    def _sample_noise_scalars(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        video_scalar = torch.randint(
            low=0,
            high=self.training_config.video_num_train_timesteps,
            size=(batch_size,),
            device=device,
        ).float() / max(1, self.training_config.video_num_train_timesteps)
        action_scalar = torch.randint(
            low=0,
            high=self.training_config.action_num_train_timesteps,
            size=(batch_size,),
            device=device,
        ).float() / max(1, self.training_config.action_num_train_timesteps)
        return video_scalar, action_scalar

    def _pack_sequence(
        self,
        visual_outputs: VisualStageOutputs,
        action_tokens: torch.Tensor,
        video_noisy: torch.Tensor,
        action_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, ParallelPackedSequenceLayout]:
        layout = build_parallel_layout(
            token_grid=visual_outputs.frontend.token_grid,
            action_per_frame=self.config.action_per_frame,
            frame_chunk_size=self.config.frame_chunk_size,
            sequence_order=self.config.sequence_order,
            device=visual_outputs.frontend.video_tokens.device,
        )
        stream_map = {
            "video_noisy": video_noisy,
            "video_condition": visual_outputs.frontend.video_tokens,
            "action_noisy": action_noisy,
            "action_condition": action_tokens,
        }
        packed_tokens = torch.cat([stream_map[name] for name in self.config.sequence_order], dim=1)
        return packed_tokens, layout

    def _build_parallel_grid_ids(
        self,
        visual_outputs: VisualStageOutputs,
        layout: ParallelPackedSequenceLayout,
    ) -> torch.Tensor:
        contexts = []
        for name in self.config.sequence_order:
            if name.startswith("video"):
                contexts.append(
                    build_video_grid_ids(
                        visual_outputs.frontend.token_grid,
                        device=visual_outputs.frontend.video_tokens.device,
                    )
                )
            else:
                contexts.append(
                    build_action_grid_ids(
                        num_frames=visual_outputs.frontend.token_grid.num_frames,
                        action_per_frame=self.config.action_per_frame,
                        device=visual_outputs.frontend.video_tokens.device,
                    )
                )
        return torch.cat(contexts, dim=1)

    def _build_parallel_timestep_values(
        self,
        layout: ParallelPackedSequenceLayout,
        batch_size: int,
        device: torch.device,
        video_scalar: torch.Tensor,
        action_scalar: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        zero = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for name in self.config.sequence_order:
            start, end = layout.spans[name]
            length = end - start
            if name == "video_noisy":
                base = video_scalar
            elif name == "action_noisy":
                base = action_scalar
            else:
                base = zero
            values.append(base[:, None].expand(-1, length))
        return torch.cat(values, dim=1)

    def _run_parallel_core(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        action_inputs: torch.Tensor,
        video_noise_scale: float,
        action_noise_scale: float,
    ) -> tuple[torch.Tensor, ParallelPackedSequenceLayout]:
        video_tokens = visual_outputs.frontend.video_tokens
        action_tokens = self._build_action_tokens(action_inputs)
        video_noisy = video_tokens + video_noise_scale * torch.randn_like(video_tokens)
        action_noisy = action_tokens + action_noise_scale * torch.randn_like(action_tokens)
        packed_tokens, layout = self._pack_sequence(
            visual_outputs=visual_outputs,
            action_tokens=action_tokens,
            video_noisy=video_noisy,
            action_noisy=action_noisy,
        )
        batch_size = packed_tokens.shape[0]
        position_context = build_parallel_position_context(
            token_grid=visual_outputs.frontend.token_grid,
            layout=layout,
            hidden_size=self.config.hidden_size,
            action_per_frame=self.config.action_per_frame,
            device=packed_tokens.device,
        )[None, :, :].expand(batch_size, -1, -1)
        video_scalar, action_scalar = self._sample_noise_scalars(batch_size, packed_tokens.device)
        timestep_context = build_parallel_timestep_context(
            batch_size=batch_size,
            layout=layout,
            hidden_size=self.config.hidden_size,
            device=packed_tokens.device,
            video_scalar=video_scalar,
            action_scalar=action_scalar,
        )
        timestep_values = self._build_parallel_timestep_values(
            layout=layout,
            batch_size=batch_size,
            device=packed_tokens.device,
            video_scalar=video_scalar,
            action_scalar=action_scalar,
        )
        attention_mask = build_parallel_attention_mask(layout, batch_size=batch_size, device=packed_tokens.device)
        core_output = visual_tower.run_core(
            VisualCoreInput(
                tokens=packed_tokens,
                token_layout=layout,
                position_context=position_context,
                timestep_context=timestep_context,
                grid_ids=self._build_parallel_grid_ids(visual_outputs, layout),
                timestep_values=timestep_values,
                attention_mask=attention_mask,
                conditioning=visual_outputs.frontend.conditioning,
            )
        )
        action_start, action_end = layout.spans["action_noisy"]
        return core_output.tokens[:, action_start:action_end, :], layout

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        action_tokens, layout = self._run_parallel_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            action_inputs=prepared_inputs.batch.actions,
            video_noise_scale=self.training_config.video_sigma_shift / max(1.0, float(self.training_config.video_num_train_timesteps)),
            action_noise_scale=self.training_config.action_sigma_shift / max(1.0, float(self.training_config.action_num_train_timesteps)),
        )
        return PolicyTrainOutput(
            policy_features=action_tokens,
            metrics={"packed_sequence_length": torch.tensor(float(layout.frame_ids.numel()), device=action_tokens.device)},
            aux={"variant": self.config.name, "layout": layout},
        )

    def prepare_infer_state(
        self,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        if previous_state is not None:
            return previous_state
        return PolicyInferState(step_index=0, cache=init_parallel_cache(self.config.frame_chunk_size))

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        batch_size = visual_outputs.frontend.video_tokens.shape[0]
        dtype = visual_outputs.frontend.video_tokens.dtype
        device = visual_outputs.frontend.video_tokens.device
        previous_actions = expand_previous_action(
            previous_action=context.previous_action,
            batch_size=batch_size,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            device=device,
            dtype=dtype,
        )
        action_tokens, layout = self._run_parallel_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            action_inputs=previous_actions,
            video_noise_scale=0.0,
            action_noise_scale=0.0,
        )
        next_cache = advance_parallel_cache({key: int(value) for key, value in infer_state.cache.items()})
        return PolicyInferOutput(
            policy_features=action_tokens,
            next_state=PolicyInferState(step_index=infer_state.step_index + 1, cache=next_cache),
            aux={"variant": self.config.name, "layout": layout},
        )
