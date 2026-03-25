from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, RegisterAttachedPolicyConfig, TrainingConfig
from open_wam.models.policy_variants.common.layouts import expand_previous_action
from open_wam.models.policy_variants.common.positions import build_sequence_position_context
from open_wam.models.visual_tower.grid_ids import build_sequence_grid_ids, build_video_grid_ids
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
from .cache import advance_register_cache, init_register_cache
from .layout import RegisterSequenceLayout, build_register_sequence_layout
from .masks import build_register_attention_mask
from .positions import build_register_position_context
from .timesteps import build_action_register_timestep_context


class RegisterAttachedPolicyVariant(PolicyVariant):
    """DreamZero-style register-attached policy variant."""

    def __init__(
        self,
        config: RegisterAttachedPolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
        state_horizon: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.state_horizon = state_horizon
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.register_role_embedding = nn.Embedding(2, config.hidden_size)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def _build_layout(self, visual_outputs: VisualStageOutputs) -> RegisterSequenceLayout:
        return build_register_sequence_layout(
            token_grid=visual_outputs.frontend.token_grid,
            action_horizon=self.action_horizon,
            state_horizon=self.state_horizon,
            num_frame_per_block=self.config.num_frame_per_block,
            num_action_per_block=self.config.num_action_per_block,
            num_state_per_block=self.config.num_state_per_block,
        )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        self._build_layout(visual_outputs)
        if batch.actions.shape[1] != self.action_horizon or batch.actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected actions with shape [B, {self.action_horizon}, {self.action_dim}], "
                f"got {tuple(batch.actions.shape)}"
            )
        if batch.state is None:
            raise ValueError("Register-attached variant requires state inputs.")
        if batch.state.shape[1] != self.state_horizon or batch.state.shape[2] != self.state_dim:
            raise ValueError(
                f"Expected state with shape [B, {self.state_horizon}, {self.state_dim}], "
                f"got {tuple(batch.state.shape)}"
            )
        return PolicyPreparedInputs(batch=batch)

    def _run_packed_core(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        action_inputs: torch.Tensor,
        state_inputs: torch.Tensor,
        current_start_frame: int,
    ) -> tuple[torch.Tensor, RegisterSequenceLayout, dict[str, object]]:
        layout = self._build_layout(visual_outputs)
        video_tokens = visual_outputs.frontend.video_tokens
        batch_size = video_tokens.shape[0]
        action_hidden = self.action_encoder(action_inputs)
        action_hidden = action_hidden + build_action_register_timestep_context(
            batch_size=batch_size,
            action_horizon=self.action_horizon,
            hidden_size=self.config.hidden_size,
            device=action_hidden.device,
        )
        action_hidden = action_hidden + self.register_role_embedding.weight[0][None, None, :]
        state_hidden = self.state_encoder(state_inputs)
        state_hidden = state_hidden + self.register_role_embedding.weight[1][None, None, :]
        # Packed register-attached sequence:
        # - `video_tokens`: `[B, T_video, H]`
        # - `action_hidden`: `[B, H_action, H]`
        # - `state_hidden`: `[B, H_state, H]`
        # Concatenation stays 1D over sequence length because action/state
        # registers are inserted as compact slots after the video region.
        packed_tokens = torch.cat([video_tokens, action_hidden, state_hidden], dim=1)
        stream_ids = torch.cat(
            [
                torch.zeros(batch_size, video_tokens.shape[1], device=packed_tokens.device, dtype=torch.long),
                torch.ones(batch_size, action_hidden.shape[1] + state_hidden.shape[1], device=packed_tokens.device, dtype=torch.long),
            ],
            dim=1,
        )
        # Grid ids mix two position systems in one shared core input:
        # - video tokens use the spatial-temporal patch grid
        # - action/state registers use 1D sequence ids with offsets so the core
        #   can keep action slots and state slots distinct.
        packed_grid_ids = torch.cat(
            [
                build_video_grid_ids(
                    visual_outputs.frontend.token_grid,
                    device=packed_tokens.device,
                    frame_shift=float(current_start_frame),
                ),
                build_sequence_grid_ids(self.action_horizon, device=packed_tokens.device, offset=0.0),
                build_sequence_grid_ids(self.state_horizon, device=packed_tokens.device, offset=float(self.action_horizon)),
            ],
            dim=1,
        )
        position_context = build_register_position_context(
            layout=layout,
            token_grid=visual_outputs.frontend.token_grid,
            hidden_size=self.config.hidden_size,
            device=packed_tokens.device,
            current_start_frame=current_start_frame,
        )[None, :, :].expand(batch_size, -1, -1)
        attention_mask = build_register_attention_mask(layout, batch_size=batch_size, device=packed_tokens.device)
        # The shared core still receives one generic packed representation:
        # `[B, S_total, H]` plus side channels that describe where video ends
        # and registers begin.
        core_output = visual_tower.run_core(
            VisualCoreInput(
                tokens=packed_tokens,
                token_layout=layout,
                position_context=position_context,
                grid_ids=packed_grid_ids,
                timestep_values=torch.zeros(batch_size, packed_tokens.shape[1], device=packed_tokens.device, dtype=torch.float32),
                stream_ids=stream_ids,
                attention_mask=attention_mask,
                conditioning=visual_outputs.frontend.conditioning,
            )
        )
        action_start = layout.action_block_spans[0][0]
        action_end = layout.action_block_spans[-1][1]
        return core_output.tokens[:, action_start:action_end, :], layout, dict(core_output.aux)

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        batch = prepared_inputs.batch
        if batch.state is None:
            raise ValueError("Register-attached variant requires state inputs.")
        action_tokens, layout, core_aux = self._run_packed_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            action_inputs=batch.actions,
            state_inputs=batch.state,
            current_start_frame=0,
        )
        return PolicyTrainOutput(
            policy_features=action_tokens,
            metrics={"num_image_blocks": torch.tensor(float(layout.num_image_blocks), device=action_tokens.device)},
            aux={"variant": self.config.name, "layout": layout, "core_aux": core_aux},
        )

    def prepare_infer_state(
        self,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        if previous_state is not None:
            return previous_state
        return PolicyInferState(step_index=0, cache=init_register_cache(self.config.num_frame_per_block))

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
        if context.state is None:
            state_inputs = torch.zeros(batch_size, self.state_horizon, self.state_dim, device=device, dtype=dtype)
        else:
            state_inputs = context.state.to(device=device, dtype=dtype)
        action_tokens, layout, core_aux = self._run_packed_core(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            action_inputs=previous_actions,
            state_inputs=state_inputs,
            current_start_frame=int(infer_state.cache.get("current_start_frame", 0)),
        )
        next_cache = advance_register_cache({key: int(value) for key, value in infer_state.cache.items()})
        return PolicyInferOutput(
            policy_features=action_tokens,
            next_state=PolicyInferState(step_index=infer_state.step_index + 1, cache=next_cache),
            aux={"variant": self.config.name, "layout": layout, "core_aux": core_aux},
        )
