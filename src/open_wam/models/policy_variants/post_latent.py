from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import (
    InferenceConfig,
    PoolingMode,
    PostLatentPolicyConfig,
    TrainingConfig,
    VisualReadoutSourceFamily,
)
from open_wam.models.visual_tower import VisualReadoutRequest, VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import (
    SharedVisualReadout,
    advance_default_runtime_infer_state,
    prepare_default_runtime_infer_state,
)
from .common.layouts import align_sequence_length, pool_frame_tokens, tokens_to_frame_major
from .contracts import (
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class PostLatentPolicyVariant(PolicyVariant):
    """Post-latent policy baseline with temporal structure preservation."""

    def __init__(
        self,
        config: PostLatentPolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_horizon: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.state_proj = nn.Linear(state_dim, config.hidden_size) if config.use_state_projection else None
        self.query_tokens = nn.Parameter(torch.randn(config.query_count, config.hidden_size)) if config.query_count > 0 else None
        self.query_norm = nn.LayerNorm(config.hidden_size) if config.query_count > 0 else None
        self.visual_readout = SharedVisualReadout(config.visual_readout, hidden_size=config.hidden_size)
        if config.visual_readout is not None and config.visual_readout.source_family not in {
            VisualReadoutSourceFamily.FINAL_CORE_TOKENS,
            VisualReadoutSourceFamily.CORE_LAYER_TOKENS,
            VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS,
        }:
            raise ValueError(
                "Post-latent currently supports only core-based shared visual readout families, "
                f"got {config.visual_readout.source_family!r}."
            )

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend", "core")

    def requested_visual_readout(self) -> VisualReadoutRequest | None:
        return self.visual_readout.requested_capture()

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        if batch.actions.ndim != 3:
            raise ValueError(
                "Expected actions with shape [B, H_action, D_action], "
                f"got {tuple(batch.actions.shape)}"
            )
        return PolicyPreparedInputs(batch=batch)

    def _resolve_visual_readout(self, visual_outputs: VisualStageOutputs):
        if visual_outputs.core is None:
            raise ValueError("Post-latent variant requires core outputs for post-visual-core attachment.")
        return self.visual_readout.resolve_from_core(visual_outputs.core)

    def _extract_policy_features(self, visual_outputs: VisualStageOutputs) -> torch.Tensor:
        resolved_readout = self._resolve_visual_readout(visual_outputs)
        tokens = resolved_readout.tokens
        if self.config.pooling_mode == PoolingMode.COMPAT_GLOBAL_MEAN:
            return tokens.mean(dim=1, keepdim=True).expand(-1, self.action_horizon, -1)
        frame_tokens = tokens_to_frame_major(tokens, visual_outputs.frontend.token_grid)
        if self.query_tokens is not None and self.query_norm is not None:
            frame_features = pool_frame_tokens(frame_tokens, mode="mean")
            queries = self.query_norm(self.query_tokens)[None, :, :].expand(frame_features.shape[0], -1, -1)
            attn_scores = torch.matmul(queries, frame_features.transpose(1, 2)) / math.sqrt(frame_features.shape[-1])
            attn_weights = F.softmax(attn_scores, dim=-1)
            queried_features = torch.matmul(attn_weights, frame_features)
            return align_sequence_length(queried_features, self.action_horizon)
        frame_features = pool_frame_tokens(frame_tokens, mode="mean")
        return align_sequence_length(frame_features, self.action_horizon)

    def _build_decoder_sequence_context(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        state: torch.Tensor | None,
    ) -> DecoderSequenceContext:
        resolved_readout = self._resolve_visual_readout(visual_outputs)
        tokens = resolved_readout.tokens
        frame_tokens = tokens_to_frame_major(tokens, visual_outputs.frontend.token_grid)
        return DecoderSequenceContext(
            sequence_tokens=frame_tokens,
            sequence_layout={
                "family": "video_feature_policy",
                "kind": "frame_token_grid",
                "attach_site": str(self.config.attach_site),
                "pooling_mode": str(self.config.pooling_mode),
                **resolved_readout.metadata,
            },
            token_grid=visual_outputs.frontend.token_grid,
            frame_count=frame_tokens.shape[1],
            source_stage=resolved_readout.source_stage,
            state_sequence=state,
        )

    def _fuse_state(self, policy_features: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None or self.state_proj is None:
            return policy_features
        state_summary = state.mean(dim=1)
        return policy_features + self.state_proj(state_summary)[:, None, :]

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        policy_features = self._extract_policy_features(visual_outputs)
        policy_features = self._fuse_state(policy_features, prepared_inputs.batch.state)
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={"policy_feature_norm": policy_features.norm(dim=-1).mean().detach()},
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=prepared_inputs.batch.state,
            ),
            aux={"variant": self.config.name, "attach_site": self.config.attach_site},
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del visual_outputs, context
        return prepare_default_runtime_infer_state(
            visual_tower,
            previous_state=previous_state,
            stage="post_latent",
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        policy_features = self._extract_policy_features(visual_outputs)
        policy_features = self._fuse_state(policy_features, context.state)
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=advance_default_runtime_infer_state(
                visual_tower,
                infer_state=infer_state,
                stage="post_latent",
            ),
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=context.state,
            ),
            aux={"variant": self.config.name, "attach_site": self.config.attach_site},
        )
