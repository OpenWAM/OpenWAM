from __future__ import annotations

import torch

from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput

from .base import ActionDecoder, ActionDecoderInferOutput, ActionDecoderTrainOutput, align_policy_features


class RegisterActionDecoder(ActionDecoder):
    """Thin decoder wrapper for register-attached joint diffusion.

    Method-2 should keep its main generation logic in the shared backbone/runtime.
    This decoder only closes the contract:
    - train: read precomputed joint train artifacts and expose the final loss/output
    - infer: align the final action tensor to the configured horizon and package metadata
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_horizon: int,
        *,
        training_config,
        inference_config,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.training_config = training_config
        self.inference_config = inference_config
        self.dropout = dropout

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        del batch
        artifacts = policy_output.aux.get("joint_train_decoder_artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("RegisterActionDecoder expected `joint_train_decoder_artifacts` in policy_output.aux.")

        action_pred = artifacts["action_pred"]
        loss = artifacts["loss"]
        metrics = dict(artifacts.get("metrics", {}))
        aux = dict(artifacts.get("aux", {}))
        aux.setdefault("decoder", self.__class__.__name__)
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics=metrics,
            aux=aux,
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: object | None = None,
    ) -> ActionDecoderInferOutput:
        del previous_state
        action_pred = self._apply_action_sampler_mask(align_policy_features(policy_output.policy_features, self.action_horizon))
        aux = {
            "decoder": self.__class__.__name__,
        }
        predicted_latents = policy_output.aux.get("predicted_latents")
        if isinstance(predicted_latents, torch.Tensor):
            aux["predicted_latents"] = predicted_latents.detach()
        for key in (
            "video_num_inference_steps",
            "action_num_inference_steps",
            "joint_sampler",
            "joint_cfg_mode",
            "joint_cfg_enabled",
        ):
            if key in policy_output.aux:
                aux[key] = policy_output.aux[key]
        return ActionDecoderInferOutput(
            action_pred=action_pred,
            aux=aux,
        )
