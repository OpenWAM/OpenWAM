"""Non-runtime template for a new PolicyVariant implementation."""

from __future__ import annotations


class TemplatePolicyVariant:
    """Skeleton only. Implement the real `PolicyVariant` contract in src/open_wam."""

    def required_visual_stages(self):
        raise NotImplementedError

    def prepare_train_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_infer_state(self, *args, **kwargs):
        raise NotImplementedError

    def forward_infer_step(self, *args, **kwargs):
        raise NotImplementedError
