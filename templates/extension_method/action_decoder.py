"""Non-runtime template for a new ActionDecoder implementation."""

from __future__ import annotations


class TemplateActionDecoder:
    """Skeleton only. Implement the real `ActionDecoder` contract in src/open_wam."""

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def forward_infer(self, *args, **kwargs):
        raise NotImplementedError
