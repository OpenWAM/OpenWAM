"""Separate transformer execution order from parameter registration."""

from collections.abc import Sequence

from torch import nn


class TransformerBlockSequence(nn.Module):
    """A stack may execute blocks owned by a larger architecture module.

    Borrowed blocks are a non-registering view. The architecture remains the
    single owner for checkpointing, optimizer construction, placement and FSDP.
    Bind once during assembly, never when switching between train and eval.
    """

    blocks: nn.ModuleList

    def __init__(self) -> None:
        super().__init__()
        self._execution_blocks: tuple[nn.Module, ...] | None = None

    @property
    def execution_blocks(self) -> Sequence[nn.Module]:
        return self.blocks if self._execution_blocks is None else self._execution_blocks

    def bind_execution_blocks(self, blocks: Sequence[nn.Module]) -> None:
        if self._execution_blocks is not None or len(self.blocks):
            raise RuntimeError(
                "Bind execution blocks once, after transferring their ownership."
            )
        self._execution_blocks = tuple(blocks)
